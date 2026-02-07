"""
DC Decomposition for torch.gather operation. Forward/Backward: [4*batch] -> [4*batch]

Gather operation is linear - the same indices are applied to both pos and neg streams.
"""

import torch
import torch.nn as nn
from torch import Tensor

from .base import split_input_4, DC_IS_OUTPUT_LAYER
from .patch_builder import (
    ForwardBuilder, BackwardBuilder, get_cache_info,
    create_patch_function, create_unpatch_function,
)


class Gather(nn.Module):
    """
    Module wrapper for torch.gather that can be patched for DC decomposition.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, input: Tensor, index: Tensor) -> Tensor:
        return torch.gather(input, self.dim, index)

    def extra_repr(self) -> str:
        return f'dim={self.dim}'


class DCGatherFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_4: Tensor, index: Tensor, dim: int,
                is_output_layer: bool, cache, layer_name, alpha: float) -> Tensor:
        
        def compute(ctx, pos, neg, input_4, index, dim):
            out_pos = torch.gather(pos, dim, index)
            out_neg = torch.gather(neg, dim, index)

            ctx.save_for_backward(input_4, index)
            ctx.dim = dim

            return out_pos, out_neg

        return ForwardBuilder.run(
            ctx, input_4, compute, is_output_layer, cache, layer_name, alpha,
            extra_args=(input_4, index, dim)
        )

    @staticmethod
    def backward(ctx, grad_4: Tensor):
        def compute(ctx, delta_pp, delta_np, delta_pn, delta_nn):
            input_4, index = ctx.saved_tensors
            input_shape = split_input_4(input_4)[0].shape
            dim = ctx.dim

            new_pp = torch.zeros(input_shape, dtype=delta_pp.dtype, device=delta_pp.device)
            new_np = torch.zeros(input_shape, dtype=delta_np.dtype, device=delta_np.device)
            new_pn = torch.zeros(input_shape, dtype=delta_pn.dtype, device=delta_pn.device)
            new_nn = torch.zeros(input_shape, dtype=delta_nn.dtype, device=delta_nn.device)

            new_pp.scatter_add_(dim, index, delta_pp)
            new_np.scatter_add_(dim, index, delta_np)
            new_pn.scatter_add_(dim, index, delta_pn)
            new_nn.scatter_add_(dim, index, delta_nn)

            return new_pp, new_np, new_pn, new_nn

        return BackwardBuilder.run(ctx, grad_4, compute, num_extra_returns=6)


def dc_forward_gather(module: Gather, input: Tensor, index: Tensor) -> Tensor:
    cache, layer_name, alpha = get_cache_info(module)
    return DCGatherFunction.apply(
        input, index, module.dim,
        getattr(module, DC_IS_OUTPUT_LAYER, False),
        cache, layer_name, alpha,
    )


patch_gather = create_patch_function(dc_forward_gather)
unpatch_gather = create_unpatch_function()
