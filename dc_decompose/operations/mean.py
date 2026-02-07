"""
DC Decomposition for torch.mean operation. Forward/Backward: [4*batch] -> [4*batch]

Mean operation is linear - pos and neg streams are averaged independently.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Optional, Union

from .base import DC_IS_OUTPUT_LAYER
from .patch_builder import (
    ForwardBuilder, BackwardBuilder, get_cache_info,
    create_patch_function, create_unpatch_function,
)


class Mean(nn.Module):
    """
    Module wrapper for torch.mean that can be patched for DC decomposition.

    This is used by functional_replacer to replace torch.mean calls.
    """

    def __init__(self, dim: Optional[Union[int, Tuple[int, ...]]] = None, keepdim: bool = False):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, input: Tensor) -> Tensor:
        if self.dim is None:
            return torch.mean(input)
        return torch.mean(input, dim=self.dim, keepdim=self.keepdim)

    def extra_repr(self) -> str:
        return f'dim={self.dim}, keepdim={self.keepdim}'


class DCMeanFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_4: Tensor, dim, keepdim: bool,
                is_output_layer: bool, cache, layer_name, alpha: float) -> Tensor:
        
        def compute(ctx, pos, neg, dim, keepdim):
            # Apply mean to both streams
            if dim is None:
                out_pos = torch.mean(pos)
                out_neg = torch.mean(neg)
            else:
                out_pos = torch.mean(pos, dim=dim, keepdim=keepdim)
                out_neg = torch.mean(neg, dim=dim, keepdim=keepdim)

            ctx.input_shape = pos.shape
            ctx.dim = dim
            ctx.keepdim = keepdim

            return out_pos, out_neg

        return ForwardBuilder.run(
            ctx, input_4, compute, is_output_layer, cache, layer_name, alpha,
            extra_args=(dim, keepdim)
        )

    @staticmethod
    def backward(ctx, grad_4: Tensor):
        def compute(ctx, delta_pp, delta_np, delta_pn, delta_nn):
            input_shape = ctx.input_shape
            dim = ctx.dim
            keepdim = ctx.keepdim

            # For mean, gradients are distributed evenly across reduced dimensions
            if dim is None:
                # Global mean - divide by total number of elements
                n = torch.tensor(input_shape).prod().item()
                new_pp = (delta_pp / n).expand(input_shape)
                new_np = (delta_np / n).expand(input_shape)
                new_pn = (delta_pn / n).expand(input_shape)
                new_nn = (delta_nn / n).expand(input_shape)
            else:
                # Mean along specific dimensions
                if isinstance(dim, int):
                    dims = [dim]
                else:
                    dims = list(dim)

                # Calculate number of elements along reduced dimensions
                n = 1
                for d in dims:
                    n *= input_shape[d]

                # Expand gradients back to original shape
                if not keepdim:
                    # Add back the reduced dimensions
                    for d in sorted(dims):
                        delta_pp = delta_pp.unsqueeze(d)
                        delta_np = delta_np.unsqueeze(d)
                        delta_pn = delta_pn.unsqueeze(d)
                        delta_nn = delta_nn.unsqueeze(d)

                new_pp = (delta_pp / n).expand(input_shape)
                new_np = (delta_np / n).expand(input_shape)
                new_pn = (delta_pn / n).expand(input_shape)
                new_nn = (delta_nn / n).expand(input_shape)

            return new_pp, new_np, new_pn, new_nn

        return BackwardBuilder.run(ctx, grad_4, compute, num_extra_returns=6)


def dc_forward_mean(module: Mean, input: Tensor) -> Tensor:
    """DC forward for mean operation."""
    cache, layer_name, alpha = get_cache_info(module)
    return DCMeanFunction.apply(
        input, module.dim, module.keepdim,
        getattr(module, DC_IS_OUTPUT_LAYER, False),
        cache, layer_name, alpha,
    )


patch_mean = create_patch_function(dc_forward_mean)
unpatch_mean = create_unpatch_function()
