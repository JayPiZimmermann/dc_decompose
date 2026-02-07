"""
DC Decomposition for tensor.contiguous() operation. Forward/Backward: [4*batch] -> [4*batch]

Contiguous operation is a memory layout operation that doesn't change values.
"""

import torch
import torch.nn as nn
from torch import Tensor

from .base import DC_IS_OUTPUT_LAYER
from .patch_builder import (
    ForwardBuilder, BackwardBuilder, get_cache_info,
    create_patch_function, create_unpatch_function,
)


class Contiguous(nn.Module):
    """
    Module wrapper for tensor.contiguous() that can be patched for DC decomposition.
    """

    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        return input.contiguous()


class DCContiguousFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_4: Tensor, is_output_layer: bool, cache, layer_name, alpha: float) -> Tensor:
        
        def compute(ctx, pos, neg):
            out_pos = pos.contiguous()
            out_neg = neg.contiguous()
            return out_pos, out_neg

        return ForwardBuilder.run(
            ctx, input_4, compute, is_output_layer, cache, layer_name, alpha
        )

    @staticmethod
    def backward(ctx, grad_4: Tensor):
        def compute(_ctx, delta_pp, delta_np, delta_pn, delta_nn):
            return delta_pp, delta_np, delta_pn, delta_nn

        return BackwardBuilder.run(ctx, grad_4, compute, num_extra_returns=4)


def dc_forward_contiguous(module: Contiguous, input: Tensor) -> Tensor:
    cache, layer_name, alpha = get_cache_info(module)
    return DCContiguousFunction.apply(
        input,
        getattr(module, DC_IS_OUTPUT_LAYER, False),
        cache, layer_name, alpha,
    )


patch_contiguous = create_patch_function(dc_forward_contiguous)
unpatch_contiguous = create_unpatch_function()
