"""
DC Decomposition for tensor.contiguous() operation. Forward/Backward: [4*batch] -> [4*batch]

Contiguous operation is a memory layout operation that doesn't change values - 
pos and neg streams are made contiguous independently.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple

from .base import (
    split_input_4, make_output_4, make_grad_4,
    init_backward, recenter_forward,
    DC_ENABLED, DC_ORIGINAL_FORWARD, DC_IS_OUTPUT_LAYER, DC_BETA
)


class Contiguous(nn.Module):
    """
    Module wrapper for tensor.contiguous() that can be patched for DC decomposition.
    
    This is used by functional_replacer to replace .contiguous() calls.
    """
    
    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        return input.contiguous()


class DCContiguousFunction(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input_4: Tensor, is_output_layer: bool, beta: float) -> Tensor:
        pos, neg = split_input_4(input_4)
        
        # Make both streams contiguous
        out_pos = pos.contiguous()
        out_neg = neg.contiguous()
        
        ctx.is_output_layer = is_output_layer
        ctx.beta = beta
        
        output = make_output_4(out_pos, out_neg)
        return recenter_forward(output)
    
    @staticmethod
    def backward(ctx, grad_4: Tensor) -> Tuple[Tensor, None, None]:
        delta_pp, delta_np, delta_pn, delta_nn = init_backward(
            grad_4, ctx.is_output_layer, ctx.beta)
        
        # Contiguous is a no-op for gradients - just pass them through
        grad_input = make_grad_4(delta_pp, delta_np, delta_pn, delta_nn)
        return grad_input, None, None


def dc_forward_contiguous(module: Contiguous, input: Tensor) -> Tensor:
    """DC forward for contiguous operation."""
    return DCContiguousFunction.apply(
        input,
        getattr(module, DC_IS_OUTPUT_LAYER, False),
        getattr(module, DC_BETA, 0.5)
    )


def patch_contiguous(module: Contiguous) -> None:
    """Patch Contiguous module for DC decomposition."""
    if hasattr(module, DC_ORIGINAL_FORWARD):
        return
    setattr(module, DC_ORIGINAL_FORWARD, module.forward)
    setattr(module, DC_ENABLED, True)
    setattr(module, DC_IS_OUTPUT_LAYER, False)
    setattr(module, DC_BETA, 0.5)

    def patched(input):
        if getattr(module, DC_ENABLED, False):
            return dc_forward_contiguous(module, input)
        else:
            return getattr(module, DC_ORIGINAL_FORWARD)(input)

    module.forward = patched


def unpatch_contiguous(module: Contiguous) -> None:
    """Unpatch Contiguous module."""
    if hasattr(module, DC_ORIGINAL_FORWARD):
        module.forward = getattr(module, DC_ORIGINAL_FORWARD)
        for attr in [DC_ORIGINAL_FORWARD, DC_ENABLED, DC_IS_OUTPUT_LAYER, DC_BETA]:
            if hasattr(module, attr):
                delattr(module, attr)