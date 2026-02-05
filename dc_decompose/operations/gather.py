"""
DC Decomposition for torch.gather operation. Forward/Backward: [4*batch] -> [4*batch]

Gather operation is linear - the same indices are applied to both pos and neg streams.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple

from .base import (
    split_input_4, make_output_4, make_grad_4,
    init_backward, recenter_forward,
    DC_ENABLED, DC_ORIGINAL_FORWARD, DC_IS_OUTPUT_LAYER
)


class Gather(nn.Module):
    """
    Module wrapper for torch.gather that can be patched for DC decomposition.
    
    This is used by functional_replacer to replace torch.gather calls.
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
                is_output_layer: bool) -> Tensor:
        pos, neg = split_input_4(input_4)
        
        # Apply gather to both streams with the same indices
        out_pos = torch.gather(pos, dim, index)
        out_neg = torch.gather(neg, dim, index)
        
        ctx.save_for_backward(input_4, index)
        ctx.dim = dim
        ctx.is_output_layer = is_output_layer
        
        
        output = make_output_4(out_pos, out_neg)
        return recenter_forward(output)
    
    @staticmethod
    def backward(ctx, grad_4: Tensor) -> Tuple[Tensor, None, None, None, None]:
        input_4, index = ctx.saved_tensors
        
        delta_pp, delta_np, delta_pn, delta_nn = init_backward(
            grad_4, ctx.is_output_layer)
        
        # Scatter gradients back using the same indices
        input_shape = split_input_4(input_4)[0].shape
        
        new_pp = torch.zeros(input_shape, dtype=delta_pp.dtype, device=delta_pp.device)
        new_np = torch.zeros(input_shape, dtype=delta_np.dtype, device=delta_np.device)
        new_pn = torch.zeros(input_shape, dtype=delta_pn.dtype, device=delta_pn.device)
        new_nn = torch.zeros(input_shape, dtype=delta_nn.dtype, device=delta_nn.device)
        
        new_pp.scatter_add_(ctx.dim, index, delta_pp)
        new_np.scatter_add_(ctx.dim, index, delta_np)
        new_pn.scatter_add_(ctx.dim, index, delta_pn)
        new_nn.scatter_add_(ctx.dim, index, delta_nn)
        
        grad_input = make_grad_4(new_pp, new_np, new_pn, new_nn)
        
        return grad_input, None, None, None, None


def dc_forward_gather(module: Gather, input: Tensor, index: Tensor) -> Tensor:
    """DC forward for gather operation."""
    return DCGatherFunction.apply(
        input, index, module.dim,
        getattr(module, DC_IS_OUTPUT_LAYER, False),
        0.5
    )


def patch_gather(module: Gather) -> None:
    """Patch Gather module for DC decomposition."""
    if hasattr(module, DC_ORIGINAL_FORWARD):
        return
    setattr(module, DC_ORIGINAL_FORWARD, module.forward)
    setattr(module, DC_ENABLED, True)
    setattr(module, DC_IS_OUTPUT_LAYER, False)
    

    def patched(input, index):
        if getattr(module, DC_ENABLED, False):
            return dc_forward_gather(module, input, index)
        else:
            return getattr(module, DC_ORIGINAL_FORWARD)(input, index)

    module.forward = patched


def unpatch_gather(module: Gather) -> None:
    """Unpatch Gather module."""
    if hasattr(module, DC_ORIGINAL_FORWARD):
        module.forward = getattr(module, DC_ORIGINAL_FORWARD)
        for attr in [DC_ORIGINAL_FORWARD, DC_ENABLED, DC_IS_OUTPUT_LAYER]:
            if hasattr(module, attr):
                delattr(module, attr)