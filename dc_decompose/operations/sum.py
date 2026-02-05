"""
DC Decomposition for torch.sum operation. Forward/Backward: [4*batch] -> [4*batch]

Sum operation is linear - pos and neg streams are summed independently.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Optional, Union

from .base import (
    split_input_4, make_output_4, make_grad_4,
    init_backward, recenter_forward,
    DC_ENABLED, DC_ORIGINAL_FORWARD, DC_IS_OUTPUT_LAYER
)


class Sum(nn.Module):
    """
    Module wrapper for torch.sum that can be patched for DC decomposition.
    
    This is used by functional_replacer to replace torch.sum calls.
    """
    
    def __init__(self, dim: Optional[Union[int, Tuple[int, ...]]] = None, keepdim: bool = False):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, input: Tensor) -> Tensor:
        if self.dim is None:
            return torch.sum(input)
        return torch.sum(input, dim=self.dim, keepdim=self.keepdim)

    def extra_repr(self) -> str:
        return f'dim={self.dim}, keepdim={self.keepdim}'


class DCSumFunction(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input_4: Tensor, dim, keepdim: bool,
                is_output_layer: bool) -> Tensor:
        pos, neg = split_input_4(input_4)
        
        # Apply sum to both streams
        if dim is None:
            out_pos = torch.sum(pos)
            out_neg = torch.sum(neg)
        else:
            out_pos = torch.sum(pos, dim=dim, keepdim=keepdim)
            out_neg = torch.sum(neg, dim=dim, keepdim=keepdim)
        
        ctx.input_shape = pos.shape
        ctx.dim = dim
        ctx.keepdim = keepdim
        ctx.is_output_layer = is_output_layer
        
        
        output = make_output_4(out_pos, out_neg)
        return recenter_forward(output)
    
    @staticmethod
    def backward(ctx, grad_4: Tensor) -> Tuple[Tensor, None, None, None, None]:
        input_shape = ctx.input_shape
        dim = ctx.dim
        keepdim = ctx.keepdim
        
        delta_pp, delta_np, delta_pn, delta_nn = init_backward(
            grad_4, ctx.is_output_layer)
        
        # For sum, gradients are broadcast back to original shape
        if dim is None:
            # Global sum - broadcast to original shape
            new_pp = delta_pp.expand(input_shape)
            new_np = delta_np.expand(input_shape)
            new_pn = delta_pn.expand(input_shape)
            new_nn = delta_nn.expand(input_shape)
        else:
            # Sum along specific dimensions
            if not keepdim:
                # Add back the reduced dimensions
                if isinstance(dim, int):
                    dims = [dim]
                else:
                    dims = list(dim)
                
                for d in sorted(dims):
                    delta_pp = delta_pp.unsqueeze(d)
                    delta_np = delta_np.unsqueeze(d)
                    delta_pn = delta_pn.unsqueeze(d)
                    delta_nn = delta_nn.unsqueeze(d)
            
            # Broadcast to original shape
            new_pp = delta_pp.expand(input_shape)
            new_np = delta_np.expand(input_shape)
            new_pn = delta_pn.expand(input_shape)
            new_nn = delta_nn.expand(input_shape)
        
        grad_input = make_grad_4(new_pp, new_np, new_pn, new_nn)
        return grad_input, None, None, None, None


def dc_forward_sum(module: Sum, input: Tensor) -> Tensor:
    """DC forward for sum operation."""
    return DCSumFunction.apply(
        input, module.dim, module.keepdim,
        getattr(module, DC_IS_OUTPUT_LAYER, False),
        0.5
    )


def patch_sum(module: Sum) -> None:
    """Patch Sum module for DC decomposition."""
    if hasattr(module, DC_ORIGINAL_FORWARD):
        return
    setattr(module, DC_ORIGINAL_FORWARD, module.forward)
    setattr(module, DC_ENABLED, True)
    setattr(module, DC_IS_OUTPUT_LAYER, False)
    

    def patched(input):
        if getattr(module, DC_ENABLED, False):
            return dc_forward_sum(module, input)
        else:
            return getattr(module, DC_ORIGINAL_FORWARD)(input)

    module.forward = patched


def unpatch_sum(module: Sum) -> None:
    """Unpatch Sum module."""
    if hasattr(module, DC_ORIGINAL_FORWARD):
        module.forward = getattr(module, DC_ORIGINAL_FORWARD)
        for attr in [DC_ORIGINAL_FORWARD, DC_ENABLED, DC_IS_OUTPUT_LAYER]:
            if hasattr(module, attr):
                delattr(module, attr)