"""
DC Decomposition for torch.mean operation. Forward/Backward: [4*batch] -> [4*batch]

Mean operation is linear - pos and neg streams are averaged independently.
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
                is_output_layer: bool, beta: float) -> Tensor:
        pos, neg = split_input_4(input_4)
        
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
        ctx.is_output_layer = is_output_layer
        ctx.beta = beta
        
        output = make_output_4(out_pos, out_neg)
        return recenter_forward(output)
    
    @staticmethod
    def backward(ctx, grad_4: Tensor) -> Tuple[Tensor, None, None, None, None]:
        input_shape = ctx.input_shape
        dim = ctx.dim
        keepdim = ctx.keepdim
        
        delta_pp, delta_np, delta_pn, delta_nn = init_backward(
            grad_4, ctx.is_output_layer, ctx.beta)
        
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
        
        grad_input = make_grad_4(new_pp, new_np, new_pn, new_nn)
        return grad_input, None, None, None, None


def dc_forward_mean(module: Mean, input: Tensor) -> Tensor:
    """DC forward for mean operation."""
    return DCMeanFunction.apply(
        input, module.dim, module.keepdim,
        getattr(module, DC_IS_OUTPUT_LAYER, False),
        0.5
    )


def patch_mean(module: Mean) -> None:
    """Patch Mean module for DC decomposition."""
    if hasattr(module, DC_ORIGINAL_FORWARD):
        return
    setattr(module, DC_ORIGINAL_FORWARD, module.forward)
    setattr(module, DC_ENABLED, True)
    setattr(module, DC_IS_OUTPUT_LAYER, False)
    

    def patched(input):
        if getattr(module, DC_ENABLED, False):
            return dc_forward_mean(module, input)
        else:
            return getattr(module, DC_ORIGINAL_FORWARD)(input)

    module.forward = patched


def unpatch_mean(module: Mean) -> None:
    """Unpatch Mean module."""
    if hasattr(module, DC_ORIGINAL_FORWARD):
        module.forward = getattr(module, DC_ORIGINAL_FORWARD)
        for attr in [DC_ORIGINAL_FORWARD, DC_ENABLED, DC_IS_OUTPUT_LAYER]:
            if hasattr(module, attr):
                delattr(module, attr)