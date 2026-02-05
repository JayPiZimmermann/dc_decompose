"""DC Decomposition for Element-wise Multiplication. Forward/Backward: [4*batch] -> [4*batch]"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional

from .base import (
    split_input_4, make_output_4, make_grad_4,
    init_backward, recenter_forward,
    DC_ENABLED, DC_ORIGINAL_FORWARD, DC_IS_OUTPUT_LAYER
)


class DCMul(nn.Module):
    """
    Element-wise multiplication module that can be patched for DC decomposition.
    
    This module wraps element-wise multiplication (* operator, torch.mul) and 
    provides a patchable interface for DC decomposition.
    
    Usage:
        # For functional replacement of torch.mul or * operator:
        mul = DCMul()
        result = mul(A, B)  # Computes A * B element-wise
        
        # With scalar multiplication:
        mul = DCMul(scalar=2.0)
        result = mul(A)  # Computes A * 2.0
    """
    
    def __init__(self, scalar: Optional[float] = None):
        super().__init__()
        self.scalar = scalar
        
    def forward(self, A: Tensor, B: Optional[Tensor] = None) -> Tensor:
        """Forward pass: compute A * B element-wise."""
        if self.scalar is not None:
            # Scalar multiplication
            return A * self.scalar
        elif B is not None:
            # Element-wise multiplication
            return A * B
        else:
            raise ValueError("Either provide B tensor or set scalar in __init__")


class DCMulFunction(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input_4: Tensor, operand_4: Tensor,
                is_output_layer: bool, beta: float) -> Tensor:
        A_pos, A_neg = split_input_4(input_4)
        B_pos, B_neg = split_input_4(operand_4)
        
        # DC element-wise mul: (A+ - A-)(B+ - B-) = (A+B+ + A-B-) - (A+B- + A-B+)
        out_pos = A_pos * B_pos + A_neg * B_neg
        out_neg = A_pos * B_neg + A_neg * B_pos
        
        # Save for backward
        ctx.save_for_backward(A_pos, A_neg, B_pos, B_neg)
        ctx.is_output_layer = is_output_layer
        ctx.beta = beta
        
        output = make_output_4(out_pos, out_neg)
        return recenter_forward(output)
    
    @staticmethod  
    def backward(ctx, grad_4: Tensor) -> Tuple[Tensor, Tensor, None, None]:
        A_pos, A_neg, B_pos, B_neg = ctx.saved_tensors
        
        delta_pp, delta_np, delta_pn, delta_nn = init_backward(
            grad_4, ctx.is_output_layer, ctx.beta)
        
        # Backward w.r.t. A using product rule
        new_pp_A = delta_pp * B_pos + delta_pn * B_neg
        new_np_A = delta_pp * B_neg + delta_pn * B_pos
        new_pn_A = delta_np * B_pos + delta_nn * B_neg
        new_nn_A = delta_np * B_neg + delta_nn * B_pos
        
        # Backward w.r.t. B using product rule
        new_pp_B = delta_pp * A_pos + delta_pn * A_neg
        new_np_B = delta_pp * A_neg + delta_pn * A_pos
        new_pn_B = delta_np * A_pos + delta_nn * A_neg
        new_nn_B = delta_np * A_neg + delta_nn * A_pos
        
        grad_A = make_grad_4(new_pp_A, new_np_A, new_pn_A, new_nn_A)
        grad_B = make_grad_4(new_pp_B, new_np_B, new_pn_B, new_nn_B)
        
        return grad_A, grad_B, None, None


class DCScalarMulFunction(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input_4: Tensor, scalar: float,
                is_output_layer: bool, beta: float) -> Tensor:
        A_pos, A_neg = split_input_4(input_4)
        
        # DC scalar mul: (A+ - A-) * c = A+ * c - A- * c
        if scalar >= 0:
            out_pos = A_pos * scalar
            out_neg = A_neg * scalar
        else:
            # Negative scalar flips sign
            out_pos = A_neg * abs(scalar)
            out_neg = A_pos * abs(scalar)
        
        ctx.scalar = scalar
        ctx.is_output_layer = is_output_layer
        ctx.beta = beta
        
        output = make_output_4(out_pos, out_neg)
        return recenter_forward(output)
    
    @staticmethod  
    def backward(ctx, grad_4: Tensor) -> Tuple[Tensor, None, None, None]:
        delta_pp, delta_np, delta_pn, delta_nn = init_backward(
            grad_4, ctx.is_output_layer, ctx.beta)
        
        # Backward for scalar multiplication
        scalar = ctx.scalar
        if scalar >= 0:
            new_pp = delta_pp * scalar
            new_np = delta_np * scalar
            new_pn = delta_pn * scalar
            new_nn = delta_nn * scalar
        else:
            # Negative scalar flips sign
            abs_scalar = abs(scalar)
            new_pp = delta_pn * abs_scalar
            new_np = delta_nn * abs_scalar
            new_pn = delta_pp * abs_scalar
            new_nn = delta_np * abs_scalar
        
        grad_A = make_grad_4(new_pp, new_np, new_pn, new_nn)
        
        return grad_A, None, None, None


def dc_forward_mul(module: DCMul, A: Tensor, B: Optional[Tensor] = None) -> Tensor:
    """DC forward for element-wise multiplication."""
    if module.scalar is not None:
        # Scalar multiplication
        return DCScalarMulFunction.apply(
            A, module.scalar,
            getattr(module, DC_IS_OUTPUT_LAYER, False),
            0.5
        )
    elif B is not None:
        # Element-wise multiplication
        operand_4 = make_output_4(F.relu(B), F.relu(-B))
        return DCMulFunction.apply(
            A, operand_4,
            getattr(module, DC_IS_OUTPUT_LAYER, False),
            0.5
        )
    else:
        raise ValueError("Either provide B tensor or set scalar in __init__")


def patch_dcmul(module: DCMul) -> None:
    """Patch DCMul module for DC decomposition."""
    if hasattr(module, DC_ORIGINAL_FORWARD):
        return
    setattr(module, DC_ORIGINAL_FORWARD, module.forward)
    setattr(module, DC_ENABLED, True)
    setattr(module, DC_IS_OUTPUT_LAYER, False)
    

    def patched(A, B=None):
        if getattr(module, DC_ENABLED, False):
            return dc_forward_mul(module, A, B)
        else:
            return getattr(module, DC_ORIGINAL_FORWARD)(A, B)

    module.forward = patched


def unpatch_dcmul(module: DCMul) -> None:
    """Unpatch DCMul module."""
    if hasattr(module, DC_ORIGINAL_FORWARD):
        module.forward = getattr(module, DC_ORIGINAL_FORWARD)
        for attr in [DC_ORIGINAL_FORWARD, DC_ENABLED, DC_IS_OUTPUT_LAYER]:
            if hasattr(module, attr):
                delattr(module, attr)