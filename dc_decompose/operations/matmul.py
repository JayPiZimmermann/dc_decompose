"""DC Decomposition for Matrix Multiplication. Forward/Backward: [4*batch] -> [4*batch]"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional

from .base import (
    split_input_4, make_output_4, make_grad_4,
    init_backward, recenter_forward,
    DC_ENABLED, DC_ORIGINAL_FORWARD, DC_IS_OUTPUT_LAYER, DC_SPLIT_WEIGHTS_ON_FLY
)


class DCMatMul(nn.Module):
    """
    Matrix multiplication module that can be patched for DC decomposition.
    
    This module wraps torch.matmul and provides a patchable interface.
    Can be used for both learnable weights (like Linear without bias) 
    and dynamic matmul operations.
    
    Usage:
        # For functional replacement of torch.matmul:
        matmul = DCMatMul()
        matmul.set_operand(B)
        result = matmul(A)  # Computes A @ B
        
        # As learnable layer:
        matmul = DCMatMul(in_features=64, out_features=32)
        result = matmul(A)  # Computes A @ W^T
    """
    
    def __init__(
        self, 
        in_features: Optional[int] = None,
        out_features: Optional[int] = None,
        transpose_b: bool = False
    ):
        super().__init__()
        self.transpose_b = transpose_b
        
        # Learnable weight mode (like nn.Linear without bias)
        if in_features is not None and out_features is not None:
            self.weight = nn.Parameter(torch.empty(out_features, in_features))
            nn.init.kaiming_uniform_(self.weight)
            self._has_weight = True
        else:
            self.register_parameter('weight', None) 
            self._has_weight = False
            
        # Dynamic operand storage
        self._operand: Optional[Tensor] = None
        
    def set_operand(self, B: Tensor):
        """Set the second operand for matrix multiplication."""
        self._operand = B
        
    def forward(self, A: Tensor, B: Optional[Tensor] = None) -> Tensor:
        """Forward pass: compute A @ B (or A @ B^T)."""
        if self._has_weight:
            # Use learnable weight
            B_weight = self.weight.t() if not self.transpose_b else self.weight
            return torch.matmul(A, B_weight)
        else:
            # Use dynamic operand - can come from set_operand() or as argument
            if B is not None:
                # B provided as argument (from functional replacement)
                operand = B
            elif self._operand is not None:
                # B set via set_operand()
                operand = self._operand
            else:
                raise RuntimeError("Operand not provided. Either pass B as argument or call set_operand().")
                
            if self.transpose_b:
                operand = operand.transpose(-2, -1)
            return torch.matmul(A, operand)


class DCMatMulFunction(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input_4: Tensor, operand_4: Tensor, transpose_b: bool,
                is_output_layer: bool, beta: float) -> Tensor:
        A_pos, A_neg = split_input_4(input_4)
        B_pos, B_neg = split_input_4(operand_4)
        
        if transpose_b:
            B_pos = B_pos.transpose(-2, -1)
            B_neg = B_neg.transpose(-2, -1)
        
        # DC matmul: (A+ - A-)(B+ - B-) = (A+B+ + A-B-) - (A+B- + A-B+)
        out_pos = torch.matmul(A_pos, B_pos) + torch.matmul(A_neg, B_neg)
        out_neg = torch.matmul(A_pos, B_neg) + torch.matmul(A_neg, B_pos)
        
        # Save for backward
        ctx.save_for_backward(A_pos, A_neg, B_pos, B_neg)
        ctx.transpose_b = transpose_b
        ctx.is_output_layer = is_output_layer
        ctx.beta = beta
        
        output = make_output_4(out_pos, out_neg)
        return recenter_forward(output)
    
    @staticmethod  
    def backward(ctx, grad_4: Tensor) -> Tuple[Tensor, Tensor, None, None, None]:
        A_pos, A_neg, B_pos, B_neg = ctx.saved_tensors
        
        delta_pp, delta_np, delta_pn, delta_nn = init_backward(
            grad_4, ctx.is_output_layer, ctx.beta)
        
        # Backward w.r.t. A using product rule
        if ctx.transpose_b:
            B_pos_T = B_pos
            B_neg_T = B_neg
        else:
            B_pos_T = B_pos.transpose(-2, -1)
            B_neg_T = B_neg.transpose(-2, -1)
            
        new_pp_A = torch.matmul(delta_pp, B_pos_T) + torch.matmul(delta_pn, B_neg_T)
        new_np_A = torch.matmul(delta_pp, B_neg_T) + torch.matmul(delta_pn, B_pos_T)
        new_pn_A = torch.matmul(delta_np, B_pos_T) + torch.matmul(delta_nn, B_neg_T)
        new_nn_A = torch.matmul(delta_np, B_neg_T) + torch.matmul(delta_nn, B_pos_T)
        
        # Backward w.r.t. B using product rule
        A_pos_T = A_pos.transpose(-2, -1)
        A_neg_T = A_neg.transpose(-2, -1)
        
        if ctx.transpose_b:
            new_pp_B = torch.matmul(A_pos_T, delta_pp) + torch.matmul(A_neg_T, delta_pn)
            new_np_B = torch.matmul(A_neg_T, delta_pp) + torch.matmul(A_pos_T, delta_pn)
            new_pn_B = torch.matmul(A_pos_T, delta_np) + torch.matmul(A_neg_T, delta_nn)
            new_nn_B = torch.matmul(A_neg_T, delta_np) + torch.matmul(A_pos_T, delta_nn)
        else:
            new_pp_B = torch.matmul(A_pos_T, delta_pp) + torch.matmul(A_neg_T, delta_pn)
            new_np_B = torch.matmul(A_neg_T, delta_pp) + torch.matmul(A_pos_T, delta_pn)
            new_pn_B = torch.matmul(A_pos_T, delta_np) + torch.matmul(A_neg_T, delta_nn)
            new_nn_B = torch.matmul(A_neg_T, delta_np) + torch.matmul(A_pos_T, delta_nn)
        
        grad_A = make_grad_4(new_pp_A, new_np_A, new_pn_A, new_nn_A)
        grad_B = make_grad_4(new_pp_B, new_np_B, new_pn_B, new_nn_B)
        
        return grad_A, grad_B, None, None, None


def dc_forward_matmul(module: DCMatMul, A: Tensor, B: Optional[Tensor] = None) -> Tensor:
    """DC forward for matrix multiplication."""
    if module._has_weight:
        # Learnable weight mode
        B_weight = module.weight.t() if not module.transpose_b else module.weight
        operand_4 = make_output_4(F.relu(B_weight), F.relu(-B_weight))
    else:
        # Dynamic operand mode - can come from argument or set_operand()
        if B is not None:
            operand = B
        elif module._operand is not None:
            operand = module._operand
        else:
            raise RuntimeError("Operand not provided.")
        operand_4 = make_output_4(F.relu(operand), F.relu(-operand))
    
    return DCMatMulFunction.apply(
        A, operand_4, module.transpose_b,
        getattr(module, DC_IS_OUTPUT_LAYER, False),
        0.5
    )


def patch_dcmatmul(module: DCMatMul) -> None:
    """Patch DCMatMul module for DC decomposition."""
    if hasattr(module, DC_ORIGINAL_FORWARD):
        return
    setattr(module, DC_ORIGINAL_FORWARD, module.forward)
    setattr(module, DC_ENABLED, True)
    setattr(module, DC_IS_OUTPUT_LAYER, False)
    

    def patched(A, B=None):
        if getattr(module, DC_ENABLED, False):
            return dc_forward_matmul(module, A, B)
        else:
            return getattr(module, DC_ORIGINAL_FORWARD)(A, B)

    module.forward = patched


def unpatch_dcmatmul(module: DCMatMul) -> None:
    """Unpatch DCMatMul module."""
    if hasattr(module, DC_ORIGINAL_FORWARD):
        module.forward = getattr(module, DC_ORIGINAL_FORWARD)
        for attr in [DC_ORIGINAL_FORWARD, DC_ENABLED, DC_IS_OUTPUT_LAYER]:
            if hasattr(module, attr):
                delattr(module, attr)