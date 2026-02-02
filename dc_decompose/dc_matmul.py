"""
DC MatMul Module for DC Decomposition

This module provides a wrapper for matrix multiplication that supports
DC decomposition with pos/neg streams.

Forward: (A+ - A-)(B+ - B-) = (A+B+ + A-B-) - (A+B- + A-B+)
Backward: Uses product rule for computing 4 sensitivities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple


class DCMatMul(nn.Module):
    """
    Matrix multiplication module that supports DC decomposition.

    This module wraps torch.matmul and provides hooks for DC decomposition.
    The second operand (B) can be either:
    1. A learnable weight matrix (like nn.Linear without bias)
    2. Set dynamically before forward pass (for attention-style matmul)

    For DC decomposition:
    (A+ - A-)(B+ - B-) = (A+B+ + A-B-) - (A+B- + A-B+)

    Usage:
        # As a learnable layer:
        matmul = DCMatMul(in_features=64, out_features=32)
        output = matmul(x)

        # As dynamic matmul (e.g., for attention):
        matmul = DCMatMul()
        matmul.set_operand(key_tensor)  # Sets B
        output = matmul(query_tensor)   # Computes Q @ K
    """

    def __init__(
        self,
        in_features: Optional[int] = None,
        out_features: Optional[int] = None,
        bias: bool = False,
        transpose_b: bool = False,
    ):
        """
        Initialize DCMatMul.

        Args:
            in_features: Input dimension (if using learnable weight)
            out_features: Output dimension (if using learnable weight)
            bias: Whether to add bias (only for learnable weight mode)
            transpose_b: If True, transpose B before multiplication (A @ B^T)
        """
        super().__init__()

        self._dc_is_matmul = True  # Flag for HookDecomposer to identify this module
        self.transpose_b = transpose_b

        # Learnable weight mode
        if in_features is not None and out_features is not None:
            self.weight = nn.Parameter(torch.empty(out_features, in_features))
            nn.init.kaiming_uniform_(self.weight)
            if bias:
                self.bias = nn.Parameter(torch.zeros(out_features))
            else:
                self.register_parameter('bias', None)
            self._has_weight = True
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
            self._has_weight = False

        # Dynamic operand storage (for attention-style usage)
        self._dc_operand_pos: Optional[Tensor] = None
        self._dc_operand_neg: Optional[Tensor] = None

    def set_operand(self, B: Tensor, B_pos: Optional[Tensor] = None, B_neg: Optional[Tensor] = None):
        """
        Set the second operand for matrix multiplication.

        Args:
            B: The original second operand tensor
            B_pos: Positive component (if already decomposed)
            B_neg: Negative component (if already decomposed)

        If B_pos and B_neg are not provided, B is decomposed using ReLU:
            B_pos = ReLU(B), B_neg = ReLU(-B)
        """
        if B_pos is not None and B_neg is not None:
            self._dc_operand_pos = B_pos
            self._dc_operand_neg = B_neg
        else:
            # Decompose B into pos/neg
            self._dc_operand_pos = F.relu(B)
            self._dc_operand_neg = F.relu(-B)

    def set_operand_decomposed(self, B_pos: Tensor, B_neg: Tensor):
        """
        Set the second operand with pre-decomposed pos/neg components.

        Args:
            B_pos: Positive component
            B_neg: Negative component
        """
        self._dc_operand_pos = B_pos
        self._dc_operand_neg = B_neg

    def _setup_weight_decomposition(self):
        """Pre-compute weight decomposition for learnable weight mode."""
        if self._has_weight:
            # Weight is stored as (out_features, in_features)
            # For A @ W^T, we need to transpose, then decompose
            W = self.weight
            if not self.transpose_b:
                W = W.t()  # Transpose to (in_features, out_features) for A @ W
            self._dc_operand_pos = F.relu(W)
            self._dc_operand_neg = F.relu(-W)

    def forward(self, A: Tensor) -> Tensor:
        """
        Forward pass: compute A @ B (or A @ B^T if transpose_b=True).

        Args:
            A: First operand tensor

        Returns:
            Result of matrix multiplication
        """
        if self._has_weight:
            # Use learnable weight (stored as out_features x in_features, like nn.Linear)
            # So we need to transpose for A @ W^T
            W = self.weight
            if self.transpose_b:
                output = torch.matmul(A, W)  # A @ W (no transpose)
            else:
                output = torch.matmul(A, W.t())  # A @ W^T (default, like nn.Linear)
            if self.bias is not None:
                output = output + self.bias
            return output
        else:
            # Use dynamically set operand
            if self._dc_operand_pos is None:
                raise RuntimeError("Second operand not set. Call set_operand() before forward().")
            B = self._dc_operand_pos - self._dc_operand_neg
            if self.transpose_b:
                B = B.transpose(-2, -1)
            return torch.matmul(A, B)


class DCMatMulFunction:
    """
    Functional interface for DC matrix multiplication.

    This provides static methods for DC matmul without requiring a module.
    Useful for integrating into existing code.
    """

    @staticmethod
    def forward(
        A_pos: Tensor, A_neg: Tensor,
        B_pos: Tensor, B_neg: Tensor,
        transpose_b: bool = False
    ) -> Tuple[Tensor, Tensor]:
        """
        DC matrix multiplication forward pass.

        (A+ - A-)(B+ - B-) = (A+B+ + A-B-) - (A+B- + A-B+)

        Args:
            A_pos: Positive component of first operand
            A_neg: Negative component of first operand
            B_pos: Positive component of second operand
            B_neg: Negative component of second operand
            transpose_b: If True, transpose B before multiplication

        Returns:
            Tuple of (output_pos, output_neg)
        """
        if transpose_b:
            B_pos = B_pos.transpose(-2, -1)
            B_neg = B_neg.transpose(-2, -1)

        output_pos = torch.matmul(A_pos, B_pos) + torch.matmul(A_neg, B_neg)
        output_neg = torch.matmul(A_pos, B_neg) + torch.matmul(A_neg, B_pos)

        return output_pos, output_neg

    @staticmethod
    def backward(
        delta_pp: Tensor, delta_np: Tensor, delta_pn: Tensor, delta_nn: Tensor,
        B_pos: Tensor, B_neg: Tensor,
        transpose_b: bool = False
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        DC matrix multiplication backward pass (w.r.t. A).

        Uses product rule to compute 4 sensitivities.

        Args:
            delta_pp, delta_np, delta_pn, delta_nn: Incoming sensitivities
            B_pos, B_neg: Second operand's pos/neg components
            transpose_b: If True, B was transposed in forward

        Returns:
            Tuple of (new_delta_pp, new_delta_np, new_delta_pn, new_delta_nn)
        """
        if transpose_b:
            # If B was transposed in forward, we need B (not B^T) for backward
            B_pos_T = B_pos
            B_neg_T = B_neg
        else:
            B_pos_T = B_pos.transpose(-2, -1)
            B_neg_T = B_neg.transpose(-2, -1)

        new_delta_pp = torch.matmul(delta_pp, B_pos_T) + torch.matmul(delta_pn, B_neg_T)
        new_delta_np = torch.matmul(delta_pp, B_neg_T) + torch.matmul(delta_pn, B_pos_T)
        new_delta_pn = torch.matmul(delta_np, B_pos_T) + torch.matmul(delta_nn, B_neg_T)
        new_delta_nn = torch.matmul(delta_np, B_neg_T) + torch.matmul(delta_nn, B_pos_T)

        return new_delta_pp, new_delta_np, new_delta_pn, new_delta_nn

    @staticmethod
    def backward_wrt_b(
        delta_pp: Tensor, delta_np: Tensor, delta_pn: Tensor, delta_nn: Tensor,
        A_pos: Tensor, A_neg: Tensor,
        transpose_b: bool = False
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        DC matrix multiplication backward pass (w.r.t. B).

        Uses product rule to compute 4 sensitivities for B.

        Args:
            delta_pp, delta_np, delta_pn, delta_nn: Incoming sensitivities
            A_pos, A_neg: First operand's pos/neg components
            transpose_b: If True, B was transposed in forward

        Returns:
            Tuple of (new_delta_pp, new_delta_np, new_delta_pn, new_delta_nn) for B
        """
        A_pos_T = A_pos.transpose(-2, -1)
        A_neg_T = A_neg.transpose(-2, -1)

        if transpose_b:
            # d(A @ B^T)/dB = A^T @ grad
            new_delta_pp = torch.matmul(A_pos_T, delta_pp) + torch.matmul(A_neg_T, delta_pn)
            new_delta_np = torch.matmul(A_neg_T, delta_pp) + torch.matmul(A_pos_T, delta_pn)
            new_delta_pn = torch.matmul(A_pos_T, delta_np) + torch.matmul(A_neg_T, delta_nn)
            new_delta_nn = torch.matmul(A_neg_T, delta_np) + torch.matmul(A_pos_T, delta_nn)
        else:
            # d(A @ B)/dB = A^T @ grad
            new_delta_pp = torch.matmul(A_pos_T, delta_pp) + torch.matmul(A_neg_T, delta_pn)
            new_delta_np = torch.matmul(A_neg_T, delta_pp) + torch.matmul(A_pos_T, delta_pn)
            new_delta_pn = torch.matmul(A_pos_T, delta_np) + torch.matmul(A_neg_T, delta_nn)
            new_delta_nn = torch.matmul(A_neg_T, delta_np) + torch.matmul(A_pos_T, delta_nn)

        return new_delta_pp, new_delta_np, new_delta_pn, new_delta_nn
