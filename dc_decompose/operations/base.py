"""
Base utilities for DC decomposition.

Both forward and backward use [4*batch, ...] format for autograd compatibility:
- Forward input: [pos; neg; pos; neg] (last two are duplicates)
- Forward output: [out_pos; out_neg; out_pos; out_neg]
- Backward: [delta_pp; delta_np; delta_pn; delta_nn]

This ensures autograd shape compatibility (backward returns same shape as input).
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import torch
from torch import Tensor


class ReLUMode(Enum):
    MAX = "max"
    MIN = "min"
    HALF = "half"


class InputMode(Enum):
    CENTER = "center"      # pos = ReLU(x), neg = ReLU(-x)
    POSITIVE = "positive"  # pos = x, neg = 0
    NEGATIVE = "negative"  # pos = 0, neg = -x


# Module attribute names
DC_ENABLED = '_dc_enabled'
DC_ORIGINAL_FORWARD = '_dc_original_forward'
DC_RELU_MODE = '_dc_relu_mode'
DC_IS_OUTPUT_LAYER = '_dc_is_output_layer'
DC_BETA = '_dc_beta'


# =============================================================================
# Core: [4*batch] format for both forward and backward
# =============================================================================

def cat4(t0: Tensor, t1: Tensor, t2: Tensor, t3: Tensor) -> Tensor:
    """Concatenate 4 tensors along batch dim."""
    return torch.cat([t0, t1, t2, t3], dim=0)


def split4(catted: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Split [4*batch] into 4 parts."""
    q = catted.shape[0] // 4
    return catted[:q], catted[q:2*q], catted[2*q:3*q], catted[3*q:]


def get_batch_size(catted: Tensor) -> int:
    """Get original batch size from [4*batch] tensor."""
    return catted.shape[0] // 4


# =============================================================================
# Forward: create [4*batch] = [pos; neg; pos; neg]
# =============================================================================

def make_input_4(pos: Tensor, neg: Tensor) -> Tensor:
    """Create [4*batch] forward input: [pos; neg; pos; neg] (duplicated)."""
    return torch.cat([pos, neg, pos, neg], dim=0)


def split_input_4(catted: Tensor) -> Tuple[Tensor, Tensor]:
    """Extract pos and neg from [4*batch] forward tensor (use first 2 quarters)."""
    q = catted.shape[0] // 4
    return catted[:q], catted[q:2*q]


def make_output_4(out_pos: Tensor, out_neg: Tensor) -> Tensor:
    """Create [4*batch] forward output: [out_pos; out_neg; out_pos; out_neg]."""
    return torch.cat([out_pos, out_neg, out_pos, out_neg], dim=0)


# =============================================================================
# Backward: [4*batch] = [delta_pp; delta_np; delta_pn; delta_nn]
# =============================================================================

def split_grad_4(catted: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Split [4*batch] backward gradient into 4 sensitivities."""
    return split4(catted)


def make_grad_4(delta_pp: Tensor, delta_np: Tensor,
                delta_pn: Tensor, delta_nn: Tensor) -> Tensor:
    """Concatenate 4 sensitivities into [4*batch]."""
    return cat4(delta_pp, delta_np, delta_pn, delta_nn)


# =============================================================================
# Input initialization
# =============================================================================

def init_pos_neg(x: Tensor, mode: InputMode = InputMode.CENTER) -> Tuple[Tensor, Tensor]:
    """Split input into pos/neg streams."""
    if mode == InputMode.CENTER:
        return torch.relu(x), torch.relu(-x)
    elif mode == InputMode.POSITIVE:
        return x, torch.zeros_like(x)
    elif mode == InputMode.NEGATIVE:
        return torch.zeros_like(x), -x
    raise ValueError(f"Unknown mode: {mode}")


def init_catted(x: Tensor, mode: InputMode = InputMode.CENTER) -> Tensor:
    """Initialize [4*batch] input for DC forward: [pos; neg; pos; neg]."""
    pos, neg = init_pos_neg(x, mode)
    return make_input_4(pos, neg)


# =============================================================================
# Output reconstruction
# =============================================================================

def reconstruct_output(output_4: Tensor) -> Tensor:
    """Reconstruct original output from [4*batch]: out_pos - out_neg."""
    q = output_4.shape[0] // 4
    out_pos = output_4[:q]
    out_neg = output_4[q:2*q]
    return out_pos - out_neg


# =============================================================================
# Re-centering (prevents magnitude explosion in residual networks)
# =============================================================================

def recenter_dc(tensor_4: Tensor) -> Tensor:
    """
    Re-center DC representation to minimize pos and neg magnitudes.

    Given [pos; neg; pos; neg] where z = pos - neg:
    - Computes new_pos = ReLU(z), new_neg = ReLU(-z)
    - Returns [new_pos; new_neg; new_pos; new_neg]

    This preserves z = new_pos - new_neg = ReLU(z) - ReLU(-z) = z,
    but ensures pos and neg have minimal magnitudes (one of them is 0 for each element).

    Use this after residual additions to prevent exponential magnitude growth.
    """
    q = tensor_4.shape[0] // 4
    pos = tensor_4[:q]
    neg = tensor_4[q:2*q]

    z = pos - neg
    new_pos = torch.relu(z)
    new_neg = torch.relu(-z)

    return make_input_4(new_pos, new_neg)


# =============================================================================
# Legacy aliases
# =============================================================================

def cat2(pos: Tensor, neg: Tensor) -> Tensor:
    """Legacy: use make_input_4 instead."""
    return torch.cat([pos, neg], dim=0)


def split2(catted: Tensor) -> Tuple[Tensor, Tensor]:
    """Legacy: use split_input_4 instead."""
    h = catted.shape[0] // 2
    return catted[:h], catted[h:]


# =============================================================================
# Cache
# =============================================================================

@dataclass
class DCCache:
    input_pos: Optional[Tensor] = None
    input_neg: Optional[Tensor] = None
    output_pos: Optional[Tensor] = None
    output_neg: Optional[Tensor] = None
    z_before: Optional[Tensor] = None
    pool_indices: Optional[Tensor] = None
    batch_size: Optional[int] = None

    def clear(self):
        for attr in ['input_pos', 'input_neg', 'output_pos', 'output_neg',
                     'z_before', 'pool_indices', 'batch_size']:
            setattr(self, attr, None)
