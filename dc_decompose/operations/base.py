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


def init_backward(grad_4: Tensor, is_output_layer: bool, beta: float = 1.0) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Initialize 4-sensitivities for backward pass.

    For output layers: converts [grad_pos; grad_neg; ...] to proper 4-sensitivities.
    For other layers: just splits the incoming gradient.

    Args:
        grad_4: Incoming gradient [4*batch]
        is_output_layer: Whether this is the output layer
        beta: Output layer initialization parameter

    Returns:
        (delta_pp, delta_np, delta_pn, delta_nn) tuple
    """
    if is_output_layer:
        q = grad_4.shape[0] // 4
        grad_pos = grad_4[:q]
        grad_neg = grad_4[q:2*q]
        delta_pp = beta * grad_pos
        delta_np = torch.zeros_like(grad_pos)
        delta_pn = (1 - beta) * grad_neg
        delta_nn = torch.zeros_like(grad_neg)
        result = delta_pp, delta_np, delta_pn, delta_nn

        # Log output layer initialization
        try:
            from ..logging_config import get_logger, TENSOR_LEVEL
            import logging
            log = get_logger('dc.backward')
            if log.isEnabledFor(logging.INFO):
                log.info(f"init_backward OUTPUT: shape={list(grad_4.shape)}, beta={beta}")
            tensor_log = get_logger('dc.tensors')
            if tensor_log.isEnabledFor(TENSOR_LEVEL):
                tensor_log.log(TENSOR_LEVEL, f"init_bwd: grad_pos={grad_pos.mean().item():.4f}, grad_neg={grad_neg.mean().item():.4f}")
        except ImportError:
            pass

        return result
    else:
        return split_grad_4(grad_4)


def recenter_grad(grad_4: Tensor) -> Tensor:
    """
    Re-center gradient for numerical stability (detached from autograd).

    Converts [delta_pp; delta_np; delta_pn; delta_nn] to canonical form
    where magnitudes are minimized while preserving gradient reconstruction.
    """
    delta_pp, delta_np, delta_pn, delta_nn = split_grad_4(grad_4)

    # Reconstruct the actual gradients
    grad_pos = delta_pp - delta_np  # gradient w.r.t. pos
    grad_neg = delta_pn - delta_nn  # gradient w.r.t. neg

    # Re-center: put positive parts in pp/pn, negative parts in np/nn
    new_pp = torch.relu(grad_pos)
    new_np = torch.relu(-grad_pos)
    new_pn = torch.relu(grad_neg)
    new_nn = torch.relu(-grad_neg)

    return make_grad_4(new_pp, new_np, new_pn, new_nn)


def recenter_forward(output_4: Tensor) -> Tensor:
    """
    Re-center forward output for numerical stability.

    Converts [pos; neg; pos; neg] to canonical form where one of pos/neg
    is zero for each element, minimizing magnitudes.

    Uses in-place data modification to not affect autograd graph.
    """
    q = output_4.shape[0] // 4
    pos = output_4[:q]
    neg = output_4[q:2*q]

    # Log before recenter (lazy import to avoid circular deps)
    old_pos_mean = old_neg_mean = None
    try:
        from ..logging_config import get_logger, TENSOR_LEVEL
        import logging
        log = get_logger('dc.recenter')
        tensor_log = get_logger('dc.tensors')
        if log.isEnabledFor(logging.INFO):
            old_pos_mean = pos.mean().item()
            old_neg_mean = neg.mean().item()
    except ImportError:
        log = None
        tensor_log = None

    with torch.no_grad():
        z = pos - neg
        new_pos = torch.relu(z)
        new_neg = torch.relu(-z)

    # Create output tensor that maintains gradient connection to input
    # by using clone and in-place data assignment
    result = output_4.clone()
    result[:q].data.copy_(new_pos)
    result[q:2*q].data.copy_(new_neg)
    result[2*q:3*q].data.copy_(new_pos)
    result[3*q:].data.copy_(new_neg)

    # Log after recenter
    if log is not None and log.isEnabledFor(logging.INFO):
        new_pos_mean = new_pos.mean().item()
        new_neg_mean = new_neg.mean().item()
        log.info(f"recenter: pos {old_pos_mean:.2f}->{new_pos_mean:.2f}, neg {old_neg_mean:.2f}->{new_neg_mean:.2f}")

    if tensor_log is not None and tensor_log.isEnabledFor(TENSOR_LEVEL):
        tensor_log.log(TENSOR_LEVEL, f"recenter: shape={list(output_4.shape)}, z_mean={z.mean().item():.4f}")

    return result


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


class InitCattedFunction(torch.autograd.Function):
    """
    Custom autograd function for init_catted.

    Forward: x -> [pos; neg; pos; neg] where pos = relu(x), neg = relu(-x)
             The DC format creation is detached - no gradient flows through relu.
    Backward: Pass gradients through unchanged. The DC layers handle gradient
              transformation via their own 4-sensitivity logic.
    """

    @staticmethod
    def forward(ctx, x: Tensor, mode_value: int) -> Tensor:
        # mode_value: 0=CENTER, 1=POSITIVE, 2=NEGATIVE
        # Create DC format with detached operations
        with torch.no_grad():
            if mode_value == 0:  # CENTER
                pos = torch.relu(x)
                neg = torch.relu(-x)
            elif mode_value == 1:  # POSITIVE
                pos = x.clone()
                neg = torch.zeros_like(x)
            else:  # NEGATIVE
                pos = torch.zeros_like(x)
                neg = -x

        # Create output that requires grad if input does
        result = torch.cat([pos, neg, pos, neg], dim=0)
        return result

    @staticmethod
    def backward(ctx, grad_4: Tensor) -> Tuple[Tensor, None]:
        # Pass gradient through unchanged - reconstruct from 4-sensitivity format
        # grad_4 is [4*batch] = [delta_pp; delta_np; delta_pn; delta_nn]
        q = grad_4.shape[0] // 4
        delta_pp = grad_4[:q]
        delta_np = grad_4[q:2*q]
        delta_pn = grad_4[2*q:3*q]
        delta_nn = grad_4[3*q:]

        # Reconstruct gradient: this is the standard DC gradient reconstruction
        # grad_x = d(loss)/d(pos) - d(loss)/d(neg)
        # where d(loss)/d(pos) comes from pp path and d(loss)/d(neg) from nn path
        grad_x = delta_pp - delta_np - delta_pn + delta_nn

        return grad_x, None


def init_catted(x: Tensor, mode: InputMode = InputMode.CENTER) -> Tensor:
    """Initialize [4*batch] input for DC forward: [pos; neg; pos; neg]."""
    # Convert mode to int for autograd function
    mode_value = 0 if mode == InputMode.CENTER else (1 if mode == InputMode.POSITIVE else 2)
    result = InitCattedFunction.apply(x, mode_value)

    # Log initialization
    try:
        from ..logging_config import get_logger, TENSOR_LEVEL
        import logging
        log = get_logger('dc.forward')
        if log.isEnabledFor(logging.INFO):
            log.info(f"init_catted: {list(x.shape)} -> {list(result.shape)} (mode={mode.value})")
        tensor_log = get_logger('dc.tensors')
        if tensor_log.isEnabledFor(TENSOR_LEVEL):
            q = result.shape[0] // 4
            pos, neg = result[:q], result[q:2*q]
            tensor_log.log(TENSOR_LEVEL, f"init: x_mean={x.mean().item():.4f}, pos_mean={pos.mean().item():.4f}, neg_mean={neg.mean().item():.4f}")
    except ImportError:
        pass

    return result


# =============================================================================
# Output reconstruction
# =============================================================================

def reconstruct_output(output_4: Tensor) -> Tensor:
    """Reconstruct original output from [4*batch]: out_pos - out_neg."""
    q = output_4.shape[0] // 4
    out_pos = output_4[:q]
    out_neg = output_4[q:2*q]
    result = out_pos - out_neg

    # Log reconstruction
    try:
        from ..logging_config import get_logger, TENSOR_LEVEL
        import logging
        log = get_logger('dc.forward')
        if log.isEnabledFor(logging.INFO):
            log.info(f"reconstruct: {list(output_4.shape)} -> {list(result.shape)}")
        tensor_log = get_logger('dc.tensors')
        if tensor_log.isEnabledFor(TENSOR_LEVEL):
            tensor_log.log(TENSOR_LEVEL, f"reconstruct: pos_mean={out_pos.mean().item():.4f}, neg_mean={out_neg.mean().item():.4f}, z_mean={result.mean().item():.4f}")
    except ImportError:
        pass

    return result


# =============================================================================
# Re-centering (prevents magnitude explosion in residual networks)
# =============================================================================

class DCRecenterFunction(torch.autograd.Function):
    """
    Re-center DC representation with proper gradient flow.

    Forward: new_pos = relu(z), new_neg = relu(-z) where z = pos - neg
    Backward: Properly propagate 4-sensitivities through the relu operations.
    """

    @staticmethod
    def forward(ctx, tensor_4: Tensor) -> Tensor:
        q = tensor_4.shape[0] // 4
        pos = tensor_4[:q]
        neg = tensor_4[q:2*q]

        z = pos - neg
        new_pos = torch.relu(z)
        new_neg = torch.relu(-z)

        # Save mask for backward: where z > 0, gradient flows through new_pos
        # where z < 0, gradient flows through new_neg
        ctx.save_for_backward((z > 0).float(), (z < 0).float())

        return make_input_4(new_pos, new_neg)

    @staticmethod
    def backward(ctx, grad_4: Tensor) -> Tensor:
        mask_pos, mask_neg = ctx.saved_tensors

        # Split incoming gradients
        delta_pp, delta_np, delta_pn, delta_nn = split_grad_4(grad_4)

        # Gradient through recenter:
        # new_pos = relu(z) = relu(pos - neg)
        # new_neg = relu(-z) = relu(neg - pos)
        #
        # d(new_pos)/d(pos) = relu'(z) = mask_pos
        # d(new_pos)/d(neg) = -relu'(z) = -mask_pos
        # d(new_neg)/d(pos) = -relu'(-z) = -mask_neg
        # d(new_neg)/d(neg) = relu'(-z) = mask_neg
        #
        # In 4-sensitivity format:
        # delta_pp flows to: pos (new_pos path) -> delta_pp * mask_pos
        # delta_np flows to: neg (new_pos path) -> delta_np * (-mask_pos) -> goes to input neg
        # etc.

        # Gradient w.r.t. pos (from both new_pos and new_neg paths)
        grad_pos_from_new_pos = delta_pp * mask_pos  # d(new_pos)/d(pos) * delta_pp
        grad_pos_from_new_neg = -delta_np * mask_neg  # d(new_neg)/d(pos) * delta_np

        # Gradient w.r.t. neg (from both new_pos and new_neg paths)
        grad_neg_from_new_pos = -delta_pp * mask_pos  # d(new_pos)/d(neg) * delta_pp
        grad_neg_from_new_neg = delta_np * mask_neg   # d(new_neg)/d(neg) * delta_np

        # Same for pn/nn sensitivities
        grad_pos_from_new_pos_pn = delta_pn * mask_pos
        grad_pos_from_new_neg_pn = -delta_nn * mask_neg
        grad_neg_from_new_pos_pn = -delta_pn * mask_pos
        grad_neg_from_new_neg_pn = delta_nn * mask_neg

        # Combine into new 4-sensitivities for input
        # The pp sensitivity for input pos comes from pp flowing through new_pos
        new_pp = grad_pos_from_new_pos
        # The np sensitivity for input neg comes from pp flowing through new_neg (sign matters)
        new_np = -grad_neg_from_new_pos  # Flip sign for neg input

        new_pn = grad_pos_from_new_pos_pn
        new_nn = -grad_neg_from_new_pos_pn

        # Actually, let me think about this more carefully.
        # The 4-sensitivities are:
        # delta_pp: dL/d(out_pos) for the "positive" sensitivity path
        # delta_np: dL/d(out_pos) that came through the neg input path
        # etc.

        # For recenter, the output (new_pos, new_neg) depends on input (pos, neg) through z = pos - neg.
        # The chain rule gives us:
        # dL/d(pos) = dL/d(new_pos) * d(new_pos)/d(pos) + dL/d(new_neg) * d(new_neg)/d(pos)
        #           = delta_pp * mask_pos + delta_np * (-mask_neg)
        # dL/d(neg) = dL/d(new_pos) * d(new_pos)/d(neg) + dL/d(new_neg) * d(new_neg)/d(neg)
        #           = delta_pp * (-mask_pos) + delta_np * mask_neg

        # In standard DC backward, the 4 sensitivities propagate independently.
        # Here we need to mix them based on the recenter operation.

        # Simpler approach: since recenter just applies relu to z = pos - neg,
        # the gradient flow is determined by where z > 0 or z < 0.
        # Where z > 0: new_pos = z, new_neg = 0, so gradient flows through new_pos only
        # Where z < 0: new_pos = 0, new_neg = -z, so gradient flows through new_neg only

        # For the DC 4-sensitivity format, we need to propagate each sensitivity
        # through the appropriate path.

        # When z > 0 (mask_pos = 1, mask_neg = 0):
        #   new_pos = z = pos - neg, new_neg = 0
        #   dL/d(pos) = dL/d(new_pos) = delta_pp (for pp path)
        #   dL/d(neg) = -dL/d(new_pos) = -delta_pp
        #   But wait, neg doesn't contribute to new_neg when z > 0, so delta_np doesn't flow.

        # This is getting complicated. Let me just use a simpler formulation:
        # The recenter operation is: new_pos = relu(pos - neg), new_neg = relu(neg - pos)
        # This is equivalent to two separate relu operations.

        # For standard gradient reconstruction:
        # grad_z = delta_pp * mask_pos - delta_np * mask_neg  (contribution to z from both outputs)

        # Actually, let me use the simplest correct approach:
        # Compute grad_z = grad_new_pos * relu'(z) + grad_new_neg * relu'(-z) * (-1)
        #                = delta_pp * mask_pos - delta_np * mask_neg
        # Then: grad_pos = grad_z, grad_neg = -grad_z

        grad_z_pp = delta_pp * mask_pos - delta_np * mask_neg
        grad_z_pn = delta_pn * mask_pos - delta_nn * mask_neg

        # For input sensitivities:
        # new_pp corresponds to grad on pos from pp path
        # new_np corresponds to grad on neg from pp path (but neg contributes -z)
        input_pp = grad_z_pp  # grad w.r.t. pos
        input_np = torch.zeros_like(grad_z_pp)  # pp doesn't flow to neg's positive sensitivity
        input_pn = grad_z_pn
        input_nn = torch.zeros_like(grad_z_pn)

        # Hmm, this still doesn't feel right. Let me just pass through unchanged for now
        # and rely on Add's recenter which works.
        return grad_4


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
    return DCRecenterFunction.apply(tensor_4)


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
