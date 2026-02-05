"""
Utility functions for DC decomposition.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

# Removed DCCache dependency - was part of deprecated HookDecomposer


def split_input(x: Tensor, beta: float = 1.0) -> Tuple[Tensor, Tensor]:
    """
    Split input tensor into positive and negative parts.

    Args:
        x: Input tensor
        beta: Split parameter (default 1.0)

    Returns:
        (pos, neg) where x = pos - neg
    """
    pos = beta * x
    neg = -(1 - beta) * x
    return pos, neg


def reconstruct(pos: Tensor, neg: Tensor) -> Tensor:
    """
    Reconstruct original tensor from pos and neg parts.

    Args:
        pos: Positive stream
        neg: Negative stream

    Returns:
        pos - neg
    """
    return pos - neg


def compute_gradient_from_sensitivities(
    delta_pp: Tensor, delta_np: Tensor, delta_pn: Tensor, delta_nn: Tensor
) -> Tensor:
    """
    Compute the combined gradient from 4 sensitivities.

    The gradient w.r.t. the original activation is:
        grad = (delta_pp - delta_np) - (delta_pn - delta_nn)

    Args:
        delta_pp, delta_np, delta_pn, delta_nn: The 4 sensitivities

    Returns:
        Combined gradient tensor
    """
    return (delta_pp - delta_np) - (delta_pn - delta_nn)


def compute_pos_gradient(delta_pp: Tensor, delta_np: Tensor) -> Tensor:
    """
    Compute gradient w.r.t. positive stream.

    Args:
        delta_pp: Sensitivity of pos output to pos input
        delta_np: Sensitivity of pos output to neg input

    Returns:
        Gradient w.r.t. pos stream
    """
    return delta_pp - delta_np


def compute_neg_gradient(delta_pn: Tensor, delta_nn: Tensor) -> Tensor:
    """
    Compute gradient w.r.t. negative stream.

    Args:
        delta_pn: Sensitivity of neg output to pos input
        delta_nn: Sensitivity of neg output to neg input

    Returns:
        Gradient w.r.t. neg stream
    """
    return delta_pn - delta_nn


def minimization_step(pos: Tensor, neg: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Minimize the magnitude of pos and neg while maintaining pos - neg.

    This removes the common positive part from both streams:
        min_val = min(pos, neg)
        pos_new = pos - min_val
        neg_new = neg - min_val

    Args:
        pos: Positive stream
        neg: Negative stream

    Returns:
        (pos_minimized, neg_minimized)
    """
    min_val = torch.minimum(pos, neg)
    pos_new = pos - min_val
    neg_new = neg - min_val
    return pos_new, neg_new


def renormalize(pos: Tensor, neg: Tensor, original: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Renormalize pos and neg to exactly match the original activation.

    This corrects for any numerical drift.

    Args:
        pos: Positive stream
        neg: Negative stream
        original: Original activation to match

    Returns:
        (pos_normalized, neg_normalized) such that pos - neg == original
    """
    reconstructed = pos - neg
    diff = original - reconstructed

    pos_new = pos + diff / 2
    neg_new = neg - diff / 2

    return pos_new, neg_new


def verify_monotonicity(pos: Tensor, neg: Tensor, tolerance: float = 1e-6) -> bool:
    """
    Verify that both pos and neg are non-negative (monotonicity property).

    Args:
        pos: Positive stream
        neg: Negative stream
        tolerance: Tolerance for numerical errors

    Returns:
        True if both streams are non-negative
    """
    return (pos >= -tolerance).all() and (neg >= -tolerance).all()


# The following functions were removed as they depend on the deprecated HookDecomposer DCCache:
# - compute_stream_norms()
# - compute_sensitivity_norms()
#
# For the current patcher API, use dc_decompose.operations.base.extract_sensitivities()
# to get sensitivities from gradient tensors.


def find_supported_layers(model: nn.Module) -> Dict[str, type]:
    """
    Find all layers in a model that support DC decomposition.

    Args:
        model: PyTorch model

    Returns:
        Dictionary mapping layer names to their types
    """
    supported_types = {
        nn.Linear, nn.Conv2d, nn.ReLU,
        nn.BatchNorm2d, nn.MaxPool2d, nn.AvgPool2d,
        nn.Flatten, nn.AdaptiveAvgPool2d,
    }

    result = {}
    for name, module in model.named_modules():
        if type(module) in supported_types:
            result[name] = type(module)

    return result


def count_decomposable_params(model: nn.Module) -> Tuple[int, int]:
    """
    Count parameters in decomposable vs non-decomposable layers.

    Args:
        model: PyTorch model

    Returns:
        (decomposable_params, total_params)
    """
    supported_types = {nn.Linear, nn.Conv2d, nn.BatchNorm2d}

    decomposable = 0
    total = 0

    for name, module in model.named_modules():
        params = sum(p.numel() for p in module.parameters(recurse=False))
        total += params

        if type(module) in supported_types:
            decomposable += params

    return decomposable, total
