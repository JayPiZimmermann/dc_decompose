"""
Model patcher for DC decomposition.

Usage:
    model = YourModel()
    patch_model(model)
    mark_output_layer(model.fc)  # Mark last layer for backward init

    # Forward
    x_catted = init_catted(x)  # [batch] -> [2*batch]
    output_catted = model(x_catted)
    out_pos, out_neg = split2(output_catted)

    # Backward
    reconstructed = out_pos - out_neg
    reconstructed.backward(gradient)  # Triggers 4-sensitivity backward
"""

import torch
import torch.nn as nn
from contextlib import contextmanager
from typing import Optional, List

from .base import DC_ENABLED, DC_ORIGINAL_FORWARD, DC_IS_OUTPUT_LAYER, DC_BETA
from .linear import patch_linear, unpatch_linear
from .conv2d import patch_conv2d, unpatch_conv2d
from .conv1d import patch_conv1d, unpatch_conv1d
from .conv_transpose import patch_conv_transpose1d, patch_conv_transpose2d, unpatch_conv_transpose1d, unpatch_conv_transpose2d
from .relu import patch_relu, unpatch_relu
from .batchnorm import patch_batchnorm, unpatch_batchnorm
from .maxpool import patch_maxpool1d, patch_maxpool2d, unpatch_maxpool1d, unpatch_maxpool2d
from .avgpool import (
    patch_avgpool1d, patch_avgpool2d, unpatch_avgpool1d, unpatch_avgpool2d,
    patch_adaptive_avgpool1d, patch_adaptive_avgpool2d, unpatch_adaptive_avgpool1d, unpatch_adaptive_avgpool2d
)
from .add import Add, patch_add, unpatch_add
from .shape_ops import (
    patch_flatten, unpatch_flatten,
    patch_unflatten, unpatch_unflatten,
    Reshape, patch_reshape, unpatch_reshape,
    View, patch_view, unpatch_view,
    Squeeze, patch_squeeze, unpatch_squeeze,
    Unsqueeze, patch_unsqueeze, unpatch_unsqueeze,
    Transpose, patch_transpose, unpatch_transpose,
    Permute, patch_permute, unpatch_permute,
    patch_dropout, unpatch_dropout
)


def patch_model(
    model: nn.Module,
    relu_mode: str = 'max',
    backprop_mode: str = 'standard',
    target_layers: Optional[List[str]] = None,
) -> None:
    """
    Patch all supported layers in a model for DC decomposition.

    Args:
        model: The PyTorch model to patch
        relu_mode: ReLU decomposition mode ('max', 'min', 'half')
        backprop_mode: ReLU backprop mode ('standard', 'mask_diff', 'sum')
            - 'standard': original DC sensitivity propagation (default)
            - 'sum': preserves gradient reconstruction
            - 'mask_diff': alternative formulation
        target_layers: Optional list of layer names (None = all)
    """
    for name, module in model.named_modules():
        if target_layers is not None and name not in target_layers:
            continue

        if isinstance(module, nn.Linear):
            patch_linear(module)
        elif isinstance(module, nn.Conv2d):
            patch_conv2d(module)
        elif isinstance(module, nn.Conv1d):
            patch_conv1d(module)
        elif isinstance(module, nn.ConvTranspose2d):
            patch_conv_transpose2d(module)
        elif isinstance(module, nn.ConvTranspose1d):
            patch_conv_transpose1d(module)
        elif isinstance(module, nn.ReLU):
            patch_relu(module, split_mode=relu_mode, backprop_mode=backprop_mode)
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            patch_batchnorm(module)
        elif isinstance(module, nn.MaxPool2d):
            patch_maxpool2d(module)
        elif isinstance(module, nn.MaxPool1d):
            patch_maxpool1d(module)
        elif isinstance(module, nn.AvgPool2d):
            patch_avgpool2d(module)
        elif isinstance(module, nn.AvgPool1d):
            patch_avgpool1d(module)
        elif isinstance(module, nn.AdaptiveAvgPool2d):
            patch_adaptive_avgpool2d(module)
        elif isinstance(module, nn.AdaptiveAvgPool1d):
            patch_adaptive_avgpool1d(module)
        elif isinstance(module, nn.Flatten):
            patch_flatten(module)
        elif isinstance(module, nn.Unflatten):
            patch_unflatten(module)
        elif isinstance(module, nn.Dropout):
            patch_dropout(module)
        elif isinstance(module, Add):
            patch_add(module)
        elif isinstance(module, Reshape):
            patch_reshape(module)
        elif isinstance(module, View):
            patch_view(module)
        elif isinstance(module, Squeeze):
            patch_squeeze(module)
        elif isinstance(module, Unsqueeze):
            patch_unsqueeze(module)
        elif isinstance(module, Transpose):
            patch_transpose(module)
        elif isinstance(module, Permute):
            patch_permute(module)


def unpatch_model(model: nn.Module) -> None:
    """Unpatch all layers, restoring original forward methods."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            unpatch_linear(module)
        elif isinstance(module, nn.Conv2d):
            unpatch_conv2d(module)
        elif isinstance(module, nn.Conv1d):
            unpatch_conv1d(module)
        elif isinstance(module, nn.ConvTranspose2d):
            unpatch_conv_transpose2d(module)
        elif isinstance(module, nn.ConvTranspose1d):
            unpatch_conv_transpose1d(module)
        elif isinstance(module, nn.ReLU):
            unpatch_relu(module)
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            unpatch_batchnorm(module)
        elif isinstance(module, nn.MaxPool2d):
            unpatch_maxpool2d(module)
        elif isinstance(module, nn.MaxPool1d):
            unpatch_maxpool1d(module)
        elif isinstance(module, nn.AvgPool2d):
            unpatch_avgpool2d(module)
        elif isinstance(module, nn.AvgPool1d):
            unpatch_avgpool1d(module)
        elif isinstance(module, nn.AdaptiveAvgPool2d):
            unpatch_adaptive_avgpool2d(module)
        elif isinstance(module, nn.AdaptiveAvgPool1d):
            unpatch_adaptive_avgpool1d(module)
        elif isinstance(module, nn.Flatten):
            unpatch_flatten(module)
        elif isinstance(module, nn.Unflatten):
            unpatch_unflatten(module)
        elif isinstance(module, nn.Dropout):
            unpatch_dropout(module)
        elif isinstance(module, Add):
            unpatch_add(module)
        elif isinstance(module, Reshape):
            unpatch_reshape(module)
        elif isinstance(module, View):
            unpatch_view(module)
        elif isinstance(module, Squeeze):
            unpatch_squeeze(module)
        elif isinstance(module, Unsqueeze):
            unpatch_unsqueeze(module)
        elif isinstance(module, Transpose):
            unpatch_transpose(module)
        elif isinstance(module, Permute):
            unpatch_permute(module)


def mark_output_layer(module: nn.Module, beta: float = 1.0) -> None:
    """
    Mark a module as the output layer for backward initialization.

    The output layer receives [2*batch] gradient from autograd and
    initializes 4 sensitivities:
        delta_pp = beta * grad_pos
        delta_np = 0
        delta_pn = (1-beta) * grad_neg
        delta_nn = 0

    Args:
        module: The output layer module
        beta: Initialization parameter (default 1.0)
    """
    setattr(module, DC_IS_OUTPUT_LAYER, True)
    setattr(module, DC_BETA, beta)


def unmark_output_layer(module: nn.Module) -> None:
    """Remove output layer marking from a module."""
    setattr(module, DC_IS_OUTPUT_LAYER, False)


def set_dc_enabled(model: nn.Module, enabled: bool = True) -> None:
    """Enable or disable DC decomposition for all patched layers."""
    for name, module in model.named_modules():
        if hasattr(module, DC_ENABLED):
            setattr(module, DC_ENABLED, enabled)


@contextmanager
def dc_disabled(model: nn.Module):
    """Context manager to temporarily disable DC decomposition."""
    states = {}
    for name, module in model.named_modules():
        if hasattr(module, DC_ENABLED):
            states[name] = getattr(module, DC_ENABLED)
            setattr(module, DC_ENABLED, False)
    try:
        yield
    finally:
        for name, module in model.named_modules():
            if name in states:
                setattr(module, DC_ENABLED, states[name])


def is_patched(module: nn.Module) -> bool:
    """Check if a module has been patched."""
    return hasattr(module, DC_ORIGINAL_FORWARD)


def get_patched_layers(model: nn.Module) -> List[str]:
    """Get names of all patched layers."""
    return [name for name, module in model.named_modules() if is_patched(module)]


def find_output_layer(model: nn.Module) -> Optional[nn.Module]:
    """
    Find the last computational layer in the model (likely the output layer).

    Excludes _dc_* modules added by make_dc_compatible() since those are
    auxiliary modules (like ReLU replacements), not the actual output.

    Returns the module or None if not found.
    """
    last_layer = None
    for name, module in model.named_modules():
        # Skip _dc_* modules added by functional_replacer
        if name.startswith('_dc_') or '._dc_' in name:
            continue
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d,
                               nn.ConvTranspose1d, nn.ConvTranspose2d)):
            last_layer = module
    return last_layer


def auto_mark_output_layer(model: nn.Module, beta: float = 1.0) -> Optional[nn.Module]:
    """
    Automatically find and mark the output layer.

    Returns the marked module or None if not found.
    """
    output_layer = find_output_layer(model)
    if output_layer is not None:
        mark_output_layer(output_layer, beta)
    return output_layer


def prepare_model_for_dc(
    model: nn.Module,
    relu_mode: str = 'max',
    backprop_mode: str = 'standard',
    beta: float = 1.0,
) -> nn.Module:
    """
    Prepare a model for DC decomposition in one step.

    This function:
    1. Replaces functional calls (torch.relu, +, etc.) with module equivalents
    2. Patches all layers for DC decomposition
    3. Automatically marks the output layer for backward initialization

    Args:
        model: The PyTorch model to prepare
        relu_mode: ReLU decomposition mode ('max', 'min', 'half')
        backprop_mode: ReLU backprop mode ('standard', 'mask_diff', 'sum')
        beta: Output layer initialization parameter (default 1.0)

    Returns:
        The prepared model (same instance after in-place modifications,
        or a new instance if functional replacement required deep copy)

    Example:
        model = prepare_model_for_dc(model)
        x_cat = init_catted(x, InputMode.CENTER)
        out_cat = model(x_cat)
        out = reconstruct_output(out_cat)
    """
    from .functional_replacer import make_dc_compatible

    # Step 1: Replace functional calls with modules
    model = make_dc_compatible(model)
    model.eval()

    # Step 2: Patch all layers
    patch_model(model, relu_mode=relu_mode, backprop_mode=backprop_mode)

    # Step 3: Find and mark output layer
    output_layer = find_output_layer(model)
    if output_layer is not None:
        mark_output_layer(output_layer, beta)

    return model
