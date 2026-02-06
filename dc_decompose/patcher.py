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
from dataclasses import dataclass
from typing import Optional, List, TYPE_CHECKING
from torch import Tensor

from .operations.base import (
    DC_ENABLED, DC_ORIGINAL_FORWARD, DC_IS_OUTPUT_LAYER,
    DC_SENSITIVITY_ALPHA, compute_frobenius_alpha,
)

if TYPE_CHECKING:
    from .alignment_cache import AlignmentCache, AlignmentMode
from .operations.linear import patch_linear, unpatch_linear
from .operations.conv import patch_conv2d, unpatch_conv2d, patch_conv1d, unpatch_conv1d
from .operations.conv_transpose import patch_conv_transpose1d, patch_conv_transpose2d, unpatch_conv_transpose1d, unpatch_conv_transpose2d
from .operations.relu import patch_relu, unpatch_relu
from .operations.batchnorm import patch_batchnorm, unpatch_batchnorm
from .operations.maxpool import patch_maxpool1d, patch_maxpool2d, unpatch_maxpool1d, unpatch_maxpool2d
from .operations.avgpool import (
    patch_avgpool1d, patch_avgpool2d, unpatch_avgpool1d, unpatch_avgpool2d,
    patch_adaptive_avgpool1d, patch_adaptive_avgpool2d, unpatch_adaptive_avgpool1d, unpatch_adaptive_avgpool2d
)
from .operations.add import Add, patch_add, unpatch_add
from .operations.shape_ops import (
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
from .operations.layernorm import patch_layernorm, unpatch_layernorm
from .operations.softmax import patch_softmax, unpatch_softmax
from .operations.matmul import DCMatMul, patch_dcmatmul, unpatch_dcmatmul
from .operations.mul import DCMul, patch_dcmul, unpatch_dcmul


# Forward declaration for Mul and Mean (defined in functional_replacer)
Mul = None
Mean = None


def _lazy_import_mul_mean():
    """Lazy import to avoid circular dependency."""
    global Mul, Mean
    if Mul is None:
        from .functional_replacer import Mul as _Mul, Mean as _Mean
        Mul = _Mul
        Mean = _Mean


def _get_mul_mean_patchers():
    """Get patch/unpatch functions for Mul and Mean."""
    from .functional_replacer import patch_mul, unpatch_mul, patch_mean, unpatch_mean
    return patch_mul, unpatch_mul, patch_mean, unpatch_mean


def patch_model(
    model: nn.Module,
    relu_mode: str = 'max',
    backprop_mode: str = 'sum',
    target_layers: Optional[List[str]] = None,
) -> None:
    """
    Patch all supported layers in a model for DC decomposition.

    Args:
        model: The PyTorch model to patch
        relu_mode: ReLU decomposition mode ('max', 'min', 'half')
        backprop_mode: ReLU backprop mode ('standard', 'mask_diff', 'sum')
            - 'sum': preserves gradient reconstruction (default)
            - 'standard': original DC sensitivity propagation
            - 'mask_diff': alternative formulation
        target_layers: Optional list of layer names (None = all)
    """
    _lazy_import_mul_mean()
    patch_mul, unpatch_mul, patch_mean, unpatch_mean = _get_mul_mean_patchers()

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
        # elif isinstance(module, nn.AvgPool2d):
        #     patch_avgpool2d(module)  # Disabled: linear operation works out of the box
        # elif isinstance(module, nn.AvgPool1d):
        #     patch_avgpool1d(module)  # Disabled: linear operation works out of the box
        # elif isinstance(module, nn.AdaptiveAvgPool2d):
        #     patch_adaptive_avgpool2d(module)  # Disabled: linear operation works out of the box
        # elif isinstance(module, nn.AdaptiveAvgPool1d):
        #     patch_adaptive_avgpool1d(module)  # Disabled: linear operation works out of the box
        # elif isinstance(module, nn.Flatten):
        #     patch_flatten(module)  # Disabled: reshaping operation works out of the box
        # elif isinstance(module, nn.Unflatten):
        #     patch_unflatten(module)  # Disabled: reshaping operation works out of the box
        elif isinstance(module, nn.Dropout):
            patch_dropout(module)
        elif isinstance(module, Add):
            patch_add(module)  # Re-enabled: DC-format addition needs special handling
        # elif isinstance(module, Reshape):
        #     patch_reshape(module)  # Disabled: reshaping operation works out of the box
        # elif isinstance(module, View):
        #     patch_view(module)  # Disabled: reshaping operation works out of the box
        # elif isinstance(module, Squeeze):
        #     patch_squeeze(module)  # Disabled: reshaping operation works out of the box
        # elif isinstance(module, Unsqueeze):
        #     patch_unsqueeze(module)  # Disabled: reshaping operation works out of the box
        elif isinstance(module, Transpose):
            patch_transpose(module)
        elif isinstance(module, Permute):
            patch_permute(module)
        elif Mul is not None and isinstance(module, Mul):
            patch_mul(module)
        elif Mean is not None and isinstance(module, Mean):
            patch_mean(module)
        elif isinstance(module, nn.LayerNorm):
            patch_layernorm(module)
        elif isinstance(module, nn.Softmax):
            patch_softmax(module)
        elif isinstance(module, DCMatMul):
            patch_dcmatmul(module)
        elif isinstance(module, DCMul):
            patch_dcmul(module)


def unpatch_model(model: nn.Module) -> None:
    """Unpatch all layers, restoring original forward methods."""
    _lazy_import_mul_mean()
    patch_mul, unpatch_mul, patch_mean, unpatch_mean = _get_mul_mean_patchers()

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
        # elif isinstance(module, nn.AvgPool2d):
        #     unpatch_avgpool2d(module)  # Disabled: linear operation works out of the box
        # elif isinstance(module, nn.AvgPool1d):
        #     unpatch_avgpool1d(module)  # Disabled: linear operation works out of the box
        # elif isinstance(module, nn.AdaptiveAvgPool2d):
        #     unpatch_adaptive_avgpool2d(module)  # Disabled: linear operation works out of the box
        # elif isinstance(module, nn.AdaptiveAvgPool1d):
        #     unpatch_adaptive_avgpool1d(module)  # Disabled: linear operation works out of the box
        # elif isinstance(module, nn.Flatten):
        #     unpatch_flatten(module)  # Disabled: reshaping operation works out of the box
        # elif isinstance(module, nn.Unflatten):
        #     unpatch_unflatten(module)  # Disabled: reshaping operation works out of the box
        elif isinstance(module, nn.Dropout):
            unpatch_dropout(module)
        elif isinstance(module, Add):
            unpatch_add(module)  # Re-enabled: DC-format addition needs special handling
        # elif isinstance(module, Reshape):
        #     unpatch_reshape(module)  # Disabled: reshaping operation works out of the box
        # elif isinstance(module, View):
        #     unpatch_view(module)  # Disabled: reshaping operation works out of the box
        # elif isinstance(module, Squeeze):
        #     unpatch_squeeze(module)  # Disabled: reshaping operation works out of the box
        # elif isinstance(module, Unsqueeze):
        #     unpatch_unsqueeze(module)  # Disabled: reshaping operation works out of the box
        elif isinstance(module, Transpose):
            unpatch_transpose(module)
        elif isinstance(module, Permute):
            unpatch_permute(module)
        elif Mul is not None and isinstance(module, Mul):
            unpatch_mul(module)
        elif Mean is not None and isinstance(module, Mean):
            unpatch_mean(module)
        elif isinstance(module, nn.LayerNorm):
            unpatch_layernorm(module)
        elif isinstance(module, nn.Softmax):
            unpatch_softmax(module)
        elif isinstance(module, DCMatMul):
            unpatch_dcmatmul(module)
        elif isinstance(module, DCMul):
            unpatch_dcmul(module)


def set_sensitivity_alpha(
    model: nn.Module,
    alpha: float = 0.0,
    mode: str = 'constant',
    default_alpha: float = 0.5,
) -> None:
    """
    Set sensitivity shift alpha on all modules for numerical stability.

    The alpha parameter controls how much to shift sensitivities to reduce
    magnitudes while preserving the gradient reconstruction.

    Args:
        model: The PyTorch model
        alpha: Alpha value for constant mode (0 = no shift, 0.5 = typical for add layers)
        mode: How to determine alpha per module:
            - 'constant': Use the same alpha for all modules
            - 'frobenius': Compute alpha from Frobenius norm of weights.
              Modules with weights use alpha = (1 - 1/rho) / 2 where rho is Frobenius norm.
              Modules without weights (ReLU, pooling) use default_alpha.
        default_alpha: Alpha to use for non-weight modules in 'frobenius' mode
    """
    _lazy_import_mul_mean()

    for name, module in model.named_modules():
        if mode == 'constant':
            setattr(module, DC_SENSITIVITY_ALPHA, alpha)
        elif mode == 'frobenius':
            # Modules with weights: compute from Frobenius norm
            if hasattr(module, 'weight') and module.weight is not None:
                computed_alpha = compute_frobenius_alpha(module.weight)
                setattr(module, DC_SENSITIVITY_ALPHA, computed_alpha)
            elif isinstance(module, Add):
                # Add modules always use default_alpha (typically 0.5)
                setattr(module, DC_SENSITIVITY_ALPHA, default_alpha)
            else:
                # Non-weight modules (ReLU, MaxPool, etc.): use default_alpha
                setattr(module, DC_SENSITIVITY_ALPHA, default_alpha)
        else:
            raise ValueError(f"Unknown alpha mode: {mode}. Use 'constant' or 'frobenius'.")


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
    Find the last computational layer in the model (the output layer).

    Uses forward hooks with a dummy input to find the actual last module call
    in execution order, rather than registration order.

    Returns the module or None if not found.
    """
    _lazy_import_mul_mean()

    # Patchable layer types that can be output layers
    output_layer_types = [
        nn.Linear, nn.Conv2d, nn.Conv1d,
        nn.ConvTranspose1d, nn.ConvTranspose2d,
        nn.ReLU, nn.GELU, nn.Sigmoid, nn.Tanh, nn.Softmax,
        nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d,
        Add,
    ]
    if Mul is not None:
        output_layer_types.append(Mul)
    if Mean is not None:
        output_layer_types.append(Mean)
    output_layer_types = tuple(output_layer_types)

    # Track execution order using hooks
    execution_order = []
    hooks = []

    def make_hook(module):
        def hook(mod, inp, out):
            if isinstance(mod, output_layer_types):
                execution_order.append(mod)
        return hook

    # Register hooks on all output_layer_types modules
    for name, module in model.named_modules():
        if isinstance(module, output_layer_types):
            hooks.append(module.register_forward_hook(make_hook(module)))

    # Try to run a forward pass to determine execution order
    try:
        model.eval()
        with torch.no_grad():
            # Create a dummy input that matches the model's expected input
            # Try common input shapes
            for shape in [
                (1, 3, 32, 32),   # Image-like
                (1, 64, 16, 16),  # Feature map
                (1, 128),         # 1D vector
                (1, 16, 64),      # Sequence-like
            ]:
                try:
                    dummy = torch.zeros(*shape)
                    model(dummy)
                    break
                except Exception:
                    execution_order.clear()
                    continue
    except Exception:
        pass
    finally:
        # Remove hooks
        for h in hooks:
            h.remove()

    # Return the last executed layer
    if execution_order:
        return execution_order[-1]

    # Fallback: use registration order
    last_layer = None
    for name, module in model.named_modules():
        if isinstance(module, output_layer_types):
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
    backprop_mode: str = 'sum',
    beta: float = 1.0,
    align_forward: bool = False,
    align_backward: bool = False,
    x: Optional[Tensor] = None,
    target_grad: Optional[Tensor] = None,
) -> nn.Module:
    """
    Prepare a model for DC decomposition in one step.

    This function:
    1. Replaces functional calls (torch.relu, +, etc.) with module equivalents
    2. Finds the output layer (before patching, so forward pass works normally)
    3. If alignment enabled: captures original activations/gradients
    4. Patches all layers for DC decomposition
    5. Marks the output layer for backward initialization

    Args:
        model: The PyTorch model to prepare
        relu_mode: ReLU decomposition mode ('max', 'min', 'half')
        backprop_mode: ReLU backprop mode ('standard', 'mask_diff', 'sum')
        beta: Output layer initialization parameter (default 1.0)
        align_forward: If True, align DC forward to match original activations
        align_backward: If True, align DC backward to match original gradients
        x: Input tensor (required if alignment is enabled)
        target_grad: Target gradient for backward (required if align_backward=True)

    Returns:
        The prepared model (same instance after in-place modifications,
        or a new instance if functional replacement required deep copy)

    Example:
        model = prepare_model_for_dc(model)
        x_cat = init_catted(x, InputMode.CENTER)
        out_cat = model(x_cat)
        out = reconstruct_output(out_cat)

    Example with alignment:
        model = prepare_model_for_dc(
            model, align_forward=True, align_backward=True,
            x=x, target_grad=target_grad
        )
    """
    from .functional_replacer import make_dc_compatible

    # Step 1: Replace functional calls with modules
    model = make_dc_compatible(model)
    model.eval()

    # Step 2: Find output layer BEFORE patching (so forward pass works normally)
    output_layer = find_output_layer(model)

    # Step 3: Create and populate alignment cache if alignment is enabled
    alignment_cache = None
    if align_forward or align_backward:
        if x is None:
            raise ValueError("x is required when alignment is enabled")
        if align_backward and target_grad is None:
            raise ValueError("target_grad is required when align_backward=True")

        from .alignment_cache import AlignmentCache, AlignmentMode

        # Determine mode based on flags
        if align_forward and align_backward:
            mode = AlignmentMode.BOTH
        elif align_forward:
            mode = AlignmentMode.FORWARD_ONLY
        else:
            mode = AlignmentMode.BACKWARD_ONLY

        alignment_cache = AlignmentCache(mode=mode)
        alignment_cache.capture_original(
            model, x, target_grad,
            capture_forward=align_forward,
            capture_backward=align_backward,
        )
        alignment_cache.attach_to_model(model)

    # Step 4: Patch all layers
    patch_model(model, relu_mode=relu_mode, backprop_mode=backprop_mode)

    # Step 5: Mark output layer
    if output_layer is not None:
        mark_output_layer(output_layer, beta)

    return model


# =============================================================================
# Aligned DC Forward Context Manager
# =============================================================================

@dataclass
class AlignedDCContext:
    """Context object for aligned_dc_forward."""
    cache: 'AlignmentCache'
    original_output: Tensor
    _x_cat: Optional[Tensor] = None
    _output: Optional[Tensor] = None

    @property
    def output(self) -> Tensor:
        """The reconstructed DC output (pos - neg)."""
        if self._output is None:
            raise RuntimeError("Output not available")
        return self._output

    @property
    def input_4(self) -> Tensor:
        """The [4*batch] input tensor (for accessing .grad after backward)."""
        if self._x_cat is None:
            raise RuntimeError("Input not available")
        return self._x_cat


@contextmanager
def aligned_dc_forward(
    model: nn.Module,
    x: Tensor,
    mode: Optional['AlignmentMode'] = None,
    target_grad: Optional[Tensor] = None,
    relu_mode: str = 'max',
    backprop_mode: str = 'sum',
):
    """
    Context manager for DC forward with alignment correction.

    This provides a convenient API that:
    1. Replaces functional calls in the model
    2. Captures original activations/gradients (before DC)
    3. Patches the model for DC decomposition
    4. Runs DC forward with automatic alignment

    Usage:
        with aligned_dc_forward(model, x, mode=AlignmentMode.BOTH) as ctx:
            out = ctx.output
            loss = criterion(out, target)
            loss.backward()

        # Get correction statistics
        stats = ctx.cache.get_correction_stats()

    Args:
        model: PyTorch model (will be modified in place)
        x: Input tensor
        mode: Alignment mode (default: AlignmentMode.BOTH)
        target_grad: Target gradient for backward capture (optional)
        relu_mode: ReLU decomposition mode ('max', 'min', 'half')
        backprop_mode: ReLU backprop mode ('standard', 'mask_diff', 'sum')

    Yields:
        AlignedDCContext with output and cache

    Note:
        The model is unpatched when the context exits. Make a copy if
        you need to keep the patched version.
    """
    from .alignment_cache import AlignmentCache, AlignmentMode
    from .functional_replacer import make_dc_compatible
    from .operations.base import init_catted, reconstruct_output

    # Default mode
    if mode is None:
        mode = AlignmentMode.BOTH

    # Step 1: Replace functional calls
    model = make_dc_compatible(model)
    model.eval()

    # Step 2: Create cache and capture original behavior
    cache = AlignmentCache(mode=mode)
    capture_forward = mode in (AlignmentMode.FORWARD_ONLY, AlignmentMode.BOTH)
    capture_backward = mode in (AlignmentMode.BACKWARD_ONLY, AlignmentMode.BOTH)

    orig_output = cache.capture_original(
        model, x, target_grad,
        capture_forward=capture_forward,
        capture_backward=capture_backward,
    )

    # Step 3: Prepare for DC
    output_layer = find_output_layer(model)
    cache.attach_to_model(model)
    patch_model(model, relu_mode=relu_mode, backprop_mode=backprop_mode)
    if output_layer:
        mark_output_layer(output_layer)

    # Step 4: Create result container
    ctx = AlignedDCContext(cache=cache, original_output=orig_output)

    # Step 5: Run DC forward
    ctx._x_cat = init_catted(x)
    out_cat = model(ctx._x_cat)
    ctx._output = reconstruct_output(out_cat)

    try:
        yield ctx
    finally:
        cache.detach_from_model(model)
        unpatch_model(model)


# =============================================================================
# Logging Integration
# =============================================================================

_logging_enabled = False


def enable_dc_logging(
    level: str = 'INFO',
    channels: Optional[List[str]] = None,
    include_tensors: bool = False
):
    """
    Enable logging for DC decomposition operations.

    Args:
        level: Log level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        channels: Specific channels to enable (default: all except tensors)
            - 'dc.forward': Forward pass logging
            - 'dc.backward': Backward pass logging
            - 'dc.tensors': Tensor values (very verbose)
            - 'dc.recenter': Re-centering operations
            - 'dc.patch': Patching operations
        include_tensors: Include detailed tensor logging

    Example:
        enable_dc_logging('DEBUG')  # Verbose logging
        enable_dc_logging('INFO', channels=['dc.forward'])  # Only forward pass
        enable_dc_logging('DEBUG', include_tensors=True)  # Include tensor values
    """
    global _logging_enabled
    import logging as _logging
    from .logging_config import enable_logging, wrap_autograd_function

    level_map = {
        'DEBUG': _logging.DEBUG,
        'INFO': _logging.INFO,
        'WARNING': _logging.WARNING,
        'ERROR': _logging.ERROR,
    }
    log_level = level_map.get(level.upper(), _logging.INFO)

    enable_logging(log_level, channels, include_tensors)

    # Wrap all DC autograd functions with logging (only once)
    if not _logging_enabled:
        _wrap_all_dc_functions()
        _logging_enabled = True


def disable_dc_logging():
    """Disable DC logging."""
    from .logging_config import disable_logging
    disable_logging()


def _wrap_all_dc_functions():
    """Wrap all DC autograd functions with logging."""
    from .logging_config import wrap_autograd_function

    # Import all DC function classes
    from .operations.linear import DCLinearFunction
    from .operations.conv import DCConv1dFunction, DCConv2dFunction
    from .operations.conv_transpose import DCConvTranspose1dFunction, DCConvTranspose2dFunction
    from .operations.relu import DCReLUFunction
    from .operations.batchnorm import DCBatchNormFunction
    from .operations.maxpool import DCMaxPool1dFunction, DCMaxPool2dFunction
    from .operations.avgpool import (
        DCAvgPool1dFunction, DCAvgPool2dFunction,
        DCAdaptiveAvgPool1dFunction, DCAdaptiveAvgPool2dFunction
    )
    from .operations.add import DCAddFunction
    from .operations.shape_ops import (
        DCFlattenFunction, DCUnflattenFunction, DCReshapeFunction,
        DCSqueezeFunction, DCUnsqueezeFunction, DCTransposeFunction,
        DCPermuteFunction, DCDropoutFunction
    )
    from .operations.layernorm import DCLayerNormFunction
    from .operations.softmax import DCSoftmaxFunction
    from .functional_replacer import DCMulFunction, DCMeanFunction

    # Wrap each function class
    function_classes = [
        (DCLinearFunction, "Linear"),
        (DCConv1dFunction, "Conv1d"),
        (DCConv2dFunction, "Conv2d"),
        (DCConvTranspose1dFunction, "ConvT1d"),
        (DCConvTranspose2dFunction, "ConvT2d"),
        (DCReLUFunction, "ReLU"),
        (DCBatchNormFunction, "BatchNorm"),
        (DCMaxPool1dFunction, "MaxPool1d"),
        (DCMaxPool2dFunction, "MaxPool2d"),
        (DCAvgPool1dFunction, "AvgPool1d"),
        (DCAvgPool2dFunction, "AvgPool2d"),
        (DCAdaptiveAvgPool1dFunction, "AdaptAvgPool1d"),
        (DCAdaptiveAvgPool2dFunction, "AdaptAvgPool2d"),
        (DCAddFunction, "Add"),
        (DCFlattenFunction, "Flatten"),
        (DCUnflattenFunction, "Unflatten"),
        (DCReshapeFunction, "Reshape"),
        (DCSqueezeFunction, "Squeeze"),
        (DCUnsqueezeFunction, "Unsqueeze"),
        (DCTransposeFunction, "Transpose"),
        (DCPermuteFunction, "Permute"),
        (DCDropoutFunction, "Dropout"),
        (DCLayerNormFunction, "LayerNorm"),
        (DCSoftmaxFunction, "Softmax"),
        (DCMulFunction, "Mul"),
        (DCMeanFunction, "Mean"),
    ]

    for func_cls, name in function_classes:
        wrap_autograd_function(func_cls, name)
