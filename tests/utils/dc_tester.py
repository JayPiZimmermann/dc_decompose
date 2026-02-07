"""
DC Decomposition Testing Utilities.

Provides a unified testing API for DC decomposition validation.
Tests verify:
1. Forward: DC output (pos - neg) matches original output
2. Backward: Reconstructed gradient (pp - np - pn + nn) matches original gradient

Two APIs are tested:
- Functional API: init_catted + reconstruct_output
- Context Manager API: dc_forward

Usage:
    from utils import run_model_tests

    MODELS = {
        'ModelName': (model_or_factory, input_tensor_or_shape),
        ...
    }

    if __name__ == "__main__":
        success = run_model_tests(MODELS, title="My Tests")
        sys.exit(0 if success else 1)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import copy
import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field

from dc_decompose.patcher import prepare_model_for_dc, unpatch_model, set_sensitivity_alpha
from dc_decompose.functional_replacer import replace_functional_with_modules
from dc_decompose.operations.base import (
    init_catted, reconstruct_output, InputMode, split4,
    Sensitivities, extract_sensitivities, dc_forward,
)


# =============================================================================
# Type Aliases
# =============================================================================

ModelSpec = Union[nn.Module, Callable[[], nn.Module]]
InputSpec = Union[Tensor, Tuple[int, ...]]
LossFn = Callable[[Tensor, Tensor], Tensor]
LossModelSpec = Tuple[ModelSpec, InputSpec, Tensor, LossFn]  # (model, input, target, loss_fn)


# =============================================================================
# Default Tolerances (Only Relative Thresholds)
# =============================================================================

# Relative error thresholds for forward/backward pass accuracy
DEFAULT_FWD_REL_TOL = 1e-4
DEFAULT_BWD_REL_TOL = 1e-4

# Relative correction thresholds (corrections relative to original values)
DEFAULT_FWD_CORRECTION_REL_TOL = 1e-3  # Forward corrections relative to original activations
DEFAULT_BWD_CORRECTION_REL_TOL = 1e-3  # Backward corrections relative to original gradients


# =============================================================================
# Column Display Configuration
# =============================================================================

SHOW_SENSITIVITY_NORMS = True      # Show ||δ_pp||, ||δ_np||, ||δ_pn||, ||δ_nn|| columns
SHOW_ACTIVATION_NORMS = True       # Show ||pos||, ||neg|| columns
SHOW_ORIGINAL_GRAD_NORMS = True    # Show ||∇_orig|| column
SHOW_CORRECTION_NORMS = True      # Show ||Δfwd||, ||Δbwd|| alignment correction columns
INIT_RANDOM_BIASES = True          # Initialize biases randomly (not zeros)
INIT_LARGER_WEIGHTS = True         # Initialize weights with larger values for more significant gradients

# Alignment settings - when enabled, DC outputs are corrected to match original values exactly
ALIGN_FORWARD = True               # Align forward pass (DC pos-neg = original activation)
ALIGN_BACKWARD = True              # Align backward pass (DC sensitivities = original gradient)

# Sensitivity shift alpha - reduces sensitivity magnitudes for numerical stability
# Set to a float for constant alpha (e.g., 0.5), or "frobenius" to compute from weights
ALPHA = 0.5  # 0.0 = disabled, "frobenius" = compute from Frobenius norm, float = constant


# =============================================================================
# Result Data Classes
# =============================================================================

@dataclass
class LayerResult:
    """Result for a single layer."""
    name: str
    module_type: str
    display_name: str = ""  # Custom display name with input info

    # Relative errors only
    fwd_rel_error: float = 0.0
    bwd_rel_error: float = 0.0

    # Relative correction errors (corrections relative to original values)
    fwd_correction_rel_error: float = 0.0  # Correction relative to original activation
    bwd_correction_rel_error: float = 0.0  # Correction relative to original gradient

    # L2 norms of sensitivity components
    sens_pp_norm: float = 0.0
    sens_np_norm: float = 0.0  
    sens_pn_norm: float = 0.0
    sens_nn_norm: float = 0.0

    # L2 norms of pos/neg activation components
    act_pos_norm: float = 0.0
    act_neg_norm: float = 0.0

    # L2 norm of original gradient
    orig_grad_norm: float = 0.0

    # Alignment correction norms (from AlignmentCache)
    fwd_correction_norm: float = 0.0
    bwd_correction_norm: float = 0.0

    # Pass/fail
    fwd_pass: bool = True
    bwd_pass: bool = True


@dataclass
class TestResult:
    """Result for a single model test."""
    name: str
    success: bool = False
    error_message: str = ""

    # Overall errors (max across layers) - relative only
    fwd_rel_error: float = float('inf')
    bwd_rel_error: float = float('inf')
    
    # Overall correction errors (max across layers)
    fwd_correction_rel_error: float = float('inf')  # Max forward correction relative to original
    bwd_correction_rel_error: float = float('inf')  # Max backward correction relative to original

    # Pass/fail for each phase
    fwd_pass: bool = False
    bwd_pass: bool = False
    fwd_correction_pass: bool = False
    bwd_correction_pass: bool = False

    # Layer-wise results
    layer_results: List[LayerResult] = field(default_factory=list)

    # API used for this test
    api_used: str = "functional"

    # Tolerances used (relative only)
    fwd_rel_tol: float = DEFAULT_FWD_REL_TOL
    bwd_rel_tol: float = DEFAULT_BWD_REL_TOL
    fwd_correction_rel_tol: float = DEFAULT_FWD_CORRECTION_REL_TOL
    bwd_correction_rel_tol: float = DEFAULT_BWD_CORRECTION_REL_TOL


# =============================================================================
# Layer Cache for Hook-based Testing
# =============================================================================

class LayerCache:
    """Cache activations and gradients for each layer."""

    def __init__(self):
        self.orig_outputs: Dict[str, Tensor] = {}
        self.orig_grad_inputs: Dict[str, Tensor] = {}
        self.dc_outputs: Dict[str, Tensor] = {}
        self.dc_grad_inputs: Dict[str, Tensor] = {}
        self.dc_grad_inputs_raw: Dict[str, Tensor] = {}  # Raw [4*batch] gradients for sum check
        self.dc_activations_raw: Dict[str, Tensor] = {}  # Raw [4*batch] activations for norm computation
        self.layer_types: Dict[str, str] = {}
        self.layer_order: List[str] = []  # Will be populated in execution order
        self.execution_order: List[str] = []  # Track actual execution order during forward pass
        self._handles: List = []

    def clear(self):
        self.orig_outputs.clear()
        self.orig_grad_inputs.clear()
        self.dc_outputs.clear()
        self.dc_grad_inputs.clear()
        self.dc_grad_inputs_raw.clear()
        self.dc_activations_raw.clear()
        self.layer_types.clear()
        self.layer_order.clear()
        self.execution_order.clear()
        self._remove_hooks()

    def _remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def register_orig_hooks(self, model: nn.Module):
        """Register hooks to capture original activations/gradients and track execution order."""
        self._remove_hooks()
        self.execution_order.clear()

        for name, module in model.named_modules():
            if self._is_trackable(module) and name:
                self.layer_types[name] = module.__class__.__name__

                def make_fwd_hook(n):
                    def hook(m, inp, out):
                        # Track execution order during forward pass
                        if n not in self.execution_order:
                            self.execution_order.append(n)
                        self.orig_outputs[n] = out.detach().clone()
                    return hook

                def make_bwd_hook(n):
                    def hook(m, grad_in, grad_out):
                        if grad_in[0] is not None:
                            self.orig_grad_inputs[n] = grad_in[0].detach().clone()
                    return hook

                self._handles.append(module.register_forward_hook(make_fwd_hook(name)))
                self._handles.append(module.register_full_backward_hook(make_bwd_hook(name)))

        # After all hooks are registered, layer_order will be populated during forward pass

    def register_dc_hooks(self, model: nn.Module):
        """Register hooks to capture DC activations/gradients."""
        self._remove_hooks()

        for name, module in model.named_modules():
            if self._is_trackable(module) and name:
                self.layer_types[name] = module.__class__.__name__

                def make_fwd_hook(n):
                    def hook(m, inp, out):
                        # Store raw activations for norm computation
                        self.dc_activations_raw[n] = out.detach().clone()
                        # Reconstruct from [4*batch] format
                        if out.shape[0] % 4 == 0:
                            q = out.shape[0] // 4
                            pos, neg = out[:q], out[q:2*q]
                            self.dc_outputs[n] = (pos - neg).detach().clone()
                        else:
                            self.dc_outputs[n] = out.detach().clone()
                    return hook

                def make_bwd_hook(n):
                    def hook(m, grad_in, grad_out):
                        if grad_in[0] is not None:
                            grad = grad_in[0].detach()
                            # Store raw gradient for sum check
                            self.dc_grad_inputs_raw[n] = grad.clone()
                            # Reconstruct from [4*batch] sensitivities
                            if grad.shape[0] % 4 == 0:
                                q = grad.shape[0] // 4
                                pp, np, pn, nn = grad[:q], grad[q:2*q], grad[2*q:3*q], grad[3*q:]
                                self.dc_grad_inputs[n] = (pp - np - pn + nn).clone()
                            else:
                                self.dc_grad_inputs[n] = grad.clone()
                    return hook

                self._handles.append(module.register_forward_hook(make_fwd_hook(name)))
                self._handles.append(module.register_full_backward_hook(make_bwd_hook(name)))

    def _is_trackable(self, module: nn.Module) -> bool:
        """Check if module should be tracked."""
        # Import custom modules
        try:
            from dc_decompose.operations.add import Add
            from dc_decompose.operations.mul import DCMul  
            from dc_decompose.operations.mean import Mean
            from dc_decompose.operations.contiguous import Contiguous
            from dc_decompose.operations.shape_ops import Reshape, View, Squeeze, Unsqueeze
            from dc_decompose.operations.tensor_ops import DCSplit, DCChunk, DCCat, DCSlice
            custom_modules = (Add, DCMul, Mean, Contiguous, Reshape, View, Squeeze, Unsqueeze, 
                            DCSplit, DCChunk, DCCat, DCSlice)
        except ImportError:
            custom_modules = ()
        
        return isinstance(module, (
            nn.Linear, nn.Conv1d, nn.Conv2d,
            nn.ReLU, nn.LeakyReLU, nn.GELU,
            nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm,
            nn.MaxPool1d, nn.MaxPool2d,
            nn.AvgPool1d, nn.AvgPool2d,
            nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d,
            nn.Flatten, nn.Dropout,
        ) + custom_modules)

    def finalize_execution_order(self):
        """Update layer_order to match execution_order after forward pass is complete."""
        if self.execution_order:
            # Update layer_order to reflect actual execution order
            self.layer_order = self.execution_order.copy()

    def compute_errors(self, fwd_rel_tol: float, bwd_rel_tol: float,
                       fwd_correction_rel_tol: float, bwd_correction_rel_tol: float,
                       alignment_cache = None,
                       output_orig: Optional[Tensor] = None,
                       output_dc: Optional[Tensor] = None,
                       sens_grad: Optional[Tensor] = None,
                       orig_grad: Optional[Tensor] = None,
                       input_grad_raw: Optional[Tensor] = None) -> List[LayerResult]:
        """Compute layer-wise errors."""
        results = []

        # Input layer (sensitivities -> reconstructed gradient) - FIRST in execution order
        if sens_grad is not None and orig_grad is not None:
            lr = LayerResult(name=">>> INPUT", module_type="sensitivities")
            lr.fwd_rel_error = 0.0
            lr.fwd_pass = True
            diff = (sens_grad - orig_grad).abs()
            orig_norm = orig_grad.abs().max().item()
            lr.bwd_rel_error = diff.max().item() / (orig_norm + 1e-10)
            lr.bwd_pass = check_pass_relative_only(lr.bwd_rel_error, bwd_rel_tol)
            
            # Compute original gradient norm for input
            lr.orig_grad_norm = orig_grad.norm().item()
            
            # Compute L2 norms of sensitivity components for input
            if input_grad_raw is not None and input_grad_raw.shape[0] % 4 == 0:
                q = input_grad_raw.shape[0] // 4
                pp, np, pn, nn = input_grad_raw[:q], input_grad_raw[q:2*q], input_grad_raw[2*q:3*q], input_grad_raw[3*q:]
                lr.sens_pp_norm = pp.norm().item()
                lr.sens_np_norm = np.norm().item()
                lr.sens_pn_norm = pn.norm().item()
                lr.sens_nn_norm = nn.norm().item()
            
            results.append(lr)

        # Use execution order captured during forward pass, fall back to layer_order if needed
        execution_order = self.execution_order if self.execution_order else self.layer_order
        
        # Model layers - MIDDLE in execution order
        for name in execution_order:
            layer_type = self.layer_types.get(name, "Unknown")
            lr = LayerResult(name=name, module_type=layer_type)
            lr.display_name = _get_add_display_name(name, layer_type)

            # Forward error (relative only)
            if name in self.orig_outputs and name in self.dc_outputs:
                orig = self.orig_outputs[name]
                dc = self.dc_outputs[name]
                if orig.shape == dc.shape:
                    diff = (orig - dc).abs()
                    orig_norm = orig.abs().max().item()
                    lr.fwd_rel_error = diff.max().item() / (orig_norm + 1e-10)
                    lr.fwd_pass = check_pass_relative_only(lr.fwd_rel_error, fwd_rel_tol)

            # Backward error (relative only)
            if name in self.orig_grad_inputs and name in self.dc_grad_inputs:
                orig = self.orig_grad_inputs[name]
                dc = self.dc_grad_inputs[name]
                if orig.shape == dc.shape:
                    diff = (orig - dc).abs()
                    orig_norm = orig.abs().max().item()
                    lr.bwd_rel_error = diff.max().item() / (orig_norm + 1e-10)
                    lr.bwd_pass = check_pass_relative_only(lr.bwd_rel_error, bwd_rel_tol)
                
                # Compute original gradient norm
                lr.orig_grad_norm = orig.norm().item()

            # Compute L2 norms of sensitivity components
            if name in self.dc_grad_inputs_raw:
                raw_grad = self.dc_grad_inputs_raw[name]
                if raw_grad.shape[0] % 4 == 0:
                    q = raw_grad.shape[0] // 4
                    pp, np, pn, nn = raw_grad[:q], raw_grad[q:2*q], raw_grad[2*q:3*q], raw_grad[3*q:]
                    lr.sens_pp_norm = pp.norm().item()
                    lr.sens_np_norm = np.norm().item()
                    lr.sens_pn_norm = pn.norm().item()
                    lr.sens_nn_norm = nn.norm().item()

            # Compute L2 norms of activation components
            if name in self.dc_activations_raw:
                raw_act = self.dc_activations_raw[name]
                if raw_act.shape[0] % 4 == 0:
                    q = raw_act.shape[0] // 4
                    pos, neg = raw_act[:q], raw_act[q:2*q]
                    lr.act_pos_norm = pos.norm().item()
                    lr.act_neg_norm = neg.norm().item()
            
            # Correction errors will be populated later from AlignmentCache stats

            results.append(lr)

        # Output layer (reconstruct_output) - LAST in execution order
        if output_orig is not None and output_dc is not None:
            lr = LayerResult(name="<<< OUTPUT", module_type="reconstruct")
            diff = (output_orig - output_dc).abs()
            orig_norm = output_orig.abs().max().item()
            lr.fwd_rel_error = diff.max().item() / (orig_norm + 1e-10)
            lr.fwd_pass = check_pass_relative_only(lr.fwd_rel_error, fwd_rel_tol)
            # Backward at output is just the initialization, no error there
            lr.bwd_rel_error = 0.0
            lr.bwd_pass = True
            results.append(lr)

        return results


# =============================================================================
# Core Testing Functions  
# =============================================================================

def _get_add_display_name(layer_name: str, module_type: str) -> str:
    """Generate display name for Add modules showing input sources."""
    if module_type == "Add" and "_dc_add_" in layer_name:
        # Parse ResNet Add module names like "blocks.0._dc_add_0" 
        if "blocks." in layer_name and "._dc_add_" in layer_name:
            # Extract block number, e.g., "blocks.0._dc_add_0" -> "0"
            parts = layer_name.split(".")
            if len(parts) >= 2 and parts[0] == "blocks":
                block_num = parts[1]
                main_path = f"blocks.{block_num}.bn2"
                skip_path = f"blocks.{block_num} identity"
                return f"Add({main_path} + {skip_path})"
        
        # Parse simple Add module names like "_dc_add_0"
        elif layer_name.startswith("_dc_add_"):
            return f"Add(conv2 + identity)"
            
        # Fallback for other Add patterns
        return f"Add(* + *)"
    
    return layer_name


def init_random_biases(model: nn.Module) -> None:
    """Initialize weights and biases to keep gradient norm ~1.0.

    Scales weight initialization based on the number of linear/conv layers
    to prevent gradient explosion/vanishing.
    """
    # Count linear/conv layers for scaling
    num_layers = sum(1 for m in model.modules()
                     if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)))
    num_layers = max(num_layers, 1)

    # Scale factor to keep gradient norm ~1.0
    # Each layer contributes to gradient magnitude; scale down to compensate
    scale_factor = 1.0 / (num_layers ** 0.5) if not INIT_LARGER_WEIGHTS else 3.0

    for module in model.modules():
        with torch.no_grad():
            if hasattr(module, 'weight') and module.weight is not None:
                if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                    # Xavier init scaled by layer count
                    fan_in = module.weight.shape[1] if len(module.weight.shape) > 1 else module.weight.shape[0]
                    fan_out = module.weight.shape[0]
                    std = scale_factor * (2.0 / (fan_in + fan_out)) ** 0.5
                    module.weight.normal_(0.0, std)
                elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    # BatchNorm weight (gamma): close to 1
                    module.weight.uniform_(0.9, 1.1)

            # Initialize bias with random values if enabled
            if INIT_RANDOM_BIASES and hasattr(module, 'bias') and module.bias is not None:
                if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                    # Small bias to ensure some activations
                    module.bias.uniform_(0.0, 0.1)
                elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    # BatchNorm bias (beta): small random values around 0
                    module.bias.uniform_(-0.05, 0.05)


def check_pass_relative_only(rel_err: float, rel_tol: float) -> bool:
    """Pass if relative error < rel_tol.
    
    Only uses relative thresholds as requested.
    """
    return rel_err < rel_tol


def check_layer_wise_pass(layer_results: List[LayerResult], 
                         fwd_rel_tol: float,
                         bwd_rel_tol: float,
                         fwd_correction_rel_tol: float,
                         bwd_correction_rel_tol: float) -> Tuple[bool, bool, bool, bool]:
    """Check pass/fail for all layers and return overall pass status.
    
    Returns:
        Tuple of (fwd_pass, bwd_pass, fwd_correction_pass, bwd_correction_pass)
    """
    fwd_pass = True
    bwd_pass = True  
    fwd_correction_pass = True
    bwd_correction_pass = True
    
    for layer in layer_results:
        # Check forward pass
        layer.fwd_pass = check_pass_relative_only(layer.fwd_rel_error, fwd_rel_tol)
        if not layer.fwd_pass:
            fwd_pass = False
            
        # Check backward pass  
        layer.bwd_pass = check_pass_relative_only(layer.bwd_rel_error, bwd_rel_tol)
        if not layer.bwd_pass:
            bwd_pass = False
            
        # Check forward correction
        fwd_corr_pass = check_pass_relative_only(layer.fwd_correction_rel_error, fwd_correction_rel_tol)
        if not fwd_corr_pass:
            fwd_correction_pass = False
            
        # Check backward correction
        bwd_corr_pass = check_pass_relative_only(layer.bwd_correction_rel_error, bwd_correction_rel_tol)
        if not bwd_corr_pass:
            bwd_correction_pass = False
    
    return fwd_pass, bwd_pass, fwd_correction_pass, bwd_correction_pass


def test_model_functional(
    model: nn.Module,
    x: Tensor,
    name: str = "model",
    fwd_rel_tol: float = DEFAULT_FWD_REL_TOL,
    bwd_rel_tol: float = DEFAULT_BWD_REL_TOL,
    fwd_correction_rel_tol: float = DEFAULT_FWD_CORRECTION_REL_TOL,
    bwd_correction_rel_tol: float = DEFAULT_BWD_CORRECTION_REL_TOL,
) -> TestResult:
    """Test using the functional API (init_catted + reconstruct_output)."""

    result = TestResult(
        name=name,
        api_used="functional",
        fwd_rel_tol=fwd_rel_tol,
        bwd_rel_tol=bwd_rel_tol,
        fwd_correction_rel_tol=fwd_correction_rel_tol,
        bwd_correction_rel_tol=bwd_correction_rel_tol,
    )

    cache = LayerCache()

    try:
        model.eval()
        init_random_biases(model)

        # =====================================================================
        # Phase 1: Replace functional calls with modules
        # =====================================================================
        model = replace_functional_with_modules(model, inplace=True)

        # =====================================================================
        # Phase 2: Prepare for DC (captures originals if alignment enabled)
        # =====================================================================
        # Generate target_grad first by doing a dummy forward
        with torch.no_grad():
            dummy_out = model(x)
            target_grad = torch.randn_like(dummy_out)

        # Apply sensitivity alpha BEFORE cache capture to ensure consistency
        if ALPHA == "frobenius":
            set_sensitivity_alpha(model, mode='frobenius')
        elif isinstance(ALPHA, (int, float)) and ALPHA > 0:
            set_sensitivity_alpha(model, alpha=float(ALPHA), mode='constant')

        model = prepare_model_for_dc(
            model,
            align_forward=ALIGN_FORWARD,
            align_backward=ALIGN_BACKWARD,
            x=x,
            target_grad=target_grad,
        )

        # Get alignment cache from model (if alignment is enabled)
        alignment_cache = None
        for module in model.modules():
            if hasattr(module, '_dc_alignment_cache'):
                alignment_cache = getattr(module, '_dc_alignment_cache')
                break

        # If alignment is enabled, use cached originals; otherwise capture them
        if alignment_cache is not None and (ALIGN_FORWARD or ALIGN_BACKWARD):
            # Use alignment cache's captured originals for comparison
            # Note: get_layer_names() returns layers in execution order from alignment cache
            cache.execution_order = list(alignment_cache.get_layer_names())
            for layer_name in alignment_cache.get_layer_names():
                data = alignment_cache.get_layer_data(layer_name)
                if data is not None:
                    cache.layer_types[layer_name] = "Unknown"  # Will be filled by DC hooks
                    if layer_name not in cache.layer_order:
                        cache.layer_order.append(layer_name)
                    if data.original_activation is not None:
                        cache.orig_outputs[layer_name] = data.original_activation
                    if data.original_gradient is not None:
                        cache.orig_grad_inputs[layer_name] = data.original_gradient

            # Use alignment cache's captured output and gradient for comparison
            orig_out = alignment_cache.original_output
            grad_orig = alignment_cache.original_input_grad
        else:
            # Capture originals the old way (DC disabled)
            from dc_decompose.patcher import set_dc_enabled
            set_dc_enabled(model, False)
            cache.register_orig_hooks(model)
            x_orig = x.clone().requires_grad_(True)
            orig_out = model(x_orig)
            orig_out.backward(target_grad)
            grad_orig = x_orig.grad.clone()
            cache._remove_hooks()
            set_dc_enabled(model, True)

        # =====================================================================
        # Phase 3: DC forward/backward using FUNCTIONAL API
        # =====================================================================
        cache.register_dc_hooks(model)

        # Functional API: init_catted is preprocessing (detached)
        x_cat = init_catted(x, InputMode.CENTER)
        out_cat = model(x_cat)
        dc_out = reconstruct_output(out_cat)
        dc_out.backward(target_grad)

        # Get sensitivities and reconstruct gradient
        sens = extract_sensitivities(x_cat.grad)
        grad_dc = sens.reconstruct_gradient()

        cache._remove_hooks()
        unpatch_model(model)

        # Finalize execution order based on captured forward pass
        cache.finalize_execution_order()

        # =====================================================================
        # Compute layer-wise errors
        # =====================================================================
        result.layer_results = cache.compute_errors(
            fwd_rel_tol, bwd_rel_tol, fwd_correction_rel_tol, bwd_correction_rel_tol,
            alignment_cache=alignment_cache,
            output_orig=orig_out.detach(), output_dc=dc_out.detach(),
            sens_grad=grad_dc, orig_grad=grad_orig,
            input_grad_raw=x_cat.grad
        )

        # =====================================================================
        # Populate correction norms from AlignmentCache
        # =====================================================================
        if alignment_cache is not None:
            correction_stats = {s.layer_name: s for s in alignment_cache.get_correction_stats()}
            for lr in result.layer_results:
                if lr.name in correction_stats:
                    stats = correction_stats[lr.name]
                    lr.fwd_correction_norm = stats.forward_correction_norm
                    lr.bwd_correction_norm = stats.backward_correction_norm
                    # Use the relative correction values from alignment cache
                    lr.fwd_correction_rel_error = stats.forward_relative_correction
                    lr.bwd_correction_rel_error = stats.backward_relative_correction

        # =====================================================================
        # Layer-wise checking and overall pass/fail determination
        # =====================================================================

        # Use layer-wise checking to determine overall pass/fail
        result.fwd_pass, result.bwd_pass, result.fwd_correction_pass, result.bwd_correction_pass = check_layer_wise_pass(
            result.layer_results, fwd_rel_tol, bwd_rel_tol, fwd_correction_rel_tol, bwd_correction_rel_tol
        )
        
        # Compute max errors across layers (for reporting)
        result.fwd_rel_error = max((lr.fwd_rel_error for lr in result.layer_results), default=0.0)
        result.bwd_rel_error = max((lr.bwd_rel_error for lr in result.layer_results), default=0.0)
        result.fwd_correction_rel_error = max((lr.fwd_correction_rel_error for lr in result.layer_results), default=0.0)
        result.bwd_correction_rel_error = max((lr.bwd_correction_rel_error for lr in result.layer_results), default=0.0)

        # Overall success requires all phases to pass
        result.success = (result.fwd_pass and result.bwd_pass and 
                         result.fwd_correction_pass and result.bwd_correction_pass)

    except Exception as e:
        import traceback
        result.error_message = f"{e}\n{traceback.format_exc()}"
        result.success = False

    return result


def test_model_context_manager(
    model: nn.Module,
    x: Tensor,
    name: str = "model",
    fwd_rel_tol: float = DEFAULT_FWD_REL_TOL,
    bwd_rel_tol: float = DEFAULT_BWD_REL_TOL,
    fwd_correction_rel_tol: float = DEFAULT_FWD_CORRECTION_REL_TOL,
    bwd_correction_rel_tol: float = DEFAULT_BWD_CORRECTION_REL_TOL,
) -> TestResult:
    """Test using the context manager API (dc_forward)."""

    result = TestResult(
        name=name,
        api_used="context_manager",
        fwd_rel_tol=fwd_rel_tol,
        bwd_rel_tol=bwd_rel_tol,
        fwd_correction_rel_tol=fwd_correction_rel_tol,
        bwd_correction_rel_tol=bwd_correction_rel_tol,
    )

    cache = LayerCache()

    try:
        model.eval()
        init_random_biases(model)

        # =====================================================================
        # Phase 1: Replace functional calls with modules
        # =====================================================================
        model = replace_functional_with_modules(model, inplace=True)

        # =====================================================================
        # Phase 2: Prepare for DC (captures originals if alignment enabled)
        # =====================================================================
        # Generate target_grad first by doing a dummy forward
        with torch.no_grad():
            dummy_out = model(x)
            target_grad = torch.randn_like(dummy_out)

        # Apply sensitivity alpha BEFORE cache capture to ensure consistency
        if ALPHA == "frobenius":
            set_sensitivity_alpha(model, mode='frobenius')
        elif isinstance(ALPHA, (int, float)) and ALPHA > 0:
            set_sensitivity_alpha(model, alpha=float(ALPHA), mode='constant')

        model = prepare_model_for_dc(
            model,
            align_forward=ALIGN_FORWARD,
            align_backward=ALIGN_BACKWARD,
            x=x,
            target_grad=target_grad,
        )

        # Get alignment cache from model (if alignment is enabled)
        alignment_cache = None
        for module in model.modules():
            if hasattr(module, '_dc_alignment_cache'):
                alignment_cache = getattr(module, '_dc_alignment_cache')
                break

        # If alignment is enabled, use cached originals; otherwise capture them
        if alignment_cache is not None and (ALIGN_FORWARD or ALIGN_BACKWARD):
            # Use alignment cache's captured originals for comparison
            for layer_name in alignment_cache.get_layer_names():
                data = alignment_cache.get_layer_data(layer_name)
                if data is not None:
                    cache.layer_types[layer_name] = "Unknown"
                    if layer_name not in cache.layer_order:
                        cache.layer_order.append(layer_name)
                    if data.original_activation is not None:
                        cache.orig_outputs[layer_name] = data.original_activation
                    if data.original_gradient is not None:
                        cache.orig_grad_inputs[layer_name] = data.original_gradient

            # Use alignment cache's captured output and gradient for comparison
            orig_out = alignment_cache.original_output
            grad_orig = alignment_cache.original_input_grad
        else:
            # Capture originals the old way (DC disabled)
            from dc_decompose.patcher import set_dc_enabled
            set_dc_enabled(model, False)
            cache.register_orig_hooks(model)
            x_orig = x.clone().requires_grad_(True)
            orig_out = model(x_orig)
            orig_out.backward(target_grad)
            grad_orig = x_orig.grad.clone()
            cache._remove_hooks()
            set_dc_enabled(model, True)

        # =====================================================================
        # Phase 3: DC forward/backward using CONTEXT MANAGER API
        # =====================================================================
        cache.register_dc_hooks(model)

        with dc_forward(model, x, beta=1.0) as dc:
            dc_out = dc.output
            dc_out.backward(target_grad)

        # Get reconstructed gradient from sensitivities
        grad_dc = dc.reconstruct_gradient()
        input_grad_raw = dc.input_4.grad

        cache._remove_hooks()
        unpatch_model(model)

        # Finalize execution order based on captured forward pass
        cache.finalize_execution_order()

        # =====================================================================
        # Compute layer-wise errors
        # =====================================================================
        result.layer_results = cache.compute_errors(
            fwd_rel_tol, bwd_rel_tol, fwd_correction_rel_tol, bwd_correction_rel_tol,
            alignment_cache=None,  # Context manager doesn't use alignment cache in the same way
            output_orig=orig_out.detach(), output_dc=dc_out.detach(),
            sens_grad=grad_dc, orig_grad=grad_orig,
            input_grad_raw=input_grad_raw
        )

        # =====================================================================
        # Populate correction norms from AlignmentCache
        # =====================================================================
        if alignment_cache is not None:
            correction_stats = {s.layer_name: s for s in alignment_cache.get_correction_stats()}
            for lr in result.layer_results:
                if lr.name in correction_stats:
                    stats = correction_stats[lr.name]
                    lr.fwd_correction_norm = stats.forward_correction_norm
                    lr.bwd_correction_norm = stats.backward_correction_norm
                    # Use the relative correction values from alignment cache
                    lr.fwd_correction_rel_error = stats.forward_relative_correction
                    lr.bwd_correction_rel_error = stats.backward_relative_correction

        # =====================================================================
        # Layer-wise checking and overall pass/fail determination
        # =====================================================================

        # Use layer-wise checking to determine overall pass/fail
        result.fwd_pass, result.bwd_pass, result.fwd_correction_pass, result.bwd_correction_pass = check_layer_wise_pass(
            result.layer_results, fwd_rel_tol, bwd_rel_tol, fwd_correction_rel_tol, bwd_correction_rel_tol
        )
        
        # Compute max errors across layers (for reporting)
        result.fwd_rel_error = max((lr.fwd_rel_error for lr in result.layer_results), default=0.0)
        result.bwd_rel_error = max((lr.bwd_rel_error for lr in result.layer_results), default=0.0)
        result.fwd_correction_rel_error = max((lr.fwd_correction_rel_error for lr in result.layer_results), default=0.0)
        result.bwd_correction_rel_error = max((lr.bwd_correction_rel_error for lr in result.layer_results), default=0.0)

        # Overall success requires all phases to pass
        result.success = (result.fwd_pass and result.bwd_pass and 
                         result.fwd_correction_pass and result.bwd_correction_pass)

    except Exception as e:
        import traceback
        result.error_message = f"{e}\n{traceback.format_exc()}"
        result.success = False

    return result


# =============================================================================
# Test with Loss Function (non-split loss on reconstructed output)
# =============================================================================


def test_model_with_loss_functional(
    model: nn.Module,
    x: Tensor,
    target: Tensor,
    loss_fn: LossFn,
    name: str = "model",
    backward_only: bool = False,
    fwd_rel_tol: float = DEFAULT_FWD_REL_TOL,
    bwd_rel_tol: float = DEFAULT_BWD_REL_TOL,
    fwd_correction_rel_tol: float = DEFAULT_FWD_CORRECTION_REL_TOL,
    bwd_correction_rel_tol: float = DEFAULT_BWD_CORRECTION_REL_TOL,
) -> TestResult:
    """
    Test DC decomposition with a loss function applied to reconstructed output.

    This tests the scenario where:
    1. Model has multi-dimensional output (not just scalar)
    2. Loss is computed on the reconstructed output (pos - neg), not split
    3. Optionally uses backward-only mode with cached masks from original forward

    Args:
        model: The model to test
        x: Input tensor
        target: Target tensor for loss computation
        loss_fn: Loss function (output, target) -> scalar loss
        backward_only: If True, use cached masks from original forward (no DC forward decomposition)
        fwd_rel_tol, bwd_rel_tol: Forward/backward pass tolerances
        fwd_correction_rel_tol, bwd_correction_rel_tol: Correction tolerances
    """
    result = TestResult(
        name=name,
        api_used="functional",
        fwd_rel_tol=fwd_rel_tol,
        bwd_rel_tol=bwd_rel_tol,
        fwd_correction_rel_tol=fwd_correction_rel_tol,
        bwd_correction_rel_tol=bwd_correction_rel_tol,
    )

    try:
        model.eval()
        init_random_biases(model)

        # Phase 1: Replace functional calls with modules
        model = replace_functional_with_modules(model, inplace=True)

        # Phase 2: Compute original forward/backward with loss
        x_orig = x.clone().requires_grad_(True)
        orig_out = model(x_orig)
        orig_loss = loss_fn(orig_out, target)
        orig_loss.backward()
        grad_orig = x_orig.grad.clone()
        orig_out_detached = orig_out.detach().clone()

        # Phase 3: Prepare model for DC (no alignment - loss gradient is computed dynamically)
        # For loss-based testing, we don't use alignment since the gradient comes from the loss
        from dc_decompose.patcher import prepare_model_for_dc
        
        # Generate target_grad for consistency
        with torch.no_grad():
            dummy_out = model(x)
            target_grad = torch.randn_like(dummy_out)
        
        model = prepare_model_for_dc(
            model,
            align_forward=False,  # No alignment for loss-based tests
            align_backward=False,
            x=x,
            target_grad=target_grad,
        )

        # Apply sensitivity alpha if configured
        if ALPHA == "frobenius":
            set_sensitivity_alpha(model, mode='frobenius')
        elif isinstance(ALPHA, (int, float)) and ALPHA > 0:
            set_sensitivity_alpha(model, alpha=float(ALPHA), mode='constant')

        # Phase 4: DC forward/backward with loss
        x_cat = init_catted(x, InputMode.CENTER)
        out_cat = model(x_cat)
        dc_out = reconstruct_output(out_cat)

        # Apply loss on reconstructed output (not split)
        dc_loss = loss_fn(dc_out, target)
        dc_loss.backward()

        # Reconstruct gradient from sensitivities
        sens = extract_sensitivities(x_cat.grad)
        grad_dc = sens.reconstruct_gradient()

        unpatch_model(model)

        # Compute errors (relative only)
        fwd_err = torch.abs(dc_out.detach() - orig_out_detached)
        orig_norm = orig_out_detached.abs().max().item()
        result.fwd_rel_error = fwd_err.max().item() / max(orig_norm, 1e-10)

        bwd_err = torch.abs(grad_dc - grad_orig)
        grad_norm = grad_orig.abs().max().item()
        result.bwd_rel_error = bwd_err.max().item() / max(grad_norm, 1e-10)

        # No correction errors for these simpler tests (set to 0)
        result.fwd_correction_rel_error = 0.0
        result.bwd_correction_rel_error = 0.0

        result.fwd_pass = check_pass_relative_only(result.fwd_rel_error, fwd_rel_tol)
        result.bwd_pass = check_pass_relative_only(result.bwd_rel_error, bwd_rel_tol)
        result.fwd_correction_pass = True  # No corrections to check
        result.bwd_correction_pass = True  # No corrections to check
        result.success = (result.fwd_pass and result.bwd_pass and 
                         result.fwd_correction_pass and result.bwd_correction_pass)

    except Exception as e:
        import traceback
        result.error_message = f"{e}\n{traceback.format_exc()}"
        result.success = False

    return result


def test_model_with_loss_context_manager(
    model: nn.Module,
    x: Tensor,
    target: Tensor,
    loss_fn: LossFn,
    name: str = "model",
    backward_only: bool = False,
    fwd_rel_tol: float = DEFAULT_FWD_REL_TOL,
    bwd_rel_tol: float = DEFAULT_BWD_REL_TOL,
    fwd_correction_rel_tol: float = DEFAULT_FWD_CORRECTION_REL_TOL,
    bwd_correction_rel_tol: float = DEFAULT_BWD_CORRECTION_REL_TOL,
) -> TestResult:
    """
    Test DC decomposition with a loss function using the context manager API.

    Same as test_model_with_loss_functional but uses dc_forward context manager.
    """
    result = TestResult(
        name=name,
        api_used="context_manager",
        fwd_rel_tol=fwd_rel_tol,
        bwd_rel_tol=bwd_rel_tol,
        fwd_correction_rel_tol=fwd_correction_rel_tol,
        bwd_correction_rel_tol=bwd_correction_rel_tol,
    )

    try:
        model.eval()
        init_random_biases(model)

        # Phase 1: Replace functional calls with modules  
        from dc_decompose.functional_replacer import replace_functional_with_modules
        model = replace_functional_with_modules(model, inplace=True)

        # Phase 2: Compute original forward/backward with loss
        x_orig = x.clone().requires_grad_(True)
        orig_out = model(x_orig)
        orig_loss = loss_fn(orig_out, target)
        orig_loss.backward()
        grad_orig = x_orig.grad.clone()
        orig_out_detached = orig_out.detach().clone()

        # Phase 3: Prepare model for DC (no alignment for loss-based testing)
        from dc_decompose.patcher import patch_model, find_output_layer, mark_output_layer
        
        # Find and mark output layer before patching
        output_layer = find_output_layer(model)
        patch_model(model)
        if output_layer is not None:
            mark_output_layer(output_layer, beta=1.0)

        # Apply sensitivity alpha if configured
        if ALPHA == "frobenius":
            set_sensitivity_alpha(model, mode='frobenius')
        elif isinstance(ALPHA, (int, float)) and ALPHA > 0:
            set_sensitivity_alpha(model, alpha=float(ALPHA), mode='constant')

        # Phase 4: DC forward/backward with loss using context manager
        with dc_forward(model, x, beta=1.0) as dc:
            dc_out = dc.output
            dc_loss = loss_fn(dc_out, target)
            dc_loss.backward()

        grad_dc = dc.reconstruct_gradient()
        unpatch_model(model)

        # Compute errors (relative only)
        fwd_err = torch.abs(dc_out.detach() - orig_out_detached)
        orig_norm = orig_out_detached.abs().max().item()
        result.fwd_rel_error = fwd_err.max().item() / max(orig_norm, 1e-10)

        bwd_err = torch.abs(grad_dc - grad_orig)
        grad_norm = grad_orig.abs().max().item()
        result.bwd_rel_error = bwd_err.max().item() / max(grad_norm, 1e-10)

        # No correction errors for these simpler tests (set to 0)
        result.fwd_correction_rel_error = 0.0
        result.bwd_correction_rel_error = 0.0

        result.fwd_pass = check_pass_relative_only(result.fwd_rel_error, fwd_rel_tol)
        result.bwd_pass = check_pass_relative_only(result.bwd_rel_error, bwd_rel_tol)
        result.fwd_correction_pass = True  # No corrections to check
        result.bwd_correction_pass = True  # No corrections to check
        result.success = (result.fwd_pass and result.bwd_pass and 
                         result.fwd_correction_pass and result.bwd_correction_pass)

    except Exception as e:
        import traceback
        result.error_message = f"{e}\n{traceback.format_exc()}"
        result.success = False

    return result


def test_model_with_loss(
    model: nn.Module,
    x: Tensor,
    target: Tensor,
    loss_fn: LossFn,
    name: str = "model",
    backward_only: bool = False,
    fwd_rel_tol: float = DEFAULT_FWD_REL_TOL,
    bwd_rel_tol: float = DEFAULT_BWD_REL_TOL,
    fwd_correction_rel_tol: float = DEFAULT_FWD_CORRECTION_REL_TOL,
    bwd_correction_rel_tol: float = DEFAULT_BWD_CORRECTION_REL_TOL,
    api: str = "both",
) -> Union[TestResult, Tuple[TestResult, TestResult]]:
    """
    Test DC decomposition with a loss function.

    Args:
        model: Model to test (will be copied for each API test)
        x: Input tensor
        target: Target tensor for loss computation
        loss_fn: Loss function (output, target) -> scalar
        backward_only: If True, use cached masks (no DC forward decomposition)
        api: "functional", "context_manager", or "both"

    Returns:
        TestResult or tuple of TestResults
    """
    import copy

    if api == "functional":
        return test_model_with_loss_functional(
            copy.deepcopy(model), x, target, loss_fn, name, backward_only,
            fwd_rel_tol, bwd_rel_tol, fwd_correction_rel_tol, bwd_correction_rel_tol
        )
    elif api == "context_manager":
        return test_model_with_loss_context_manager(
            copy.deepcopy(model), x, target, loss_fn, name, backward_only,
            fwd_rel_tol, bwd_rel_tol, fwd_correction_rel_tol, bwd_correction_rel_tol
        )
    else:  # both
        r1 = test_model_with_loss_functional(
            copy.deepcopy(model), x, target, loss_fn, f"{name} [functional]", backward_only,
            fwd_rel_tol, bwd_rel_tol, fwd_correction_rel_tol, bwd_correction_rel_tol
        )
        r2 = test_model_with_loss_context_manager(
            copy.deepcopy(model), x, target, loss_fn, f"{name} [context_mgr]", backward_only,
            fwd_rel_tol, bwd_rel_tol, fwd_correction_rel_tol, bwd_correction_rel_tol
        )
        return r1, r2


def run_loss_model_tests(
    models: Dict[str, LossModelSpec],
    title: str = "DC Decomposition Tests with Loss",
    backward_only: bool = False,
    fwd_rel_tol: float = DEFAULT_FWD_REL_TOL,
    bwd_rel_tol: float = DEFAULT_BWD_REL_TOL,
    fwd_correction_rel_tol: float = DEFAULT_FWD_CORRECTION_REL_TOL,
    bwd_correction_rel_tol: float = DEFAULT_BWD_CORRECTION_REL_TOL,
    seed: int = 42,
    verbose: bool = True,
    test_both_apis: bool = True,
) -> bool:
    """
    Run DC decomposition tests with loss functions on multiple models.

    Args:
        models: Dict mapping name -> (model_or_factory, input_spec, target, loss_fn)
        backward_only: If True, test backward-only mode with cached masks
        title: Title for test output
        test_both_apis: Whether to test both functional and context manager APIs

    Returns:
        True if all tests pass
    """
    torch.manual_seed(seed)

    mode_str = "backward-only (cached masks)" if backward_only else "full DC decomposition"

    if verbose:
        print("=" * 90)
        print(title)
        print("=" * 90)
        print(f"Mode: {mode_str}")
        print(f"Tolerances: fwd_rel={fwd_rel_tol:.0e}, bwd_rel={bwd_rel_tol:.0e}, "
              f"fwd_corr_rel={fwd_correction_rel_tol:.0e}, bwd_corr_rel={bwd_correction_rel_tol:.0e}")
        if test_both_apis:
            print("Testing both APIs: functional and context_manager")
        print()

    results: List[TestResult] = []

    for name, (model_spec, input_spec, target, loss_fn) in models.items():
        # Get model instance
        if callable(model_spec) and not isinstance(model_spec, nn.Module):
            model = model_spec()
        else:
            model = model_spec

        # Get input tensor
        if isinstance(input_spec, Tensor):
            x = input_spec
        else:
            x = torch.randn(*input_spec)

        # Run test(s)
        if test_both_apis:
            r1, r2 = test_model_with_loss(
                model, x, target, loss_fn, name, backward_only,
                fwd_rel_tol, bwd_rel_tol, fwd_correction_rel_tol, bwd_correction_rel_tol,
                api="both"
            )
            results.extend([r1, r2])
            if verbose:
                _print_result(r1, show_layers=True)
                _print_result(r2, show_layers=True)
        else:
            result = test_model_with_loss(
                model, x, target, loss_fn, name, backward_only,
                fwd_rel_tol, bwd_rel_tol, fwd_correction_rel_tol, bwd_correction_rel_tol,
                api="functional"
            )
            results.append(result)
            if verbose:
                _print_result(result, show_layers=True)

    # Print summary
    if verbose:
        _print_summary(results)

    return all(r.success for r in results)


def test_model(
    model: nn.Module,
    x: Tensor,
    name: str = "model",
    fwd_rel_tol: float = DEFAULT_FWD_REL_TOL,
    bwd_rel_tol: float = DEFAULT_BWD_REL_TOL,
    fwd_correction_rel_tol: float = DEFAULT_FWD_CORRECTION_REL_TOL,
    bwd_correction_rel_tol: float = DEFAULT_BWD_CORRECTION_REL_TOL,
    api: str = "both",
) -> Union[TestResult, Tuple[TestResult, TestResult]]:
    """
    Test a model's DC decomposition.

    Args:
        api: "functional", "context_manager", or "both" (default)
    """
    if api == "functional":
        return test_model_functional(model, x, name, fwd_rel_tol, bwd_rel_tol, fwd_correction_rel_tol, bwd_correction_rel_tol)
    elif api == "context_manager":
        return test_model_context_manager(model, x, name, fwd_rel_tol, bwd_rel_tol, fwd_correction_rel_tol, bwd_correction_rel_tol)
    else:  # both
        r1 = test_model_functional(model, x, f"{name} [functional]", fwd_rel_tol, bwd_rel_tol, fwd_correction_rel_tol, bwd_correction_rel_tol)
        r2 = test_model_context_manager(model, x, f"{name} [context_mgr]", fwd_rel_tol, bwd_rel_tol, fwd_correction_rel_tol, bwd_correction_rel_tol)
        return r1, r2


# =============================================================================
# Batch Testing with Layer-wise Output
# =============================================================================


def run_model_tests(
    models: Dict[str, Tuple[ModelSpec, InputSpec]],
    title: str = "DC Decomposition Tests",
    fwd_rel_tol: float = DEFAULT_FWD_REL_TOL,
    bwd_rel_tol: float = DEFAULT_BWD_REL_TOL,
    fwd_correction_rel_tol: float = DEFAULT_FWD_CORRECTION_REL_TOL,
    bwd_correction_rel_tol: float = DEFAULT_BWD_CORRECTION_REL_TOL,
    seed: int = 42,
    verbose: bool = True,
    show_layers: bool = True,
    test_both_apis: bool = True,
) -> bool:
    """
    Run DC decomposition tests on multiple models.

    Args:
        models: Dict mapping name -> (model_or_factory, input_tensor_or_shape)
        title: Title for test output
        show_layers: Whether to show layer-wise errors
        test_both_apis: Whether to test both functional and context manager APIs
    """
    torch.manual_seed(seed)

    if verbose:
        print("=" * 90)
        print(title)
        print("=" * 90)
        print(f"Tolerances: fwd_rel={fwd_rel_tol:.0e}, bwd_rel={bwd_rel_tol:.0e}, "
              f"fwd_corr_rel={fwd_correction_rel_tol:.0e}, bwd_corr_rel={bwd_correction_rel_tol:.0e}")
        if test_both_apis:
            print("Testing both APIs: functional and context_manager")
        print()

    results: List[TestResult] = []

    for name, (model_spec, input_spec) in models.items():
        # Get model instance
        if callable(model_spec) and not isinstance(model_spec, nn.Module):
            model = model_spec()
        else:
            model = model_spec

        # Get input tensor
        if isinstance(input_spec, Tensor):
            x = input_spec
        else:
            x = torch.randn(*input_spec)

        # Run test(s)
        if test_both_apis:
            r1, r2 = test_model(
                model, x, name,
                fwd_rel_tol=fwd_rel_tol,
                bwd_rel_tol=bwd_rel_tol,
                fwd_correction_rel_tol=fwd_correction_rel_tol,
                bwd_correction_rel_tol=bwd_correction_rel_tol,
                api="both",
            )
            results.extend([r1, r2])
            if verbose:
                _print_result(r1, show_layers=show_layers)
                _print_result(r2, show_layers=show_layers)
        else:
            result = test_model(
                model, x, name,
                fwd_rel_tol=fwd_rel_tol,
                bwd_rel_tol=bwd_rel_tol,
                fwd_correction_rel_tol=fwd_correction_rel_tol,
                bwd_correction_rel_tol=bwd_correction_rel_tol,
                api="functional",
            )
            results.append(result)
            if verbose:
                _print_result(result, show_layers=show_layers)

    # Print summary
    if verbose:
        _print_summary(results)

    return all(r.success for r in results)


def _print_result(result: TestResult, show_layers: bool = True) -> None:
    """Print a single test result with optional layer-wise details."""
    status = "PASS" if result.success else "FAIL"

    if result.error_message:
        print(f"{'='*100}")
        print(f"Model: {result.name} [{status}]")
        print(f"{'='*100}")
        print(f"  ERROR: {result.error_message}")
        print()
        return

    fwd_status = "ok" if result.fwd_pass else "FAIL"
    bwd_status = "ok" if result.bwd_pass else "FAIL"
    fwd_corr_status = "ok" if result.fwd_correction_pass else "FAIL"
    bwd_corr_status = "ok" if result.bwd_correction_pass else "FAIL"

    print(f"{'='*100}")
    print(f"Model: {result.name} [{status}]")
    print(f"{'='*100}")
    print(f"  Overall Forward:       rel={result.fwd_rel_error:.2e} [{fwd_status}]")
    print(f"  Overall Backward:      rel={result.bwd_rel_error:.2e} [{bwd_status}]")
    print(f"  Forward Correction:    rel={result.fwd_correction_rel_error:.2e} [{fwd_corr_status}]")
    print(f"  Backward Correction:   rel={result.bwd_correction_rel_error:.2e} [{bwd_corr_status}]")

    if show_layers and result.layer_results:
        print()
        # Build header dynamically based on configuration
        header = f"  {'Layer':<35} {'Type':<15} {'Fwd Rel':<10} {'Bwd Rel':<10} {'FwdCorr':<10} {'BwdCorr':<10}"
        header_len = 90
        
        if SHOW_SENSITIVITY_NORMS:
            header += f" {'||δ_pp||':<10} {'||δ_np||':<10} {'||δ_pn||':<10} {'||δ_nn||':<10}"
            header_len += 40
            
        if SHOW_ACTIVATION_NORMS:
            header += f" {'||pos||':<10} {'||neg||':<10}"
            header_len += 20
            
        if SHOW_ORIGINAL_GRAD_NORMS:
            header += f" {'||∇_orig||':<10}"
            header_len += 10

        if SHOW_CORRECTION_NORMS:
            header += f" {'||Δfwd||':<10} {'||Δbwd||':<10}"
            header_len += 20

        print(header)
        print(f"  {'-'*header_len}")

        for lr in result.layer_results:
            # Use display_name for Add modules, otherwise use regular name
            display_name = lr.display_name if lr.display_name and lr.display_name != lr.name else lr.name
            layer_name = display_name[:33] + '..' if len(display_name) > 35 else display_name
            fwd_mark = "" if lr.fwd_pass else "*"
            bwd_mark = "" if lr.bwd_pass else "*"
            # Build row dynamically based on configuration
            row = (f"  {layer_name:<35} {lr.module_type:<15} "
                   f"{lr.fwd_rel_error:<10.2e}{fwd_mark} {lr.bwd_rel_error:<10.2e}{bwd_mark} "
                   f"{lr.fwd_correction_rel_error:<10.2e} {lr.bwd_correction_rel_error:<10.2e}")
            
            if SHOW_SENSITIVITY_NORMS:
                row += (f" {lr.sens_pp_norm:<10.2e} {lr.sens_np_norm:<10.2e} "
                        f"{lr.sens_pn_norm:<10.2e} {lr.sens_nn_norm:<10.2e}")
                        
            if SHOW_ACTIVATION_NORMS:
                row += f" {lr.act_pos_norm:<10.2e} {lr.act_neg_norm:<10.2e}"
                
            if SHOW_ORIGINAL_GRAD_NORMS:
                row += f" {lr.orig_grad_norm:<10.2e}"

            if SHOW_CORRECTION_NORMS:
                row += f" {lr.fwd_correction_norm:<10.2e} {lr.bwd_correction_norm:<10.2e}"

            print(row)

    print()


def _print_summary(results: List[TestResult]) -> None:
    """Print test summary."""
    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)

    print(f"{'Model':<40} {'Fwd Rel':<12} {'Bwd Rel':<12} {'FwdCorr':<12} {'BwdCorr':<12} {'Status'}")
    print("-" * 90)

    for r in results:
        display_name = r.name[:38] + '..' if len(r.name) > 40 else r.name
        if r.error_message:
            print(f"{display_name:<40} {'ERROR':<12} {'':<12} {'':<12} {'':<12} FAIL")
        else:
            status = "PASS" if r.success else "FAIL"
            print(f"{display_name:<40} {r.fwd_rel_error:<12.2e} {r.bwd_rel_error:<12.2e} "
                  f"{r.fwd_correction_rel_error:<12.2e} {r.bwd_correction_rel_error:<12.2e} {status}")

    print("-" * 90)

    passed = sum(1 for r in results if r.success)
    total = len(results)

    if passed == total:
        print(f"\nAll {total} tests: PASS")
    else:
        print(f"\n{passed}/{total} tests passed, {total - passed} FAILED")


# =============================================================================
# Backward Compatibility
# =============================================================================

def test_model_simple(model: nn.Module, x: Tensor, name: str = "model",
                      fwd_tol: float = 1e-5, bwd_tol: float = 0.1) -> Dict:
    """Simple test interface for backward compatibility."""
    result = test_model_functional(
        model, x, name,
        fwd_rel_tol=fwd_tol, bwd_rel_tol=bwd_tol,
        fwd_correction_rel_tol=1e-3, bwd_correction_rel_tol=1e-3,
    )

    return {
        'name': name,
        'success': result.success,
        'forward_rel': result.fwd_rel_error,
        'backward_rel': result.bwd_rel_error,
        'forward_correction_rel': result.fwd_correction_rel_error,
        'backward_correction_rel': result.bwd_correction_rel_error,
        'error': result.error_message if result.error_message else None,
        'layer_results': result.layer_results,
    }
