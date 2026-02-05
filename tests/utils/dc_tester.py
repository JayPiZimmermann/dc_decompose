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

from dc_decompose.patcher import prepare_model_for_dc, unpatch_model
from dc_decompose.functional_replacer import replace_functional_with_modules
from dc_decompose.operations.base import (
    init_catted, reconstruct_output, InputMode, split4,
    Sensitivities, extract_sensitivities, dc_forward,
)


# =============================================================================
# Default Tolerances
# =============================================================================

DEFAULT_FWD_ABS_TOL = 1e-5
DEFAULT_FWD_REL_TOL = 1e-5
DEFAULT_BWD_ABS_TOL = 1e-4
DEFAULT_BWD_REL_TOL = 1e-2  # 1% relative error maximum (was 10%)


# =============================================================================
# Column Display Configuration
# =============================================================================

SHOW_SENSITIVITY_NORMS = True      # Show ||δ_pp||, ||δ_np||, ||δ_pn||, ||δ_nn|| columns
SHOW_ACTIVATION_NORMS = True       # Show ||pos||, ||neg|| columns
SHOW_ORIGINAL_GRAD_NORMS = True    # Show ||∇_orig|| column  
INIT_RANDOM_BIASES = True          # Initialize biases randomly (not zeros)
INIT_LARGER_WEIGHTS = True         # Initialize weights with larger values for more significant gradients


# =============================================================================
# Result Data Classes
# =============================================================================

@dataclass
class LayerResult:
    """Result for a single layer."""
    name: str
    module_type: str
    display_name: str = ""  # Custom display name with input info

    # Forward errors
    fwd_abs_error: float = 0.0
    fwd_rel_error: float = 0.0

    # Backward errors
    bwd_abs_error: float = 0.0
    bwd_rel_error: float = 0.0

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

    # Pass/fail
    fwd_pass: bool = True
    bwd_pass: bool = True


@dataclass
class TestResult:
    """Result for a single model test."""
    name: str
    success: bool = False
    error_message: str = ""

    # Overall errors (max across layers)
    fwd_abs_error: float = float('inf')
    fwd_rel_error: float = float('inf')
    bwd_abs_error: float = float('inf')
    bwd_rel_error: float = float('inf')

    # Pass/fail for each phase
    fwd_pass: bool = False
    bwd_pass: bool = False

    # Layer-wise results
    layer_results: List[LayerResult] = field(default_factory=list)

    # API used for this test
    api_used: str = "functional"

    # Tolerances used
    fwd_abs_tol: float = DEFAULT_FWD_ABS_TOL
    fwd_rel_tol: float = DEFAULT_FWD_REL_TOL
    bwd_abs_tol: float = DEFAULT_BWD_ABS_TOL
    bwd_rel_tol: float = DEFAULT_BWD_REL_TOL


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
        self.layer_order: List[str] = []
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
        self._remove_hooks()

    def _remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def register_orig_hooks(self, model: nn.Module):
        """Register hooks to capture original activations/gradients."""
        self._remove_hooks()

        for name, module in model.named_modules():
            if self._is_trackable(module) and name:
                self.layer_types[name] = module.__class__.__name__
                if name not in self.layer_order:
                    self.layer_order.append(name)

                def make_fwd_hook(n):
                    def hook(m, inp, out):
                        self.orig_outputs[n] = out.detach().clone()
                    return hook

                def make_bwd_hook(n):
                    def hook(m, grad_in, grad_out):
                        if grad_in[0] is not None:
                            self.orig_grad_inputs[n] = grad_in[0].detach().clone()
                    return hook

                self._handles.append(module.register_forward_hook(make_fwd_hook(name)))
                self._handles.append(module.register_full_backward_hook(make_bwd_hook(name)))

    def register_dc_hooks(self, model: nn.Module):
        """Register hooks to capture DC activations/gradients."""
        self._remove_hooks()

        for name, module in model.named_modules():
            if self._is_trackable(module) and name:
                self.layer_types[name] = module.__class__.__name__
                if name not in self.layer_order:
                    self.layer_order.append(name)

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

    def compute_errors(self, fwd_abs_tol: float, fwd_rel_tol: float,
                       bwd_abs_tol: float, bwd_rel_tol: float,
                       output_orig: Optional[Tensor] = None,
                       output_dc: Optional[Tensor] = None,
                       sens_grad: Optional[Tensor] = None,
                       orig_grad: Optional[Tensor] = None,
                       input_grad_raw: Optional[Tensor] = None) -> List[LayerResult]:
        """Compute layer-wise errors."""
        results = []

        # Model layers
        for name in self.layer_order:
            layer_type = self.layer_types.get(name, "Unknown")
            lr = LayerResult(name=name, module_type=layer_type)
            lr.display_name = _get_add_display_name(name, layer_type)

            # Forward error
            if name in self.orig_outputs and name in self.dc_outputs:
                orig = self.orig_outputs[name]
                dc = self.dc_outputs[name]
                if orig.shape == dc.shape:
                    diff = (orig - dc).abs()
                    lr.fwd_abs_error = diff.max().item()
                    lr.fwd_rel_error = lr.fwd_abs_error / (orig.abs().max().item() + 1e-10)
                    lr.fwd_pass = lr.fwd_abs_error < fwd_abs_tol or lr.fwd_rel_error < fwd_rel_tol

            # Backward error
            if name in self.orig_grad_inputs and name in self.dc_grad_inputs:
                orig = self.orig_grad_inputs[name]
                dc = self.dc_grad_inputs[name]
                if orig.shape == dc.shape:
                    diff = (orig - dc).abs()
                    lr.bwd_abs_error = diff.max().item()
                    lr.bwd_rel_error = lr.bwd_abs_error / (orig.abs().max().item() + 1e-10)
                    lr.bwd_pass = lr.bwd_abs_error < bwd_abs_tol or lr.bwd_rel_error < bwd_rel_tol
                
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

            results.append(lr)

        # Output layer (reconstruct_output)
        if output_orig is not None and output_dc is not None:
            lr = LayerResult(name="<<< OUTPUT", module_type="reconstruct")
            diff = (output_orig - output_dc).abs()
            lr.fwd_abs_error = diff.max().item()
            lr.fwd_rel_error = lr.fwd_abs_error / (output_orig.abs().max().item() + 1e-10)
            lr.fwd_pass = lr.fwd_abs_error < fwd_abs_tol or lr.fwd_rel_error < fwd_rel_tol
            # Backward at output is just the initialization, no error there
            lr.bwd_abs_error = 0.0
            lr.bwd_rel_error = 0.0
            lr.bwd_pass = True
            results.append(lr)

        # Input layer (sensitivities -> reconstructed gradient)
        if sens_grad is not None and orig_grad is not None:
            lr = LayerResult(name=">>> INPUT", module_type="sensitivities")
            lr.fwd_abs_error = 0.0
            lr.fwd_rel_error = 0.0
            lr.fwd_pass = True
            diff = (sens_grad - orig_grad).abs()
            lr.bwd_abs_error = diff.max().item()
            lr.bwd_rel_error = lr.bwd_abs_error / (orig_grad.abs().max().item() + 1e-10)
            lr.bwd_pass = lr.bwd_abs_error < bwd_abs_tol or lr.bwd_rel_error < bwd_rel_tol
            
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
            
            # Insert at beginning
            results.insert(0, lr)

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
    """Initialize weights and biases with larger values for more significant gradients."""
    
    for module in model.modules():
        with torch.no_grad():
            # Initialize weights with larger values if enabled
            if INIT_LARGER_WEIGHTS and hasattr(module, 'weight') and module.weight is not None:
                if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                    # Use larger standard deviation for more significant gradients
                    # Standard PyTorch uses sqrt(1/fan_in), we use 3x that
                    fan_in = module.weight.shape[1] if len(module.weight.shape) > 1 else module.weight.shape[0]
                    std = 3.0 / torch.sqrt(torch.tensor(fan_in, dtype=torch.float32))
                    module.weight.normal_(0.0, std)
                elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    # BatchNorm weight (gamma): start closer to 1 but with variation
                    module.weight.uniform_(0.8, 1.2)
                    
            # Initialize bias with random values if enabled  
            if INIT_RANDOM_BIASES and hasattr(module, 'bias') and module.bias is not None:
                if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                    # Positive bias range for more activations
                    module.bias.uniform_(0.0, 0.5)
                elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    # BatchNorm bias (beta): small random values around 0
                    module.bias.uniform_(-0.1, 0.1)


def check_pass(abs_err: float, rel_err: float, abs_tol: float, rel_tol: float) -> bool:
    """Pass if BOTH absolute error < abs_tol AND relative error < rel_tol."""
    return abs_err < abs_tol and rel_err < rel_tol


def test_model_functional(
    model: nn.Module,
    x: Tensor,
    name: str = "model",
    fwd_abs_tol: float = DEFAULT_FWD_ABS_TOL,
    fwd_rel_tol: float = DEFAULT_FWD_REL_TOL,
    bwd_abs_tol: float = DEFAULT_BWD_ABS_TOL,
    bwd_rel_tol: float = DEFAULT_BWD_REL_TOL,
) -> TestResult:
    """Test using the functional API (init_catted + reconstruct_output)."""

    result = TestResult(
        name=name,
        api_used="functional",
        fwd_abs_tol=fwd_abs_tol,
        fwd_rel_tol=fwd_rel_tol,
        bwd_abs_tol=bwd_abs_tol,
        bwd_rel_tol=bwd_rel_tol,
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
        # Phase 2: Original forward/backward with hooks (BEFORE DC patching)
        # =====================================================================
        cache.register_orig_hooks(model)

        x_orig = x.clone().requires_grad_(True)
        orig_out = model(x_orig)
        target_grad = torch.randn_like(orig_out)
        orig_out.backward(target_grad)
        grad_orig = x_orig.grad.clone()

        cache._remove_hooks()

        # =====================================================================
        # Phase 3: DC forward/backward using FUNCTIONAL API
        # =====================================================================
        model = prepare_model_for_dc(model)
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

        # =====================================================================
        # Compute layer-wise errors
        # =====================================================================
        result.layer_results = cache.compute_errors(
            fwd_abs_tol, fwd_rel_tol, bwd_abs_tol, bwd_rel_tol,
            output_orig=orig_out.detach(), output_dc=dc_out.detach(),
            sens_grad=grad_dc, orig_grad=grad_orig,
            input_grad_raw=x_cat.grad
        )

        # =====================================================================
        # Compute overall errors
        # =====================================================================

        # Forward: compare final output
        fwd_diff = (orig_out.detach() - dc_out.detach()).abs()
        result.fwd_abs_error = fwd_diff.max().item()
        result.fwd_rel_error = result.fwd_abs_error / (orig_out.abs().max().item() + 1e-10)

        for lr in result.layer_results:
            if lr.fwd_abs_error > result.fwd_abs_error:
                result.fwd_abs_error = lr.fwd_abs_error
            if lr.fwd_rel_error > result.fwd_rel_error:
                result.fwd_rel_error = lr.fwd_rel_error

        result.fwd_pass = check_pass(
            result.fwd_abs_error, result.fwd_rel_error,
            fwd_abs_tol, fwd_rel_tol
        )

        # Backward: compare reconstructed gradient
        bwd_diff = (grad_orig - grad_dc).abs()
        result.bwd_abs_error = bwd_diff.max().item()
        result.bwd_rel_error = result.bwd_abs_error / (grad_orig.abs().max().item() + 1e-10)

        for lr in result.layer_results:
            if lr.bwd_abs_error > result.bwd_abs_error:
                result.bwd_abs_error = lr.bwd_abs_error
            if lr.bwd_rel_error > result.bwd_rel_error:
                result.bwd_rel_error = lr.bwd_rel_error

        result.bwd_pass = check_pass(
            result.bwd_abs_error, result.bwd_rel_error,
            bwd_abs_tol, bwd_rel_tol
        )

        # Sum check: |delta_pp - delta_pn + delta_np - delta_nn| < tol

        result.success = result.fwd_pass and result.bwd_pass

    except Exception as e:
        import traceback
        result.error_message = f"{e}\n{traceback.format_exc()}"
        result.success = False

    return result


def test_model_context_manager(
    model: nn.Module,
    x: Tensor,
    name: str = "model",
    fwd_abs_tol: float = DEFAULT_FWD_ABS_TOL,
    fwd_rel_tol: float = DEFAULT_FWD_REL_TOL,
    bwd_abs_tol: float = DEFAULT_BWD_ABS_TOL,
    bwd_rel_tol: float = DEFAULT_BWD_REL_TOL,
) -> TestResult:
    """Test using the context manager API (dc_forward)."""

    result = TestResult(
        name=name,
        api_used="context_manager",
        fwd_abs_tol=fwd_abs_tol,
        fwd_rel_tol=fwd_rel_tol,
        bwd_abs_tol=bwd_abs_tol,
        bwd_rel_tol=bwd_rel_tol,
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
        # Phase 2: Original forward/backward with hooks (BEFORE DC patching)
        # =====================================================================
        cache.register_orig_hooks(model)

        x_orig = x.clone().requires_grad_(True)
        orig_out = model(x_orig)
        target_grad = torch.randn_like(orig_out)
        orig_out.backward(target_grad)
        grad_orig = x_orig.grad.clone()

        cache._remove_hooks()

        # =====================================================================
        # Phase 3: DC forward/backward using CONTEXT MANAGER API
        # =====================================================================
        model = prepare_model_for_dc(model)
        cache.register_dc_hooks(model)

        with dc_forward(model, x, beta=1.0) as dc:
            dc_out = dc.output
            dc_out.backward(target_grad)

        # Get reconstructed gradient from sensitivities
        grad_dc = dc.reconstruct_gradient()
        input_grad_raw = dc.input_4.grad

        cache._remove_hooks()
        unpatch_model(model)

        # =====================================================================
        # Compute layer-wise errors
        # =====================================================================
        result.layer_results = cache.compute_errors(
            fwd_abs_tol, fwd_rel_tol, bwd_abs_tol, bwd_rel_tol,
            output_orig=orig_out.detach(), output_dc=dc_out.detach(),
            sens_grad=grad_dc, orig_grad=grad_orig,
            input_grad_raw=input_grad_raw
        )

        # =====================================================================
        # Compute overall errors
        # =====================================================================

        # Forward: compare final output
        fwd_diff = (orig_out.detach() - dc_out.detach()).abs()
        result.fwd_abs_error = fwd_diff.max().item()
        result.fwd_rel_error = result.fwd_abs_error / (orig_out.abs().max().item() + 1e-10)

        for lr in result.layer_results:
            if lr.fwd_abs_error > result.fwd_abs_error:
                result.fwd_abs_error = lr.fwd_abs_error
            if lr.fwd_rel_error > result.fwd_rel_error:
                result.fwd_rel_error = lr.fwd_rel_error

        result.fwd_pass = check_pass(
            result.fwd_abs_error, result.fwd_rel_error,
            fwd_abs_tol, fwd_rel_tol
        )

        # Backward: compare reconstructed gradient
        bwd_diff = (grad_orig - grad_dc).abs()
        result.bwd_abs_error = bwd_diff.max().item()
        result.bwd_rel_error = result.bwd_abs_error / (grad_orig.abs().max().item() + 1e-10)

        for lr in result.layer_results:
            if lr.bwd_abs_error > result.bwd_abs_error:
                result.bwd_abs_error = lr.bwd_abs_error
            if lr.bwd_rel_error > result.bwd_rel_error:
                result.bwd_rel_error = lr.bwd_rel_error

        result.bwd_pass = check_pass(
            result.bwd_abs_error, result.bwd_rel_error,
            bwd_abs_tol, bwd_rel_tol
        )

        # Sum check: |delta_pp - delta_pn + delta_np - delta_nn| < tol

        result.success = result.fwd_pass and result.bwd_pass

    except Exception as e:
        import traceback
        result.error_message = f"{e}\n{traceback.format_exc()}"
        result.success = False

    return result


def test_model(
    model: nn.Module,
    x: Tensor,
    name: str = "model",
    fwd_abs_tol: float = DEFAULT_FWD_ABS_TOL,
    fwd_rel_tol: float = DEFAULT_FWD_REL_TOL,
    bwd_abs_tol: float = DEFAULT_BWD_ABS_TOL,
    bwd_rel_tol: float = DEFAULT_BWD_REL_TOL,
    api: str = "both",
) -> Union[TestResult, Tuple[TestResult, TestResult]]:
    """
    Test a model's DC decomposition.

    Args:
        api: "functional", "context_manager", or "both" (default)
    """
    if api == "functional":
        return test_model_functional(model, x, name, fwd_abs_tol, fwd_rel_tol, bwd_abs_tol, bwd_rel_tol)
    elif api == "context_manager":
        return test_model_context_manager(model, x, name, fwd_abs_tol, fwd_rel_tol, bwd_abs_tol, bwd_rel_tol)
    else:  # both
        r1 = test_model_functional(model, x, f"{name} [functional]", fwd_abs_tol, fwd_rel_tol, bwd_abs_tol, bwd_rel_tol)
        r2 = test_model_context_manager(model, x, f"{name} [context_mgr]", fwd_abs_tol, fwd_rel_tol, bwd_abs_tol, bwd_rel_tol)
        return r1, r2


# =============================================================================
# Batch Testing with Layer-wise Output
# =============================================================================

ModelSpec = Union[nn.Module, Callable[[], nn.Module]]
InputSpec = Union[Tensor, Tuple[int, ...]]


def run_model_tests(
    models: Dict[str, Tuple[ModelSpec, InputSpec]],
    title: str = "DC Decomposition Tests",
    fwd_abs_tol: float = DEFAULT_FWD_ABS_TOL,
    fwd_rel_tol: float = DEFAULT_FWD_REL_TOL,
    bwd_abs_tol: float = DEFAULT_BWD_ABS_TOL,
    bwd_rel_tol: float = DEFAULT_BWD_REL_TOL,
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
        print(f"Tolerances: fwd_abs={fwd_abs_tol:.0e}, fwd_rel={fwd_rel_tol:.0e}, "
              f"bwd_abs={bwd_abs_tol:.0e}, bwd_rel={bwd_rel_tol:.0e}")
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
                fwd_abs_tol=fwd_abs_tol,
                fwd_rel_tol=fwd_rel_tol,
                bwd_abs_tol=bwd_abs_tol,
                bwd_rel_tol=bwd_rel_tol,
                api="both",
            )
            results.extend([r1, r2])
            if verbose:
                _print_result(r1, show_layers=show_layers)
                _print_result(r2, show_layers=show_layers)
        else:
            result = test_model(
                model, x, name,
                fwd_abs_tol=fwd_abs_tol,
                fwd_rel_tol=fwd_rel_tol,
                bwd_abs_tol=bwd_abs_tol,
                bwd_rel_tol=bwd_rel_tol,
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

    print(f"{'='*100}")
    print(f"Model: {result.name} [{status}]")
    print(f"{'='*100}")
    print(f"  Overall Forward:  abs={result.fwd_abs_error:.2e}, rel={result.fwd_rel_error:.2e} [{fwd_status}]")
    print(f"  Overall Backward: abs={result.bwd_abs_error:.2e}, rel={result.bwd_rel_error:.2e} [{bwd_status}]")

    if show_layers and result.layer_results:
        print()
        # Build header dynamically based on configuration
        header = f"  {'Layer':<35} {'Type':<15} {'Fwd Abs':<10} {'Fwd Rel':<10} {'Bwd Abs':<10} {'Bwd Rel':<10}"
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
                   f"{lr.fwd_abs_error:<10.2e}{fwd_mark} {lr.fwd_rel_error:<10.2e} "
                   f"{lr.bwd_abs_error:<10.2e}{bwd_mark} {lr.bwd_rel_error:<10.2e}")
            
            if SHOW_SENSITIVITY_NORMS:
                row += (f" {lr.sens_pp_norm:<10.2e} {lr.sens_np_norm:<10.2e} "
                        f"{lr.sens_pn_norm:<10.2e} {lr.sens_nn_norm:<10.2e}")
                        
            if SHOW_ACTIVATION_NORMS:
                row += f" {lr.act_pos_norm:<10.2e} {lr.act_neg_norm:<10.2e}"
                
            if SHOW_ORIGINAL_GRAD_NORMS:
                row += f" {lr.orig_grad_norm:<10.2e}"
                
            print(row)

    print()


def _print_summary(results: List[TestResult]) -> None:
    """Print test summary."""
    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)

    print(f"{'Model':<40} {'Fwd Abs':<12} {'Fwd Rel':<12} {'Bwd Abs':<12} {'Bwd Rel':<12} {'Status'}")
    print("-" * 90)

    for r in results:
        display_name = r.name[:38] + '..' if len(r.name) > 40 else r.name
        if r.error_message:
            print(f"{display_name:<40} {'ERROR':<12} {'':<12} {'':<12} {'':<12} FAIL")
        else:
            status = "PASS" if r.success else "FAIL"
            print(f"{display_name:<40} {r.fwd_abs_error:<12.2e} {r.fwd_rel_error:<12.2e} "
                  f"{r.bwd_abs_error:<12.2e} {r.bwd_rel_error:<12.2e} {status}")

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
        fwd_abs_tol=fwd_tol, fwd_rel_tol=fwd_tol,
        bwd_abs_tol=bwd_tol, bwd_rel_tol=bwd_tol,
    )

    return {
        'name': name,
        'success': result.success,
        'forward_error': result.fwd_abs_error,
        'forward_rel': result.fwd_rel_error,
        'backward_error': result.bwd_abs_error,
        'backward_rel': result.bwd_rel_error,
        'error': result.error_message if result.error_message else None,
        'layer_results': result.layer_results,
    }
