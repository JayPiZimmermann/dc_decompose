"""
DC Decomposition Testing Utilities.

Provides a unified testing API for DC decomposition validation.
All test files should use these utilities - no testing logic in test files.

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
from dc_decompose.operations.base import (
    init_catted, reconstruct_output, InputMode, split4
)


# =============================================================================
# Default Tolerances
# =============================================================================

DEFAULT_FWD_ABS_TOL = 1e-5
DEFAULT_FWD_REL_TOL = 1e-5
DEFAULT_BWD_ABS_TOL = 1e-4
DEFAULT_BWD_REL_TOL = 0.1


# =============================================================================
# Result Data Classes
# =============================================================================

@dataclass
class LayerResult:
    """Result for a single layer."""
    name: str
    module_type: str

    # Forward errors
    fwd_abs_error: float = 0.0
    fwd_rel_error: float = 0.0

    # Backward errors
    bwd_abs_error: float = 0.0
    bwd_rel_error: float = 0.0

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
        self.layer_types: Dict[str, str] = {}
        self.layer_order: List[str] = []
        self._handles: List = []

    def clear(self):
        self.orig_outputs.clear()
        self.orig_grad_inputs.clear()
        self.dc_outputs.clear()
        self.dc_grad_inputs.clear()
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
                            # Reconstruct from [4*batch] format
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
        return isinstance(module, (
            nn.Linear, nn.Conv1d, nn.Conv2d,
            nn.ReLU, nn.LeakyReLU, nn.GELU,
            nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm,
            nn.MaxPool1d, nn.MaxPool2d,
            nn.AvgPool1d, nn.AvgPool2d,
            nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d,
            nn.Flatten, nn.Dropout,
        ))

    def compute_errors(self, fwd_abs_tol: float, fwd_rel_tol: float,
                       bwd_abs_tol: float, bwd_rel_tol: float,
                       input_orig: Optional[Tensor] = None,
                       input_dc: Optional[Tensor] = None,
                       input_grad_orig: Optional[Tensor] = None,
                       input_grad_dc: Optional[Tensor] = None,
                       output_orig: Optional[Tensor] = None,
                       output_dc: Optional[Tensor] = None) -> List[LayerResult]:
        """Compute layer-wise errors including input/output."""
        results = []

        # Input layer (init_catted forward)
        if input_orig is not None and input_dc is not None:
            lr = LayerResult(name=">>> INPUT", module_type="init_catted")
            # Forward: input should be identical (just reformatted)
            diff = (input_orig - input_dc).abs()
            lr.fwd_abs_error = diff.max().item()
            lr.fwd_rel_error = lr.fwd_abs_error / (input_orig.abs().max().item() + 1e-10)
            lr.fwd_pass = lr.fwd_abs_error < fwd_abs_tol or lr.fwd_rel_error < fwd_rel_tol
            # Backward: gradient at input
            if input_grad_orig is not None and input_grad_dc is not None:
                diff = (input_grad_orig - input_grad_dc).abs()
                lr.bwd_abs_error = diff.max().item()
                lr.bwd_rel_error = lr.bwd_abs_error / (input_grad_orig.abs().max().item() + 1e-10)
                lr.bwd_pass = lr.bwd_abs_error < bwd_abs_tol or lr.bwd_rel_error < bwd_rel_tol
            results.append(lr)

        # Model layers
        for name in self.layer_order:
            layer_type = self.layer_types.get(name, "Unknown")
            lr = LayerResult(name=name, module_type=layer_type)

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

            results.append(lr)

        # Output layer (reconstruct_output)
        if output_orig is not None and output_dc is not None:
            lr = LayerResult(name="<<< OUTPUT", module_type="reconstruct")
            diff = (output_orig - output_dc).abs()
            lr.fwd_abs_error = diff.max().item()
            lr.fwd_rel_error = lr.fwd_abs_error / (output_orig.abs().max().item() + 1e-10)
            lr.fwd_pass = lr.fwd_abs_error < fwd_abs_tol or lr.fwd_rel_error < fwd_rel_tol
            results.append(lr)

        return results


# =============================================================================
# Core Testing Functions
# =============================================================================

def check_pass(abs_err: float, rel_err: float, abs_tol: float, rel_tol: float) -> bool:
    """Pass if EITHER absolute error < abs_tol OR relative error < rel_tol."""
    return abs_err < abs_tol or rel_err < rel_tol


def test_model(
    model: nn.Module,
    x: Tensor,
    name: str = "model",
    fwd_abs_tol: float = DEFAULT_FWD_ABS_TOL,
    fwd_rel_tol: float = DEFAULT_FWD_REL_TOL,
    bwd_abs_tol: float = DEFAULT_BWD_ABS_TOL,
    bwd_rel_tol: float = DEFAULT_BWD_REL_TOL,
) -> TestResult:
    """Test a model's DC decomposition with layer-wise error tracking."""

    result = TestResult(
        name=name,
        fwd_abs_tol=fwd_abs_tol,
        fwd_rel_tol=fwd_rel_tol,
        bwd_abs_tol=bwd_abs_tol,
        bwd_rel_tol=bwd_rel_tol,
    )

    cache = LayerCache()

    try:
        model = copy.deepcopy(model)
        model.eval()

        # =====================================================================
        # Phase 1: Original forward/backward with hooks
        # =====================================================================
        model_orig = copy.deepcopy(model)
        model_orig.eval()
        cache.register_orig_hooks(model_orig)

        x_orig = x.clone().requires_grad_(True)
        orig_out = model_orig(x_orig)
        target_grad = torch.randn_like(orig_out)
        orig_out.backward(target_grad)
        grad_orig = x_orig.grad.clone()

        cache._remove_hooks()

        # =====================================================================
        # Phase 2: DC forward/backward with hooks
        # =====================================================================
        model_dc = copy.deepcopy(model)
        model_dc = prepare_model_for_dc(model_dc)
        cache.register_dc_hooks(model_dc)

        x_dc = x.clone().requires_grad_(True)
        x_cat = init_catted(x_dc, InputMode.CENTER)
        out_cat = model_dc(x_cat)
        dc_out = reconstruct_output(out_cat)
        dc_out.backward(target_grad)
        grad_dc = x_dc.grad

        cache._remove_hooks()
        unpatch_model(model_dc)

        # =====================================================================
        # Compute layer-wise errors (including input/output)
        # =====================================================================
        result.layer_results = cache.compute_errors(
            fwd_abs_tol, fwd_rel_tol, bwd_abs_tol, bwd_rel_tol,
            input_orig=x, input_dc=x,  # Same input
            input_grad_orig=grad_orig, input_grad_dc=grad_dc,
            output_orig=orig_out.detach(), output_dc=dc_out.detach()
        )

        # =====================================================================
        # Compute overall errors (max across layers + input/output)
        # =====================================================================

        # Forward: compare final output
        fwd_diff = (orig_out.detach() - dc_out.detach()).abs()
        result.fwd_abs_error = fwd_diff.max().item()
        result.fwd_rel_error = result.fwd_abs_error / (orig_out.abs().max().item() + 1e-10)

        # Also consider max layer error
        for lr in result.layer_results:
            if lr.fwd_abs_error > result.fwd_abs_error:
                result.fwd_abs_error = lr.fwd_abs_error
            if lr.fwd_rel_error > result.fwd_rel_error:
                result.fwd_rel_error = lr.fwd_rel_error

        result.fwd_pass = check_pass(
            result.fwd_abs_error, result.fwd_rel_error,
            fwd_abs_tol, fwd_rel_tol
        )

        # Backward: compare input gradient
        bwd_diff = (grad_orig - grad_dc).abs()
        result.bwd_abs_error = bwd_diff.max().item()
        result.bwd_rel_error = result.bwd_abs_error / (grad_orig.abs().max().item() + 1e-10)

        # Also consider max layer error
        for lr in result.layer_results:
            if lr.bwd_abs_error > result.bwd_abs_error:
                result.bwd_abs_error = lr.bwd_abs_error
            if lr.bwd_rel_error > result.bwd_rel_error:
                result.bwd_rel_error = lr.bwd_rel_error

        result.bwd_pass = check_pass(
            result.bwd_abs_error, result.bwd_rel_error,
            bwd_abs_tol, bwd_rel_tol
        )

        result.success = result.fwd_pass and result.bwd_pass

    except Exception as e:
        import traceback
        result.error_message = str(e)
        result.success = False

    return result


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
) -> bool:
    """
    Run DC decomposition tests on multiple models.

    Args:
        models: Dict mapping name -> (model_or_factory, input_tensor_or_shape)
        title: Title for test output
        show_layers: Whether to show layer-wise errors
    """
    torch.manual_seed(seed)

    if verbose:
        print("=" * 90)
        print(title)
        print("=" * 90)
        print(f"Tolerances: fwd_abs={fwd_abs_tol:.0e}, fwd_rel={fwd_rel_tol:.0e}, "
              f"bwd_abs={bwd_abs_tol:.0e}, bwd_rel={bwd_rel_tol:.0e}")
        print()

    results: Dict[str, TestResult] = {}

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

        # Run test
        result = test_model(
            model, x, name,
            fwd_abs_tol=fwd_abs_tol,
            fwd_rel_tol=fwd_rel_tol,
            bwd_abs_tol=bwd_abs_tol,
            bwd_rel_tol=bwd_rel_tol,
        )
        results[name] = result

        if verbose:
            _print_result(result, show_layers=show_layers)

    # Print summary
    if verbose:
        _print_summary(results)

    return all(r.success for r in results.values())


def _print_result(result: TestResult, show_layers: bool = True) -> None:
    """Print a single test result with optional layer-wise details."""
    status = "PASS" if result.success else "FAIL"

    if result.error_message:
        print(f"{'='*90}")
        print(f"Model: {result.name} [{status}]")
        print(f"{'='*90}")
        print(f"  ERROR: {result.error_message}")
        print()
        return

    fwd_status = "ok" if result.fwd_pass else "FAIL"
    bwd_status = "ok" if result.bwd_pass else "FAIL"

    print(f"{'='*90}")
    print(f"Model: {result.name} [{status}]")
    print(f"{'='*90}")
    print(f"  Overall Forward:  abs={result.fwd_abs_error:.2e}, rel={result.fwd_rel_error:.2e} [{fwd_status}]")
    print(f"  Overall Backward: abs={result.bwd_abs_error:.2e}, rel={result.bwd_rel_error:.2e} [{bwd_status}]")

    if show_layers and result.layer_results:
        print()
        print(f"  {'Layer':<35} {'Type':<15} {'Fwd Abs':<10} {'Fwd Rel':<10} {'Bwd Abs':<10} {'Bwd Rel':<10}")
        print(f"  {'-'*100}")

        for lr in result.layer_results:
            layer_name = lr.name[:33] + '..' if len(lr.name) > 35 else lr.name
            fwd_mark = "" if lr.fwd_pass else "*"
            bwd_mark = "" if lr.bwd_pass else "*"
            print(f"  {layer_name:<35} {lr.module_type:<15} "
                  f"{lr.fwd_abs_error:<10.2e} {lr.fwd_rel_error:<10.2e} "
                  f"{lr.bwd_abs_error:<10.2e}{bwd_mark} {lr.bwd_rel_error:<10.2e}{bwd_mark}")

    print()


def _print_summary(results: Dict[str, TestResult]) -> None:
    """Print test summary."""
    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)

    print(f"{'Model':<30} {'Fwd Abs':<12} {'Fwd Rel':<12} {'Bwd Abs':<12} {'Bwd Rel':<12} {'Status'}")
    print("-" * 90)

    for name, r in results.items():
        display_name = name[:28] + '..' if len(name) > 30 else name
        if r.error_message:
            print(f"{display_name:<30} {'ERROR':<12} {'':<12} {'':<12} {'':<12} FAIL")
        else:
            status = "PASS" if r.success else "FAIL"
            print(f"{display_name:<30} {r.fwd_abs_error:<12.2e} {r.fwd_rel_error:<12.2e} "
                  f"{r.bwd_abs_error:<12.2e} {r.bwd_rel_error:<12.2e} {status}")

    print("-" * 90)

    passed = sum(1 for r in results.values() if r.success)
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
    result = test_model(
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
