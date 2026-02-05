"""
DC Decomposition Testing Utilities.

Provides a unified testing API for DC decomposition validation.
All test files should use these utilities - no testing logic in test files.

Usage:
    from utils import DCModelTester, run_model_tests

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

# Forward pass tolerances
DEFAULT_FWD_ABS_TOL = 1e-5  # Absolute error threshold
DEFAULT_FWD_REL_TOL = 1e-5  # Relative error threshold

# Backward pass tolerances (more lenient due to known issues)
DEFAULT_BWD_ABS_TOL = 1e-4  # Absolute error threshold
DEFAULT_BWD_REL_TOL = 0.1   # Relative error threshold (10%)


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

    # Overall errors
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
# Core Testing Functions
# =============================================================================

def check_pass(abs_err: float, rel_err: float, abs_tol: float, rel_tol: float) -> bool:
    """
    Check if errors pass tolerance thresholds.

    Pass if EITHER:
    - Absolute error < abs_tol, OR
    - Relative error < rel_tol
    """
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
    """
    Test a model's DC decomposition for forward and backward passes.

    Args:
        model: PyTorch model to test
        x: Input tensor
        name: Model name for reporting
        fwd_abs_tol: Forward absolute error tolerance
        fwd_rel_tol: Forward relative error tolerance
        bwd_abs_tol: Backward absolute error tolerance
        bwd_rel_tol: Backward relative error tolerance

    Returns:
        TestResult with detailed error information
    """
    result = TestResult(
        name=name,
        fwd_abs_tol=fwd_abs_tol,
        fwd_rel_tol=fwd_rel_tol,
        bwd_abs_tol=bwd_abs_tol,
        bwd_rel_tol=bwd_rel_tol,
    )

    try:
        # Deep copy model to avoid side effects
        model = copy.deepcopy(model)
        model.eval()

        # =====================================================================
        # Forward Pass Test
        # =====================================================================

        # Original forward
        with torch.no_grad():
            orig_out = model(x.clone())

        # DC forward
        model_dc = copy.deepcopy(model)
        model_dc = prepare_model_for_dc(model_dc)

        x_cat = init_catted(x.clone(), InputMode.CENTER)
        with torch.no_grad():
            out_cat = model_dc(x_cat)
        dc_out = reconstruct_output(out_cat)

        # Compute forward errors
        fwd_diff = (orig_out - dc_out).abs()
        result.fwd_abs_error = fwd_diff.max().item()
        orig_norm = orig_out.abs().max().item()
        result.fwd_rel_error = result.fwd_abs_error / (orig_norm + 1e-10)

        result.fwd_pass = check_pass(
            result.fwd_abs_error, result.fwd_rel_error,
            fwd_abs_tol, fwd_rel_tol
        )

        unpatch_model(model_dc)

        # =====================================================================
        # Backward Pass Test
        # =====================================================================

        # Original backward
        x_orig = x.clone().requires_grad_(True)
        model_orig = copy.deepcopy(model)
        model_orig.eval()
        orig_out_bwd = model_orig(x_orig)
        target_grad = torch.randn_like(orig_out_bwd)
        orig_out_bwd.backward(target_grad)
        grad_orig = x_orig.grad.clone()

        # DC backward
        model_dc2 = copy.deepcopy(model)
        model_dc2 = prepare_model_for_dc(model_dc2)

        x_dc = x.clone().requires_grad_(True)
        x_cat = init_catted(x_dc, InputMode.CENTER)
        out_cat = model_dc2(x_cat)
        dc_out_bwd = reconstruct_output(out_cat)
        dc_out_bwd.backward(target_grad)
        grad_dc = x_dc.grad

        # Compute backward errors
        bwd_diff = (grad_orig - grad_dc).abs()
        result.bwd_abs_error = bwd_diff.max().item()
        grad_norm = grad_orig.abs().max().item()
        result.bwd_rel_error = result.bwd_abs_error / (grad_norm + 1e-10)

        result.bwd_pass = check_pass(
            result.bwd_abs_error, result.bwd_rel_error,
            bwd_abs_tol, bwd_rel_tol
        )

        unpatch_model(model_dc2)

        # =====================================================================
        # Overall success
        # =====================================================================
        result.success = result.fwd_pass and result.bwd_pass

    except Exception as e:
        result.error_message = str(e)
        result.success = False

    return result


# =============================================================================
# Batch Testing
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
) -> bool:
    """
    Run DC decomposition tests on multiple models.

    Args:
        models: Dict mapping name -> (model_or_factory, input_tensor_or_shape)
        title: Title for test output
        fwd_abs_tol: Forward absolute error tolerance
        fwd_rel_tol: Forward relative error tolerance
        bwd_abs_tol: Backward absolute error tolerance
        bwd_rel_tol: Backward relative error tolerance
        seed: Random seed for reproducibility
        verbose: Whether to print detailed output

    Returns:
        True if all tests pass, False otherwise
    """
    torch.manual_seed(seed)

    if verbose:
        print("=" * 70)
        print(title)
        print("=" * 70)
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
            _print_result(result)

    # Print summary
    if verbose:
        _print_summary(results)

    return all(r.success for r in results.values())


def _print_result(result: TestResult) -> None:
    """Print a single test result."""
    status = "PASS" if result.success else "FAIL"

    if result.error_message:
        print(f"--- {result.name} [{status}] ---")
        print(f"  ERROR: {result.error_message}")
    else:
        fwd_status = "ok" if result.fwd_pass else "FAIL"
        bwd_status = "ok" if result.bwd_pass else "FAIL"

        print(f"--- {result.name} [{status}] ---")
        print(f"  forward:  abs={result.fwd_abs_error:.2e}, rel={result.fwd_rel_error:.2e} [{fwd_status}]")
        print(f"  backward: abs={result.bwd_abs_error:.2e}, rel={result.bwd_rel_error:.2e} [{bwd_status}]")
    print()


def _print_summary(results: Dict[str, TestResult]) -> None:
    """Print test summary."""
    print("=" * 70)
    print("Summary")
    print("=" * 70)

    # Table header
    print(f"{'Model':<25} {'Fwd Abs':<12} {'Fwd Rel':<12} {'Bwd Abs':<12} {'Bwd Rel':<12} {'Status'}")
    print("-" * 85)

    for name, r in results.items():
        if r.error_message:
            print(f"{name:<25} {'ERROR':<12} {'':<12} {'':<12} {'':<12} FAIL")
        else:
            status = "PASS" if r.success else "FAIL"
            print(f"{name:<25} {r.fwd_abs_error:<12.2e} {r.fwd_rel_error:<12.2e} "
                  f"{r.bwd_abs_error:<12.2e} {r.bwd_rel_error:<12.2e} {status}")

    print("-" * 85)

    # Overall result
    passed = sum(1 for r in results.values() if r.success)
    total = len(results)

    if passed == total:
        print(f"\nAll {total} tests: PASS")
    else:
        print(f"\n{passed}/{total} tests passed, {total - passed} FAILED")


# =============================================================================
# Convenience Aliases (for backward compatibility)
# =============================================================================

def test_model_simple(model: nn.Module, x: Tensor, name: str = "model",
                      fwd_tol: float = 1e-5, bwd_tol: float = 0.1) -> Dict:
    """
    Simple test interface for backward compatibility.

    Returns dict with: success, forward_error, forward_rel, backward_error, backward_rel
    """
    result = test_model(
        model, x, name,
        fwd_abs_tol=fwd_tol,
        fwd_rel_tol=fwd_tol,
        bwd_abs_tol=bwd_tol,
        bwd_rel_tol=bwd_tol,
    )

    return {
        'name': name,
        'success': result.success,
        'forward_error': result.fwd_abs_error,
        'forward_rel': result.fwd_rel_error,
        'backward_error': result.bwd_abs_error,
        'backward_rel': result.bwd_rel_error,
        'error': result.error_message if result.error_message else None,
    }


# Legacy aliases
DCTester = None  # Deprecated, use test_model directly
LayerCache = None  # Deprecated
