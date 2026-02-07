"""
Test utilities for DC decomposition.

Usage:
    from utils import run_model_tests, test_model

    MODELS = {
        'ModelName': (model_or_factory, input_tensor_or_shape),
        ...
    }

    if __name__ == "__main__":
        success = run_model_tests(MODELS, title="My Tests")
        sys.exit(0 if success else 1)
"""

from .dc_tester import (
    # Main testing functions
    run_model_tests,
    test_model,
    test_model_functional,
    test_model_context_manager,
    test_model_simple,

    # Result classes
    TestResult,
    LayerResult,

    # Tolerances
    DEFAULT_FWD_REL_TOL,
    DEFAULT_BWD_REL_TOL,
    DEFAULT_FWD_CORRECTION_REL_TOL,
    DEFAULT_BWD_CORRECTION_REL_TOL,

    # Helper
    check_pass_relative_only,
)

__all__ = [
    'run_model_tests',
    'test_model',
    'test_model_functional',
    'test_model_context_manager',
    'test_model_simple',
    'TestResult',
    'LayerResult',
    'DEFAULT_FWD_REL_TOL',
    'DEFAULT_BWD_REL_TOL',
    'DEFAULT_FWD_CORRECTION_REL_TOL',
    'DEFAULT_BWD_CORRECTION_REL_TOL',
    'check_pass_relative_only',
]
