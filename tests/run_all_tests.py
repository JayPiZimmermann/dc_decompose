#!/usr/bin/env python
"""
Run all DC decomposition tests.

Usage:
    python tests/run_all_tests.py
"""

import sys
import os
import subprocess

TEST_FILES = [
    'test_basic_layers.py',
    'test_models_mlp.py',
    'test_models_cnn.py',
    'test_models_resnet.py',
    'test_complex_models.py',
    'test_functional_replacer.py',
    'test_patcher_resnet.py',
    'test_residual_connections.py',
    'test_recenter_solution.py',
    'test_backward_pass.py',
]


def main():
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    all_pass = True

    print("=" * 70)
    print("DC Decomposition - Running All Tests")
    print("=" * 70)
    print()

    for test_file in TEST_FILES:
        test_path = os.path.join(tests_dir, test_file)
        if not os.path.exists(test_path):
            print(f"SKIP: {test_file} (not found)")
            continue

        print(f"Running {test_file}...")
        result = subprocess.run(
            [sys.executable, test_path],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            all_pass = False
            print(f"  FAILED (exit code {result.returncode})")
        else:
            print(f"  PASSED")

    print()
    print("=" * 70)
    if all_pass:
        print("All tests PASSED")
    else:
        print("Some tests FAILED")
    print("=" * 70)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
