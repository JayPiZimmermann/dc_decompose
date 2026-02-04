"""
Test script for DC decomposition operations.

Tests various model architectures to validate that:
1. pos - neg reconstructs original activations
2. All supported layers work correctly
3. Different model complexities are handled

Run with: python -m dc_decompose.operations.test_operations
"""

import torch
import torch.nn as nn
from typing import Tuple

from .testing import DCTester, test_model
from .base import InputMode


# =============================================================================
# Test Models
# =============================================================================

class SimpleLinear(nn.Module):
    """Single linear layer."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)


class LinearReLULinear(nn.Module):
    """Linear -> ReLU -> Linear."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class DeepMLP(nn.Module):
    """Deep MLP with multiple layers."""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
        )

    def forward(self, x):
        return self.layers(x)


class SimpleConvNet(nn.Module):
    """Simple CNN with Conv2d, ReLU, and pooling."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ConvBNReLU(nn.Module):
    """Conv -> BatchNorm -> ReLU."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 5)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ResidualBlock(nn.Module):
    """Residual block with skip connection."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = out + identity  # Skip connection
        out = self.relu2(out)
        return out


class SimpleResNet(nn.Module):
    """Simple ResNet-like model with skip connections."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.res1 = ResidualBlock(16)
        self.res2 = ResidualBlock(16)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Conv1dModel(nn.Module):
    """Model with Conv1d layers."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(8, 16, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.fc = nn.Linear(16 * 8, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# =============================================================================
# Test Runner
# =============================================================================

def run_all_tests(verbose: bool = True) -> Tuple[int, int, str]:
    """
    Run all model tests.

    Returns:
        (passed_count, total_count, full_report)
    """
    test_cases = [
        ("SimpleLinear", SimpleLinear(), (4, 10)),
        ("LinearReLULinear", LinearReLULinear(), (4, 10)),
        ("DeepMLP", DeepMLP(), (4, 20)),
        ("SimpleConvNet", SimpleConvNet(), (2, 3, 16, 16)),
        ("ConvBNReLU", ConvBNReLU(), (2, 3, 8, 8)),
        ("SimpleResNet", SimpleResNet(), (2, 3, 16, 16)),
        # ("Conv1dModel", Conv1dModel(), (2, 8, 16)),  # Conv1d not yet implemented
    ]

    reports = []
    passed = 0
    total = len(test_cases)

    reports.append("=" * 70)
    reports.append("DC DECOMPOSITION TEST SUITE")
    reports.append("=" * 70)
    reports.append("")

    for name, model, input_shape in test_cases:
        reports.append(f"\n{'='*70}")
        reports.append(f"Testing: {name}")
        reports.append(f"Input shape: {input_shape}")
        reports.append("=" * 70)

        try:
            test_passed, report, tester = test_model(
                model,
                input_shape,
                input_mode=InputMode.CENTER,
                relu_mode='max',
            )

            if test_passed:
                passed += 1
                status = "PASS ✓"
            else:
                status = "FAIL ✗"

            reports.append(f"Status: {status}")
            reports.append(f"Max Error: {tester.max_activation_error():.2e}")

            if verbose or not test_passed:
                reports.append("")
                reports.append(report)

        except Exception as e:
            reports.append(f"Status: ERROR ✗")
            reports.append(f"Exception: {type(e).__name__}: {e}")
            import traceback
            reports.append(traceback.format_exc())

    # Summary
    reports.append("\n" + "=" * 70)
    reports.append("SUMMARY")
    reports.append("=" * 70)
    reports.append(f"Passed: {passed}/{total}")
    reports.append(f"Status: {'ALL TESTS PASSED' if passed == total else 'SOME TESTS FAILED'}")
    reports.append("=" * 70)

    return passed, total, "\n".join(reports)


def test_relu_modes() -> str:
    """Test different ReLU modes."""
    reports = []
    reports.append("\n" + "=" * 70)
    reports.append("RELU MODE COMPARISON")
    reports.append("=" * 70)

    model = LinearReLULinear()
    x = torch.randn(4, 10)

    for mode in ['max', 'min', 'half']:
        reports.append(f"\nMode: {mode}")

        passed, report, tester = test_model(
            model,
            (4, 10),
            relu_mode=mode,
        )

        reports.append(f"  Max Error: {tester.max_activation_error():.2e}")
        reports.append(f"  Status: {'PASS' if passed else 'FAIL'}")

    return "\n".join(reports)


if __name__ == "__main__":
    # Run main tests
    passed, total, report = run_all_tests(verbose=True)
    print(report)

    # Test ReLU modes
    relu_report = test_relu_modes()
    print(relu_report)

    # Exit code
    exit(0 if passed == total else 1)
