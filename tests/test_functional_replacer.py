"""
Test functional_replacer.py for automatic conversion of torch.relu and + to modules.

The workflow is:
1. Write model with torch.relu() and + operations (standard PyTorch)
2. Call make_dc_compatible(model) to replace functional calls with modules
3. Call patch_model(model) to enable DC decomposition
4. Run DC forward pass
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from dc_decompose.operations.patcher import patch_model, unpatch_model
from dc_decompose.operations.base import init_catted, reconstruct_output, InputMode
from dc_decompose.operations.functional_replacer import make_dc_compatible


class ResBlockFunctional(nn.Module):
    """Standard ResBlock using torch.relu() and + (no special modules)."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)  # Functional call
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity   # Standard addition (residual)
        out = torch.relu(out)  # Functional call
        return out


class SimpleResNetFunctional(nn.Module):
    """Standard ResNet using functional calls."""
    def __init__(self, num_blocks=3):
        super().__init__()
        self.stem_conv = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(64)
        self.blocks = nn.Sequential(*[ResBlockFunctional(64) for _ in range(num_blocks)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.stem_conv(x)
        x = self.stem_bn(x)
        x = torch.relu(x)  # Functional call
        x = self.blocks(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def test_functional_replacer_basic():
    """Test that functional_replacer correctly replaces torch.relu."""
    print("="*70)
    print("Test: Functional Replacer Basic")
    print("="*70)

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 10)

        def forward(self, x):
            return torch.relu(self.linear(x))

    model = SimpleModel()

    # Check modules before replacement
    modules_before = list(model.named_modules())
    print(f"Modules before: {[name for name, _ in modules_before]}")

    # Replace functional calls
    model = make_dc_compatible(model)

    # Check modules after replacement
    modules_after = list(model.named_modules())
    print(f"Modules after: {[name for name, _ in modules_after]}")

    # Should have a ReLU module now
    relu_modules = [name for name, m in model.named_modules() if isinstance(m, nn.ReLU)]
    print(f"ReLU modules found: {relu_modules}")

    # Test forward still works
    x = torch.randn(2, 10)
    y = model(x)
    print(f"Forward output shape: {y.shape}")

    return len(relu_modules) > 0


def test_functional_replacer_addition():
    """Test that functional_replacer correctly replaces + with Add modules."""
    print("\n" + "="*70)
    print("Test: Functional Replacer Addition")
    print("="*70)

    class AddModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 10)

        def forward(self, x):
            return x + self.linear(x)

    model = AddModel()

    # Check modules before replacement
    modules_before = list(model.named_modules())
    print(f"Modules before: {[name for name, _ in modules_before]}")

    # Replace functional calls
    model = make_dc_compatible(model)

    # Check modules after replacement
    modules_after = list(model.named_modules())
    print(f"Modules after: {[name for name, _ in modules_after]}")

    # Check for Add modules
    from dc_decompose.operations.add import Add
    add_modules = [name for name, m in model.named_modules() if isinstance(m, Add)]
    print(f"Add modules found: {add_modules}")

    # Test forward still works
    x = torch.randn(2, 10)
    y = model(x)
    print(f"Forward output shape: {y.shape}")

    return len(add_modules) > 0


def test_resblock_full_workflow():
    """Test full workflow: functional model -> make_dc_compatible -> patch -> DC forward."""
    print("\n" + "="*70)
    print("Test: ResBlock Full Workflow")
    print("="*70)

    torch.manual_seed(42)
    x = torch.randn(1, 64, 8, 8)

    # Create functional model
    block = ResBlockFunctional(64)
    block.eval()

    # Get original output before any transformation
    orig_out = block(x)
    print(f"Original output norm: {orig_out.norm():.4f}")

    # Make DC compatible (replace torch.relu and + with modules)
    block = make_dc_compatible(block)

    # Verify forward still works after replacement
    out_after_replace = block(x)
    replace_error = torch.norm(orig_out - out_after_replace).item()
    print(f"After make_dc_compatible, forward error: {replace_error:.2e}")

    # Check for inserted modules
    from dc_decompose.operations.add import Add
    relu_count = sum(1 for _, m in block.named_modules() if isinstance(m, nn.ReLU))
    add_count = sum(1 for _, m in block.named_modules() if isinstance(m, Add))
    print(f"Inserted modules: {relu_count} ReLU, {add_count} Add")

    # Now patch the model for DC decomposition
    patch_model(block)

    # DC forward
    x_cat = init_catted(x, InputMode.CENTER)
    out_cat = block(x_cat)
    dc_out = reconstruct_output(out_cat)

    # Unpatch
    unpatch_model(block)

    dc_error = torch.norm(orig_out - dc_out).item()
    rel_error = dc_error / (orig_out.norm().item() + 1e-10)
    print(f"DC reconstruction error: {dc_error:.2e}, relative: {rel_error:.2e}")

    return dc_error


def test_resnet_full_workflow():
    """Test full workflow on a multi-block ResNet."""
    print("\n" + "="*70)
    print("Test: SimpleResNet Full Workflow (3 blocks)")
    print("="*70)

    torch.manual_seed(42)
    x = torch.randn(1, 3, 32, 32)

    # Create functional model
    model = SimpleResNetFunctional(num_blocks=3)
    model.eval()

    # Get original output
    orig_out = model(x)
    print(f"Original output: {orig_out[:, :5]}")

    # Make DC compatible
    model = make_dc_compatible(model)

    # Verify forward still works
    out_after = model(x)
    replace_error = torch.norm(orig_out - out_after).item()
    print(f"After make_dc_compatible, error: {replace_error:.2e}")

    # Count inserted modules
    from dc_decompose.operations.add import Add
    relu_count = sum(1 for _, m in model.named_modules() if isinstance(m, nn.ReLU))
    add_count = sum(1 for _, m in model.named_modules() if isinstance(m, Add))
    print(f"Total modules: {relu_count} ReLU, {add_count} Add")

    # Patch and run DC
    patch_model(model)

    x_cat = init_catted(x, InputMode.CENTER)
    out_cat = model(x_cat)
    dc_out = reconstruct_output(out_cat)

    unpatch_model(model)

    dc_error = torch.norm(orig_out - dc_out).item()
    rel_error = dc_error / (orig_out.norm().item() + 1e-10)
    print(f"DC reconstruction error: {dc_error:.2e}, relative: {rel_error:.2e}")

    return dc_error


def run_all_tests():
    """Run all functional replacer tests."""
    print("="*70)
    print("DC Decomposition - Functional Replacer Tests")
    print("="*70)

    results = {}

    # Basic tests
    results['relu_replaced'] = test_functional_replacer_basic()
    results['add_replaced'] = test_functional_replacer_addition()

    # Full workflow tests
    results['resblock_error'] = test_resblock_full_workflow()
    results['resnet_error'] = test_resnet_full_workflow()

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"ReLU replacement: {'PASS' if results['relu_replaced'] else 'FAIL'}")
    print(f"Add replacement: {'PASS' if results['add_replaced'] else 'FAIL'}")
    print(f"ResBlock DC error: {results['resblock_error']:.2e}")
    print(f"ResNet DC error: {results['resnet_error']:.2e}")

    if results['resnet_error'] < 1e-3:
        print("\n✓ Functional replacer workflow: PASS")
    else:
        print(f"\n✗ Functional replacer workflow: FAIL (error too high)")


if __name__ == "__main__":
    run_all_tests()
