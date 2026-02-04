"""
Test patcher-based DC decomposition on ResNet models.

The workflow is:
1. Write standard PyTorch models with torch.relu() and + operations
2. Call make_dc_compatible(model) to replace functional calls with modules
3. Call patch_model(model) to enable DC decomposition
4. Run DC forward pass with init_catted and reconstruct_output
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from dc_decompose.operations.patcher import patch_model, unpatch_model, mark_output_layer
from dc_decompose.operations.base import init_catted, reconstruct_output, InputMode
from dc_decompose.operations.functional_replacer import make_dc_compatible


class ResBlock(nn.Module):
    """Standard ResBlock using torch.relu() and + (standard PyTorch)."""
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
        out = torch.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity  # Residual connection
        out = torch.relu(out)
        return out


class SimpleResNet(nn.Module):
    """Simple ResNet using standard PyTorch operations."""
    def __init__(self, num_blocks=3):
        super().__init__()
        self.stem_conv = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(64)
        self.blocks = nn.Sequential(*[ResBlock(64) for _ in range(num_blocks)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.stem_conv(x)
        x = self.stem_bn(x)
        x = torch.relu(x)
        x = self.blocks(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def test_single_layer(layer_fn, input_shape, name):
    """Test a single layer type."""
    layer = layer_fn()
    layer.eval()
    x = torch.randn(*input_shape)

    # Original output
    orig_out = layer(x)

    # Patch and run DC
    patch_model(layer)
    x_cat = init_catted(x, InputMode.CENTER)
    out_cat = layer(x_cat)
    dc_out = reconstruct_output(out_cat)

    # Unpatch
    unpatch_model(layer)

    error = torch.norm(orig_out - dc_out).item()
    rel_error = error / (torch.norm(orig_out).item() + 1e-10)

    print(f"{name}: abs_error={error:.2e}, rel_error={rel_error:.2e}")
    return error, rel_error


def test_residual_addition():
    """Test that residual addition works correctly with DC format."""
    print("\n" + "="*60)
    print("Residual Addition Test")
    print("="*60)

    batch, channels, h, w = 2, 8, 4, 4
    x = torch.randn(batch, channels, h, w)
    y = torch.randn(batch, channels, h, w)

    # Original addition
    orig = x + y

    # DC format
    x_cat = init_catted(x, InputMode.CENTER)
    y_cat = init_catted(y, InputMode.CENTER)
    sum_cat = x_cat + y_cat
    dc_sum = reconstruct_output(sum_cat)

    error = torch.norm(orig - dc_sum).item()
    print(f"Residual addition error: {error:.2e}")

    return error


def test_individual_components():
    """Test individual components."""
    print("\n" + "="*60)
    print("Individual Component Tests")
    print("="*60)

    results = {}

    results['Conv2d'] = test_single_layer(
        lambda: nn.Conv2d(16, 16, 3, padding=1, bias=False),
        (2, 16, 8, 8), "Conv2d"
    )

    results['BatchNorm2d'] = test_single_layer(
        lambda: nn.BatchNorm2d(16).eval(),
        (2, 16, 8, 8), "BatchNorm2d"
    )

    results['ReLU'] = test_single_layer(
        lambda: nn.ReLU(),
        (2, 16, 8, 8), "ReLU"
    )

    def seq_fn():
        seq = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        seq.eval()
        return seq

    results['Conv+BN+ReLU'] = test_single_layer(
        seq_fn, (2, 16, 8, 8), "Conv+BN+ReLU"
    )

    return results


def test_resblock_with_functional_replacer():
    """Test ResBlock using functional_replacer to handle torch.relu and +."""
    print("\n" + "="*60)
    print("ResBlock with Functional Replacer")
    print("="*60)

    torch.manual_seed(42)
    x = torch.randn(1, 64, 8, 8)

    # Create standard PyTorch model
    block = ResBlock(64)
    block.eval()

    # Get original output
    orig_out = block(x)

    # Make DC compatible (replaces torch.relu and + with patchable modules)
    block = make_dc_compatible(block)

    # Verify forward still works
    out_after = block(x)
    replace_error = torch.norm(orig_out - out_after).item()
    print(f"After make_dc_compatible, error: {replace_error:.2e}")

    # Patch for DC decomposition
    patch_model(block)

    # DC forward
    x_cat = init_catted(x, InputMode.CENTER)
    out_cat = block(x_cat)
    dc_out = reconstruct_output(out_cat)

    unpatch_model(block)

    dc_error = torch.norm(orig_out - dc_out).item()
    rel_error = dc_error / (orig_out.norm().item() + 1e-10)
    print(f"DC reconstruction error: {dc_error:.2e}, relative: {rel_error:.2e}")

    return dc_error


def test_chained_resblocks():
    """Test chaining multiple ResBlocks."""
    print("\n" + "="*60)
    print("Chained ResBlocks Test")
    print("="*60)

    torch.manual_seed(42)
    x = torch.randn(1, 64, 8, 8)

    for n_blocks in [1, 2, 3, 5]:
        # Create standard PyTorch model
        blocks = nn.Sequential(*[ResBlock(64) for _ in range(n_blocks)])
        blocks.eval()

        # Get original output
        orig_out = blocks(x)

        # Make DC compatible
        blocks = make_dc_compatible(blocks)

        # Patch
        patch_model(blocks)

        # DC forward
        x_cat = init_catted(x, InputMode.CENTER)
        out_cat = blocks(x_cat)
        dc_out = reconstruct_output(out_cat)

        unpatch_model(blocks)

        error = torch.norm(orig_out - dc_out).item()
        rel_error = error / (orig_out.norm().item() + 1e-10)

        print(f"{n_blocks} blocks: error={error:.2e}, rel_error={rel_error:.2e}")


def test_simple_resnet():
    """Test SimpleResNet with multiple blocks."""
    print("\n" + "="*60)
    print("SimpleResNet Test (3 blocks)")
    print("="*60)

    torch.manual_seed(42)
    x = torch.randn(1, 3, 32, 32)

    # Create standard PyTorch model
    model = SimpleResNet(num_blocks=3)
    model.eval()

    # Get original output
    orig_out = model(x)

    # Make DC compatible
    model = make_dc_compatible(model)

    # Patch
    patch_model(model)
    mark_output_layer(model.fc)

    # DC forward
    x_cat = init_catted(x, InputMode.CENTER)
    out_cat = model(x_cat)
    dc_out = reconstruct_output(out_cat)

    unpatch_model(model)

    error = torch.norm(orig_out - dc_out).item()
    rel_error = error / (orig_out.norm().item() + 1e-10)

    print(f"Error: {error:.2e}, Relative: {rel_error:.2e}")

    return error


def run_all_tests():
    """Run all patcher ResNet tests."""
    print("="*60)
    print("DC Decomposition - Patcher ResNet Tests")
    print("="*60)
    print("\nWorkflow: model -> make_dc_compatible() -> patch_model() -> DC forward")

    torch.manual_seed(42)

    # Test residual addition
    add_error = test_residual_addition()
    assert add_error < 1e-6, f"Residual addition error too high: {add_error}"

    # Test individual components
    test_individual_components()

    # Test ResBlock with functional replacer
    resblock_error = test_resblock_with_functional_replacer()

    # Test chained blocks
    test_chained_resblocks()

    # Test full ResNet
    resnet_error = test_simple_resnet()

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Residual addition: PASS (error={add_error:.2e})")
    print(f"ResBlock: error={resblock_error:.2e}")
    print(f"SimpleResNet (3 blocks): error={resnet_error:.2e}")

    if resnet_error < 1e-3:
        print("\n✓ DC decomposition with functional replacer: PASS")
    else:
        print(f"\n✗ DC decomposition: FAIL (error too high)")

    print("\nUsage:")
    print("  1. model = make_dc_compatible(model)  # Replace torch.relu, + with modules")
    print("  2. patch_model(model)                  # Enable DC decomposition")
    print("  3. x_cat = init_catted(x)              # Convert input to DC format")
    print("  4. out_cat = model(x_cat)              # Forward pass")
    print("  5. out = reconstruct_output(out_cat)   # Reconstruct output")


if __name__ == "__main__":
    run_all_tests()
