"""
Test DC decomposition on ResNet models with residual connections.

These tests use prepare_model_for_dc() to handle torch.relu() and + operations.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from dc_decompose.operations.patcher import prepare_model_for_dc, unpatch_model
from dc_decompose.operations.base import init_catted, reconstruct_output, InputMode, split_grad_4


def test_model_forward(model, x, name="model"):
    """Test forward pass reconstruction."""
    model.eval()

    # Original output
    with torch.no_grad():
        orig_out = model(x)

    # Prepare for DC (replaces functional, patches, marks output)
    model = prepare_model_for_dc(model)

    x_cat = init_catted(x, InputMode.CENTER)
    with torch.no_grad():
        out_cat = model(x_cat)
    dc_out = reconstruct_output(out_cat)

    unpatch_model(model)

    error = torch.norm(orig_out - dc_out).item()
    rel_error = error / (torch.norm(orig_out).item() + 1e-10)

    print(f"{name}: forward error={error:.2e}, rel={rel_error:.2e}")
    return error, rel_error


def test_model_backward(model, x, name="model"):
    """Test backward pass gradient reconstruction."""
    model.eval()

    # Original backward
    x_orig = x.clone().detach().requires_grad_(True)
    orig_out = model(x_orig)
    target_grad = torch.randn_like(orig_out)
    orig_out.backward(target_grad)
    grad_orig = x_orig.grad.clone()

    # Prepare for DC (replaces functional, patches, marks output)
    model = prepare_model_for_dc(model)

    x_dc = x.clone().detach()
    x_cat = init_catted(x_dc, InputMode.CENTER)
    x_cat.requires_grad_(True)

    out_cat = model(x_cat)
    dc_out = reconstruct_output(out_cat)
    dc_out.backward(target_grad)

    # Reconstruct gradient from 4-sensitivities
    grad_cat = x_cat.grad
    delta_pp, delta_np, delta_pn, delta_nn = split_grad_4(grad_cat)
    grad_dc = delta_pp - delta_np - delta_pn + delta_nn

    unpatch_model(model)

    error = torch.norm(grad_orig - grad_dc).item()
    rel_error = error / (torch.norm(grad_orig).item() + 1e-10)

    print(f"{name}: backward error={error:.2e}, rel={rel_error:.2e}")
    return error, rel_error


# ============================================================================
# ResNet-style Models (using torch.relu and +)
# ============================================================================

class ResBlock(nn.Module):
    """Standard ResBlock using torch.relu() and + (standard PyTorch)."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels, 1)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity  # Residual connection
        out = torch.relu(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class ResBlockNoBN(nn.Module):
    """ResBlock without BatchNorm."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels, 1)

    def forward(self, x):
        identity = x
        out = torch.relu(self.conv1(x))
        out = self.conv2(out)
        out = torch.relu(out + identity)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class SimpleResNet(nn.Module):
    """Simple ResNet with multiple blocks."""
    def __init__(self, num_blocks=2, channels=16):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.blocks = nn.Sequential(*[ResBlockInternal(channels) for _ in range(num_blocks)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels, 1)

    def forward(self, x):
        x = self.stem(x)
        x = torch.relu(x)
        x = self.blocks(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ResBlockInternal(nn.Module):
    """Internal ResBlock (no final fc) for use in SimpleResNet."""
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
        out = out + identity
        out = torch.relu(out)
        return out


def ResBlock_single():
    model = ResBlock(16)
    model.eval()
    return model

def ResBlockNoBN_single():
    model = ResBlockNoBN(16)
    model.eval()
    return model

def ResNet_2blocks():
    model = SimpleResNet(num_blocks=2, channels=16)
    model.eval()
    return model

def ResNet_3blocks():
    model = SimpleResNet(num_blocks=3, channels=16)
    model.eval()
    return model


RESNET_MODELS = {
    "ResBlock": (ResBlock_single, (1, 16, 8, 8)),
    "ResBlockNoBN": (ResBlockNoBN_single, (1, 16, 8, 8)),
    "ResNet_2blocks": (ResNet_2blocks, (1, 3, 16, 16)),
    "ResNet_3blocks": (ResNet_3blocks, (1, 3, 16, 16)),
}


def run_tests():
    """Run all ResNet tests."""
    print("=" * 70)
    print("ResNet Model Tests (with prepare_model_for_dc)")
    print("=" * 70)

    torch.manual_seed(42)

    results = {}
    for name, (model_fn, input_shape) in RESNET_MODELS.items():
        print(f"\n--- {name} ---")
        model = model_fn()
        x = torch.randn(*input_shape)

        fwd_err, fwd_rel = test_model_forward(model, x, name)

        # Re-create model for backward test
        model = model_fn()
        bwd_err, bwd_rel = test_model_backward(model, x, name)

        results[name] = {
            "forward_error": fwd_err,
            "forward_rel": fwd_rel,
            "backward_error": bwd_err,
            "backward_rel": bwd_rel,
        }

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    all_pass = True
    for name, res in results.items():
        fwd_pass = res["forward_rel"] < 1e-4
        bwd_pass = res["backward_rel"] < 1e-2
        status = "PASS" if (fwd_pass and bwd_pass) else "FAIL"
        if not (fwd_pass and bwd_pass):
            all_pass = False
        print(f"{name:20s}: fwd_rel={res['forward_rel']:.2e}, bwd_rel={res['backward_rel']:.2e} [{status}]")

    if all_pass:
        print("\nAll ResNet tests: PASS")
    else:
        print("\nSome ResNet tests: FAIL")

    return all_pass


if __name__ == "__main__":
    run_tests()
