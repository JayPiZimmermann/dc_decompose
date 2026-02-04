"""
Test DC decomposition on CNN models.

Tests forward pass reconstruction and backward pass gradient reconstruction.
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

    # Prepare for DC (patches and marks output)
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

    # Prepare for DC (patches and marks output)
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
# CNN Models (all with output dim 1)
# ============================================================================

def CNN_2conv():
    return nn.Sequential(
        nn.Conv2d(1, 2, 2), nn.ReLU(),
        nn.Conv2d(2, 4, 2), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(4, 1)
    )

def CNN_2conv_maxpool():
    return nn.Sequential(
        nn.Conv2d(1, 4, 3, stride=2), nn.ReLU(),
        nn.Conv2d(4, 8, 2), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(8, 1)
    )

def CNN_deep_relu():
    return nn.Sequential(
        nn.Conv2d(1, 4, 2), nn.ReLU(),
        nn.Conv2d(4, 8, 2), nn.ReLU(),
        nn.Conv2d(8, 4, 2), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(4, 1)
    )

def CNN_inner_maxpool():
    return nn.Sequential(
        nn.Conv2d(1, 4, 2), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(4, 8, 2), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(8, 1)
    )

def CNN_inner_avgpool():
    return nn.Sequential(
        nn.Conv2d(1, 4, 2), nn.ReLU(),
        nn.AvgPool2d(2),
        nn.Conv2d(4, 8, 2), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(8, 1)
    )

def CNN_BN_Relu():
    model = nn.Sequential(
        nn.Conv2d(1, 4, 2),
        nn.BatchNorm2d(4),
        nn.ReLU(),
        nn.Conv2d(4, 8, 3),
        nn.BatchNorm2d(8),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(8, 1)
    )
    model.eval()
    return model

def CNN_stride_padding():
    return nn.Sequential(
        nn.Conv2d(1, 4, 3, stride=2, padding=1), nn.ReLU(),
        nn.Conv2d(4, 8, 2, stride=2), nn.ReLU(),
        nn.Conv2d(8, 8, 2, stride=2, padding=1), nn.ReLU(),
        nn.Conv2d(8, 4, 2), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(4, 1)
    )

def CNN_Pointwise():
    model = nn.Sequential(
        nn.Conv2d(1, 16, 3), nn.ReLU(),
        nn.Conv2d(16, 8, 1),  # Pointwise conv
        nn.BatchNorm2d(8), nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(8, 1)
    )
    model.eval()
    return model

def CNN_VGG_Block():
    model = nn.Sequential(
        nn.Conv2d(1, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(),
        nn.Conv2d(8, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(8, 1)
    )
    model.eval()
    return model


CNN_MODELS = {
    "CNN_2conv": (CNN_2conv, (1, 1, 8, 8)),
    "CNN_2conv_maxpool": (CNN_2conv_maxpool, (1, 1, 16, 16)),
    "CNN_deep_relu": (CNN_deep_relu, (1, 1, 8, 8)),
    "CNN_inner_maxpool": (CNN_inner_maxpool, (1, 1, 8, 8)),
    "CNN_inner_avgpool": (CNN_inner_avgpool, (1, 1, 8, 8)),
    "CNN_BN_Relu": (CNN_BN_Relu, (1, 1, 8, 8)),
    "CNN_stride_padding": (CNN_stride_padding, (1, 1, 32, 32)),
    "CNN_Pointwise": (CNN_Pointwise, (1, 1, 16, 16)),
    "CNN_VGG_Block": (CNN_VGG_Block, (1, 1, 16, 16)),
}


def run_tests():
    """Run all CNN tests."""
    print("=" * 70)
    print("CNN Model Tests (with prepare_model_for_dc)")
    print("=" * 70)

    torch.manual_seed(42)

    results = {}
    for name, (model_fn, input_shape) in CNN_MODELS.items():
        print(f"\n--- {name} ---")
        model = model_fn()
        x = torch.randn(*input_shape)

        fwd_err, fwd_rel = test_model_forward(model, x, name)
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
        fwd_pass = res["forward_rel"] < 1e-5
        bwd_pass = res["backward_rel"] < 1e-3
        status = "PASS" if (fwd_pass and bwd_pass) else "FAIL"
        if not (fwd_pass and bwd_pass):
            all_pass = False
        print(f"{name:20s}: fwd_rel={res['forward_rel']:.2e}, bwd_rel={res['backward_rel']:.2e} [{status}]")

    if all_pass:
        print("\nAll CNN tests: PASS")
    else:
        print("\nSome CNN tests: FAIL")

    return all_pass


if __name__ == "__main__":
    run_tests()
