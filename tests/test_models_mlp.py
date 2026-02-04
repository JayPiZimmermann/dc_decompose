"""
Test DC decomposition on MLP models.

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
# MLP Models
# ============================================================================

def MLP_1layer():
    return nn.Sequential(nn.Linear(3, 2))

def MLP_1layer_relu():
    return nn.Sequential(nn.Linear(3, 2), nn.ReLU())

def MLP_2layer():
    return nn.Sequential(nn.Linear(3, 2), nn.ReLU(), nn.Linear(2, 1))

def MLP_3layer():
    return nn.Sequential(nn.Linear(3, 2), nn.ReLU(), nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 1))

def MLP_4layer():
    return nn.Sequential(
        nn.Linear(3, 4), nn.ReLU(),
        nn.Linear(4, 3), nn.ReLU(),
        nn.Linear(3, 2), nn.ReLU(),
        nn.Linear(2, 1)
    )

def MLP_5layer():
    return nn.Sequential(
        nn.Linear(3, 4), nn.ReLU(),
        nn.Linear(4, 4), nn.ReLU(),
        nn.Linear(4, 3), nn.ReLU(),
        nn.Linear(3, 2), nn.ReLU(),
        nn.Linear(2, 1)
    )


MLP_MODELS = {
    "MLP_1layer": (MLP_1layer, (1, 3)),
    "MLP_1layer_relu": (MLP_1layer_relu, (1, 3)),
    "MLP_2layer": (MLP_2layer, (1, 3)),
    "MLP_3layer": (MLP_3layer, (1, 3)),
    "MLP_4layer": (MLP_4layer, (1, 3)),
    "MLP_5layer": (MLP_5layer, (1, 3)),
}


def run_tests():
    """Run all MLP tests."""
    print("=" * 70)
    print("MLP Model Tests (with prepare_model_for_dc)")
    print("=" * 70)

    torch.manual_seed(42)

    results = {}
    for name, (model_fn, input_shape) in MLP_MODELS.items():
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
        print("\nAll MLP tests: PASS")
    else:
        print("\nSome MLP tests: FAIL")

    return all_pass


if __name__ == "__main__":
    run_tests()
