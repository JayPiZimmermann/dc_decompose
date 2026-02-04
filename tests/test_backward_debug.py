"""
Debug backward pass step by step.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from dc_decompose.operations.patcher import patch_model, unpatch_model, mark_output_layer
from dc_decompose.operations.base import init_catted, reconstruct_output, InputMode


def test_linear_backward():
    """Test Linear layer backward step by step."""
    print("="*70)
    print("Test: Linear Backward - Step by Step")
    print("="*70)

    torch.manual_seed(42)

    layer = nn.Linear(4, 3, bias=False)
    x = torch.randn(2, 4, requires_grad=True)
    target_grad = torch.randn(2, 3)

    # ===== Original backward =====
    out_orig = layer(x)
    out_orig.backward(target_grad)
    grad_orig = x.grad.clone()

    print(f"Original output: {out_orig}")
    print(f"Original input grad: {grad_orig}")
    print(f"Expected: W.T @ target_grad = {(layer.weight.T @ target_grad.T).T}")

    # ===== DC backward =====
    x.grad = None

    patch_model(layer)
    mark_output_layer(layer)

    # Create DC input
    pos = torch.relu(x)
    neg = torch.relu(-x)
    x_cat = torch.cat([pos, neg, pos, neg], dim=0)
    x_cat = x_cat.clone().detach().requires_grad_(True)

    print(f"\nDC input x_cat shape: {x_cat.shape}")
    print(f"pos:\n{pos}")
    print(f"neg:\n{neg}")

    out_cat = layer(x_cat)
    print(f"\nDC output shape: {out_cat.shape}")

    # Reconstruct and backward
    q = out_cat.shape[0] // 4
    out_pos = out_cat[:q]
    out_neg = out_cat[q:2*q]
    out_dc = out_pos - out_neg

    print(f"out_pos:\n{out_pos}")
    print(f"out_neg:\n{out_neg}")
    print(f"out_dc (reconstructed):\n{out_dc}")
    print(f"Forward error: {torch.norm(out_orig - out_dc).item():.2e}")

    out_dc.backward(target_grad)

    # Get DC gradient
    grad_cat = x_cat.grad
    print(f"\nDC gradient x_cat.grad shape: {grad_cat.shape}")

    q = grad_cat.shape[0] // 4
    delta_pp = grad_cat[:q]
    delta_np = grad_cat[q:2*q]
    delta_pn = grad_cat[2*q:3*q]
    delta_nn = grad_cat[3*q:]

    print(f"delta_pp:\n{delta_pp}")
    print(f"delta_np:\n{delta_np}")
    print(f"delta_pn:\n{delta_pn}")
    print(f"delta_nn:\n{delta_nn}")

    # Reconstruct gradient using pp - np - pn + nn
    grad_dc = delta_pp - delta_np - delta_pn + delta_nn
    print(f"\nReconstructed grad (pp - np - pn + nn):\n{grad_dc}")
    print(f"Original grad:\n{grad_orig}")

    error = torch.norm(grad_orig - grad_dc).item()
    print(f"\nGradient error: {error:.2e}")

    unpatch_model(layer)

    return error


def test_relu_backward():
    """Test ReLU backward step by step."""
    print("\n" + "="*70)
    print("Test: ReLU Backward - Step by Step")
    print("="*70)

    torch.manual_seed(42)

    layer = nn.ReLU()
    x = torch.tensor([[-1.0, 2.0], [0.5, -0.5]], requires_grad=True)
    target_grad = torch.tensor([[1.0, 1.0], [1.0, 1.0]])

    # ===== Original backward =====
    out_orig = layer(x)
    out_orig.backward(target_grad)
    grad_orig = x.grad.clone()

    print(f"Input x:\n{x}")
    print(f"Original output (ReLU(x)):\n{out_orig}")
    print(f"Original grad (should be target * (x > 0)):\n{grad_orig}")

    # ===== DC backward =====
    x.grad = None

    patch_model(layer)
    mark_output_layer(layer)

    # Create DC input
    pos = torch.relu(x)
    neg = torch.relu(-x)
    x_cat = torch.cat([pos, neg, pos, neg], dim=0)
    x_cat = x_cat.clone().detach().requires_grad_(True)

    print(f"\nDC input:")
    print(f"pos = relu(x):\n{pos}")
    print(f"neg = relu(-x):\n{neg}")
    print(f"z = pos - neg:\n{pos - neg}")

    out_cat = layer(x_cat)

    # Reconstruct
    q = out_cat.shape[0] // 4
    out_pos = out_cat[:q]
    out_neg = out_cat[q:2*q]
    out_dc = out_pos - out_neg

    print(f"\nDC output:")
    print(f"out_pos:\n{out_pos}")
    print(f"out_neg:\n{out_neg}")
    print(f"out_dc = out_pos - out_neg:\n{out_dc}")
    print(f"Forward error: {torch.norm(out_orig - out_dc).item():.2e}")

    out_dc.backward(target_grad)

    grad_cat = x_cat.grad
    q = grad_cat.shape[0] // 4
    delta_pp = grad_cat[:q]
    delta_np = grad_cat[q:2*q]
    delta_pn = grad_cat[2*q:3*q]
    delta_nn = grad_cat[3*q:]

    print(f"\nDC gradients:")
    print(f"delta_pp:\n{delta_pp}")
    print(f"delta_np:\n{delta_np}")
    print(f"delta_pn:\n{delta_pn}")
    print(f"delta_nn:\n{delta_nn}")

    # Reconstruct gradient
    grad_dc = delta_pp - delta_np - delta_pn + delta_nn
    print(f"\nReconstructed grad (pp - np - pn + nn):\n{grad_dc}")
    print(f"Original grad:\n{grad_orig}")

    error = torch.norm(grad_orig - grad_dc).item()
    print(f"\nGradient error: {error:.2e}")

    unpatch_model(layer)

    return error


def test_mlp_backward():
    """Test MLP backward."""
    print("\n" + "="*70)
    print("Test: MLP Backward")
    print("="*70)

    torch.manual_seed(42)

    model = nn.Sequential(
        nn.Linear(4, 4, bias=False),
        nn.ReLU(),
        nn.Linear(4, 2, bias=False)
    )

    x = torch.randn(2, 4, requires_grad=True)
    target_grad = torch.randn(2, 2)

    # Original
    out_orig = model(x)
    out_orig.backward(target_grad)
    grad_orig = x.grad.clone()

    print(f"Original output:\n{out_orig}")
    print(f"Original grad:\n{grad_orig}")

    # DC
    x.grad = None

    patch_model(model)
    mark_output_layer(model[2])  # Last Linear

    pos = torch.relu(x)
    neg = torch.relu(-x)
    x_cat = torch.cat([pos, neg, pos, neg], dim=0).clone().detach().requires_grad_(True)

    out_cat = model(x_cat)
    q = out_cat.shape[0] // 4
    out_dc = out_cat[:q] - out_cat[q:2*q]

    print(f"\nDC output:\n{out_dc}")
    print(f"Forward error: {torch.norm(out_orig - out_dc).item():.2e}")

    out_dc.backward(target_grad)

    grad_cat = x_cat.grad
    q = grad_cat.shape[0] // 4
    grad_dc = grad_cat[:q] - grad_cat[q:2*q] - grad_cat[2*q:3*q] + grad_cat[3*q:]

    print(f"\nDC grad:\n{grad_dc}")
    print(f"Original grad:\n{grad_orig}")

    error = torch.norm(grad_orig - grad_dc).item()
    print(f"\nGradient error: {error:.2e}")

    unpatch_model(model)

    return error


if __name__ == "__main__":
    test_linear_backward()
    test_relu_backward()
    test_mlp_backward()
