"""
Test backward pass for DC decomposition.

Uses hooks to cache gradients and compare original vs DC gradients.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from dc_decompose.operations.patcher import patch_model, unpatch_model, mark_output_layer
from dc_decompose.operations.base import init_catted, reconstruct_output, InputMode, split_input_4, make_input_4
from dc_decompose.operations.functional_replacer import make_dc_compatible


class GradientCache:
    """Cache for storing gradients captured by hooks."""

    def __init__(self):
        self.input_grads: Dict[str, Tensor] = {}
        self.output_grads: Dict[str, Tensor] = {}
        self.hooks: List = []

    def register_hooks(self, model: nn.Module):
        """Register backward hooks on all modules."""
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                hook = module.register_full_backward_hook(
                    lambda m, grad_in, grad_out, n=name: self._save_grads(n, grad_in, grad_out)
                )
                self.hooks.append(hook)

    def _save_grads(self, name: str, grad_input: Tuple, grad_output: Tuple):
        """Save gradients from backward hook."""
        if grad_input[0] is not None:
            self.input_grads[name] = grad_input[0].clone()
        if grad_output[0] is not None:
            self.output_grads[name] = grad_output[0].clone()

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def clear(self):
        """Clear cached gradients."""
        self.input_grads = {}
        self.output_grads = {}


def compare_gradients(orig_cache: GradientCache, dc_cache: GradientCache, prefix: str = ""):
    """Compare gradients between original and DC caches."""
    print(f"\n{prefix}Gradient Comparison:")

    all_names = set(orig_cache.input_grads.keys()) | set(dc_cache.input_grads.keys())

    errors = {}
    for name in sorted(all_names):
        if name in orig_cache.input_grads and name in dc_cache.input_grads:
            orig_grad = orig_cache.input_grads[name]
            dc_grad = dc_cache.input_grads[name]

            # DC gradient is [4*batch], need to reconstruct
            if dc_grad.shape[0] == 4 * orig_grad.shape[0]:
                q = dc_grad.shape[0] // 4
                # Reconstruct gradient: pp - np - pn + nn
                delta_pp = dc_grad[:q]
                delta_np = dc_grad[q:2*q]
                delta_pn = dc_grad[2*q:3*q]
                delta_nn = dc_grad[3*q:]
                dc_grad_reconstructed = delta_pp - delta_np - delta_pn + delta_nn
            else:
                dc_grad_reconstructed = dc_grad

            error = torch.norm(orig_grad - dc_grad_reconstructed).item()
            rel_error = error / (torch.norm(orig_grad).item() + 1e-10)
            errors[name] = (error, rel_error)

            print(f"  {name:30s}: error={error:.2e}, rel={rel_error:.2e}")

    return errors


def test_backward_linear():
    """Test backward pass through Linear layer."""
    print("="*70)
    print("Test: Backward Pass - Linear")
    print("="*70)

    torch.manual_seed(42)

    # Create model
    model = nn.Linear(10, 5)
    x = torch.randn(2, 10, requires_grad=True)
    target_grad = torch.randn(2, 5)

    # Original backward
    orig_cache = GradientCache()
    orig_cache.register_hooks(model)

    out_orig = model(x)
    out_orig.backward(target_grad)

    orig_cache.remove_hooks()

    print(f"Original output: {out_orig.sum():.4f}")

    # DC backward
    dc_cache = GradientCache()
    patch_model(model)
    mark_output_layer(model)
    dc_cache.register_hooks(model)

    x_cat = init_catted(x, InputMode.CENTER)
    out_cat = model(x_cat)
    out_dc = reconstruct_output(out_cat)

    # Backward with same target gradient
    out_dc.backward(target_grad)

    dc_cache.remove_hooks()
    unpatch_model(model)

    print(f"DC output: {out_dc.sum():.4f}")
    print(f"Forward error: {torch.norm(out_orig - out_dc).item():.2e}")

    errors = compare_gradients(orig_cache, dc_cache)

    return errors


def test_backward_relu():
    """Test backward pass through ReLU."""
    print("\n" + "="*70)
    print("Test: Backward Pass - ReLU")
    print("="*70)

    torch.manual_seed(42)

    model = nn.ReLU()
    x = torch.randn(2, 10, requires_grad=True)
    target_grad = torch.randn(2, 10)

    # Original backward
    orig_cache = GradientCache()
    orig_cache.register_hooks(model)

    out_orig = model(x)
    out_orig.backward(target_grad)

    orig_cache.remove_hooks()

    # DC backward
    dc_cache = GradientCache()
    patch_model(model)
    mark_output_layer(model)
    dc_cache.register_hooks(model)

    x_cat = init_catted(x, InputMode.CENTER)
    out_cat = model(x_cat)
    out_dc = reconstruct_output(out_cat)

    out_dc.backward(target_grad)

    dc_cache.remove_hooks()
    unpatch_model(model)

    print(f"Forward error: {torch.norm(out_orig - out_dc).item():.2e}")

    errors = compare_gradients(orig_cache, dc_cache)

    return errors


def test_backward_conv_bn_relu():
    """Test backward pass through Conv -> BN -> ReLU."""
    print("\n" + "="*70)
    print("Test: Backward Pass - Conv+BN+ReLU")
    print("="*70)

    torch.manual_seed(42)

    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU()
    )
    model.eval()

    x = torch.randn(1, 3, 8, 8, requires_grad=True)

    # Original forward/backward
    orig_cache = GradientCache()
    orig_cache.register_hooks(model)

    out_orig = model(x)
    target_grad = torch.randn_like(out_orig)
    out_orig.backward(target_grad)

    orig_cache.remove_hooks()

    # DC forward/backward
    dc_cache = GradientCache()
    patch_model(model)
    mark_output_layer(model[2])  # ReLU
    dc_cache.register_hooks(model)

    x_cat = init_catted(x, InputMode.CENTER)
    out_cat = model(x_cat)
    out_dc = reconstruct_output(out_cat)

    out_dc.backward(target_grad)

    dc_cache.remove_hooks()
    unpatch_model(model)

    print(f"Forward error: {torch.norm(out_orig - out_dc).item():.2e}")

    errors = compare_gradients(orig_cache, dc_cache)

    return errors


def test_backward_resblock():
    """Test backward pass through ResBlock with functional replacer."""
    print("\n" + "="*70)
    print("Test: Backward Pass - ResBlock")
    print("="*70)

    torch.manual_seed(42)

    class ResBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(channels)
            self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(channels)

        def forward(self, x):
            identity = x
            out = torch.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out = torch.relu(out + identity)
            return out

    model = ResBlock(16)
    model.eval()

    x = torch.randn(1, 16, 8, 8, requires_grad=True)

    # Original forward/backward
    orig_cache = GradientCache()
    orig_cache.register_hooks(model)

    out_orig = model(x)
    target_grad = torch.randn_like(out_orig)
    out_orig.backward(target_grad)

    orig_cache.remove_hooks()

    # Make DC compatible
    model_dc = ResBlock(16)
    model_dc.load_state_dict(model.state_dict())
    model_dc.eval()
    model_dc = make_dc_compatible(model_dc)

    # DC forward/backward
    dc_cache = GradientCache()
    patch_model(model_dc)

    # Find last ReLU for output layer marking
    for name, m in model_dc.named_modules():
        if isinstance(m, nn.ReLU):
            last_relu = m
    mark_output_layer(last_relu)

    dc_cache.register_hooks(model_dc)

    x_cat = init_catted(x, InputMode.CENTER)
    out_cat = model_dc(x_cat)
    out_dc = reconstruct_output(out_cat)

    out_dc.backward(target_grad)

    dc_cache.remove_hooks()
    unpatch_model(model_dc)

    print(f"Forward error: {torch.norm(out_orig - out_dc).item():.2e}")

    # Note: Module names may differ after make_dc_compatible
    print("\nOriginal model gradients:")
    for name, grad in orig_cache.input_grads.items():
        print(f"  {name}: shape={grad.shape}, norm={grad.norm():.4f}")

    print("\nDC model gradients:")
    for name, grad in dc_cache.input_grads.items():
        print(f"  {name}: shape={grad.shape}, norm={grad.norm():.4f}")

    return {}


def test_backward_mlp():
    """Test backward pass through MLP."""
    print("\n" + "="*70)
    print("Test: Backward Pass - MLP")
    print("="*70)

    torch.manual_seed(42)

    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10),
        nn.ReLU(),
        nn.Linear(10, 5)
    )

    x = torch.randn(2, 10, requires_grad=True)
    target_grad = torch.randn(2, 5)

    # Original
    orig_cache = GradientCache()
    orig_cache.register_hooks(model)

    out_orig = model(x)
    out_orig.backward(target_grad)

    orig_cache.remove_hooks()

    # DC
    dc_cache = GradientCache()
    patch_model(model)
    mark_output_layer(model[4])  # Last Linear
    dc_cache.register_hooks(model)

    x_cat = init_catted(x, InputMode.CENTER)
    out_cat = model(x_cat)
    out_dc = reconstruct_output(out_cat)

    out_dc.backward(target_grad)

    dc_cache.remove_hooks()
    unpatch_model(model)

    print(f"Forward error: {torch.norm(out_orig - out_dc).item():.2e}")

    errors = compare_gradients(orig_cache, dc_cache)

    return errors


def run_all_tests():
    """Run all backward pass tests."""
    print("="*70)
    print("DC Decomposition - Backward Pass Tests")
    print("="*70)

    results = {}
    results['linear'] = test_backward_linear()
    results['relu'] = test_backward_relu()
    results['conv_bn_relu'] = test_backward_conv_bn_relu()
    results['mlp'] = test_backward_mlp()
    test_backward_resblock()  # Separate analysis

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("Gradient reconstruction: grad = delta_pp - delta_np - delta_pn + delta_nn\n")

    all_pass = True
    for name, errors in results.items():
        if errors:
            max_error = max(e[0] for e in errors.values()) if errors else 0
            status = "PASS" if max_error < 1e-3 else "FAIL"
            if max_error >= 1e-3:
                all_pass = False
            print(f"{name:20s}: max_error={max_error:.2e} [{status}]")
        else:
            print(f"{name:20s}: no gradients captured")

    if all_pass:
        print("\n✓ All backward pass tests: PASS")
    else:
        print("\n✗ Some backward pass tests: FAIL")


if __name__ == "__main__":
    run_all_tests()
