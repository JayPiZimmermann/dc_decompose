"""
Testing module for DC decomposition.

Tests both forward (activation reconstruction) and backward (gradient reconstruction).
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .base import split4, init_catted, InputMode, DC_ENABLED
from .patcher import patch_model, unpatch_model, mark_output_layer, auto_mark_output_layer


@dataclass
class LayerCache:
    """Cache for a single layer's activations and gradients."""
    name: str
    module_type: str

    # Original model
    original_input: Optional[Tensor] = None
    original_output: Optional[Tensor] = None
    original_grad_input: Optional[Tensor] = None
    original_grad_output: Optional[Tensor] = None

    # DC decomposition - forward
    dc_input_pos: Optional[Tensor] = None
    dc_input_neg: Optional[Tensor] = None
    dc_output_pos: Optional[Tensor] = None
    dc_output_neg: Optional[Tensor] = None

    # DC decomposition - backward (4 sensitivities)
    dc_grad_delta_pp: Optional[Tensor] = None
    dc_grad_delta_np: Optional[Tensor] = None
    dc_grad_delta_pn: Optional[Tensor] = None
    dc_grad_delta_nn: Optional[Tensor] = None

    # Reconstructed values
    dc_output_reconstructed: Optional[Tensor] = None
    dc_grad_reconstructed: Optional[Tensor] = None

    # Errors
    output_error: Optional[float] = None
    relative_output_error: Optional[float] = None
    grad_error: Optional[float] = None
    relative_grad_error: Optional[float] = None


class DCTester:
    """Test harness for DC decomposition validation."""

    def __init__(self, model: nn.Module, input_mode: InputMode = InputMode.CENTER,
                 relu_mode: str = 'max', backprop_mode: str = 'standard', beta: float = 1.0):
        self.model = model
        self.input_mode = input_mode
        self.relu_mode = relu_mode
        self.backprop_mode = backprop_mode
        self.beta = beta

        self.layer_caches: Dict[str, LayerCache] = {}
        self.layer_order: List[str] = []
        self._handles: List = []
        self._has_run = False

    def _get_supported_layers(self) -> List[Tuple[str, nn.Module]]:
        supported_types = (
            nn.Linear, nn.Conv2d, nn.Conv1d, nn.ReLU,
            nn.BatchNorm1d, nn.BatchNorm2d,
            nn.MaxPool1d, nn.MaxPool2d,
            nn.AvgPool1d, nn.AvgPool2d,
            nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d,
            nn.Flatten,
        )
        return [(name, m) for name, m in self.model.named_modules()
                if isinstance(m, supported_types) and name != '']

    def _register_original_hooks(self) -> None:
        for name, module in self._get_supported_layers():
            if name not in self.layer_caches:
                self.layer_caches[name] = LayerCache(name=name, module_type=module.__class__.__name__)
                self.layer_order.append(name)

            cache = self.layer_caches[name]

            def make_fwd_hook(c):
                def hook(m, inp, out):
                    c.original_input = inp[0].detach().clone()
                    c.original_output = out.detach().clone()
                return hook

            def make_bwd_hook(c):
                def hook(m, grad_in, grad_out):
                    if grad_in[0] is not None:
                        c.original_grad_input = grad_in[0].detach().clone()
                    if grad_out[0] is not None:
                        c.original_grad_output = grad_out[0].detach().clone()
                return hook

            self._handles.append(module.register_forward_hook(make_fwd_hook(cache)))
            self._handles.append(module.register_full_backward_hook(make_bwd_hook(cache)))

    def _register_dc_forward_hooks(self) -> None:
        for name, module in self._get_supported_layers():
            cache = self.layer_caches[name]

            def make_hook(c):
                def hook(m, inp, out):
                    inp_cat = inp[0].detach()
                    out_cat = out.detach()
                    # [4*batch] format: [pos; neg; pos; neg]
                    q_in = inp_cat.shape[0] // 4
                    q_out = out_cat.shape[0] // 4
                    inp_pos, inp_neg = inp_cat[:q_in], inp_cat[q_in:2*q_in]
                    out_pos, out_neg = out_cat[:q_out], out_cat[q_out:2*q_out]
                    c.dc_input_pos = inp_pos.clone()
                    c.dc_input_neg = inp_neg.clone()
                    c.dc_output_pos = out_pos.clone()
                    c.dc_output_neg = out_neg.clone()
                    c.dc_output_reconstructed = (out_pos - out_neg).clone()
                return hook

            self._handles.append(module.register_forward_hook(make_hook(cache)))

    def _register_dc_backward_hooks(self) -> None:
        for name, module in self._get_supported_layers():
            cache = self.layer_caches[name]

            def make_hook(c):
                def hook(m, grad_in, grad_out):
                    # grad_in is the 4-sensitivity gradient returned by backward
                    if grad_in[0] is not None:
                        grad = grad_in[0].detach()
                        # Check if it's [4*batch] (4 sensitivities)
                        if grad.shape[0] % 4 == 0:
                            delta_pp, delta_np, delta_pn, delta_nn = split4(grad)
                            c.dc_grad_delta_pp = delta_pp.clone()
                            c.dc_grad_delta_np = delta_np.clone()
                            c.dc_grad_delta_pn = delta_pn.clone()
                            c.dc_grad_delta_nn = delta_nn.clone()
                            # Reconstruct: delta_pp - delta_np - delta_pn + delta_nn
                            c.dc_grad_reconstructed = (delta_pp - delta_np - delta_pn + delta_nn).clone()
                return hook

            self._handles.append(module.register_full_backward_hook(make_hook(cache)))

    def _remove_hooks(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def _compute_errors(self) -> None:
        for cache in self.layer_caches.values():
            # Output error
            if cache.original_output is not None and cache.dc_output_reconstructed is not None:
                diff = (cache.original_output - cache.dc_output_reconstructed).abs()
                cache.output_error = diff.max().item()
                norm = cache.original_output.abs().max().item()
                cache.relative_output_error = cache.output_error / (norm + 1e-10)

            # Gradient error
            if cache.original_grad_input is not None and cache.dc_grad_reconstructed is not None:
                diff = (cache.original_grad_input - cache.dc_grad_reconstructed).abs()
                cache.grad_error = diff.max().item()
                norm = cache.original_grad_input.abs().max().item()
                cache.relative_grad_error = cache.grad_error / (norm + 1e-10)

    def run(self, x: Tensor, target: Optional[Tensor] = None) -> Tensor:
        """Run both original and DC forward/backward passes."""
        self.model.eval()
        self.layer_caches.clear()
        self.layer_order.clear()
        self._remove_hooks()

        # Phase 1: Original forward/backward
        self._register_original_hooks()
        x_orig = x.clone().requires_grad_(True)
        output_orig = self.model(x_orig)

        if target is None:
            target = torch.ones(output_orig.shape[0], 1, device=output_orig.device)
        # Ensure output has single dimension for clean backward
        if output_orig.dim() > 1 and output_orig.shape[-1] > 1:
            output_scalar = output_orig.sum(dim=-1, keepdim=True)
        else:
            output_scalar = output_orig
        loss = ((output_scalar - target) ** 2).mean()
        loss.backward()
        self._remove_hooks()

        # Phase 2: DC forward
        patch_model(self.model, relu_mode=self.relu_mode, backprop_mode=self.backprop_mode)
        auto_mark_output_layer(self.model, beta=self.beta)

        self._register_dc_forward_hooks()
        self._register_dc_backward_hooks()

        x_catted = init_catted(x, mode=self.input_mode)
        x_catted.requires_grad_(True)
        output_catted = self.model(x_catted)

        # [4*batch] format: [pos; neg; pos; neg]
        q = output_catted.shape[0] // 4
        out_pos, out_neg = output_catted[:q], output_catted[q:2*q]
        reconstructed = out_pos - out_neg

        # Same loss computation
        if reconstructed.dim() > 1 and reconstructed.shape[-1] > 1:
            recon_scalar = reconstructed.sum(dim=-1, keepdim=True)
        else:
            recon_scalar = reconstructed
        dc_loss = ((recon_scalar - target) ** 2).mean()
        dc_loss.backward()

        self._remove_hooks()
        unpatch_model(self.model)

        # Phase 3: Compute errors
        self._compute_errors()
        self._has_run = True

        return output_orig.detach()

    def max_activation_error(self) -> float:
        if not self._has_run:
            raise RuntimeError("Must call run() first")
        return max((c.output_error or 0) for c in self.layer_caches.values())

    def max_relative_activation_error(self) -> float:
        if not self._has_run:
            raise RuntimeError("Must call run() first")
        return max((c.relative_output_error or 0) for c in self.layer_caches.values())

    def max_gradient_error(self) -> float:
        if not self._has_run:
            raise RuntimeError("Must call run() first")
        return max((c.grad_error or 0) for c in self.layer_caches.values())

    def max_relative_gradient_error(self) -> float:
        if not self._has_run:
            raise RuntimeError("Must call run() first")
        return max((c.relative_grad_error or 0) for c in self.layer_caches.values())

    def report(self, verbose: bool = True) -> str:
        if not self._has_run:
            raise RuntimeError("Must call run() first")

        lines = ["=" * 80, "DC Decomposition Test Report", "=" * 80, ""]

        max_act = self.max_activation_error()
        max_rel_act = self.max_relative_activation_error()
        max_grad = self.max_gradient_error()
        max_rel_grad = self.max_relative_gradient_error()

        lines.append("ACTIVATIONS:")
        lines.append(f"  Max Absolute Error:  {max_act:.6e}")
        lines.append(f"  Max Relative Error:  {max_rel_act:.6e}")
        lines.append(f"  Status: {'PASS' if max_rel_act < 1e-2 else 'FAIL'}")
        lines.append("")

        lines.append("GRADIENTS (reconstructed = delta_pp - delta_np - delta_pn + delta_nn):")
        lines.append(f"  Max Absolute Error:  {max_grad:.6e}")
        lines.append(f"  Max Relative Error:  {max_rel_grad:.6e}")
        lines.append(f"  Status: {'PASS' if max_rel_grad < 1e-2 else 'FAIL'}")
        lines.append("")

        lines.append("-" * 80)
        lines.append(f"{'Layer':<30} {'Type':<12} {'Act Err':<12} {'Grad Err':<12}")
        lines.append("-" * 80)

        for name in self.layer_order:
            c = self.layer_caches[name]
            act_err = c.output_error if c.output_error is not None else float('nan')
            grad_err = c.grad_error if c.grad_error is not None else float('nan')
            display = name[:28] + '..' if len(name) > 30 else name
            lines.append(f"{display:<30} {c.module_type:<12} {act_err:<12.2e} {grad_err:<12.2e}")

        lines.append("-" * 80)
        return "\n".join(lines)


def test_model(model: nn.Module, input_shape: Tuple[int, ...],
               input_mode: InputMode = InputMode.CENTER,
               relu_mode: str = 'max', backprop_mode: str = 'standard',
               beta: float = 1.0, device: str = 'cpu') -> Tuple[bool, str, DCTester]:
    model = model.to(device).eval()
    x = torch.randn(*input_shape, device=device)

    tester = DCTester(model, input_mode=input_mode, relu_mode=relu_mode,
                      backprop_mode=backprop_mode, beta=beta)
    tester.run(x)

    act_ok = tester.max_relative_activation_error() < 1e-2
    grad_ok = tester.max_relative_gradient_error() < 1e-2 or tester.max_gradient_error() < 1e-6
    passed = act_ok and grad_ok

    return passed, tester.report(), tester
