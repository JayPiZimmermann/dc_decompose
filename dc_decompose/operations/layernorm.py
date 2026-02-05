"""DC Decomposition for LayerNorm. Forward/Backward: [4*batch] -> [4*batch]

LayerNorm computes: y = (x - mean) / sqrt(var + eps) * weight + bias
For DC: we compute mean and var on z = pos - neg, cache them for backward.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, List

from .base import (
    split_input_4, make_output_4, make_grad_4,
    init_backward, recenter_forward,
    DC_ENABLED, DC_ORIGINAL_FORWARD, DC_IS_OUTPUT_LAYER, DC_BETA
)


class DCLayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_4: Tensor, normalized_shape: List[int],
                weight: Tensor, bias: Tensor, eps: float,
                is_output_layer: bool, beta: float) -> Tensor:
        pos, neg = split_input_4(input_4)
        z = pos - neg

        # Compute LayerNorm on z
        # Mean and variance are computed over normalized_shape dimensions
        dims = list(range(-len(normalized_shape), 0))
        mean = z.mean(dim=dims, keepdim=True)
        var = z.var(dim=dims, unbiased=False, keepdim=True)

        z_norm = (z - mean) / torch.sqrt(var + eps)

        if weight is not None:
            z_norm = z_norm * weight
        if bias is not None:
            z_norm = z_norm + bias

        # Output in DC format: pos = relu(result), neg = relu(-result)
        out_pos = F.relu(z_norm)
        out_neg = F.relu(-z_norm)

        # Cache for backward
        ctx.save_for_backward(z, mean, var, weight)
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        ctx.is_output_layer = is_output_layer
        ctx.beta = beta

        return make_output_4(out_pos, out_neg)

    @staticmethod
    def backward(ctx, grad_4: Tensor) -> Tuple[Tensor, None, None, None, None, None, None]:
        z, mean, var, weight = ctx.saved_tensors
        normalized_shape = ctx.normalized_shape
        eps = ctx.eps

        delta_pp, delta_np, delta_pn, delta_nn = init_backward(
            grad_4, ctx.is_output_layer, ctx.beta)

        # Reconstruct gradient w.r.t. z for each sensitivity
        # LayerNorm backward: dL/dz = dL/dy * dy/dz
        # dy/dz = weight / sqrt(var + eps) * (I - 1/N - z_centered * z_centered^T / (N * var))

        dims = list(range(-len(normalized_shape), 0))
        N = 1
        for d in normalized_shape:
            N *= d

        std_inv = 1.0 / torch.sqrt(var + eps)
        z_centered = z - mean

        def layernorm_backward(delta):
            # delta is dL/dy, we compute dL/dz
            if weight is not None:
                delta = delta * weight

            # dL/dz = std_inv * (delta - mean(delta) - z_centered * mean(delta * z_centered) / var)
            delta_mean = delta.mean(dim=dims, keepdim=True)
            delta_z_mean = (delta * z_centered).mean(dim=dims, keepdim=True)

            dz = std_inv * (delta - delta_mean - z_centered * delta_z_mean / (var + eps))
            return dz

        # Apply LayerNorm backward to each sensitivity
        # Since output was relu(z_norm) and relu(-z_norm), we need to account for relu gradient
        z_norm = (z - mean) * std_inv
        if weight is not None:
            z_norm_scaled = z_norm * weight
        else:
            z_norm_scaled = z_norm

        # Gradient through relu: only passes where input > 0
        relu_grad_pos = (z_norm_scaled > 0).float()
        relu_grad_neg = (-z_norm_scaled > 0).float()

        # Chain rule: delta for layernorm = relu_grad * upstream_delta
        grad_pp = layernorm_backward(delta_pp * relu_grad_pos - delta_np * relu_grad_neg)
        grad_np = layernorm_backward(delta_np * relu_grad_pos - delta_pp * relu_grad_neg)
        grad_pn = layernorm_backward(delta_pn * relu_grad_pos - delta_nn * relu_grad_neg)
        grad_nn = layernorm_backward(delta_nn * relu_grad_pos - delta_pn * relu_grad_neg)

        return make_grad_4(grad_pp, grad_np, grad_pn, grad_nn), None, None, None, None, None, None


def dc_forward_layernorm(module: nn.LayerNorm, x: Tensor) -> Tensor:
    return DCLayerNormFunction.apply(
        x, list(module.normalized_shape),
        module.weight, module.bias, module.eps,
        getattr(module, DC_IS_OUTPUT_LAYER, False),
        getattr(module, DC_BETA, 0.5)
    )


def patch_layernorm(module: nn.LayerNorm) -> None:
    if hasattr(module, DC_ORIGINAL_FORWARD):
        return
    setattr(module, DC_ORIGINAL_FORWARD, module.forward)
    setattr(module, DC_ENABLED, True)
    setattr(module, DC_IS_OUTPUT_LAYER, False)
    setattr(module, DC_BETA, 0.5)

    def patched(x):
        if getattr(module, DC_ENABLED, False):
            return dc_forward_layernorm(module, x)
        else:
            return getattr(module, DC_ORIGINAL_FORWARD)(x)

    module.forward = patched


def unpatch_layernorm(module: nn.LayerNorm) -> None:
    if hasattr(module, DC_ORIGINAL_FORWARD):
        module.forward = getattr(module, DC_ORIGINAL_FORWARD)
        for a in [DC_ORIGINAL_FORWARD, DC_ENABLED, DC_IS_OUTPUT_LAYER, DC_BETA]:
            if hasattr(module, a):
                delattr(module, a)
