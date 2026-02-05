"""DC Decomposition for MaxPool layers. Forward/Backward: [4*batch] -> [4*batch]

Uses winner-takes-all: indices from z = pos - neg applied to both streams.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple

from .base import split_input_4, make_output_4, split_grad_4, make_grad_4, DC_ENABLED, DC_ORIGINAL_FORWARD, DC_IS_OUTPUT_LAYER


class DCMaxPool2dFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_4: Tensor, kernel_size, stride, padding,
                is_output_layer: bool, beta: float) -> Tensor:
        pos, neg = split_input_4(input_4)

        z = pos - neg
        _, indices = F.max_pool2d(z, kernel_size, stride, padding, return_indices=True)

        batch, channels, h_in, w_in = pos.shape
        h_out, w_out = indices.shape[2], indices.shape[3]

        pos_flat = pos.view(batch, channels, -1)
        neg_flat = neg.view(batch, channels, -1)
        indices_flat = indices.view(batch, channels, -1)

        out_pos = torch.gather(pos_flat, 2, indices_flat).view(batch, channels, h_out, w_out)
        out_neg = torch.gather(neg_flat, 2, indices_flat).view(batch, channels, h_out, w_out)

        ctx.save_for_backward(indices)
        ctx.input_shape = pos.shape
        ctx.is_output_layer = is_output_layer
        ctx.beta = beta

        return make_output_4(out_pos, out_neg)

    @staticmethod
    def backward(ctx, grad_4: Tensor) -> Tuple[Tensor, None, None, None, None, None]:
        indices, = ctx.saved_tensors
        batch, channels, h_in, w_in = ctx.input_shape

        if ctx.is_output_layer:
            q = grad_4.shape[0] // 4
            gp, gn = grad_4[:q], grad_4[q:2*q]
            delta_pp, delta_np = ctx.beta * gp, torch.zeros_like(gp)
            delta_pn, delta_nn = (1 - ctx.beta) * gn, torch.zeros_like(gn)
        else:
            delta_pp, delta_np, delta_pn, delta_nn = split_grad_4(grad_4)

        indices_flat = indices.view(batch, channels, -1)

        def scatter_grad(delta: Tensor) -> Tensor:
            delta_flat = delta.view(batch, channels, -1)
            out_flat = torch.zeros(batch, channels, h_in * w_in, device=delta.device, dtype=delta.dtype)
            out_flat.scatter_(2, indices_flat, delta_flat)
            return out_flat.view(batch, channels, h_in, w_in)

        new_pp = scatter_grad(delta_pp)
        new_np = scatter_grad(delta_np)
        new_pn = scatter_grad(delta_pn)
        new_nn = scatter_grad(delta_nn)

        return make_grad_4(new_pp, new_np, new_pn, new_nn), None, None, None, None, None


class DCMaxPool1dFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_4: Tensor, kernel_size, stride, padding,
                is_output_layer: bool, beta: float) -> Tensor:
        pos, neg = split_input_4(input_4)

        z = pos - neg
        _, indices = F.max_pool1d(z, kernel_size, stride, padding, return_indices=True)

        batch, channels, l_in = pos.shape
        l_out = indices.shape[2]

        pos_flat = pos.view(batch, channels, -1)
        neg_flat = neg.view(batch, channels, -1)
        indices_flat = indices.view(batch, channels, -1)

        out_pos = torch.gather(pos_flat, 2, indices_flat).view(batch, channels, l_out)
        out_neg = torch.gather(neg_flat, 2, indices_flat).view(batch, channels, l_out)

        ctx.save_for_backward(indices)
        ctx.input_shape = pos.shape
        ctx.is_output_layer = is_output_layer
        ctx.beta = beta

        return make_output_4(out_pos, out_neg)

    @staticmethod
    def backward(ctx, grad_4: Tensor) -> Tuple[Tensor, None, None, None, None, None]:
        indices, = ctx.saved_tensors
        batch, channels, l_in = ctx.input_shape

        if ctx.is_output_layer:
            q = grad_4.shape[0] // 4
            gp, gn = grad_4[:q], grad_4[q:2*q]
            delta_pp, delta_np = ctx.beta * gp, torch.zeros_like(gp)
            delta_pn, delta_nn = (1 - ctx.beta) * gn, torch.zeros_like(gn)
        else:
            delta_pp, delta_np, delta_pn, delta_nn = split_grad_4(grad_4)

        indices_flat = indices.view(batch, channels, -1)

        def scatter_grad(delta: Tensor) -> Tensor:
            delta_flat = delta.view(batch, channels, -1)
            out_flat = torch.zeros(batch, channels, l_in, device=delta.device, dtype=delta.dtype)
            out_flat.scatter_(2, indices_flat, delta_flat)
            return out_flat.view(batch, channels, l_in)

        new_pp = scatter_grad(delta_pp)
        new_np = scatter_grad(delta_np)
        new_pn = scatter_grad(delta_pn)
        new_nn = scatter_grad(delta_nn)

        return make_grad_4(new_pp, new_np, new_pn, new_nn), None, None, None, None, None


def dc_forward_maxpool2d(module: nn.MaxPool2d, x: Tensor) -> Tensor:
    kernel_size = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
    stride = module.stride if isinstance(module.stride, tuple) else (module.stride, module.stride)
    padding = module.padding if isinstance(module.padding, tuple) else (module.padding, module.padding)
    return DCMaxPool2dFunction.apply(x, kernel_size, stride, padding,
                                      getattr(module, DC_IS_OUTPUT_LAYER, False), 0.5)


def dc_forward_maxpool1d(module: nn.MaxPool1d, x: Tensor) -> Tensor:
    stride = module.stride if module.stride is not None else module.kernel_size
    return DCMaxPool1dFunction.apply(x, module.kernel_size, stride, module.padding,
                                      getattr(module, DC_IS_OUTPUT_LAYER, False), 0.5)


def patch_maxpool2d(module: nn.MaxPool2d) -> None:
    if hasattr(module, DC_ORIGINAL_FORWARD): return
    setattr(module, DC_ORIGINAL_FORWARD, module.forward)
    setattr(module, DC_ENABLED, True)
    setattr(module, DC_IS_OUTPUT_LAYER, False)
    

    def patched(x):
        if getattr(module, DC_ENABLED, False):
            return dc_forward_maxpool2d(module, x)
        else:
            return getattr(module, DC_ORIGINAL_FORWARD)(x)

    module.forward = patched


def patch_maxpool1d(module: nn.MaxPool1d) -> None:
    if hasattr(module, DC_ORIGINAL_FORWARD): return
    setattr(module, DC_ORIGINAL_FORWARD, module.forward)
    setattr(module, DC_ENABLED, True)
    setattr(module, DC_IS_OUTPUT_LAYER, False)
    

    def patched(x):
        if getattr(module, DC_ENABLED, False):
            return dc_forward_maxpool1d(module, x)
        else:
            return getattr(module, DC_ORIGINAL_FORWARD)(x)

    module.forward = patched


def unpatch_maxpool2d(module: nn.MaxPool2d) -> None:
    if hasattr(module, DC_ORIGINAL_FORWARD):
        module.forward = getattr(module, DC_ORIGINAL_FORWARD)
        for a in [DC_ORIGINAL_FORWARD, DC_ENABLED, DC_IS_OUTPUT_LAYER]:
            if hasattr(module, a): delattr(module, a)


def unpatch_maxpool1d(module: nn.MaxPool1d) -> None:
    if hasattr(module, DC_ORIGINAL_FORWARD):
        module.forward = getattr(module, DC_ORIGINAL_FORWARD)
        for a in [DC_ORIGINAL_FORWARD, DC_ENABLED, DC_IS_OUTPUT_LAYER]:
            if hasattr(module, a): delattr(module, a)
