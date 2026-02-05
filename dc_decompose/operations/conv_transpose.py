"""DC Decomposition for ConvTranspose layers. Forward/Backward: [4*batch] -> [4*batch]"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple

from .base import split_input_4, make_output_4, make_grad_4, init_backward, DC_ENABLED, DC_ORIGINAL_FORWARD, DC_IS_OUTPUT_LAYER, DC_BETA


class DCConvTranspose2dFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_4: Tensor, weight: Tensor, bias: Optional[Tensor],
                stride, padding, output_padding, dilation, groups,
                is_output_layer: bool, beta: float) -> Tensor:
        pos, neg = split_input_4(input_4)

        weight_pos = F.relu(weight)
        weight_neg = F.relu(-weight)

        out_pos = F.conv_transpose2d(pos, weight_pos, None, stride, padding, output_padding, groups, dilation)
        out_pos = out_pos + F.conv_transpose2d(neg, weight_neg, None, stride, padding, output_padding, groups, dilation)
        out_neg = F.conv_transpose2d(pos, weight_neg, None, stride, padding, output_padding, groups, dilation)
        out_neg = out_neg + F.conv_transpose2d(neg, weight_pos, None, stride, padding, output_padding, groups, dilation)

        if bias is not None:
            out_pos = out_pos + F.relu(bias).view(1, -1, 1, 1)
            out_neg = out_neg + F.relu(-bias).view(1, -1, 1, 1)

        ctx.save_for_backward(weight_pos, weight_neg)
        ctx.stride, ctx.padding, ctx.dilation, ctx.groups = stride, padding, dilation, groups
        ctx.is_output_layer, ctx.beta = is_output_layer, beta
        ctx.input_shape = pos.shape

        return make_output_4(out_pos, out_neg)

    @staticmethod
    def backward(ctx, grad_4: Tensor):
        weight_pos, weight_neg = ctx.saved_tensors

        delta_pp, delta_np, delta_pn, delta_nn = init_backward(
            grad_4, ctx.is_output_layer, ctx.beta)

        # Backward of conv_transpose is conv
        def conv(d, w):
            return F.conv2d(d, w, None, ctx.stride, ctx.padding, ctx.dilation, ctx.groups)

        new_pp = conv(delta_pp, weight_pos) + conv(delta_np, weight_neg)
        new_np = conv(delta_pp, weight_neg) + conv(delta_np, weight_pos)
        new_pn = conv(delta_pn, weight_pos) + conv(delta_nn, weight_neg)
        new_nn = conv(delta_pn, weight_neg) + conv(delta_nn, weight_pos)

        return make_grad_4(new_pp, new_np, new_pn, new_nn), None, None, None, None, None, None, None, None, None


class DCConvTranspose1dFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_4: Tensor, weight: Tensor, bias: Optional[Tensor],
                stride, padding, output_padding, dilation, groups,
                is_output_layer: bool, beta: float) -> Tensor:
        pos, neg = split_input_4(input_4)

        weight_pos = F.relu(weight)
        weight_neg = F.relu(-weight)

        out_pos = F.conv_transpose1d(pos, weight_pos, None, stride, padding, output_padding, groups, dilation)
        out_pos = out_pos + F.conv_transpose1d(neg, weight_neg, None, stride, padding, output_padding, groups, dilation)
        out_neg = F.conv_transpose1d(pos, weight_neg, None, stride, padding, output_padding, groups, dilation)
        out_neg = out_neg + F.conv_transpose1d(neg, weight_pos, None, stride, padding, output_padding, groups, dilation)

        if bias is not None:
            out_pos = out_pos + F.relu(bias).view(1, -1, 1)
            out_neg = out_neg + F.relu(-bias).view(1, -1, 1)

        ctx.save_for_backward(weight_pos, weight_neg)
        ctx.stride, ctx.padding, ctx.dilation, ctx.groups = stride, padding, dilation, groups
        ctx.is_output_layer, ctx.beta = is_output_layer, beta
        ctx.input_shape = pos.shape

        return make_output_4(out_pos, out_neg)

    @staticmethod
    def backward(ctx, grad_4: Tensor):
        weight_pos, weight_neg = ctx.saved_tensors

        delta_pp, delta_np, delta_pn, delta_nn = init_backward(
            grad_4, ctx.is_output_layer, ctx.beta)

        def conv(d, w):
            return F.conv1d(d, w, None, ctx.stride, ctx.padding, ctx.dilation, ctx.groups)

        new_pp = conv(delta_pp, weight_pos) + conv(delta_np, weight_neg)
        new_np = conv(delta_pp, weight_neg) + conv(delta_np, weight_pos)
        new_pn = conv(delta_pn, weight_pos) + conv(delta_nn, weight_neg)
        new_nn = conv(delta_pn, weight_neg) + conv(delta_nn, weight_pos)

        return make_grad_4(new_pp, new_np, new_pn, new_nn), None, None, None, None, None, None, None, None, None


def dc_forward_conv_transpose2d(m: nn.ConvTranspose2d, x: Tensor) -> Tensor:
    stride = m.stride if isinstance(m.stride, tuple) else (m.stride, m.stride)
    padding = m.padding if isinstance(m.padding, tuple) else (m.padding, m.padding)
    output_padding = m.output_padding if isinstance(m.output_padding, tuple) else (m.output_padding, m.output_padding)
    dilation = m.dilation if isinstance(m.dilation, tuple) else (m.dilation, m.dilation)
    return DCConvTranspose2dFunction.apply(
        x, m.weight, m.bias, stride, padding, output_padding, dilation, m.groups,
        getattr(m, DC_IS_OUTPUT_LAYER, False), getattr(m, DC_BETA, 0.5)
    )


def dc_forward_conv_transpose1d(m: nn.ConvTranspose1d, x: Tensor) -> Tensor:
    return DCConvTranspose1dFunction.apply(
        x, m.weight, m.bias, m.stride[0], m.padding[0], m.output_padding[0], m.dilation[0], m.groups,
        getattr(m, DC_IS_OUTPUT_LAYER, False), getattr(m, DC_BETA, 0.5)
    )


def patch_conv_transpose2d(m: nn.ConvTranspose2d) -> None:
    if hasattr(m, DC_ORIGINAL_FORWARD): return
    setattr(m, DC_ORIGINAL_FORWARD, m.forward)
    setattr(m, DC_ENABLED, True)
    setattr(m, DC_IS_OUTPUT_LAYER, False)
    setattr(m, DC_BETA, 0.5)

    def patched(x):
        if getattr(m, DC_ENABLED, False):
            return dc_forward_conv_transpose2d(m, x)
        else:
            return getattr(m, DC_ORIGINAL_FORWARD)(x)

    m.forward = patched


def patch_conv_transpose1d(m: nn.ConvTranspose1d) -> None:
    if hasattr(m, DC_ORIGINAL_FORWARD): return
    setattr(m, DC_ORIGINAL_FORWARD, m.forward)
    setattr(m, DC_ENABLED, True)
    setattr(m, DC_IS_OUTPUT_LAYER, False)
    setattr(m, DC_BETA, 0.5)

    def patched(x):
        if getattr(m, DC_ENABLED, False):
            return dc_forward_conv_transpose1d(m, x)
        else:
            return getattr(m, DC_ORIGINAL_FORWARD)(x)

    m.forward = patched


def unpatch_conv_transpose2d(m: nn.ConvTranspose2d) -> None:
    if hasattr(m, DC_ORIGINAL_FORWARD):
        m.forward = getattr(m, DC_ORIGINAL_FORWARD)
        for a in [DC_ORIGINAL_FORWARD, DC_ENABLED, DC_IS_OUTPUT_LAYER, DC_BETA]:
            if hasattr(m, a): delattr(m, a)


def unpatch_conv_transpose1d(m: nn.ConvTranspose1d) -> None:
    if hasattr(m, DC_ORIGINAL_FORWARD):
        m.forward = getattr(m, DC_ORIGINAL_FORWARD)
        for a in [DC_ORIGINAL_FORWARD, DC_ENABLED, DC_IS_OUTPUT_LAYER, DC_BETA]:
            if hasattr(m, a): delattr(m, a)
