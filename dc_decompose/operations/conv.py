"""DC Decomposition for Conv1d and Conv2d layers. Forward/Backward: [4*batch] -> [4*batch]"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple

from .base import (
    split_input_4, make_output_4, make_grad_4,
    init_backward, recenter_forward,
    DC_ENABLED, DC_ORIGINAL_FORWARD, DC_IS_OUTPUT_LAYER, DC_BETA, DC_SPLIT_WEIGHTS_ON_FLY
)


# =============================================================================
# Conv1d
# =============================================================================

class DCConv1dFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_4: Tensor, weight: Tensor, bias: Optional[Tensor],
                stride, padding, dilation, groups, is_output_layer: bool, beta: float, split_on_fly: bool) -> Tensor:
        pos, neg = split_input_4(input_4)

        # Split weights on-the-fly (temporary tensors, auto-cleanup)
        weight_pos = F.relu(weight)
        weight_neg = F.relu(-weight)

        out_pos = F.conv1d(pos, weight_pos, None, stride, padding, dilation, groups)
        out_pos = out_pos + F.conv1d(neg, weight_neg, None, stride, padding, dilation, groups)
        out_neg = F.conv1d(pos, weight_neg, None, stride, padding, dilation, groups)
        out_neg = out_neg + F.conv1d(neg, weight_pos, None, stride, padding, dilation, groups)

        if bias is not None:
            out_pos = out_pos + F.relu(bias).view(1, -1, 1)
            out_neg = out_neg + F.relu(-bias).view(1, -1, 1)

        # Save for backward: only save original weight if split_on_fly=True
        if split_on_fly:
            ctx.save_for_backward(weight)
        else:
            ctx.save_for_backward(weight_pos, weight_neg)
            
        ctx.stride, ctx.padding, ctx.dilation, ctx.groups = stride, padding, dilation, groups
        ctx.is_output_layer, ctx.beta = is_output_layer, beta
        ctx.split_on_fly = split_on_fly
        ctx.input_shape = pos.shape

        output = make_output_4(out_pos, out_neg)
        return recenter_forward(output)

    @staticmethod
    def backward(ctx, grad_4: Tensor):
        delta_pp, delta_np, delta_pn, delta_nn = init_backward(
            grad_4, ctx.is_output_layer, ctx.beta)

        if ctx.split_on_fly:
            # Split weights on-the-fly in backward
            weight, = ctx.saved_tensors
            
            # Split on-the-fly (temporary tensors, auto-cleanup)
            weight_pos = F.relu(weight)
            weight_neg = F.relu(-weight)
        else:
            # Use pre-computed split weights
            weight_pos, weight_neg = ctx.saved_tensors

        # output_padding for conv_transpose1d
        out_pad = 0
        if ctx.stride > 1:
            l_in = ctx.input_shape[2]
            l_out = delta_pp.shape[2]
            k = weight_pos.shape[2]
            c_l = (l_out - 1) * ctx.stride - 2 * ctx.padding + k
            out_pad = max(0, l_in - c_l)

        def conv_t(d, w):
            return F.conv_transpose1d(d, w, None, ctx.stride, ctx.padding, out_pad, ctx.groups, ctx.dilation)

        new_pp = conv_t(delta_pp, weight_pos) + conv_t(delta_np, weight_neg)
        new_np = conv_t(delta_pp, weight_neg) + conv_t(delta_np, weight_pos)
        new_pn = conv_t(delta_pn, weight_pos) + conv_t(delta_nn, weight_neg)
        new_nn = conv_t(delta_pn, weight_neg) + conv_t(delta_nn, weight_pos)

        return make_grad_4(new_pp, new_np, new_pn, new_nn), None, None, None, None, None, None, None, None, None


def dc_forward_conv1d(m: nn.Conv1d, x: Tensor) -> Tensor:
    split_on_fly = getattr(m, DC_SPLIT_WEIGHTS_ON_FLY, True)
    return DCConv1dFunction.apply(x, m.weight, m.bias, m.stride[0], m.padding[0], m.dilation[0], m.groups,
                                   getattr(m, DC_IS_OUTPUT_LAYER, False), getattr(m, DC_BETA, 0.5), split_on_fly)


def patch_conv1d(m: nn.Conv1d) -> None:
    if hasattr(m, DC_ORIGINAL_FORWARD): return
    setattr(m, DC_ORIGINAL_FORWARD, m.forward)
    setattr(m, DC_ENABLED, True)
    setattr(m, DC_IS_OUTPUT_LAYER, False)
    setattr(m, DC_BETA, 0.5)
    setattr(m, DC_SPLIT_WEIGHTS_ON_FLY, True)

    def patched(x):
        if getattr(m, DC_ENABLED, False):
            return dc_forward_conv1d(m, x)
        else:
            return getattr(m, DC_ORIGINAL_FORWARD)(x)

    m.forward = patched


def unpatch_conv1d(m: nn.Conv1d) -> None:
    if hasattr(m, DC_ORIGINAL_FORWARD):
        m.forward = getattr(m, DC_ORIGINAL_FORWARD)
        for a in [DC_ORIGINAL_FORWARD, DC_ENABLED, DC_IS_OUTPUT_LAYER, DC_BETA, DC_SPLIT_WEIGHTS_ON_FLY]:
            if hasattr(m, a): delattr(m, a)


# =============================================================================
# Conv2d
# =============================================================================

class DCConv2dFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_4: Tensor, weight: Tensor, bias: Optional[Tensor],
                stride, padding, dilation, groups, is_output_layer: bool, beta: float, split_on_fly: bool) -> Tensor:
        pos, neg = split_input_4(input_4)

        # Split weights on-the-fly (temporary tensors, auto-cleanup)
        weight_pos = F.relu(weight)
        weight_neg = F.relu(-weight)

        out_pos = F.conv2d(pos, weight_pos, None, stride, padding, dilation, groups)
        out_pos = out_pos + F.conv2d(neg, weight_neg, None, stride, padding, dilation, groups)
        out_neg = F.conv2d(pos, weight_neg, None, stride, padding, dilation, groups)
        out_neg = out_neg + F.conv2d(neg, weight_pos, None, stride, padding, dilation, groups)

        if bias is not None:
            out_pos = out_pos + F.relu(bias).view(1, -1, 1, 1)
            out_neg = out_neg + F.relu(-bias).view(1, -1, 1, 1)

        # Save for backward: only save original weight if split_on_fly=True
        if split_on_fly:
            ctx.save_for_backward(weight)
        else:
            ctx.save_for_backward(weight_pos, weight_neg)
            
        ctx.stride, ctx.padding, ctx.dilation, ctx.groups = stride, padding, dilation, groups
        ctx.is_output_layer, ctx.beta = is_output_layer, beta
        ctx.split_on_fly = split_on_fly
        ctx.input_shape = pos.shape

        output = make_output_4(out_pos, out_neg)
        return recenter_forward(output)

    @staticmethod
    def backward(ctx, grad_4: Tensor):
        delta_pp, delta_np, delta_pn, delta_nn = init_backward(
            grad_4, ctx.is_output_layer, ctx.beta)

        if ctx.split_on_fly:
            # Split weights on-the-fly in backward
            weight, = ctx.saved_tensors
            
            # Split on-the-fly (temporary tensors, auto-cleanup)
            weight_pos = F.relu(weight)
            weight_neg = F.relu(-weight)
        else:
            # Use pre-computed split weights
            weight_pos, weight_neg = ctx.saved_tensors

        # output_padding for conv_transpose2d
        out_pad = (0, 0)
        if any(s > 1 for s in ctx.stride):
            ih, iw = ctx.input_shape[2], ctx.input_shape[3]
            oh, ow = delta_pp.shape[2], delta_pp.shape[3]
            kh, kw = weight_pos.shape[2], weight_pos.shape[3]
            ch = (oh - 1) * ctx.stride[0] - 2 * ctx.padding[0] + kh
            cw = (ow - 1) * ctx.stride[1] - 2 * ctx.padding[1] + kw
            out_pad = (max(0, ih - ch), max(0, iw - cw))

        def conv_t(d, w):
            return F.conv_transpose2d(d, w, None, ctx.stride, ctx.padding, out_pad, ctx.groups, ctx.dilation)

        new_pp = conv_t(delta_pp, weight_pos) + conv_t(delta_np, weight_neg)
        new_np = conv_t(delta_pp, weight_neg) + conv_t(delta_np, weight_pos)
        new_pn = conv_t(delta_pn, weight_pos) + conv_t(delta_nn, weight_neg)
        new_nn = conv_t(delta_pn, weight_neg) + conv_t(delta_nn, weight_pos)

        return make_grad_4(new_pp, new_np, new_pn, new_nn), None, None, None, None, None, None, None, None, None


def dc_forward_conv2d(m: nn.Conv2d, x: Tensor) -> Tensor:
    split_on_fly = getattr(m, DC_SPLIT_WEIGHTS_ON_FLY, True)
    return DCConv2dFunction.apply(x, m.weight, m.bias, m.stride, m.padding, m.dilation, m.groups,
                                   getattr(m, DC_IS_OUTPUT_LAYER, False), getattr(m, DC_BETA, 0.5), split_on_fly)


def patch_conv2d(m: nn.Conv2d) -> None:
    if hasattr(m, DC_ORIGINAL_FORWARD): return
    setattr(m, DC_ORIGINAL_FORWARD, m.forward)
    setattr(m, DC_ENABLED, True)
    setattr(m, DC_IS_OUTPUT_LAYER, False)
    setattr(m, DC_BETA, 0.5)
    setattr(m, DC_SPLIT_WEIGHTS_ON_FLY, True)

    def patched(x):
        if getattr(m, DC_ENABLED, False):
            return dc_forward_conv2d(m, x)
        else:
            return getattr(m, DC_ORIGINAL_FORWARD)(x)

    m.forward = patched


def unpatch_conv2d(m: nn.Conv2d) -> None:
    if hasattr(m, DC_ORIGINAL_FORWARD):
        m.forward = getattr(m, DC_ORIGINAL_FORWARD)
        for a in [DC_ORIGINAL_FORWARD, DC_ENABLED, DC_IS_OUTPUT_LAYER, DC_BETA, DC_SPLIT_WEIGHTS_ON_FLY]:
            if hasattr(m, a): delattr(m, a)
