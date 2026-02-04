"""DC Decomposition for Conv1d layers. Forward/Backward: [4*batch] -> [4*batch]"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple

from .base import split_input_4, make_output_4, split_grad_4, make_grad_4, DC_ENABLED, DC_ORIGINAL_FORWARD, DC_IS_OUTPUT_LAYER, DC_BETA


class DCConv1dFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_4: Tensor, weight: Tensor, bias: Optional[Tensor],
                stride, padding, dilation, groups, is_output_layer: bool, beta: float) -> Tensor:
        pos, neg = split_input_4(input_4)

        weight_pos = F.relu(weight)
        weight_neg = F.relu(-weight)

        out_pos = F.conv1d(pos, weight_pos, None, stride, padding, dilation, groups)
        out_pos = out_pos + F.conv1d(neg, weight_neg, None, stride, padding, dilation, groups)
        out_neg = F.conv1d(pos, weight_neg, None, stride, padding, dilation, groups)
        out_neg = out_neg + F.conv1d(neg, weight_pos, None, stride, padding, dilation, groups)

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

        if ctx.is_output_layer:
            q = grad_4.shape[0] // 4
            gp, gn = grad_4[:q], grad_4[q:2*q]
            delta_pp, delta_np = ctx.beta * gp, torch.zeros_like(gp)
            delta_pn, delta_nn = (1 - ctx.beta) * gn, torch.zeros_like(gn)
        else:
            delta_pp, delta_np, delta_pn, delta_nn = split_grad_4(grad_4)

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

        return make_grad_4(new_pp, new_np, new_pn, new_nn), None, None, None, None, None, None, None, None


def dc_forward_conv1d(m: nn.Conv1d, x: Tensor) -> Tensor:
    return DCConv1dFunction.apply(x, m.weight, m.bias, m.stride[0], m.padding[0], m.dilation[0], m.groups,
                                   getattr(m, DC_IS_OUTPUT_LAYER, False), getattr(m, DC_BETA, 1.0))


def patch_conv1d(m: nn.Conv1d) -> None:
    if hasattr(m, DC_ORIGINAL_FORWARD): return
    setattr(m, DC_ORIGINAL_FORWARD, m.forward)
    setattr(m, DC_ENABLED, True)
    setattr(m, DC_IS_OUTPUT_LAYER, False)
    setattr(m, DC_BETA, 1.0)

    def patched(x):
        if getattr(m, DC_ENABLED, False):
            return dc_forward_conv1d(m, x)
        else:
            return getattr(m, DC_ORIGINAL_FORWARD)(x)

    m.forward = patched


def unpatch_conv1d(m: nn.Conv1d) -> None:
    if hasattr(m, DC_ORIGINAL_FORWARD):
        m.forward = getattr(m, DC_ORIGINAL_FORWARD)
        for a in [DC_ORIGINAL_FORWARD, DC_ENABLED, DC_IS_OUTPUT_LAYER, DC_BETA]:
            if hasattr(m, a): delattr(m, a)
