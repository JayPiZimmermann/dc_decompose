"""DC Decomposition for Conv1d and Conv2d layers. Forward/Backward: [4*batch] -> [4*batch]"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

from .base import DC_IS_OUTPUT_LAYER
from .patch_builder import (
    ForwardBuilder, BackwardBuilder, get_cache_info,
    create_patch_function, create_unpatch_function,
)


# =============================================================================
# Conv1d
# =============================================================================

class DCConv1dFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_4: Tensor, weight: Tensor, bias: Optional[Tensor],
                stride, padding, dilation, groups,
                is_output_layer: bool, cache, layer_name) -> Tensor:
        fb = ForwardBuilder(ctx, is_output_layer, cache, layer_name)
        pos, neg = fb.split_input(input_4)

        weight_pos = F.relu(weight)
        weight_neg = F.relu(-weight)

        out_pos = F.conv1d(pos, weight_pos, None, stride, padding, dilation, groups)
        out_pos = out_pos + F.conv1d(neg, weight_neg, None, stride, padding, dilation, groups)
        out_neg = F.conv1d(pos, weight_neg, None, stride, padding, dilation, groups)
        out_neg = out_neg + F.conv1d(neg, weight_pos, None, stride, padding, dilation, groups)

        if bias is not None:
            out_pos = out_pos + F.relu(bias).view(1, -1, 1)
            out_neg = out_neg + F.relu(-bias).view(1, -1, 1)

        ctx.save_for_backward(weight)
        ctx.stride, ctx.padding, ctx.dilation, ctx.groups = stride, padding, dilation, groups
        ctx.input_shape = pos.shape

        return fb.build_output(out_pos, out_neg)

    @staticmethod
    def backward(ctx, grad_4: Tensor):
        def compute(ctx, delta_pp, delta_np, delta_pn, delta_nn):
            weight, = ctx.saved_tensors
            weight_pos = F.relu(weight)
            weight_neg = F.relu(-weight)

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

            return new_pp, new_np, new_pn, new_nn

        return BackwardBuilder.run(ctx, grad_4, compute, num_extra_returns=9)


def dc_forward_conv1d(m: nn.Conv1d, x: Tensor) -> Tensor:
    cache, layer_name = get_cache_info(m)
    return DCConv1dFunction.apply(
        x, m.weight, m.bias,
        m.stride[0], m.padding[0], m.dilation[0], m.groups,
        getattr(m, DC_IS_OUTPUT_LAYER, False),
        cache, layer_name,
    )


patch_conv1d = create_patch_function(dc_forward_conv1d)
unpatch_conv1d = create_unpatch_function()


# =============================================================================
# Conv2d
# =============================================================================

class DCConv2dFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_4: Tensor, weight: Tensor, bias: Optional[Tensor],
                stride, padding, dilation, groups,
                is_output_layer: bool, cache, layer_name) -> Tensor:
        fb = ForwardBuilder(ctx, is_output_layer, cache, layer_name)
        pos, neg = fb.split_input(input_4)

        weight_pos = F.relu(weight)
        weight_neg = F.relu(-weight)

        out_pos = F.conv2d(pos, weight_pos, None, stride, padding, dilation, groups)
        out_pos = out_pos + F.conv2d(neg, weight_neg, None, stride, padding, dilation, groups)
        out_neg = F.conv2d(pos, weight_neg, None, stride, padding, dilation, groups)
        out_neg = out_neg + F.conv2d(neg, weight_pos, None, stride, padding, dilation, groups)

        if bias is not None:
            out_pos = out_pos + F.relu(bias).view(1, -1, 1, 1)
            out_neg = out_neg + F.relu(-bias).view(1, -1, 1, 1)

        ctx.save_for_backward(weight)
        ctx.stride, ctx.padding, ctx.dilation, ctx.groups = stride, padding, dilation, groups
        ctx.input_shape = pos.shape

        return fb.build_output(out_pos, out_neg)

    @staticmethod
    def backward(ctx, grad_4: Tensor):
        def compute(ctx, delta_pp, delta_np, delta_pn, delta_nn):
            weight, = ctx.saved_tensors
            weight_pos = F.relu(weight)
            weight_neg = F.relu(-weight)

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

            return new_pp, new_np, new_pn, new_nn

        return BackwardBuilder.run(ctx, grad_4, compute, num_extra_returns=9)


def dc_forward_conv2d(m: nn.Conv2d, x: Tensor) -> Tensor:
    cache, layer_name = get_cache_info(m)
    return DCConv2dFunction.apply(
        x, m.weight, m.bias,
        m.stride, m.padding, m.dilation, m.groups,
        getattr(m, DC_IS_OUTPUT_LAYER, False),
        cache, layer_name,
    )


patch_conv2d = create_patch_function(dc_forward_conv2d)
unpatch_conv2d = create_unpatch_function()
