"""DC Decomposition for ConvTranspose layers. Forward/Backward: [4*batch] -> [4*batch]"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple

from .base import DC_IS_OUTPUT_LAYER
from .patch_builder import (
    ForwardBuilder, BackwardBuilder, get_cache_info,
    create_patch_function, create_unpatch_function,
)


class DCConvTranspose2dFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_4: Tensor, weight: Tensor, bias: Optional[Tensor],
                stride, padding, output_padding, dilation, groups,
                is_output_layer: bool, cache, layer_name, alpha: float) -> Tensor:
        
        def compute(ctx, pos, neg, weight, bias, stride, padding, output_padding, dilation, groups):
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
            ctx.input_shape = pos.shape

            return out_pos, out_neg

        return ForwardBuilder.run(
            ctx, input_4, compute, is_output_layer, cache, layer_name, alpha,
            extra_args=(weight, bias, stride, padding, output_padding, dilation, groups)
        )

    @staticmethod
    def backward(ctx, grad_4: Tensor):
        def compute(ctx, delta_pp, delta_np, delta_pn, delta_nn):
            weight_pos, weight_neg = ctx.saved_tensors

            # Backward of conv_transpose is conv
            def conv(d, w):
                return F.conv2d(d, w, None, ctx.stride, ctx.padding, ctx.dilation, ctx.groups)

            new_pp = conv(delta_pp, weight_pos) + conv(delta_np, weight_neg)
            new_np = conv(delta_pp, weight_neg) + conv(delta_np, weight_pos)
            new_pn = conv(delta_pn, weight_pos) + conv(delta_nn, weight_neg)
            new_nn = conv(delta_pn, weight_neg) + conv(delta_nn, weight_pos)

            return new_pp, new_np, new_pn, new_nn

        return BackwardBuilder.run(ctx, grad_4, compute, num_extra_returns=11)


class DCConvTranspose1dFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_4: Tensor, weight: Tensor, bias: Optional[Tensor],
                stride, padding, output_padding, dilation, groups,
                is_output_layer: bool, cache, layer_name, alpha: float) -> Tensor:
        
        def compute(ctx, pos, neg, weight, bias, stride, padding, output_padding, dilation, groups):
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
            ctx.input_shape = pos.shape

            return out_pos, out_neg

        return ForwardBuilder.run(
            ctx, input_4, compute, is_output_layer, cache, layer_name, alpha,
            extra_args=(weight, bias, stride, padding, output_padding, dilation, groups)
        )

    @staticmethod
    def backward(ctx, grad_4: Tensor):
        def compute(ctx, delta_pp, delta_np, delta_pn, delta_nn):
            weight_pos, weight_neg = ctx.saved_tensors

            def conv(d, w):
                return F.conv1d(d, w, None, ctx.stride, ctx.padding, ctx.dilation, ctx.groups)

            new_pp = conv(delta_pp, weight_pos) + conv(delta_np, weight_neg)
            new_np = conv(delta_pp, weight_neg) + conv(delta_np, weight_pos)
            new_pn = conv(delta_pn, weight_pos) + conv(delta_nn, weight_neg)
            new_nn = conv(delta_pn, weight_neg) + conv(delta_nn, weight_pos)

            return new_pp, new_np, new_pn, new_nn

        return BackwardBuilder.run(ctx, grad_4, compute, num_extra_returns=11)


def dc_forward_conv_transpose2d(m: nn.ConvTranspose2d, x: Tensor) -> Tensor:
    stride = m.stride if isinstance(m.stride, tuple) else (m.stride, m.stride)
    padding = m.padding if isinstance(m.padding, tuple) else (m.padding, m.padding)
    output_padding = m.output_padding if isinstance(m.output_padding, tuple) else (m.output_padding, m.output_padding)
    dilation = m.dilation if isinstance(m.dilation, tuple) else (m.dilation, m.dilation)
    cache, layer_name, alpha = get_cache_info(m)
    return DCConvTranspose2dFunction.apply(
        x, m.weight, m.bias, stride, padding, output_padding, dilation, m.groups,
        getattr(m, DC_IS_OUTPUT_LAYER, False),
        cache, layer_name, alpha,
    )


def dc_forward_conv_transpose1d(m: nn.ConvTranspose1d, x: Tensor) -> Tensor:
    cache, layer_name, alpha = get_cache_info(m)
    return DCConvTranspose1dFunction.apply(
        x, m.weight, m.bias, m.stride[0], m.padding[0], m.output_padding[0], m.dilation[0], m.groups,
        getattr(m, DC_IS_OUTPUT_LAYER, False),
        cache, layer_name, alpha,
    )


patch_conv_transpose2d = create_patch_function(dc_forward_conv_transpose2d)
patch_conv_transpose1d = create_patch_function(dc_forward_conv_transpose1d)
unpatch_conv_transpose2d = create_unpatch_function()
unpatch_conv_transpose1d = create_unpatch_function()
