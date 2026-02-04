"""DC Decomposition for BatchNorm layers. Forward/Backward: [4*batch] -> [4*batch]"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Union

from .base import split_input_4, make_output_4, split_grad_4, make_grad_4, DC_ENABLED, DC_ORIGINAL_FORWARD, DC_IS_OUTPUT_LAYER, DC_BETA


class DCBatchNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_4: Tensor, running_mean: Tensor, running_var: Tensor,
                weight: Tensor, bias: Tensor, eps: float, is_2d: bool,
                is_output_layer: bool, beta: float) -> Tensor:
        pos, neg = split_input_4(input_4)

        with torch.no_grad():
            scale = weight / torch.sqrt(running_var + eps)
            bias_eff = bias - scale * running_mean

        scale_pos = F.relu(scale)
        scale_neg = F.relu(-scale)
        bias_pos = F.relu(bias_eff)
        bias_neg = F.relu(-bias_eff)

        if is_2d:
            scale_pos = scale_pos.view(1, -1, 1, 1)
            scale_neg = scale_neg.view(1, -1, 1, 1)
            bias_pos = bias_pos.view(1, -1, 1, 1)
            bias_neg = bias_neg.view(1, -1, 1, 1)
        else:
            scale_pos = scale_pos.view(1, -1)
            scale_neg = scale_neg.view(1, -1)
            bias_pos = bias_pos.view(1, -1)
            bias_neg = bias_neg.view(1, -1)

        out_pos = scale_pos * pos + scale_neg * neg + bias_pos
        out_neg = scale_neg * pos + scale_pos * neg + bias_neg

        ctx.save_for_backward(scale_pos, scale_neg)
        ctx.is_output_layer = is_output_layer
        ctx.beta = beta
        ctx.batch_size = pos.shape[0]

        return make_output_4(out_pos, out_neg)

    @staticmethod
    def backward(ctx, grad_4: Tensor) -> Tuple[Tensor, None, None, None, None, None, None, None, None]:
        scale_pos, scale_neg = ctx.saved_tensors

        if ctx.is_output_layer:
            q = grad_4.shape[0] // 4
            gp, gn = grad_4[:q], grad_4[q:2*q]
            delta_pp, delta_np = ctx.beta * gp, torch.zeros_like(gp)
            delta_pn, delta_nn = (1 - ctx.beta) * gn, torch.zeros_like(gn)
        else:
            delta_pp, delta_np, delta_pn, delta_nn = split_grad_4(grad_4)

        new_pp = scale_pos * delta_pp + scale_neg * delta_np
        new_np = scale_neg * delta_pp + scale_pos * delta_np
        new_pn = scale_pos * delta_pn + scale_neg * delta_nn
        new_nn = scale_neg * delta_pn + scale_pos * delta_nn

        return make_grad_4(new_pp, new_np, new_pn, new_nn), None, None, None, None, None, None, None, None


def dc_forward_batchnorm(module: Union[nn.BatchNorm1d, nn.BatchNorm2d], x: Tensor) -> Tensor:
    is_2d = isinstance(module, nn.BatchNorm2d)
    return DCBatchNormFunction.apply(
        x, module.running_mean, module.running_var,
        module.weight, module.bias, module.eps, is_2d,
        getattr(module, DC_IS_OUTPUT_LAYER, False), getattr(module, DC_BETA, 1.0)
    )


def patch_batchnorm(module: Union[nn.BatchNorm1d, nn.BatchNorm2d]) -> None:
    if hasattr(module, DC_ORIGINAL_FORWARD): return
    setattr(module, DC_ORIGINAL_FORWARD, module.forward)
    setattr(module, DC_ENABLED, True)
    setattr(module, DC_IS_OUTPUT_LAYER, False)
    setattr(module, DC_BETA, 1.0)

    def patched(x):
        if getattr(module, DC_ENABLED, False):
            return dc_forward_batchnorm(module, x)
        else:
            return getattr(module, DC_ORIGINAL_FORWARD)(x)

    module.forward = patched


def unpatch_batchnorm(module: Union[nn.BatchNorm1d, nn.BatchNorm2d]) -> None:
    if hasattr(module, DC_ORIGINAL_FORWARD):
        module.forward = getattr(module, DC_ORIGINAL_FORWARD)
        for a in [DC_ORIGINAL_FORWARD, DC_ENABLED, DC_IS_OUTPUT_LAYER, DC_BETA]:
            if hasattr(module, a): delattr(module, a)
