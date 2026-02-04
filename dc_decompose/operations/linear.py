"""
DC Decomposition for Linear layers.

Forward: [4*batch] -> [4*batch]
- Input: [pos; neg; pos; neg]
- Output: [out_pos; out_neg; out_pos; out_neg]

Backward: [4*batch] -> [4*batch]
- Output layer: interprets first 2 quarters as [grad_pos; grad_neg], initializes 4 sensitivities
- Other layers: receives [delta_pp; delta_np; delta_pn; delta_nn]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple

from .base import (
    split_input_4, make_output_4, split_grad_4, make_grad_4, get_batch_size,
    DC_ENABLED, DC_ORIGINAL_FORWARD, DC_IS_OUTPUT_LAYER, DC_BETA
)


class DCLinearFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_4: Tensor, weight: Tensor, bias: Optional[Tensor],
                is_output_layer: bool, beta: float) -> Tensor:
        # Extract pos, neg from first 2 quarters
        pos, neg = split_input_4(input_4)

        weight_pos = F.relu(weight)
        weight_neg = F.relu(-weight)

        out_pos = F.linear(pos, weight_pos) + F.linear(neg, weight_neg)
        out_neg = F.linear(pos, weight_neg) + F.linear(neg, weight_pos)

        if bias is not None:
            bias_pos = F.relu(bias)
            bias_neg = F.relu(-bias)
            out_pos = out_pos + bias_pos
            out_neg = out_neg + bias_neg

        ctx.save_for_backward(weight_pos, weight_neg)
        ctx.is_output_layer = is_output_layer
        ctx.beta = beta
        ctx.batch_size = pos.shape[0]

        return make_output_4(out_pos, out_neg)

    @staticmethod
    def backward(ctx, grad_4: Tensor) -> Tuple[Tensor, None, None, None, None]:
        weight_pos, weight_neg = ctx.saved_tensors
        batch = ctx.batch_size

        if ctx.is_output_layer:
            # Output layer: interpret first 2 quarters as [grad_pos; grad_neg]
            q = grad_4.shape[0] // 4
            grad_pos = grad_4[:q]
            grad_neg = grad_4[q:2*q]

            delta_pp = ctx.beta * grad_pos
            delta_np = torch.zeros_like(grad_pos)
            delta_pn = (1 - ctx.beta) * grad_neg
            delta_nn = torch.zeros_like(grad_neg)
        else:
            delta_pp, delta_np, delta_pn, delta_nn = split_grad_4(grad_4)

        new_pp = F.linear(delta_pp, weight_pos.t()) + F.linear(delta_np, weight_neg.t())
        new_np = F.linear(delta_pp, weight_neg.t()) + F.linear(delta_np, weight_pos.t())
        new_pn = F.linear(delta_pn, weight_pos.t()) + F.linear(delta_nn, weight_neg.t())
        new_nn = F.linear(delta_pn, weight_neg.t()) + F.linear(delta_nn, weight_pos.t())

        return make_grad_4(new_pp, new_np, new_pn, new_nn), None, None, None, None


def dc_forward_linear(module: nn.Linear, input_4: Tensor) -> Tensor:
    return DCLinearFunction.apply(
        input_4, module.weight, module.bias,
        getattr(module, DC_IS_OUTPUT_LAYER, False),
        getattr(module, DC_BETA, 1.0)
    )


def patch_linear(module: nn.Linear) -> None:
    if hasattr(module, DC_ORIGINAL_FORWARD):
        return
    setattr(module, DC_ORIGINAL_FORWARD, module.forward)
    setattr(module, DC_ENABLED, True)
    setattr(module, DC_IS_OUTPUT_LAYER, False)
    setattr(module, DC_BETA, 1.0)

    def patched(x):
        if getattr(module, DC_ENABLED, False):
            return dc_forward_linear(module, x)
        else:
            return getattr(module, DC_ORIGINAL_FORWARD)(x)

    module.forward = patched


def unpatch_linear(module: nn.Linear) -> None:
    if hasattr(module, DC_ORIGINAL_FORWARD):
        module.forward = getattr(module, DC_ORIGINAL_FORWARD)
        for a in [DC_ORIGINAL_FORWARD, DC_ENABLED, DC_IS_OUTPUT_LAYER, DC_BETA]:
            if hasattr(module, a):
                delattr(module, a)
