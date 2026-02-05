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
    split_input_4, make_output_4, make_grad_4,
    init_backward, recenter_forward,
    DC_ENABLED, DC_ORIGINAL_FORWARD, DC_IS_OUTPUT_LAYER
)


class DCLinearFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_4: Tensor, weight: Tensor, bias: Optional[Tensor],
                is_output_layer: bool) -> Tensor:
        # Extract pos, neg from first 2 quarters
        pos, neg = split_input_4(input_4)

        # Split weights on-the-fly (temporary tensors, auto-cleanup)
        weight_pos = F.relu(weight)
        weight_neg = F.relu(-weight)

        out_pos = F.linear(pos, weight_pos) + F.linear(neg, weight_neg)
        out_neg = F.linear(pos, weight_neg) + F.linear(neg, weight_pos)

        if bias is not None:
            bias_pos = F.relu(bias)
            bias_neg = F.relu(-bias)
            out_pos = out_pos + bias_pos
            out_neg = out_neg + bias_neg

        # Always save original weight
        ctx.save_for_backward(weight)
        ctx.is_output_layer = is_output_layer

        output = make_output_4(out_pos, out_neg)
        return recenter_forward(output)

    @staticmethod
    def backward(ctx, grad_4: Tensor) -> Tuple[Tensor, None, None, None]:
        delta_pp, delta_np, delta_pn, delta_nn = init_backward(
            grad_4, ctx.is_output_layer)

        # Always split weights on-the-fly (temporary tensors, auto-cleanup)
        weight, = ctx.saved_tensors
        weight_pos = F.relu(weight)
        weight_neg = F.relu(-weight)

        new_pp = F.linear(delta_pp, weight_pos.t()) + F.linear(delta_np, weight_neg.t())
        new_np = F.linear(delta_pp, weight_neg.t()) + F.linear(delta_np, weight_pos.t())
        new_pn = F.linear(delta_pn, weight_pos.t()) + F.linear(delta_nn, weight_neg.t())
        new_nn = F.linear(delta_pn, weight_neg.t()) + F.linear(delta_nn, weight_pos.t())

        return make_grad_4(new_pp, new_np, new_pn, new_nn), None, None, None


def dc_forward_linear(module: nn.Linear, input_4: Tensor) -> Tensor:
    return DCLinearFunction.apply(
        input_4, module.weight, module.bias,
        getattr(module, DC_IS_OUTPUT_LAYER, False)
    )


def patch_linear(module: nn.Linear) -> None:
    if hasattr(module, DC_ORIGINAL_FORWARD):
        return
    setattr(module, DC_ORIGINAL_FORWARD, module.forward)
    setattr(module, DC_ENABLED, True)
    setattr(module, DC_IS_OUTPUT_LAYER, False)

    def patched(x):
        if getattr(module, DC_ENABLED, False):
            return dc_forward_linear(module, x)
        else:
            return getattr(module, DC_ORIGINAL_FORWARD)(x)

    module.forward = patched


def unpatch_linear(module: nn.Linear) -> None:
    if hasattr(module, DC_ORIGINAL_FORWARD):
        module.forward = getattr(module, DC_ORIGINAL_FORWARD)
        for a in [DC_ORIGINAL_FORWARD, DC_ENABLED, DC_IS_OUTPUT_LAYER]:
            if hasattr(module, a):
                delattr(module, a)
