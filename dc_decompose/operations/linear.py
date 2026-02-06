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
from typing import Optional

from .base import DC_IS_OUTPUT_LAYER
from .patch_builder import (
    ForwardBuilder, BackwardBuilder, get_cache_info,
    create_patch_function, create_unpatch_function,
)


class DCLinearFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_4: Tensor, weight: Tensor, bias: Optional[Tensor],
                is_output_layer: bool, cache, layer_name, alpha: float) -> Tensor:
        fb = ForwardBuilder(ctx, is_output_layer, cache, layer_name, alpha)
        pos, neg = fb.split_input(input_4)

        # Split weights into positive and negative parts
        weight_pos = F.relu(weight)
        weight_neg = F.relu(-weight)

        # DC forward: track positive and negative weight contributions separately
        out_pos = F.linear(pos, weight_pos) + F.linear(neg, weight_neg)
        out_neg = F.linear(pos, weight_neg) + F.linear(neg, weight_pos)

        if bias is not None:
            bias_pos = F.relu(bias)
            bias_neg = F.relu(-bias)
            out_pos = out_pos + bias_pos
            out_neg = out_neg + bias_neg

        ctx.save_for_backward(weight)
        return fb.build_output(out_pos, out_neg)

    @staticmethod
    def backward(ctx, grad_4: Tensor):
        def compute(ctx, delta_pp, delta_np, delta_pn, delta_nn):
            weight, = ctx.saved_tensors
            weight_pos = F.relu(weight)
            weight_neg = F.relu(-weight)

            new_pp = F.linear(delta_pp, weight_pos.t()) + F.linear(delta_np, weight_neg.t())
            new_np = F.linear(delta_pp, weight_neg.t()) + F.linear(delta_np, weight_pos.t())
            new_pn = F.linear(delta_pn, weight_pos.t()) + F.linear(delta_nn, weight_neg.t())
            new_nn = F.linear(delta_pn, weight_neg.t()) + F.linear(delta_nn, weight_pos.t())

            return new_pp, new_np, new_pn, new_nn

        return BackwardBuilder.run(ctx, grad_4, compute, num_extra_returns=6)


def dc_forward_linear(module: nn.Linear, input_4: Tensor) -> Tensor:
    cache, layer_name, alpha = get_cache_info(module)
    return DCLinearFunction.apply(
        input_4, module.weight, module.bias,
        getattr(module, DC_IS_OUTPUT_LAYER, False),
        cache, layer_name, alpha,
    )


patch_linear = create_patch_function(dc_forward_linear)
unpatch_linear = create_unpatch_function()
