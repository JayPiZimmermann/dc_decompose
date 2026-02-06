"""DC Decomposition for Softmax. Forward/Backward: [4*batch] -> [4*batch]

Softmax: s = exp(z) / sum(exp(z))
For DC: compute softmax on z = pos - neg, cache z and s for backward.
Output is always positive, so: out_pos = s, out_neg = 0
Backward: Jacobian J = diag(s) - s @ s^T, apply to each sensitivity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple

from .base import make_output_4, DC_IS_OUTPUT_LAYER
from .patch_builder import (
    ForwardBuilder, BackwardBuilder, get_cache_info,
    create_patch_function, create_unpatch_function,
)


class DCSoftmaxFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_4: Tensor, dim: int,
                is_output_layer: bool, cache, layer_name) -> Tensor:
        fb = ForwardBuilder(ctx, is_output_layer, cache, layer_name)
        pos, neg = fb.split_input(input_4)
        z = pos - neg

        # Compute softmax on z
        s = F.softmax(z, dim=dim)

        # Softmax output is always positive, so out_pos = s, out_neg = 0
        out_pos = s
        out_neg = torch.zeros_like(s)

        # Cache for backward
        ctx.save_for_backward(s)
        ctx.dim = dim

        return make_output_4(out_pos, out_neg)

    @staticmethod
    def backward(ctx, grad_4: Tensor) -> Tuple[Tensor, None, None, None, None]:
        def compute(ctx, delta_pp, delta_np, delta_pn, delta_nn):
            s, = ctx.saved_tensors
            dim = ctx.dim

            def softmax_backward(delta):
                # Jacobian of softmax: J = diag(s) - s @ s^T
                # J @ delta = s * delta - s * sum(s * delta, dim=dim)
                # This is the standard softmax backward formula
                sum_term = (s * delta).sum(dim=dim, keepdim=True)
                return s * delta - s * sum_term

            # Gradient through softmax for each sensitivity path
            new_pp = softmax_backward(delta_pp)
            new_np = softmax_backward(delta_np)
            new_pn = softmax_backward(delta_pn)
            new_nn = softmax_backward(delta_nn)

            return new_pp, new_np, new_pn, new_nn

        return BackwardBuilder.run(ctx, grad_4, compute, num_extra_returns=4)


def dc_forward_softmax(module: nn.Softmax, x: Tensor) -> Tensor:
    cache, layer_name = get_cache_info(module)
    return DCSoftmaxFunction.apply(
        x, module.dim,
        getattr(module, DC_IS_OUTPUT_LAYER, False),
        cache, layer_name,
    )


patch_softmax = create_patch_function(dc_forward_softmax)
unpatch_softmax = create_unpatch_function()
