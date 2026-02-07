"""
DC Decomposition for Embedding layers. Forward/Backward: [4*batch] -> [4*batch]

For embedding layers, the same embedding indices are applied to both pos and neg streams.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple

from .base import DC_IS_OUTPUT_LAYER
from .patch_builder import (
    ForwardBuilder, BackwardBuilder, get_cache_info,
    create_patch_function, create_unpatch_function,
)


class DCEmbeddingFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_4: Tensor, weight: Tensor, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse,
                is_output_layer: bool, cache, layer_name, alpha: float) -> Tensor:
        
        def compute(ctx, pos_indices, neg_indices, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse):
            out_pos = torch.nn.functional.embedding(pos_indices, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse)
            out_neg = torch.nn.functional.embedding(neg_indices, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse)

            ctx.save_for_backward(pos_indices, neg_indices, weight)
            ctx.padding_idx = padding_idx
            ctx.max_norm = max_norm
            ctx.norm_type = norm_type
            ctx.scale_grad_by_freq = scale_grad_by_freq
            ctx.sparse = sparse

            return out_pos, out_neg

        return ForwardBuilder.run(
            ctx, input_4, compute, is_output_layer, cache, layer_name, alpha,
            extra_args=(weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse)
        )

    @staticmethod
    def backward(ctx, grad_4: Tensor) -> Tuple[Tensor, ...]:
        def compute(ctx, delta_pp, delta_np, delta_pn, delta_nn):
            # For embedding, gradients flow back to the weight matrix
            # Input gradients are typically not needed for embedding indices
            return None, None, None, None

        # Note: This is a simplified implementation
        return (None, None, None, None, None, None, None, None, None, None)


def dc_forward_embedding(module: nn.Embedding, input: Tensor) -> Tensor:
    cache, layer_name, alpha = get_cache_info(module)
    return DCEmbeddingFunction.apply(
        input, module.weight, module.padding_idx, module.max_norm,
        module.norm_type, module.scale_grad_by_freq, module.sparse,
        getattr(module, DC_IS_OUTPUT_LAYER, False),
        cache, layer_name, alpha,
    )


patch_embedding = create_patch_function(dc_forward_embedding)
unpatch_embedding = create_unpatch_function()
