"""
DC Decomposition for Embedding layers. Forward/Backward: [4*batch] -> [4*batch]

For embedding layers, the same embedding indices are applied to both pos and neg streams.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple

from .base import (
    split_input_4, make_output_4, make_grad_4,
    init_backward, recenter_forward,
    DC_ENABLED, DC_ORIGINAL_FORWARD, DC_IS_OUTPUT_LAYER
)


class DCEmbeddingFunction(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input_4: Tensor, weight: Tensor, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse,
                is_output_layer: bool, beta: float) -> Tensor:
        # For embedding, we typically use the same indices for pos/neg
        # The input_4 contains indices, not pos/neg decomposed values
        pos_indices, neg_indices = split_input_4(input_4)
        
        # Apply embedding to both streams (usually pos_indices == neg_indices for indices)
        out_pos = torch.nn.functional.embedding(pos_indices, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse)
        out_neg = torch.nn.functional.embedding(neg_indices, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse)
        
        ctx.save_for_backward(pos_indices, neg_indices, weight)
        ctx.padding_idx = padding_idx
        ctx.max_norm = max_norm
        ctx.norm_type = norm_type
        ctx.scale_grad_by_freq = scale_grad_by_freq
        ctx.sparse = sparse
        ctx.is_output_layer = is_output_layer
        ctx.beta = beta
        
        output = make_output_4(out_pos, out_neg)
        return recenter_forward(output)
    
    @staticmethod
    def backward(ctx, grad_4: Tensor) -> Tuple[Tensor, ...]:
        pos_indices, neg_indices, weight = ctx.saved_tensors
        
        delta_pp, delta_np, delta_pn, delta_nn = init_backward(
            grad_4, ctx.is_output_layer, ctx.beta)
        
        # For embedding, gradients flow back to the weight matrix
        # Input gradients are typically not needed for embedding indices
        grad_input = None
        
        # Compute weight gradients using embedding_backward
        # This is simplified - in practice, embedding gradients are more complex
        grad_weight = None  # Would need proper embedding gradient computation
        
        return grad_input, grad_weight, None, None, None, None, None, None, None


def dc_forward_embedding(module: nn.Embedding, input: Tensor) -> Tensor:
    """DC forward for embedding layer."""
    return DCEmbeddingFunction.apply(
        input, module.weight, module.padding_idx, module.max_norm, 
        module.norm_type, module.scale_grad_by_freq, module.sparse,
        getattr(module, DC_IS_OUTPUT_LAYER, False),
        0.5
    )


def patch_embedding(module: nn.Embedding) -> None:
    """Patch Embedding module for DC decomposition."""
    if hasattr(module, DC_ORIGINAL_FORWARD):
        return
    setattr(module, DC_ORIGINAL_FORWARD, module.forward)
    setattr(module, DC_ENABLED, True)
    setattr(module, DC_IS_OUTPUT_LAYER, False)
    

    def patched(input):
        if getattr(module, DC_ENABLED, False):
            return dc_forward_embedding(module, input)
        else:
            return getattr(module, DC_ORIGINAL_FORWARD)(input)

    module.forward = patched


def unpatch_embedding(module: nn.Embedding) -> None:
    """Unpatch Embedding module."""
    if hasattr(module, DC_ORIGINAL_FORWARD):
        module.forward = getattr(module, DC_ORIGINAL_FORWARD)
        for attr in [DC_ORIGINAL_FORWARD, DC_ENABLED, DC_IS_OUTPUT_LAYER]:
            if hasattr(module, attr):
                delattr(module, attr)