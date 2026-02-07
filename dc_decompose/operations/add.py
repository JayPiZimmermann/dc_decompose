"""
DC Decomposition for Addition (residual connections).

When adding two DC-format tensors, magnitude can grow exponentially.
This module provides re-centering to prevent numerical precision loss.

The Add module is designed to be inserted by functional_replacer.py when it
detects `+` or `torch.add` operations, then patched by patch_model().
"""

import torch
import torch.nn as nn
from torch import Tensor

from .base import recenter_dc, DC_IS_OUTPUT_LAYER
from .patch_builder import create_patch_function, create_unpatch_function, ForwardBuilder, BackwardBuilder, get_cache_info


class Add(nn.Module):
    """
    Simple addition module that can be patched for DC decomposition.

    This module is inserted by functional_replacer.py to replace `+` and
    `torch.add` operations. When patched, it performs DC-aware addition
    with re-centering to prevent magnitude explosion in residual networks.

    Usage (automatic via functional_replacer):
        model = make_dc_compatible(model)  # Replaces + with Add modules
        patch_model(model)  # Patches Add modules for DC mode
    """

    def __init__(self, recenter: bool = True):
        super().__init__()
        self.recenter = recenter

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return x + y


class DCAddFunction(torch.autograd.Function):
    """Proper DC-aware addition function using ForwardBuilder pattern."""
    
    @staticmethod
    def forward(ctx, x: Tensor, y: Tensor, recenter: bool,
                is_output_layer: bool, cache, layer_name, alpha: float) -> Tensor:
        
        def compute(ctx, x_pos, x_neg, y, recenter):
            from .base import split_input_4
            # Split second input
            y_pos, y_neg = split_input_4(y)
            
            # Add corresponding components
            result_pos = x_pos + y_pos
            result_neg = x_neg + y_neg
            
            return result_pos, result_neg

        return ForwardBuilder.run(
            ctx, x, compute, is_output_layer, cache, layer_name, alpha,
            use_recenter_dc=recenter, recenter=recenter,
            extra_args=(y, recenter)
        )
    
    @staticmethod
    def backward(ctx, grad_4: Tensor) -> tuple:
        def compute(ctx, delta_pp, delta_np, delta_pn, delta_nn):
            # Addition distributes gradients to both inputs identically
            return (delta_pp, delta_np, delta_pn, delta_nn), (delta_pp, delta_np, delta_pn, delta_nn)
        
        return BackwardBuilder.run_multi(ctx, grad_4, compute, num_outputs=2, num_extra_returns=5)


def dc_forward_add(module: Add, x: Tensor, y: Tensor) -> Tensor:
    """DC-aware forward for Add module using proper ForwardBuilder pattern."""
    cache, layer_name, alpha = get_cache_info(module)
    return DCAddFunction.apply(
        x, y, module.recenter,
        getattr(module, DC_IS_OUTPUT_LAYER, False),
        cache, layer_name, alpha
    )


patch_add = create_patch_function(dc_forward_add)
unpatch_add = create_unpatch_function()

# Legacy aliases for backward compatibility
DCAdd = Add
patch_dcadd = patch_add
unpatch_dcadd = unpatch_add


def dc_add(x: Tensor, y: Tensor, recenter: bool = True) -> Tensor:
    """Functional version of DC addition with re-centering."""
    result = x + y
    if recenter:
        result = recenter_dc(result)
    return result
