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

from .base import (
    recenter_dc,
    DC_ENABLED, DC_ORIGINAL_FORWARD, DC_IS_OUTPUT_LAYER, DC_BETA
)


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


def dc_forward_add(module: Add, x: Tensor, y: Tensor) -> Tensor:
    """DC-aware forward for Add module.

    Simply adds the two tensors and optionally re-centers.
    The recenter_dc function handles gradient flow correctly.
    """
    result = x + y
    if module.recenter:
        result = recenter_dc(result)
    return result


def patch_add(module: Add) -> None:
    """Patch Add module for DC decomposition."""
    if hasattr(module, DC_ORIGINAL_FORWARD):
        return
    setattr(module, DC_ORIGINAL_FORWARD, module.forward)
    setattr(module, DC_ENABLED, True)
    setattr(module, DC_IS_OUTPUT_LAYER, False)
    setattr(module, DC_BETA, 1.0)

    def patched(x, y):
        if getattr(module, DC_ENABLED, False):
            return dc_forward_add(module, x, y)
        else:
            return getattr(module, DC_ORIGINAL_FORWARD)(x, y)

    module.forward = patched


def unpatch_add(module: Add) -> None:
    """Unpatch Add module, restoring original forward."""
    if hasattr(module, DC_ORIGINAL_FORWARD):
        module.forward = getattr(module, DC_ORIGINAL_FORWARD)
        for attr in [DC_ORIGINAL_FORWARD, DC_ENABLED]:
            if hasattr(module, attr):
                delattr(module, attr)


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
