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
from typing import Tuple

from .base import (
    split_input_4, make_input_4, split_grad_4, make_grad_4, recenter_dc,
    DC_ENABLED, DC_ORIGINAL_FORWARD
)


class DCAddFunction(torch.autograd.Function):
    """
    Addition with automatic re-centering for DC format.

    Forward: adds two [4*batch] tensors and re-centers the result.
    Backward: distributes gradients to both inputs.

    Note: recenter preserves z = pos - neg, so gradient reconstruction
    (g = pp - np - pn + nn) works correctly with unchanged gradients.
    """

    @staticmethod
    def forward(ctx, x_4: Tensor, y_4: Tensor, recenter: bool) -> Tensor:
        sum_4 = x_4 + y_4

        if recenter:
            result = recenter_dc(sum_4)
        else:
            result = sum_4

        return result

    @staticmethod
    def backward(ctx, grad_4: Tensor) -> Tuple[Tensor, Tensor, None]:
        # Addition distributes gradients to both inputs unchanged.
        # Recenter preserves z = pos - neg, so gradient reconstruction works.
        return grad_4, grad_4, None


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
    """DC-aware forward for Add module."""
    return DCAddFunction.apply(x, y, module.recenter)


def patch_add(module: Add) -> None:
    """Patch Add module for DC decomposition."""
    if hasattr(module, DC_ORIGINAL_FORWARD):
        return
    setattr(module, DC_ORIGINAL_FORWARD, module.forward)
    setattr(module, DC_ENABLED, True)

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
    return DCAddFunction.apply(x, y, recenter)
