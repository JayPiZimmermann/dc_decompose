"""DC Decomposition for ReLU. Forward/Backward: [4*batch] -> [4*batch]"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Type

from .base import (
    split_input_4, make_output_4, make_grad_4,
    init_backward, recenter_forward,
    DC_ENABLED, DC_ORIGINAL_FORWARD, DC_RELU_MODE, DC_IS_OUTPUT_LAYER, DC_BETA
)

DC_BACKPROP_MODE = '_dc_backprop_mode'
DC_RELU_FUNCTION = '_dc_relu_function'


# Forward positive subfunctions
def _forward_pos_max(pos: Tensor, neg: Tensor) -> Tensor:
    """Max mode: forward_pos = max{a, b}"""
    return torch.max(pos, neg)


def _forward_pos_min(pos: Tensor, _neg: Tensor) -> Tensor:
    """Min mode: forward_pos = a"""
    return pos


# Forward negative subfunctions
def _forward_neg_max(_pos: Tensor, neg: Tensor) -> Tensor:
    """Max mode: forward_neg = b"""
    return neg


def _forward_neg_min(pos: Tensor, neg: Tensor) -> Tensor:
    """Min mode: forward_neg = min{a, b}"""
    return torch.min(pos, neg)


def forward_relu(pos: Tensor, neg: Tensor, split_mode: str) -> Tuple[Tensor, Tensor]:
    """
    DC forward for ReLU with mode dispatch.

    Args:
        pos: Positive component (a)
        neg: Negative component (b)
        split_mode: 'max', 'min', or 'half'

    Returns:
        (out_pos, out_neg) tuple
    """
    if split_mode == 'max':
        return _forward_pos_max(pos, neg), _forward_neg_max(pos, neg)
    elif split_mode == 'min':
        return _forward_pos_min(pos, neg), _forward_neg_min(pos, neg)
    elif split_mode == 'half':
        out_pos = (_forward_pos_max(pos, neg) + _forward_pos_min(pos, neg)) / 2
        out_neg = (_forward_neg_max(pos, neg) + _forward_neg_min(pos, neg)) / 2
        return out_pos, out_neg
    else:
        raise ValueError(f"Unknown split_mode: {split_mode}")


def backward_relu(
    delta_pp: Tensor, delta_np: Tensor, delta_pn: Tensor, delta_nn: Tensor,
    mp: Tensor, mn: Tensor, split_mode: str, backprop_mode: str
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    DC backward for ReLU with mode dispatch.

    Args:
        delta_pp, delta_np, delta_pn, delta_nn: Incoming gradients
        mp: Mask where z >= 0 (pos >= neg)
        mn: Mask where z < 0 (pos < neg)
        split_mode: 'max', 'min', or 'half'
        backprop_mode: 'standard', 'mask_diff', or 'sum'

    Returns:
        (new_pp, new_np, new_pn, new_nn) tuple
    """
    if split_mode == 'max':
        return _backprop_max(delta_pp, delta_np, delta_pn, delta_nn, mp, mn, backprop_mode)
    elif split_mode == 'min':
        return _backprop_min(delta_pp, delta_np, delta_pn, delta_nn, mp, mn, backprop_mode)
    elif split_mode == 'half':
        pp_max, np_max, pn_max, nn_max = _backprop_max(
            delta_pp, delta_np, delta_pn, delta_nn, mp, mn, backprop_mode)
        pp_min, np_min, pn_min, nn_min = _backprop_min(
            delta_pp, delta_np, delta_pn, delta_nn, mp, mn, backprop_mode)
        return (pp_max + pp_min) / 2, (np_max + np_min) / 2, (pn_max + pn_min) / 2, (nn_max + nn_min) / 2
    else:
        raise ValueError(f"Unknown split_mode: {split_mode}")


def _make_relu_function(split_mode: str, backprop_mode: str) -> Type[torch.autograd.Function]:
    """Factory that creates a DCReLUFunction class with modes baked in."""

    class DCReLUFunction(torch.autograd.Function):
        """ReLU DC function with split_mode and backprop_mode captured from closure."""

        @staticmethod
        def forward(ctx, input_4: Tensor, is_output_layer: bool, beta: float) -> Tensor:
            pos, neg = split_input_4(input_4)
            z = pos - neg

            out_pos, out_neg = forward_relu(pos, neg, split_mode)

            ctx.save_for_backward(z)
            ctx.is_output_layer, ctx.beta = is_output_layer, beta

            output = make_output_4(out_pos, out_neg)
            return recenter_forward(output)

        @staticmethod
        def backward(ctx, grad_4: Tensor) -> Tuple[Tensor, None, None]:
            z, = ctx.saved_tensors
            mp, mn = (z >= 0).float(), (z < 0).float()

            delta_pp, delta_np, delta_pn, delta_nn = init_backward(
                grad_4, ctx.is_output_layer, ctx.beta)

            # For intermediate layers: reconstruct and re-split for ReLU backward
            if not ctx.is_output_layer:
                g = delta_pp - delta_np - delta_pn + delta_nn
                delta_pp = g
                delta_np = -g * mn
                delta_pn = torch.zeros_like(g)
                delta_nn = torch.zeros_like(g)

            new_pp, new_np, new_pn, new_nn = backward_relu(
                delta_pp, delta_np, delta_pn, delta_nn, mp, mn, split_mode, backprop_mode)

            return make_grad_4(new_pp, new_np, new_pn, new_nn), None, None

    return DCReLUFunction


def _backprop_max(delta_pp, delta_np, delta_pn, delta_nn, mp, mn, backprop_mode):
    """Max mode backward: forward_pos = max{a, b}, forward_neg = b

    Gradient reconstruction: grad = pp - np - pn + nn
    ReLU gradient: grad_out = grad_in * (z >= 0) = grad_in * mp

    Reference formulas from direct_relu_split.py
    """
    if backprop_mode == 'standard':
        new_pp = delta_pp * mp
        new_np = delta_np + delta_pp * mn
        new_pn = delta_pn * mp
        new_nn = delta_nn + delta_pn * mn
    elif backprop_mode == 'mask_diff':
        new_pp = delta_pp * (mp - mn)
        new_np = delta_np
        new_pn = delta_pn * (mp - mn)
        new_nn = delta_nn
    elif backprop_mode == 'sum':
        new_pp = delta_pp * mp + delta_np * mn
        new_np = delta_np
        new_pn = delta_pn * mp + delta_nn * mn
        new_nn = delta_nn
    else:
        raise ValueError(f"Unknown backprop_mode: {backprop_mode}")
    return new_pp, new_np, new_pn, new_nn


def _backprop_min(delta_pp, delta_np, delta_pn, delta_nn, mp, mn, backprop_mode):
    """Min mode backward: forward_pos = a, forward_neg = min{a, b}

    Gradient reconstruction: grad = pp - np - pn + nn
    ReLU gradient: grad_out = grad_in * (z >= 0) = grad_in * mp

    Reference formulas from direct_relu_split.py
    """
    if backprop_mode == 'standard':
        new_pp = delta_pp + delta_np * mn
        new_np = delta_np * mp
        new_pn = delta_pn + delta_nn * mn
        new_nn = delta_nn * mp
    elif backprop_mode == 'mask_diff':
        new_pp = delta_pp
        new_np = delta_np * (mp - mn)
        new_pn = delta_pn
        new_nn = delta_nn * (mp - mn)
    elif backprop_mode == 'sum':
        new_pp = delta_pp
        new_np = delta_np * mp + delta_pp * mn
        new_pn = delta_pn
        new_nn = delta_nn * mp + delta_pn * mn
    else:
        raise ValueError(f"Unknown backprop_mode: {backprop_mode}")
    return new_pp, new_np, new_pn, new_nn


def dc_forward_relu(m: nn.ReLU, x: Tensor) -> Tensor:
    func = getattr(m, DC_RELU_FUNCTION)
    return func.apply(
        x,
        getattr(m, DC_IS_OUTPUT_LAYER, False),
        getattr(m, DC_BETA, 1.0)
    )


def patch_relu(m: nn.ReLU, split_mode: str = 'max', backprop_mode: str = 'standard') -> None:
    if hasattr(m, DC_ORIGINAL_FORWARD): return
    setattr(m, DC_ORIGINAL_FORWARD, m.forward)
    setattr(m, DC_ENABLED, True)
    setattr(m, DC_RELU_MODE, split_mode)
    setattr(m, DC_BACKPROP_MODE, backprop_mode)
    setattr(m, DC_RELU_FUNCTION, _make_relu_function(split_mode, backprop_mode))
    setattr(m, DC_IS_OUTPUT_LAYER, False)
    setattr(m, DC_BETA, 1.0)

    def patched(x):
        if getattr(m, DC_ENABLED, False):
            return dc_forward_relu(m, x)
        else:
            return getattr(m, DC_ORIGINAL_FORWARD)(x)

    m.forward = patched


def unpatch_relu(m: nn.ReLU) -> None:
    if hasattr(m, DC_ORIGINAL_FORWARD):
        m.forward = getattr(m, DC_ORIGINAL_FORWARD)
        for a in [DC_ORIGINAL_FORWARD, DC_ENABLED, DC_RELU_MODE, DC_BACKPROP_MODE, DC_RELU_FUNCTION, DC_IS_OUTPUT_LAYER, DC_BETA]:
            if hasattr(m, a): delattr(m, a)


# Default DCReLUFunction for backward compatibility
DCReLUFunction = _make_relu_function('max', 'standard')
