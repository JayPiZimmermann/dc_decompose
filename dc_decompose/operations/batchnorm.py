"""DC Decomposition for BatchNorm layers. Forward/Backward: [4*batch] -> [4*batch]"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Union

from .base import (
    split_input_4, make_output_4, make_grad_4,
    init_backward, recenter_forward,
    DC_ENABLED, DC_ORIGINAL_FORWARD, DC_IS_OUTPUT_LAYER, DC_BETA, DC_SPLIT_WEIGHTS_ON_FLY
)


class DCBatchNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_4: Tensor, weight: Tensor, bias: Tensor, 
                running_mean: Tensor, running_var: Tensor, eps: float, 
                momentum: float, is_training: bool, is_2d: bool,
                is_output_layer: bool, beta: float, split_on_fly: bool) -> Tensor:
        pos, neg = split_input_4(input_4)

        with torch.no_grad():
            if is_training:
                # Training mode: compute batch statistics
                if is_2d:
                    # For 2D: mean over batch, height, width dims (keep channel dim)
                    dims = (0, 2, 3)
                else:
                    # For 1D: Check if we have 3D [N,C,L] or 2D [N,C] input
                    if pos.dim() == 3:
                        # 3D case: mean over batch and sequence dims (keep channel dim)
                        dims = (0, 2)
                    else:
                        # 2D case: mean over batch dim only (keep channel dim)
                        dims = (0,)
                
                batch_mean = pos.mean(dim=dims, keepdim=False)
                batch_var = pos.var(dim=dims, unbiased=False, keepdim=False)
                
                # Update running statistics (detached)
                with torch.no_grad():
                    running_mean.mul_(1 - momentum).add_(batch_mean, alpha=momentum)
                    running_var.mul_(1 - momentum).add_(batch_var, alpha=momentum)
                
                # Use batch statistics for normalization
                mean_to_use = batch_mean
                var_to_use = batch_var
            else:
                # Eval mode: use running statistics
                mean_to_use = running_mean
                var_to_use = running_var

            scale = weight / torch.sqrt(var_to_use + eps)
            bias_eff = bias - scale * mean_to_use

        # Split scale and bias on-the-fly (will be cleaned up automatically)
        scale_pos = F.relu(scale)
        scale_neg = F.relu(-scale)
        bias_pos = F.relu(bias_eff)
        bias_neg = F.relu(-bias_eff)

        if is_2d:
            scale_pos = scale_pos.view(1, -1, 1, 1)
            scale_neg = scale_neg.view(1, -1, 1, 1)
            bias_pos = bias_pos.view(1, -1, 1, 1)
            bias_neg = bias_neg.view(1, -1, 1, 1)
        else:
            # For 1D: Check if we have 3D [N,C,L] or 2D [N,C] input
            if pos.dim() == 3:
                # 3D case: need to reshape to [1, C, 1] to broadcast with [B, C, L]
                scale_pos = scale_pos.view(1, -1, 1)
                scale_neg = scale_neg.view(1, -1, 1)
                bias_pos = bias_pos.view(1, -1, 1)
                bias_neg = bias_neg.view(1, -1, 1)
            else:
                # 2D case: need to reshape to [1, C] to broadcast with [B, C]
                scale_pos = scale_pos.view(1, -1)
                scale_neg = scale_neg.view(1, -1)
                bias_pos = bias_pos.view(1, -1)
                bias_neg = bias_neg.view(1, -1)

        out_pos = scale_pos * pos + scale_neg * neg + bias_pos
        out_neg = scale_neg * pos + scale_pos * neg + bias_neg

        # Save for backward: only save original weight if split_on_fly=True
        if split_on_fly:
            ctx.save_for_backward(weight, bias, mean_to_use, var_to_use)
        else:
            ctx.save_for_backward(scale_pos, scale_neg)
            
        ctx.eps = eps
        ctx.is_2d = is_2d
        ctx.is_output_layer = is_output_layer
        ctx.beta = beta
        ctx.split_on_fly = split_on_fly
        ctx.pos_dim = pos.dim()  # Save input dimensionality

        output = make_output_4(out_pos, out_neg)
        return recenter_forward(output)

    @staticmethod
    def backward(ctx, grad_4: Tensor) -> Tuple[Tensor, None, None, None, None, None, None, None, None, None, None, None]:
        delta_pp, delta_np, delta_pn, delta_nn = init_backward(
            grad_4, ctx.is_output_layer, ctx.beta)

        if ctx.split_on_fly:
            # Split weights on-the-fly in backward
            weight, bias, mean_to_use, var_to_use = ctx.saved_tensors
            
            with torch.no_grad():
                scale = weight / torch.sqrt(var_to_use + ctx.eps)

            # Split on-the-fly (temporary tensors, auto-cleanup)
            scale_pos = F.relu(scale)
            scale_neg = F.relu(-scale)

            if ctx.is_2d:
                scale_pos = scale_pos.view(1, -1, 1, 1)
                scale_neg = scale_neg.view(1, -1, 1, 1)
            else:
                # For 1D: Check if we have 3D [N,C,L] or 2D [N,C] input
                if ctx.pos_dim == 3:
                    # 3D case: need to reshape to [1, C, 1] to broadcast with [B, C, L]
                    scale_pos = scale_pos.view(1, -1, 1)
                    scale_neg = scale_neg.view(1, -1, 1)
                else:
                    # 2D case: need to reshape to [1, C] to broadcast with [B, C]
                    scale_pos = scale_pos.view(1, -1)
                    scale_neg = scale_neg.view(1, -1)
        else:
            # Use pre-computed split weights
            scale_pos, scale_neg = ctx.saved_tensors

        new_pp = scale_pos * delta_pp + scale_neg * delta_np
        new_np = scale_neg * delta_pp + scale_pos * delta_np
        new_pn = scale_pos * delta_pn + scale_neg * delta_nn
        new_nn = scale_neg * delta_pn + scale_pos * delta_nn

        return make_grad_4(new_pp, new_np, new_pn, new_nn), None, None, None, None, None, None, None, None, None, None, None


def dc_forward_batchnorm(module: Union[nn.BatchNorm1d, nn.BatchNorm2d], x: Tensor) -> Tensor:
    is_2d = isinstance(module, nn.BatchNorm2d)
    split_on_fly = getattr(module, DC_SPLIT_WEIGHTS_ON_FLY, True)
    
    return DCBatchNormFunction.apply(
        x, module.weight, module.bias, 
        module.running_mean, module.running_var, module.eps, module.momentum,
        module.training, is_2d,
        getattr(module, DC_IS_OUTPUT_LAYER, False), getattr(module, DC_BETA, 0.5),
        split_on_fly
    )


def patch_batchnorm(module: Union[nn.BatchNorm1d, nn.BatchNorm2d]) -> None:
    if hasattr(module, DC_ORIGINAL_FORWARD): return
    setattr(module, DC_ORIGINAL_FORWARD, module.forward)
    setattr(module, DC_ENABLED, True)
    setattr(module, DC_IS_OUTPUT_LAYER, False)
    setattr(module, DC_BETA, 0.5)
    setattr(module, DC_SPLIT_WEIGHTS_ON_FLY, True)

    def patched(x):
        if getattr(module, DC_ENABLED, False):
            return dc_forward_batchnorm(module, x)
        else:
            return getattr(module, DC_ORIGINAL_FORWARD)(x)

    module.forward = patched


def unpatch_batchnorm(module: Union[nn.BatchNorm1d, nn.BatchNorm2d]) -> None:
    if hasattr(module, DC_ORIGINAL_FORWARD):
        module.forward = getattr(module, DC_ORIGINAL_FORWARD)
        for a in [DC_ORIGINAL_FORWARD, DC_ENABLED, DC_IS_OUTPUT_LAYER, DC_BETA, DC_SPLIT_WEIGHTS_ON_FLY]:
            if hasattr(module, a): delattr(module, a)
