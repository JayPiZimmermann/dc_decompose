"""DC Decomposition for BatchNorm layers. Forward/Backward: [4*batch] -> [4*batch]"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Union

from .base import DC_IS_OUTPUT_LAYER
from .patch_builder import (
    ForwardBuilder, BackwardBuilder, get_cache_info,
    create_patch_function, create_unpatch_function,
)


class DCBatchNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_4: Tensor, weight: Tensor, bias: Tensor,
                running_mean: Tensor, running_var: Tensor, eps: float,
                momentum: float, is_training: bool, is_2d: bool,
                is_output_layer: bool, cache, layer_name, alpha: float) -> Tensor:
        
        def compute(ctx, pos, neg, weight, bias, running_mean, running_var, eps, momentum, is_training, is_2d):
            with torch.no_grad():
                if is_training:
                    # Training mode: compute batch statistics
                    if is_2d:
                        dims = (0, 2, 3)
                    else:
                        if pos.dim() == 3:
                            dims = (0, 2)
                        else:
                            dims = (0,)

                    batch_mean = pos.mean(dim=dims, keepdim=False)
                    batch_var = pos.var(dim=dims, unbiased=False, keepdim=False)

                    with torch.no_grad():
                        running_mean.mul_(1 - momentum).add_(batch_mean, alpha=momentum)
                        running_var.mul_(1 - momentum).add_(batch_var, alpha=momentum)

                    mean_to_use = batch_mean
                    var_to_use = batch_var
                else:
                    mean_to_use = running_mean
                    var_to_use = running_var

                scale = weight / torch.sqrt(var_to_use + eps)
                bias_eff = bias - scale * mean_to_use

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
                if pos.dim() == 3:
                    scale_pos = scale_pos.view(1, -1, 1)
                    scale_neg = scale_neg.view(1, -1, 1)
                    bias_pos = bias_pos.view(1, -1, 1)
                    bias_neg = bias_neg.view(1, -1, 1)
                else:
                    scale_pos = scale_pos.view(1, -1)
                    scale_neg = scale_neg.view(1, -1)
                    bias_pos = bias_pos.view(1, -1)
                    bias_neg = bias_neg.view(1, -1)

            out_pos = scale_pos * pos + scale_neg * neg + bias_pos
            out_neg = scale_neg * pos + scale_pos * neg + bias_neg

            ctx.save_for_backward(weight, bias, mean_to_use, var_to_use)
            ctx.eps = eps
            ctx.is_2d = is_2d
            ctx.pos_dim = pos.dim()

            return out_pos, out_neg

        return ForwardBuilder.run(
            ctx, input_4, compute, is_output_layer, cache, layer_name, alpha,
            extra_args=(weight, bias, running_mean, running_var, eps, momentum, is_training, is_2d)
        )

    @staticmethod
    def backward(ctx, grad_4: Tensor):
        def compute(ctx, delta_pp, delta_np, delta_pn, delta_nn):
            weight, bias, mean_to_use, var_to_use = ctx.saved_tensors

            with torch.no_grad():
                scale = weight / torch.sqrt(var_to_use + ctx.eps)

            scale_pos = F.relu(scale)
            scale_neg = F.relu(-scale)

            if ctx.is_2d:
                scale_pos = scale_pos.view(1, -1, 1, 1)
                scale_neg = scale_neg.view(1, -1, 1, 1)
            else:
                if ctx.pos_dim == 3:
                    scale_pos = scale_pos.view(1, -1, 1)
                    scale_neg = scale_neg.view(1, -1, 1)
                else:
                    scale_pos = scale_pos.view(1, -1)
                    scale_neg = scale_neg.view(1, -1)

            new_pp = scale_pos * delta_pp + scale_neg * delta_np
            new_np = scale_neg * delta_pp + scale_pos * delta_np
            new_pn = scale_pos * delta_pn + scale_neg * delta_nn
            new_nn = scale_neg * delta_pn + scale_pos * delta_nn

            return new_pp, new_np, new_pn, new_nn

        return BackwardBuilder.run(ctx, grad_4, compute, num_extra_returns=12)


def dc_forward_batchnorm(module: Union[nn.BatchNorm1d, nn.BatchNorm2d], x: Tensor) -> Tensor:
    is_2d = isinstance(module, nn.BatchNorm2d)
    cache, layer_name, alpha = get_cache_info(module)

    return DCBatchNormFunction.apply(
        x, module.weight, module.bias,
        module.running_mean, module.running_var, module.eps, module.momentum,
        module.training, is_2d,
        getattr(module, DC_IS_OUTPUT_LAYER, False),
        cache, layer_name, alpha,
    )


patch_batchnorm = create_patch_function(dc_forward_batchnorm)
unpatch_batchnorm = create_unpatch_function()
