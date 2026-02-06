"""DC Decomposition for MaxPool layers. Forward/Backward: [4*batch] -> [4*batch]

Uses winner-takes-all: indices from z = pos - neg applied to both streams.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional, TYPE_CHECKING

from .base import DC_IS_OUTPUT_LAYER
from .patch_builder import (
    ForwardBuilder, BackwardBuilder, get_cache_info,
    create_patch_function, create_unpatch_function,
)

if TYPE_CHECKING:
    from ..alignment_cache import AlignmentCache


class DCMaxPool2dFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_4: Tensor, kernel_size, stride, padding,
                is_output_layer: bool, cache: Optional['AlignmentCache'],
                layer_name: Optional[str], alpha: float) -> Tensor:
        fb = ForwardBuilder(ctx, is_output_layer, cache, layer_name, alpha)
        pos, neg = fb.split_input(input_4)

        # Check for cached indices in backward-only mode
        use_cached_indices = False
        if fb.should_use_cached_mask():
            cached_indices = fb.get_cached_maxpool_indices()
            if cached_indices is not None:
                use_cached_indices = True
                indices = cached_indices

        if not use_cached_indices:
            z = pos - neg
            _, indices = F.max_pool2d(z, kernel_size, stride, padding, return_indices=True)

        batch, channels, h_in, w_in = pos.shape
        h_out, w_out = indices.shape[2], indices.shape[3]

        pos_flat = pos.view(batch, channels, -1)
        neg_flat = neg.view(batch, channels, -1)
        indices_flat = indices.view(batch, channels, -1)

        out_pos = torch.gather(pos_flat, 2, indices_flat).view(batch, channels, h_out, w_out)
        out_neg = torch.gather(neg_flat, 2, indices_flat).view(batch, channels, h_out, w_out)

        ctx.save_for_backward(indices)
        ctx.input_shape = pos.shape

        return fb.build_output(out_pos, out_neg, recenter=False)

    @staticmethod
    def backward(ctx, grad_4: Tensor):
        def compute(ctx, delta_pp, delta_np, delta_pn, delta_nn):
            indices, = ctx.saved_tensors
            batch, channels, h_in, w_in = ctx.input_shape
            indices_flat = indices.view(batch, channels, -1)

            def scatter_grad(delta: Tensor) -> Tensor:
                delta_flat = delta.view(batch, channels, -1)
                out_flat = torch.zeros(batch, channels, h_in * w_in, device=delta.device, dtype=delta.dtype)
                out_flat.scatter_(2, indices_flat, delta_flat)
                return out_flat.view(batch, channels, h_in, w_in)

            new_pp = scatter_grad(delta_pp)
            new_np = scatter_grad(delta_np)
            new_pn = scatter_grad(delta_pn)
            new_nn = scatter_grad(delta_nn)

            return new_pp, new_np, new_pn, new_nn

        return BackwardBuilder.run(ctx, grad_4, compute, num_extra_returns=7)


class DCMaxPool1dFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_4: Tensor, kernel_size, stride, padding,
                is_output_layer: bool, cache: Optional['AlignmentCache'],
                layer_name: Optional[str], alpha: float) -> Tensor:
        fb = ForwardBuilder(ctx, is_output_layer, cache, layer_name, alpha)
        pos, neg = fb.split_input(input_4)

        # Check for cached indices in backward-only mode
        use_cached_indices = False
        if fb.should_use_cached_mask():
            cached_indices = fb.get_cached_maxpool_indices()
            if cached_indices is not None:
                use_cached_indices = True
                indices = cached_indices

        if not use_cached_indices:
            z = pos - neg
            _, indices = F.max_pool1d(z, kernel_size, stride, padding, return_indices=True)

        batch, channels, l_in = pos.shape
        l_out = indices.shape[2]

        pos_flat = pos.view(batch, channels, -1)
        neg_flat = neg.view(batch, channels, -1)
        indices_flat = indices.view(batch, channels, -1)

        out_pos = torch.gather(pos_flat, 2, indices_flat).view(batch, channels, l_out)
        out_neg = torch.gather(neg_flat, 2, indices_flat).view(batch, channels, l_out)

        ctx.save_for_backward(indices)
        ctx.input_shape = pos.shape

        return fb.build_output(out_pos, out_neg, recenter=False)

    @staticmethod
    def backward(ctx, grad_4: Tensor):
        def compute(ctx, delta_pp, delta_np, delta_pn, delta_nn):
            indices, = ctx.saved_tensors
            batch, channels, l_in = ctx.input_shape
            indices_flat = indices.view(batch, channels, -1)

            def scatter_grad(delta: Tensor) -> Tensor:
                delta_flat = delta.view(batch, channels, -1)
                out_flat = torch.zeros(batch, channels, l_in, device=delta.device, dtype=delta.dtype)
                out_flat.scatter_(2, indices_flat, delta_flat)
                return out_flat.view(batch, channels, l_in)

            new_pp = scatter_grad(delta_pp)
            new_np = scatter_grad(delta_np)
            new_pn = scatter_grad(delta_pn)
            new_nn = scatter_grad(delta_nn)

            return new_pp, new_np, new_pn, new_nn

        return BackwardBuilder.run(ctx, grad_4, compute, num_extra_returns=7)


def dc_forward_maxpool2d(module: nn.MaxPool2d, x: Tensor) -> Tensor:
    kernel_size = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
    stride = module.stride if isinstance(module.stride, tuple) else (module.stride, module.stride)
    padding = module.padding if isinstance(module.padding, tuple) else (module.padding, module.padding)
    cache, layer_name, alpha = get_cache_info(module)
    return DCMaxPool2dFunction.apply(
        x, kernel_size, stride, padding,
        getattr(module, DC_IS_OUTPUT_LAYER, False),
        cache, layer_name, alpha,
    )


def dc_forward_maxpool1d(module: nn.MaxPool1d, x: Tensor) -> Tensor:
    stride = module.stride if module.stride is not None else module.kernel_size
    cache, layer_name, alpha = get_cache_info(module)
    return DCMaxPool1dFunction.apply(
        x, module.kernel_size, stride, module.padding,
        getattr(module, DC_IS_OUTPUT_LAYER, False),
        cache, layer_name, alpha,
    )


patch_maxpool2d = create_patch_function(dc_forward_maxpool2d)
patch_maxpool1d = create_patch_function(dc_forward_maxpool1d)
unpatch_maxpool2d = create_unpatch_function()
unpatch_maxpool1d = create_unpatch_function()
