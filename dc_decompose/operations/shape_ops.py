"""DC Decomposition for shape operations. Forward/Backward: [4*batch] -> [4*batch]

Shape operations (flatten, reshape, transpose, etc.) are linear operations that apply
the same transformation to both pos and neg streams.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Union, List

from .base import DC_IS_OUTPUT_LAYER
from .patch_builder import (
    ForwardBuilder, BackwardBuilder, get_cache_info,
    create_patch_function, create_unpatch_function,
)


# =============================================================================
# Flatten
# =============================================================================

class DCFlattenFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_4: Tensor, start_dim: int, end_dim: int,
                is_output_layer: bool, cache, layer_name) -> Tensor:
        fb = ForwardBuilder(ctx, is_output_layer, cache, layer_name)
        pos, neg = fb.split_input(input_4)

        ctx.input_shape = pos.shape

        out_pos = torch.flatten(pos, start_dim, end_dim)
        out_neg = torch.flatten(neg, start_dim, end_dim)

        return fb.build_output(out_pos, out_neg, recenter=False)

    @staticmethod
    def backward(ctx, grad_4: Tensor) -> Tuple[Tensor, None, None, None, None, None]:
        def compute(ctx, delta_pp, delta_np, delta_pn, delta_nn):
            new_pp = delta_pp.view(ctx.input_shape)
            new_np = delta_np.view(ctx.input_shape)
            new_pn = delta_pn.view(ctx.input_shape)
            new_nn = delta_nn.view(ctx.input_shape)
            return new_pp, new_np, new_pn, new_nn

        return BackwardBuilder.run(ctx, grad_4, compute, num_extra_returns=5)


def dc_forward_flatten(m: nn.Flatten, x: Tensor) -> Tensor:
    cache, layer_name = get_cache_info(m)
    return DCFlattenFunction.apply(x, m.start_dim, m.end_dim,
                                    getattr(m, DC_IS_OUTPUT_LAYER, False),
                                    cache, layer_name)


patch_flatten = create_patch_function(dc_forward_flatten)
unpatch_flatten = create_unpatch_function()


# =============================================================================
# Unflatten
# =============================================================================

class DCUnflattenFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_4: Tensor, dim: int, unflattened_size: Tuple[int, ...],
                is_output_layer: bool, cache, layer_name) -> Tensor:
        fb = ForwardBuilder(ctx, is_output_layer, cache, layer_name)
        pos, neg = fb.split_input(input_4)

        ctx.input_shape = pos.shape
        ctx.dim = dim

        out_pos = pos.unflatten(dim, unflattened_size)
        out_neg = neg.unflatten(dim, unflattened_size)

        return fb.build_output(out_pos, out_neg, recenter=False)

    @staticmethod
    def backward(ctx, grad_4: Tensor) -> Tuple[Tensor, None, None, None, None, None]:
        def compute(ctx, delta_pp, delta_np, delta_pn, delta_nn):
            # Flatten back to original shape
            end_dim = ctx.dim + len(delta_pp.shape) - len(ctx.input_shape)
            new_pp = delta_pp.flatten(ctx.dim, end_dim)
            new_np = delta_np.flatten(ctx.dim, end_dim)
            new_pn = delta_pn.flatten(ctx.dim, end_dim)
            new_nn = delta_nn.flatten(ctx.dim, end_dim)
            return new_pp, new_np, new_pn, new_nn

        return BackwardBuilder.run(ctx, grad_4, compute, num_extra_returns=5)


def dc_forward_unflatten(m: nn.Unflatten, x: Tensor) -> Tensor:
    cache, layer_name = get_cache_info(m)
    return DCUnflattenFunction.apply(x, m.dim, m.unflattened_size,
                                      getattr(m, DC_IS_OUTPUT_LAYER, False),
                                      cache, layer_name)


patch_unflatten = create_patch_function(dc_forward_unflatten)
unpatch_unflatten = create_unpatch_function()


# =============================================================================
# Reshape Module (for functional torch.reshape replacement)
# =============================================================================

class Reshape(nn.Module):
    """Module wrapper for torch.reshape that can be patched for DC decomposition."""

    def __init__(self, shape: Tuple[int, ...]):
        super().__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        return x.reshape(self.shape)


class DCReshapeFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_4: Tensor, shape: Tuple[int, ...],
                is_output_layer: bool, cache, layer_name) -> Tensor:
        fb = ForwardBuilder(ctx, is_output_layer, cache, layer_name)
        pos, neg = fb.split_input(input_4)

        ctx.input_shape = pos.shape

        # Adjust shape for batch dimension (shape[0] should be -1 or match)
        out_pos = pos.reshape(shape)
        out_neg = neg.reshape(shape)

        return fb.build_output(out_pos, out_neg, recenter=False)

    @staticmethod
    def backward(ctx, grad_4: Tensor) -> Tuple[Tensor, None, None, None, None]:
        def compute(ctx, delta_pp, delta_np, delta_pn, delta_nn):
            new_pp = delta_pp.reshape(ctx.input_shape)
            new_np = delta_np.reshape(ctx.input_shape)
            new_pn = delta_pn.reshape(ctx.input_shape)
            new_nn = delta_nn.reshape(ctx.input_shape)
            return new_pp, new_np, new_pn, new_nn

        return BackwardBuilder.run(ctx, grad_4, compute, num_extra_returns=4)


def dc_forward_reshape(m: Reshape, x: Tensor) -> Tensor:
    cache, layer_name = get_cache_info(m)
    return DCReshapeFunction.apply(x, m.shape,
                                    getattr(m, DC_IS_OUTPUT_LAYER, False),
                                    cache, layer_name)


patch_reshape = create_patch_function(dc_forward_reshape)
unpatch_reshape = create_unpatch_function()


# =============================================================================
# View Module (for functional tensor.view replacement)
# =============================================================================

class View(nn.Module):
    """Module wrapper for tensor.view that can be patched for DC decomposition."""

    def __init__(self, shape: Tuple[int, ...]):
        super().__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        return x.view(self.shape)


def dc_forward_view(m: View, x: Tensor) -> Tensor:
    # View and reshape have same behavior for DC
    cache, layer_name = get_cache_info(m)
    return DCReshapeFunction.apply(x, m.shape,
                                    getattr(m, DC_IS_OUTPUT_LAYER, False),
                                    cache, layer_name)


patch_view = create_patch_function(dc_forward_view)
unpatch_view = create_unpatch_function()


# =============================================================================
# Squeeze Module
# =============================================================================

class Squeeze(nn.Module):
    """Module wrapper for tensor.squeeze that can be patched for DC decomposition."""

    def __init__(self, dim: int = None):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        if self.dim is None:
            return x.squeeze()
        return x.squeeze(self.dim)


class DCSqueezeFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_4: Tensor, dim: int,
                is_output_layer: bool, cache, layer_name) -> Tensor:
        fb = ForwardBuilder(ctx, is_output_layer, cache, layer_name)
        pos, neg = fb.split_input(input_4)

        ctx.input_shape = pos.shape
        ctx.dim = dim

        if dim is None:
            out_pos = pos.squeeze()
            out_neg = neg.squeeze()
        else:
            out_pos = pos.squeeze(dim)
            out_neg = neg.squeeze(dim)

        return fb.build_output(out_pos, out_neg, recenter=False)

    @staticmethod
    def backward(ctx, grad_4: Tensor) -> Tuple[Tensor, None, None, None, None]:
        def compute(ctx, delta_pp, delta_np, delta_pn, delta_nn):
            new_pp = delta_pp.view(ctx.input_shape)
            new_np = delta_np.view(ctx.input_shape)
            new_pn = delta_pn.view(ctx.input_shape)
            new_nn = delta_nn.view(ctx.input_shape)
            return new_pp, new_np, new_pn, new_nn

        return BackwardBuilder.run(ctx, grad_4, compute, num_extra_returns=4)


def dc_forward_squeeze(m: Squeeze, x: Tensor) -> Tensor:
    cache, layer_name = get_cache_info(m)
    return DCSqueezeFunction.apply(x, m.dim,
                                    getattr(m, DC_IS_OUTPUT_LAYER, False),
                                    cache, layer_name)


patch_squeeze = create_patch_function(dc_forward_squeeze)
unpatch_squeeze = create_unpatch_function()


# =============================================================================
# Unsqueeze Module
# =============================================================================

class Unsqueeze(nn.Module):
    """Module wrapper for tensor.unsqueeze that can be patched for DC decomposition."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return x.unsqueeze(self.dim)


class DCUnsqueezeFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_4: Tensor, dim: int,
                is_output_layer: bool, cache, layer_name) -> Tensor:
        fb = ForwardBuilder(ctx, is_output_layer, cache, layer_name)
        pos, neg = fb.split_input(input_4)

        ctx.dim = dim

        out_pos = pos.unsqueeze(dim)
        out_neg = neg.unsqueeze(dim)

        return fb.build_output(out_pos, out_neg, recenter=False)

    @staticmethod
    def backward(ctx, grad_4: Tensor) -> Tuple[Tensor, None, None, None, None]:
        def compute(ctx, delta_pp, delta_np, delta_pn, delta_nn):
            new_pp = delta_pp.squeeze(ctx.dim)
            new_np = delta_np.squeeze(ctx.dim)
            new_pn = delta_pn.squeeze(ctx.dim)
            new_nn = delta_nn.squeeze(ctx.dim)
            return new_pp, new_np, new_pn, new_nn

        return BackwardBuilder.run(ctx, grad_4, compute, num_extra_returns=4)


def dc_forward_unsqueeze(m: Unsqueeze, x: Tensor) -> Tensor:
    cache, layer_name = get_cache_info(m)
    return DCUnsqueezeFunction.apply(x, m.dim,
                                      getattr(m, DC_IS_OUTPUT_LAYER, False),
                                      cache, layer_name)


patch_unsqueeze = create_patch_function(dc_forward_unsqueeze)
unpatch_unsqueeze = create_unpatch_function()


# =============================================================================
# Transpose Module
# =============================================================================

class Transpose(nn.Module):
    """Module wrapper for tensor.transpose that can be patched for DC decomposition."""

    def __init__(self, dim0: int, dim1: int):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x: Tensor) -> Tensor:
        return x.transpose(self.dim0, self.dim1)


class DCTransposeFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_4: Tensor, dim0: int, dim1: int,
                is_output_layer: bool, cache, layer_name) -> Tensor:
        fb = ForwardBuilder(ctx, is_output_layer, cache, layer_name)
        pos, neg = fb.split_input(input_4)

        ctx.dim0, ctx.dim1 = dim0, dim1

        out_pos = pos.transpose(dim0, dim1)
        out_neg = neg.transpose(dim0, dim1)

        return fb.build_output(out_pos, out_neg, recenter=False)

    @staticmethod
    def backward(ctx, grad_4: Tensor) -> Tuple[Tensor, None, None, None, None, None]:
        def compute(ctx, delta_pp, delta_np, delta_pn, delta_nn):
            # Transpose is its own inverse
            new_pp = delta_pp.transpose(ctx.dim0, ctx.dim1)
            new_np = delta_np.transpose(ctx.dim0, ctx.dim1)
            new_pn = delta_pn.transpose(ctx.dim0, ctx.dim1)
            new_nn = delta_nn.transpose(ctx.dim0, ctx.dim1)
            return new_pp, new_np, new_pn, new_nn

        return BackwardBuilder.run(ctx, grad_4, compute, num_extra_returns=5)


def dc_forward_transpose(m: Transpose, x: Tensor) -> Tensor:
    cache, layer_name = get_cache_info(m)
    return DCTransposeFunction.apply(x, m.dim0, m.dim1,
                                      getattr(m, DC_IS_OUTPUT_LAYER, False),
                                      cache, layer_name)


patch_transpose = create_patch_function(dc_forward_transpose)
unpatch_transpose = create_unpatch_function()


# =============================================================================
# Permute Module
# =============================================================================

class Permute(nn.Module):
    """Module wrapper for tensor.permute that can be patched for DC decomposition."""

    def __init__(self, dims: Tuple[int, ...]):
        super().__init__()
        self.dims = dims

    def forward(self, x: Tensor) -> Tensor:
        return x.permute(self.dims)


class DCPermuteFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_4: Tensor, dims: Tuple[int, ...],
                is_output_layer: bool, cache, layer_name) -> Tensor:
        fb = ForwardBuilder(ctx, is_output_layer, cache, layer_name)
        pos, neg = fb.split_input(input_4)

        ctx.dims = dims

        out_pos = pos.permute(dims)
        out_neg = neg.permute(dims)

        return fb.build_output(out_pos, out_neg, recenter=False)

    @staticmethod
    def backward(ctx, grad_4: Tensor) -> Tuple[Tensor, None, None, None, None]:
        def compute(ctx, delta_pp, delta_np, delta_pn, delta_nn):
            # Compute inverse permutation
            inv_dims = [0] * len(ctx.dims)
            for i, d in enumerate(ctx.dims):
                inv_dims[d] = i

            new_pp = delta_pp.permute(inv_dims)
            new_np = delta_np.permute(inv_dims)
            new_pn = delta_pn.permute(inv_dims)
            new_nn = delta_nn.permute(inv_dims)
            return new_pp, new_np, new_pn, new_nn

        return BackwardBuilder.run(ctx, grad_4, compute, num_extra_returns=4)


def dc_forward_permute(m: Permute, x: Tensor) -> Tensor:
    cache, layer_name = get_cache_info(m)
    return DCPermuteFunction.apply(x, m.dims,
                                    getattr(m, DC_IS_OUTPUT_LAYER, False),
                                    cache, layer_name)


patch_permute = create_patch_function(dc_forward_permute)
unpatch_permute = create_unpatch_function()


# =============================================================================
# Dropout Module (passes through in eval mode, identity in DC mode)
# =============================================================================

class DCDropoutFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_4: Tensor, is_output_layer: bool, cache, layer_name) -> Tensor:
        # In DC mode, dropout is identity (should only be used in eval mode)
        fb = ForwardBuilder(ctx, is_output_layer, cache, layer_name)
        return input_4

    @staticmethod
    def backward(ctx, grad_4: Tensor) -> Tuple[Tensor, None, None, None]:
        # Identity backward
        return grad_4, None, None, None


def dc_forward_dropout(m: nn.Dropout, x: Tensor) -> Tensor:
    cache, layer_name = get_cache_info(m)
    return DCDropoutFunction.apply(x,
                                    getattr(m, DC_IS_OUTPUT_LAYER, False),
                                    cache, layer_name)


patch_dropout = create_patch_function(dc_forward_dropout)
unpatch_dropout = create_unpatch_function()
