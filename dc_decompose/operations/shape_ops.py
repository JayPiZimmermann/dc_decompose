"""DC Decomposition for shape operations. Forward/Backward: [4*batch] -> [4*batch]

Shape operations (flatten, reshape, transpose, etc.) are linear operations that apply
the same transformation to both pos and neg streams.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Union, List

from .base import split_input_4, make_output_4, split_grad_4, make_grad_4, DC_ENABLED, DC_ORIGINAL_FORWARD, DC_IS_OUTPUT_LAYER, DC_BETA


# =============================================================================
# Flatten
# =============================================================================

class DCFlattenFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_4: Tensor, start_dim: int, end_dim: int,
                is_output_layer: bool, beta: float) -> Tensor:
        pos, neg = split_input_4(input_4)

        ctx.input_shape = pos.shape
        ctx.is_output_layer = is_output_layer
        ctx.beta = beta

        out_pos = torch.flatten(pos, start_dim, end_dim)
        out_neg = torch.flatten(neg, start_dim, end_dim)

        return make_output_4(out_pos, out_neg)

    @staticmethod
    def backward(ctx, grad_4: Tensor) -> Tuple[Tensor, None, None, None, None]:
        if ctx.is_output_layer:
            q = grad_4.shape[0] // 4
            gp, gn = grad_4[:q], grad_4[q:2*q]
            delta_pp, delta_np = ctx.beta * gp, torch.zeros_like(gp)
            delta_pn, delta_nn = (1 - ctx.beta) * gn, torch.zeros_like(gn)
        else:
            delta_pp, delta_np, delta_pn, delta_nn = split_grad_4(grad_4)

        new_pp = delta_pp.view(ctx.input_shape)
        new_np = delta_np.view(ctx.input_shape)
        new_pn = delta_pn.view(ctx.input_shape)
        new_nn = delta_nn.view(ctx.input_shape)

        return make_grad_4(new_pp, new_np, new_pn, new_nn), None, None, None, None


def dc_forward_flatten(m: nn.Flatten, x: Tensor) -> Tensor:
    return DCFlattenFunction.apply(x, m.start_dim, m.end_dim,
                                    getattr(m, DC_IS_OUTPUT_LAYER, False), getattr(m, DC_BETA, 1.0))


def patch_flatten(module: nn.Flatten) -> None:
    if hasattr(module, DC_ORIGINAL_FORWARD): return
    setattr(module, DC_ORIGINAL_FORWARD, module.forward)
    setattr(module, DC_ENABLED, True)
    setattr(module, DC_IS_OUTPUT_LAYER, False)
    setattr(module, DC_BETA, 1.0)

    def patched(x):
        if getattr(module, DC_ENABLED, False):
            return dc_forward_flatten(module, x)
        else:
            return getattr(module, DC_ORIGINAL_FORWARD)(x)

    module.forward = patched


def unpatch_flatten(module: nn.Flatten) -> None:
    if hasattr(module, DC_ORIGINAL_FORWARD):
        module.forward = getattr(module, DC_ORIGINAL_FORWARD)
        for a in [DC_ORIGINAL_FORWARD, DC_ENABLED, DC_IS_OUTPUT_LAYER, DC_BETA]:
            if hasattr(module, a): delattr(module, a)


# =============================================================================
# Unflatten
# =============================================================================

class DCUnflattenFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_4: Tensor, dim: int, unflattened_size: Tuple[int, ...],
                is_output_layer: bool, beta: float) -> Tensor:
        pos, neg = split_input_4(input_4)

        ctx.input_shape = pos.shape
        ctx.dim = dim
        ctx.is_output_layer = is_output_layer
        ctx.beta = beta

        out_pos = pos.unflatten(dim, unflattened_size)
        out_neg = neg.unflatten(dim, unflattened_size)

        return make_output_4(out_pos, out_neg)

    @staticmethod
    def backward(ctx, grad_4: Tensor) -> Tuple[Tensor, None, None, None, None]:
        if ctx.is_output_layer:
            q = grad_4.shape[0] // 4
            gp, gn = grad_4[:q], grad_4[q:2*q]
            delta_pp, delta_np = ctx.beta * gp, torch.zeros_like(gp)
            delta_pn, delta_nn = (1 - ctx.beta) * gn, torch.zeros_like(gn)
        else:
            delta_pp, delta_np, delta_pn, delta_nn = split_grad_4(grad_4)

        # Flatten back to original shape
        new_pp = delta_pp.flatten(ctx.dim, ctx.dim + len(grad_4.shape) - len(ctx.input_shape))
        new_np = delta_np.flatten(ctx.dim, ctx.dim + len(grad_4.shape) - len(ctx.input_shape))
        new_pn = delta_pn.flatten(ctx.dim, ctx.dim + len(grad_4.shape) - len(ctx.input_shape))
        new_nn = delta_nn.flatten(ctx.dim, ctx.dim + len(grad_4.shape) - len(ctx.input_shape))

        return make_grad_4(new_pp, new_np, new_pn, new_nn), None, None, None, None


def dc_forward_unflatten(m: nn.Unflatten, x: Tensor) -> Tensor:
    return DCUnflattenFunction.apply(x, m.dim, m.unflattened_size,
                                      getattr(m, DC_IS_OUTPUT_LAYER, False), getattr(m, DC_BETA, 1.0))


def patch_unflatten(module: nn.Unflatten) -> None:
    if hasattr(module, DC_ORIGINAL_FORWARD): return
    setattr(module, DC_ORIGINAL_FORWARD, module.forward)
    setattr(module, DC_ENABLED, True)
    setattr(module, DC_IS_OUTPUT_LAYER, False)
    setattr(module, DC_BETA, 1.0)

    def patched(x):
        if getattr(module, DC_ENABLED, False):
            return dc_forward_unflatten(module, x)
        else:
            return getattr(module, DC_ORIGINAL_FORWARD)(x)

    module.forward = patched


def unpatch_unflatten(module: nn.Unflatten) -> None:
    if hasattr(module, DC_ORIGINAL_FORWARD):
        module.forward = getattr(module, DC_ORIGINAL_FORWARD)
        for a in [DC_ORIGINAL_FORWARD, DC_ENABLED, DC_IS_OUTPUT_LAYER, DC_BETA]:
            if hasattr(module, a): delattr(module, a)


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
                is_output_layer: bool, beta: float) -> Tensor:
        pos, neg = split_input_4(input_4)

        ctx.input_shape = pos.shape
        ctx.is_output_layer = is_output_layer
        ctx.beta = beta

        # Adjust shape for batch dimension (shape[0] should be -1 or match)
        out_pos = pos.reshape(shape)
        out_neg = neg.reshape(shape)

        return make_output_4(out_pos, out_neg)

    @staticmethod
    def backward(ctx, grad_4: Tensor) -> Tuple[Tensor, None, None, None]:
        if ctx.is_output_layer:
            q = grad_4.shape[0] // 4
            gp, gn = grad_4[:q], grad_4[q:2*q]
            delta_pp, delta_np = ctx.beta * gp, torch.zeros_like(gp)
            delta_pn, delta_nn = (1 - ctx.beta) * gn, torch.zeros_like(gn)
        else:
            delta_pp, delta_np, delta_pn, delta_nn = split_grad_4(grad_4)

        new_pp = delta_pp.reshape(ctx.input_shape)
        new_np = delta_np.reshape(ctx.input_shape)
        new_pn = delta_pn.reshape(ctx.input_shape)
        new_nn = delta_nn.reshape(ctx.input_shape)

        return make_grad_4(new_pp, new_np, new_pn, new_nn), None, None, None


def dc_forward_reshape(m: Reshape, x: Tensor) -> Tensor:
    return DCReshapeFunction.apply(x, m.shape,
                                    getattr(m, DC_IS_OUTPUT_LAYER, False), getattr(m, DC_BETA, 1.0))


def patch_reshape(module: Reshape) -> None:
    if hasattr(module, DC_ORIGINAL_FORWARD): return
    setattr(module, DC_ORIGINAL_FORWARD, module.forward)
    setattr(module, DC_ENABLED, True)
    setattr(module, DC_IS_OUTPUT_LAYER, False)
    setattr(module, DC_BETA, 1.0)

    def patched(x):
        if getattr(module, DC_ENABLED, False):
            return dc_forward_reshape(module, x)
        else:
            return getattr(module, DC_ORIGINAL_FORWARD)(x)

    module.forward = patched


def unpatch_reshape(module: Reshape) -> None:
    if hasattr(module, DC_ORIGINAL_FORWARD):
        module.forward = getattr(module, DC_ORIGINAL_FORWARD)
        for a in [DC_ORIGINAL_FORWARD, DC_ENABLED, DC_IS_OUTPUT_LAYER, DC_BETA]:
            if hasattr(module, a): delattr(module, a)


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
    return DCReshapeFunction.apply(x, m.shape,
                                    getattr(m, DC_IS_OUTPUT_LAYER, False), getattr(m, DC_BETA, 1.0))


def patch_view(module: View) -> None:
    if hasattr(module, DC_ORIGINAL_FORWARD): return
    setattr(module, DC_ORIGINAL_FORWARD, module.forward)
    setattr(module, DC_ENABLED, True)
    setattr(module, DC_IS_OUTPUT_LAYER, False)
    setattr(module, DC_BETA, 1.0)

    def patched(x):
        if getattr(module, DC_ENABLED, False):
            return dc_forward_view(module, x)
        else:
            return getattr(module, DC_ORIGINAL_FORWARD)(x)

    module.forward = patched


def unpatch_view(module: View) -> None:
    if hasattr(module, DC_ORIGINAL_FORWARD):
        module.forward = getattr(module, DC_ORIGINAL_FORWARD)
        for a in [DC_ORIGINAL_FORWARD, DC_ENABLED, DC_IS_OUTPUT_LAYER, DC_BETA]:
            if hasattr(module, a): delattr(module, a)


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
                is_output_layer: bool, beta: float) -> Tensor:
        pos, neg = split_input_4(input_4)

        ctx.input_shape = pos.shape
        ctx.dim = dim
        ctx.is_output_layer = is_output_layer
        ctx.beta = beta

        if dim is None:
            out_pos = pos.squeeze()
            out_neg = neg.squeeze()
        else:
            out_pos = pos.squeeze(dim)
            out_neg = neg.squeeze(dim)

        return make_output_4(out_pos, out_neg)

    @staticmethod
    def backward(ctx, grad_4: Tensor) -> Tuple[Tensor, None, None, None]:
        if ctx.is_output_layer:
            q = grad_4.shape[0] // 4
            gp, gn = grad_4[:q], grad_4[q:2*q]
            delta_pp, delta_np = ctx.beta * gp, torch.zeros_like(gp)
            delta_pn, delta_nn = (1 - ctx.beta) * gn, torch.zeros_like(gn)
        else:
            delta_pp, delta_np, delta_pn, delta_nn = split_grad_4(grad_4)

        new_pp = delta_pp.view(ctx.input_shape)
        new_np = delta_np.view(ctx.input_shape)
        new_pn = delta_pn.view(ctx.input_shape)
        new_nn = delta_nn.view(ctx.input_shape)

        return make_grad_4(new_pp, new_np, new_pn, new_nn), None, None, None


def dc_forward_squeeze(m: Squeeze, x: Tensor) -> Tensor:
    return DCSqueezeFunction.apply(x, m.dim,
                                    getattr(m, DC_IS_OUTPUT_LAYER, False), getattr(m, DC_BETA, 1.0))


def patch_squeeze(module: Squeeze) -> None:
    if hasattr(module, DC_ORIGINAL_FORWARD): return
    setattr(module, DC_ORIGINAL_FORWARD, module.forward)
    setattr(module, DC_ENABLED, True)
    setattr(module, DC_IS_OUTPUT_LAYER, False)
    setattr(module, DC_BETA, 1.0)

    def patched(x):
        if getattr(module, DC_ENABLED, False):
            return dc_forward_squeeze(module, x)
        else:
            return getattr(module, DC_ORIGINAL_FORWARD)(x)

    module.forward = patched


def unpatch_squeeze(module: Squeeze) -> None:
    if hasattr(module, DC_ORIGINAL_FORWARD):
        module.forward = getattr(module, DC_ORIGINAL_FORWARD)
        for a in [DC_ORIGINAL_FORWARD, DC_ENABLED, DC_IS_OUTPUT_LAYER, DC_BETA]:
            if hasattr(module, a): delattr(module, a)


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
                is_output_layer: bool, beta: float) -> Tensor:
        pos, neg = split_input_4(input_4)

        ctx.dim = dim
        ctx.is_output_layer = is_output_layer
        ctx.beta = beta

        out_pos = pos.unsqueeze(dim)
        out_neg = neg.unsqueeze(dim)

        return make_output_4(out_pos, out_neg)

    @staticmethod
    def backward(ctx, grad_4: Tensor) -> Tuple[Tensor, None, None, None]:
        if ctx.is_output_layer:
            q = grad_4.shape[0] // 4
            gp, gn = grad_4[:q], grad_4[q:2*q]
            delta_pp, delta_np = ctx.beta * gp, torch.zeros_like(gp)
            delta_pn, delta_nn = (1 - ctx.beta) * gn, torch.zeros_like(gn)
        else:
            delta_pp, delta_np, delta_pn, delta_nn = split_grad_4(grad_4)

        new_pp = delta_pp.squeeze(ctx.dim)
        new_np = delta_np.squeeze(ctx.dim)
        new_pn = delta_pn.squeeze(ctx.dim)
        new_nn = delta_nn.squeeze(ctx.dim)

        return make_grad_4(new_pp, new_np, new_pn, new_nn), None, None, None


def dc_forward_unsqueeze(m: Unsqueeze, x: Tensor) -> Tensor:
    return DCUnsqueezeFunction.apply(x, m.dim,
                                      getattr(m, DC_IS_OUTPUT_LAYER, False), getattr(m, DC_BETA, 1.0))


def patch_unsqueeze(module: Unsqueeze) -> None:
    if hasattr(module, DC_ORIGINAL_FORWARD): return
    setattr(module, DC_ORIGINAL_FORWARD, module.forward)
    setattr(module, DC_ENABLED, True)
    setattr(module, DC_IS_OUTPUT_LAYER, False)
    setattr(module, DC_BETA, 1.0)

    def patched(x):
        if getattr(module, DC_ENABLED, False):
            return dc_forward_unsqueeze(module, x)
        else:
            return getattr(module, DC_ORIGINAL_FORWARD)(x)

    module.forward = patched


def unpatch_unsqueeze(module: Unsqueeze) -> None:
    if hasattr(module, DC_ORIGINAL_FORWARD):
        module.forward = getattr(module, DC_ORIGINAL_FORWARD)
        for a in [DC_ORIGINAL_FORWARD, DC_ENABLED, DC_IS_OUTPUT_LAYER, DC_BETA]:
            if hasattr(module, a): delattr(module, a)


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
                is_output_layer: bool, beta: float) -> Tensor:
        pos, neg = split_input_4(input_4)

        ctx.dim0, ctx.dim1 = dim0, dim1
        ctx.is_output_layer = is_output_layer
        ctx.beta = beta

        out_pos = pos.transpose(dim0, dim1)
        out_neg = neg.transpose(dim0, dim1)

        return make_output_4(out_pos, out_neg)

    @staticmethod
    def backward(ctx, grad_4: Tensor) -> Tuple[Tensor, None, None, None, None]:
        if ctx.is_output_layer:
            q = grad_4.shape[0] // 4
            gp, gn = grad_4[:q], grad_4[q:2*q]
            delta_pp, delta_np = ctx.beta * gp, torch.zeros_like(gp)
            delta_pn, delta_nn = (1 - ctx.beta) * gn, torch.zeros_like(gn)
        else:
            delta_pp, delta_np, delta_pn, delta_nn = split_grad_4(grad_4)

        # Transpose is its own inverse
        new_pp = delta_pp.transpose(ctx.dim0, ctx.dim1)
        new_np = delta_np.transpose(ctx.dim0, ctx.dim1)
        new_pn = delta_pn.transpose(ctx.dim0, ctx.dim1)
        new_nn = delta_nn.transpose(ctx.dim0, ctx.dim1)

        return make_grad_4(new_pp, new_np, new_pn, new_nn), None, None, None, None


def dc_forward_transpose(m: Transpose, x: Tensor) -> Tensor:
    return DCTransposeFunction.apply(x, m.dim0, m.dim1,
                                      getattr(m, DC_IS_OUTPUT_LAYER, False), getattr(m, DC_BETA, 1.0))


def patch_transpose(module: Transpose) -> None:
    if hasattr(module, DC_ORIGINAL_FORWARD): return
    setattr(module, DC_ORIGINAL_FORWARD, module.forward)
    setattr(module, DC_ENABLED, True)
    setattr(module, DC_IS_OUTPUT_LAYER, False)
    setattr(module, DC_BETA, 1.0)

    def patched(x):
        if getattr(module, DC_ENABLED, False):
            return dc_forward_transpose(module, x)
        else:
            return getattr(module, DC_ORIGINAL_FORWARD)(x)

    module.forward = patched


def unpatch_transpose(module: Transpose) -> None:
    if hasattr(module, DC_ORIGINAL_FORWARD):
        module.forward = getattr(module, DC_ORIGINAL_FORWARD)
        for a in [DC_ORIGINAL_FORWARD, DC_ENABLED, DC_IS_OUTPUT_LAYER, DC_BETA]:
            if hasattr(module, a): delattr(module, a)


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
                is_output_layer: bool, beta: float) -> Tensor:
        pos, neg = split_input_4(input_4)

        ctx.dims = dims
        ctx.is_output_layer = is_output_layer
        ctx.beta = beta

        out_pos = pos.permute(dims)
        out_neg = neg.permute(dims)

        return make_output_4(out_pos, out_neg)

    @staticmethod
    def backward(ctx, grad_4: Tensor) -> Tuple[Tensor, None, None, None]:
        if ctx.is_output_layer:
            q = grad_4.shape[0] // 4
            gp, gn = grad_4[:q], grad_4[q:2*q]
            delta_pp, delta_np = ctx.beta * gp, torch.zeros_like(gp)
            delta_pn, delta_nn = (1 - ctx.beta) * gn, torch.zeros_like(gn)
        else:
            delta_pp, delta_np, delta_pn, delta_nn = split_grad_4(grad_4)

        # Compute inverse permutation
        inv_dims = [0] * len(ctx.dims)
        for i, d in enumerate(ctx.dims):
            inv_dims[d] = i

        new_pp = delta_pp.permute(inv_dims)
        new_np = delta_np.permute(inv_dims)
        new_pn = delta_pn.permute(inv_dims)
        new_nn = delta_nn.permute(inv_dims)

        return make_grad_4(new_pp, new_np, new_pn, new_nn), None, None, None


def dc_forward_permute(m: Permute, x: Tensor) -> Tensor:
    return DCPermuteFunction.apply(x, m.dims,
                                    getattr(m, DC_IS_OUTPUT_LAYER, False), getattr(m, DC_BETA, 1.0))


def patch_permute(module: Permute) -> None:
    if hasattr(module, DC_ORIGINAL_FORWARD): return
    setattr(module, DC_ORIGINAL_FORWARD, module.forward)
    setattr(module, DC_ENABLED, True)
    setattr(module, DC_IS_OUTPUT_LAYER, False)
    setattr(module, DC_BETA, 1.0)

    def patched(x):
        if getattr(module, DC_ENABLED, False):
            return dc_forward_permute(module, x)
        else:
            return getattr(module, DC_ORIGINAL_FORWARD)(x)

    module.forward = patched


def unpatch_permute(module: Permute) -> None:
    if hasattr(module, DC_ORIGINAL_FORWARD):
        module.forward = getattr(module, DC_ORIGINAL_FORWARD)
        for a in [DC_ORIGINAL_FORWARD, DC_ENABLED, DC_IS_OUTPUT_LAYER, DC_BETA]:
            if hasattr(module, a): delattr(module, a)


# =============================================================================
# Dropout Module (passes through in eval mode, identity in DC mode)
# =============================================================================

class DCDropoutFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_4: Tensor, is_output_layer: bool, beta: float) -> Tensor:
        # In DC mode, dropout is identity (should only be used in eval mode)
        ctx.is_output_layer = is_output_layer
        ctx.beta = beta
        return input_4

    @staticmethod
    def backward(ctx, grad_4: Tensor) -> Tuple[Tensor, None, None]:
        # Identity backward
        return grad_4, None, None


def dc_forward_dropout(m: nn.Dropout, x: Tensor) -> Tensor:
    return DCDropoutFunction.apply(x,
                                    getattr(m, DC_IS_OUTPUT_LAYER, False), getattr(m, DC_BETA, 1.0))


def patch_dropout(module: nn.Dropout) -> None:
    if hasattr(module, DC_ORIGINAL_FORWARD): return
    setattr(module, DC_ORIGINAL_FORWARD, module.forward)
    setattr(module, DC_ENABLED, True)
    setattr(module, DC_IS_OUTPUT_LAYER, False)
    setattr(module, DC_BETA, 1.0)

    def patched(x):
        if getattr(module, DC_ENABLED, False):
            return dc_forward_dropout(module, x)
        else:
            return getattr(module, DC_ORIGINAL_FORWARD)(x)

    module.forward = patched


def unpatch_dropout(module: nn.Dropout) -> None:
    if hasattr(module, DC_ORIGINAL_FORWARD):
        module.forward = getattr(module, DC_ORIGINAL_FORWARD)
        for a in [DC_ORIGINAL_FORWARD, DC_ENABLED, DC_IS_OUTPUT_LAYER, DC_BETA]:
            if hasattr(module, a): delattr(module, a)
