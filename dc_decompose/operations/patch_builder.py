"""
Unified patch building utilities for DC decomposition.

This module provides clean, reusable patterns for DC operations to eliminate
code duplication. All DC autograd functions should use ForwardBuilder and
BackwardBuilder for consistent handling of:
- Input splitting (split_input_4)
- Output creation with optional re-centering and alignment
- Backward initialization (init_backward)
- Gradient creation with optional alignment
- Cached mask access for backward-only mode

Usage:
    class DCLinearFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input_4, weight, bias, is_output_layer, cache, layer_name):
            fb = ForwardBuilder(ctx, is_output_layer, cache, layer_name, alpha)
            pos, neg = fb.split_input(input_4)

            # ... layer-specific computation ...

            return fb.build_output(out_pos, out_neg)

        @staticmethod
        def backward(ctx, grad_4):
            def compute(ctx, delta_pp, delta_np, delta_pn, delta_nn):
                # ... layer-specific backward ...
                return new_pp, new_np, new_pn, new_nn

            return BackwardBuilder.run(ctx, grad_4, compute, num_extra_returns=5)
"""

from typing import Tuple, Optional, Callable, Any, TYPE_CHECKING

import torch
from torch import Tensor
import torch.nn as nn

from .base import (
    split_input_4, make_output_4, make_grad_4,
    init_backward, recenter_forward, recenter_dc, recenter_grad,
    shift_sensitivities,
    DC_ENABLED, DC_ORIGINAL_FORWARD, DC_IS_OUTPUT_LAYER,
    DC_ALIGNMENT_CACHE, DC_CACHE_LAYER_NAME, DC_SENSITIVITY_ALPHA,
)

if TYPE_CHECKING:
    from ..alignment_cache import AlignmentCache


def get_cache_info(module: nn.Module) -> Tuple[Optional['AlignmentCache'], Optional[str], float]:
    """Extract alignment cache, layer name, and sensitivity alpha from a module."""
    cache = getattr(module, DC_ALIGNMENT_CACHE, None)
    layer_name = getattr(module, DC_CACHE_LAYER_NAME, None)
    alpha = getattr(module, DC_SENSITIVITY_ALPHA, 0.0)
    return cache, layer_name, alpha


class ForwardBuilder:
    """
    Builder for DC forward pass with consistent handling of splitting,
    output creation, re-centering, and alignment.

    Usage:
        @staticmethod
        def forward(ctx, input_4, ..., is_output_layer, cache, layer_name):
            fb = ForwardBuilder(ctx, is_output_layer, cache, layer_name, alpha)
            pos, neg = fb.split_input(input_4)

            # ... layer-specific computation ...

            return fb.build_output(out_pos, out_neg)
    """

    def __init__(
        self,
        ctx: Any,
        is_output_layer: bool,
        cache: Optional['AlignmentCache'] = None,
        layer_name: Optional[str] = None,
        alpha: float = 0.0,
    ):
        self.ctx = ctx
        self.is_output_layer = is_output_layer
        self.cache = cache
        self.layer_name = layer_name
        self.alpha = alpha

        # Store in context for backward
        ctx.is_output_layer = is_output_layer
        ctx._dc_cache = cache
        ctx._dc_layer_name = layer_name
        ctx._dc_sensitivity_alpha = alpha

    def split_input(self, input_4: Tensor) -> Tuple[Tensor, Tensor]:
        """Split [4*batch] input into pos and neg."""
        return split_input_4(input_4)

    def build_output(
        self,
        out_pos: Tensor,
        out_neg: Tensor,
        recenter: bool = True,
    ) -> Tensor:
        """
        Create [4*batch] output with re-centering and alignment.

        Args:
            out_pos: Positive output
            out_neg: Negative output
            recenter: Whether to apply re-centering (default True)

        Returns:
            [4*batch] output tensor
        """
        # Apply forward alignment if cache is active
        if self.cache is not None and self.layer_name:
            from ..alignment_cache import AlignmentMode
            if self.cache.mode in (AlignmentMode.FORWARD_ONLY, AlignmentMode.BOTH):
                out_pos, out_neg = self.cache.align_forward(
                    self.layer_name, out_pos, out_neg
                )

        output = make_output_4(out_pos, out_neg)

        if recenter:
            output = recenter_forward(output)

        return output

    def build_output_dc(
        self,
        out_pos: Tensor,
        out_neg: Tensor,
    ) -> Tensor:
        """
        Create [4*batch] output with recenter_dc (for ReLU-like ops).

        Uses recenter_dc instead of recenter_forward.
        """
        # Apply forward alignment if cache is active
        if self.cache is not None and self.layer_name:
            from ..alignment_cache import AlignmentMode
            if self.cache.mode in (AlignmentMode.FORWARD_ONLY, AlignmentMode.BOTH):
                out_pos, out_neg = self.cache.align_forward(
                    self.layer_name, out_pos, out_neg
                )

        output = make_output_4(out_pos, out_neg)
        return recenter_dc(output)

    def get_cached_relu_mask(self) -> Optional[Tensor]:
        """Get cached ReLU mask for backward-only mode."""
        if self.cache is not None and self.layer_name:
            return self.cache.get_relu_mask(self.layer_name)
        return None

    def get_cached_maxpool_indices(self) -> Optional[Tensor]:
        """Get cached MaxPool indices for backward-only mode."""
        if self.cache is not None and self.layer_name:
            return self.cache.get_maxpool_indices(self.layer_name)
        return None

    def should_use_cached_mask(self) -> bool:
        """Check if backward-only mode with cached masks is active."""
        if self.cache is None or self.layer_name is None:
            return False
        return self.cache.should_use_cached_mask(self.layer_name)


class BackwardBuilder:
    """
    Builder for DC backward pass. Provides a static method to run the backward
    pass with consistent handling of initialization, gradient creation, and alignment.

    Usage:
        @staticmethod
        def backward(ctx, grad_4):
            def compute(ctx, delta_pp, delta_np, delta_pn, delta_nn):
                # ... layer-specific backward ...
                return new_pp, new_np, new_pn, new_nn

            return BackwardBuilder.run(ctx, grad_4, compute, num_extra_returns=5)
    """

    @staticmethod
    def run(
        ctx: Any,
        grad_4: Tensor,
        backward_fn: Callable[[Any, Tensor, Tensor, Tensor, Tensor], Tuple[Tensor, Tensor, Tensor, Tensor]],
        num_extra_returns: int = 0,
    ) -> Tuple[Tensor, ...]:
        """
        Execute backward pass with layer-specific computation.

        Args:
            ctx: Autograd context from forward pass
            grad_4: Gradient tensor [4*batch]
            backward_fn: Function(ctx, delta_pp, delta_np, delta_pn, delta_nn) -> (new_pp, new_np, new_pn, new_nn)
            num_extra_returns: Number of None values to append (for other forward args)

        Returns:
            Tuple of (grad_input, None, None, ...) with num_extra_returns Nones
        """
        is_output_layer = ctx.is_output_layer
        cache = getattr(ctx, '_dc_cache', None)
        layer_name = getattr(ctx, '_dc_layer_name', None)
        alpha = getattr(ctx, '_dc_sensitivity_alpha', 0.0)

        # Initialize deltas from gradient
        delta_pp, delta_np, delta_pn, delta_nn = init_backward(grad_4, is_output_layer)

        # Run layer-specific backward computation
        new_pp, new_np, new_pn, new_nn = backward_fn(ctx, delta_pp, delta_np, delta_pn, delta_nn)

        # Apply backward alignment if cache is active
        if cache is not None and layer_name:
            from ..alignment_cache import AlignmentMode
            if cache.mode in (AlignmentMode.BACKWARD_ONLY, AlignmentMode.BOTH):
                new_pp, new_np, new_pn, new_nn = cache.align_backward(
                    layer_name, new_pp, new_np, new_pn, new_nn
                )

        # Apply sensitivity shift for numerical stability
        if alpha > 0.0:
            new_pp, new_np, new_pn, new_nn = shift_sensitivities(
                new_pp, new_np, new_pn, new_nn, alpha
            )

        # Build gradient tensor
        grad = make_grad_4(new_pp, new_np, new_pn, new_nn)

        # Return with appropriate number of Nones for other forward args
        return (grad,) + (None,) * num_extra_returns

    @staticmethod
    def run_multi(
        ctx: Any,
        grad_4: Tensor,
        backward_fn: Callable,
        num_outputs: int,
        num_extra_returns: int = 0,
    ) -> Tuple[Tensor, ...]:
        """
        Execute backward pass for operations with multiple gradient outputs.

        Args:
            ctx: Autograd context from forward pass
            grad_4: Gradient tensor [4*batch]
            backward_fn: Function(ctx, delta_pp, delta_np, delta_pn, delta_nn) -> tuple of (pp, np, pn, nn) tuples
            num_outputs: Number of gradient outputs (e.g., 2 for matmul with A and B)
            num_extra_returns: Number of None values to append

        Returns:
            Tuple of gradient tensors and Nones
        """
        is_output_layer = ctx.is_output_layer
        cache = getattr(ctx, '_dc_cache', None)
        layer_name = getattr(ctx, '_dc_layer_name', None)
        alpha = getattr(ctx, '_dc_sensitivity_alpha', 0.0)

        # Initialize deltas from gradient
        delta_pp, delta_np, delta_pn, delta_nn = init_backward(grad_4, is_output_layer)

        # Run layer-specific backward computation
        results = backward_fn(ctx, delta_pp, delta_np, delta_pn, delta_nn)

        # Build gradient tensors for each output
        grads = []
        for i in range(num_outputs):
            new_pp, new_np, new_pn, new_nn = results[i]

            # Apply backward alignment if cache is active
            if cache is not None and layer_name:
                from ..alignment_cache import AlignmentMode
                if cache.mode in (AlignmentMode.BACKWARD_ONLY, AlignmentMode.BOTH):
                    new_pp, new_np, new_pn, new_nn = cache.align_backward(
                        layer_name, new_pp, new_np, new_pn, new_nn
                    )

            # Apply sensitivity shift for numerical stability
            if alpha > 0.0:
                new_pp, new_np, new_pn, new_nn = shift_sensitivities(
                    new_pp, new_np, new_pn, new_nn, alpha
                )

            grads.append(make_grad_4(new_pp, new_np, new_pn, new_nn))

        return tuple(grads) + (None,) * num_extra_returns


# =============================================================================
# Patch/Unpatch Function Factories
# =============================================================================

def create_patch_function(
    dc_forward_fn: Callable,
    extra_attrs: Tuple[str, ...] = (),
) -> Callable:
    """
    Create a standard patch function for a module type.

    Args:
        dc_forward_fn: Function that performs DC forward (module, *args)
        extra_attrs: Additional attributes to set on the module

    Returns:
        A patch function that can be applied to modules
    """
    def patch_fn(module: nn.Module) -> None:
        if hasattr(module, DC_ORIGINAL_FORWARD):
            return

        setattr(module, DC_ORIGINAL_FORWARD, module.forward)
        setattr(module, DC_ENABLED, True)
        setattr(module, DC_IS_OUTPUT_LAYER, False)

        for attr in extra_attrs:
            if not hasattr(module, attr):
                setattr(module, attr, None)

        def patched(*args, **kwargs):
            if getattr(module, DC_ENABLED, False):
                return dc_forward_fn(module, *args, **kwargs)
            else:
                return getattr(module, DC_ORIGINAL_FORWARD)(*args, **kwargs)

        module.forward = patched

    return patch_fn


def create_unpatch_function(
    extra_attrs: Tuple[str, ...] = (),
) -> Callable:
    """
    Create a standard unpatch function for a module type.

    Args:
        extra_attrs: Additional attributes to clean up

    Returns:
        An unpatch function
    """
    all_attrs = (
        DC_ORIGINAL_FORWARD, DC_ENABLED, DC_IS_OUTPUT_LAYER,
        DC_ALIGNMENT_CACHE, DC_CACHE_LAYER_NAME,
    ) + extra_attrs

    def unpatch_fn(module: nn.Module) -> None:
        if hasattr(module, DC_ORIGINAL_FORWARD):
            module.forward = getattr(module, DC_ORIGINAL_FORWARD)
            for attr in all_attrs:
                if hasattr(module, attr):
                    delattr(module, attr)

    return unpatch_fn
