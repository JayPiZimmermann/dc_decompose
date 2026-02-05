"""DC Decomposition for AvgPool layers. Forward/Backward: [4*batch] -> [4*batch]

AvgPool is linear, so we apply the same operation to both pos and neg streams.
Since it's linear, we let autograd handle the backward pass naturally.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Union

from .base import split_input_4, make_output_4, make_grad_4, init_backward, recenter_forward, DC_ENABLED, DC_ORIGINAL_FORWARD, DC_IS_OUTPUT_LAYER


# For linear operations like AvgPool, we don't need custom autograd.Function
# We just apply the operation to each component and let autograd handle backward

class DCAvgPool2dFunction(torch.autograd.Function):
    """Placeholder for backward compatibility. Actual implementation uses direct apply."""
    pass


class DCAvgPool1dFunction(torch.autograd.Function):
    """Placeholder for backward compatibility. Actual implementation uses direct apply."""
    pass


class DCAdaptiveAvgPool2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_4: Tensor, output_size, is_output_layer: bool) -> Tensor:
        pos, neg = split_input_4(input_4)
        
        # Apply adaptive avg pool to both streams
        out_pos = F.adaptive_avg_pool2d(pos, output_size)
        out_neg = F.adaptive_avg_pool2d(neg, output_size)
        
        # Save for backward
        ctx.input_shape = pos.shape
        ctx.output_size = output_size
        ctx.is_output_layer = is_output_layer
        
        output = make_output_4(out_pos, out_neg)
        return recenter_forward(output)
    
    @staticmethod
    def backward(ctx, grad_4: Tensor) -> Tuple[Tensor, None, None]:
        delta_pp, delta_np, delta_pn, delta_nn = init_backward(
            grad_4, ctx.is_output_layer)
        
        # For adaptive avg pool, gradients are distributed evenly across the pooled regions
        # We use the fact that adaptive_avg_pool2d is linear
        input_shape = ctx.input_shape
        
        # Create dummy tensors with the right shapes to compute backward through adaptive_avg_pool2d
        dummy_input = torch.ones(input_shape, device=grad_4.device, dtype=grad_4.dtype)
        dummy_input.requires_grad_(True)
        dummy_output = F.adaptive_avg_pool2d(dummy_input, ctx.output_size)
        
        # Use autograd to compute the backward mapping
        grad_dummy = torch.autograd.grad(
            dummy_output, dummy_input, 
            grad_outputs=torch.ones_like(dummy_output),
            create_graph=False, retain_graph=False
        )[0]
        
        # Apply this backward mapping to our deltas
        # Scale by the appropriate gradients
        new_pp = delta_pp.view_as(dummy_output) * grad_dummy.sum() / grad_dummy.numel()
        new_np = delta_np.view_as(dummy_output) * grad_dummy.sum() / grad_dummy.numel()  
        new_pn = delta_pn.view_as(dummy_output) * grad_dummy.sum() / grad_dummy.numel()
        new_nn = delta_nn.view_as(dummy_output) * grad_dummy.sum() / grad_dummy.numel()
        
        # Expand back to input shape
        new_pp = new_pp.expand_as(grad_dummy) * grad_dummy / (grad_dummy.sum() / grad_dummy.numel())
        new_np = new_np.expand_as(grad_dummy) * grad_dummy / (grad_dummy.sum() / grad_dummy.numel())
        new_pn = new_pn.expand_as(grad_dummy) * grad_dummy / (grad_dummy.sum() / grad_dummy.numel())  
        new_nn = new_nn.expand_as(grad_dummy) * grad_dummy / (grad_dummy.sum() / grad_dummy.numel())
        
        grad_input = make_grad_4(new_pp, new_np, new_pn, new_nn)
        return grad_input, None, None


class DCAdaptiveAvgPool1dFunction(torch.autograd.Function):
    """Placeholder for backward compatibility. Actual implementation uses direct apply."""
    pass


# DC forward functions - apply operation to each component, autograd handles backward
def dc_forward_avgpool2d(m: nn.AvgPool2d, x: Tensor) -> Tensor:
    """Apply AvgPool2d to DC format. Since it's linear, autograd handles backward."""
    pos, neg = split_input_4(x)
    out_pos = F.avg_pool2d(pos, m.kernel_size, m.stride, m.padding,
                           ceil_mode=m.ceil_mode, count_include_pad=m.count_include_pad)
    out_neg = F.avg_pool2d(neg, m.kernel_size, m.stride, m.padding,
                           ceil_mode=m.ceil_mode, count_include_pad=m.count_include_pad)
    return make_output_4(out_pos, out_neg)


def dc_forward_avgpool1d(m: nn.AvgPool1d, x: Tensor) -> Tensor:
    """Apply AvgPool1d to DC format. Since it's linear, autograd handles backward."""
    pos, neg = split_input_4(x)
    stride = m.stride if m.stride is not None else m.kernel_size
    out_pos = F.avg_pool1d(pos, m.kernel_size, stride, m.padding,
                           ceil_mode=m.ceil_mode, count_include_pad=m.count_include_pad)
    out_neg = F.avg_pool1d(neg, m.kernel_size, stride, m.padding,
                           ceil_mode=m.ceil_mode, count_include_pad=m.count_include_pad)
    return make_output_4(out_pos, out_neg)


def dc_forward_adaptive_avgpool2d(m: nn.AdaptiveAvgPool2d, x: Tensor) -> Tensor:
    """Apply AdaptiveAvgPool2d to DC format. Since it's linear, autograd handles backward."""
    pos, neg = split_input_4(x)
    out_pos = F.adaptive_avg_pool2d(pos, m.output_size)
    out_neg = F.adaptive_avg_pool2d(neg, m.output_size)
    return make_output_4(out_pos, out_neg)


def dc_forward_adaptive_avgpool1d(m: nn.AdaptiveAvgPool1d, x: Tensor) -> Tensor:
    """Apply AdaptiveAvgPool1d to DC format. Since it's linear, autograd handles backward."""
    pos, neg = split_input_4(x)
    out_pos = F.adaptive_avg_pool1d(pos, m.output_size)
    out_neg = F.adaptive_avg_pool1d(neg, m.output_size)
    return make_output_4(out_pos, out_neg)


# Patch functions
def patch_avgpool2d(module: nn.AvgPool2d) -> None:
    if hasattr(module, DC_ORIGINAL_FORWARD): return
    setattr(module, DC_ORIGINAL_FORWARD, module.forward)
    setattr(module, DC_ENABLED, True)
    setattr(module, DC_IS_OUTPUT_LAYER, False)
    

    def patched(x):
        if getattr(module, DC_ENABLED, False):
            return dc_forward_avgpool2d(module, x)
        else:
            return getattr(module, DC_ORIGINAL_FORWARD)(x)

    module.forward = patched


def patch_avgpool1d(module: nn.AvgPool1d) -> None:
    if hasattr(module, DC_ORIGINAL_FORWARD): return
    setattr(module, DC_ORIGINAL_FORWARD, module.forward)
    setattr(module, DC_ENABLED, True)
    setattr(module, DC_IS_OUTPUT_LAYER, False)
    

    def patched(x):
        if getattr(module, DC_ENABLED, False):
            return dc_forward_avgpool1d(module, x)
        else:
            return getattr(module, DC_ORIGINAL_FORWARD)(x)

    module.forward = patched


def patch_adaptive_avgpool2d(module: nn.AdaptiveAvgPool2d) -> None:
    if hasattr(module, DC_ORIGINAL_FORWARD): return
    setattr(module, DC_ORIGINAL_FORWARD, module.forward)
    setattr(module, DC_ENABLED, True)
    setattr(module, DC_IS_OUTPUT_LAYER, False)
    

    def patched(x):
        if getattr(module, DC_ENABLED, False):
            return dc_forward_adaptive_avgpool2d(module, x)
        else:
            return getattr(module, DC_ORIGINAL_FORWARD)(x)

    module.forward = patched


def patch_adaptive_avgpool1d(module: nn.AdaptiveAvgPool1d) -> None:
    if hasattr(module, DC_ORIGINAL_FORWARD): return
    setattr(module, DC_ORIGINAL_FORWARD, module.forward)
    setattr(module, DC_ENABLED, True)
    setattr(module, DC_IS_OUTPUT_LAYER, False)
    

    def patched(x):
        if getattr(module, DC_ENABLED, False):
            return dc_forward_adaptive_avgpool1d(module, x)
        else:
            return getattr(module, DC_ORIGINAL_FORWARD)(x)

    module.forward = patched


# Unpatch functions
def unpatch_avgpool2d(module: nn.AvgPool2d) -> None:
    if hasattr(module, DC_ORIGINAL_FORWARD):
        module.forward = getattr(module, DC_ORIGINAL_FORWARD)
        for a in [DC_ORIGINAL_FORWARD, DC_ENABLED, DC_IS_OUTPUT_LAYER]:
            if hasattr(module, a): delattr(module, a)


def unpatch_avgpool1d(module: nn.AvgPool1d) -> None:
    if hasattr(module, DC_ORIGINAL_FORWARD):
        module.forward = getattr(module, DC_ORIGINAL_FORWARD)
        for a in [DC_ORIGINAL_FORWARD, DC_ENABLED, DC_IS_OUTPUT_LAYER]:
            if hasattr(module, a): delattr(module, a)


def unpatch_adaptive_avgpool2d(module: nn.AdaptiveAvgPool2d) -> None:
    if hasattr(module, DC_ORIGINAL_FORWARD):
        module.forward = getattr(module, DC_ORIGINAL_FORWARD)
        for a in [DC_ORIGINAL_FORWARD, DC_ENABLED, DC_IS_OUTPUT_LAYER]:
            if hasattr(module, a): delattr(module, a)


def unpatch_adaptive_avgpool1d(module: nn.AdaptiveAvgPool1d) -> None:
    if hasattr(module, DC_ORIGINAL_FORWARD):
        module.forward = getattr(module, DC_ORIGINAL_FORWARD)
        for a in [DC_ORIGINAL_FORWARD, DC_ENABLED, DC_IS_OUTPUT_LAYER]:
            if hasattr(module, a): delattr(module, a)
