"""DC Decomposition for AvgPool layers. Forward/Backward: [4*batch] -> [4*batch]

AvgPool is linear, so we apply the same operation to both pos and neg streams.
Since it's linear, we let autograd handle the backward pass naturally.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .base import split_input_4, make_output_4
from .patch_builder import create_patch_function, create_unpatch_function


# For linear operations like AvgPool, we don't need custom autograd.Function
# We just apply the operation to each component and let autograd handle backward

# Placeholder classes for backward compatibility
class DCAvgPool2dFunction:
    """Placeholder for backward compatibility. AvgPool uses autograd directly."""
    pass


class DCAvgPool1dFunction:
    """Placeholder for backward compatibility. AvgPool uses autograd directly."""
    pass


class DCAdaptiveAvgPool2dFunction:
    """Placeholder for backward compatibility. AvgPool uses autograd directly."""
    pass


class DCAdaptiveAvgPool1dFunction:
    """Placeholder for backward compatibility. AvgPool uses autograd directly."""
    pass

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


# Patch/unpatch functions
patch_avgpool2d = create_patch_function(dc_forward_avgpool2d)
patch_avgpool1d = create_patch_function(dc_forward_avgpool1d)
patch_adaptive_avgpool2d = create_patch_function(dc_forward_adaptive_avgpool2d)
patch_adaptive_avgpool1d = create_patch_function(dc_forward_adaptive_avgpool1d)

unpatch_avgpool2d = create_unpatch_function()
unpatch_avgpool1d = create_unpatch_function()
unpatch_adaptive_avgpool2d = create_unpatch_function()
unpatch_adaptive_avgpool1d = create_unpatch_function()
