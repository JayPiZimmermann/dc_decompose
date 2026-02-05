"""
DC Decomposition Operations Module

Forward: [4*batch] = [pos; neg; pos; neg]
Backward: [4*batch] = [delta_pp; delta_np; delta_pn; delta_nn]

This module contains layer-level DC operations (linear, conv, relu, etc.).
For model-level utilities (patch_model, prepare_model_for_dc, make_dc_compatible),
import from dc_decompose directly.
"""

from .base import (
    cat2, split2, cat4, split4,
    init_catted, init_pos_neg, InputMode,
    ReLUMode, DCCache, recenter_dc, reconstruct_output,
    DC_ENABLED, DC_ORIGINAL_FORWARD, DC_IS_OUTPUT_LAYER, DC_BETA, DC_RELU_MODE,
    split_input_4, make_output_4, split_grad_4, make_grad_4, make_input_4,
)

from .add import Add, DCAdd, dc_add, DCAddFunction, patch_add, unpatch_add

from .linear import patch_linear, unpatch_linear, DCLinearFunction
from .conv import (
    patch_conv2d, unpatch_conv2d, DCConv2dFunction,
    patch_conv1d, unpatch_conv1d, DCConv1dFunction,
)
from .conv_transpose import (
    patch_conv_transpose1d, patch_conv_transpose2d,
    unpatch_conv_transpose1d, unpatch_conv_transpose2d,
    DCConvTranspose1dFunction, DCConvTranspose2dFunction,
)
from .relu import patch_relu, unpatch_relu, DCReLUFunction
from .batchnorm import patch_batchnorm, unpatch_batchnorm, DCBatchNormFunction
from .maxpool import (
    patch_maxpool1d, patch_maxpool2d,
    unpatch_maxpool1d, unpatch_maxpool2d,
    DCMaxPool1dFunction, DCMaxPool2dFunction,
)
from .avgpool import (
    patch_avgpool1d, patch_avgpool2d,
    unpatch_avgpool1d, unpatch_avgpool2d,
    patch_adaptive_avgpool1d, patch_adaptive_avgpool2d,
    unpatch_adaptive_avgpool1d, unpatch_adaptive_avgpool2d,
    DCAvgPool1dFunction, DCAvgPool2dFunction,
    DCAdaptiveAvgPool1dFunction, DCAdaptiveAvgPool2dFunction,
)
from .shape_ops import (
    patch_flatten, unpatch_flatten, DCFlattenFunction,
    patch_unflatten, unpatch_unflatten, DCUnflattenFunction,
    Reshape, patch_reshape, unpatch_reshape, DCReshapeFunction,
    View, patch_view, unpatch_view,
    Squeeze, patch_squeeze, unpatch_squeeze, DCSqueezeFunction,
    Unsqueeze, patch_unsqueeze, unpatch_unsqueeze, DCUnsqueezeFunction,
    Transpose, patch_transpose, unpatch_transpose, DCTransposeFunction,
    Permute, patch_permute, unpatch_permute, DCPermuteFunction,
    patch_dropout, unpatch_dropout, DCDropoutFunction,
)
from .layernorm import patch_layernorm, unpatch_layernorm
from .softmax import patch_softmax, unpatch_softmax

__all__ = [
    # Base
    'cat2', 'split2', 'cat4', 'split4',
    'init_catted', 'init_pos_neg', 'InputMode',
    'ReLUMode', 'DCCache', 'recenter_dc', 'reconstruct_output',
    'DC_ENABLED', 'DC_ORIGINAL_FORWARD', 'DC_IS_OUTPUT_LAYER', 'DC_BETA', 'DC_RELU_MODE',
    'split_input_4', 'make_output_4', 'split_grad_4', 'make_grad_4', 'make_input_4',
    # Add
    'Add', 'DCAdd', 'dc_add', 'DCAddFunction', 'patch_add', 'unpatch_add',
    # Linear
    'patch_linear', 'unpatch_linear', 'DCLinearFunction',
    # Conv
    'patch_conv2d', 'unpatch_conv2d', 'DCConv2dFunction',
    'patch_conv1d', 'unpatch_conv1d', 'DCConv1dFunction',
    # ConvTranspose
    'patch_conv_transpose1d', 'patch_conv_transpose2d',
    'unpatch_conv_transpose1d', 'unpatch_conv_transpose2d',
    'DCConvTranspose1dFunction', 'DCConvTranspose2dFunction',
    # ReLU
    'patch_relu', 'unpatch_relu', 'DCReLUFunction',
    # BatchNorm
    'patch_batchnorm', 'unpatch_batchnorm', 'DCBatchNormFunction',
    # MaxPool
    'patch_maxpool1d', 'patch_maxpool2d',
    'unpatch_maxpool1d', 'unpatch_maxpool2d',
    'DCMaxPool1dFunction', 'DCMaxPool2dFunction',
    # AvgPool
    'patch_avgpool1d', 'patch_avgpool2d',
    'unpatch_avgpool1d', 'unpatch_avgpool2d',
    'patch_adaptive_avgpool1d', 'patch_adaptive_avgpool2d',
    'unpatch_adaptive_avgpool1d', 'unpatch_adaptive_avgpool2d',
    'DCAvgPool1dFunction', 'DCAvgPool2dFunction',
    'DCAdaptiveAvgPool1dFunction', 'DCAdaptiveAvgPool2dFunction',
    # Shape ops
    'patch_flatten', 'unpatch_flatten', 'DCFlattenFunction',
    'patch_unflatten', 'unpatch_unflatten', 'DCUnflattenFunction',
    'Reshape', 'patch_reshape', 'unpatch_reshape', 'DCReshapeFunction',
    'View', 'patch_view', 'unpatch_view',
    'Squeeze', 'patch_squeeze', 'unpatch_squeeze', 'DCSqueezeFunction',
    'Unsqueeze', 'patch_unsqueeze', 'unpatch_unsqueeze', 'DCUnsqueezeFunction',
    'Transpose', 'patch_transpose', 'unpatch_transpose', 'DCTransposeFunction',
    'Permute', 'patch_permute', 'unpatch_permute', 'DCPermuteFunction',
    'patch_dropout', 'unpatch_dropout', 'DCDropoutFunction',
    # LayerNorm
    'patch_layernorm', 'unpatch_layernorm',
    # Softmax
    'patch_softmax', 'unpatch_softmax',
]
