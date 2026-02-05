"""
DC Decomposition for Neural Networks

This module implements Difference of Convex (DC) decomposition for neural networks.
The key idea: f(x) = g(x) - h(x) where g and h are monotone subnetworks.

Two APIs are provided:

1. Patcher API (recommended for most use cases):
    from dc_decompose import prepare_model_for_dc, init_catted, reconstruct_output

    model = prepare_model_for_dc(model)
    x_cat = init_catted(x)
    out_cat = model(x_cat)
    out = reconstruct_output(out_cat)

For backward compatibility, import the core utilities:
    from dc_decompose.operations.base import InputMode

Forward format: [4*batch] = [pos; neg; 0; 0]
Backward format: [4*batch] = [delta_pp; delta_np; delta_pn; delta_nn]
"""

# Backward compatibility - legacy enums now in operations.base
# ReLUMode and BackwardMode were part of the old HookDecomposer

# Patcher-based API (model-level)
from .patcher import (
    patch_model, unpatch_model,
    mark_output_layer, unmark_output_layer,
    auto_mark_output_layer, find_output_layer,
    set_dc_enabled, dc_disabled,
    is_patched, get_patched_layers,
    prepare_model_for_dc,
    enable_dc_logging, disable_dc_logging,
)
from .functional_replacer import make_dc_compatible, replace_functional_with_modules, Mul, Mean

# Core utilities from operations
from .operations.base import (
    init_catted, init_pos_neg, reconstruct_output,
    recenter_dc, InputMode as OpInputMode,
)

# Matrix operations
from .operations.matmul import DCMatMul, DCMatMulFunction

# Operations
from .operations.tensor_ops import DCSplit, DCChunk, DCCat, DCSlice
from .operations.shape_ops import Reshape as DCReshape, Transpose as DCTranspose, Permute as DCPermute
from .operations.add import Add as DCAdd
from .operations.gather import Gather as DCGather
from .operations.mean import Mean as DCMean
from .operations.sum import Sum as DCSum
from .operations.contiguous import Contiguous as DCContiguous

__all__ = [
    # Patcher API (recommended)
    "prepare_model_for_dc",
    "enable_dc_logging",
    "disable_dc_logging",
    "patch_model",
    "unpatch_model",
    "mark_output_layer",
    "unmark_output_layer",
    "auto_mark_output_layer",
    "find_output_layer",
    "set_dc_enabled",
    "dc_disabled",
    "is_patched",
    "get_patched_layers",
    "make_dc_compatible",
    "replace_functional_with_modules",
    # Core utilities
    "init_catted",
    "init_pos_neg",
    "reconstruct_output",
    "recenter_dc",
    "InputMode",
    # Functional replacer modules
    "Mul",
    "Mean",
    # MatMul
    "DCMatMul",
    "DCMatMulFunction",
    # Operations
    "DCReshape",
    "DCPermute", 
    "DCTranspose",
    "DCContiguous",
    "DCAdd",
    "DCSplit",
    "DCChunk",
    "DCCat", 
    "DCSlice",
    "DCGather",
    "DCMean",
    "DCSum",
]
