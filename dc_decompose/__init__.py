"""
DC Decomposition using PyTorch Hooks

This module implements Difference of Convex (DC) decomposition for neural networks
using PyTorch forward and backward hooks directly on original modules.

The key idea: f(x) = g(x) - h(x) where g and h are monotone subnetworks.

Each layer maintains:
- pos (positive stream): Non-negative activations
- neg (negative stream): Non-negative activations
- original: The original undecomposed activation

Forward: output = pos - neg = original
Backward: 4 local sensitivities (delta_pp, delta_np, delta_pn, delta_nn)

Shift Modes for input splitting:
- CENTER (default): pos = ReLU(x), neg = ReLU(-x) - ensures non-negativity
- POSITIVE: pos = x, neg = 0 - all in positive stream
- NEGATIVE: pos = 0, neg = -x - all in negative stream
- BETA: pos = beta * x, neg = -(1-beta) * x - configurable split

Usage:
    from dc_decompose import HookDecomposer, ShiftMode

    model = YourModel()
    decomposer = HookDecomposer(model)  # Uses CENTER mode by default

    # Forward pass - hooks compute pos/neg automatically
    decomposer.initialize()
    output = model(x)

    # Access decomposed activations
    for name, cache in decomposer.caches.items():
        pos, neg = cache.output_pos, cache.output_neg

    # Backward pass - compute 4 sensitivities
    decomposer.backward()

    # Access sensitivities
    for name, cache in decomposer.caches.items():
        delta_pp, delta_np, delta_pn, delta_nn = (
            cache.delta_pp, cache.delta_np, cache.delta_pn, cache.delta_nn
        )
"""

"""
DC Decomposition using PyTorch Hooks

This module implements Difference of Convex (DC) decomposition for neural networks
using PyTorch forward and backward hooks directly on original modules.

The key idea: f(x) = g(x) - h(x) where g and h are monotone subnetworks.

Each layer maintains:
- pos (positive stream): Non-negative activations
- neg (negative stream): Non-negative activations
- original: The original undecomposed activation

Forward: output = pos - neg = original
Backward: 4 local sensitivities (delta_pp, delta_np, delta_pn, delta_nn)

Supported Layers:
- Linear, Conv2d (weight decomposition)
- ReLU (multiple modes: MAX, MIN, HALF)
- Softmax (with Jacobian-based backward)
- LayerNorm, BatchNorm2d (variance treated as constant)
- MaxPool2d (winner-takes-all), AvgPool2d, AdaptiveAvgPool2d
- Flatten
- DCMatMul (for explicit matrix multiplication decomposition)

Shift Modes for input splitting:
- CENTER (default): pos = ReLU(x), neg = ReLU(-x) - ensures non-negativity
- POSITIVE: pos = x, neg = 0 - all in positive stream
- NEGATIVE: pos = 0, neg = -x - all in negative stream
- BETA: pos = beta * x, neg = -(1-beta) * x - configurable split

Usage:
    from dc_decompose import HookDecomposer, ShiftMode

    model = YourModel()
    decomposer = HookDecomposer(model)  # Uses CENTER mode by default

    # Forward pass - hooks compute pos/neg automatically
    decomposer.initialize()
    output = model(x)

    # Access decomposed activations
    for name, cache in decomposer.caches.items():
        pos, neg = cache.output_pos, cache.output_neg

    # Backward pass - compute 4 sensitivities
    decomposer.backward()

    # Access sensitivities
    for name, cache in decomposer.caches.items():
        delta_pp, delta_np, delta_pn, delta_nn = (
            cache.delta_pp, cache.delta_np, cache.delta_pn, cache.delta_nn
        )

For matrix multiplications (e.g., attention in transformers):
    from dc_decompose import DCMatMulFunction

    # (A+ - A-)(B+ - B-) = (A+B+ + A-B-) - (A+B- + A-B+)
    output_pos, output_neg = DCMatMulFunction.forward(A_pos, A_neg, B_pos, B_neg)
"""

from .hook_decomposer import HookDecomposer, InputMode, BackwardMode, ReLUMode, DCCache
from .dc_matmul import DCMatMul, DCMatMulFunction
from .dc_operations import (
    DCReshape, DCDynamicReshape, DCPermute, DCTranspose, DCContiguous,
    DCScalarMul, DCScalarDiv, DCAdd, DCSplit, DCChunk, DCCat, DCSlice,
    DCDropout, DCIdentity, DCEmbedding, DCGather, DCMean, DCSum,
)

__all__ = [
    # Core
    "HookDecomposer",
    "ShiftMode",
    "ReLUMode",
    "DCCache",
    # MatMul
    "DCMatMul",
    "DCMatMulFunction",
    # Operations
    "DCReshape",
    "DCDynamicReshape",
    "DCPermute",
    "DCTranspose",
    "DCContiguous",
    "DCScalarMul",
    "DCScalarDiv",
    "DCAdd",
    "DCSplit",
    "DCChunk",
    "DCCat",
    "DCSlice",
    "DCDropout",
    "DCIdentity",
    "DCEmbedding",
    "DCGather",
    "DCMean",
    "DCSum",
]
