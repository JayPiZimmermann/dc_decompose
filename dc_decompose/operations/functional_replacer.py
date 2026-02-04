"""Replace functional calls with module equivalents for DC decomposition.

This module uses torch.fx to trace models and replace:
- torch.relu, F.relu, x.relu() -> nn.ReLU()
- torch.add, operator.add (+) -> Add() (for DC re-centering in residuals)
- Other activations (gelu, sigmoid, tanh, softmax)
"""

import operator
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fx import symbolic_trace, GraphModule, Node
from typing import Dict, Any, Optional, Tuple

from .add import Add
from .shape_ops import Reshape, View, Squeeze, Unsqueeze, Transpose, Permute


def replace_functional_with_modules(model: nn.Module, inplace: bool = False) -> nn.Module:
    """
    Replace functional calls (torch.relu, F.relu, etc.) with module equivalents.

    This is necessary for DC decomposition because hooks are attached to modules,
    not functional calls.

    Args:
        model: The model to transform
        inplace: If True, modify the model in place. If False, return a new model.

    Returns:
        Transformed model with functional calls replaced by modules
    """
    if not inplace:
        import copy
        model = copy.deepcopy(model)

    # Replace functional calls in all submodules recursively
    _replace_functional_in_module(model)

    return model


def _replace_functional_in_module(module: nn.Module) -> None:
    """Recursively replace functional calls in a module's forward method."""
    # Process all child modules first
    for name, child in module.named_children():
        _replace_functional_in_module(child)

    # Try to trace and transform this module's forward
    try:
        _transform_module_forward(module)
    except Exception:
        # If tracing fails (e.g., dynamic control flow), skip this module
        pass


def _transform_module_forward(module: nn.Module) -> None:
    """Transform a single module's forward method using torch.fx."""
    # Skip leaf modules that don't have submodules
    if len(list(module.children())) == 0 and not _has_functional_calls(module):
        return

    try:
        # Trace the module
        traced = symbolic_trace(module)
    except Exception:
        return

    # Track new modules to add
    new_modules: Dict[str, nn.Module] = {}
    counter = {
        'relu': 0, 'gelu': 0, 'sigmoid': 0, 'tanh': 0, 'softmax': 0, 'add': 0,
        'flatten': 0, 'reshape': 0, 'view': 0, 'squeeze': 0, 'unsqueeze': 0,
        'transpose': 0, 'permute': 0
    }

    # Transform the graph
    for node in traced.graph.nodes:
        if node.op == 'call_function':
            replacement, keep_all_args = _get_module_replacement(node, counter, new_modules)
            if replacement:
                node.op = 'call_module'
                node.target = replacement
                if keep_all_args:
                    # Keep both arguments (for binary ops like add)
                    node.args = tuple(node.args[:2])
                else:
                    # Keep only the first argument (for unary ops like relu)
                    node.args = (node.args[0],) if node.args else ()
                node.kwargs = {}
        elif node.op == 'call_method':
            replacement = _get_method_replacement(node, counter, new_modules)
            if replacement:
                node.op = 'call_module'
                node.target = replacement
                # The self argument becomes the input
                node.args = (node.args[0],) if node.args else ()
                node.kwargs = {}

    # Add new modules to the traced module
    for name, mod in new_modules.items():
        traced.add_module(name, mod)

    # Recompile the graph
    traced.graph.lint()
    traced.recompile()

    # Replace the module's forward with the traced version
    module.forward = traced.forward

    # Copy the new submodules
    for name, mod in new_modules.items():
        module.add_module(name, mod)


def _has_functional_calls(module: nn.Module) -> bool:
    """Check if a module's forward likely contains functional calls."""
    import inspect
    try:
        source = inspect.getsource(module.forward)
        functional_patterns = [
            'torch.relu', 'F.relu', 'torch.nn.functional.relu',
            'torch.gelu', 'F.gelu',
            'torch.sigmoid', 'F.sigmoid',
            'torch.tanh', 'F.tanh',
            'torch.softmax', 'F.softmax',
            '.relu()', '.sigmoid()', '.tanh()',
            # Addition patterns (for residual connections)
            '+ identity', '+ x', 'out +', '+ out',
            'torch.add',
            # Shape operations
            'torch.flatten', 'torch.reshape', 'torch.squeeze', 'torch.unsqueeze',
            'torch.transpose', 'torch.permute',
            '.flatten()', '.reshape(', '.view(', '.squeeze(', '.unsqueeze(',
            '.transpose(', '.permute(',
        ]
        return any(p in source for p in functional_patterns)
    except Exception:
        return False


def _get_module_replacement(node: Node, counter: Dict[str, int], new_modules: Dict[str, nn.Module]) -> tuple:
    """
    Get module replacement for a functional call node.

    Returns:
        (module_name, keep_all_args) tuple, or (None, False) if no replacement
    """
    func = node.target

    # Check for addition (+ operator or torch.add)
    if func in (operator.add, torch.add):
        name = f'_dc_add_{counter["add"]}'
        counter['add'] += 1
        new_modules[name] = Add(recenter=True)
        return name, True  # Keep both arguments

    # Check for relu
    if func in (torch.relu, F.relu, torch.relu_):
        name = f'_dc_relu_{counter["relu"]}'
        counter['relu'] += 1
        inplace = node.kwargs.get('inplace', False) or (len(node.args) > 1 and node.args[1])
        new_modules[name] = nn.ReLU(inplace=inplace)
        return name, False

    # Check for gelu
    if func in (torch.nn.functional.gelu, F.gelu) or (hasattr(torch, 'gelu') and func == torch.gelu):
        name = f'_dc_gelu_{counter["gelu"]}'
        counter['gelu'] += 1
        new_modules[name] = nn.GELU()
        return name, False

    # Check for sigmoid
    if func in (torch.sigmoid, F.sigmoid):
        name = f'_dc_sigmoid_{counter["sigmoid"]}'
        counter['sigmoid'] += 1
        new_modules[name] = nn.Sigmoid()
        return name, False

    # Check for tanh
    if func in (torch.tanh, F.tanh):
        name = f'_dc_tanh_{counter["tanh"]}'
        counter['tanh'] += 1
        new_modules[name] = nn.Tanh()
        return name, False

    # Check for softmax
    if func in (torch.softmax, F.softmax):
        name = f'_dc_softmax_{counter["softmax"]}'
        counter['softmax'] += 1
        dim = node.kwargs.get('dim', node.args[1] if len(node.args) > 1 else -1)
        new_modules[name] = nn.Softmax(dim=dim)
        return name, False

    # torch.reshape: Check if batch dimension is hardcoded and needs to be made adaptive
    if func == torch.reshape:
        if len(node.args) > 1:
            shape_arg = node.args[1]
            if isinstance(shape_arg, (tuple, list)) and len(shape_arg) > 0:
                shape = list(shape_arg)
                # If first element is a hardcoded positive integer (batch dim), replace with -1
                if isinstance(shape[0], int) and shape[0] > 0:
                    shape[0] = -1
                    node.args = (node.args[0], tuple(shape))
        return None, False  # Don't replace with module

    # Other shape ops (flatten, squeeze, unsqueeze, transpose, permute) don't need modification
    # as they either don't touch batch dim or use relative dimensions

    return None, False


def _get_method_replacement(node: Node, counter: Dict[str, int], new_modules: Dict[str, nn.Module]) -> Optional[str]:
    """Get module replacement for a method call node (e.g., x.relu())."""
    method_name = node.target

    if method_name == 'relu':
        name = f'_dc_relu_{counter["relu"]}'
        counter['relu'] += 1
        new_modules[name] = nn.ReLU()
        return name

    if method_name == 'sigmoid':
        name = f'_dc_sigmoid_{counter["sigmoid"]}'
        counter['sigmoid'] += 1
        new_modules[name] = nn.Sigmoid()
        return name

    if method_name == 'tanh':
        name = f'_dc_tanh_{counter["tanh"]}'
        counter['tanh'] += 1
        new_modules[name] = nn.Tanh()
        return name

    if method_name == 'flatten':
        name = f'_dc_flatten_{counter["flatten"]}'
        counter['flatten'] += 1
        start_dim = node.kwargs.get('start_dim', node.args[0] if len(node.args) > 0 else 1)
        end_dim = node.kwargs.get('end_dim', node.args[1] if len(node.args) > 1 else -1)
        new_modules[name] = nn.Flatten(start_dim=start_dim, end_dim=end_dim)
        return name

    # view and reshape: Check if batch dimension is hardcoded and needs to be made adaptive
    if method_name in ('view', 'reshape'):
        if len(node.args) > 0:
            args = list(node.args)
            # If first arg is a hardcoded positive integer (batch dim), replace with -1
            if isinstance(args[0], int) and args[0] > 0:
                args[0] = -1
                node.args = tuple(args)
        return None  # Don't replace with module, just fix the args

    # transpose, permute, squeeze, unsqueeze don't touch batch dim, pass through unchanged

    if method_name == 'squeeze':
        name = f'_dc_squeeze_{counter["squeeze"]}'
        counter['squeeze'] += 1
        dim = node.kwargs.get('dim', node.args[0] if len(node.args) > 0 else None)
        new_modules[name] = Squeeze(dim=dim)
        return name

    if method_name == 'unsqueeze':
        name = f'_dc_unsqueeze_{counter["unsqueeze"]}'
        counter['unsqueeze'] += 1
        dim = node.kwargs.get('dim', node.args[0] if len(node.args) > 0 else 0)
        new_modules[name] = Unsqueeze(dim=dim)
        return name

    if method_name == 'transpose':
        name = f'_dc_transpose_{counter["transpose"]}'
        counter['transpose'] += 1
        dim0 = node.args[0] if len(node.args) > 0 else 0
        dim1 = node.args[1] if len(node.args) > 1 else 1
        new_modules[name] = Transpose(dim0=dim0, dim1=dim1)
        return name

    if method_name == 'permute':
        name = f'_dc_permute_{counter["permute"]}'
        counter['permute'] += 1
        dims = node.args if len(node.args) > 0 else ()
        new_modules[name] = Permute(dims=tuple(dims) if not isinstance(dims, tuple) else dims)
        return name

    return None


def make_dc_compatible(model: nn.Module) -> nn.Module:
    """
    Make a model compatible with DC decomposition by replacing functional calls.

    This is a convenience function that applies all necessary transformations.

    Args:
        model: The model to transform

    Returns:
        DC-compatible model
    """
    return replace_functional_with_modules(model, inplace=False)
