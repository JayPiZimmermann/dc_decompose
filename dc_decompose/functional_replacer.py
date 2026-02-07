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
from torch.fx import symbolic_trace, GraphModule, Node, Tracer
from typing import Dict, Any, Optional, Tuple, Set

from .operations.add import Add
from .operations.shape_ops import Reshape, View, Squeeze, Unsqueeze, Transpose, Permute
from .operations.matmul import DCMatMul
from .operations.mul import DCMul
from .operations.tensor_ops import DCCat
from .operations.base import DC_ENABLED, DC_ORIGINAL_FORWARD, DC_IS_OUTPUT_LAYER
from .operations.base import split_input_4, make_output_4, split_grad_4, make_grad_4
from .inline_module_replacer import replace_inline_modules


class LeafModuleTracer(Tracer):
    """
    Custom tracer that treats specified modules as leaf modules.

    This prevents fx from inlining the forward methods of modules that have
    already been transformed by the functional replacer.
    """

    def __init__(self, leaf_module_ids: Set[int] = None):
        super().__init__()
        self.leaf_module_ids = leaf_module_ids or set()

    def is_leaf_module(self, m: nn.Module, module_qualified_name: str) -> bool:
        """
        Mark modules as leaf if:
        1. They're in the leaf_module_ids set (already transformed)
        2. They're standard leaf modules (no children, or built-in types)
        """
        # Check if this specific module instance was already transformed
        if id(m) in self.leaf_module_ids:
            return True

        # Default leaf module check
        return super().is_leaf_module(m, module_qualified_name)


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

    # First, replace inline modules (nn.Dropout()(x) -> self.dropout_0(x))
    replace_inline_modules(model, inplace=True)

    # Then, duplicate any modules used multiple times
    _duplicate_shared_modules(model)

    # Replace functional calls in all submodules recursively
    _replace_functional_in_module(model)

    return model


def _duplicate_shared_modules(model: nn.Module) -> None:
    """
    Duplicate modules that are used multiple times in the computation graph.

    This is necessary for DC decomposition because each module instance should
    only process one input stream. When a module is used in multiple places
    (e.g., shared weights), we need separate instances for DC to work correctly.
    """
    import copy

    # Track module usage by tracing
    try:
        traced = symbolic_trace(model)
    except Exception:
        # If tracing fails, skip duplication
        return

    # Count how many times each module is called
    module_call_counts: Dict[str, int] = {}
    for node in traced.graph.nodes:
        if node.op == 'call_module':
            target = node.target
            module_call_counts[target] = module_call_counts.get(target, 0) + 1

    # Find modules called more than once
    shared_modules = {name for name, count in module_call_counts.items() if count > 1}

    if not shared_modules:
        return

    # For each shared module, we need to create copies and update the graph
    # This is complex with fx, so we'll use a simpler approach:
    # Just mark them and let the user know (for now, skip complex duplication)
    pass


def _replace_functional_in_module(module: nn.Module, processed: set = None) -> None:
    """Recursively replace functional calls in a module's forward method."""
    if processed is None:
        processed = set()

    # Process all child modules first (depth-first)
    for name, child in module.named_children():
        _replace_functional_in_module(child, processed)
        processed.add(id(child))

    # Try to trace and transform this module's forward
    try:
        _transform_module_forward(module, processed)
    except Exception:
        # If tracing fails (e.g., dynamic control flow), skip this module
        pass


def _transform_module_forward(module: nn.Module, processed: set = None) -> None:
    """Transform a single module's forward method using torch.fx."""
    if processed is None:
        processed = set()

    # Skip modules that are already the target types
    skip_types = (
        nn.ReLU, nn.GELU, nn.Sigmoid, nn.Tanh, nn.Softmax,
        nn.Flatten, nn.Unflatten,
        Add, Reshape, View, Squeeze, Unsqueeze, Transpose, Permute,
        DCMatMul, DCMul, DCCat,
    )
    if isinstance(module, skip_types):
        return

    # Skip leaf modules that don't have submodules
    if len(list(module.children())) == 0 and not _has_functional_calls(module):
        return

    try:
        tracer = LeafModuleTracer(leaf_module_ids=processed)
        graph = tracer.trace(module)
        traced = GraphModule(module, graph)
    except Exception:
        return

    # Track new modules to add
    new_modules: Dict[str, nn.Module] = {}
    counter = {
        'relu': 0, 'gelu': 0, 'sigmoid': 0, 'tanh': 0, 'softmax': 0, 'add': 0,
        'mul': 0, 'mean': 0, 'matmul': 0, 'cat': 0,
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
                    node.args = tuple(node.args[:2])
                else:
                    node.args = (node.args[0],) if node.args else ()
                node.kwargs = {}
        elif node.op == 'call_method':
            replacement = _get_method_replacement(node, counter, new_modules)
            if replacement:
                node.op = 'call_module'
                node.target = replacement
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
            '+ identity', '+ x', 'out +', '+ out',
            'torch.add', 'torch.cat',
            'torch.flatten', 'torch.reshape', 'torch.squeeze', 'torch.unsqueeze',
            'torch.transpose', 'torch.permute',
            '.flatten()', '.reshape(', '.view(', '.squeeze(', '.unsqueeze(',
            '.transpose(', '.permute(',
        ]
        return any(p in source for p in functional_patterns)
    except Exception:
        return False


class Mean(nn.Module):
    """Module wrapper for torch.mean with dim argument."""
    def __init__(self, dim=None, keepdim=False):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        if self.dim is None:
            return torch.mean(x)
        return torch.mean(x, dim=self.dim, keepdim=self.keepdim)


class Mul(nn.Module):
    """Module wrapper for element-wise multiplication."""
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return x * y


def _get_module_replacement(node: Node, counter: Dict[str, int], new_modules: Dict[str, nn.Module]) -> tuple:
    """Get module replacement for a functional call node."""
    func = node.target

    if func in (operator.add, torch.add):
        name = f'_dc_add_{counter["add"]}'
        counter['add'] += 1
        new_modules[name] = Add(recenter=True)
        return name, True

    if func in (operator.mul, torch.mul):
        name = f'_dc_mul_{counter["mul"]}'
        counter['mul'] += 1
        new_modules[name] = DCMul()
        return name, True

    if func == torch.matmul:
        name = f'_dc_matmul_{counter["matmul"]}'
        counter['matmul'] += 1
        new_modules[name] = DCMatMul()
        return name, True  # Keep both arguments for torch.matmul(A, B)

    if func in (torch.mean,):
        name = f'_dc_mean_{counter["mean"]}'
        counter['mean'] += 1
        dim = node.kwargs.get('dim', node.args[1] if len(node.args) > 1 else None)
        keepdim = node.kwargs.get('keepdim', node.args[2] if len(node.args) > 2 else False)
        new_modules[name] = Mean(dim=dim, keepdim=keepdim)
        return name, False

    if func == torch.cat:
        name = f'_dc_cat_{counter["cat"]}'
        counter['cat'] += 1
        dim = node.kwargs.get('dim', node.args[1] if len(node.args) > 1 else 0)
        new_modules[name] = DCCat(dim=dim)
        return name, True  # Keep first argument (list of tensors)

    if func in (torch.relu, F.relu, torch.relu_):
        name = f'_dc_relu_{counter["relu"]}'
        counter['relu'] += 1
        inplace = node.kwargs.get('inplace', False) or (len(node.args) > 1 and node.args[1])
        new_modules[name] = nn.ReLU(inplace=inplace)
        return name, False

    if func in (torch.nn.functional.gelu, F.gelu) or (hasattr(torch, 'gelu') and func == torch.gelu):
        name = f'_dc_gelu_{counter["gelu"]}'
        counter['gelu'] += 1
        new_modules[name] = nn.GELU()
        return name, False

    if func in (torch.sigmoid, F.sigmoid):
        name = f'_dc_sigmoid_{counter["sigmoid"]}'
        counter['sigmoid'] += 1
        new_modules[name] = nn.Sigmoid()
        return name, False

    if func in (torch.tanh, F.tanh):
        name = f'_dc_tanh_{counter["tanh"]}'
        counter['tanh'] += 1
        new_modules[name] = nn.Tanh()
        return name, False

    if func in (torch.softmax, F.softmax):
        name = f'_dc_softmax_{counter["softmax"]}'
        counter['softmax'] += 1
        dim = node.kwargs.get('dim', node.args[1] if len(node.args) > 1 else -1)
        new_modules[name] = nn.Softmax(dim=dim)
        return name, False

    if func == torch.flatten:
        name = f'_dc_flatten_{counter["flatten"]}'
        counter['flatten'] += 1
        start_dim = node.kwargs.get('start_dim', node.args[1] if len(node.args) > 1 else 1)
        end_dim = node.kwargs.get('end_dim', node.args[2] if len(node.args) > 2 else -1)
        new_modules[name] = nn.Flatten(start_dim=start_dim, end_dim=end_dim)
        return name, False

    if func == torch.reshape:
        if len(node.args) > 1:
            shape_arg = node.args[1]
            if isinstance(shape_arg, (tuple, list)) and len(shape_arg) > 0:
                shape = list(shape_arg)
                if isinstance(shape[0], int) and shape[0] > 0:
                    shape[0] = -1
                    node.args = (node.args[0], tuple(shape))
        return None, False

    return None, False


def _extract_value(arg, default=None):
    """Extract actual value from an argument, handling Node objects for constants."""
    if arg is None:
        return default
    if isinstance(arg, Node):
        return None
    return arg


def _get_method_replacement(node: Node, counter: Dict[str, int], new_modules: Dict[str, nn.Module]) -> Optional[str]:
    """Get module replacement for a method call node."""
    method_name = node.target
    method_args = node.args[1:] if len(node.args) > 1 else ()

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
        start_dim_raw = node.kwargs.get('start_dim', method_args[0] if len(method_args) > 0 else 1)
        end_dim_raw = node.kwargs.get('end_dim', method_args[1] if len(method_args) > 1 else -1)
        start_dim = _extract_value(start_dim_raw, 1)
        end_dim = _extract_value(end_dim_raw, -1)
        if start_dim is None or end_dim is None:
            return None
        name = f'_dc_flatten_{counter["flatten"]}'
        counter['flatten'] += 1
        new_modules[name] = nn.Flatten(start_dim=start_dim, end_dim=end_dim)
        return name

    if method_name in ('view', 'reshape'):
        if len(method_args) > 0:
            first_arg = _extract_value(method_args[0])
            if isinstance(first_arg, int) and first_arg > 0:
                new_args = list(node.args)
                new_args[1] = -1
                node.args = tuple(new_args)
        return None

    if method_name == 'squeeze':
        dim_raw = node.kwargs.get('dim', method_args[0] if len(method_args) > 0 else None)
        dim = _extract_value(dim_raw, None)
        if isinstance(dim_raw, Node):
            return None
        name = f'_dc_squeeze_{counter["squeeze"]}'
        counter['squeeze'] += 1
        new_modules[name] = Squeeze(dim=dim)
        return name

    if method_name == 'unsqueeze':
        dim_raw = node.kwargs.get('dim', method_args[0] if len(method_args) > 0 else 0)
        dim = _extract_value(dim_raw, 0)
        if dim is None:
            return None
        name = f'_dc_unsqueeze_{counter["unsqueeze"]}'
        counter['unsqueeze'] += 1
        new_modules[name] = Unsqueeze(dim=dim)
        return name

    if method_name == 'transpose':
        dim0_raw = method_args[0] if len(method_args) > 0 else 0
        dim1_raw = method_args[1] if len(method_args) > 1 else 1
        dim0 = _extract_value(dim0_raw, 0)
        dim1 = _extract_value(dim1_raw, 1)
        if dim0 is None or dim1 is None:
            return None
        name = f'_dc_transpose_{counter["transpose"]}'
        counter['transpose'] += 1
        new_modules[name] = Transpose(dim0=dim0, dim1=dim1)
        return name

    if method_name == 'permute':
        dims_raw = method_args if len(method_args) > 0 else ()
        dims = []
        for d in dims_raw:
            val = _extract_value(d)
            if val is None:
                return None
            dims.append(val)
        name = f'_dc_permute_{counter["permute"]}'
        counter['permute'] += 1
        new_modules[name] = Permute(dims=tuple(dims))
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


# ============================================================================
# DC Patch Functions for Mul and Mean
# ============================================================================

class DCMulFunction(torch.autograd.Function):
    """DC element-wise multiplication with proper backward."""

    @staticmethod
    def forward(ctx, x_4: torch.Tensor, y_4: torch.Tensor,
                is_output_layer: bool, beta: float) -> torch.Tensor:
        pos1, neg1 = split_input_4(x_4)
        pos2, neg2 = split_input_4(y_4)

        out_pos = pos1 * pos2 + neg1 * neg2
        out_neg = pos1 * neg2 + neg1 * pos2

        ctx.save_for_backward(pos1, neg1, pos2, neg2)
        ctx.is_output_layer = is_output_layer
        ctx.beta = beta
        return make_output_4(out_pos, out_neg)

    @staticmethod
    def backward(ctx, grad_4: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, None, None]:
        pos1, neg1, pos2, neg2 = ctx.saved_tensors

        if ctx.is_output_layer:
            q = grad_4.shape[0] // 4
            grad_pos = grad_4[:q]
            grad_neg = grad_4[q:2*q]
            delta_pp = ctx.beta * grad_pos
            delta_np = torch.zeros_like(grad_pos)
            delta_pn = (1 - ctx.beta) * grad_neg
            delta_nn = torch.zeros_like(grad_neg)
        else:
            delta_pp, delta_np, delta_pn, delta_nn = split_grad_4(grad_4)

        new_pp_x = delta_pp * pos2 + delta_pn * neg2
        new_np_x = delta_pp * neg2 + delta_pn * pos2
        new_pn_x = delta_np * pos2 + delta_nn * neg2
        new_nn_x = delta_np * neg2 + delta_nn * pos2

        new_pp_y = delta_pp * pos1 + delta_pn * neg1
        new_np_y = delta_pp * neg1 + delta_pn * pos1
        new_pn_y = delta_np * pos1 + delta_nn * neg1
        new_nn_y = delta_np * neg1 + delta_nn * pos1

        return make_grad_4(new_pp_x, new_np_x, new_pn_x, new_nn_x), \
               make_grad_4(new_pp_y, new_np_y, new_pn_y, new_nn_y), None, None


class DCMeanFunction(torch.autograd.Function):
    """DC mean operation with proper backward."""

    @staticmethod
    def forward(ctx, x_4: torch.Tensor, dim, keepdim: bool,
                is_output_layer: bool, beta: float) -> torch.Tensor:
        pos, neg = split_input_4(x_4)

        if dim is None:
            out_pos = torch.mean(pos)
            out_neg = torch.mean(neg)
        else:
            out_pos = torch.mean(pos, dim=dim, keepdim=keepdim)
            out_neg = torch.mean(neg, dim=dim, keepdim=keepdim)

        ctx.input_shape = pos.shape
        ctx.dim = dim
        ctx.keepdim = keepdim
        ctx.is_output_layer = is_output_layer
        ctx.beta = beta
        return make_output_4(out_pos, out_neg)

    @staticmethod
    def backward(ctx, grad_4: torch.Tensor) -> Tuple[torch.Tensor, None, None, None, None]:
        input_shape = ctx.input_shape
        dim = ctx.dim
        keepdim = ctx.keepdim

        if ctx.is_output_layer:
            q = grad_4.shape[0] // 4
            grad_pos = grad_4[:q]
            grad_neg = grad_4[q:2*q]
            delta_pp = ctx.beta * grad_pos
            delta_np = torch.zeros_like(grad_pos)
            delta_pn = (1 - ctx.beta) * grad_neg
            delta_nn = torch.zeros_like(grad_neg)
        else:
            delta_pp, delta_np, delta_pn, delta_nn = split_grad_4(grad_4)

        if dim is None:
            n = torch.tensor(input_shape).prod().item()
            new_pp = (delta_pp / n).expand(input_shape)
            new_np = (delta_np / n).expand(input_shape)
            new_pn = (delta_pn / n).expand(input_shape)
            new_nn = (delta_nn / n).expand(input_shape)
        else:
            n = input_shape[dim]
            if not keepdim:
                delta_pp = delta_pp.unsqueeze(dim)
                delta_np = delta_np.unsqueeze(dim)
                delta_pn = delta_pn.unsqueeze(dim)
                delta_nn = delta_nn.unsqueeze(dim)
            new_pp = (delta_pp / n).expand(input_shape)
            new_np = (delta_np / n).expand(input_shape)
            new_pn = (delta_pn / n).expand(input_shape)
            new_nn = (delta_nn / n).expand(input_shape)

        return make_grad_4(new_pp, new_np, new_pn, new_nn), None, None, None, None


def dc_forward_mul(m: Mul, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """DC forward for element-wise multiplication."""
    return DCMulFunction.apply(
        x, y,
        getattr(m, DC_IS_OUTPUT_LAYER, False),
        0.5
    )


def patch_mul(module: Mul) -> None:
    """Patch Mul module for DC decomposition."""
    if hasattr(module, DC_ORIGINAL_FORWARD):
        return
    setattr(module, DC_ORIGINAL_FORWARD, module.forward)
    setattr(module, DC_ENABLED, True)
    setattr(module, DC_IS_OUTPUT_LAYER, False)

    def patched(x, y):
        if getattr(module, DC_ENABLED, False):
            return dc_forward_mul(module, x, y)
        else:
            return getattr(module, DC_ORIGINAL_FORWARD)(x, y)

    module.forward = patched


def unpatch_mul(module: Mul) -> None:
    """Unpatch Mul module."""
    if hasattr(module, DC_ORIGINAL_FORWARD):
        module.forward = getattr(module, DC_ORIGINAL_FORWARD)
        for a in [DC_ORIGINAL_FORWARD, DC_ENABLED, DC_IS_OUTPUT_LAYER]:
            if hasattr(module, a):
                delattr(module, a)


def dc_forward_mean(m: Mean, x: torch.Tensor) -> torch.Tensor:
    """DC forward for mean operation."""
    return DCMeanFunction.apply(
        x, m.dim, m.keepdim,
        getattr(m, DC_IS_OUTPUT_LAYER, False),
        0.5
    )


def patch_mean(module: Mean) -> None:
    """Patch Mean module for DC decomposition."""
    if hasattr(module, DC_ORIGINAL_FORWARD):
        return
    setattr(module, DC_ORIGINAL_FORWARD, module.forward)
    setattr(module, DC_ENABLED, True)
    setattr(module, DC_IS_OUTPUT_LAYER, False)

    def patched(x):
        if getattr(module, DC_ENABLED, False):
            return dc_forward_mean(module, x)
        else:
            return getattr(module, DC_ORIGINAL_FORWARD)(x)

    module.forward = patched


def unpatch_mean(module: Mean) -> None:
    """Unpatch Mean module."""
    if hasattr(module, DC_ORIGINAL_FORWARD):
        module.forward = getattr(module, DC_ORIGINAL_FORWARD)
        for a in [DC_ORIGINAL_FORWARD, DC_ENABLED, DC_IS_OUTPUT_LAYER]:
            if hasattr(module, a):
                delattr(module, a)
