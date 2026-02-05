"""Replace inline module construction with registered submodules.

This module handles patterns like:
- nn.Dropout(0.1)(x) -> self.dropout_0(x) 
- nn.BatchNorm1d(64)(x) -> self.batchnorm1d_0(x)
- nn.ReLU()(x) -> self.relu_0(x)

This preprocessing step is necessary before torch.fx tracing can work properly
for the functional replacer.
"""

import ast
import inspect
import textwrap
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any, Optional


class InlineModuleReplacer(ast.NodeTransformer):
    """AST transformer that replaces inline module construction with submodules."""
    
    def __init__(self):
        self.inline_modules: Dict[str, Tuple[str, Any, Dict[str, Any]]] = {}
        self.counter = 0
    
    def visit_Call(self, node: ast.Call) -> Any:
        """Visit Call nodes to find inline module construction patterns."""
        self.generic_visit(node)
        
        # Debug: print all call nodes
        # print(f"DEBUG: Visiting call node: {ast.unparse(node)}")
        
        # Check if this is a call to a call (module construction then call)
        if (isinstance(node.func, ast.Call) and 
            isinstance(node.func.func, ast.Attribute) and
            isinstance(node.func.func.value, ast.Name) and
            node.func.func.value.id == 'nn'):
            
            # This is nn.SomeModule(args)(x) pattern
            module_name = node.func.func.attr
            module_args = node.func.args
            module_kwargs = node.func.keywords
            
            if self._is_supported_inline_module(module_name):
                # Generate a unique submodule name
                submodule_name = f'_inline_{module_name.lower()}_{self.counter}'
                self.counter += 1
                
                # Store the module info for later registration
                args = [self._ast_to_value(arg) for arg in module_args]
                kwargs = {kw.arg: self._ast_to_value(kw.value) for kw in module_kwargs}
                
                self.inline_modules[submodule_name] = (module_name, args, kwargs)
                
                # Replace the call with self.submodule_name(x)
                return ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id='self', ctx=ast.Load()),
                        attr=submodule_name,
                        ctx=ast.Load()
                    ),
                    args=node.args,
                    keywords=node.keywords
                )
        
        return node
    
    def _is_supported_inline_module(self, module_name: str) -> bool:
        """Check if the module type is supported for inline replacement."""
        supported_modules = {
            'Dropout', 'Dropout1d', 'Dropout2d',
            'BatchNorm1d', 'BatchNorm2d', 'BatchNorm3d',
            'LayerNorm', 'GroupNorm', 'InstanceNorm1d', 'InstanceNorm2d',
            'ReLU', 'GELU', 'Sigmoid', 'Tanh', 'LeakyReLU', 'ELU', 'SELU',
            'Softmax', 'LogSoftmax', 'Softplus', 'Softsign',
            'MaxPool1d', 'MaxPool2d', 'MaxPool3d',
            'AvgPool1d', 'AvgPool2d', 'AvgPool3d',
            'AdaptiveMaxPool1d', 'AdaptiveMaxPool2d', 'AdaptiveMaxPool3d',
            'AdaptiveAvgPool1d', 'AdaptiveAvgPool2d', 'AdaptiveAvgPool3d',
            'Flatten', 'Unflatten'
        }
        return module_name in supported_modules
    
    def _ast_to_value(self, node: ast.AST) -> Any:
        """Convert simple AST nodes to Python values."""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Num):  # Python < 3.8 compatibility
            return node.n
        elif isinstance(node, ast.Str):  # Python < 3.8 compatibility  
            return node.s
        elif isinstance(node, ast.NameConstant):  # Python < 3.8 compatibility
            return node.value
        elif isinstance(node, ast.Name):
            # For simple names like True, False, None
            if node.id == 'True':
                return True
            elif node.id == 'False':
                return False
            elif node.id == 'None':
                return None
        elif isinstance(node, ast.List):
            return [self._ast_to_value(elt) for elt in node.elts]
        elif isinstance(node, ast.Tuple):
            return tuple(self._ast_to_value(elt) for elt in node.elts)
        
        # For complex expressions, return a placeholder
        return f"<complex_expr_{id(node)}>"


def replace_inline_modules(model: nn.Module, inplace: bool = False) -> nn.Module:
    """
    Replace inline module construction with registered submodules.
    
    Args:
        model: The model to transform
        inplace: If True, modify the model in place
        
    Returns:
        Model with inline modules replaced by registered submodules
    """
    if not inplace:
        import copy
        model = copy.deepcopy(model)
    
    # Process all modules recursively
    _replace_inline_modules_in_module(model)
    
    return model


def _replace_inline_modules_in_module(module: nn.Module) -> None:
    """Replace inline modules in a single module and all its children."""
    
    # First process all child modules
    for child in module.children():
        _replace_inline_modules_in_module(child)
    
    # Then process this module
    try:
        _transform_module_inline_modules(module)
    except Exception:
        # If transformation fails, skip this module
        pass


def _transform_module_inline_modules(module: nn.Module) -> None:
    """Transform inline modules in a single module's forward method."""
    
    # Skip certain module types that don't need transformation
    skip_types = (
        nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d,
        nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
        nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
        nn.LayerNorm, nn.GroupNorm,
        nn.ReLU, nn.GELU, nn.Sigmoid, nn.Tanh,
        nn.MaxPool1d, nn.MaxPool2d, nn.AvgPool1d, nn.AvgPool2d,
        nn.Dropout, nn.Flatten, nn.Unflatten
    )
    
    if isinstance(module, skip_types):
        return
    
    # Get the source code of the forward method
    try:
        forward_method = module.forward
        if hasattr(forward_method, '__func__'):
            source = inspect.getsource(forward_method.__func__)
        else:
            source = inspect.getsource(forward_method)
    except (OSError, TypeError):
        return
    
    # Parse the source code
    try:
        # Remove indentation to make it parseable
        source = textwrap.dedent(source)
        tree = ast.parse(source)
    except SyntaxError:
        return
    
    # Transform the AST
    transformer = InlineModuleReplacer()
    new_tree = transformer.visit(tree)
    
    # If no inline modules were found, skip
    if not transformer.inline_modules:
        return
    
    # Register the inline modules as submodules
    for submodule_name, (module_name, args, kwargs) in transformer.inline_modules.items():
        try:
            # Create the module instance
            module_class = getattr(nn, module_name)
            
            # Filter out complex expressions from args/kwargs
            clean_args = [arg for arg in args if not isinstance(arg, str) or not arg.startswith('<complex_expr_')]
            clean_kwargs = {k: v for k, v in kwargs.items() if not isinstance(v, str) or not v.startswith('<complex_expr_')}
            
            inline_module = module_class(*clean_args, **clean_kwargs)
            module.add_module(submodule_name, inline_module)
        except Exception:
            # If module creation fails, skip this one
            continue
    
    # Compile and replace the forward method
    try:
        # Fix indentation for the method
        new_source = ast.unparse(new_tree)
        
        # Create a new forward method
        namespace = {'nn': nn, 'torch': torch}
        
        # Add the module's current namespace for any references
        if hasattr(module, '__dict__'):
            namespace.update(module.__dict__)
        
        # Execute the new code
        exec(new_source, namespace)
        
        # Replace the forward method
        if 'forward' in namespace:
            new_forward = namespace['forward']
            
            # Bind the method to the instance
            import types
            module.forward = types.MethodType(new_forward, module)
            
    except Exception:
        # If compilation fails, keep the original method but the submodules are still registered
        pass


def make_inline_module_compatible(model: nn.Module) -> nn.Module:
    """
    Convenience function to make a model compatible with inline modules.
    
    Args:
        model: The model to transform
        
    Returns:
        Model with inline modules replaced
    """
    return replace_inline_modules(model, inplace=False)