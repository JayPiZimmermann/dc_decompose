"""
Logging configuration for DC decomposition debugging.

Channels (loggers):
- dc.forward: Forward pass logging (shapes, module names)
- dc.backward: Backward pass logging (shapes, gradients)
- dc.tensors: Detailed tensor values (very verbose)
- dc.recenter: Re-centering operations
- dc.patch: Patching/unpatching operations
- dc.error: Errors and warnings

Usage:
    from dc_decompose.logging_config import get_logger, enable_logging, set_level

    # Enable all DC logging at INFO level
    enable_logging()

    # Enable specific channels
    enable_logging(channels=['dc.forward', 'dc.backward'])

    # Set level for tensor debugging (very verbose)
    set_level('dc.tensors', logging.DEBUG)

    # Disable logging
    disable_logging()
"""

import logging
import functools
from typing import Optional, List, Union, Callable, Any
from contextlib import contextmanager
import torch
from torch import Tensor


# =============================================================================
# Logger Configuration
# =============================================================================

# All DC logging channels
CHANNELS = [
    'dc',           # Root DC logger
    'dc.forward',   # Forward pass
    'dc.backward',  # Backward pass
    'dc.tensors',   # Tensor values (verbose)
    'dc.recenter',  # Re-centering operations
    'dc.patch',     # Patching operations
    'dc.error',     # Errors
]

# Custom log level for tensor details
TENSOR_LEVEL = 5  # Below DEBUG (10)
logging.addLevelName(TENSOR_LEVEL, 'TENSOR')


class DCFormatter(logging.Formatter):
    """Custom formatter for DC logging with colors for terminal."""

    COLORS = {
        'TENSOR': '\033[90m',   # Gray
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'RESET': '\033[0m',
    }

    def __init__(self, use_colors: bool = True):
        super().__init__()
        self.use_colors = use_colors

    def format(self, record):
        # Extract channel name (e.g., 'dc.forward' -> 'FWD')
        channel_map = {
            'dc.forward': 'FWD',
            'dc.backward': 'BWD',
            'dc.tensors': 'TNS',
            'dc.recenter': 'RCT',
            'dc.patch': 'PCH',
            'dc.error': 'ERR',
            'dc': 'DC',
        }
        channel = channel_map.get(record.name, record.name[-3:].upper())

        level = record.levelname
        msg = record.getMessage()

        if self.use_colors:
            color = self.COLORS.get(level, '')
            reset = self.COLORS['RESET']
            return f"{color}[{channel}] {msg}{reset}"
        else:
            return f"[{channel}] {msg}"


def _setup_logger(name: str) -> logging.Logger:
    """Set up a logger with the DC formatter."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.WARNING)  # Default to WARNING (quiet)
    logger.propagate = False

    # Remove existing handlers
    logger.handlers = []

    # Add console handler
    handler = logging.StreamHandler()
    handler.setFormatter(DCFormatter(use_colors=True))
    logger.addHandler(handler)

    return logger


# Initialize all loggers
_loggers = {name: _setup_logger(name) for name in CHANNELS}


def get_logger(channel: str = 'dc') -> logging.Logger:
    """Get a DC logger by channel name."""
    if channel not in _loggers:
        _loggers[channel] = _setup_logger(channel)
    return _loggers[channel]


def enable_logging(
    level: int = logging.INFO,
    channels: Optional[List[str]] = None,
    include_tensors: bool = False
):
    """
    Enable DC logging.

    Args:
        level: Log level (default INFO)
        channels: Specific channels to enable (default all except tensors)
        include_tensors: Include tensor value logging (very verbose)
    """
    if channels is None:
        channels = [c for c in CHANNELS if c != 'dc.tensors']

    for name in channels:
        if name in _loggers:
            _loggers[name].setLevel(level)

    if include_tensors:
        _loggers['dc.tensors'].setLevel(TENSOR_LEVEL)


def disable_logging():
    """Disable all DC logging."""
    for logger in _loggers.values():
        logger.setLevel(logging.CRITICAL)


def set_level(channel: str, level: int):
    """Set log level for a specific channel."""
    if channel in _loggers:
        _loggers[channel].setLevel(level)


@contextmanager
def logging_context(level: int = logging.DEBUG, channels: Optional[List[str]] = None):
    """Context manager to temporarily enable logging."""
    old_levels = {name: logger.level for name, logger in _loggers.items()}
    enable_logging(level, channels)
    try:
        yield
    finally:
        for name, old_level in old_levels.items():
            _loggers[name].setLevel(old_level)


# =============================================================================
# Tensor Formatting Utilities
# =============================================================================

def format_shape(t: Optional[Tensor]) -> str:
    """Format tensor shape as string."""
    if t is None:
        return "None"
    return str(list(t.shape))


def format_tensor_stats(t: Optional[Tensor], name: str = "") -> str:
    """Format tensor statistics (min, max, mean, std)."""
    if t is None:
        return f"{name}: None"
    with torch.no_grad():
        prefix = f"{name}: " if name else ""
        return (f"{prefix}shape={list(t.shape)}, "
                f"min={t.min().item():.4f}, max={t.max().item():.4f}, "
                f"mean={t.mean().item():.4f}, std={t.std().item():.4f}")


def format_tensor_values(t: Optional[Tensor], name: str = "", max_elements: int = 10) -> str:
    """Format tensor values (truncated for readability)."""
    if t is None:
        return f"{name}: None"
    with torch.no_grad():
        flat = t.flatten()
        n = min(len(flat), max_elements)
        vals = [f"{flat[i].item():.4f}" for i in range(n)]
        suffix = "..." if len(flat) > max_elements else ""
        prefix = f"{name}: " if name else ""
        return f"{prefix}[{', '.join(vals)}{suffix}]"


def format_dc_tensor(t: Optional[Tensor], name: str = "") -> str:
    """Format a [4*batch] DC tensor with pos/neg breakdown."""
    if t is None:
        return f"{name}: None"
    with torch.no_grad():
        q = t.shape[0] // 4
        pos = t[:q]
        neg = t[q:2*q]
        prefix = f"{name}: " if name else ""
        return (f"{prefix}shape={list(t.shape)}, "
                f"pos_mean={pos.mean().item():.4f}, neg_mean={neg.mean().item():.4f}, "
                f"z_mean={(pos - neg).mean().item():.4f}")


# =============================================================================
# Logging Decorators for Forward/Backward
# =============================================================================

def log_forward(module_type: str):
    """
    Decorator to log forward pass of DC operations.

    Usage:
        @log_forward("Conv2d")
        def forward(ctx, input_4, weight, ...):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            fwd_log = get_logger('dc.forward')
            tensor_log = get_logger('dc.tensors')

            # Extract input tensor (usually first arg after ctx for autograd.Function)
            ctx = args[0] if args else None
            input_tensor = args[1] if len(args) > 1 else None

            # Log before
            if fwd_log.isEnabledFor(logging.DEBUG):
                fwd_log.debug(f"{module_type}.forward ENTER: input={format_shape(input_tensor)}")

            if tensor_log.isEnabledFor(TENSOR_LEVEL) and input_tensor is not None:
                tensor_log.log(TENSOR_LEVEL, f"{module_type} input: {format_dc_tensor(input_tensor)}")

            # Execute
            result = func(*args, **kwargs)

            # Log after
            if fwd_log.isEnabledFor(logging.DEBUG):
                fwd_log.debug(f"{module_type}.forward EXIT: output={format_shape(result)}")

            if fwd_log.isEnabledFor(logging.INFO):
                fwd_log.info(f"{module_type}: {format_shape(input_tensor)} -> {format_shape(result)}")

            if tensor_log.isEnabledFor(TENSOR_LEVEL) and result is not None:
                tensor_log.log(TENSOR_LEVEL, f"{module_type} output: {format_dc_tensor(result)}")

            return result
        return wrapper
    return decorator


def log_backward(module_type: str):
    """
    Decorator to log backward pass of DC operations.

    Usage:
        @log_backward("Conv2d")
        def backward(ctx, grad_4):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            bwd_log = get_logger('dc.backward')
            tensor_log = get_logger('dc.tensors')

            ctx = args[0] if args else None
            grad_tensor = args[1] if len(args) > 1 else None

            # Log before
            if bwd_log.isEnabledFor(logging.DEBUG):
                bwd_log.debug(f"{module_type}.backward ENTER: grad={format_shape(grad_tensor)}")

            if tensor_log.isEnabledFor(TENSOR_LEVEL) and grad_tensor is not None:
                tensor_log.log(TENSOR_LEVEL, f"{module_type} grad_in: {format_dc_tensor(grad_tensor)}")

            # Execute
            result = func(*args, **kwargs)

            # Log after
            if bwd_log.isEnabledFor(logging.DEBUG):
                if isinstance(result, tuple):
                    shapes = [format_shape(r) if isinstance(r, Tensor) else str(r) for r in result]
                    bwd_log.debug(f"{module_type}.backward EXIT: grads={shapes}")
                else:
                    bwd_log.debug(f"{module_type}.backward EXIT: grad={format_shape(result)}")

            if bwd_log.isEnabledFor(logging.INFO):
                out_shape = format_shape(result[0]) if isinstance(result, tuple) else format_shape(result)
                bwd_log.info(f"{module_type} bwd: {format_shape(grad_tensor)} -> {out_shape}")

            if tensor_log.isEnabledFor(TENSOR_LEVEL):
                if isinstance(result, tuple) and len(result) > 0 and isinstance(result[0], Tensor):
                    tensor_log.log(TENSOR_LEVEL, f"{module_type} grad_out: {format_dc_tensor(result[0])}")

            return result
        return wrapper
    return decorator


def log_recenter(func: Callable) -> Callable:
    """Decorator to log re-centering operations."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        log = get_logger('dc.recenter')
        tensor_log = get_logger('dc.tensors')

        input_tensor = args[0] if args else None

        if log.isEnabledFor(logging.DEBUG):
            log.debug(f"recenter ENTER: {format_shape(input_tensor)}")

        if tensor_log.isEnabledFor(TENSOR_LEVEL) and input_tensor is not None:
            tensor_log.log(TENSOR_LEVEL, f"recenter input: {format_dc_tensor(input_tensor)}")

        result = func(*args, **kwargs)

        if log.isEnabledFor(logging.INFO) and input_tensor is not None and result is not None:
            with torch.no_grad():
                q = input_tensor.shape[0] // 4
                old_pos_mean = input_tensor[:q].mean().item()
                old_neg_mean = input_tensor[q:2*q].mean().item()
                new_pos_mean = result[:q].mean().item()
                new_neg_mean = result[q:2*q].mean().item()
                log.info(f"recenter: pos {old_pos_mean:.2f}->{new_pos_mean:.2f}, "
                        f"neg {old_neg_mean:.2f}->{new_neg_mean:.2f}")

        if tensor_log.isEnabledFor(TENSOR_LEVEL) and result is not None:
            tensor_log.log(TENSOR_LEVEL, f"recenter output: {format_dc_tensor(result)}")

        return result
    return wrapper


def log_patch(module_type: str, action: str = "patch"):
    """Decorator to log patching operations."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(module, *args, **kwargs):
            log = get_logger('dc.patch')

            module_name = getattr(module, '_dc_name', module.__class__.__name__)

            if log.isEnabledFor(logging.INFO):
                log.info(f"{action} {module_type}: {module_name}")

            return func(module, *args, **kwargs)
        return wrapper
    return decorator


# =============================================================================
# Wrapper for Autograd Functions
# =============================================================================

class LoggedFunction:
    """
    Mixin class for autograd.Function that adds logging.

    Usage:
        class DCConv2dFunction(LoggedFunction, torch.autograd.Function):
            _log_name = "Conv2d"

            @staticmethod
            def forward(ctx, ...):
                ...
    """
    _log_name = "Unknown"

    @classmethod
    def _log_forward(cls, input_tensor, output_tensor):
        fwd_log = get_logger('dc.forward')
        tensor_log = get_logger('dc.tensors')

        if fwd_log.isEnabledFor(logging.INFO):
            fwd_log.info(f"{cls._log_name}: {format_shape(input_tensor)} -> {format_shape(output_tensor)}")

        if tensor_log.isEnabledFor(TENSOR_LEVEL):
            if input_tensor is not None:
                tensor_log.log(TENSOR_LEVEL, f"{cls._log_name} in: {format_dc_tensor(input_tensor)}")
            if output_tensor is not None:
                tensor_log.log(TENSOR_LEVEL, f"{cls._log_name} out: {format_dc_tensor(output_tensor)}")

    @classmethod
    def _log_backward(cls, grad_in, grad_out):
        bwd_log = get_logger('dc.backward')
        tensor_log = get_logger('dc.tensors')

        if bwd_log.isEnabledFor(logging.INFO):
            out_shape = format_shape(grad_out[0]) if isinstance(grad_out, tuple) else format_shape(grad_out)
            bwd_log.info(f"{cls._log_name} bwd: {format_shape(grad_in)} -> {out_shape}")

        if tensor_log.isEnabledFor(TENSOR_LEVEL):
            if grad_in is not None:
                tensor_log.log(TENSOR_LEVEL, f"{cls._log_name} grad_in: {format_dc_tensor(grad_in)}")


def wrap_autograd_function(cls, name: str):
    """
    Wrap an existing autograd.Function class with logging.

    Returns a new class with logging added to forward and backward.
    """
    original_forward = cls.forward
    original_backward = cls.backward

    @staticmethod
    def logged_forward(ctx, *args, **kwargs):
        fwd_log = get_logger('dc.forward')
        tensor_log = get_logger('dc.tensors')

        input_tensor = args[0] if args else None

        if fwd_log.isEnabledFor(logging.DEBUG):
            fwd_log.debug(f"{name}.forward ENTER: input={format_shape(input_tensor)}")

        result = original_forward(ctx, *args, **kwargs)

        if fwd_log.isEnabledFor(logging.INFO):
            fwd_log.info(f"{name}: {format_shape(input_tensor)} -> {format_shape(result)}")

        if tensor_log.isEnabledFor(TENSOR_LEVEL):
            if input_tensor is not None:
                tensor_log.log(TENSOR_LEVEL, f"{name} in: {format_dc_tensor(input_tensor)}")
            if result is not None:
                tensor_log.log(TENSOR_LEVEL, f"{name} out: {format_dc_tensor(result)}")

        return result

    @staticmethod
    def logged_backward(ctx, *args, **kwargs):
        bwd_log = get_logger('dc.backward')
        tensor_log = get_logger('dc.tensors')

        grad_tensor = args[0] if args else None

        if bwd_log.isEnabledFor(logging.DEBUG):
            bwd_log.debug(f"{name}.backward ENTER: grad={format_shape(grad_tensor)}")

        result = original_backward(ctx, *args, **kwargs)

        if bwd_log.isEnabledFor(logging.INFO):
            out_shape = format_shape(result[0]) if isinstance(result, tuple) else format_shape(result)
            bwd_log.info(f"{name} bwd: {format_shape(grad_tensor)} -> {out_shape}")

        if tensor_log.isEnabledFor(TENSOR_LEVEL):
            if grad_tensor is not None:
                tensor_log.log(TENSOR_LEVEL, f"{name} grad_in: {format_dc_tensor(grad_tensor)}")
            if isinstance(result, tuple) and result[0] is not None:
                tensor_log.log(TENSOR_LEVEL, f"{name} grad_out: {format_dc_tensor(result[0])}")

        return result

    cls.forward = logged_forward
    cls.backward = logged_backward
    cls._log_name = name

    return cls


# =============================================================================
# Module-level Logging Hook
# =============================================================================

def add_logging_hooks(model: 'torch.nn.Module', prefix: str = ""):
    """
    Add logging hooks to all modules in a model.

    Returns list of hook handles for removal.
    """
    handles = []
    fwd_log = get_logger('dc.forward')

    def make_hook(name):
        def hook(module, input, output):
            if fwd_log.isEnabledFor(logging.DEBUG):
                in_shape = format_shape(input[0]) if isinstance(input, tuple) and input else "?"
                out_shape = format_shape(output) if isinstance(output, Tensor) else "?"
                fwd_log.debug(f"{name}: {in_shape} -> {out_shape}")
        return hook

    for name, module in model.named_modules():
        full_name = f"{prefix}.{name}" if prefix else name
        if full_name:  # Skip root module
            h = module.register_forward_hook(make_hook(full_name))
            handles.append(h)

    return handles


def remove_logging_hooks(handles: List):
    """Remove logging hooks."""
    for h in handles:
        h.remove()
