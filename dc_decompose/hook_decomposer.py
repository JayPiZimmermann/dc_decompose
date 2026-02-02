"""
DC Decomposition using Pure Hooks (no module wrapping)

This approach hooks the original modules directly without creating wrapper classes.
The hooks compute and cache pos/neg activations and 4 sensitivities.

Key design:
- Parameters (shift_mode, beta, flags) are stored directly on the original modules
- Hooks read configuration from the modules they're attached to
- Forward: Input split according to shift_mode (default: CENTER)
- Backward: 4 local sensitivities (delta_pp, delta_np, delta_pn, delta_nn)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ShiftMode(Enum):
    """Mode for splitting input activations into pos/neg streams."""
    CENTER = "center"    # pos = ReLU(x), neg = ReLU(-x) - ensures non-negativity (DEFAULT)
    POSITIVE = "positive"  # pos = x, neg = 0 - all in positive stream
    NEGATIVE = "negative"  # pos = 0, neg = -x - all in negative stream
    BETA = "beta"        # pos = beta * x, neg = -(1-beta) * x - configurable split


class ReLUMode(Enum):
    MAX = "max"
    MIN = "min"
    HALF = "half"


@dataclass
class DCCache:
    """Cache for a single layer's DC decomposition data."""
    # Forward pass
    input_pos: Optional[Tensor] = None
    input_neg: Optional[Tensor] = None
    output_pos: Optional[Tensor] = None
    output_neg: Optional[Tensor] = None
    original_output: Optional[Tensor] = None

    # For ReLU: pre-activation value (z_before = input_pos - input_neg)
    z_before: Optional[Tensor] = None

    # For MaxPool: winner indices
    pool_indices: Optional[Tensor] = None

    # Backward pass (4 sensitivities)
    delta_pp: Optional[Tensor] = None
    delta_np: Optional[Tensor] = None
    delta_pn: Optional[Tensor] = None
    delta_nn: Optional[Tensor] = None

    def clear(self):
        for attr in ['input_pos', 'input_neg', 'output_pos', 'output_neg',
                     'original_output', 'z_before', 'pool_indices',
                     'delta_pp', 'delta_np', 'delta_pn', 'delta_nn']:
            setattr(self, attr, None)


# Attribute names stored on original modules
DC_SHIFT_MODE = '_dc_shift_mode'
DC_BETA = '_dc_beta'
DC_RELU_MODE = '_dc_relu_mode'
DC_ENABLED = '_dc_enabled'
DC_CACHE_ACTIVATIONS = '_dc_cache_activations'
DC_WEIGHT_POS = '_dc_weight_pos'
DC_WEIGHT_NEG = '_dc_weight_neg'
DC_BIAS_POS = '_dc_bias_pos'
DC_BIAS_NEG = '_dc_bias_neg'
DC_BN_SCALE_POS = '_dc_bn_scale_pos'
DC_BN_SCALE_NEG = '_dc_bn_scale_neg'
DC_SOFTMAX_DIM = '_dc_softmax_dim'
DC_LN_SCALE_POS = '_dc_ln_scale_pos'
DC_LN_SCALE_NEG = '_dc_ln_scale_neg'
DC_LN_NORMALIZED_SHAPE = '_dc_ln_normalized_shape'


class HookDecomposer:
    """
    DC Decomposition using pure forward/backward hooks on original modules.

    Parameters are stored directly on the original modules:
    - module._dc_shift_mode: ShiftMode, how to split input activations
    - module._dc_beta: float, input split parameter (for BETA mode)
    - module._dc_relu_mode: ReLUMode, how to decompose ReLU
    - module._dc_enabled: bool, whether DC is active for this layer
    - module._dc_cache_activations: bool, whether to cache activations

    Usage:
        model = YourModel()
        decomposer = HookDecomposer(model)  # Uses CENTER mode by default

        # Forward pass - hooks compute pos/neg automatically
        output = model(x)

        # Access decomposed activations
        for name, cache in decomposer.caches.items():
            print(f"{name}: pos={cache.output_pos.shape}")

        # Backward pass - compute 4 sensitivities
        decomposer.backward()
    """

    def __init__(
        self,
        model: nn.Module,
        shift_mode: ShiftMode = ShiftMode.CENTER,
        beta: float = 1.0,
        relu_mode: ReLUMode = ReLUMode.MAX,
        cache_activations: bool = True,
        target_layers: Optional[List[str]] = None,
    ):
        """
        Initialize HookDecomposer.

        Args:
            model: The PyTorch model to decompose
            shift_mode: How to split input activations into pos/neg streams:
                - CENTER (default): pos = ReLU(x), neg = ReLU(-x)
                  Ensures both streams are non-negative (true DC property)
                - POSITIVE: pos = x, neg = 0 (all in positive stream)
                - NEGATIVE: pos = 0, neg = -x (all in negative stream)
                - BETA: pos = beta * x, neg = -(1-beta) * x (configurable)
            beta: Split parameter for BETA mode (default 0.5)
            relu_mode: How to decompose ReLU activations
            cache_activations: Whether to cache activations (required for backward)
            target_layers: Optional list of layer names to decompose (None = all)
        """
        self.model = model
        self.shift_mode = shift_mode
        self.beta = beta
        self.relu_mode = relu_mode
        self.cache_activations = cache_activations
        self.target_layers = target_layers

        # Layer caches and ordering
        self.caches: Dict[str, DCCache] = {}
        self.layer_order: List[str] = []
        self.modules: Dict[str, nn.Module] = {}

        # Current decomposed state flowing through network
        self._current_pos: Optional[Tensor] = None
        self._current_neg: Optional[Tensor] = None
        self._initialized: bool = False

        # Hook handles
        self._forward_handles: List = []

        # Register hooks and store parameters on modules
        self._register_hooks()

    def _is_supported(self, module: nn.Module) -> bool:
        """Check if module type is supported for DC decomposition."""
        # Check for DC operation modules via their marker attributes
        dc_markers = [
            '_dc_is_matmul', '_dc_is_reshape', '_dc_is_permute', '_dc_is_transpose',
            '_dc_is_contiguous', '_dc_is_scalar_mul', '_dc_is_scalar_div', '_dc_is_add',
            '_dc_is_split', '_dc_is_chunk', '_dc_is_cat', '_dc_is_slice',
            '_dc_is_dropout', '_dc_is_identity', '_dc_is_embedding', '_dc_is_gather',
            '_dc_is_mean', '_dc_is_sum',
        ]
        for marker in dc_markers:
            if getattr(module, marker, False):
                return True

        return isinstance(module, (
            nn.Linear, nn.Conv2d, nn.ReLU, nn.Softmax,
            nn.BatchNorm2d, nn.LayerNorm, nn.MaxPool2d, nn.AvgPool2d,
            nn.Flatten, nn.AdaptiveAvgPool2d, nn.Dropout, nn.Identity,
        ))

    def _register_hooks(self):
        """Register forward hooks and store DC parameters on original modules."""
        for name, module in self.model.named_modules():
            if name == "":
                continue

            if self.target_layers is not None and name not in self.target_layers:
                continue

            if not self._is_supported(module):
                continue

            # Store DC parameters directly on the module
            self._setup_module(name, module)

            # Create cache for this layer
            self.caches[name] = DCCache()
            self.layer_order.append(name)
            self.modules[name] = module

            # Register forward hook
            handle = module.register_forward_hook(
                self._make_forward_hook(name)
            )
            self._forward_handles.append(handle)

    def _setup_module(self, name: str, module: nn.Module):
        """Store DC parameters and decomposed weights on the module."""
        # Store configuration parameters
        setattr(module, DC_SHIFT_MODE, self.shift_mode)
        setattr(module, DC_BETA, self.beta)
        setattr(module, DC_RELU_MODE, self.relu_mode)
        setattr(module, DC_ENABLED, True)
        setattr(module, DC_CACHE_ACTIVATIONS, self.cache_activations)

        # Pre-compute weight decomposition for layers with weights
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            with torch.no_grad():
                # W = W_pos - W_neg where W_pos = ReLU(W), W_neg = ReLU(-W)
                weight_pos = F.relu(module.weight.data.clone())
                weight_neg = F.relu(-module.weight.data.clone())
                setattr(module, DC_WEIGHT_POS, weight_pos)
                setattr(module, DC_WEIGHT_NEG, weight_neg)

                if module.bias is not None:
                    bias_pos = F.relu(module.bias.data.clone())
                    bias_neg = F.relu(-module.bias.data.clone())
                    setattr(module, DC_BIAS_POS, bias_pos)
                    setattr(module, DC_BIAS_NEG, bias_neg)

        elif isinstance(module, nn.BatchNorm2d):
            # Treat variance as constant
            # y = (x - mean) / sqrt(var + eps) * gamma + beta
            # Effective: y = x * scale + bias where scale = gamma / sqrt(var + eps)
            with torch.no_grad():
                std = torch.sqrt(module.running_var + module.eps)
                scale = module.weight / std
                bias = module.bias - module.running_mean * scale

                scale_pos = F.relu(scale.clone())
                scale_neg = F.relu(-scale.clone())
                bias_pos = F.relu(bias.clone())
                bias_neg = F.relu(-bias.clone())

                setattr(module, DC_BN_SCALE_POS, scale_pos)
                setattr(module, DC_BN_SCALE_NEG, scale_neg)
                setattr(module, DC_BIAS_POS, bias_pos)
                setattr(module, DC_BIAS_NEG, bias_neg)

        elif isinstance(module, nn.Softmax):
            # Store the softmax dimension
            setattr(module, DC_SOFTMAX_DIM, module.dim)

        elif isinstance(module, nn.LayerNorm):
            # Store normalized shape and decompose scale/bias
            # Scale (gamma) and bias (beta) are decomposed into pos/neg
            setattr(module, DC_LN_NORMALIZED_SHAPE, module.normalized_shape)
            with torch.no_grad():
                if module.weight is not None:
                    scale_pos = F.relu(module.weight.data.clone())
                    scale_neg = F.relu(-module.weight.data.clone())
                    setattr(module, DC_LN_SCALE_POS, scale_pos)
                    setattr(module, DC_LN_SCALE_NEG, scale_neg)
                if module.bias is not None:
                    bias_pos = F.relu(module.bias.data.clone())
                    bias_neg = F.relu(-module.bias.data.clone())
                    setattr(module, DC_BIAS_POS, bias_pos)
                    setattr(module, DC_BIAS_NEG, bias_neg)

    def _make_forward_hook(self, name: str):
        """Create forward hook that reads config from the module itself."""

        def hook(module: nn.Module, inputs: Tuple[Tensor, ...], output: Tensor):
            # Check if DC is enabled for this module
            if not getattr(module, DC_ENABLED, True):
                return

            cache = self.caches[name]
            should_cache = getattr(module, DC_CACHE_ACTIVATIONS, True)

            # Get input pos/neg from previous layer or initialize
            if not self._initialized:
                # First layer: apply shift_mode-based initialization
                x = inputs[0]
                shift_mode = getattr(module, DC_SHIFT_MODE, ShiftMode.CENTER)
                beta = getattr(module, DC_BETA, 0.5)

                if shift_mode == ShiftMode.CENTER:
                    # pos = ReLU(x), neg = ReLU(-x) - ensures non-negativity
                    input_pos = F.relu(x)
                    input_neg = F.relu(-x)
                elif shift_mode == ShiftMode.POSITIVE:
                    # pos = x, neg = 0 - all in positive stream
                    input_pos = x
                    input_neg = torch.zeros_like(x)
                elif shift_mode == ShiftMode.NEGATIVE:
                    # pos = 0, neg = -x - all in negative stream
                    input_pos = torch.zeros_like(x)
                    input_neg = -x
                elif shift_mode == ShiftMode.BETA:
                    # pos = beta * x, neg = -(1-beta) * x - configurable split
                    input_pos = beta * x
                    input_neg = -(1 - beta) * x
                else:
                    # Default to CENTER
                    input_pos = F.relu(x)
                    input_neg = F.relu(-x)

                self._initialized = True
            else:
                input_pos = self._current_pos
                input_neg = self._current_neg

            # Cache inputs if requested
            if should_cache:
                cache.input_pos = input_pos.detach()
                cache.input_neg = input_neg.detach()
                cache.original_output = output.detach()

            # Compute DC decomposition based on layer type
            if isinstance(module, nn.Linear):
                output_pos, output_neg = self._forward_linear(module, input_pos, input_neg)

            elif isinstance(module, nn.Conv2d):
                output_pos, output_neg = self._forward_conv2d(module, input_pos, input_neg)

            elif isinstance(module, nn.ReLU):
                relu_mode = getattr(module, DC_RELU_MODE, ReLUMode.MAX)
                output_pos, output_neg = self._forward_relu(
                    cache, input_pos, input_neg, relu_mode, should_cache
                )

            elif isinstance(module, nn.BatchNorm2d):
                output_pos, output_neg = self._forward_batchnorm(module, input_pos, input_neg)

            elif isinstance(module, nn.MaxPool2d):
                # Winner-takes-all only
                output_pos, output_neg = self._forward_maxpool_wta(
                    module, cache, input_pos, input_neg, should_cache
                )

            elif isinstance(module, nn.AvgPool2d):
                output_pos, output_neg = self._forward_avgpool(module, input_pos, input_neg)

            elif isinstance(module, nn.AdaptiveAvgPool2d):
                output_pos, output_neg = self._forward_adaptive_avgpool(module, input_pos, input_neg)

            elif isinstance(module, nn.Flatten):
                output_pos = input_pos.flatten(module.start_dim, module.end_dim)
                output_neg = input_neg.flatten(module.start_dim, module.end_dim)

            elif isinstance(module, nn.Softmax):
                output_pos, output_neg = self._forward_softmax(
                    module, cache, input_pos, input_neg, should_cache
                )

            elif isinstance(module, nn.LayerNorm):
                output_pos, output_neg = self._forward_layernorm(
                    module, cache, input_pos, input_neg, should_cache
                )

            elif hasattr(module, '_dc_is_matmul') and module._dc_is_matmul:
                # DCMatMul module - special handling for two-input matmul
                output_pos, output_neg = self._forward_dc_matmul(
                    module, cache, input_pos, input_neg, should_cache
                )

            # =========================================================
            # DC Operation Modules (linear operations)
            # For linear operations: apply same operation to both streams
            # =========================================================

            elif getattr(module, '_dc_is_reshape', False):
                # Reshape: apply same operation to both streams
                output_pos = input_pos.view(output.shape)
                output_neg = input_neg.view(output.shape)

            elif getattr(module, '_dc_is_permute', False):
                # Permute: apply same operation to both streams
                output_pos = input_pos.permute(*module.dims)
                output_neg = input_neg.permute(*module.dims)

            elif getattr(module, '_dc_is_transpose', False):
                # Transpose: apply same operation to both streams
                output_pos = input_pos.transpose(module.dim0, module.dim1)
                output_neg = input_neg.transpose(module.dim0, module.dim1)

            elif getattr(module, '_dc_is_contiguous', False):
                # Contiguous: apply same operation to both streams
                output_pos = input_pos.contiguous()
                output_neg = input_neg.contiguous()

            elif getattr(module, '_dc_is_scalar_mul', False):
                # Scalar multiplication
                if module.is_negative:
                    # Negative scalar swaps pos and neg
                    output_pos = module.abs_scalar * input_neg
                    output_neg = module.abs_scalar * input_pos
                else:
                    output_pos = module.scalar * input_pos
                    output_neg = module.scalar * input_neg

            elif getattr(module, '_dc_is_scalar_div', False):
                # Scalar division
                if module.is_negative:
                    # Negative scalar swaps pos and neg
                    output_pos = input_neg / module.abs_scalar
                    output_neg = input_pos / module.abs_scalar
                else:
                    output_pos = input_pos / module.scalar
                    output_neg = input_neg / module.scalar

            elif getattr(module, '_dc_is_add', False):
                # Addition: (a_pos - a_neg) + (b_pos - b_neg) = (a_pos + b_pos) - (a_neg + b_neg)
                if module._dc_operand_pos is not None:
                    output_pos = input_pos + module._dc_operand_pos
                    output_neg = input_neg + module._dc_operand_neg
                else:
                    # Fallback: use original output
                    output_pos = output
                    output_neg = torch.zeros_like(output)

            elif getattr(module, '_dc_is_slice', False):
                # Slice: apply same slice to both streams
                slices = [slice(None)] * input_pos.dim()
                slices[module.dim] = slice(module.start, module.end)
                output_pos = input_pos[tuple(slices)]
                output_neg = input_neg[tuple(slices)]

            elif getattr(module, '_dc_is_dropout', False) or isinstance(module, nn.Dropout):
                # Dropout: identity in eval, same mask in train
                if module.training:
                    # Generate mask from output
                    mask = (output != 0).float() if hasattr(module, 'p') and module.p > 0 else None
                    if mask is not None:
                        scale = 1.0 / (1.0 - module.p) if hasattr(module, 'p') else 1.0
                        output_pos = input_pos * mask * scale
                        output_neg = input_neg * mask * scale
                    else:
                        output_pos = input_pos
                        output_neg = input_neg
                else:
                    output_pos = input_pos
                    output_neg = input_neg

            elif getattr(module, '_dc_is_identity', False) or isinstance(module, nn.Identity):
                # Identity: pass through
                output_pos = input_pos
                output_neg = input_neg

            elif getattr(module, '_dc_is_mean', False):
                # Mean: linear operation
                if module.dim is None:
                    output_pos = input_pos.mean()
                    output_neg = input_neg.mean()
                else:
                    output_pos = input_pos.mean(dim=module.dim, keepdim=module.keepdim)
                    output_neg = input_neg.mean(dim=module.dim, keepdim=module.keepdim)

            elif getattr(module, '_dc_is_sum', False):
                # Sum: linear operation
                if module.dim is None:
                    output_pos = input_pos.sum()
                    output_neg = input_neg.sum()
                else:
                    output_pos = input_pos.sum(dim=module.dim, keepdim=module.keepdim)
                    output_neg = input_neg.sum(dim=module.dim, keepdim=module.keepdim)

            elif getattr(module, '_dc_is_gather', False):
                # Gather: linear operation (need index from forward pass)
                # Use output shape to infer the gather was done
                output_pos = output
                output_neg = torch.zeros_like(output)  # Fallback

            else:
                # Fallback: shouldn't reach here for supported modules
                output_pos = output
                output_neg = torch.zeros_like(output)

            # Cache outputs if requested
            if should_cache:
                cache.output_pos = output_pos.detach()
                cache.output_neg = output_neg.detach()

            # Update current state for next layer
            self._current_pos = output_pos
            self._current_neg = output_neg

        return hook

    def _forward_linear(self, module: nn.Linear, input_pos: Tensor, input_neg: Tensor) -> Tuple[Tensor, Tensor]:
        """DC forward for Linear layer using weights stored on module."""
        W_pos = getattr(module, DC_WEIGHT_POS)
        W_neg = getattr(module, DC_WEIGHT_NEG)

        # pos_out = W_pos @ pos_in + W_neg @ neg_in + bias_pos
        output_pos = F.linear(input_pos, W_pos) + F.linear(input_neg, W_neg)
        if hasattr(module, DC_BIAS_POS):
            output_pos = output_pos + getattr(module, DC_BIAS_POS)

        # neg_out = W_neg @ pos_in + W_pos @ neg_in + bias_neg
        output_neg = F.linear(input_pos, W_neg) + F.linear(input_neg, W_pos)
        if hasattr(module, DC_BIAS_NEG):
            output_neg = output_neg + getattr(module, DC_BIAS_NEG)

        return output_pos, output_neg

    def _forward_conv2d(self, module: nn.Conv2d, input_pos: Tensor, input_neg: Tensor) -> Tuple[Tensor, Tensor]:
        """DC forward for Conv2d layer using weights stored on module."""
        W_pos = getattr(module, DC_WEIGHT_POS)
        W_neg = getattr(module, DC_WEIGHT_NEG)

        # pos_out = conv(pos_in, W_pos) + conv(neg_in, W_neg) + bias_pos
        output_pos = F.conv2d(input_pos, W_pos, None, module.stride, module.padding, module.dilation, module.groups)
        output_pos = output_pos + F.conv2d(input_neg, W_neg, None, module.stride, module.padding, module.dilation, module.groups)
        if hasattr(module, DC_BIAS_POS):
            output_pos = output_pos + getattr(module, DC_BIAS_POS).view(1, -1, 1, 1)

        # neg_out = conv(pos_in, W_neg) + conv(neg_in, W_pos) + bias_neg
        output_neg = F.conv2d(input_pos, W_neg, None, module.stride, module.padding, module.dilation, module.groups)
        output_neg = output_neg + F.conv2d(input_neg, W_pos, None, module.stride, module.padding, module.dilation, module.groups)
        if hasattr(module, DC_BIAS_NEG):
            output_neg = output_neg + getattr(module, DC_BIAS_NEG).view(1, -1, 1, 1)

        return output_pos, output_neg

    def _forward_relu(
        self, cache: DCCache, input_pos: Tensor, input_neg: Tensor,
        relu_mode: ReLUMode, should_cache: bool
    ) -> Tuple[Tensor, Tensor]:
        """DC forward for ReLU layer."""
        # Pre-activation value
        z_before = input_pos - input_neg
        if should_cache:
            cache.z_before = z_before.detach()

        if relu_mode == ReLUMode.MAX:
            # pos_out = ReLU(z_before), neg_out = 0
            output_pos = F.relu(z_before)
            output_neg = torch.zeros_like(output_pos)

        elif relu_mode == ReLUMode.MIN:
            # Minimize pos stream magnitude
            min_val = torch.minimum(input_pos, input_neg)
            output_pos = input_pos - min_val + F.relu(-z_before)
            output_neg = input_neg - min_val + F.relu(-z_before)

        elif relu_mode == ReLUMode.HALF:
            # Average of MAX and MIN
            max_val = torch.maximum(input_pos, input_neg)
            min_val = torch.minimum(input_pos, input_neg)
            output_pos = F.relu((max_val + input_pos - min_val) / 2)
            output_neg = F.relu((input_neg + min_val - max_val) / 2)

        else:
            raise ValueError(f"Unknown ReLU mode: {relu_mode}")

        return output_pos, output_neg

    def _forward_batchnorm(self, module: nn.BatchNorm2d, input_pos: Tensor, input_neg: Tensor) -> Tuple[Tensor, Tensor]:
        """DC forward for BatchNorm2d layer (variance treated as constant)."""
        scale_pos = getattr(module, DC_BN_SCALE_POS).view(1, -1, 1, 1)
        scale_neg = getattr(module, DC_BN_SCALE_NEG).view(1, -1, 1, 1)
        bias_pos = getattr(module, DC_BIAS_POS).view(1, -1, 1, 1)
        bias_neg = getattr(module, DC_BIAS_NEG).view(1, -1, 1, 1)

        # pos_out = scale_pos * pos_in + scale_neg * neg_in + bias_pos
        output_pos = scale_pos * input_pos + scale_neg * input_neg + bias_pos

        # neg_out = scale_neg * pos_in + scale_pos * neg_in + bias_neg
        output_neg = scale_neg * input_pos + scale_pos * input_neg + bias_neg

        return output_pos, output_neg

    def _forward_maxpool_wta(
        self, module: nn.MaxPool2d, cache: DCCache,
        input_pos: Tensor, input_neg: Tensor, should_cache: bool
    ) -> Tuple[Tensor, Tensor]:
        """
        DC forward for MaxPool2d layer using winner-takes-all.

        The argmax is determined from the original activation (pos - neg),
        then the same positions are selected from both pos and neg streams.
        """
        z_before = input_pos - input_neg

        # Get argmax indices from original activation
        _, indices = F.max_pool2d(
            z_before, module.kernel_size, module.stride, module.padding,
            return_indices=True
        )

        if should_cache:
            cache.pool_indices = indices.detach()

        # Gather from pos and neg streams using winner indices
        batch, channels, h_out, w_out = indices.shape

        pos_flat = input_pos.view(batch, channels, -1)
        neg_flat = input_neg.view(batch, channels, -1)
        indices_flat = indices.view(batch, channels, -1)

        output_pos = torch.gather(pos_flat, 2, indices_flat).view(batch, channels, h_out, w_out)
        output_neg = torch.gather(neg_flat, 2, indices_flat).view(batch, channels, h_out, w_out)

        return output_pos, output_neg

    def _forward_avgpool(self, module: nn.AvgPool2d, input_pos: Tensor, input_neg: Tensor) -> Tuple[Tensor, Tensor]:
        """DC forward for AvgPool2d layer (linear operation)."""
        output_pos = F.avg_pool2d(input_pos, module.kernel_size, module.stride, module.padding)
        output_neg = F.avg_pool2d(input_neg, module.kernel_size, module.stride, module.padding)
        return output_pos, output_neg

    def _forward_adaptive_avgpool(self, module: nn.AdaptiveAvgPool2d, input_pos: Tensor, input_neg: Tensor) -> Tuple[Tensor, Tensor]:
        """DC forward for AdaptiveAvgPool2d layer (linear operation)."""
        output_pos = F.adaptive_avg_pool2d(input_pos, module.output_size)
        output_neg = F.adaptive_avg_pool2d(input_neg, module.output_size)
        return output_pos, output_neg

    def _forward_softmax(
        self, module: nn.Softmax, cache: DCCache,
        input_pos: Tensor, input_neg: Tensor, should_cache: bool
    ) -> Tuple[Tensor, Tensor]:
        """
        DC forward for Softmax layer.

        Forward: output_pos = softmax(input_pos - input_neg), output_neg = 0
        The original input (pos - neg) is cached for Jacobian computation in backward.
        """
        dim = getattr(module, DC_SOFTMAX_DIM, -1)

        # Compute original input
        z = input_pos - input_neg

        # Cache the original input for backward Jacobian computation
        if should_cache:
            cache.z_before = z.detach()

        # Forward: softmax applied to original input, output in pos stream only
        output_pos = F.softmax(z, dim=dim)
        output_neg = torch.zeros_like(output_pos)

        return output_pos, output_neg

    def _forward_layernorm(
        self, module: nn.LayerNorm, cache: DCCache,
        input_pos: Tensor, input_neg: Tensor, should_cache: bool
    ) -> Tuple[Tensor, Tensor]:
        """
        DC forward for LayerNorm layer.

        Variance is computed on the original input (pos - neg) and treated as constant.
        y = (x - mean) / sqrt(var + eps) * gamma + beta

        For DC decomposition:
        - Compute normalized z from original input
        - Apply decomposed scale and bias
        """
        # Original input
        z = input_pos - input_neg

        # Cache original input for backward
        if should_cache:
            cache.z_before = z.detach()

        # Compute mean and variance along normalized dimensions
        normalized_shape = getattr(module, DC_LN_NORMALIZED_SHAPE, module.normalized_shape)
        dims = tuple(range(-len(normalized_shape), 0))

        mean = z.mean(dim=dims, keepdim=True)
        var = z.var(dim=dims, unbiased=False, keepdim=True)
        std = torch.sqrt(var + module.eps)

        # Normalize the original input
        z_norm = (z - mean) / std

        # Cache std for backward pass (variance treated as constant)
        if should_cache:
            cache.pool_indices = std.detach()  # Reuse pool_indices to store std

        # Apply scale and bias decomposition
        # y = gamma * z_norm + beta
        # gamma = gamma_pos - gamma_neg, beta = beta_pos - beta_neg
        # y_pos = gamma_pos * ReLU(z_norm) + gamma_neg * ReLU(-z_norm) + beta_pos
        # y_neg = gamma_neg * ReLU(z_norm) + gamma_pos * ReLU(-z_norm) + beta_neg

        if hasattr(module, DC_LN_SCALE_POS):
            scale_pos = getattr(module, DC_LN_SCALE_POS)
            scale_neg = getattr(module, DC_LN_SCALE_NEG)

            # Decompose z_norm into pos/neg
            z_norm_pos = F.relu(z_norm)
            z_norm_neg = F.relu(-z_norm)

            output_pos = scale_pos * z_norm_pos + scale_neg * z_norm_neg
            output_neg = scale_neg * z_norm_pos + scale_pos * z_norm_neg
        else:
            # No learnable scale
            output_pos = F.relu(z_norm)
            output_neg = F.relu(-z_norm)

        # Add bias
        if hasattr(module, DC_BIAS_POS):
            bias_pos = getattr(module, DC_BIAS_POS)
            bias_neg = getattr(module, DC_BIAS_NEG)
            output_pos = output_pos + bias_pos
            output_neg = output_neg + bias_neg

        return output_pos, output_neg

    def _forward_dc_matmul(
        self, module: nn.Module, cache: DCCache,
        input_pos: Tensor, input_neg: Tensor, should_cache: bool
    ) -> Tuple[Tensor, Tensor]:
        """
        DC forward for DCMatMul module.

        (A+ - A-)(B+ - B-) = (A+B+ + A-B-) - (A+B- + A-B+)

        The second operand (B) is stored on the module with its pos/neg decomposition.
        """
        # Get B's pos/neg from the module (set before forward pass)
        B_pos = module._dc_operand_pos
        B_neg = module._dc_operand_neg

        # Cache inputs
        if should_cache:
            cache.z_before = (B_pos.detach(), B_neg.detach())  # Store B for backward

        # Forward: (A+ - A-)(B+ - B-) = (A+B+ + A-B-) - (A+B- + A-B+)
        output_pos = torch.matmul(input_pos, B_pos) + torch.matmul(input_neg, B_neg)
        output_neg = torch.matmul(input_pos, B_neg) + torch.matmul(input_neg, B_pos)

        return output_pos, output_neg

    # =========================================================================
    # Initialization and State Management
    # =========================================================================

    def initialize(self, x: Optional[Tensor] = None):
        """
        Initialize for a new forward pass.

        Call this before model(x) to reset the DC state.
        """
        self._current_pos = None
        self._current_neg = None
        self._initialized = False

        for cache in self.caches.values():
            cache.clear()

    def set_shift_mode(self, mode: ShiftMode):
        """
        Update the shift mode on all modules.

        Args:
            mode: New shift mode for input splitting
        """
        self.shift_mode = mode
        for module in self.modules.values():
            setattr(module, DC_SHIFT_MODE, mode)

    def set_beta(self, beta: float):
        """
        Update the beta parameter on all modules (for BETA shift mode).

        Args:
            beta: New input split parameter
        """
        self.beta = beta
        for module in self.modules.values():
            setattr(module, DC_BETA, beta)

    def set_relu_mode(self, mode: ReLUMode):
        """
        Update the ReLU mode on all ReLU modules.

        Args:
            mode: New ReLU decomposition mode
        """
        self.relu_mode = mode
        for module in self.modules.values():
            if isinstance(module, nn.ReLU):
                setattr(module, DC_RELU_MODE, mode)

    def enable_layer(self, name: str, enabled: bool = True):
        """Enable or disable DC decomposition for a specific layer."""
        if name in self.modules:
            setattr(self.modules[name], DC_ENABLED, enabled)

    def enable_caching(self, name: str, enabled: bool = True):
        """Enable or disable activation caching for a specific layer."""
        if name in self.modules:
            setattr(self.modules[name], DC_CACHE_ACTIVATIONS, enabled)

    # =========================================================================
    # Backward Pass: Compute 4 Local Sensitivities
    # =========================================================================

    def backward(
        self,
        grad_output_pos: Optional[Tensor] = None,
        grad_output_neg: Optional[Tensor] = None,
    ):
        """
        Compute 4 sensitivities via backward pass through cached activations.

        The backward pass propagates 4 sensitivity tensors:
        - delta_pp: sensitivity of output_pos w.r.t. input_pos
        - delta_np: sensitivity of output_pos w.r.t. input_neg
        - delta_pn: sensitivity of output_neg w.r.t. input_pos
        - delta_nn: sensitivity of output_neg w.r.t. input_neg

        Args:
            grad_output_pos: Gradient w.r.t. final output_pos (default: ones)
            grad_output_neg: Gradient w.r.t. final output_neg (default: zeros)
        """
        if len(self.layer_order) == 0:
            return

        # Get final layer's output shape for initialization
        final_name = self.layer_order[-1]
        final_cache = self.caches[final_name]

        if final_cache.output_pos is None:
            raise RuntimeError("No cached activations. Run forward pass with caching enabled first.")

        # Initialize output sensitivities
        if grad_output_pos is None:
            grad_output_pos = torch.ones_like(final_cache.output_pos)
        if grad_output_neg is None:
            grad_output_neg = torch.zeros_like(final_cache.output_neg)

        # For output = pos - neg, gradient flows as:
        # delta_pp: grad from pos_out
        # delta_pn: grad from neg_out (negative contribution to loss)
        delta_pp = grad_output_pos.clone()
        delta_np = torch.zeros_like(grad_output_pos)
        delta_pn = grad_output_neg.clone()
        delta_nn = torch.zeros_like(grad_output_neg)

        # Backpropagate through layers in reverse order
        for name in reversed(self.layer_order):
            cache = self.caches[name]
            module = self.modules[name]

            # Check if this layer was enabled
            if not getattr(module, DC_ENABLED, True):
                continue

            if isinstance(module, nn.Linear):
                delta_pp, delta_np, delta_pn, delta_nn = self._backward_linear(
                    module, delta_pp, delta_np, delta_pn, delta_nn
                )

            elif isinstance(module, nn.Conv2d):
                delta_pp, delta_np, delta_pn, delta_nn = self._backward_conv2d(
                    module, delta_pp, delta_np, delta_pn, delta_nn
                )

            elif isinstance(module, nn.ReLU):
                relu_mode = getattr(module, DC_RELU_MODE, ReLUMode.MAX)
                delta_pp, delta_np, delta_pn, delta_nn = self._backward_relu(
                    cache, delta_pp, delta_np, delta_pn, delta_nn, relu_mode
                )

            elif isinstance(module, nn.BatchNorm2d):
                delta_pp, delta_np, delta_pn, delta_nn = self._backward_batchnorm(
                    module, delta_pp, delta_np, delta_pn, delta_nn
                )

            elif isinstance(module, nn.MaxPool2d):
                delta_pp, delta_np, delta_pn, delta_nn = self._backward_maxpool_wta(
                    cache, delta_pp, delta_np, delta_pn, delta_nn
                )

            elif isinstance(module, (nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
                delta_pp, delta_np, delta_pn, delta_nn = self._backward_avgpool(
                    cache, module, delta_pp, delta_np, delta_pn, delta_nn
                )

            elif isinstance(module, nn.Softmax):
                delta_pp, delta_np, delta_pn, delta_nn = self._backward_softmax(
                    module, cache, delta_pp, delta_np, delta_pn, delta_nn
                )

            elif isinstance(module, nn.LayerNorm):
                delta_pp, delta_np, delta_pn, delta_nn = self._backward_layernorm(
                    module, cache, delta_pp, delta_np, delta_pn, delta_nn
                )

            elif hasattr(module, '_dc_is_matmul') and module._dc_is_matmul:
                delta_pp, delta_np, delta_pn, delta_nn = self._backward_dc_matmul(
                    module, cache, delta_pp, delta_np, delta_pn, delta_nn
                )

            elif isinstance(module, nn.Flatten):
                input_shape = cache.input_pos.shape
                delta_pp = delta_pp.view(input_shape)
                delta_np = delta_np.view(input_shape)
                delta_pn = delta_pn.view(input_shape)
                delta_nn = delta_nn.view(input_shape)

            # =========================================================
            # DC Operation Modules Backward (linear operations)
            # =========================================================

            elif getattr(module, '_dc_is_reshape', False):
                # Reshape backward: reshape to input shape
                input_shape = cache.input_pos.shape
                delta_pp = delta_pp.view(input_shape)
                delta_np = delta_np.view(input_shape)
                delta_pn = delta_pn.view(input_shape)
                delta_nn = delta_nn.view(input_shape)

            elif getattr(module, '_dc_is_permute', False):
                # Permute backward: inverse permutation
                inv_dims = [0] * len(module.dims)
                for i, d in enumerate(module.dims):
                    inv_dims[d] = i
                delta_pp = delta_pp.permute(*inv_dims)
                delta_np = delta_np.permute(*inv_dims)
                delta_pn = delta_pn.permute(*inv_dims)
                delta_nn = delta_nn.permute(*inv_dims)

            elif getattr(module, '_dc_is_transpose', False):
                # Transpose backward: same transpose (self-inverse)
                delta_pp = delta_pp.transpose(module.dim0, module.dim1)
                delta_np = delta_np.transpose(module.dim0, module.dim1)
                delta_pn = delta_pn.transpose(module.dim0, module.dim1)
                delta_nn = delta_nn.transpose(module.dim0, module.dim1)

            elif getattr(module, '_dc_is_contiguous', False):
                # Contiguous backward: identity
                pass

            elif getattr(module, '_dc_is_scalar_mul', False):
                # Scalar multiplication backward
                if module.is_negative:
                    # Negative scalar swapped streams in forward
                    delta_pp, delta_np = delta_np * module.abs_scalar, delta_pp * module.abs_scalar
                    delta_pn, delta_nn = delta_nn * module.abs_scalar, delta_pn * module.abs_scalar
                else:
                    delta_pp = delta_pp * module.scalar
                    delta_np = delta_np * module.scalar
                    delta_pn = delta_pn * module.scalar
                    delta_nn = delta_nn * module.scalar

            elif getattr(module, '_dc_is_scalar_div', False):
                # Scalar division backward
                if module.is_negative:
                    # Negative scalar swapped streams in forward
                    delta_pp, delta_np = delta_np / module.abs_scalar, delta_pp / module.abs_scalar
                    delta_pn, delta_nn = delta_nn / module.abs_scalar, delta_pn / module.abs_scalar
                else:
                    delta_pp = delta_pp / module.scalar
                    delta_np = delta_np / module.scalar
                    delta_pn = delta_pn / module.scalar
                    delta_nn = delta_nn / module.scalar

            elif getattr(module, '_dc_is_add', False):
                # Addition backward: identity (gradient flows through both operands)
                # Only handle the first operand here; second operand gradient not tracked
                pass

            elif getattr(module, '_dc_is_slice', False):
                # Slice backward: scatter gradients back
                input_shape = cache.input_pos.shape
                new_delta_pp = torch.zeros(input_shape, device=delta_pp.device, dtype=delta_pp.dtype)
                new_delta_np = torch.zeros(input_shape, device=delta_np.device, dtype=delta_np.dtype)
                new_delta_pn = torch.zeros(input_shape, device=delta_pn.device, dtype=delta_pn.dtype)
                new_delta_nn = torch.zeros(input_shape, device=delta_nn.device, dtype=delta_nn.dtype)
                slices = [slice(None)] * len(input_shape)
                slices[module.dim] = slice(module.start, module.end)
                new_delta_pp[tuple(slices)] = delta_pp
                new_delta_np[tuple(slices)] = delta_np
                new_delta_pn[tuple(slices)] = delta_pn
                new_delta_nn[tuple(slices)] = delta_nn
                delta_pp, delta_np, delta_pn, delta_nn = new_delta_pp, new_delta_np, new_delta_pn, new_delta_nn

            elif getattr(module, '_dc_is_dropout', False) or isinstance(module, nn.Dropout):
                # Dropout backward: identity in eval, masked in train
                pass

            elif getattr(module, '_dc_is_identity', False) or isinstance(module, nn.Identity):
                # Identity backward: identity
                pass

            elif getattr(module, '_dc_is_mean', False):
                # Mean backward: expand and divide
                input_shape = cache.input_pos.shape
                if module.dim is None:
                    n = cache.input_pos.numel()
                    delta_pp = delta_pp.expand(input_shape) / n
                    delta_np = delta_np.expand(input_shape) / n
                    delta_pn = delta_pn.expand(input_shape) / n
                    delta_nn = delta_nn.expand(input_shape) / n
                else:
                    dims = (module.dim,) if isinstance(module.dim, int) else module.dim
                    n = 1
                    for d in dims:
                        n *= input_shape[d]
                    if not module.keepdim:
                        for d in sorted(dims):
                            delta_pp = delta_pp.unsqueeze(d)
                            delta_np = delta_np.unsqueeze(d)
                            delta_pn = delta_pn.unsqueeze(d)
                            delta_nn = delta_nn.unsqueeze(d)
                    delta_pp = delta_pp.expand(input_shape) / n
                    delta_np = delta_np.expand(input_shape) / n
                    delta_pn = delta_pn.expand(input_shape) / n
                    delta_nn = delta_nn.expand(input_shape) / n

            elif getattr(module, '_dc_is_sum', False):
                # Sum backward: expand
                input_shape = cache.input_pos.shape
                if module.dim is None:
                    delta_pp = delta_pp.expand(input_shape)
                    delta_np = delta_np.expand(input_shape)
                    delta_pn = delta_pn.expand(input_shape)
                    delta_nn = delta_nn.expand(input_shape)
                else:
                    dims = (module.dim,) if isinstance(module.dim, int) else module.dim
                    if not module.keepdim:
                        for d in sorted(dims):
                            delta_pp = delta_pp.unsqueeze(d)
                            delta_np = delta_np.unsqueeze(d)
                            delta_pn = delta_pn.unsqueeze(d)
                            delta_nn = delta_nn.unsqueeze(d)
                    delta_pp = delta_pp.expand(input_shape)
                    delta_np = delta_np.expand(input_shape)
                    delta_pn = delta_pn.expand(input_shape)
                    delta_nn = delta_nn.expand(input_shape)

            # Cache sensitivities for this layer
            cache.delta_pp = delta_pp.detach().clone()
            cache.delta_np = delta_np.detach().clone()
            cache.delta_pn = delta_pn.detach().clone()
            cache.delta_nn = delta_nn.detach().clone()

    def _backward_linear(
        self, module: nn.Linear,
        delta_pp: Tensor, delta_np: Tensor, delta_pn: Tensor, delta_nn: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Backward through Linear layer."""
        W_pos = getattr(module, DC_WEIGHT_POS)
        W_neg = getattr(module, DC_WEIGHT_NEG)

        # Gradient w.r.t. input_pos comes from both output_pos and output_neg
        # d(out_pos)/d(in_pos) = W_pos, d(out_neg)/d(in_pos) = W_neg
        new_delta_pp = F.linear(delta_pp, W_pos.t()) + F.linear(delta_pn, W_neg.t())
        new_delta_np = F.linear(delta_pp, W_neg.t()) + F.linear(delta_pn, W_pos.t())
        new_delta_pn = F.linear(delta_np, W_pos.t()) + F.linear(delta_nn, W_neg.t())
        new_delta_nn = F.linear(delta_np, W_neg.t()) + F.linear(delta_nn, W_pos.t())

        return new_delta_pp, new_delta_np, new_delta_pn, new_delta_nn

    def _backward_conv2d(
        self, module: nn.Conv2d,
        delta_pp: Tensor, delta_np: Tensor, delta_pn: Tensor, delta_nn: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Backward through Conv2d layer."""
        W_pos = getattr(module, DC_WEIGHT_POS)
        W_neg = getattr(module, DC_WEIGHT_NEG)

        kwargs = dict(stride=module.stride, padding=module.padding,
                      dilation=module.dilation, groups=module.groups)

        new_delta_pp = F.conv_transpose2d(delta_pp, W_pos, **kwargs) + F.conv_transpose2d(delta_pn, W_neg, **kwargs)
        new_delta_np = F.conv_transpose2d(delta_pp, W_neg, **kwargs) + F.conv_transpose2d(delta_pn, W_pos, **kwargs)
        new_delta_pn = F.conv_transpose2d(delta_np, W_pos, **kwargs) + F.conv_transpose2d(delta_nn, W_neg, **kwargs)
        new_delta_nn = F.conv_transpose2d(delta_np, W_neg, **kwargs) + F.conv_transpose2d(delta_nn, W_pos, **kwargs)

        return new_delta_pp, new_delta_np, new_delta_pn, new_delta_nn

    def _backward_relu(
        self, cache: DCCache,
        delta_pp: Tensor, delta_np: Tensor, delta_pn: Tensor, delta_nn: Tensor,
        relu_mode: ReLUMode
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Backward through ReLU layer."""
        z_before = cache.z_before

        if relu_mode == ReLUMode.MAX:
            # For MAX mode:
            # output_pos = ReLU(z_before) = ReLU(in_pos - in_neg)
            # output_neg = 0
            #
            # d(out_pos)/d(in_pos) = 1 if z_before >= 0, else 0
            # d(out_pos)/d(in_neg) = -1 if z_before >= 0, else 0
            # But since out_neg = 0, gradient flows only through out_pos
            mask_pos = (z_before >= 0).float()
            mask_neg = (z_before < 0).float()

            # delta_pp propagates where z_before >= 0
            # Where z_before < 0, gradient goes to neg input (delta_np)
            new_delta_pp = delta_pp * mask_pos
            new_delta_np = delta_np + delta_pp * mask_neg
            new_delta_pn = delta_pn * mask_pos
            new_delta_nn = delta_nn + delta_pn * mask_neg

        elif relu_mode == ReLUMode.MIN:
            # Simplified backward for MIN mode
            new_delta_pp = delta_pp
            new_delta_np = delta_np
            new_delta_pn = delta_pn
            new_delta_nn = delta_nn

        elif relu_mode == ReLUMode.HALF:
            # Simplified backward for HALF mode
            new_delta_pp = delta_pp * 0.5
            new_delta_np = delta_np * 0.5
            new_delta_pn = delta_pn * 0.5
            new_delta_nn = delta_nn * 0.5

        else:
            new_delta_pp, new_delta_np = delta_pp, delta_np
            new_delta_pn, new_delta_nn = delta_pn, delta_nn

        return new_delta_pp, new_delta_np, new_delta_pn, new_delta_nn

    def _backward_batchnorm(
        self, module: nn.BatchNorm2d,
        delta_pp: Tensor, delta_np: Tensor, delta_pn: Tensor, delta_nn: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Backward through BatchNorm2d layer (variance as constant)."""
        scale_pos = getattr(module, DC_BN_SCALE_POS).view(1, -1, 1, 1)
        scale_neg = getattr(module, DC_BN_SCALE_NEG).view(1, -1, 1, 1)

        new_delta_pp = delta_pp * scale_pos + delta_pn * scale_neg
        new_delta_np = delta_pp * scale_neg + delta_pn * scale_pos
        new_delta_pn = delta_np * scale_pos + delta_nn * scale_neg
        new_delta_nn = delta_np * scale_neg + delta_nn * scale_pos

        return new_delta_pp, new_delta_np, new_delta_pn, new_delta_nn

    def _backward_maxpool_wta(
        self, cache: DCCache,
        delta_pp: Tensor, delta_np: Tensor, delta_pn: Tensor, delta_nn: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Backward through MaxPool2d layer (winner-takes-all)."""
        indices = cache.pool_indices
        input_shape = cache.input_pos.shape

        def unpool(grad: Tensor) -> Tensor:
            batch, channels, h_out, w_out = grad.shape
            h_in, w_in = input_shape[2], input_shape[3]

            out = torch.zeros(batch, channels, h_in * w_in, device=grad.device, dtype=grad.dtype)
            indices_flat = indices.view(batch, channels, -1)
            grad_flat = grad.view(batch, channels, -1)
            out.scatter_add_(2, indices_flat, grad_flat)

            return out.view(batch, channels, h_in, w_in)

        return unpool(delta_pp), unpool(delta_np), unpool(delta_pn), unpool(delta_nn)

    def _backward_avgpool(
        self, cache: DCCache, module: nn.Module,
        delta_pp: Tensor, delta_np: Tensor, delta_pn: Tensor, delta_nn: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Backward through AvgPool2d or AdaptiveAvgPool2d layer."""
        input_shape = cache.input_pos.shape

        def unpool_avg(grad: Tensor) -> Tensor:
            return F.interpolate(grad, size=(input_shape[2], input_shape[3]), mode='nearest')

        # Compute kernel area for gradient scaling
        if isinstance(module, nn.AvgPool2d):
            k = module.kernel_size
            kernel_area = k * k if isinstance(k, int) else k[0] * k[1]
        else:
            # AdaptiveAvgPool2d
            kernel_area = (input_shape[2] * input_shape[3]) / (delta_pp.shape[2] * delta_pp.shape[3])

        return (
            unpool_avg(delta_pp) / kernel_area,
            unpool_avg(delta_np) / kernel_area,
            unpool_avg(delta_pn) / kernel_area,
            unpool_avg(delta_nn) / kernel_area
        )

    def _backward_softmax(
        self, module: nn.Softmax, cache: DCCache,
        delta_pp: Tensor, delta_np: Tensor, delta_pn: Tensor, delta_nn: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Backward through Softmax layer using the Jacobian.

        The Jacobian of softmax is: J[i,j] = s[i] * (delta[i,j] - s[j])
        For gradient v, the Jacobian-vector product is: s * (v - <s, v>)
        where <s, v> is the dot product along the softmax dimension.

        Since forward outputs: output_pos = softmax(z), output_neg = 0
        All sensitivities are multiplied by the Jacobian.
        """
        dim = getattr(module, DC_SOFTMAX_DIM, -1)

        # Recompute softmax from cached input
        z = cache.z_before
        s = F.softmax(z, dim=dim)

        def jacobian_vector_product(v: Tensor) -> Tensor:
            """Compute J @ v = s * (v - <s, v>)"""
            # <s, v> summed along softmax dim, keeping dims for broadcasting
            sv = (s * v).sum(dim=dim, keepdim=True)
            return s * (v - sv)

        # Apply Jacobian to all 4 sensitivities
        # Since output_pos = softmax(input_pos - input_neg) and output_neg = 0:
        # - delta_pp flows through pos -> pos path
        # - delta_np flows through neg -> pos path (negative sign in chain rule)
        # - delta_pn and delta_nn are zero from output_neg but may carry upstream gradients

        new_delta_pp = jacobian_vector_product(delta_pp)
        new_delta_np = -jacobian_vector_product(delta_pp)  # Negative because d(pos-neg)/d(neg) = -1
        new_delta_pn = jacobian_vector_product(delta_pn)
        new_delta_nn = -jacobian_vector_product(delta_pn)  # Negative because d(pos-neg)/d(neg) = -1

        return new_delta_pp, new_delta_np, new_delta_pn, new_delta_nn

    def _backward_layernorm(
        self, module: nn.LayerNorm, cache: DCCache,
        delta_pp: Tensor, delta_np: Tensor, delta_pn: Tensor, delta_nn: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Backward through LayerNorm layer (variance treated as constant).

        Since variance is constant, the backward is similar to a linear scaling.
        The Jacobian of LayerNorm w.r.t. input x is:
        dy/dx = gamma / std * (I - 1/n * 11^T)

        For simplicity with constant variance, we treat it as:
        dy/dx ≈ gamma / std (ignoring mean subtraction gradient)
        """
        z = cache.z_before
        std = cache.pool_indices  # We stored std here

        normalized_shape = getattr(module, DC_LN_NORMALIZED_SHAPE, module.normalized_shape)
        dims = tuple(range(-len(normalized_shape), 0))
        n = 1
        for d in normalized_shape:
            n *= d

        # Compute normalized value for determining pos/neg regions
        mean = z.mean(dim=dims, keepdim=True)
        z_norm = (z - mean) / std

        if hasattr(module, DC_LN_SCALE_POS):
            scale_pos = getattr(module, DC_LN_SCALE_POS)
            scale_neg = getattr(module, DC_LN_SCALE_NEG)

            # Mask for where z_norm is positive or negative
            mask_pos = (z_norm >= 0).float()
            mask_neg = (z_norm < 0).float()

            # Effective scale based on region
            # Where z_norm >= 0: d(out_pos)/d(z_norm) = scale_pos, d(out_neg)/d(z_norm) = scale_neg
            # Where z_norm < 0: d(out_pos)/d(z_norm) = -scale_neg, d(out_neg)/d(z_norm) = -scale_pos

            # Gradient through normalization (d(z_norm)/d(z) = 1/std with variance constant)
            inv_std = 1.0 / std

            # Combined gradient computation
            # delta w.r.t. z_norm
            grad_z_norm_from_pp = delta_pp * scale_pos * mask_pos - delta_pp * scale_neg * mask_neg
            grad_z_norm_from_np = delta_np * scale_neg * mask_pos - delta_np * scale_pos * mask_neg
            grad_z_norm_from_pn = delta_pn * scale_neg * mask_pos - delta_pn * scale_pos * mask_neg
            grad_z_norm_from_nn = delta_nn * scale_pos * mask_pos - delta_nn * scale_neg * mask_neg

            # Gradient w.r.t. z (through normalization, variance constant)
            new_delta_pp = (grad_z_norm_from_pp + grad_z_norm_from_pn) * inv_std
            new_delta_np = -new_delta_pp  # Because d(z)/d(neg) = -1
            new_delta_pn = (grad_z_norm_from_np + grad_z_norm_from_nn) * inv_std
            new_delta_nn = -new_delta_pn

        else:
            # No learnable scale - just normalization
            inv_std = 1.0 / std
            new_delta_pp = delta_pp * inv_std
            new_delta_np = -delta_pp * inv_std
            new_delta_pn = delta_pn * inv_std
            new_delta_nn = -delta_pn * inv_std

        return new_delta_pp, new_delta_np, new_delta_pn, new_delta_nn

    def _backward_dc_matmul(
        self, module: nn.Module, cache: DCCache,
        delta_pp: Tensor, delta_np: Tensor, delta_pn: Tensor, delta_nn: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Backward through DCMatMul using product rule.

        Forward: C_pos = A_pos @ B_pos + A_neg @ B_neg
                 C_neg = A_pos @ B_neg + A_neg @ B_pos

        Backward (w.r.t. A, B is treated as second operand):
        d(C_pos)/d(A_pos) = grad @ B_pos^T
        d(C_pos)/d(A_neg) = grad @ B_neg^T
        d(C_neg)/d(A_pos) = grad @ B_neg^T
        d(C_neg)/d(A_neg) = grad @ B_pos^T

        Combined sensitivities:
        new_delta_pp = delta_pp @ B_pos^T + delta_pn @ B_neg^T
        new_delta_np = delta_pp @ B_neg^T + delta_pn @ B_pos^T
        new_delta_pn = delta_np @ B_pos^T + delta_nn @ B_neg^T
        new_delta_nn = delta_np @ B_neg^T + delta_nn @ B_pos^T
        """
        # Get B's pos/neg from cache
        B_pos, B_neg = cache.z_before

        # Transpose B for backward pass
        B_pos_T = B_pos.transpose(-2, -1)
        B_neg_T = B_neg.transpose(-2, -1)

        # Compute new sensitivities
        new_delta_pp = torch.matmul(delta_pp, B_pos_T) + torch.matmul(delta_pn, B_neg_T)
        new_delta_np = torch.matmul(delta_pp, B_neg_T) + torch.matmul(delta_pn, B_pos_T)
        new_delta_pn = torch.matmul(delta_np, B_pos_T) + torch.matmul(delta_nn, B_neg_T)
        new_delta_nn = torch.matmul(delta_np, B_neg_T) + torch.matmul(delta_nn, B_pos_T)

        return new_delta_pp, new_delta_np, new_delta_pn, new_delta_nn

    # =========================================================================
    # Access Methods
    # =========================================================================

    def get_activation(self, name: str) -> Optional[Tuple[Tensor, Tensor]]:
        """Get (pos, neg) activation for a layer."""
        if name not in self.caches:
            return None
        cache = self.caches[name]
        if cache.output_pos is None:
            return None
        return cache.output_pos, cache.output_neg

    def get_sensitivity(self, name: str) -> Optional[Tuple[Tensor, Tensor, Tensor, Tensor]]:
        """Get (delta_pp, delta_np, delta_pn, delta_nn) for a layer."""
        if name not in self.caches:
            return None
        cache = self.caches[name]
        if cache.delta_pp is None:
            return None
        return cache.delta_pp, cache.delta_np, cache.delta_pn, cache.delta_nn

    def get_combined_gradient(self, name: str) -> Optional[Tensor]:
        """
        Get the combined gradient for a layer.

        The combined gradient w.r.t. the original activation is:
        grad = (delta_pp - delta_np) - (delta_pn - delta_nn)
        """
        sens = self.get_sensitivity(name)
        if sens is None:
            return None
        delta_pp, delta_np, delta_pn, delta_nn = sens
        return (delta_pp - delta_np) - (delta_pn - delta_nn)

    def verify_reconstruction(self, tolerance: float = 1e-5) -> Dict[str, float]:
        """Verify pos - neg = original for all layers."""
        errors = {}
        for name, cache in self.caches.items():
            if cache.output_pos is not None and cache.original_output is not None:
                reconstructed = cache.output_pos - cache.output_neg
                error = (reconstructed - cache.original_output).abs().max().item()
                errors[name] = error
        return errors

    # =========================================================================
    # Cleanup
    # =========================================================================

    def remove_hooks(self, remove_attributes: bool = False):
        """
        Remove all hooks.

        Args:
            remove_attributes: If True, also remove DC attributes from modules.
                               Default False to avoid issues when reusing models.
        """
        for handle in self._forward_handles:
            handle.remove()
        self._forward_handles.clear()

        if remove_attributes:
            self._remove_attributes()

    def _remove_attributes(self):
        """Remove DC attributes from modules."""
        for module in self.modules.values():
            for attr in [DC_SHIFT_MODE, DC_BETA, DC_RELU_MODE, DC_ENABLED, DC_CACHE_ACTIVATIONS,
                         DC_WEIGHT_POS, DC_WEIGHT_NEG, DC_BIAS_POS, DC_BIAS_NEG,
                         DC_BN_SCALE_POS, DC_BN_SCALE_NEG, DC_SOFTMAX_DIM,
                         DC_LN_SCALE_POS, DC_LN_SCALE_NEG, DC_LN_NORMALIZED_SHAPE]:
                if hasattr(module, attr):
                    delattr(module, attr)

    def __del__(self):
        try:
            # Only remove hooks, not attributes (to avoid race conditions)
            for handle in self._forward_handles:
                handle.remove()
        except Exception:
            pass
