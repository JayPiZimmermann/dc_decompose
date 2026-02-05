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


class InputMode(Enum):
    """Mode for splitting input activations into pos/neg streams (input layer only)."""
    CENTER = "center"    # pos = ReLU(x), neg = ReLU(-x) - ensures non-negativity (DEFAULT)
    POSITIVE = "positive"  # pos = x, neg = 0 - all in positive stream
    NEGATIVE = "negative"  # pos = 0, neg = -x - all in negative stream
    BETA = "beta"        # pos = beta * x, neg = -(1-beta) * x - configurable split


class BackwardMode(Enum):
    """Mode for gradient shifting in backward pass."""
    NONE = "none"        # No gradient shifting
    ALPHA = "alpha"      # α-shifting: δ_pp -= α(δ_pp + δ_np), δ_np -= α(δ_pp + δ_np), etc.


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


# Import helpers from operations module
from .operations.relu import forward_relu, backward_relu
from .operations.add import Add


def _recenter_stacked(stacked: Tensor) -> Tensor:
    """
    Re-center stacked DC representation [2, batch, ...] to minimize magnitudes.

    Given stacked[0] = pos, stacked[1] = neg where z = pos - neg:
    - Computes new_pos = ReLU(z), new_neg = ReLU(-z)
    - Returns stacked [new_pos, new_neg]

    This preserves z = new_pos - new_neg but ensures minimal magnitudes.
    """
    pos, neg = stacked[0], stacked[1]
    z = pos - neg
    new_pos = torch.relu(z)
    new_neg = torch.relu(-z)
    return torch.stack([new_pos, new_neg], dim=0)

# Attribute names stored on original modules
DC_INPUT_MODE = '_dc_input_mode'
DC_BETA = '_dc_beta'
DC_BACKWARD_MODE = '_dc_backward_mode'
DC_ALPHA = '_dc_alpha'
DC_RELU_MODE = '_dc_relu_mode'
DC_BACKPROP_MODE = '_dc_backprop_mode'
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
        input_mode: InputMode = InputMode.CENTER,
        beta: float = 1.0,
        backward_mode: BackwardMode = BackwardMode.ALPHA,
        alpha: float = 0.35,
        relu_mode: ReLUMode = ReLUMode.MAX,
        relu_backprop_mode: str = 'standard',
        cache_activations: bool = True,
        target_layers: Optional[List[str]] = None,
        auto_convert_functional: bool = True,
    ):
        """
        Initialize HookDecomposer.

        Args:
            model: The PyTorch model to decompose
            input_mode: How to split input activations into pos/neg streams (input layer only):
                - CENTER (default): pos = ReLU(x), neg = ReLU(-x)
                  Ensures both streams are non-negative (true DC property)
                - POSITIVE: pos = x, neg = 0 (all in positive stream)
                - NEGATIVE: pos = 0, neg = -x (all in negative stream)
                - BETA: pos = beta * x, neg = -(1-beta) * x (configurable)
            beta: Split parameter for BETA input mode (default 1.0)
            backward_mode: Gradient shifting strategy for backward pass:
                - ALPHA (default): Apply α-shifting to stabilize gradients
                - NONE: No gradient shifting
            alpha: α parameter for gradient shifting (default 0.35, recommended range 0.2-0.5)
            relu_mode: How to decompose ReLU activations (max/min/half)
            relu_backprop_mode: ReLU backprop mode ('standard', 'mask_diff', 'sum')
            cache_activations: Whether to cache activations (required for backward)
            target_layers: Optional list of layer names to decompose (None = all)
            auto_convert_functional: If True, automatically convert functional operations
                (torch.relu, +, torch.softmax, etc.) to module equivalents (default True)
        """
        # Auto-convert functional operations to modules if requested
        if auto_convert_functional:
            from .operations.functional_replacer import make_dc_compatible
            model = make_dc_compatible(model)

        self.model = model
        self.input_mode = input_mode
        self.beta = beta
        self.backward_mode = backward_mode
        self.alpha = alpha
        self.relu_mode = relu_mode
        self.relu_backprop_mode = relu_backprop_mode
        self.cache_activations = cache_activations
        self.target_layers = target_layers

        # Layer caches and ordering
        self.caches: Dict[str, DCCache] = {}
        self.layer_order: List[str] = []  # Module tree order (for module registration)
        self.execution_order: List[str] = []  # Actual execution order (populated during forward)
        self.modules: Dict[str, nn.Module] = {}

        # Stacked tensor cache: maps original tensor data_ptr to stacked version
        # This handles branching where multiple modules receive the same input
        self._stacked_cache: Dict[int, Tensor] = {}
        self._initialized: bool = False

        # Hook handles
        self._forward_handles: List = []
        
        # Gradient accumulation for residual connections
        self._gradient_accumulators: Dict[str, Dict[str, Tensor]] = {}
        self._tensor_hooks: List = []
        
        # Hook bypass functionality
        self._hooks_enabled: bool = True
        self._original_forwards: Dict[str, Any] = {}

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
            nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.MaxPool2d, nn.AvgPool2d,
            nn.Flatten, nn.AdaptiveAvgPool2d, nn.Dropout, nn.Identity, Add,
        ))

    def _register_hooks(self):
        """Register forward hooks and store DC parameters on original modules."""
        for name, module in self.model.named_modules():
            # Handle root module case - if it's a supported single layer model
            if name == "":
                # Only register root module if it's the only module and supported
                modules_list = list(self.model.named_modules())
                if len(modules_list) == 1 and self._is_supported(module):
                    name = "root"  # Give it a name
                else:
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
        setattr(module, DC_INPUT_MODE, self.input_mode)
        setattr(module, DC_BETA, self.beta)
        setattr(module, DC_BACKWARD_MODE, self.backward_mode)
        setattr(module, DC_ALPHA, self.alpha)
        setattr(module, DC_RELU_MODE, self.relu_mode)
        setattr(module, DC_BACKPROP_MODE, self.relu_backprop_mode)
        setattr(module, DC_ENABLED, True)
        setattr(module, DC_CACHE_ACTIVATIONS, self.cache_activations)

        # No weight copying! All decomposition done with masked views on-the-fly

        if isinstance(module, nn.Softmax):
            # Store the softmax dimension
            setattr(module, DC_SOFTMAX_DIM, module.dim)

        elif isinstance(module, nn.LayerNorm):
            # Store normalized shape for LayerNorm (weights computed on-the-fly)
            setattr(module, DC_LN_NORMALIZED_SHAPE, module.normalized_shape)

    def _make_forward_hook(self, name: str):
        """Create forward hook that reads config from the module itself."""

        def hook(module: nn.Module, inputs: Tuple[Tensor, ...], output: Tensor):
            # Check if hooks are globally enabled
            if not self._hooks_enabled:
                return
                
            # Check if DC is enabled for this module
            if not getattr(module, DC_ENABLED, True):
                return

            cache = self.caches[name]
            should_cache = getattr(module, DC_CACHE_ACTIVATIONS, True)

            # Get stacked input from cache (handles branching) or initialize
            x = inputs[0]
            input_ptr = x.data_ptr()

            if input_ptr in self._stacked_cache:
                # Use cached stacked version for this input tensor (handles branching)
                stacked_input = self._stacked_cache[input_ptr]
            elif not self._initialized:
                # First layer: apply input_mode-based initialization
                input_mode = getattr(module, DC_INPUT_MODE, InputMode.CENTER)
                beta = getattr(module, DC_BETA, 1.0)

                if input_mode == InputMode.CENTER:
                    # pos = ReLU(x), neg = ReLU(-x) - ensures non-negativity
                    input_pos = F.relu(x)
                    input_neg = F.relu(-x)
                elif input_mode == InputMode.POSITIVE:
                    # pos = x, neg = 0 - all in positive stream
                    input_pos = x
                    input_neg = torch.zeros_like(x)
                elif input_mode == InputMode.NEGATIVE:
                    # pos = 0, neg = -x - all in negative stream
                    input_pos = torch.zeros_like(x)
                    input_neg = -x
                elif input_mode == InputMode.BETA:
                    # pos = beta * x, neg = -(1-beta) * x - configurable split
                    input_pos = beta * x
                    input_neg = -(1 - beta) * x
                else:
                    # Default to CENTER
                    input_pos = F.relu(x)
                    input_neg = F.relu(-x)

                # Stack into format [2, batch, ...]
                stacked_input = torch.stack([input_pos, input_neg], dim=0)
                self._stacked_cache[input_ptr] = stacked_input
                self._initialized = True
            else:
                # Use the most recent stacked output as fallback
                stacked_input = self._current_stacked
                
            # Cache inputs if requested (unstack only for caching)
            if should_cache:
                input_pos, input_neg = stacked_input[0], stacked_input[1]
                cache.input_pos = input_pos.detach()
                cache.input_neg = input_neg.detach()
                cache.original_output = output.detach()

            # Compute DC decomposition using ONLY stacked tensor approach
            if isinstance(module, nn.Linear):
                stacked_output = self._forward_linear_stacked(module, stacked_input)

            elif isinstance(module, nn.Conv2d):
                stacked_output = self._forward_conv2d_stacked(module, stacked_input)

            elif isinstance(module, nn.ReLU):
                stacked_output = self._forward_relu_stacked(
                    module, cache, stacked_input, should_cache
                )

            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                stacked_output = self._forward_batchnorm_stacked(module, stacked_input)

            elif isinstance(module, nn.MaxPool2d):
                stacked_output = self._forward_maxpool_stacked(
                    module, cache, stacked_input, should_cache
                )

            elif isinstance(module, nn.AvgPool2d):
                stacked_output = self._forward_avgpool_stacked(module, stacked_input)

            elif isinstance(module, nn.AdaptiveAvgPool2d):
                stacked_output = self._forward_adaptive_avgpool_stacked(module, stacked_input)

            elif isinstance(module, nn.Flatten):
                # Flatten: apply to each component separately to avoid dim issues
                input_pos, input_neg = stacked_input[0], stacked_input[1]
                output_pos = torch.flatten(input_pos, module.start_dim, module.end_dim)
                output_neg = torch.flatten(input_neg, module.start_dim, module.end_dim)
                stacked_output = torch.stack([output_pos, output_neg], dim=0)

            elif isinstance(module, nn.Softmax):
                stacked_output = self._forward_softmax_stacked(
                    module, cache, stacked_input, should_cache
                )

            elif isinstance(module, nn.LayerNorm):
                stacked_output = self._forward_layernorm_stacked(
                    module, cache, stacked_input, should_cache
                )

            elif isinstance(module, Add):
                # Add module has two inputs - get stacked versions of both
                stacked_output = self._forward_add_stacked(module, inputs, stacked_input)

            elif hasattr(module, '_dc_is_matmul') and module._dc_is_matmul:
                stacked_output = self._forward_dc_matmul_stacked(
                    module, cache, stacked_input, should_cache
                )

            # =========================================================
            # DC Operation Modules (native stacked tensor operations)
            # =========================================================
            elif getattr(module, '_dc_is_reshape', False):
                # Reshape: apply to stacked tensor, preserving stack dimension
                new_shape = (stacked_input.shape[0],) + module.target_shape
                stacked_output = stacked_input.view(*new_shape)

            elif getattr(module, '_dc_is_permute', False):
                # Permute: handle stacked tensor format [2, batch, ...]
                adjusted_dims = [0] + [d + 1 for d in module.dims]  # +1 because dim 0 is stack dim
                stacked_output = stacked_input.permute(*adjusted_dims)

            elif getattr(module, '_dc_is_transpose', False):
                # Transpose: handle stacked tensor format [2, batch, ...]
                adj_dim0 = module.dim0 + 1 if module.dim0 >= 0 else module.dim0
                adj_dim1 = module.dim1 + 1 if module.dim1 >= 0 else module.dim1
                stacked_output = stacked_input.transpose(adj_dim0, adj_dim1)

            elif getattr(module, '_dc_is_contiguous', False):
                # Contiguous: apply to stacked tensor
                stacked_output = stacked_input.contiguous()

            elif getattr(module, '_dc_is_scalar_mul', False):
                # Scalar multiplication on stacked tensor
                if module.is_negative:
                    # For negative scalar: swap streams [0,1] -> [1,0] and use absolute value
                    stacked_output = module.abs_scalar * stacked_input[[1, 0]]
                else:
                    # For positive scalar: multiply stacked tensor
                    stacked_output = module.scalar * stacked_input

            elif getattr(module, '_dc_is_scalar_div', False):
                # Scalar division on stacked tensor
                if module.is_negative:
                    # For negative scalar: swap streams [0,1] -> [1,0] and use absolute value
                    stacked_output = stacked_input[[1, 0]] / module.abs_scalar
                else:
                    # For positive scalar: divide stacked tensor
                    stacked_output = stacked_input / module.scalar

            elif getattr(module, '_dc_is_add', False):
                # Element-wise addition on stacked tensor
                # If we have stored operand decomposition, use it
                if hasattr(module, '_dc_operand_pos') and module._dc_operand_pos is not None:
                    stacked_operand = torch.stack([module._dc_operand_pos, module._dc_operand_neg], dim=0)
                    stacked_output = stacked_input + stacked_operand
                else:
                    # Automatic operand decomposition using CENTER mode like input
                    if hasattr(module, '_last_operand_cache') and module._last_operand_cache is not None:
                        operand = module._last_operand_cache
                        operand_pos = torch.relu(operand)
                        operand_neg = torch.relu(-operand)
                        stacked_operand = torch.stack([operand_pos, operand_neg], dim=0)
                        stacked_output = stacked_input + stacked_operand
                    else:
                        # Fallback: pass through (no addition)
                        stacked_output = stacked_input

            elif getattr(module, '_dc_is_slice', False):
                # Slice: apply to stacked tensor, adjusting for stack dimension
                slices = [slice(None)] * stacked_input.dim()
                slices[module.dim + 1] = slice(module.start, module.end)  # +1 for stack dim
                stacked_output = stacked_input[tuple(slices)]

            elif getattr(module, '_dc_is_dropout', False) or isinstance(module, nn.Dropout):
                # Dropout: identity in eval, same mask in train
                if module.training:
                    # Generate mask from output and apply to stacked tensor
                    mask = (output != 0).float() if hasattr(module, 'p') and module.p > 0 else None
                    if mask is not None:
                        scale = 1.0 / (1.0 - module.p) if hasattr(module, 'p') else 1.0
                        # Broadcast mask to match stacked tensor shape
                        mask_stacked = mask.unsqueeze(0).expand_as(stacked_input)
                        stacked_output = stacked_input * mask_stacked * scale
                    else:
                        stacked_output = stacked_input
                else:
                    stacked_output = stacked_input

            elif getattr(module, '_dc_is_identity', False) or isinstance(module, nn.Identity):
                # Identity: pass through stacked tensor
                stacked_output = stacked_input

            elif getattr(module, '_dc_is_mean', False):
                # Mean: linear operation on stacked tensor
                if module.dim is None:
                    stacked_output = stacked_input.mean()
                else:
                    # Adjust dimension for stack
                    adj_dim = module.dim + 1 if isinstance(module.dim, int) else tuple(d + 1 for d in module.dim)
                    stacked_output = stacked_input.mean(dim=adj_dim, keepdim=module.keepdim)

            elif getattr(module, '_dc_is_sum', False):
                # Sum: linear operation on stacked tensor
                if module.dim is None:
                    stacked_output = stacked_input.sum()
                else:
                    # Adjust dimension for stack
                    adj_dim = module.dim + 1 if isinstance(module.dim, int) else tuple(d + 1 for d in module.dim)
                    stacked_output = stacked_input.sum(dim=adj_dim, keepdim=module.keepdim)

            else:
                # Fallback: identity (pass through stacked tensor unchanged)
                stacked_output = stacked_input

            # Cache outputs if requested (unstack only for caching)
            if should_cache:
                output_pos, output_neg = stacked_output[0], stacked_output[1]
                cache.output_pos = output_pos.detach()
                cache.output_neg = output_neg.detach()

            # Update current stacked state for next layer
            self._current_stacked = stacked_output

            # Cache stacked output by output tensor's data_ptr for branching support
            # This allows subsequent layers that receive this output to use the stacked version
            output_ptr = output.data_ptr()
            self._stacked_cache[output_ptr] = stacked_output

        return hook

    def _forward_linear(self, module: nn.Linear, input_pos: Tensor, input_neg: Tensor) -> Tuple[Tensor, Tensor]:
        """DC forward for Linear layer using masked weight views (no copying!)."""
        # Create masked views of weights (no copying!)
        pos_mask = module.weight >= 0
        neg_mask = module.weight < 0
        W_pos = module.weight * pos_mask
        W_neg = -module.weight * neg_mask

        # pos_out = W_pos @ pos_in + W_neg @ neg_in + bias_pos
        output_pos = F.linear(input_pos, W_pos) + F.linear(input_neg, W_neg)
        
        # neg_out = W_neg @ pos_in + W_pos @ neg_in + bias_neg
        output_neg = F.linear(input_pos, W_neg) + F.linear(input_neg, W_pos)
        
        # Handle bias with masked views
        if module.bias is not None:
            bias_pos_mask = module.bias >= 0
            bias_neg_mask = module.bias < 0
            bias_pos = module.bias * bias_pos_mask
            bias_neg = -module.bias * bias_neg_mask
            
            output_pos = output_pos + bias_pos
            output_neg = output_neg + bias_neg

        return output_pos, output_neg

    def _forward_conv2d(self, module: nn.Conv2d, input_pos: Tensor, input_neg: Tensor) -> Tuple[Tensor, Tensor]:
        """DC forward for Conv2d layer using masked weight views (no copying!)."""
        # Create masked views of weights (no copying!)
        pos_mask = module.weight >= 0
        neg_mask = module.weight < 0
        W_pos = module.weight * pos_mask
        W_neg = -module.weight * neg_mask

        # pos_out = conv(pos_in, W_pos) + conv(neg_in, W_neg) + bias_pos
        output_pos = F.conv2d(input_pos, W_pos, None, module.stride, module.padding, module.dilation, module.groups)
        output_pos = output_pos + F.conv2d(input_neg, W_neg, None, module.stride, module.padding, module.dilation, module.groups)
        
        # neg_out = conv(pos_in, W_neg) + conv(neg_in, W_pos) + bias_neg
        output_neg = F.conv2d(input_pos, W_neg, None, module.stride, module.padding, module.dilation, module.groups)
        output_neg = output_neg + F.conv2d(input_neg, W_pos, None, module.stride, module.padding, module.dilation, module.groups)
        
        # Handle bias with masked views
        if module.bias is not None:
            bias_pos_mask = module.bias >= 0
            bias_neg_mask = module.bias < 0
            bias_pos = module.bias * bias_pos_mask
            bias_neg = -module.bias * bias_neg_mask
            
            output_pos = output_pos + bias_pos.view(1, -1, 1, 1)
            output_neg = output_neg + bias_neg.view(1, -1, 1, 1)

        return output_pos, output_neg

    def _forward_relu(
        self, module: nn.ReLU, cache: DCCache, input_pos: Tensor, input_neg: Tensor, should_cache: bool
    ) -> Tuple[Tensor, Tensor]:
        """DC forward for ReLU layer. Reads config from module."""
        # Cache z_before for backward pass
        if should_cache:
            cache.z_before = (input_pos - input_neg).detach()

        # Read split_mode from module
        relu_mode = getattr(module, DC_RELU_MODE, ReLUMode.MAX)
        split_mode = relu_mode.value if isinstance(relu_mode, ReLUMode) else relu_mode
        return forward_relu(input_pos, input_neg, split_mode)

    def _forward_batchnorm(self, module: nn.BatchNorm2d, input_pos: Tensor, input_neg: Tensor) -> Tuple[Tensor, Tensor]:
        """DC forward for BatchNorm2d layer (variance treated as constant) using masked views."""
        # Compute effective scale: gamma / sqrt(var + eps)
        with torch.no_grad():
            std = torch.sqrt(module.running_var + module.eps)
            scale = module.weight / std
            bias = module.bias - module.running_mean * scale
        
        # Create masked views (no copying!)
        scale_pos_mask = scale >= 0
        scale_neg_mask = scale < 0
        scale_pos = scale * scale_pos_mask
        scale_neg = -scale * scale_neg_mask
        
        bias_pos_mask = bias >= 0
        bias_neg_mask = bias < 0
        bias_pos = bias * bias_pos_mask
        bias_neg = -bias * bias_neg_mask
        
        # Reshape for broadcasting
        scale_pos = scale_pos.view(1, -1, 1, 1)
        scale_neg = scale_neg.view(1, -1, 1, 1)
        bias_pos = bias_pos.view(1, -1, 1, 1)
        bias_neg = bias_neg.view(1, -1, 1, 1)

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

        if module.weight is not None:
            # Create masked views of weights (no copying!)
            scale_pos_mask = module.weight >= 0
            scale_neg_mask = module.weight < 0
            scale_pos = module.weight * scale_pos_mask
            scale_neg = -module.weight * scale_neg_mask

            # Decompose z_norm into pos/neg
            z_norm_pos = F.relu(z_norm)
            z_norm_neg = F.relu(-z_norm)

            output_pos = scale_pos * z_norm_pos + scale_neg * z_norm_neg
            output_neg = scale_neg * z_norm_pos + scale_pos * z_norm_neg
        else:
            # No learnable scale
            output_pos = F.relu(z_norm)
            output_neg = F.relu(-z_norm)

        # Add bias with masked views
        if module.bias is not None:
            bias_pos_mask = module.bias >= 0
            bias_neg_mask = module.bias < 0
            bias_pos = module.bias * bias_pos_mask
            bias_neg = -module.bias * bias_neg_mask
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
    # Stacked Tensor Forward Methods (ONLY APPROACH)
    # =========================================================================

    def _forward_linear_stacked(self, module: nn.Linear, stacked_input: Tensor) -> Tensor:
        """DC forward for Linear layer using stacked tensors and masked weight views."""
        # stacked_input: [2, batch, features] where [0] = pos, [1] = neg
        
        # Create masked views of weights (no copying!)
        pos_mask = module.weight >= 0
        neg_mask = module.weight < 0
        W_pos = module.weight * pos_mask
        W_neg = -module.weight * neg_mask
        
        input_pos, input_neg = stacked_input[0], stacked_input[1]
        
        # pos_out = W_pos @ pos_in + W_neg @ neg_in
        output_pos = F.linear(input_pos, W_pos) + F.linear(input_neg, W_neg)
        
        # neg_out = W_neg @ pos_in + W_pos @ neg_in  
        output_neg = F.linear(input_pos, W_neg) + F.linear(input_neg, W_pos)
        
        # Handle bias with masked views
        if module.bias is not None:
            bias_pos_mask = module.bias >= 0
            bias_neg_mask = module.bias < 0
            bias_pos = module.bias * bias_pos_mask
            bias_neg = -module.bias * bias_neg_mask
            
            output_pos = output_pos + bias_pos
            output_neg = output_neg + bias_neg
        
        return torch.stack([output_pos, output_neg], dim=0)

    def _forward_conv2d_stacked(self, module: nn.Conv2d, stacked_input: Tensor) -> Tensor:
        """DC forward for Conv2d layer using stacked tensors and masked weight views."""
        # stacked_input: [2, batch, channels, height, width]
        
        pos_mask = module.weight >= 0
        neg_mask = module.weight < 0
        W_pos = module.weight * pos_mask
        W_neg = -module.weight * neg_mask
        
        input_pos, input_neg = stacked_input[0], stacked_input[1]
        
        output_pos = F.conv2d(input_pos, W_pos, None, module.stride, module.padding, module.dilation, module.groups)
        output_pos += F.conv2d(input_neg, W_neg, None, module.stride, module.padding, module.dilation, module.groups)
        
        output_neg = F.conv2d(input_pos, W_neg, None, module.stride, module.padding, module.dilation, module.groups)
        output_neg += F.conv2d(input_neg, W_pos, None, module.stride, module.padding, module.dilation, module.groups)
        
        if module.bias is not None:
            bias_pos_mask = module.bias >= 0
            bias_neg_mask = module.bias < 0
            bias_pos = module.bias * bias_pos_mask
            bias_neg = -module.bias * bias_neg_mask
            
            output_pos = output_pos + bias_pos.view(1, -1, 1, 1)
            output_neg = output_neg + bias_neg.view(1, -1, 1, 1)
        
        return torch.stack([output_pos, output_neg], dim=0)

    def _forward_relu_stacked(
        self, module: nn.ReLU, cache: DCCache, stacked_input: Tensor, should_cache: bool
    ) -> Tensor:
        """DC forward for ReLU using stacked tensors. Reads config from module."""
        input_pos, input_neg = stacked_input[0], stacked_input[1]

        # Cache z_before for backward pass if requested
        if should_cache:
            cache.z_before = input_pos - input_neg

        # Read split_mode from module
        relu_mode = getattr(module, DC_RELU_MODE, ReLUMode.MAX)
        split_mode = relu_mode.value if isinstance(relu_mode, ReLUMode) else relu_mode
        output_pos, output_neg = forward_relu(input_pos, input_neg, split_mode)

        return torch.stack([output_pos, output_neg], dim=0)

    def _forward_batchnorm_stacked(self, module, stacked_input: Tensor) -> Tensor:
        """DC forward for BatchNorm1d/2d using stacked tensors."""
        input_pos, input_neg = stacked_input[0], stacked_input[1]
        
        # Compute effective scale: gamma / sqrt(var + eps)
        with torch.no_grad():
            scale = module.weight / torch.sqrt(module.running_var + module.eps)
            bias = module.bias - scale * module.running_mean
        
        # Create masked views
        scale_pos = F.relu(scale)
        scale_neg = F.relu(-scale)
        bias_pos = F.relu(bias)
        bias_neg = F.relu(-bias)
        
        # Reshape for broadcasting - handle both 1D and 2D cases
        if isinstance(module, nn.BatchNorm1d):
            # 1D case: [batch, channels] -> [1, channels]
            scale_pos = scale_pos.view(1, -1)
            scale_neg = scale_neg.view(1, -1)
            bias_pos = bias_pos.view(1, -1)
            bias_neg = bias_neg.view(1, -1)
        else:
            # 2D case: [batch, channels, height, width] -> [1, channels, 1, 1]
            scale_pos = scale_pos.view(1, -1, 1, 1)
            scale_neg = scale_neg.view(1, -1, 1, 1)
            bias_pos = bias_pos.view(1, -1, 1, 1)
            bias_neg = bias_neg.view(1, -1, 1, 1)
        
        output_pos = scale_pos * input_pos + scale_neg * input_neg + bias_pos
        output_neg = scale_neg * input_pos + scale_pos * input_neg + bias_neg
        
        return torch.stack([output_pos, output_neg], dim=0)

    def _forward_maxpool_stacked(
        self, module: nn.MaxPool2d, cache: DCCache,
        stacked_input: Tensor, should_cache: bool
    ) -> Tensor:
        """DC forward for MaxPool2d using stacked tensors with winner-takes-all."""
        input_pos, input_neg = stacked_input[0], stacked_input[1]

        # Get original activation for determining winners
        z_before = input_pos - input_neg

        # Get argmax indices from original activation
        _, indices = F.max_pool2d(
            z_before, module.kernel_size, module.stride, module.padding,
            return_indices=True
        )

        # Cache indices for backward pass
        if should_cache:
            cache.pool_indices = indices.detach()

        # Apply same indices to both pos and neg streams
        batch, channels, h_in, w_in = input_pos.shape
        h_out, w_out = indices.shape[2], indices.shape[3]

        # Flatten for gather operation
        pos_flat = input_pos.view(batch, channels, -1)
        neg_flat = input_neg.view(batch, channels, -1)
        indices_flat = indices.view(batch, channels, -1)

        # Gather using indices
        output_pos = torch.gather(pos_flat, 2, indices_flat).view(batch, channels, h_out, w_out)
        output_neg = torch.gather(neg_flat, 2, indices_flat).view(batch, channels, h_out, w_out)

        return torch.stack([output_pos, output_neg], dim=0)

    def _forward_avgpool_stacked(self, module: nn.AvgPool2d, stacked_input: Tensor) -> Tensor:
        """DC forward for AvgPool2d using stacked tensors (linear operation)."""
        input_pos, input_neg = stacked_input[0], stacked_input[1]
        output_pos = F.avg_pool2d(input_pos, module.kernel_size, module.stride, module.padding)
        output_neg = F.avg_pool2d(input_neg, module.kernel_size, module.stride, module.padding)
        return torch.stack([output_pos, output_neg], dim=0)

    def _forward_adaptive_avgpool_stacked(self, module: nn.AdaptiveAvgPool2d, stacked_input: Tensor) -> Tensor:
        """DC forward for AdaptiveAvgPool2d using stacked tensors (linear operation)."""
        input_pos, input_neg = stacked_input[0], stacked_input[1]
        output_pos = F.adaptive_avg_pool2d(input_pos, module.output_size)
        output_neg = F.adaptive_avg_pool2d(input_neg, module.output_size)
        return torch.stack([output_pos, output_neg], dim=0)

    def _forward_add_stacked(self, module: Add, inputs: Tuple[Tensor, ...], stacked_x: Tensor) -> Tensor:
        """DC forward for Add module using stacked tensors with re-centering."""
        # Add module has two inputs: x and y
        # stacked_x is the stacked version of x (first input)
        # We need to get or create stacked version of y (second input)

        if len(inputs) < 2:
            # Fallback: if only one input, return it unchanged
            return stacked_x

        y = inputs[1]
        y_ptr = y.data_ptr()

        # Get or create stacked version of y
        if y_ptr in self._stacked_cache:
            stacked_y = self._stacked_cache[y_ptr]
        else:
            # Initialize y using CENTER mode (same as input initialization)
            input_mode = getattr(module, DC_INPUT_MODE, InputMode.CENTER)
            beta = getattr(module, DC_BETA, 1.0)

            if input_mode == InputMode.CENTER:
                y_pos = F.relu(y)
                y_neg = F.relu(-y)
            elif input_mode == InputMode.POSITIVE:
                y_pos = y
                y_neg = torch.zeros_like(y)
            elif input_mode == InputMode.NEGATIVE:
                y_pos = torch.zeros_like(y)
                y_neg = -y
            elif input_mode == InputMode.BETA:
                y_pos = beta * y
                y_neg = -(1 - beta) * y
            else:
                y_pos = F.relu(y)
                y_neg = F.relu(-y)

            stacked_y = torch.stack([y_pos, y_neg], dim=0)
            self._stacked_cache[y_ptr] = stacked_y

        # Add the two stacked tensors
        stacked_sum = stacked_x + stacked_y

        # Apply re-centering if enabled (prevents magnitude explosion)
        if getattr(module, 'recenter', True):
            stacked_sum = _recenter_stacked(stacked_sum)

        return stacked_sum

    def _forward_softmax_stacked(
        self, module: nn.Softmax, cache: DCCache,
        stacked_input: Tensor, should_cache: bool
    ) -> Tensor:
        """DC forward for Softmax using stacked tensors."""
        input_pos, input_neg = stacked_input[0], stacked_input[1]
        
        # Reconstruct original activation
        z = input_pos - input_neg
        
        # Apply softmax
        dim = getattr(module, DC_SOFTMAX_DIM, module.dim)
        output_pos = F.softmax(z, dim=dim)
        output_neg = torch.zeros_like(output_pos)
        
        return torch.stack([output_pos, output_neg], dim=0)

    def _forward_layernorm_stacked(
        self, module: nn.LayerNorm, cache: DCCache,
        stacked_input: Tensor, should_cache: bool
    ) -> Tensor:
        """DC forward for LayerNorm using stacked tensors."""
        input_pos, input_neg = stacked_input[0], stacked_input[1]
        
        # Reconstruct original for normalization stats
        z = input_pos - input_neg
        normalized_shape = getattr(module, DC_LN_NORMALIZED_SHAPE, module.normalized_shape)
        
        # Compute normalization statistics from original
        dims = tuple(range(-len(normalized_shape), 0))
        mean = z.mean(dim=dims, keepdim=True)
        var = z.var(dim=dims, keepdim=True, unbiased=False)
        
        # Normalize original
        z_norm = (z - mean) / torch.sqrt(var + module.eps)
        
        # Create decomposed normalized output
        output_pos = F.relu(z_norm)
        output_neg = F.relu(-z_norm)
        
        # Apply weight and bias with masking
        if module.weight is not None:
            weight_pos = F.relu(module.weight)
            weight_neg = F.relu(-module.weight)
            
            output_pos = weight_pos * output_pos + weight_neg * output_neg
            output_neg = weight_neg * output_pos + weight_pos * output_neg
        
        if module.bias is not None:
            bias_pos = F.relu(module.bias)
            bias_neg = F.relu(-module.bias)
            
            output_pos = output_pos + bias_pos
            output_neg = output_neg + bias_neg
        
        return torch.stack([output_pos, output_neg], dim=0)

    def _forward_dc_matmul_stacked(
        self, module: nn.Module, cache: DCCache,
        stacked_input: Tensor, should_cache: bool
    ) -> Tensor:
        """DC forward for DCMatMul using stacked tensors."""
        input_pos, input_neg = stacked_input[0], stacked_input[1]
        
        # Get decomposed B matrix from module
        if hasattr(module, '_dc_B_pos') and hasattr(module, '_dc_B_neg'):
            B_pos = module._dc_B_pos
            B_neg = module._dc_B_neg
        else:
            # Fallback: decompose B on the fly
            B = module.B if hasattr(module, 'B') else torch.eye(input_pos.shape[-1])
            B_pos = F.relu(B)
            B_neg = F.relu(-B)
        
        # DC matmul: (A_pos - A_neg) @ (B_pos - B_neg) = (A_pos @ B_pos + A_neg @ B_neg) - (A_pos @ B_neg + A_neg @ B_pos)
        output_pos = torch.matmul(input_pos, B_pos) + torch.matmul(input_neg, B_neg)
        output_neg = torch.matmul(input_pos, B_neg) + torch.matmul(input_neg, B_pos)
        
        return torch.stack([output_pos, output_neg], dim=0)

    # =========================================================================
    # Initialization and State Management
    # =========================================================================

    def initialize(self, x: Optional[Tensor] = None):
        """
        Initialize for a new forward pass.

        Call this before model(x) to reset the DC state.
        """
        self._current_stacked = None
        self._initialized = False
        self._stacked_cache.clear()

        for cache in self.caches.values():
            cache.clear()

    def set_input_mode(self, mode: InputMode):
        """
        Update the input mode on all modules.

        Args:
            mode: New input mode for input splitting (first layer only)
        """
        self.input_mode = mode
        for module in self.modules.values():
            setattr(module, DC_INPUT_MODE, mode)

    def set_backward_mode(self, mode: BackwardMode):
        """
        Update the backward mode on all modules.

        Args:
            mode: New backward mode for gradient shifting
        """
        self.backward_mode = mode
        for module in self.modules.values():
            setattr(module, DC_BACKWARD_MODE, mode)

    def set_alpha(self, alpha: float):
        """
        Update the α parameter on all modules (for ALPHA backward mode).

        Args:
            alpha: New α parameter for gradient shifting (recommended: 0.2-0.5)
        """
        self.alpha = alpha
        for module in self.modules.values():
            setattr(module, DC_ALPHA, alpha)

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
    
    def enable_hooks(self, enabled: bool = True):
        """Enable or disable DC decomposition hooks globally.
        
        When disabled, the model behaves as the original model without decomposition.
        This allows for easy comparison between DC and original model behavior.
        
        Args:
            enabled: If True, enable DC decomposition. If False, use original model behavior.
        """
        self._hooks_enabled = enabled
    
    def disable_hooks(self):
        """Convenience method to disable hooks (use original model behavior)."""
        self.enable_hooks(False)
        
    @property
    def hooks_enabled(self) -> bool:
        """Check if hooks are currently enabled."""
        return self._hooks_enabled

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

        # Stack into format [4, batch, ...] where:
        # [0] = delta_pp, [1] = delta_np, [2] = delta_pn, [3] = delta_nn
        stacked_gradients = torch.stack([
            grad_output_pos.clone(),  # delta_pp
            torch.zeros_like(grad_output_pos),  # delta_np  
            grad_output_neg.clone(),  # delta_pn
            torch.zeros_like(grad_output_neg),  # delta_nn
        ], dim=0)

        # Backpropagate through layers in reverse order using ONLY stacked tensors
        for name in reversed(self.layer_order):
            cache = self.caches[name]
            module = self.modules[name]

            # Check if this layer was enabled
            if not getattr(module, DC_ENABLED, True):
                continue

            if isinstance(module, nn.Linear):
                stacked_gradients = self._backward_linear_stacked(module, stacked_gradients)

            elif isinstance(module, nn.Conv2d):
                stacked_gradients = self._backward_conv2d_stacked(module, stacked_gradients)

            elif isinstance(module, nn.ReLU):
                stacked_gradients = self._backward_relu_stacked(
                    module, cache, stacked_gradients
                )

            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                stacked_gradients = self._backward_batchnorm_stacked(module, stacked_gradients)

            elif isinstance(module, nn.MaxPool2d):
                stacked_gradients = self._backward_maxpool_stacked(cache, stacked_gradients)

            elif isinstance(module, (nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
                stacked_gradients = self._backward_avgpool_stacked(cache, module, stacked_gradients)

            elif isinstance(module, nn.Softmax):
                stacked_gradients = self._backward_softmax_stacked(module, cache, stacked_gradients)

            elif isinstance(module, nn.LayerNorm):
                stacked_gradients = self._backward_layernorm_stacked(module, cache, stacked_gradients)

            elif hasattr(module, '_dc_is_matmul') and module._dc_is_matmul:
                stacked_gradients = self._backward_dc_matmul_stacked(module, cache, stacked_gradients)

            elif isinstance(module, nn.Flatten):
                # Flatten backward: reshape stacked gradients to input shape  
                input_shape = cache.input_pos.shape
                new_shape = (stacked_gradients.shape[0],) + input_shape  # [4, batch, ...]
                stacked_gradients = stacked_gradients.view(*new_shape)

            # =========================================================
            # DC Operation Modules Backward (native stacked operations)
            # =========================================================

            elif getattr(module, '_dc_is_reshape', False):
                # Reshape backward: reshape stacked gradients to input shape
                input_shape = cache.input_pos.shape
                new_shape = (stacked_gradients.shape[0],) + input_shape  # [4, batch, ...]
                stacked_gradients = stacked_gradients.view(*new_shape)

            elif getattr(module, '_dc_is_permute', False):
                # Permute backward: apply inverse permutation to stacked gradients
                inv_dims = [0] * len(module.dims)
                for i, d in enumerate(module.dims):
                    inv_dims[d] = i
                
                # Adjust for stack dimension: [0] + [inv_dims + 1]
                adjusted_inv_dims = [0] + [d + 1 for d in inv_dims]
                stacked_gradients = stacked_gradients.permute(*adjusted_inv_dims)

            elif getattr(module, '_dc_is_transpose', False):
                # Transpose backward: same transpose (self-inverse)
                # Apply transpose to stacked gradients: adjust dims for [4, batch, ...] format
                adj_dim0 = module.dim0 + 1 if module.dim0 >= 0 else module.dim0
                adj_dim1 = module.dim1 + 1 if module.dim1 >= 0 else module.dim1
                stacked_gradients = stacked_gradients.transpose(adj_dim0, adj_dim1)

            elif getattr(module, '_dc_is_contiguous', False):
                # Contiguous backward: identity
                pass

            elif getattr(module, '_dc_is_scalar_mul', False):
                # Scalar multiplication backward
                if module.is_negative:
                    # Negative scalar swapped streams in forward: swap [0,1] and [2,3]
                    stacked_gradients = torch.stack([
                        stacked_gradients[1] * module.abs_scalar,  # delta_np -> delta_pp
                        stacked_gradients[0] * module.abs_scalar,  # delta_pp -> delta_np
                        stacked_gradients[3] * module.abs_scalar,  # delta_nn -> delta_pn
                        stacked_gradients[2] * module.abs_scalar   # delta_pn -> delta_nn
                    ], dim=0)
                else:
                    stacked_gradients = stacked_gradients * module.scalar

            elif getattr(module, '_dc_is_scalar_div', False):
                # Scalar division backward
                if module.is_negative:
                    # Negative scalar swapped streams in forward: swap [0,1] and [2,3]
                    stacked_gradients = torch.stack([
                        stacked_gradients[1] / module.abs_scalar,  # delta_np -> delta_pp
                        stacked_gradients[0] / module.abs_scalar,  # delta_pp -> delta_np
                        stacked_gradients[3] / module.abs_scalar,  # delta_nn -> delta_pn
                        stacked_gradients[2] / module.abs_scalar   # delta_pn -> delta_nn
                    ], dim=0)
                else:
                    stacked_gradients = stacked_gradients / module.scalar

            elif getattr(module, '_dc_is_add', False):
                # Addition backward: gradients distribute identically to all inputs
                # For z = x + y: ∂z/∂x = 1, ∂z/∂y = 1
                # Since addition is linear: gradient flows unchanged to first operand
                # Note: Second operand gradient would need separate handling if tracked
                # Current implementation: first operand gets full gradient (correct)
                pass

            elif getattr(module, '_dc_is_slice', False):
                # Slice backward: scatter gradients back
                input_shape = cache.input_pos.shape
                full_shape = (4,) + input_shape  # [4, batch, ...]
                new_stacked = torch.zeros(full_shape, device=stacked_gradients.device, dtype=stacked_gradients.dtype)
                slices = [slice(None)] + [slice(None)] * len(input_shape)  # [slice(None), slice(None), ...]
                slices[module.dim + 1] = slice(module.start, module.end)  # +1 for stack dimension
                new_stacked[tuple(slices)] = stacked_gradients
                stacked_gradients = new_stacked

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
                    full_shape = (4,) + input_shape
                    stacked_gradients = stacked_gradients.expand(full_shape) / n
                else:
                    dims = (module.dim,) if isinstance(module.dim, int) else module.dim
                    n = 1
                    for d in dims:
                        n *= input_shape[d]
                    if not module.keepdim:
                        for d in sorted(dims, reverse=True):  # Insert dims in reverse order
                            stacked_gradients = stacked_gradients.unsqueeze(d + 1)  # +1 for stack dim
                    full_shape = (4,) + input_shape
                    stacked_gradients = stacked_gradients.expand(full_shape) / n

            elif getattr(module, '_dc_is_sum', False):
                # Sum backward: expand
                input_shape = cache.input_pos.shape
                if module.dim is None:
                    full_shape = (4,) + input_shape
                    stacked_gradients = stacked_gradients.expand(full_shape)
                else:
                    dims = (module.dim,) if isinstance(module.dim, int) else module.dim
                    if not module.keepdim:
                        for d in sorted(dims, reverse=True):  # Insert dims in reverse order
                            stacked_gradients = stacked_gradients.unsqueeze(d + 1)  # +1 for stack dim
                    full_shape = (4,) + input_shape
                    stacked_gradients = stacked_gradients.expand(full_shape)

            elif getattr(module, '_dc_is_embedding', False):
                # Embedding backward: scatter gradients back to input indices
                # For now, treat as identity since embeddings typically receive index inputs
                pass

            elif getattr(module, '_dc_is_gather', False):
                # Gather backward: scatter gradients back using same indices
                input_shape = cache.input_pos.shape
                full_shape = (4,) + input_shape
                new_stacked = torch.zeros(full_shape, device=stacked_gradients.device, dtype=stacked_gradients.dtype)
                # This would need the index tensor to properly scatter - for now approximate as identity
                # TODO: Implement proper scatter when index is available
                pass

            # Apply α-shifting if enabled (to stacked gradients)
            backward_mode = getattr(module, DC_BACKWARD_MODE, BackwardMode.ALPHA)
            if backward_mode == BackwardMode.ALPHA:
                alpha = getattr(module, DC_ALPHA, 0.35)
                # α-shifting strategy: preserves invariant δ_pp - δ_np - δ_pn + δ_nn = δ
                # Unstack for shifting
                delta_pp, delta_np, delta_pn, delta_nn = stacked_gradients[0], stacked_gradients[1], stacked_gradients[2], stacked_gradients[3]
                
                shift_p = alpha * (delta_pp + delta_np)
                shift_n = alpha * (delta_pn + delta_nn)
                
                delta_pp = delta_pp - shift_p
                delta_np = delta_np - shift_p
                delta_pn = delta_pn - shift_n
                delta_nn = delta_nn - shift_n
                
                # Restack after shifting
                stacked_gradients = torch.stack([delta_pp, delta_np, delta_pn, delta_nn], dim=0)

            # Cache stacked gradients for this layer
            cache.stacked_gradients = stacked_gradients.detach().clone()
            
            # Also cache individual components for compatibility (unstack only for caching)
            delta_pp, delta_np, delta_pn, delta_nn = stacked_gradients[0], stacked_gradients[1], stacked_gradients[2], stacked_gradients[3]
            cache.delta_pp = delta_pp.detach().clone()
            cache.delta_np = delta_np.detach().clone()
            cache.delta_pn = delta_pn.detach().clone()
            cache.delta_nn = delta_nn.detach().clone()


    # =========================================================================
    # Stacked Tensor Backward Methods (ONLY APPROACH) 
    # =========================================================================

    def _backward_linear_stacked(self, module: nn.Linear, stacked_gradients: Tensor) -> Tensor:
        """Backward through Linear layer using stacked gradients [4, batch, ...]."""
        # Unstack: [0]=delta_pp, [1]=delta_np, [2]=delta_pn, [3]=delta_nn
        delta_pp, delta_np, delta_pn, delta_nn = stacked_gradients[0], stacked_gradients[1], stacked_gradients[2], stacked_gradients[3]
        
        # Create masked views of weights (no copying!)
        pos_mask = module.weight >= 0
        neg_mask = module.weight < 0
        W_pos = module.weight * pos_mask
        W_neg = -module.weight * neg_mask

        # Compute backward pass
        new_delta_pp = F.linear(delta_pp, W_pos.t()) + F.linear(delta_pn, W_neg.t())
        new_delta_np = F.linear(delta_pp, W_neg.t()) + F.linear(delta_pn, W_pos.t())
        new_delta_pn = F.linear(delta_np, W_pos.t()) + F.linear(delta_nn, W_neg.t())
        new_delta_nn = F.linear(delta_np, W_neg.t()) + F.linear(delta_nn, W_pos.t())

        # Restack
        return torch.stack([new_delta_pp, new_delta_np, new_delta_pn, new_delta_nn], dim=0)

    def _backward_conv2d_stacked(self, module: nn.Conv2d, stacked_gradients: Tensor) -> Tensor:
        """Backward through Conv2d layer using stacked gradients."""
        delta_pp, delta_np, delta_pn, delta_nn = stacked_gradients[0], stacked_gradients[1], stacked_gradients[2], stacked_gradients[3]
        
        # Create masked views of weights
        pos_mask = module.weight >= 0
        neg_mask = module.weight < 0
        W_pos = module.weight * pos_mask
        W_neg = -module.weight * neg_mask

        # Backward convolution
        new_delta_pp = F.conv_transpose2d(delta_pp, W_pos, None, module.stride, module.padding, 0, module.groups, module.dilation)
        new_delta_pp += F.conv_transpose2d(delta_pn, W_neg, None, module.stride, module.padding, 0, module.groups, module.dilation)
        
        new_delta_np = F.conv_transpose2d(delta_pp, W_neg, None, module.stride, module.padding, 0, module.groups, module.dilation)
        new_delta_np += F.conv_transpose2d(delta_pn, W_pos, None, module.stride, module.padding, 0, module.groups, module.dilation)
        
        new_delta_pn = F.conv_transpose2d(delta_np, W_pos, None, module.stride, module.padding, 0, module.groups, module.dilation)
        new_delta_pn += F.conv_transpose2d(delta_nn, W_neg, None, module.stride, module.padding, 0, module.groups, module.dilation)
        
        new_delta_nn = F.conv_transpose2d(delta_np, W_neg, None, module.stride, module.padding, 0, module.groups, module.dilation)
        new_delta_nn += F.conv_transpose2d(delta_nn, W_pos, None, module.stride, module.padding, 0, module.groups, module.dilation)

        return torch.stack([new_delta_pp, new_delta_np, new_delta_pn, new_delta_nn], dim=0)

    def _backward_relu_stacked(
        self, module: nn.ReLU, cache: DCCache, stacked_gradients: Tensor
    ) -> Tensor:
        """Backward through ReLU using stacked gradients. Reads config from module."""
        delta_pp, delta_np, delta_pn, delta_nn = stacked_gradients[0], stacked_gradients[1], stacked_gradients[2], stacked_gradients[3]

        if not hasattr(cache, 'z_before') or cache.z_before is None:
            # Fallback: pass through unchanged if no cached z_before
            return stacked_gradients

        # Compute masks from cached z_before
        mp = (cache.z_before >= 0).float()  # pos >= neg
        mn = (cache.z_before < 0).float()   # pos < neg

        # Read config from module
        relu_mode = getattr(module, DC_RELU_MODE, ReLUMode.MAX)
        split_mode = relu_mode.value if isinstance(relu_mode, ReLUMode) else relu_mode
        backprop_mode = getattr(module, DC_BACKPROP_MODE, 'standard')

        new_pp, new_np, new_pn, new_nn = backward_relu(
            delta_pp, delta_np, delta_pn, delta_nn, mp, mn, split_mode, backprop_mode)

        return torch.stack([new_pp, new_np, new_pn, new_nn], dim=0)

    def _backward_batchnorm_stacked(self, module, stacked_gradients: Tensor) -> Tensor:
        """Backward through BatchNorm using stacked gradients."""
        # For simplicity, treat BatchNorm as identity in backward pass (variance constant)
        # This is an approximation - full implementation would need running stats
        return stacked_gradients

    def _backward_maxpool_stacked(self, cache: DCCache, stacked_gradients: Tensor) -> Tensor:
        """Backward through MaxPool using stacked gradients [4, batch, C, H_out, W_out]."""
        if not hasattr(cache, 'input_pos') or cache.input_pos is None:
            return stacked_gradients

        input_shape = cache.input_pos.shape  # [batch, C, H_in, W_in]
        batch, channels, h_in, w_in = input_shape

        # Get pool indices from cache (if available)
        if hasattr(cache, 'pool_indices') and cache.pool_indices is not None:
            indices = cache.pool_indices  # [batch, C, H_out, W_out]
            h_out, w_out = indices.shape[2], indices.shape[3]

            # Process each of the 4 gradient components
            result_components = []
            for i in range(4):
                grad_component = stacked_gradients[i]  # [batch, C, H_out, W_out]

                # Create output tensor filled with zeros
                output = torch.zeros(batch, channels, h_in * w_in,
                                   device=grad_component.device, dtype=grad_component.dtype)

                # Flatten for scatter operation
                indices_flat = indices.view(batch, channels, -1)
                grad_flat = grad_component.view(batch, channels, -1)

                # Scatter gradients to winner positions
                output.scatter_(2, indices_flat, grad_flat)

                # Reshape to spatial dimensions
                output = output.view(batch, channels, h_in, w_in)
                result_components.append(output)

            return torch.stack(result_components, dim=0)
        else:
            # Fallback: use nearest neighbor upsampling if no indices cached
            h_out, w_out = stacked_gradients.shape[3], stacked_gradients.shape[4]
            result_components = []
            for i in range(4):
                grad_component = stacked_gradients[i]
                upsampled = F.interpolate(grad_component, size=(h_in, w_in), mode='nearest')
                result_components.append(upsampled)
            return torch.stack(result_components, dim=0)

    def _backward_avgpool_stacked(self, cache: DCCache, module: nn.Module, stacked_gradients: Tensor) -> Tensor:
        """Backward through AvgPool using stacked gradients [4, batch, C, H, W]."""
        if not hasattr(cache, 'input_pos') or cache.input_pos is None:
            return stacked_gradients

        input_size = cache.input_pos.shape[2:]  # [H_in, W_in]

        # Process each of the 4 gradient components separately
        # stacked_gradients: [4, batch, channels, H_out, W_out]
        result_components = []
        for i in range(4):
            grad_component = stacked_gradients[i]  # [batch, channels, H_out, W_out]
            upsampled = F.interpolate(grad_component, size=input_size, mode='nearest')
            result_components.append(upsampled)

        # Stack back together: [4, batch, channels, H_in, W_in]
        result = torch.stack(result_components, dim=0)

        # For AvgPool, divide by kernel area
        if isinstance(module, nn.AvgPool2d):
            ks = module.kernel_size
            kh = ks if isinstance(ks, int) else ks[0]
            kw = ks if isinstance(ks, int) else ks[1]
            result = result / (kh * kw)
        elif isinstance(module, nn.AdaptiveAvgPool2d):
            # Compute effective kernel size
            h_in, w_in = input_size
            out_size = module.output_size
            oh = out_size if isinstance(out_size, int) else out_size[0]
            ow = out_size if isinstance(out_size, int) else out_size[1]
            kh = h_in // oh
            kw = w_in // ow
            result = result / (kh * kw)

        return result

    def _backward_softmax_stacked(self, module: nn.Softmax, cache: DCCache, stacked_gradients: Tensor) -> Tensor:
        """Backward through Softmax using stacked gradients."""
        # Softmax backward is complex - approximate as identity for now
        # Full implementation would need Jacobian computation
        return stacked_gradients

    def _backward_layernorm_stacked(self, module: nn.LayerNorm, cache: DCCache, stacked_gradients: Tensor) -> Tensor:
        """Backward through LayerNorm using stacked gradients."""
        # LayerNorm backward is complex - approximate as identity for now
        # Full implementation would need Jacobian computation
        return stacked_gradients
    
    def _backward_dc_matmul_stacked(self, module, cache: DCCache, stacked_gradients: Tensor) -> Tensor:
        """Backward through DC MatMul using stacked gradients."""
        # DCMatMul backward is like linear layer
        W = module.weight
        
        # Create masked views
        pos_mask = W >= 0
        neg_mask = W < 0
        W_pos = W * pos_mask
        W_neg = -W * neg_mask
        
        delta_pp, delta_np, delta_pn, delta_nn = stacked_gradients[0], stacked_gradients[1], stacked_gradients[2], stacked_gradients[3]
        
        # Backward through DC linear layer
        new_delta_pp = F.linear(delta_pp, W_pos.t()) + F.linear(delta_pn, W_neg.t())
        new_delta_np = F.linear(delta_pp, W_neg.t()) + F.linear(delta_pn, W_pos.t())
        new_delta_pn = F.linear(delta_np, W_pos.t()) + F.linear(delta_nn, W_neg.t())
        new_delta_nn = F.linear(delta_np, W_neg.t()) + F.linear(delta_nn, W_pos.t())
        
        return torch.stack([new_delta_pp, new_delta_np, new_delta_pn, new_delta_nn], dim=0)

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

    def get_stacked_gradients(self, layer_name: Optional[str] = None) -> Optional[Tensor]:
        """
        Get stacked gradients [delta_pp, delta_np, delta_pn, delta_nn] for specified layer.
        
        Returns stacked tensor of shape [4, batch, ...] for PyTorch autograd compatibility.
        This enables automatic gradient accumulation at residual connection split points.
        """
        if layer_name is None:
            layer_name = self.layer_order[0] if self.layer_order else None

        if layer_name is None or layer_name not in self.caches:
            return None

        cache = self.caches[layer_name]
        if hasattr(cache, 'stacked_gradients'):
            return cache.stacked_gradients
        
        # Fallback: construct from individual sensitivities
        sens = self.get_sensitivities(layer_name)
        if sens is None:
            return None
        delta_pp, delta_np, delta_pn, delta_nn = sens
        return torch.stack([delta_pp, delta_np, delta_pn, delta_nn], dim=0)

    def verify_reconstruction(self, tolerance: float = 1e-5) -> Dict[str, float]:
        """Verify pos - neg = original for all layers."""
        errors = {}
        for name, cache in self.caches.items():
            if cache.output_pos is not None and cache.original_output is not None:
                reconstructed = cache.output_pos - cache.output_neg
                error = (reconstructed - cache.original_output).abs().max().item()
                errors[name] = error
        return errors

    def find_output_layer_name(self) -> Optional[str]:
        """
        Find the actual output layer name (typically a Linear or Conv layer).

        Skips _dc_* modules added by functional_replacer since those are
        auxiliary modules (like ReLU replacements), not the actual output.

        Returns:
            Name of the output layer, or None if not found.
        """
        output_name = None
        for name in self.layer_order:
            # Skip _dc_* modules added by functional_replacer
            if name.startswith('_dc_') or '._dc_' in name:
                continue
            module = self.modules.get(name)
            if module is not None and isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                output_name = name
        return output_name

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
            for attr in [DC_INPUT_MODE, DC_BETA, DC_BACKWARD_MODE, DC_ALPHA, DC_RELU_MODE, 
                         DC_ENABLED, DC_CACHE_ACTIVATIONS, DC_WEIGHT_POS, DC_WEIGHT_NEG, 
                         DC_BIAS_POS, DC_BIAS_NEG, DC_BN_SCALE_POS, DC_BN_SCALE_NEG, 
                         DC_SOFTMAX_DIM, DC_LN_SCALE_POS, DC_LN_SCALE_NEG, DC_LN_NORMALIZED_SHAPE]:
                if hasattr(module, attr):
                    delattr(module, attr)

    def __del__(self):
        try:
            # Only remove hooks, not attributes (to avoid race conditions)
            for handle in self._forward_handles:
                handle.remove()
        except Exception:
            pass
