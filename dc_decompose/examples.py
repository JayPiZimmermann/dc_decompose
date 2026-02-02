"""
Examples demonstrating DC decomposition with PyTorch hooks.

This module shows how to use the DC decomposition for:
1. Simple MLPs
2. CNNs (convolutional networks)
3. ResNets (residual networks)
"""

import torch
import torch.nn as nn
from torch import Tensor

from .hook_decomposer import HookDecomposer, ReLUMode, ShiftMode
from .dc_matmul import DCMatMul, DCMatMulFunction


# =============================================================================
# Example Models
# =============================================================================

class SimpleMLP(nn.Module):
    """Simple MLP for demonstration."""

    def __init__(self, input_dim: int = 784, hidden_dim: int = 256, output_dim: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


class SimpleCNN(nn.Module):
    """Simple CNN for demonstration."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x


class BasicBlock(nn.Module):
    """Basic residual block for ResNet."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        out = self.relu2(out)
        return out


class MLPWithSoftmax(nn.Module):
    """MLP with intermediate softmax layers (not just at the end)."""

    def __init__(self, input_dim: int = 100, hidden_dim: int = 64, output_dim: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.softmax1 = nn.Softmax(dim=-1)  # Softmax after first layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.softmax2 = nn.Softmax(dim=-1)  # Another intermediate softmax
        self.fc4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.softmax1(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.softmax2(x)
        x = self.fc4(x)
        return x


class SequentialSoftmaxMLP(nn.Module):
    """Sequential MLP with multiple softmax layers (for testing DC decomposition)."""

    def __init__(self, input_dim: int = 64, hidden_dim: int = 32, output_dim: int = 10):
        super().__init__()
        # Sequential structure compatible with hook-based decomposition
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.softmax1 = nn.Softmax(dim=-1)  # First intermediate softmax
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.softmax2 = nn.Softmax(dim=-1)  # Second intermediate softmax
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.softmax3 = nn.Softmax(dim=-1)  # Third intermediate softmax
        self.fc5 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.softmax1(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.softmax2(x)
        x = self.fc4(x)
        x = self.softmax3(x)
        x = self.fc5(x)
        return x


class MultiSoftmaxCNN(nn.Module):
    """CNN with multiple softmax layers at different stages."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        # Softmax over channel dimension (like channel attention)
        self.channel_softmax = nn.Softmax(dim=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.softmax_hidden = nn.Softmax(dim=-1)  # Intermediate softmax
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.channel_softmax(x)  # Channel-wise softmax
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.softmax_hidden(x)  # Hidden layer softmax
        x = self.fc2(x)
        return x


class SimpleResNet(nn.Module):
    """Simple ResNet for demonstration."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        self.layer1 = BasicBlock(64, 64)
        self.layer2 = BasicBlock(64, 128, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x


# =============================================================================
# Examples
# =============================================================================

def example_mlp():
    """Example: DC decomposition of a simple MLP."""
    print("=" * 60)
    print("Example: MLP DC Decomposition")
    print("=" * 60)

    # Create model
    model = SimpleMLP(input_dim=100, hidden_dim=64, output_dim=10)
    model.eval()

    # Attach decomposer via hooks (uses CENTER shift mode by default)
    decomposer = HookDecomposer(model, relu_mode=ReLUMode.MAX)

    # Create input
    x = torch.randn(2, 100)

    # Initialize and run forward pass
    decomposer.initialize()
    output = model(x)

    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # Access decomposed activations
    print("\nLayer Activations (pos, neg):")
    for name in decomposer.layer_order:
        pos, neg = decomposer.get_activation(name)
        if pos is not None:
            recon_error = (pos - neg - decomposer.caches[name].original_output).abs().max().item()
            print(f"  {name}:")
            print(f"    pos norm: {pos.norm():.4f}, neg norm: {neg.norm():.4f}")
            print(f"    reconstruction error: {recon_error:.2e}")

    # Backward pass - compute 4 sensitivities
    decomposer.backward()

    print("\nLayer Sensitivities (4 deltas):")
    for name in decomposer.layer_order:
        sens = decomposer.get_sensitivity(name)
        if sens is not None:
            delta_pp, delta_np, delta_pn, delta_nn = sens
            print(f"  {name}:")
            print(f"    delta_pp: {delta_pp.norm():.4f}, delta_np: {delta_np.norm():.4f}")
            print(f"    delta_pn: {delta_pn.norm():.4f}, delta_nn: {delta_nn.norm():.4f}")

    # Verify reconstruction
    errors = decomposer.verify_reconstruction()
    max_error = max(errors.values()) if errors else 0
    print(f"\nMax reconstruction error: {max_error:.2e}")

    # Cleanup
    decomposer.remove_hooks()

    return decomposer


def example_cnn():
    """Example: DC decomposition of a CNN."""
    print("\n" + "=" * 60)
    print("Example: CNN DC Decomposition")
    print("=" * 60)

    # Create CNN
    model = SimpleCNN(num_classes=10)
    model.eval()

    # Attach decomposer (uses CENTER shift mode by default)
    decomposer = HookDecomposer(model, relu_mode=ReLUMode.MAX)

    # Create input
    x = torch.randn(2, 1, 28, 28)

    # Forward
    decomposer.initialize()
    output = model(x)

    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Decomposed layers: {len(decomposer.layer_order)}")

    # Show layer activations
    print("\nLayer activations:")
    for name in decomposer.layer_order:
        pos, neg = decomposer.get_activation(name)
        if pos is not None:
            print(f"  {name}: pos={tuple(pos.shape)}, neg={tuple(neg.shape)}")

    # Backward
    decomposer.backward()

    print("\nLayer sensitivities:")
    for name in decomposer.layer_order:
        sens = decomposer.get_sensitivity(name)
        if sens is not None:
            delta_pp, _, _, _ = sens
            print(f"  {name}: delta_pp shape={tuple(delta_pp.shape)}")

    # Verify
    errors = decomposer.verify_reconstruction()
    max_error = max(errors.values()) if errors else 0
    print(f"\nMax reconstruction error: {max_error:.2e}")

    decomposer.remove_hooks()
    return decomposer


def example_shift_modes():
    """Example: Different shift modes for input initialization."""
    print("\n" + "=" * 60)
    print("Example: Shift Modes")
    print("=" * 60)

    model = SimpleMLP(input_dim=10, hidden_dim=8, output_dim=5)
    model.eval()

    x = torch.randn(1, 10)
    print(f"Input x[0,:5]: {x[0, :5].tolist()}")

    # Test each shift mode
    for mode in [ShiftMode.CENTER, ShiftMode.POSITIVE, ShiftMode.NEGATIVE, ShiftMode.BETA]:
        beta = 0.7 if mode == ShiftMode.BETA else 0.5
        decomposer = HookDecomposer(model, shift_mode=mode, beta=beta)
        decomposer.initialize()
        output = model(x)

        # Get first layer's input
        first_layer = decomposer.layer_order[0]
        cache = decomposer.caches[first_layer]

        print(f"\n{mode.value.upper()} mode{' (beta=0.7)' if mode == ShiftMode.BETA else ''}:")
        print(f"  input_pos[0,:5]: {cache.input_pos[0, :5].tolist()}")
        print(f"  input_neg[0,:5]: {cache.input_neg[0, :5].tolist()}")
        print(f"  pos - neg = x:   {(cache.input_pos - cache.input_neg)[0, :5].tolist()}")

        # Verify non-negativity for CENTER mode
        if mode == ShiftMode.CENTER:
            pos_nonneg = (cache.input_pos >= 0).all().item()
            neg_nonneg = (cache.input_neg >= 0).all().item()
            print(f"  pos non-negative: {pos_nonneg}, neg non-negative: {neg_nonneg}")

        decomposer.remove_hooks()


def example_relu_modes():
    """Example: Different ReLU decomposition modes."""
    print("\n" + "=" * 60)
    print("Example: ReLU Decomposition Modes")
    print("=" * 60)

    model = nn.Sequential(
        nn.Linear(10, 8),
        nn.ReLU(),
        nn.Linear(8, 5),
    )
    model.eval()

    x = torch.randn(1, 10)
    original_output = model(x)

    for mode in [ReLUMode.MAX, ReLUMode.MIN, ReLUMode.HALF]:
        decomposer = HookDecomposer(model, relu_mode=mode)  # Uses CENTER shift mode
        decomposer.initialize()
        output = model(x)

        # Get ReLU layer output
        relu_name = [n for n in decomposer.layer_order if 'relu' in n.lower() or n == '1'][0]
        pos, neg = decomposer.get_activation(relu_name)

        print(f"\n{mode.value.upper()} mode:")
        print(f"  pos norm: {pos.norm():.4f}")
        print(f"  neg norm: {neg.norm():.4f}")
        print(f"  output matches original: {torch.allclose(output, original_output)}")

        decomposer.remove_hooks()


def example_runtime_configuration():
    """Example: Changing configuration at runtime."""
    print("\n" + "=" * 60)
    print("Example: Runtime Configuration")
    print("=" * 60)

    model = SimpleMLP(input_dim=10, hidden_dim=8, output_dim=5)
    model.eval()

    # Start with CENTER shift mode (default)
    decomposer = HookDecomposer(model, shift_mode=ShiftMode.CENTER)

    x = torch.randn(1, 10)

    # Run with CENTER mode
    decomposer.initialize()
    model(x)
    cache = decomposer.caches[decomposer.layer_order[0]]
    pos_nonneg = (cache.input_pos >= 0).all().item()
    print(f"With CENTER mode: input_pos non-negative = {pos_nonneg}")

    # Change shift mode at runtime to BETA
    decomposer.set_shift_mode(ShiftMode.BETA)
    decomposer.set_beta(0.5)
    decomposer.initialize()
    model(x)
    print(f"With BETA(0.5) mode: first layer input_neg norm = {decomposer.caches[decomposer.layer_order[0]].input_neg.norm():.4f}")

    # Change beta at runtime
    decomposer.set_beta(0.8)
    decomposer.initialize()
    model(x)
    print(f"With BETA(0.8) mode: first layer input_neg norm = {decomposer.caches[decomposer.layer_order[0]].input_neg.norm():.4f}")

    # Disable a layer
    decomposer.enable_layer("fc2", False)
    decomposer.initialize()
    model(x)
    print(f"With fc2 disabled: fc2 has cached output = {decomposer.caches['fc2'].output_pos is not None}")

    decomposer.remove_hooks()


def example_softmax_mlp():
    """Example: MLP with intermediate softmax layers."""
    print("\n" + "=" * 60)
    print("Example: MLP with Intermediate Softmax")
    print("=" * 60)

    model = MLPWithSoftmax(input_dim=50, hidden_dim=32, output_dim=10)
    model.eval()

    decomposer = HookDecomposer(model)

    x = torch.randn(2, 50)

    decomposer.initialize()
    output = model(x)

    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # Show layer activations with focus on softmax layers
    print("\nLayer activations:")
    for name in decomposer.layer_order:
        pos, neg = decomposer.get_activation(name)
        if pos is not None:
            is_softmax = 'softmax' in name.lower()
            marker = " [SOFTMAX]" if is_softmax else ""
            print(f"  {name}{marker}:")
            print(f"    pos norm: {pos.norm():.4f}, neg norm: {neg.norm():.4f}")
            if is_softmax:
                # Verify softmax output: should sum to 1 along last dim
                sum_check = pos.sum(dim=-1)
                print(f"    softmax sum (should be ~1): {sum_check[0].item():.6f}")

    # Backward pass
    decomposer.backward()

    print("\nLayer sensitivities (softmax layers):")
    for name in decomposer.layer_order:
        if 'softmax' in name.lower():
            sens = decomposer.get_sensitivity(name)
            if sens is not None:
                delta_pp, delta_np, delta_pn, delta_nn = sens
                print(f"  {name}:")
                print(f"    delta_pp: {delta_pp.norm():.4f}, delta_np: {delta_np.norm():.4f}")
                print(f"    delta_pn: {delta_pn.norm():.4f}, delta_nn: {delta_nn.norm():.4f}")

    # Verify reconstruction
    errors = decomposer.verify_reconstruction()
    max_error = max(errors.values()) if errors else 0
    print(f"\nMax reconstruction error: {max_error:.2e}")

    decomposer.remove_hooks()
    return decomposer


def example_multi_softmax_mlp():
    """Example: Sequential MLP with multiple softmax layers."""
    print("\n" + "=" * 60)
    print("Example: Sequential MLP with Multiple Softmax Layers")
    print("=" * 60)

    model = SequentialSoftmaxMLP(input_dim=64, hidden_dim=32, output_dim=10)
    model.eval()

    decomposer = HookDecomposer(model)

    x = torch.randn(2, 64)

    decomposer.initialize()
    output = model(x)

    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    print("\nLayer activations:")
    for name in decomposer.layer_order:
        pos, neg = decomposer.get_activation(name)
        if pos is not None:
            is_softmax = 'softmax' in name.lower()
            marker = " [SOFTMAX]" if is_softmax else ""
            recon_error = (pos - neg - decomposer.caches[name].original_output).abs().max().item()
            print(f"  {name}{marker}: recon_error={recon_error:.2e}")
            if is_softmax:
                # Verify softmax sums to 1
                sum_check = pos.sum(dim=-1).mean().item()
                print(f"    softmax sum (should be ~1): {sum_check:.6f}")

    decomposer.backward()

    print("\nAll layer sensitivities:")
    for name in decomposer.layer_order:
        sens = decomposer.get_sensitivity(name)
        if sens is not None:
            delta_pp, delta_np, delta_pn, delta_nn = sens
            is_softmax = 'softmax' in name.lower()
            marker = " [SOFTMAX]" if is_softmax else ""
            print(f"  {name}{marker}:")
            print(f"    delta_pp: {delta_pp.norm():.4f}, delta_np: {delta_np.norm():.4f}")
            print(f"    delta_pn: {delta_pn.norm():.4f}, delta_nn: {delta_nn.norm():.4f}")

    errors = decomposer.verify_reconstruction()
    max_error = max(errors.values()) if errors else 0
    print(f"\nMax reconstruction error: {max_error:.2e}")

    decomposer.remove_hooks()
    return decomposer


def example_softmax_cnn():
    """Example: CNN with multiple softmax layers."""
    print("\n" + "=" * 60)
    print("Example: CNN with Multiple Softmax Layers")
    print("=" * 60)

    model = MultiSoftmaxCNN(num_classes=10)
    model.eval()

    decomposer = HookDecomposer(model)

    x = torch.randn(2, 1, 28, 28)

    decomposer.initialize()
    output = model(x)

    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Decomposed layers: {len(decomposer.layer_order)}")

    print("\nLayer activations:")
    for name in decomposer.layer_order:
        pos, neg = decomposer.get_activation(name)
        if pos is not None:
            is_softmax = 'softmax' in name.lower()
            marker = " [SOFTMAX]" if is_softmax else ""
            print(f"  {name}{marker}: pos={tuple(pos.shape)}, neg={tuple(neg.shape)}")

    decomposer.backward()

    print("\nSoftmax layer sensitivities:")
    for name in decomposer.layer_order:
        if 'softmax' in name.lower():
            sens = decomposer.get_sensitivity(name)
            if sens is not None:
                delta_pp, _, _, _ = sens
                print(f"  {name}: delta_pp shape={tuple(delta_pp.shape)}")

    errors = decomposer.verify_reconstruction()
    max_error = max(errors.values()) if errors else 0
    print(f"\nMax reconstruction error: {max_error:.2e}")

    decomposer.remove_hooks()
    return decomposer


class MLPWithLayerNorm(nn.Module):
    """MLP with LayerNorm layers (like transformer blocks)."""

    def __init__(self, input_dim: int = 64, hidden_dim: int = 128, output_dim: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.ln2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


class MatMulMLP(nn.Module):
    """MLP using DCMatMul for matrix operations."""

    def __init__(self, input_dim: int = 64, hidden_dim: int = 32, output_dim: int = 10):
        super().__init__()
        self.matmul1 = DCMatMul(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.matmul2 = DCMatMul(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.matmul3 = DCMatMul(hidden_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        # Setup weight decomposition for DC
        self.matmul1._setup_weight_decomposition()
        self.matmul2._setup_weight_decomposition()
        self.matmul3._setup_weight_decomposition()

        x = self.matmul1(x)
        x = self.relu1(x)
        x = self.matmul2(x)
        x = self.relu2(x)
        x = self.matmul3(x)
        return x


class SimpleTransformerBlock(nn.Module):
    """
    Simplified transformer block with LayerNorm (sequential structure).

    Note: This is a simplified version that uses purely sequential operations,
    which works well with hook-based DC decomposition. For full attention
    mechanisms with matrix multiplications, see the ViT example.
    """

    def __init__(self, dim: int = 64):
        super().__init__()
        self.dim = dim

        # Pre-norm transformer block (sequential structure)
        self.ln1 = nn.LayerNorm(dim)
        self.attn_fc = nn.Linear(dim, dim)  # Simplified "attention" as single linear

        self.ln2 = nn.LayerNorm(dim)
        self.mlp_fc1 = nn.Linear(dim, 4 * dim)
        self.relu = nn.ReLU()
        self.mlp_fc2 = nn.Linear(4 * dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        # Attention block (simplified)
        x = self.ln1(x)
        x = self.attn_fc(x)

        # MLP block
        x = self.ln2(x)
        x = self.mlp_fc1(x)
        x = self.relu(x)
        x = self.mlp_fc2(x)

        return x


def example_layernorm():
    """Example: MLP with LayerNorm layers."""
    print("\n" + "=" * 60)
    print("Example: MLP with LayerNorm")
    print("=" * 60)

    model = MLPWithLayerNorm(input_dim=64, hidden_dim=128, output_dim=10)
    model.eval()

    decomposer = HookDecomposer(model)

    x = torch.randn(2, 64)

    decomposer.initialize()
    output = model(x)

    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    print("\nLayer activations:")
    for name in decomposer.layer_order:
        pos, neg = decomposer.get_activation(name)
        if pos is not None:
            is_ln = 'ln' in name.lower()
            marker = " [LAYERNORM]" if is_ln else ""
            recon_error = (pos - neg - decomposer.caches[name].original_output).abs().max().item()
            print(f"  {name}{marker}: recon_error={recon_error:.2e}")

    decomposer.backward()

    print("\nLayerNorm sensitivities:")
    for name in decomposer.layer_order:
        if 'ln' in name.lower():
            sens = decomposer.get_sensitivity(name)
            if sens is not None:
                delta_pp, delta_np, delta_pn, delta_nn = sens
                print(f"  {name}:")
                print(f"    delta_pp: {delta_pp.norm():.4f}, delta_np: {delta_np.norm():.4f}")
                print(f"    delta_pn: {delta_pn.norm():.4f}, delta_nn: {delta_nn.norm():.4f}")

    errors = decomposer.verify_reconstruction()
    max_error = max(errors.values()) if errors else 0
    print(f"\nMax reconstruction error: {max_error:.2e}")

    decomposer.remove_hooks()
    return decomposer


def example_matmul():
    """Example: MLP using DCMatMul."""
    print("\n" + "=" * 60)
    print("Example: MLP with DCMatMul")
    print("=" * 60)

    model = MatMulMLP(input_dim=64, hidden_dim=32, output_dim=10)
    model.eval()

    decomposer = HookDecomposer(model)

    x = torch.randn(2, 64)

    decomposer.initialize()
    output = model(x)

    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    print("\nLayer activations:")
    for name in decomposer.layer_order:
        pos, neg = decomposer.get_activation(name)
        if pos is not None:
            is_matmul = 'matmul' in name.lower()
            marker = " [MATMUL]" if is_matmul else ""
            recon_error = (pos - neg - decomposer.caches[name].original_output).abs().max().item()
            print(f"  {name}{marker}: recon_error={recon_error:.2e}")

    decomposer.backward()

    print("\nMatMul sensitivities:")
    for name in decomposer.layer_order:
        if 'matmul' in name.lower():
            sens = decomposer.get_sensitivity(name)
            if sens is not None:
                delta_pp, delta_np, delta_pn, delta_nn = sens
                print(f"  {name}:")
                print(f"    delta_pp: {delta_pp.norm():.4f}, delta_np: {delta_np.norm():.4f}")
                print(f"    delta_pn: {delta_pn.norm():.4f}, delta_nn: {delta_nn.norm():.4f}")

    errors = decomposer.verify_reconstruction()
    max_error = max(errors.values()) if errors else 0
    print(f"\nMax reconstruction error: {max_error:.2e}")

    decomposer.remove_hooks()
    return decomposer


def example_dc_matmul_functional():
    """Example: Direct usage of DCMatMulFunction."""
    print("\n" + "=" * 60)
    print("Example: DCMatMulFunction (Functional API)")
    print("=" * 60)

    # Create test tensors
    A = torch.randn(2, 4, 8)  # batch=2, seq=4, dim=8
    B = torch.randn(8, 16)    # (8, 16) weight matrix

    # Decompose into pos/neg
    A_pos = torch.relu(A)
    A_neg = torch.relu(-A)
    B_pos = torch.relu(B)
    B_neg = torch.relu(-B)

    # Verify A = A_pos - A_neg
    A_reconstructed = A_pos - A_neg
    print(f"A reconstruction error: {(A - A_reconstructed).abs().max():.2e}")

    # Forward pass
    C_pos, C_neg = DCMatMulFunction.forward(A_pos, A_neg, B_pos, B_neg)

    # Verify C = C_pos - C_neg = A @ B
    C_original = torch.matmul(A, B)
    C_reconstructed = C_pos - C_neg
    print(f"C reconstruction error: {(C_original - C_reconstructed).abs().max():.2e}")

    # Backward pass
    delta_pp = torch.ones_like(C_pos)
    delta_np = torch.zeros_like(C_pos)
    delta_pn = torch.zeros_like(C_neg)
    delta_nn = torch.zeros_like(C_neg)

    new_delta_pp, new_delta_np, new_delta_pn, new_delta_nn = DCMatMulFunction.backward(
        delta_pp, delta_np, delta_pn, delta_nn, B_pos, B_neg
    )

    print(f"Backward delta_pp shape: {new_delta_pp.shape}")
    print(f"Backward delta norms: pp={new_delta_pp.norm():.4f}, np={new_delta_np.norm():.4f}")


def example_transformer_block():
    """Example: Simplified transformer block with LayerNorm."""
    print("\n" + "=" * 60)
    print("Example: Simplified Transformer Block")
    print("=" * 60)

    model = SimpleTransformerBlock(dim=64)
    model.eval()

    decomposer = HookDecomposer(model)

    # Sequence input: (batch, seq_len, dim)
    x = torch.randn(2, 8, 64)

    decomposer.initialize()
    output = model(x)

    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Decomposed layers: {len(decomposer.layer_order)}")

    print("\nLayer activations:")
    for name in decomposer.layer_order:
        pos, neg = decomposer.get_activation(name)
        if pos is not None:
            layer_type = ""
            if 'ln' in name.lower():
                layer_type = " [LAYERNORM]"
            elif 'relu' in name.lower():
                layer_type = " [RELU]"
            print(f"  {name}{layer_type}: pos={tuple(pos.shape)}")

    decomposer.backward()

    errors = decomposer.verify_reconstruction()
    max_error = max(errors.values()) if errors else 0
    print(f"\nMax reconstruction error: {max_error:.2e}")

    decomposer.remove_hooks()
    return decomposer


def run_all_examples():
    """Run all examples."""
    example_mlp()
    example_cnn()
    example_shift_modes()
    example_relu_modes()
    example_runtime_configuration()
    example_softmax_mlp()
    example_multi_softmax_mlp()
    example_softmax_cnn()
    example_layernorm()
    example_matmul()
    example_dc_matmul_functional()
    example_transformer_block()


if __name__ == "__main__":
    run_all_examples()
