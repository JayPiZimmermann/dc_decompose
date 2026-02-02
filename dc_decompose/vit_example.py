"""
Example: DC Decomposition of ViT (Vision Transformer)

This example demonstrates how to apply DC decomposition to a
HuggingFace ViT model with ReLU activations.

Requirements:
    pip install transformers pillow

Note: This example modifies the ViT config to use ReLU instead of GELU,
as DC decomposition requires ReLU for proper splitting.
"""

import torch
import torch.nn as nn

try:
    from transformers import ViTConfig, ViTModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("transformers not installed. Install with: pip install transformers")

from .hook_decomposer import HookDecomposer, ShiftMode
from .dc_matmul import DCMatMul, DCMatMulFunction


def create_vit_with_relu(model_name: str = "google/vit-base-patch16-224"):
    """
    Create a ViT model with ReLU activations instead of GELU.

    Args:
        model_name: HuggingFace model name

    Returns:
        ViT model with ReLU activations
    """
    if not HAS_TRANSFORMERS:
        raise ImportError("transformers package required. Install with: pip install transformers")

    config = ViTConfig.from_pretrained(model_name)
    config.hidden_act = "relu"  # Override GELU with ReLU

    model = ViTModel.from_pretrained(model_name, config=config)
    return model


def analyze_vit_structure(model: nn.Module):
    """Analyze and print the structure of a ViT model."""
    print("ViT Model Structure:")
    print("=" * 60)

    supported_types = (nn.Linear, nn.LayerNorm, nn.ReLU, nn.Softmax)

    for name, module in model.named_modules():
        if isinstance(module, supported_types):
            type_name = type(module).__name__
            if isinstance(module, nn.Linear):
                print(f"  {name}: {type_name}({module.in_features}, {module.out_features})")
            elif isinstance(module, nn.LayerNorm):
                print(f"  {name}: {type_name}({module.normalized_shape})")
            else:
                print(f"  {name}: {type_name}")


def example_vit_decomposition():
    """
    Example: DC decomposition of ViT MLP layers only.

    ViT has complex internal dataflow in attention (Q@K^T, attn@V) that
    requires special handling. This example focuses on the MLP blocks
    which have sequential structure compatible with hook-based decomposition.

    For full attention decomposition, you would need to:
    1. Extract and decompose the attention computation separately
    2. Use DCMatMulFunction for Q@K^T and attn@V operations
    3. Reconnect the decomposed flows
    """
    print("=" * 60)
    print("Example: ViT MLP Decomposition")
    print("=" * 60)

    if not HAS_TRANSFORMERS:
        print("Skipping: transformers not installed")
        return None

    # Create ViT with ReLU
    print("\nLoading ViT model with ReLU activations...")
    model = create_vit_with_relu()
    model.eval()

    # Identify MLP-related layers only (intermediate and output dense layers)
    # These have sequential structure that works with our hooks
    mlp_layers = []
    for name, module in model.named_modules():
        # Select only the MLP layers (intermediate and output)
        if 'intermediate' in name or ('output.dense' in name and 'attention' not in name):
            if isinstance(module, (nn.Linear, nn.ReLU)):
                mlp_layers.append(name)

    print(f"\nTargeting MLP layers only ({len(mlp_layers)} layers)")
    print(f"  Example layers: {mlp_layers[:3]}...")

    # Create decomposer with target layers
    decomposer = HookDecomposer(model, shift_mode=ShiftMode.CENTER, target_layers=mlp_layers)

    print(f"  Decomposed layers: {len(decomposer.layer_order)}")

    # Create dummy input (224x224 image)
    x = torch.randn(1, 3, 224, 224)

    print("\nRunning forward pass...")
    decomposer.initialize()
    with torch.no_grad():
        output = model(x)

    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.last_hidden_state.shape}")

    # Show layer activations
    print("\nLayer activations (MLP layers):")
    shown = 0
    for name in decomposer.layer_order:
        if shown >= 6:
            print("  ...")
            break
        pos, neg = decomposer.get_activation(name)
        if pos is not None:
            recon_error = (pos - neg - decomposer.caches[name].original_output).abs().max().item()
            print(f"  {name}: shape={tuple(pos.shape)}, recon_err={recon_error:.2e}")
            shown += 1

    # Verify reconstruction
    errors = decomposer.verify_reconstruction()
    if errors:
        max_error = max(errors.values())
        print(f"\nMax reconstruction error: {max_error:.2e}")
    else:
        print("\nNo layers with cached outputs")

    decomposer.remove_hooks()

    print("\nNote: For full ViT decomposition including attention, use")
    print("      DCMatMulFunction to handle Q@K^T and attn@V separately.")

    return decomposer


def example_attention_matmul():
    """
    Example: DC decomposition of attention-like matrix multiplication.

    This shows how to use DCMatMulFunction for Q @ K^T style operations
    where both operands have pos/neg decomposition.
    """
    print("\n" + "=" * 60)
    print("Example: Attention-style MatMul DC Decomposition")
    print("=" * 60)

    # Simulate attention dimensions
    batch_size = 2
    num_heads = 4
    seq_len = 16
    head_dim = 64

    # Create Q and K tensors
    Q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    K = torch.randn(batch_size, num_heads, seq_len, head_dim)

    # Decompose Q and K into pos/neg
    Q_pos = torch.relu(Q)
    Q_neg = torch.relu(-Q)
    K_pos = torch.relu(K)
    K_neg = torch.relu(-K)

    print(f"Q shape: {Q.shape}")
    print(f"K shape: {K.shape}")

    # Compute Q @ K^T using DC decomposition
    # Note: For Q @ K^T, we use transpose_b=True
    attn_pos, attn_neg = DCMatMulFunction.forward(
        Q_pos, Q_neg, K_pos, K_neg, transpose_b=True
    )

    # Verify reconstruction
    attn_original = torch.matmul(Q, K.transpose(-2, -1))
    attn_reconstructed = attn_pos - attn_neg
    recon_error = (attn_original - attn_reconstructed).abs().max().item()

    print(f"\nAttention scores shape: {attn_pos.shape}")
    print(f"Reconstruction error: {recon_error:.2e}")

    # Backward pass
    delta_pp = torch.ones_like(attn_pos)
    delta_np = torch.zeros_like(attn_pos)
    delta_pn = torch.zeros_like(attn_neg)
    delta_nn = torch.zeros_like(attn_neg)

    # Backward w.r.t. Q
    new_delta_pp, new_delta_np, new_delta_pn, new_delta_nn = DCMatMulFunction.backward(
        delta_pp, delta_np, delta_pn, delta_nn,
        K_pos, K_neg, transpose_b=True
    )

    print(f"\nBackward delta shapes (w.r.t. Q):")
    print(f"  delta_pp: {new_delta_pp.shape}, norm={new_delta_pp.norm():.4f}")
    print(f"  delta_np: {new_delta_np.shape}, norm={new_delta_np.norm():.4f}")

    # Backward w.r.t. K
    grad_K_pp, grad_K_np, grad_K_pn, grad_K_nn = DCMatMulFunction.backward_wrt_b(
        delta_pp, delta_np, delta_pn, delta_nn,
        Q_pos, Q_neg, transpose_b=True
    )

    print(f"\nBackward delta shapes (w.r.t. K):")
    print(f"  delta_pp: {grad_K_pp.shape}, norm={grad_K_pp.norm():.4f}")
    print(f"  delta_np: {grad_K_np.shape}, norm={grad_K_np.norm():.4f}")


def example_simple_layernorm_mlp():
    """
    Example: Simple transformer-style MLP with LayerNorm (no attention).

    This is useful for testing DC decomposition on the non-attention
    parts of a transformer.
    """
    print("\n" + "=" * 60)
    print("Example: Transformer MLP Block (LayerNorm + Linear + ReLU)")
    print("=" * 60)

    class TransformerMLP(nn.Module):
        def __init__(self, dim: int = 768, mlp_ratio: int = 4):
            super().__init__()
            self.norm = nn.LayerNorm(dim)
            self.fc1 = nn.Linear(dim, dim * mlp_ratio)
            self.act = nn.ReLU()
            self.fc2 = nn.Linear(dim * mlp_ratio, dim)

        def forward(self, x):
            x = self.norm(x)
            x = self.fc1(x)
            x = self.act(x)
            x = self.fc2(x)
            return x

    model = TransformerMLP(dim=768, mlp_ratio=4)
    model.eval()

    decomposer = HookDecomposer(model)

    # Sequence input
    x = torch.randn(2, 197, 768)  # Like ViT: batch=2, seq=197 (196 patches + CLS), dim=768

    decomposer.initialize()
    output = model(x)

    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    print("\nLayer activations:")
    for name in decomposer.layer_order:
        pos, neg = decomposer.get_activation(name)
        if pos is not None:
            recon_error = (pos - neg - decomposer.caches[name].original_output).abs().max().item()
            print(f"  {name}: shape={tuple(pos.shape)}, recon_err={recon_error:.2e}")

    decomposer.backward()

    print("\nSensitivities:")
    for name in decomposer.layer_order:
        sens = decomposer.get_sensitivity(name)
        if sens is not None:
            delta_pp, delta_np, delta_pn, delta_nn = sens
            print(f"  {name}:")
            print(f"    delta_pp: {delta_pp.norm():.4f}, delta_np: {delta_np.norm():.4f}")

    errors = decomposer.verify_reconstruction()
    max_error = max(errors.values()) if errors else 0
    print(f"\nMax reconstruction error: {max_error:.2e}")

    decomposer.remove_hooks()


def run_vit_examples():
    """Run all ViT-related examples."""
    example_simple_layernorm_mlp()
    example_attention_matmul()

    if HAS_TRANSFORMERS:
        example_vit_decomposition()
    else:
        print("\n" + "=" * 60)
        print("Skipping ViT example: transformers not installed")
        print("Install with: pip install transformers")
        print("=" * 60)


if __name__ == "__main__":
    run_vit_examples()
