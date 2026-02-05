"""
Debug magnitude growth in DC decomposition.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from dc_decompose.patcher import patch_model, unpatch_model
from dc_decompose.operations.base import init_catted, reconstruct_output, InputMode, split_input_4


class SimpleBlock(nn.Module):
    """Simple block without residual."""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ResBlock(nn.Module):
    """ResBlock with residual."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu2(out + identity)
        return out


def analyze_magnitudes(name, tensor_dc, tensor_orig=None):
    """Analyze DC tensor magnitudes."""
    q = tensor_dc.shape[0] // 4
    pos = tensor_dc[:q]
    neg = tensor_dc[q:2*q]

    pos_mean = pos.abs().mean().item()
    neg_mean = neg.abs().mean().item()
    recon = pos - neg
    recon_mean = recon.abs().mean().item()

    if tensor_orig is not None:
        error = torch.norm(tensor_orig - recon).item()
        rel_error = error / (torch.norm(tensor_orig).item() + 1e-10)
    else:
        error = 0
        rel_error = 0

    print(f"{name:20s}: pos_mean={pos_mean:8.3f}, neg_mean={neg_mean:8.3f}, recon_mean={recon_mean:8.3f}, error={error:.2e}, rel={rel_error:.2e}")


def test_chains():
    """Compare chain of simple blocks vs chain of ResBlocks."""
    print("="*100)
    print("Magnitude Growth Analysis")
    print("="*100)

    torch.manual_seed(42)
    x = torch.randn(1, 16, 8, 8)

    # Chain of SimpleBlocks
    print("\n--- Chain of Simple Blocks (no residual) ---")
    blocks_simple = nn.Sequential(*[SimpleBlock(16) for _ in range(5)])
    blocks_simple.eval()

    patch_model(blocks_simple)
    x_cat = init_catted(x, InputMode.CENTER)

    current = x_cat
    current_orig = x
    for i, block in enumerate(blocks_simple):
        with torch.no_grad():
            current_orig = block._modules['relu'](block._modules['bn'](block._modules['conv'](current_orig)))

        unpatch_model(block)  # Temporarily unpatch to get original
        with torch.no_grad():
            orig_block_out = block(current_orig)
        patch_model(block)

        current = block(current)
        analyze_magnitudes(f"After block {i}", current, current_orig)

    unpatch_model(blocks_simple)

    # Chain of ResBlocks
    print("\n--- Chain of ResBlocks (with residual) ---")
    blocks_res = nn.Sequential(*[ResBlock(16) for _ in range(5)])
    blocks_res.eval()

    # Get original outputs
    orig_outputs = []
    current_orig = x
    for block in blocks_res:
        with torch.no_grad():
            current_orig = block(current_orig)
            orig_outputs.append(current_orig.clone())

    patch_model(blocks_res)
    x_cat = init_catted(x, InputMode.CENTER)

    current = x_cat
    for i, block in enumerate(blocks_res):
        current = block(current)
        analyze_magnitudes(f"After block {i}", current, orig_outputs[i])

    unpatch_model(blocks_res)


def test_relu_effect():
    """Analyze ReLU's effect on magnitude growth."""
    print("\n" + "="*100)
    print("ReLU Effect on DC Format")
    print("="*100)

    torch.manual_seed(42)

    # Create a tensor with both positive and negative values
    x = torch.randn(1, 16, 8, 8)
    x_cat = init_catted(x, InputMode.CENTER)

    q = x_cat.shape[0] // 4
    pos = x_cat[:q]
    neg = x_cat[q:2*q]

    print(f"\nOriginal DC values:")
    print(f"  pos: mean={pos.mean():.3f}, min={pos.min():.3f}, max={pos.max():.3f}")
    print(f"  neg: mean={neg.mean():.3f}, min={neg.min():.3f}, max={neg.max():.3f}")
    print(f"  recon: mean={x.mean():.3f}, min={x.min():.3f}, max={x.max():.3f}")

    # Simulate what ReLU does to DC format
    # ReLU(z) = ReLU(pos - neg)
    # DC ReLU: forward_pos = max(pos, neg), forward_neg = neg
    relu = nn.ReLU()
    patch_model(relu)

    out = relu(x_cat)
    unpatch_model(relu)

    out_pos = out[:q]
    out_neg = out[q:2*q]
    recon = out_pos - out_neg

    print(f"\nAfter DC ReLU (max mode):")
    print(f"  pos: mean={out_pos.mean():.3f}, min={out_pos.min():.3f}, max={out_pos.max():.3f}")
    print(f"  neg: mean={out_neg.mean():.3f}, min={out_neg.min():.3f}, max={out_neg.max():.3f}")

    # Compare with regular ReLU
    reg_out = torch.relu(x)
    print(f"  recon: mean={recon.mean():.3f}, min={recon.min():.3f}, max={recon.max():.3f}")
    print(f"  original relu: mean={reg_out.mean():.3f}, min={reg_out.min():.3f}, max={reg_out.max():.3f}")
    print(f"  error: {torch.norm(recon - reg_out).item():.2e}")


def test_scaling():
    """Test if the issue is related to numerical scaling."""
    print("\n" + "="*100)
    print("Numerical Scaling Test")
    print("="*100)

    torch.manual_seed(42)

    for scale in [0.1, 1.0, 10.0, 100.0]:
        x = scale * torch.randn(1, 16, 8, 8)

        blocks = nn.Sequential(*[ResBlock(16) for _ in range(3)])
        blocks.eval()

        # Original
        orig_out = blocks(x)

        # DC
        patch_model(blocks)
        x_cat = init_catted(x, InputMode.CENTER)
        dc_out = blocks(x_cat)
        dc_recon = reconstruct_output(dc_out)
        unpatch_model(blocks)

        error = torch.norm(orig_out - dc_recon).item()
        rel_error = error / (torch.norm(orig_out).item() + 1e-10)

        print(f"Scale {scale:5.1f}: abs_error={error:.2e}, rel_error={rel_error:.2e}")


if __name__ == "__main__":
    test_relu_effect()
    test_chains()
    test_scaling()
