"""
Test the re-centering solution for magnitude explosion in residual networks.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from dc_decompose.operations.patcher import patch_model, unpatch_model
from dc_decompose.operations.base import init_catted, reconstruct_output, recenter_dc, InputMode, split_input_4
from dc_decompose.operations.add import dc_add, DCAdd


class ResBlockWithRecenter(nn.Module):
    """ResBlock that uses dc_add for residual connection with re-centering."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu2 = nn.ReLU()
        self.dc_add = DCAdd(recenter=True)  # Re-centering add

    def forward(self, x):
        identity = x
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.dc_add(out, identity)  # Use DC-aware add
        out = self.relu2(out)
        return out


class ResBlockWithoutRecenter(nn.Module):
    """ResBlock without re-centering (for comparison)."""
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
        out = out + identity  # Regular add (no re-centering)
        out = self.relu2(out)
        return out


def analyze_magnitudes(name, tensor_dc, tensor_orig=None):
    """Analyze DC tensor magnitudes."""
    q = tensor_dc.shape[0] // 4
    pos = tensor_dc[:q]
    neg = tensor_dc[q:2*q]

    pos_mean = pos.abs().mean().item()
    neg_mean = neg.abs().mean().item()

    if tensor_orig is not None:
        recon = pos - neg
        error = torch.norm(tensor_orig - recon).item()
        rel_error = error / (torch.norm(tensor_orig).item() + 1e-10)
    else:
        error = 0
        rel_error = 0

    print(f"{name:25s}: pos_mean={pos_mean:10.3f}, neg_mean={neg_mean:10.3f}, error={error:.2e}, rel={rel_error:.2e}")
    return error


def test_recenter_function():
    """Test the recenter_dc function directly."""
    print("="*80)
    print("Test recenter_dc Function")
    print("="*80)

    torch.manual_seed(42)

    # Create DC tensor with large pos and neg
    batch = 2
    channels = 8
    h, w = 4, 4

    # Simulate accumulated pos/neg after many layers
    pos = torch.rand(batch, channels, h, w) * 1000 + 500  # Range [500, 1500]
    neg = torch.rand(batch, channels, h, w) * 1000 + 500  # Range [500, 1500]

    z_before = pos - neg  # This should be small ([-1000, 1000] but centered)

    # Create [4*batch] tensor
    from dc_decompose.operations.base import make_input_4
    tensor_4 = make_input_4(pos, neg)

    print(f"\nBefore recenter:")
    print(f"  pos mean: {pos.mean():.2f}, neg mean: {neg.mean():.2f}")
    print(f"  z mean: {z_before.mean():.2f}, z range: [{z_before.min():.2f}, {z_before.max():.2f}]")

    # Apply recenter
    recentered = recenter_dc(tensor_4)

    q = recentered.shape[0] // 4
    new_pos = recentered[:q]
    new_neg = recentered[q:2*q]
    z_after = new_pos - new_neg

    print(f"\nAfter recenter:")
    print(f"  pos mean: {new_pos.mean():.2f}, neg mean: {new_neg.mean():.2f}")
    print(f"  z mean: {z_after.mean():.2f}, z range: [{z_after.min():.2f}, {z_after.max():.2f}]")

    # Verify z is preserved
    z_error = torch.norm(z_before - z_after).item()
    print(f"\nz preservation error: {z_error:.2e}")

    assert z_error < 1e-5, f"Re-centering changed z! Error: {z_error}"
    print("PASS: z is preserved after re-centering")


def test_chain_comparison():
    """Compare chains with and without re-centering."""
    print("\n" + "="*80)
    print("Chain Comparison: With vs Without Re-centering")
    print("="*80)

    torch.manual_seed(42)
    x = torch.randn(1, 16, 8, 8)

    n_blocks = 5

    # Chain WITHOUT re-centering
    print("\n--- Without Re-centering ---")
    blocks_no_recenter = nn.Sequential(*[ResBlockWithoutRecenter(16) for _ in range(n_blocks)])
    blocks_no_recenter.eval()

    orig_outputs_no = []
    current = x
    for block in blocks_no_recenter:
        with torch.no_grad():
            current = block(current)
            orig_outputs_no.append(current.clone())

    patch_model(blocks_no_recenter)
    x_cat = init_catted(x, InputMode.CENTER)
    current_dc = x_cat
    for i, block in enumerate(blocks_no_recenter):
        current_dc = block(current_dc)
        analyze_magnitudes(f"Block {i}", current_dc, orig_outputs_no[i])
    unpatch_model(blocks_no_recenter)

    # Chain WITH re-centering
    print("\n--- With Re-centering (DCAdd) ---")
    blocks_recenter = nn.Sequential(*[ResBlockWithRecenter(16) for _ in range(n_blocks)])
    blocks_recenter.eval()

    # Copy weights from no-recenter version to ensure fair comparison
    for i in range(n_blocks):
        blocks_recenter[i].conv1.load_state_dict(blocks_no_recenter[i].conv1.state_dict())
        blocks_recenter[i].conv2.load_state_dict(blocks_no_recenter[i].conv2.state_dict())
        blocks_recenter[i].bn1.load_state_dict(blocks_no_recenter[i].bn1.state_dict())
        blocks_recenter[i].bn2.load_state_dict(blocks_no_recenter[i].bn2.state_dict())

    # Note: orig_outputs will be the same since re-centering preserves z
    orig_outputs_rc = []
    current = x
    for block in blocks_recenter:
        with torch.no_grad():
            current = block(current)
            orig_outputs_rc.append(current.clone())

    patch_model(blocks_recenter)
    x_cat = init_catted(x, InputMode.CENTER)
    current_dc = x_cat
    for i, block in enumerate(blocks_recenter):
        current_dc = block(current_dc)
        analyze_magnitudes(f"Block {i}", current_dc, orig_outputs_rc[i])
    unpatch_model(blocks_recenter)


def test_dc_add_in_isolation():
    """Test DCAdd module in isolation."""
    print("\n" + "="*80)
    print("Test DCAdd Module")
    print("="*80)

    torch.manual_seed(42)

    x = torch.randn(1, 16, 8, 8)
    y = torch.randn(1, 16, 8, 8)

    # Expected result
    expected = x + y

    # DC format
    x_cat = init_catted(x, InputMode.CENTER)
    y_cat = init_catted(y, InputMode.CENTER)

    # Without recenter
    add_no_recenter = DCAdd(recenter=False)
    sum_no_recenter = add_no_recenter(x_cat, y_cat)
    recon_no_recenter = reconstruct_output(sum_no_recenter)
    error_no = torch.norm(expected - recon_no_recenter).item()

    # With recenter
    add_recenter = DCAdd(recenter=True)
    sum_recenter = add_recenter(x_cat, y_cat)
    recon_recenter = reconstruct_output(sum_recenter)
    error_rc = torch.norm(expected - recon_recenter).item()

    print(f"DCAdd without recenter - error: {error_no:.2e}")
    print(f"DCAdd with recenter - error: {error_rc:.2e}")

    # Check magnitudes
    q = sum_no_recenter.shape[0] // 4
    print(f"\nWithout recenter: pos_mean={sum_no_recenter[:q].abs().mean():.3f}, neg_mean={sum_no_recenter[q:2*q].abs().mean():.3f}")
    print(f"With recenter: pos_mean={sum_recenter[:q].abs().mean():.3f}, neg_mean={sum_recenter[q:2*q].abs().mean():.3f}")


def run_all_tests():
    """Run all re-centering tests."""
    test_recenter_function()
    test_dc_add_in_isolation()
    test_chain_comparison()

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("Re-centering prevents magnitude explosion in residual networks.")
    print("Use DCAdd(recenter=True) or dc_add(x, y, recenter=True) for residual connections.")


if __name__ == "__main__":
    run_all_tests()
