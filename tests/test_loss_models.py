"""
Test DC decomposition with multi-dimensional output models and loss functions.

Tests:
1. Models with 2D/3D/4D outputs (not just scalar)
2. Loss functions applied to reconstructed output (not split)
3. Both full DC decomposition and backward-only mode (cached masks)
4. Both functional and context manager APIs
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from tests.utils.dc_tester import (
    run_loss_model_tests,
    test_model_with_loss,
    _print_result,
    _print_summary,
)


# =============================================================================
# Models with Multi-Dimensional Output
# =============================================================================

class SimpleSegmentationNet(nn.Module):
    """Simple encoder-decoder for segmentation-like task. Output: [B, C, H, W]"""

    def __init__(self, in_channels=3, num_classes=5):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        # Decoder
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.dec2 = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.dec1(x)
        x = self.dec2(x)
        return x


class DeepSegmentationNet(nn.Module):
    """Deeper segmentation network with skip connections. Output: [B, C, H, W]"""

    def __init__(self, in_channels=3, num_classes=5):
        super().__init__()
        # Encoder path
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.pool2 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        # Decoder path
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),  # 64 + 64 from skip
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),  # 32 + 32 from skip
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.output = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        # Bottleneck
        b = self.bottleneck(p2)

        # Decoder with skip connections
        u1 = self.up1(b)
        d1 = self.dec1(torch.cat([u1, e2], dim=1))

        u2 = self.up2(d1)
        d2 = self.dec2(torch.cat([u2, e1], dim=1))

        return self.output(d2)


class FeatureExtractor(nn.Module):
    """Feature extraction network. Output: [B, C, H, W] feature maps"""

    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, out_channels, 3, padding=1),
        )

    def forward(self, x):
        return self.features(x)


class Seq2SeqLike(nn.Module):
    """Sequence model with multi-dimensional output. Output: [B, T, C]"""

    def __init__(self, input_size=32, hidden_size=64, output_size=16):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        # x: [B, T, input_size]
        x = F.relu(self.fc1(x))
        x = self.norm(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x  # [B, T, output_size]


class AutoEncoder(nn.Module):
    """Autoencoder with same-size output. Output: [B, C, H, W]"""

    def __init__(self, in_channels=3):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, in_channels, 3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


class ResBlock(nn.Module):
    """Residual block for deep network."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)


class DeepResAutoEncoder(nn.Module):
    """Deep autoencoder with residual blocks. Output: [B, C, H, W]"""

    def __init__(self, in_channels=3, num_blocks=3):
        super().__init__()
        # Encoder
        self.enc_conv = nn.Conv2d(in_channels, 32, 3, stride=2, padding=1)
        self.enc_blocks = nn.Sequential(*[ResBlock(32) for _ in range(num_blocks)])

        # Decoder
        self.dec_blocks = nn.Sequential(*[ResBlock(32) for _ in range(num_blocks)])
        self.dec_conv = nn.ConvTranspose2d(32, in_channels, 3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x = F.relu(self.enc_conv(x))
        x = self.enc_blocks(x)
        x = self.dec_blocks(x)
        x = self.dec_conv(x)
        return x


# =============================================================================
# Loss Functions
# =============================================================================

def mse_loss(output: Tensor, target: Tensor) -> Tensor:
    """Mean squared error loss."""
    return F.mse_loss(output, target)


def l1_loss(output: Tensor, target: Tensor) -> Tensor:
    """L1 / MAE loss."""
    return F.l1_loss(output, target)


def cross_entropy_loss(output: Tensor, target: Tensor) -> Tensor:
    """Cross entropy for segmentation (target is class indices)."""
    # output: [B, C, H, W], target: [B, H, W] with class indices
    return F.cross_entropy(output, target)


def smooth_l1_loss(output: Tensor, target: Tensor) -> Tensor:
    """Smooth L1 / Huber loss."""
    return F.smooth_l1_loss(output, target)


# =============================================================================
# Test Configurations
# =============================================================================

def create_test_models():
    """Create test models with their inputs, targets, and loss functions."""
    torch.manual_seed(42)

    models = {}

    # 1. Simple Segmentation with MSE loss (regression target)
    models['SimpleSegNet_MSE'] = (
        SimpleSegmentationNet(in_channels=3, num_classes=5),
        torch.randn(2, 3, 32, 32),  # input
        torch.randn(2, 5, 32, 32),  # target (same shape as output)
        mse_loss,
    )

    # 2. Simple Segmentation with CrossEntropy (classification target)
    models['SimpleSegNet_CE'] = (
        SimpleSegmentationNet(in_channels=3, num_classes=5),
        torch.randn(2, 3, 32, 32),
        torch.randint(0, 5, (2, 32, 32)),  # class indices
        cross_entropy_loss,
    )

    # 3. Feature Extractor with L1 loss
    models['FeatureExtractor_L1'] = (
        FeatureExtractor(in_channels=3, out_channels=64),
        torch.randn(2, 3, 16, 16),
        torch.randn(2, 64, 16, 16),
        l1_loss,
    )

    # 4. Sequence model with MSE
    models['Seq2Seq_MSE'] = (
        Seq2SeqLike(input_size=32, hidden_size=64, output_size=16),
        torch.randn(2, 10, 32),  # [B, T, input_size]
        torch.randn(2, 10, 16),  # [B, T, output_size]
        mse_loss,
    )

    # 5. AutoEncoder with MSE (reconstruction)
    input_ae = torch.randn(2, 3, 32, 32)
    models['AutoEncoder_MSE'] = (
        AutoEncoder(in_channels=3),
        input_ae,
        input_ae.clone(),  # reconstruct input
        mse_loss,
    )

    # 6. Deep Segmentation with Smooth L1
    models['DeepSegNet_SmoothL1'] = (
        DeepSegmentationNet(in_channels=3, num_classes=5),
        torch.randn(2, 3, 64, 64),
        torch.randn(2, 5, 64, 64),
        smooth_l1_loss,
    )

    # 7. Deep ResNet AutoEncoder
    input_res = torch.randn(2, 3, 32, 32)
    models['DeepResAutoEncoder_MSE'] = (
        DeepResAutoEncoder(in_channels=3, num_blocks=2),
        input_res,
        input_res.clone(),
        mse_loss,
    )

    return models


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    models = create_test_models()

    print("\n" + "=" * 90)
    print("PART 1: Full DC Decomposition Mode")
    print("=" * 90)
    success1 = run_loss_model_tests(
        models,
        title="Multi-Dimensional Output Models with Loss Functions",
        backward_only=False,
        test_both_apis=True,
    )

    print("\n" + "=" * 90)
    print("PART 2: Backward-Only Mode (Cached Masks)")
    print("=" * 90)
    success2 = run_loss_model_tests(
        models,
        title="Multi-Dimensional Output Models - Backward Only Mode",
        backward_only=True,
        test_both_apis=True,
    )

    print("\n" + "=" * 90)
    print("FINAL SUMMARY")
    print("=" * 90)
    print(f"Full DC Mode: {'PASS' if success1 else 'FAIL'}")
    print(f"Backward-Only Mode: {'PASS' if success2 else 'FAIL'}")

    sys.exit(0 if (success1 and success2) else 1)
