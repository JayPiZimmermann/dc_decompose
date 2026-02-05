"""
Test DC decomposition on ResNet models with residual connections.

These models use torch.relu() and + operations which get replaced by
prepare_model_for_dc().
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from utils import run_model_tests


# =============================================================================
# Model Definitions
# =============================================================================

class ResBlock(nn.Module):
    """Standard ResBlock using torch.relu() and + (standard PyTorch)."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels, 1)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity  # Residual connection
        out = torch.relu(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class ResBlockNoBN(nn.Module):
    """ResBlock without BatchNorm."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels, 1)

    def forward(self, x):
        identity = x
        out = torch.relu(self.conv1(x))
        out = self.conv2(out)
        out = torch.relu(out + identity)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class ResBlockInternal(nn.Module):
    """Internal ResBlock (no final fc) for use in SimpleResNet."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        out = torch.relu(out)
        return out


class SimpleResNet(nn.Module):
    """Simple ResNet with multiple blocks."""
    def __init__(self, num_blocks=2, channels=16):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.blocks = nn.Sequential(*[ResBlockInternal(channels) for _ in range(num_blocks)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels, 1)

    def forward(self, x):
        x = self.stem(x)
        x = torch.relu(x)
        x = self.blocks(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def ResBlock_single():
    model = ResBlock(16)
    model.eval()
    return model

def ResBlockNoBN_single():
    model = ResBlockNoBN(16)
    model.eval()
    return model

def ResNet_2blocks():
    model = SimpleResNet(num_blocks=2, channels=16)
    model.eval()
    return model

def ResNet_3blocks():
    model = SimpleResNet(num_blocks=3, channels=16)
    model.eval()
    return model


# =============================================================================
# Test Configuration
# =============================================================================

MODELS = {
    "ResBlock": (ResBlock_single, (1, 16, 8, 8)),
    "ResBlockNoBN": (ResBlockNoBN_single, (1, 16, 8, 8)),
    "ResNet_2blocks": (ResNet_2blocks, (1, 3, 16, 16)),
    "ResNet_3blocks": (ResNet_3blocks, (1, 3, 16, 16)),
}


if __name__ == "__main__":
    success = run_model_tests(MODELS, title="ResNet Model Tests")
    sys.exit(0 if success else 1)
