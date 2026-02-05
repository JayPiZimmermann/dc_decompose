"""
Test the re-centering solution for magnitude explosion in residual networks.

Re-centering converts pos/neg to relu(z)/relu(-z) where z = pos - neg,
minimizing magnitudes while preserving the reconstruction.
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

class ResBlockWithAdd(nn.Module):
    """ResBlock using + operator (handled by functional_replacer)."""
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
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity  # Residual connection
        out = torch.relu(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        return self.fc(out)


def chain_1block():
    model = ResBlockWithAdd(16)
    model.eval()
    return model

def chain_3blocks():
    """Chain of 3 residual blocks."""
    class ChainedResBlocks(nn.Module):
        def __init__(self, channels, n_blocks):
            super().__init__()
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(channels),
                )
                for _ in range(n_blocks)
            ])
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(channels, 1)

        def forward(self, x):
            for block in self.blocks:
                identity = x
                out = block(x)
                out = out + identity
                x = torch.relu(out)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)

    model = ChainedResBlocks(16, 3)
    model.eval()
    return model

def chain_5blocks():
    """Chain of 5 residual blocks - tests magnitude stability."""
    class ChainedResBlocks(nn.Module):
        def __init__(self, channels, n_blocks):
            super().__init__()
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(channels),
                )
                for _ in range(n_blocks)
            ])
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(channels, 1)

        def forward(self, x):
            for block in self.blocks:
                identity = x
                out = block(x)
                out = out + identity
                x = torch.relu(out)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)

    model = ChainedResBlocks(16, 5)
    model.eval()
    return model


# =============================================================================
# Test Configuration
# =============================================================================

MODELS = {
    'ResBlock_1': (chain_1block, (1, 16, 8, 8)),
    'ResBlocks_x3': (chain_3blocks, (1, 16, 8, 8)),
    'ResBlocks_x5': (chain_5blocks, (1, 16, 8, 8)),
}


if __name__ == "__main__":
    success = run_model_tests(MODELS, title="Re-centering Solution Tests")
    sys.exit(0 if success else 1)
