"""
Test patcher-based DC decomposition on ResNet models.

The workflow is:
1. Write standard PyTorch models with torch.relu() and + operations
2. Call prepare_model_for_dc(model) which does make_dc_compatible + patch_model
3. Run DC forward pass with init_catted and reconstruct_output
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

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity  # Residual connection
        out = torch.relu(out)
        return out


class SimpleResNet(nn.Module):
    """Simple ResNet using standard PyTorch operations."""
    def __init__(self, num_blocks=3):
        super().__init__()
        self.stem_conv = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(64)
        self.blocks = nn.Sequential(*[ResBlock(64) for _ in range(num_blocks)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.stem_conv(x)
        x = self.stem_bn(x)
        x = torch.relu(x)
        x = self.blocks(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# Individual component models for testing
def conv2d_layer():
    model = nn.Conv2d(16, 16, 3, padding=1, bias=False)
    return model

def batchnorm_layer():
    model = nn.BatchNorm2d(16)
    model.eval()
    return model

def relu_layer():
    return nn.ReLU()

def conv_bn_relu():
    model = nn.Sequential(
        nn.Conv2d(16, 16, 3, padding=1, bias=False),
        nn.BatchNorm2d(16),
        nn.ReLU()
    )
    model.eval()
    return model

def resblock_single():
    model = ResBlock(64)
    model.eval()
    return model

def resblocks_chained(n):
    def factory():
        model = nn.Sequential(*[ResBlock(64) for _ in range(n)])
        model.eval()
        return model
    return factory

def simple_resnet(num_blocks):
    def factory():
        model = SimpleResNet(num_blocks=num_blocks)
        model.eval()
        return model
    return factory


# =============================================================================
# Test Configuration
# =============================================================================

MODELS = {
    # Individual components
    'Conv2d': (conv2d_layer, (2, 16, 8, 8)),
    'BatchNorm2d': (batchnorm_layer, (2, 16, 8, 8)),
    'ReLU': (relu_layer, (2, 16, 8, 8)),
    'Conv+BN+ReLU': (conv_bn_relu, (2, 16, 8, 8)),

    # ResBlock
    'ResBlock': (resblock_single, (1, 64, 8, 8)),

    # Chained ResBlocks
    'ResBlocks_x1': (resblocks_chained(1), (1, 64, 8, 8)),
    'ResBlocks_x2': (resblocks_chained(2), (1, 64, 8, 8)),
    'ResBlocks_x3': (resblocks_chained(3), (1, 64, 8, 8)),
    'ResBlocks_x5': (resblocks_chained(5), (1, 64, 8, 8)),

    # Full ResNet
    'SimpleResNet_3': (simple_resnet(3), (1, 3, 32, 32)),
}


if __name__ == "__main__":
    success = run_model_tests(MODELS, title="Patcher ResNet Tests")
    sys.exit(0 if success else 1)
