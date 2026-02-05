"""
Test functional_replacer.py for automatic conversion of torch.relu and + to modules.

These models use torch.relu() and + operations which get replaced by
prepare_model_for_dc() via make_dc_compatible().
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

class ResBlockFunctional(nn.Module):
    """Standard ResBlock using torch.relu() and + (no special modules)."""
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
        out = torch.relu(out)  # Functional call
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity   # Standard addition (residual)
        out = torch.relu(out)  # Functional call
        return out


class SimpleResNetFunctional(nn.Module):
    """Standard ResNet using functional calls."""
    def __init__(self, num_blocks=3):
        super().__init__()
        self.stem_conv = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(64)
        self.blocks = nn.Sequential(*[ResBlockFunctional(64) for _ in range(num_blocks)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.stem_conv(x)
        x = self.stem_bn(x)
        x = torch.relu(x)  # Functional call
        x = self.blocks(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class SimpleModelWithRelu(nn.Module):
    """Simple model using torch.relu."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        return torch.relu(self.linear(x))


class AddModel(nn.Module):
    """Model with tensor addition."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        return x + self.linear(x)


def resblock_single():
    model = ResBlockFunctional(64)
    model.eval()
    return model

def resnet_3blocks():
    model = SimpleResNetFunctional(num_blocks=3)
    model.eval()
    return model


# =============================================================================
# Test Configuration
# =============================================================================

MODELS = {
    'SimpleRelu': (SimpleModelWithRelu, (2, 10)),
    'AddModel': (AddModel, (2, 10)),
    'ResBlockFunctional': (resblock_single, (1, 64, 8, 8)),
    'SimpleResNet_3blocks': (resnet_3blocks, (1, 3, 32, 32)),
}


if __name__ == "__main__":
    success = run_model_tests(MODELS, title="Functional Replacer Tests")
    sys.exit(0 if success else 1)
