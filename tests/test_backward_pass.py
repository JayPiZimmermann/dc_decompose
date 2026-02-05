"""
Test backward pass for DC decomposition.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from utils import run_model_tests


# =============================================================================
# Model Definitions - Focus on backward pass accuracy
# =============================================================================

def linear_single():
    return nn.Linear(10, 5)

def linear_chain():
    return nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10),
        nn.ReLU(),
        nn.Linear(10, 5)
    )

def conv_bn_relu():
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(16, 1)
    )
    model.eval()
    return model


class ResBlock(nn.Module):
    """ResBlock for backward pass testing."""
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
        out = torch.relu(out + identity)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        return self.fc(out)


def resblock_model():
    model = ResBlock(16)
    model.eval()
    return model


# =============================================================================
# Test Configuration
# =============================================================================

MODELS = {
    'Linear': (linear_single, (2, 10)),
    'LinearChain': (linear_chain, (2, 10)),
    'Conv+BN+ReLU': (conv_bn_relu, (1, 3, 8, 8)),
    'ResBlock': (resblock_model, (1, 16, 8, 8)),
}


if __name__ == "__main__":
    success = run_model_tests(MODELS, title="Backward Pass Tests")
    sys.exit(0 if success else 1)
