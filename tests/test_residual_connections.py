"""
Test residual connections and tensor addition operations.
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

class SimpleResidual(nn.Module):
    """Very simple residual connection test."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 1)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = torch.relu(out)
        out = self.conv2(out)
        out = out + identity  # Residual connection
        out = torch.relu(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class LinearResidual(nn.Module):
    """Linear residual connection."""
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(64, 64)
        self.linear2 = nn.Linear(64, 64)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        identity = x
        out = self.linear1(x)
        out = torch.relu(out)
        out = self.linear2(out)
        out = out + identity  # Residual connection
        out = torch.relu(out)
        out = self.fc(out)
        return out


class MultipleAdditions(nn.Module):
    """Multiple tensor additions."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(8, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 8, 3, padding=1)
        self.conv3 = nn.Conv2d(8, 8, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, 1)

    def forward(self, x):
        branch1 = self.conv1(x)
        branch2 = self.conv2(x)
        branch3 = self.conv3(x)

        # Multiple additions
        out = branch1 + branch2 + branch3
        out = torch.relu(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class NestedResidual(nn.Module):
    """Nested residual connections."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        # First residual
        out1 = self.conv1(x)
        out1 = out1 + x
        out1 = torch.relu(out1)

        # Second residual
        out2 = self.conv2(out1)
        out2 = out2 + out1
        out2 = torch.relu(out2)

        # Third residual
        out3 = self.conv3(out2)
        out3 = out3 + out2
        out3 = torch.relu(out3)

        out = self.pool(out3)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


# =============================================================================
# Test Configuration
# =============================================================================

MODELS = {
    'SimpleResidual': (SimpleResidual, (1, 16, 8, 8)),
    'LinearResidual': (LinearResidual, (2, 64)),
    'MultipleAdditions': (MultipleAdditions, (1, 8, 16, 16)),
    'NestedResidual': (NestedResidual, (1, 32, 8, 8)),
}


if __name__ == "__main__":
    success = run_model_tests(MODELS, title="Residual Connection Tests")
    sys.exit(0 if success else 1)
