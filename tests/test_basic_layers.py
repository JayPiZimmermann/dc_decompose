"""
Test basic layer functionality - Linear, Conv, ReLU, BatchNorm, etc.
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

# Single layers
def single_linear():
    return nn.Linear(10, 5)

def single_conv2d():
    return nn.Conv2d(3, 16, 3)

def single_relu():
    return nn.ReLU()

def single_batchnorm1d():
    model = nn.BatchNorm1d(10)
    model.eval()
    return model

def single_batchnorm2d():
    model = nn.BatchNorm2d(8)
    model.eval()
    return model

# Pooling layers
def single_maxpool2d():
    return nn.MaxPool2d(2)

def single_avgpool2d():
    return nn.AvgPool2d(2)

def single_adaptive_avgpool2d():
    return nn.AdaptiveAvgPool2d((4, 4))

# Utility layers
def single_flatten():
    return nn.Flatten()

def single_dropout():
    model = nn.Dropout(0.5)
    model.eval()  # Dropout is identity in eval mode
    return model

def single_identity():
    return nn.Identity()

# Simple chains
def linear_chain():
    return nn.Sequential(
        nn.Linear(8, 16),
        nn.ReLU(),
        nn.Linear(16, 4)
    )

def conv_chain():
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(32, 10)
    )
    model.eval()
    return model

def mixed_chain():
    model = nn.Sequential(
        nn.Linear(20, 50),
        nn.BatchNorm1d(50),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(50, 10),
        nn.ReLU()
    )
    model.eval()
    return model


# =============================================================================
# Test Configuration
# =============================================================================

MODELS = {
    # Single layers
    'Linear': (single_linear, (3, 10)),
    'Conv2d': (single_conv2d, (2, 3, 8, 8)),
    'ReLU': (single_relu, (2, 10)),
    'BatchNorm1d': (single_batchnorm1d, (4, 10)),
    'BatchNorm2d': (single_batchnorm2d, (2, 8, 16, 16)),

    # Pooling layers
    'MaxPool2d': (single_maxpool2d, (2, 4, 16, 16)),
    'AvgPool2d': (single_avgpool2d, (2, 4, 16, 16)),
    'AdaptiveAvgPool2d': (single_adaptive_avgpool2d, (2, 8, 16, 16)),

    # Utility layers
    'Flatten': (single_flatten, (2, 4, 8, 8)),
    'Dropout': (single_dropout, (2, 10)),
    'Identity': (single_identity, (2, 10)),

    # Simple chains
    'LinearChain': (linear_chain, (2, 8)),
    'ConvChain': (conv_chain, (2, 3, 16, 16)),
    'MixedChain': (mixed_chain, (4, 20)),
}


if __name__ == "__main__":
    success = run_model_tests(MODELS, title="Basic Layer Tests")
    sys.exit(0 if success else 1)
