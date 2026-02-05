"""
Test DC decomposition on CNN models.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch.nn as nn
from utils import run_model_tests


# =============================================================================
# Model Definitions
# =============================================================================

def CNN_2conv():
    return nn.Sequential(
        nn.Conv2d(1, 2, 2), nn.ReLU(),
        nn.Conv2d(2, 4, 2), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(4, 1)
    )

def CNN_2conv_maxpool():
    return nn.Sequential(
        nn.Conv2d(1, 4, 3, stride=2), nn.ReLU(),
        nn.Conv2d(4, 8, 2), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(8, 1)
    )

def CNN_deep_relu():
    return nn.Sequential(
        nn.Conv2d(1, 4, 2), nn.ReLU(),
        nn.Conv2d(4, 8, 2), nn.ReLU(),
        nn.Conv2d(8, 4, 2), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(4, 1)
    )

def CNN_inner_maxpool():
    return nn.Sequential(
        nn.Conv2d(1, 4, 2), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(4, 8, 2), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(8, 1)
    )

def CNN_inner_avgpool():
    return nn.Sequential(
        nn.Conv2d(1, 4, 2), nn.ReLU(),
        nn.AvgPool2d(2),
        nn.Conv2d(4, 8, 2), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(8, 1)
    )

def CNN_BN_Relu():
    model = nn.Sequential(
        nn.Conv2d(1, 4, 2),
        nn.BatchNorm2d(4),
        nn.ReLU(),
        nn.Conv2d(4, 8, 3),
        nn.BatchNorm2d(8),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(8, 1)
    )
    model.eval()
    return model

def CNN_stride_padding():
    return nn.Sequential(
        nn.Conv2d(1, 4, 3, stride=2, padding=1), nn.ReLU(),
        nn.Conv2d(4, 8, 2, stride=2), nn.ReLU(),
        nn.Conv2d(8, 8, 2, stride=2, padding=1), nn.ReLU(),
        nn.Conv2d(8, 4, 2), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(4, 1)
    )

def CNN_Pointwise():
    model = nn.Sequential(
        nn.Conv2d(1, 16, 3), nn.ReLU(),
        nn.Conv2d(16, 8, 1),  # Pointwise conv
        nn.BatchNorm2d(8), nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(8, 1)
    )
    model.eval()
    return model

def CNN_VGG_Block():
    model = nn.Sequential(
        nn.Conv2d(1, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(),
        nn.Conv2d(8, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(8, 1)
    )
    model.eval()
    return model


# =============================================================================
# Test Configuration
# =============================================================================

MODELS = {
    "CNN_2conv": (CNN_2conv, (1, 1, 8, 8)),
    "CNN_2conv_maxpool": (CNN_2conv_maxpool, (1, 1, 16, 16)),
    "CNN_deep_relu": (CNN_deep_relu, (1, 1, 8, 8)),
    "CNN_inner_maxpool": (CNN_inner_maxpool, (1, 1, 8, 8)),
    "CNN_inner_avgpool": (CNN_inner_avgpool, (1, 1, 8, 8)),
    "CNN_BN_Relu": (CNN_BN_Relu, (1, 1, 8, 8)),
    "CNN_stride_padding": (CNN_stride_padding, (1, 1, 32, 32)),
    "CNN_Pointwise": (CNN_Pointwise, (1, 1, 16, 16)),
    "CNN_VGG_Block": (CNN_VGG_Block, (1, 1, 16, 16)),
}


if __name__ == "__main__":
    success = run_model_tests(MODELS, title="CNN Model Tests")
    sys.exit(0 if success else 1)
