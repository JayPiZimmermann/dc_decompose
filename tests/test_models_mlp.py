"""
Test DC decomposition on MLP models.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch.nn as nn
from utils import run_model_tests


# =============================================================================
# Model Definitions
# =============================================================================

def MLP_1layer():
    return nn.Sequential(nn.Linear(3, 2))

def MLP_1layer_relu():
    return nn.Sequential(nn.Linear(3, 2), nn.ReLU())

def MLP_2layer():
    return nn.Sequential(nn.Linear(3, 2), nn.ReLU(), nn.Linear(2, 1))

def MLP_3layer():
    return nn.Sequential(nn.Linear(3, 2), nn.ReLU(), nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 1))

def MLP_4layer():
    return nn.Sequential(
        nn.Linear(3, 4), nn.ReLU(),
        nn.Linear(4, 3), nn.ReLU(),
        nn.Linear(3, 2), nn.ReLU(),
        nn.Linear(2, 1)
    )

def MLP_5layer():
    return nn.Sequential(
        nn.Linear(3, 4), nn.ReLU(),
        nn.Linear(4, 4), nn.ReLU(),
        nn.Linear(4, 3), nn.ReLU(),
        nn.Linear(3, 2), nn.ReLU(),
        nn.Linear(2, 1)
    )


# =============================================================================
# Test Configuration
# =============================================================================

MODELS = {
    "MLP_1layer": (MLP_1layer, (1, 3)),
    "MLP_1layer_relu": (MLP_1layer_relu, (1, 3)),
    "MLP_2layer": (MLP_2layer, (1, 3)),
    "MLP_3layer": (MLP_3layer, (1, 3)),
    "MLP_4layer": (MLP_4layer, (1, 3)),
    "MLP_5layer": (MLP_5layer, (1, 3)),
}


if __name__ == "__main__":
    success = run_model_tests(MODELS, title="MLP Model Tests")
    sys.exit(0 if success else 1)
