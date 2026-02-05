"""
Test complex models - Deep networks, residual connections, complex architectures.
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

class DeepMLP(nn.Module):
    """Deep multi-layer perceptron."""
    def __init__(self, input_size=128, hidden_sizes=[256, 512, 256, 128], output_size=1):
        super().__init__()
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
            ])
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class ResidualBlock(nn.Module):
    """Simple residual block."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        identity = self.skip(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = torch.relu(out)

        return out


class DeepResNet(nn.Module):
    """Deep ResNet-like architecture."""
    def __init__(self, input_channels=3, num_classes=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        # Residual layers
        self.layer1 = ResidualBlock(64, 64)
        self.layer2 = ResidualBlock(64, 128)
        self.layer3 = ResidualBlock(128, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class SimpleAttentionBlock(nn.Module):
    """Simplified attention without einsum operations."""
    def __init__(self, embed_size):
        super().__init__()
        self.embed_size = embed_size
        self.attention_weights = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        # x shape: [batch, seq_len, embed_size]
        attn = torch.softmax(self.attention_weights(x), dim=1)
        out = x * attn  # Element-wise attention
        out = self.fc_out(out)
        return out


class TransformerLike(nn.Module):
    """Simplified transformer-like architecture."""
    def __init__(self, embed_size=128, seq_len=16, num_classes=1):
        super().__init__()
        self.embed_size = embed_size
        self.seq_len = seq_len

        # Simple embedding
        self.embedding = nn.Linear(embed_size, embed_size)

        # Attention blocks
        self.attention1 = SimpleAttentionBlock(embed_size)
        self.norm1 = nn.LayerNorm(embed_size)

        self.attention2 = SimpleAttentionBlock(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        # Feed forward
        self.ff = nn.Sequential(
            nn.Linear(embed_size, embed_size * 2),
            nn.ReLU(),
            nn.Linear(embed_size * 2, embed_size)
        )
        self.norm3 = nn.LayerNorm(embed_size)

        # Classification head
        self.classifier = nn.Linear(embed_size, num_classes)

    def forward(self, x):
        # x shape: [batch, seq_len, embed_size]
        x = self.embedding(x)

        # Attention block 1
        x = self.attention1(x)
        x = self.norm1(x)

        # Attention block 2
        x = self.attention2(x)
        x = self.norm2(x)

        # Feed forward
        x = self.ff(x)
        x = self.norm3(x)

        # Classification
        x = torch.mean(x, dim=1)
        x = self.classifier(x)

        return x


def make_very_deep_linear():
    """Very deep linear network with 10 layers."""
    layers = []
    prev_size = 32
    for _ in range(10):
        layers.extend([
            nn.Linear(prev_size, prev_size),
            nn.ReLU(),
            nn.BatchNorm1d(prev_size),
        ])
    layers.append(nn.Linear(prev_size, 1))
    return nn.Sequential(*layers)


def make_complex_cnn():
    """Complex CNN with multiple blocks."""
    return nn.Sequential(
        # First block
        nn.Conv2d(3, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2),

        # Second block
        nn.Conv2d(32, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),

        # Third block
        nn.Conv2d(64, 128, 3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((2, 2)),
        nn.Flatten(),
        nn.Linear(128 * 4, 256),
        nn.ReLU(),
        nn.Linear(256, 1)
    )


# =============================================================================
# Test Configuration
# =============================================================================

MODELS = {
    'DeepMLP': (lambda: DeepMLP(64, [128, 256, 128], 1), (2, 64)),
    'DeepResNet': (lambda: DeepResNet(3, 1), (1, 3, 64, 64)),
    'TransformerLike': (lambda: TransformerLike(64, 8, 1), (2, 8, 64)),
    'VeryDeepLinear': (make_very_deep_linear, (3, 32)),
    'ComplexCNN': (make_complex_cnn, (2, 3, 32, 32)),
}


if __name__ == "__main__":
    success = run_model_tests(MODELS, title="Complex Model Tests")
    sys.exit(0 if success else 1)
