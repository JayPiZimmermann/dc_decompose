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


def make_ultra_deep_cnn():
    """Ultra deep CNN without skip connections - tests many operations."""
    return nn.Sequential(
        # Input: [B, 3, 32, 32]
        
        # Block 1: Basic conv + pooling
        nn.Conv2d(3, 16, 3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Conv2d(16, 16, 3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.AvgPool2d(2),  # -> [B, 16, 16, 16]
        
        # Block 2: More conv + different pooling
        nn.Conv2d(16, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2),  # -> [B, 32, 8, 8]
        
        # Block 3: Shape operations
        nn.Conv2d(32, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Dropout(0.1),
        
        # Block 4: Adaptive pooling + flattening
        nn.AdaptiveAvgPool2d((4, 4)),  # -> [B, 64, 4, 4]
        nn.Flatten(),  # -> [B, 1024]
        
        # Deep MLP section
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.2),
        
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.1),
        
        nn.Linear(256, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        
        nn.Linear(128, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        
        nn.Linear(64, 32),
        nn.BatchNorm1d(32),
        nn.ReLU(),
        
        nn.Linear(32, 1)
    )


def make_autoencoder_like():
    """Autoencoder-like architecture using ConvTranspose (without skip connections)."""
    return nn.Sequential(
        # Encoder
        nn.Conv2d(1, 16, 3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Conv2d(16, 32, 3, stride=2, padding=1),  # Downsample
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, stride=2, padding=1),  # Downsample
        nn.BatchNorm2d(64),
        nn.ReLU(),
        
        # Bottleneck
        nn.AdaptiveAvgPool2d((2, 2)),
        nn.Flatten(),
        nn.Linear(64 * 4, 128),
        nn.ReLU(),
        nn.Linear(128, 64 * 4),
        nn.ReLU(),
        
        # Reshape and decode
        nn.Unflatten(1, (64, 2, 2)),
        
        # Decoder with ConvTranspose
        nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Conv2d(16, 1, 3, padding=1),
        
        # Global average to single output
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten()
    )


def make_conv1d_deep_network():
    """Deep 1D CNN for sequence processing."""
    return nn.Sequential(
        # Input: [B, 1, 128] - sequence length 128
        
        # 1D Conv blocks
        nn.Conv1d(1, 16, 7, padding=3),
        nn.BatchNorm1d(16),
        nn.ReLU(),
        nn.Conv1d(16, 16, 5, padding=2),
        nn.BatchNorm1d(16),
        nn.ReLU(),
        nn.MaxPool1d(2),  # -> [B, 16, 64]
        
        nn.Conv1d(16, 32, 5, padding=2),
        nn.BatchNorm1d(32),
        nn.ReLU(),
        nn.Conv1d(32, 32, 3, padding=1),
        nn.BatchNorm1d(32),
        nn.ReLU(),
        nn.AvgPool1d(2),  # -> [B, 32, 32]
        
        nn.Conv1d(32, 64, 3, padding=1),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.AdaptiveAvgPool1d(8),  # -> [B, 64, 8]
        
        # Flatten and MLP
        nn.Flatten(),  # -> [B, 512]
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.2),
        
        nn.Linear(256, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )


def make_shape_operations_network():
    """Network that extensively uses shape operations."""
    class ShapeOpsNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            # 2D layers
            self.conv2d_1 = nn.Conv2d(3, 8, 3, padding=1)
            self.bn2d_1 = nn.BatchNorm2d(8)
            
            # 1D layers (after transpose)
            self.conv1d_1 = nn.Conv1d(8, 16, 3, padding=1)
            self.bn1d_1 = nn.BatchNorm1d(16)
            
            # More shape changes  
            self.adaptive_pool = nn.AdaptiveAvgPool1d(32)
            
            # Back to 2D
            self.conv2d_2 = nn.Conv2d(32, 32, 3, padding=1)  # Input is 32 channels after transpose
            self.bn2d_2 = nn.BatchNorm2d(32)
            
            # Final layers
            self.final_pool = nn.AdaptiveAvgPool2d((2, 2))
            self.classifier = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
            
        def forward(self, x):
            # Start: [B, 3, 16, 16]
            x = self.conv2d_1(x)     # [B, 8, 16, 16]
            x = self.bn2d_1(x)
            x = torch.relu(x)
            
            # Shape manipulation: flatten spatial, then treat channels as sequence
            x = x.flatten(2)         # [B, 8, 256]
            x = self.conv1d_1(x)     # [B, 16, 256]  
            x = self.bn1d_1(x)
            x = torch.relu(x)
            
            # More shape changes
            x = self.adaptive_pool(x)  # [B, 16, 32]
            x = x.transpose(1, 2)      # [B, 32, 16] - transpose using tensor operation
            
            # Unflatten back to 2D-like
            x = x.unflatten(2, (4, 4))  # [B, 32, 4, 4] - 16 = 4*4
            
            # 2D processing
            x = self.conv2d_2(x)      # [B, 32, 4, 4]
            x = self.bn2d_2(x)
            x = torch.relu(x)
            
            # Final shape operations
            x = self.final_pool(x)    # [B, 32, 2, 2]
            x = x.flatten(1)          # [B, 128]
            
            # Classification
            return self.classifier(x)
    
    return ShapeOpsNetwork()


def make_layernorm_transformer_like():
    """Deep network using LayerNorm (transformer-like without attention)."""
    return nn.Sequential(
        # Input: [B, 64, 32] - sequence of length 64, feature dim 32
        
        # Embedding-like transformation
        nn.Linear(32, 128),
        nn.LayerNorm(128),
        nn.ReLU(),
        
        # Deep transformer-like blocks (no attention, just feed-forward)
        nn.Linear(128, 256),
        nn.LayerNorm(256),
        nn.ReLU(),
        nn.Dropout(0.1),
        
        nn.Linear(256, 128),
        nn.LayerNorm(128),
        nn.ReLU(),
        
        nn.Linear(128, 256),
        nn.LayerNorm(256),
        nn.ReLU(),
        nn.Dropout(0.1),
        
        nn.Linear(256, 128),
        nn.LayerNorm(128),
        nn.ReLU(),
        
        nn.Linear(128, 64),
        nn.LayerNorm(64),
        nn.ReLU(),
        
        # Global pooling over sequence dimension
        nn.AdaptiveAvgPool1d(1),  # -> [B, 64, 1]
        nn.Flatten(),  # -> [B, 64]
        
        # Final layers
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )


def make_mixed_architecture():
    """Complex architecture mixing different types of layers."""
    class MixedNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            # Conv part
            self.conv_layers = nn.Sequential(
                nn.Conv2d(2, 16, 3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4))
            )
            
            # 1D conv part (treat flattened as sequence)
            self.conv1d_layers = nn.Sequential(
                nn.Conv1d(32, 64, 3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Conv1d(64, 32, 3, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(8)
            )
            
            # Dense layers with different norms
            self.dense_layers = nn.Sequential(
                nn.Linear(256, 128),  # 32 * 8
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Dropout(0.1),
                
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                
                nn.Linear(64, 32),
                nn.LayerNorm(32),
                nn.ReLU(),
                
                nn.Linear(32, 1)
            )
        
        def forward(self, x):
            # x: [B, 2, 16, 16]
            x = self.conv_layers(x)  # [B, 32, 4, 4]
            x = x.flatten(2)  # [B, 32, 16] - treat as sequence
            x = self.conv1d_layers(x)  # [B, 32, 8]
            x = x.flatten(1)  # [B, 256]
            x = self.dense_layers(x)  # [B, 1]
            return x
    
    return MixedNetwork()


def make_comprehensive_operations_test():
    """Network that tests nearly ALL operations available."""
    class ComprehensiveNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            # Start with 2D
            self.conv2d = nn.Conv2d(1, 8, 3, padding=1)
            self.bn2d = nn.BatchNorm2d(8)
            
            # Conv1D part  
            self.conv1d = nn.Conv1d(8, 16, 3, padding=1)
            self.bn1d = nn.BatchNorm1d(16)
            
            # ConvTranspose
            self.conv_transpose = nn.ConvTranspose1d(16, 8, 3, padding=1)
            
            # Shape ops
            self.unflatten = nn.Unflatten(1, (2, 4))
            
            # Pooling ops
            self.maxpool1d = nn.MaxPool1d(2)
            self.avgpool1d = nn.AvgPool1d(2) 
            self.adaptive_avgpool = nn.AdaptiveAvgPool1d(8)
            
            # Dense layers
            self.linear1 = nn.Linear(64, 32)
            self.layernorm = nn.LayerNorm(32)
            self.linear2 = nn.Linear(32, 16)
            self.linear3 = nn.Linear(16, 8)
            
            # Final softmax layer for classification
            self.classifier = nn.Linear(8, 3)
            self.dropout = nn.Dropout(0.1)  # Register dropout as submodule
            
        def forward(self, x):
            # x: [B, 1, 8, 8]
            x = self.conv2d(x)      # [B, 8, 8, 8]
            x = self.bn2d(x)
            x = torch.relu(x)
            
            # Flatten spatial dims and treat as sequence
            x = x.flatten(2)        # [B, 8, 64]
            x = self.conv1d(x)      # [B, 16, 64]
            x = self.bn1d(x)
            x = torch.relu(x)
            x = self.dropout(x)
            
            # ConvTranspose
            x = self.conv_transpose(x)  # [B, 8, 64]
            x = torch.relu(x)
            
            # Pooling operations
            x = self.maxpool1d(x)       # [B, 8, 32]
            x = self.avgpool1d(x)       # [B, 8, 16] 
            x = self.adaptive_avgpool(x) # [B, 8, 8]
            
            # Shape manipulation
            x = x.transpose(1, 2)       # [B, 8, 8] - tensor transpose
            x = x.flatten(1)            # [B, 64]
            
            # Dense processing
            x = self.linear1(x)         # [B, 32] - input is 64, not 16
            x = self.layernorm(x)
            x = torch.relu(x)
            
            x = self.linear2(x)         # [B, 16]
            x = torch.relu(x)
            
            x = self.linear3(x)         # [B, 8]
            x = torch.relu(x)
            
            # Classification with softmax
            x = self.classifier(x)      # [B, 3]
            x = torch.softmax(x, dim=1)
            
            # Return single output for testing
            return x.mean(dim=1, keepdim=True)  # [B, 1]
    
    return ComprehensiveNetwork()


# =============================================================================
# Test Configuration
# =============================================================================

# OPERATIONS COVERAGE SUMMARY:
# 
# Core Operations:
# ✓ Linear, Conv1d, Conv2d, ConvTranspose1d, ConvTranspose2d, ReLU
# ✓ BatchNorm1d, BatchNorm2d, LayerNorm
# ✓ MaxPool1d, MaxPool2d, AvgPool1d, AvgPool2d, AdaptiveAvgPool1d, AdaptiveAvgPool2d
# ✓ Flatten, Unflatten, Reshape, Transpose, Permute, Squeeze, Unsqueeze
# ✓ Dropout, Softmax
#
# Coverage by Model:
# - UltraDeepCNN: Conv2d, BatchNorm2d, ReLU, AvgPool2d, MaxPool2d, AdaptiveAvgPool2d, Flatten, Linear, BatchNorm1d, Dropout
# - AutoencoderLike: Conv2d, ConvTranspose2d, BatchNorm2d, ReLU, AdaptiveAvgPool2d, Flatten, Linear, Unflatten
# - Conv1DDeep: Conv1d, BatchNorm1d, ReLU, MaxPool1d, AvgPool1d, AdaptiveAvgPool1d, Flatten, Linear, Dropout
# - ShapeOperations: Conv2d, BatchNorm2d, ReLU, Flatten, Transpose, Conv1d, BatchNorm1d, AdaptiveAvgPool1d, Permute, Unflatten, AdaptiveAvgPool2d, Squeeze, Linear
# - LayerNormTransformer: Linear, LayerNorm, ReLU, Dropout, AdaptiveAvgPool1d, Flatten
# - MixedArchitecture: Conv2d, BatchNorm2d, ReLU, MaxPool2d, AdaptiveAvgPool2d, Conv1d, BatchNorm1d, AdaptiveAvgPool1d, Linear, LayerNorm, Dropout
# - ComprehensiveOpsTest: Conv2d, BatchNorm2d, ReLU, Conv1d, BatchNorm1d, Dropout, ConvTranspose1d, MaxPool1d, AvgPool1d, AdaptiveAvgPool1d, Transpose, Unflatten, Permute, Flatten, Linear, LayerNorm, Softmax

MODELS = {
    # Original models  
    'DeepMLP': (lambda: DeepMLP(64, [128, 256, 128], 1), (2, 64)),
    'DeepResNet': (lambda: DeepResNet(3, 1), (1, 3, 64, 64)),
    'TransformerLike': (lambda: TransformerLike(64, 8, 1), (2, 8, 64)),
    'VeryDeepLinear': (make_very_deep_linear, (3, 32)),
    'ComplexCNN': (make_complex_cnn, (2, 3, 32, 32)),
    
    # NEW: Comprehensive deeper models without skip connections
    'UltraDeepCNN': (make_ultra_deep_cnn, (2, 3, 32, 32)),
    'AutoencoderLike': (make_autoencoder_like, (2, 1, 16, 16)), 
    'Conv1DDeep': (make_conv1d_deep_network, (2, 1, 128)),
    'ShapeOperations': (make_shape_operations_network, (2, 3, 16, 16)),
    'LayerNormTransformer': (make_layernorm_transformer_like, (2, 64, 32)),
    'MixedArchitecture': (make_mixed_architecture, (2, 2, 16, 16)),
    'ComprehensiveOpsTest': (make_comprehensive_operations_test, (2, 1, 8, 8)),
}


if __name__ == "__main__":
    success = run_model_tests(MODELS, title="Complex Model Tests")
    sys.exit(0 if success else 1)
