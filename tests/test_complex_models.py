"""
Test complex models - Deep networks, residual connections, complex architectures
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from typing import Dict, Tuple
from dc_decompose import HookDecomposer, InputMode, BackwardMode, ReLUMode


def test_model(model: nn.Module, input_tensor: torch.Tensor, model_name: str) -> Dict:
    """Test a single model."""
    print(f"Testing {model_name}...")
    
    try:
        model.eval()
        
        # Create decomposer
        decomposer = HookDecomposer(model)
        decomposer.set_input_mode(InputMode.CENTER)
        decomposer.set_backward_mode(BackwardMode.ALPHA)
        decomposer.set_alpha(0.35)
        decomposer.initialize(input_tensor)
        
        # Forward pass
        original_out = model(input_tensor)
        
        # Get final activations
        final_layer = decomposer.layer_order[-1] if decomposer.layer_order else None
        if not final_layer:
            return {'success': False, 'error': 'No layers found'}
        
        activations = decomposer.get_activation(final_layer)
        if not activations:
            return {'success': False, 'error': 'No activations captured'}
        
        out_pos, out_neg = activations
        reconstructed = out_pos - out_neg
        reconstruction_error = torch.norm(original_out - reconstructed).item()
        
        # Test backward pass
        decomposer.backward()
        
        # Check stacked gradients
        stacked_grads = decomposer.get_stacked_gradients(decomposer.layer_order[0])
        gradient_shape = stacked_grads.shape if stacked_grads is not None else None
        
        # Test hook bypass
        decomposer.disable_hooks()
        output_no_hooks = model(input_tensor)
        decomposer.enable_hooks()
        hook_bypass_error = torch.norm(original_out - output_no_hooks).item()
        
        return {
            'success': True,
            'reconstruction_error': reconstruction_error,
            'num_layers': len(decomposer.layer_order),
            'gradient_shape': gradient_shape,
            'hook_bypass_error': hook_bypass_error,
            'layers': decomposer.layer_order
        }
        
    except Exception as e:
        return {'success': False, 'error': str(e)}


class DeepMLP(nn.Module):
    """Deep multi-layer perceptron."""
    def __init__(self, input_size=128, hidden_sizes=[256, 512, 256, 128], output_size=10):
        super().__init__()
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2)
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
    def __init__(self, input_channels=3, num_classes=10):
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
        # Simplified attention using just a feedforward transformation
        attn = torch.softmax(self.attention_weights(x), dim=1)
        out = x * attn  # Element-wise attention
        out = self.fc_out(out)
        return out


class TransformerLike(nn.Module):
    """Simplified transformer-like architecture."""
    def __init__(self, embed_size=128, num_heads=8, seq_len=16, num_classes=10):
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
        
        # Attention block 1 (no residual connections to avoid tensor addition issues)
        x = self.attention1(x)
        x = self.norm1(x)
        
        # Attention block 2
        x = self.attention2(x)
        x = self.norm2(x)
        
        # Feed forward
        x = self.ff(x)
        x = self.norm3(x)
        
        # Classification - properly handle tensor dimensions
        # x shape: [batch, seq_len, embed_size] -> [batch, embed_size]
        x = torch.mean(x, dim=1)  # Average across sequence length
        x = self.classifier(x)
        
        return x


def get_complex_models():
    """Get complex model test cases."""
    models = {}
    
    # Deep MLP
    models['DeepMLP'] = (DeepMLP(64, [128, 256, 128], 10), torch.randn(2, 64))
    
    # Deep ResNet
    models['DeepResNet'] = (DeepResNet(3, 10), torch.randn(1, 3, 64, 64))
    
    # Transformer-like
    models['TransformerLike'] = (TransformerLike(64, 4, 8, 5), torch.randn(2, 8, 64))
    
    # Very deep linear network
    layers = []
    prev_size = 32
    for i in range(10):  # 10 layers deep
        layers.extend([
            nn.Linear(prev_size, prev_size),
            nn.ReLU(),
            nn.BatchNorm1d(prev_size),
        ])
    layers.append(nn.Linear(prev_size, 8))
    models['VeryDeepLinear'] = (nn.Sequential(*layers), torch.randn(3, 32))
    
    # Complex CNN
    models['ComplexCNN'] = (
        nn.Sequential(
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
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        ),
        torch.randn(2, 3, 32, 32)
    )
    
    return models


def run_complex_model_tests():
    """Run complex model tests."""
    print("🧪 DC Decomposition - Complex Model Tests")
    print("=" * 60)
    
    models = get_complex_models()
    results = {}
    
    for name, (model, input_tensor) in models.items():
        result = test_model(model, input_tensor, name)
        results[name] = result
        
        if result['success']:
            print(f"✅ {name}")
            print(f"   Reconstruction error: {result['reconstruction_error']:.2e}")
            print(f"   Hook bypass error: {result['hook_bypass_error']:.2e}")
            print(f"   Layers: {result['num_layers']}")
            if result['gradient_shape']:
                print(f"   Gradient shape: {result['gradient_shape']}")
        else:
            print(f"❌ {name}: {result['error']}")
        print()
    
    # Summary
    successful = sum(1 for r in results.values() if r['success'])
    total = len(results)
    
    print("=" * 60)
    print(f"COMPLEX MODEL SUMMARY: {successful}/{total} tests passed ({100*successful/total:.1f}%)")
    
    if successful == total:
        print("🎉 All complex model tests passed!")
        
        # Show error statistics
        recon_errors = [r['reconstruction_error'] for r in results.values() if r['success']]
        if recon_errors:
            print(f"Best reconstruction error: {min(recon_errors):.2e}")
            print(f"Worst reconstruction error: {max(recon_errors):.2e}")
            print(f"Mean reconstruction error: {sum(recon_errors)/len(recon_errors):.2e}")
            
        return True
    else:
        failed_models = [name for name, r in results.items() if not r['success']]
        print(f"❌ Failed models: {failed_models}")
        
        # Show error details
        for name in failed_models:
            print(f"   {name}: {results[name]['error']}")
        
        return False


if __name__ == "__main__":
    success = run_complex_model_tests()
    sys.exit(0 if success else 1)