"""
Test residual connections and tensor addition operations.
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
        
        return {
            'success': True,
            'reconstruction_error': reconstruction_error,
            'num_layers': len(decomposer.layer_order),
            'gradient_shape': gradient_shape,
            'layers': decomposer.layer_order
        }
        
    except Exception as e:
        return {'success': False, 'error': str(e)}


class SimpleResidual(nn.Module):
    """Very simple residual connection test."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = torch.relu(out)
        out = self.conv2(out)
        out = out + identity  # Residual connection
        return torch.relu(out)


class LinearResidual(nn.Module):
    """Linear residual connection."""
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(64, 64)
        self.linear2 = nn.Linear(64, 64)
        
    def forward(self, x):
        identity = x
        out = self.linear1(x)
        out = torch.relu(out)
        out = self.linear2(out)
        out = out + identity  # Residual connection
        return torch.relu(out)


class MultipleAdditions(nn.Module):
    """Multiple tensor additions."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(8, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 8, 3, padding=1)
        self.conv3 = nn.Conv2d(8, 8, 3, padding=1)
        
    def forward(self, x):
        branch1 = self.conv1(x)
        branch2 = self.conv2(x)
        branch3 = self.conv3(x)
        
        # Multiple additions
        out = branch1 + branch2 + branch3
        return torch.relu(out)


class NestedResidual(nn.Module):
    """Nested residual connections."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)
        
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
        
        return out3


def get_residual_models():
    """Get residual connection test models."""
    models = {}
    
    models['SimpleResidual'] = (SimpleResidual(), torch.randn(1, 16, 8, 8))
    models['LinearResidual'] = (LinearResidual(), torch.randn(2, 64))
    models['MultipleAdditions'] = (MultipleAdditions(), torch.randn(1, 8, 16, 16))
    models['NestedResidual'] = (NestedResidual(), torch.randn(1, 32, 8, 8))
    
    return models


def run_residual_tests():
    """Run residual connection tests."""
    print("🧪 DC Decomposition - Residual Connection Tests")
    print("=" * 60)
    
    models = get_residual_models()
    results = {}
    
    for name, (model, input_tensor) in models.items():
        result = test_model(model, input_tensor, name)
        results[name] = result
        
        if result['success']:
            print(f"✅ {name}")
            print(f"   Reconstruction error: {result['reconstruction_error']:.2e}")
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
    print(f"RESIDUAL CONNECTION SUMMARY: {successful}/{total} tests passed ({100*successful/total:.1f}%)")
    
    if successful == total:
        print("🎉 All residual connection tests passed!")
        return True
    else:
        failed_models = [name for name, r in results.items() if not r['success']]
        print(f"❌ Failed models: {failed_models}")
        
        # Show error details
        for name in failed_models:
            print(f"   {name}: {results[name]['error']}")
        
        # Instructions for fixing
        if failed_models:
            print("\n🔧 To fix residual connections:")
            print("   1. Implement tensor addition operator hook")
            print("   2. Handle multiple input branches properly") 
            print("   3. Ensure gradient accumulation for shared tensors")
        
        return False


if __name__ == "__main__":
    success = run_residual_tests()
    sys.exit(0 if success else 1)