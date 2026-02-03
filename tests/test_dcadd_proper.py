"""
Test proper DCAdd usage with operand setup.

This demonstrates how to correctly use DCAdd modules by setting up
the second operand's decomposition before the forward pass.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from dc_decompose import HookDecomposer, InputMode, BackwardMode, ReLUMode, DCAdd


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


class ProperDCResidual(nn.Module):
    """Properly configured DCAdd residual connection."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.add = DCAdd()
        
    def forward(self, x):
        # Store identity for DCAdd
        identity = x
        
        # Process through convolutions
        out = self.conv1(x)
        out = torch.relu(out)
        out = self.conv2(out)
        
        # Setup DCAdd with proper decomposition of identity
        # For a proper implementation, we need to decompose identity
        identity_pos = F.relu(identity)
        identity_neg = F.relu(-identity)
        self.add.set_operand_decomposed(identity_pos, identity_neg)
        
        # Perform addition
        out = self.add(out, identity)
        return torch.relu(out)


class SimpleDCAdd(nn.Module):
    """Simple test of DCAdd functionality."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(64, 64)
        self.add = DCAdd()
        
    def forward(self, x):
        # Transform input
        y = self.linear(x)
        
        # Add x back (residual connection)
        # Set up operand decomposition
        x_pos = F.relu(x)
        x_neg = F.relu(-x)
        self.add.set_operand_decomposed(x_pos, x_neg)
        
        # Perform addition
        result = self.add(y, x)
        return result


def get_proper_dc_models():
    """Get properly configured DC models."""
    models = {}
    
    models['SimpleDCAdd'] = (SimpleDCAdd(), torch.randn(2, 64))
    models['ProperDCResidual'] = (ProperDCResidual(), torch.randn(1, 16, 8, 8))
    
    return models


def run_proper_dc_tests():
    """Run proper DC tests."""
    print("🧪 DC Decomposition - Proper DCAdd Usage Tests")
    print("=" * 60)
    
    models = get_proper_dc_models()
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
    print(f"PROPER DC USAGE SUMMARY: {successful}/{total} tests passed ({100*successful/total:.1f}%)")
    
    if successful == total:
        print("🎉 All proper DC usage tests passed!")
        
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
    success = run_proper_dc_tests()
    sys.exit(0 if success else 1)