"""
Test basic layer functionality - Linear, Conv, ReLU, BatchNorm, etc.
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


def get_basic_layer_models():
    """Get basic layer test models."""
    models = {}
    
    # Single layers
    models['Linear'] = (nn.Linear(10, 5), torch.randn(3, 10))
    models['Conv2d'] = (nn.Conv2d(3, 16, 3), torch.randn(2, 3, 8, 8))
    models['ReLU'] = (nn.ReLU(), torch.randn(2, 10))
    models['BatchNorm1d'] = (nn.BatchNorm1d(10), torch.randn(4, 10))
    models['BatchNorm2d'] = (nn.BatchNorm2d(8), torch.randn(2, 8, 16, 16))
    
    # Pooling layers
    models['MaxPool2d'] = (nn.MaxPool2d(2), torch.randn(2, 4, 16, 16))
    models['AvgPool2d'] = (nn.AvgPool2d(2), torch.randn(2, 4, 16, 16))
    models['AdaptiveAvgPool2d'] = (nn.AdaptiveAvgPool2d((4, 4)), torch.randn(2, 8, 16, 16))
    
    # Utility layers
    models['Flatten'] = (nn.Flatten(), torch.randn(2, 4, 8, 8))
    models['Dropout'] = (nn.Dropout(0.5), torch.randn(2, 10))
    models['Identity'] = (nn.Identity(), torch.randn(2, 10))
    
    # Simple chains
    models['LinearChain'] = (
        nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 4)
        ),
        torch.randn(2, 8)
    )
    
    models['ConvChain'] = (
        nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, 10)
        ),
        torch.randn(2, 3, 16, 16)
    )
    
    models['MixedChain'] = (
        nn.Sequential(
            nn.Linear(20, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(50, 10),
            nn.ReLU()
        ),
        torch.randn(4, 20)
    )
    
    return models


def run_basic_layer_tests():
    """Run all basic layer tests."""
    print("🧪 DC Decomposition - Basic Layer Tests")
    print("=" * 50)
    
    models = get_basic_layer_models()
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
    
    print("=" * 50)
    print(f"BASIC LAYER SUMMARY: {successful}/{total} tests passed ({100*successful/total:.1f}%)")
    
    if successful == total:
        print("🎉 All basic layer tests passed!")
        
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
    success = run_basic_layer_tests()
    sys.exit(0 if success else 1)