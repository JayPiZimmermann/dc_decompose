"""
Simple test script to verify stacked tensor DC decomposition works correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('/home/jakob/code/src/dc_decompose/Imaginative_Polytopes')

import torch
import torch.nn as nn
from typing import Dict, List
import importlib.util

from dc_decompose import HookDecomposer, InputMode, BackwardMode, ReLUMode

# Import the edgecase test models
try:
    spec = importlib.util.spec_from_file_location(
        "edgecase_test_models", 
        "/home/jakob/code/src/dc_decompose/Imaginative_Polytopes/dc_decomposition/test/edgecase_test_models.py"
    )
    edgecase_models = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(edgecase_models)
    EDGECASE_MODELS_AVAILABLE = True
except Exception as e:
    print(f"Warning: Could not load edgecase models: {e}")
    EDGECASE_MODELS_AVAILABLE = False


def test_model(model: nn.Module, input_tensor: torch.Tensor, model_name: str) -> Dict:
    """Test a single model and return results."""
    print(f"Testing {model_name}...")
    
    try:
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


def get_test_models():
    """Get models to test."""
    models = {}
    
    # Basic models
    models['LinearOnly'] = (nn.Linear(10, 5), torch.randn(3, 10))
    models['SimpleChain'] = (
        nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4)),
        torch.randn(2, 8)
    )
    
    models['ConvChain'] = (
        nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, 10)
        ),
        torch.randn(2, 3, 16, 16)
    )
    
    # Add edgecase models if available
    if EDGECASE_MODELS_AVAILABLE:
        try:
            models['ComplexNet'] = (edgecase_models.ComplexNet(), torch.randn(2, 2))
            models['ComplexNetDeeper'] = (edgecase_models.ComplexNetDeeper(), torch.randn(2, 2))
            models['MaxPoolHeavyModel'] = (edgecase_models.MaxPoolHeavyModel(), torch.randn(1, 3, 64, 64))
            models['DropoutHeavyModel'] = (edgecase_models.DropoutHeavyModel(), torch.randn(1, 3, 32, 32))
        except Exception as e:
            print(f"Warning: Could not create edgecase models: {e}")
    
    return models


def run_tests():
    """Run all tests."""
    print("DC Decomposition Simple Test Suite")
    print("=" * 50)
    
    models = get_test_models()
    results = {}
    
    for name, (model, input_tensor) in models.items():
        model.eval()  # Set to eval mode
        result = test_model(model, input_tensor, name)
        results[name] = result
        
        if result['success']:
            print(f"✅ {name}")
            print(f"   Reconstruction error: {result['reconstruction_error']:.2e}")
            print(f"   Hook bypass error: {result['hook_bypass_error']:.2e}")  
            print(f"   Layers: {result['num_layers']}")
            print(f"   Gradient shape: {result['gradient_shape']}")
        else:
            print(f"❌ {name}: {result['error']}")
        print()
    
    # Summary
    successful = sum(1 for r in results.values() if r['success'])
    total = len(results)
    
    print("=" * 50)
    print(f"SUMMARY: {successful}/{total} tests passed ({100*successful/total:.1f}%)")
    
    if successful == total:
        print("🎉 All tests passed!")
        
        # Show best and worst reconstruction errors
        recon_errors = [r['reconstruction_error'] for r in results.values() if r['success']]
        if recon_errors:
            print(f"Best reconstruction error: {min(recon_errors):.2e}")
            print(f"Worst reconstruction error: {max(recon_errors):.2e}")
            
        return True
    else:
        failed_models = [name for name, r in results.items() if not r['success']]
        print(f"❌ Failed models: {failed_models}")
        return False


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)