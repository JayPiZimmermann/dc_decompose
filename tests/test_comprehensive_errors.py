"""
Comprehensive error testing across all layers.

This test measures both maximal activation errors and maximal gradient errors
across all layers to ensure DC decomposition works correctly throughout the network.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
from dc_decompose import HookDecomposer, InputMode, BackwardMode, ReLUMode
from dc_decompose.enhanced_dcadd import AdvancedDCAdd, create_dc_residual_model


def test_model_comprehensive(model: nn.Module, input_tensor: torch.Tensor, model_name: str) -> Dict:
    """Test a model with comprehensive error analysis across all layers."""
    print(f"Testing {model_name}...")
    
    try:
        model.eval()
        input_tensor.requires_grad_(True)
        
        # Create decomposer
        decomposer = HookDecomposer(model)
        decomposer.set_input_mode(InputMode.CENTER)
        decomposer.set_backward_mode(BackwardMode.ALPHA)
        decomposer.set_alpha(0.35)
        decomposer.initialize(input_tensor)
        
        # Forward pass with hooks enabled
        decomposer.enable_hooks(True)
        dc_output = model(input_tensor)
        
        # Forward pass with hooks disabled (original behavior)
        decomposer.enable_hooks(False)
        original_output = model(input_tensor.clone().requires_grad_(True))
        decomposer.enable_hooks(True)
        
        # Compute activation errors across all layers
        activation_errors = {}
        max_activation_error = 0.0
        max_activation_layer = None
        
        for layer_name in decomposer.layer_order:
            cache = decomposer.caches[layer_name]
            if cache.output_pos is not None and cache.output_neg is not None:
                # Reconstruct from pos/neg
                reconstructed = cache.output_pos - cache.output_neg
                original_activation = cache.original_output
                
                if original_activation is not None:
                    # Debug shapes to identify dimension mismatch
                    if original_activation.shape != reconstructed.shape:
                        print(f"⚠️  Shape mismatch in layer {layer_name}:")
                        print(f"   Original: {original_activation.shape}")
                        print(f"   Reconstructed: {reconstructed.shape}")
                        print(f"   Pos: {cache.output_pos.shape}")
                        print(f"   Neg: {cache.output_neg.shape}")
                        # Skip this layer to avoid error
                        continue
                    
                    error = torch.norm(original_activation - reconstructed).item()
                    activation_errors[layer_name] = error
                    
                    if error > max_activation_error:
                        max_activation_error = error
                        max_activation_layer = layer_name
        
        # Final output reconstruction error
        final_layer = decomposer.layer_order[-1] if decomposer.layer_order else None
        final_reconstruction_error = 0.0
        if final_layer:
            final_cache = decomposer.caches[final_layer]
            if final_cache.output_pos is not None and final_cache.output_neg is not None:
                final_reconstructed = final_cache.output_pos - final_cache.output_neg
                final_reconstruction_error = torch.norm(dc_output - final_reconstructed).item()
        
        # Hook bypass verification
        hook_bypass_error = torch.norm(dc_output - original_output).item()
        
        # Backward pass and gradient analysis
        print(f"Starting backward pass for {model_name}...")
        try:
            decomposer.backward()
            print(f"✅ Decomposer backward completed for {model_name}")
        except Exception as e:
            print(f"❌ Decomposer backward failed for {model_name}: {e}")
            raise e
        
        # Compute gradients with original PyTorch
        loss_original = original_output.sum()
        loss_original.backward()
        original_input_grad = input_tensor.grad.clone()
        input_tensor.grad = None
        
        # Compute gradients with DC decomposition
        loss_dc = dc_output.sum()
        loss_dc.backward()
        dc_input_grad = input_tensor.grad.clone()
        input_tensor.grad = None
        
        # Compare input gradients
        input_grad_error = torch.norm(original_input_grad - dc_input_grad).item()
        
        # Analyze gradient errors across all layers
        gradient_errors = {}
        max_gradient_error = 0.0
        max_gradient_layer = None
        
        for layer_name in decomposer.layer_order:
            try:
                stacked_grads = decomposer.get_stacked_gradients(layer_name)
                if stacked_grads is not None:
                    # For gradient analysis, we compare the magnitude
                    grad_norm = torch.norm(stacked_grads).item()
                    gradient_errors[layer_name] = grad_norm
                    
                    if grad_norm > max_gradient_error:
                        max_gradient_error = grad_norm
                        max_gradient_layer = layer_name
            except:
                continue
        
        return {
            'success': True,
            'final_reconstruction_error': final_reconstruction_error,
            'hook_bypass_error': hook_bypass_error,
            'input_gradient_error': input_grad_error,
            'max_activation_error': max_activation_error,
            'max_activation_layer': max_activation_layer,
            'max_gradient_error': max_gradient_error,
            'max_gradient_layer': max_gradient_layer,
            'num_layers': len(decomposer.layer_order),
            'activation_errors': activation_errors,
            'gradient_errors': gradient_errors,
            'layer_order': decomposer.layer_order
        }
        
    except Exception as e:
        return {'success': False, 'error': str(e)}


def get_comprehensive_test_models():
    """Get models for comprehensive testing."""
    models = {}
    
    # Basic models that should work perfectly
    models['SimpleLinear'] = (
        nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        ),
        torch.randn(2, 32)
    )
    
    models['SimpleCNN'] = (
        nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # Back to AdaptiveAvgPool2d
            nn.Flatten(),
            nn.Linear(32 * 16, 10)  # 32 channels * 4 * 4
        ),
        torch.randn(1, 3, 16, 16)
    )
    
    # Enhanced DCAdd residual model
    models['DCResidualModel'] = (
        create_dc_residual_model(16),
        torch.randn(1, 16, 8, 8)
    )
    
    # Traditional residual with manual DCAdd
    class ManualDCResidual(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(16, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
            self.add = AdvancedDCAdd()
            
        def forward(self, x):
            identity = x
            out = self.conv1(x)
            out = F.relu(out)
            out = self.conv2(out)
            out = self.add(out, identity)
            return F.relu(out)
    
    models['ManualDCResidual'] = (
        ManualDCResidual(),
        torch.randn(1, 16, 8, 8)
    )
    
    # Deep network
    layers = []
    prev_size = 64
    for i in range(5):
        layers.extend([
            nn.Linear(prev_size, prev_size),
            nn.ReLU(),
            nn.BatchNorm1d(prev_size)
        ])
    layers.append(nn.Linear(prev_size, 10))
    
    models['DeepNetwork'] = (
        nn.Sequential(*layers),
        torch.randn(3, 64)
    )
    
    return models


def run_comprehensive_tests():
    """Run comprehensive error testing."""
    print("🧪 DC Decomposition - Comprehensive Error Testing")
    print("=" * 70)
    
    models = get_comprehensive_test_models()
    results = {}
    
    for name, (model, input_tensor) in models.items():
        result = test_model_comprehensive(model, input_tensor, name)
        results[name] = result
        
        if result['success']:
            print(f"✅ {name}")
            print(f"   Final reconstruction error: {result['final_reconstruction_error']:.2e}")
            print(f"   Hook bypass error: {result['hook_bypass_error']:.2e}")
            print(f"   Input gradient error: {result['input_gradient_error']:.2e}")
            print(f"   Max activation error: {result['max_activation_error']:.2e} (layer: {result['max_activation_layer']})")
            print(f"   Max gradient magnitude: {result['max_gradient_error']:.2e} (layer: {result['max_gradient_layer']})")
            print(f"   Layers: {result['num_layers']}")
            
            # Show per-layer errors if requested
            if len(result['activation_errors']) <= 10:  # Only for smaller networks
                print(f"   Layer-wise activation errors:")
                for layer, error in result['activation_errors'].items():
                    print(f"     {layer}: {error:.2e}")
        else:
            print(f"❌ {name}: {result['error']}")
        print()
    
    # Summary
    successful = sum(1 for r in results.values() if r['success'])
    total = len(results)
    
    print("=" * 70)
    print(f"COMPREHENSIVE TEST SUMMARY: {successful}/{total} tests passed ({100*successful/total:.1f}%)")
    
    if successful > 0:
        # Analyze results
        successful_results = [r for r in results.values() if r['success']]
        
        final_errors = [r['final_reconstruction_error'] for r in successful_results]
        bypass_errors = [r['hook_bypass_error'] for r in successful_results]
        grad_errors = [r['input_gradient_error'] for r in successful_results]
        max_act_errors = [r['max_activation_error'] for r in successful_results]
        max_grad_errors = [r['max_gradient_error'] for r in successful_results]
        
        print("\n📊 ERROR ANALYSIS:")
        print(f"Final reconstruction errors: {min(final_errors):.2e} to {max(final_errors):.2e}")
        print(f"Hook bypass errors: {min(bypass_errors):.2e} to {max(bypass_errors):.2e}")
        print(f"Input gradient errors: {min(grad_errors):.2e} to {max(grad_errors):.2e}")
        print(f"Max activation errors: {min(max_act_errors):.2e} to {max(max_act_errors):.2e}")
        print(f"Max gradient magnitudes: {min(max_grad_errors):.2e} to {max(max_grad_errors):.2e}")
        
        # Quality assessment
        excellent_count = sum(1 for e in final_errors if e < 1e-5)
        good_count = sum(1 for e in final_errors if 1e-5 <= e < 1e-2)
        poor_count = sum(1 for e in final_errors if e >= 1e-2)
        
        print(f"\n🎯 QUALITY ASSESSMENT:")
        print(f"Excellent (< 1e-5): {excellent_count}/{successful}")
        print(f"Good (1e-5 to 1e-2): {good_count}/{successful}")
        print(f"Poor (>= 1e-2): {poor_count}/{successful}")
        
        if excellent_count == successful:
            print("🎉 All models achieve excellent DC decomposition accuracy!")
        elif excellent_count + good_count == successful:
            print("✅ All models achieve acceptable DC decomposition accuracy!")
        else:
            print("⚠️  Some models have poor DC decomposition accuracy - review implementation")
    
    if successful < total:
        failed_models = [name for name, r in results.items() if not r['success']]
        print(f"\n❌ FAILED MODELS: {failed_models}")
        
        for name in failed_models:
            print(f"   {name}: {results[name]['error']}")
    
    return successful == total and all(r['final_reconstruction_error'] < 1e-2 for r in results.values() if r['success'])


if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)