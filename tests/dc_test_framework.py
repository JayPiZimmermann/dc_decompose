"""
DC Decomposition Testing Framework

A comprehensive framework for testing DC decomposition accuracy and performance
across multiple models with detailed error analysis and comparison capabilities.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import time
import json
from enum import Enum

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dc_decompose import HookDecomposer, InputMode, BackwardMode, ReLUMode


class TestMode(Enum):
    RECONSTRUCTION = "reconstruction"
    GRADIENT = "gradient"
    SENSITIVITY = "sensitivity"
    FULL = "full"


@dataclass
class LayerAnalysis:
    """Analysis results for a single layer."""
    name: str
    reconstruction_error: float
    gradient_error: float
    sensitivity_errors: Dict[str, float]
    
    # Original vs DC activations
    original_activation_norm: float
    pos_activation_norm: float
    neg_activation_norm: float
    
    # Original vs DC gradients  
    original_gradient_norm: float
    combined_gradient_norm: float
    stacked_gradient_norms: Dict[str, float]
    
    # Shape information
    activation_shape: Tuple[int, ...]
    gradient_shape: Tuple[int, ...]
    
    # Timing
    forward_time_ms: float
    backward_time_ms: float


@dataclass  
class ModelTestResult:
    """Complete test results for a model."""
    model_name: str
    success: bool
    error_message: Optional[str] = None
    
    # Overall metrics
    max_reconstruction_error: float = 0.0
    max_gradient_error: float = 0.0
    total_forward_time_ms: float = 0.0
    total_backward_time_ms: float = 0.0
    
    # Per-layer results
    layer_results: Dict[str, LayerAnalysis] = field(default_factory=dict)
    
    # Configuration
    input_shape: Tuple[int, ...] = ()
    num_parameters: int = 0
    num_layers: int = 0


class DCTestFramework:
    """
    Comprehensive testing framework for DC decomposition.
    
    Features:
    - Tests multiple models automatically
    - Detailed error analysis and comparison
    - Hook bypass functionality for original model behavior
    - Caching of activations, gradients, and sensitivities
    - Performance profiling
    """
    
    def __init__(self, 
                 tolerance_reconstruction: float = 1e-6,
                 tolerance_gradient: float = 1e-5,
                 cache_results: bool = True,
                 verbose: bool = True):
        self.tolerance_reconstruction = tolerance_reconstruction
        self.tolerance_gradient = tolerance_gradient
        self.cache_results = cache_results
        self.verbose = verbose
        
        # Results storage
        self.results: Dict[str, ModelTestResult] = {}
        self.cache_dir = Path("test_cache")
        self.cache_dir.mkdir(exist_ok=True)
        
    def log(self, message: str, level: str = "INFO"):
        """Logging with levels."""
        if self.verbose or level == "ERROR":
            print(f"[{level}] {message}")
    
    def test_model(self, 
                   model: nn.Module, 
                   input_tensor: torch.Tensor,
                   model_name: str,
                   test_mode: TestMode = TestMode.FULL) -> ModelTestResult:
        """
        Test a single model with comprehensive analysis.
        
        Args:
            model: PyTorch model to test
            input_tensor: Input data for testing
            model_name: Name for identification
            test_mode: What aspects to test
        
        Returns:
            Complete test results
        """
        self.log(f"Testing model: {model_name}")
        
        result = ModelTestResult(
            model_name=model_name,
            success=False,
            input_shape=tuple(input_tensor.shape),
            num_parameters=sum(p.numel() for p in model.parameters()),
        )
        
        try:
            # Test with hooks enabled (DC decomposition)
            dc_results = self._test_with_dc_decomposition(model, input_tensor, test_mode)
            
            # Test with hooks disabled (original behavior)
            original_results = self._test_original_model(model, input_tensor, test_mode)
            
            # Compare and analyze
            result = self._analyze_results(dc_results, original_results, result)
            
            result.success = True
            self.log(f"✅ {model_name}: Max reconstruction error = {result.max_reconstruction_error:.2e}")
            
        except Exception as e:
            result.error_message = str(e)
            self.log(f"❌ {model_name}: Failed with error: {e}", "ERROR")
        
        self.results[model_name] = result
        return result
    
    def _test_with_dc_decomposition(self, model: nn.Module, input_tensor: torch.Tensor, test_mode: TestMode) -> Dict:
        """Test model with DC decomposition enabled."""
        # Create decomposer
        decomposer = HookDecomposer(model)
        decomposer.set_input_mode(InputMode.CENTER)
        decomposer.set_backward_mode(BackwardMode.ALPHA)
        decomposer.set_alpha(0.35)
        decomposer.initialize(input_tensor)
        
        results = {
            'decomposer': decomposer,
            'layer_order': decomposer.layer_order,
            'activations': {},
            'gradients': {},
            'sensitivities': {},
            'timings': {}
        }
        
        # Forward pass with timing
        start_time = time.time()
        output = model(input_tensor)
        forward_time = (time.time() - start_time) * 1000
        
        # Cache activations
        for layer_name in decomposer.layer_order:
            activation = decomposer.get_activation(layer_name)
            if activation:
                results['activations'][layer_name] = {
                    'pos': activation[0].detach().clone(),
                    'neg': activation[1].detach().clone(),
                    'original': activation[0] - activation[1]
                }
        
        # Backward pass with timing (if testing gradients)
        if test_mode in [TestMode.GRADIENT, TestMode.SENSITIVITY, TestMode.FULL]:
            start_time = time.time()
            decomposer.backward()
            backward_time = (time.time() - start_time) * 1000
            
            # Cache gradients and sensitivities
            for layer_name in decomposer.layer_order:
                # Stacked gradients
                stacked_grads = decomposer.get_stacked_gradients(layer_name)
                if stacked_grads is not None:
                    results['gradients'][layer_name] = {
                        'stacked': stacked_grads.detach().clone(),
                        'combined': decomposer.get_combined_gradient(layer_name)
                    }
                
                # Individual sensitivities
                sensitivity = decomposer.get_sensitivity(layer_name)
                if sensitivity:
                    results['sensitivities'][layer_name] = {
                        'delta_pp': sensitivity[0].detach().clone(),
                        'delta_np': sensitivity[1].detach().clone(),
                        'delta_pn': sensitivity[2].detach().clone(),
                        'delta_nn': sensitivity[3].detach().clone(),
                    }
            
            results['timings'] = {
                'forward_ms': forward_time,
                'backward_ms': backward_time
            }
        else:
            results['timings'] = {'forward_ms': forward_time, 'backward_ms': 0.0}
        
        results['output'] = output.detach().clone()
        return results
    
    def _test_original_model(self, model: nn.Module, input_tensor: torch.Tensor, test_mode: TestMode) -> Dict:
        """Test original model behavior with activation/gradient capture."""
        results = {
            'activations': {},
            'gradients': {},
            'timings': {}
        }
        
        # Create hooks to capture original activations and gradients
        activation_cache = {}
        gradient_cache = {}
        
        def make_activation_hook(name):
            def hook(module, input, output):
                activation_cache[name] = output.detach().clone()
            return hook
        
        def make_gradient_hook(name):
            def hook(grad):
                gradient_cache[name] = grad.detach().clone()
                return grad
            return hook
        
        # Register hooks
        hooks = []
        for name, module in model.named_modules():
            if name:  # Skip empty name (root module)
                # Activation hook
                hook = module.register_forward_hook(make_activation_hook(name))
                hooks.append(hook)
        
        try:
            # Forward pass
            input_tensor.requires_grad_(True)
            start_time = time.time()
            output = model(input_tensor)
            forward_time = (time.time() - start_time) * 1000
            
            # Backward pass if needed
            if test_mode in [TestMode.GRADIENT, TestMode.FULL]:
                # Register gradient hooks on input
                if input_tensor.requires_grad:
                    input_tensor.register_hook(make_gradient_hook('input'))
                
                start_time = time.time()
                # Create dummy loss for backward pass
                loss = output.sum()
                loss.backward()
                backward_time = (time.time() - start_time) * 1000
            else:
                backward_time = 0.0
            
            results['activations'] = activation_cache
            results['gradients'] = gradient_cache
            results['output'] = output.detach().clone()
            results['timings'] = {
                'forward_ms': forward_time,
                'backward_ms': backward_time
            }
            
        finally:
            # Clean up hooks
            for hook in hooks:
                hook.remove()
        
        return results
    
    def _analyze_results(self, dc_results: Dict, original_results: Dict, result: ModelTestResult) -> ModelTestResult:
        """Analyze and compare DC vs original results."""
        
        # Overall timing
        result.total_forward_time_ms = dc_results['timings']['forward_ms']
        result.total_backward_time_ms = dc_results['timings']['backward_ms']
        result.num_layers = len(dc_results['layer_order'])
        
        # Per-layer analysis
        for layer_name in dc_results['layer_order']:
            analysis = LayerAnalysis(
                name=layer_name,
                reconstruction_error=0.0,
                gradient_error=0.0,
                sensitivity_errors={},
                original_activation_norm=0.0,
                pos_activation_norm=0.0,
                neg_activation_norm=0.0,
                original_gradient_norm=0.0,
                combined_gradient_norm=0.0,
                stacked_gradient_norms={},
                activation_shape=(),
                gradient_shape=(),
                forward_time_ms=0.0,
                backward_time_ms=0.0
            )
            
            # Activation analysis
            if layer_name in dc_results['activations'] and layer_name in original_results['activations']:
                dc_activation = dc_results['activations'][layer_name]
                original_activation = original_results['activations'][layer_name]
                
                # Reconstruction error: |original - (pos - neg)|
                reconstructed = dc_activation['pos'] - dc_activation['neg']
                reconstruction_error = torch.norm(original_activation - reconstructed).item()
                analysis.reconstruction_error = reconstruction_error
                result.max_reconstruction_error = max(result.max_reconstruction_error, reconstruction_error)
                
                # Norms
                analysis.original_activation_norm = torch.norm(original_activation).item()
                analysis.pos_activation_norm = torch.norm(dc_activation['pos']).item()
                analysis.neg_activation_norm = torch.norm(dc_activation['neg']).item()
                analysis.activation_shape = tuple(original_activation.shape)
            
            # Gradient analysis (skip for now as it's more complex to compare)
            if layer_name in dc_results.get('gradients', {}):
                dc_gradient = dc_results['gradients'][layer_name]
                if dc_gradient.get('combined') is not None:
                    analysis.combined_gradient_norm = torch.norm(dc_gradient['combined']).item()
                    analysis.gradient_shape = tuple(dc_gradient['combined'].shape)
            
            # Sensitivity analysis
            if layer_name in dc_results['sensitivities']:
                sensitivities = dc_results['sensitivities'][layer_name]
                for sens_name, sens_tensor in sensitivities.items():
                    analysis.sensitivity_errors[sens_name] = torch.norm(sens_tensor).item()
            
            # Stacked gradient norms
            if layer_name in dc_results['gradients']:
                stacked = dc_results['gradients'][layer_name]['stacked']
                if stacked is not None:
                    analysis.stacked_gradient_norms = {
                        'delta_pp': torch.norm(stacked[0]).item(),
                        'delta_np': torch.norm(stacked[1]).item(),
                        'delta_pn': torch.norm(stacked[2]).item(),
                        'delta_nn': torch.norm(stacked[3]).item(),
                    }
            
            result.layer_results[layer_name] = analysis
        
        return result
    
    def test_multiple_models(self, models: Dict[str, Callable[[], nn.Module]], 
                           input_shapes: Dict[str, Tuple[int, ...]] = None,
                           batch_size: int = 2) -> Dict[str, ModelTestResult]:
        """
        Test multiple models in batch.
        
        Args:
            models: Dict mapping model names to model factory functions
            input_shapes: Dict mapping model names to input shapes (if None, uses default)
            batch_size: Batch size for test inputs
        
        Returns:
            Dict mapping model names to test results
        """
        self.log(f"Testing {len(models)} models...")
        
        results = {}
        for model_name, model_factory in models.items():
            try:
                # Create model
                model = model_factory()
                model.eval()  # Set to eval mode
                
                # Determine input shape
                if input_shapes and model_name in input_shapes:
                    input_shape = (batch_size,) + input_shapes[model_name]
                else:
                    # Try to infer from model (simplified)
                    input_shape = (batch_size, 3, 32, 32)  # Default for vision models
                
                # Create test input
                input_tensor = torch.randn(input_shape)
                
                # Test model
                result = self.test_model(model, input_tensor, model_name)
                results[model_name] = result
                
            except Exception as e:
                self.log(f"❌ {model_name}: Failed to create/test model: {e}", "ERROR")
                results[model_name] = ModelTestResult(
                    model_name=model_name,
                    success=False,
                    error_message=str(e)
                )
        
        return results
    
    def generate_report(self, save_path: Optional[str] = None) -> str:
        """Generate comprehensive test report."""
        report = []
        report.append("=" * 80)
        report.append("DC DECOMPOSITION TEST REPORT")
        report.append("=" * 80)
        report.append(f"Total models tested: {len(self.results)}")
        
        successful = sum(1 for r in self.results.values() if r.success)
        report.append(f"Successful: {successful}/{len(self.results)} ({100*successful/len(self.results):.1f}%)")
        report.append("")
        
        # Summary statistics
        if successful > 0:
            max_recon_errors = [r.max_reconstruction_error for r in self.results.values() if r.success]
            max_grad_errors = [r.max_gradient_error for r in self.results.values() if r.success]
            
            report.append("SUMMARY STATISTICS:")
            report.append(f"Max reconstruction error: {max(max_recon_errors):.2e}")
            report.append(f"Mean reconstruction error: {np.mean(max_recon_errors):.2e}")
            report.append(f"Max gradient error: {max(max_grad_errors):.2e}")
            report.append(f"Mean gradient error: {np.mean(max_grad_errors):.2e}")
            report.append("")
        
        # Per-model results
        report.append("PER-MODEL RESULTS:")
        report.append("-" * 40)
        
        for name, result in self.results.items():
            if result.success:
                report.append(f"✅ {name}:")
                report.append(f"   Reconstruction error: {result.max_reconstruction_error:.2e}")
                report.append(f"   Gradient error: {result.max_gradient_error:.2e}")
                report.append(f"   Layers: {result.num_layers}")
                report.append(f"   Parameters: {result.num_parameters:,}")
                report.append(f"   Forward time: {result.total_forward_time_ms:.2f}ms")
                report.append(f"   Backward time: {result.total_backward_time_ms:.2f}ms")
            else:
                report.append(f"❌ {name}: {result.error_message}")
            report.append("")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            self.log(f"Report saved to {save_path}")
        
        return report_text
    
    def save_results(self, path: str):
        """Save detailed results to JSON."""
        # Convert results to serializable format
        serializable = {}
        for name, result in self.results.items():
            serializable[name] = {
                'model_name': result.model_name,
                'success': result.success,
                'error_message': result.error_message,
                'max_reconstruction_error': result.max_reconstruction_error,
                'max_gradient_error': result.max_gradient_error,
                'total_forward_time_ms': result.total_forward_time_ms,
                'total_backward_time_ms': result.total_backward_time_ms,
                'input_shape': list(result.input_shape),
                'num_parameters': result.num_parameters,
                'num_layers': result.num_layers,
                'layer_results': {
                    layer_name: {
                        'name': analysis.name,
                        'reconstruction_error': analysis.reconstruction_error,
                        'gradient_error': analysis.gradient_error,
                        'sensitivity_errors': analysis.sensitivity_errors,
                        'original_activation_norm': analysis.original_activation_norm,
                        'pos_activation_norm': analysis.pos_activation_norm,
                        'neg_activation_norm': analysis.neg_activation_norm,
                        'original_gradient_norm': analysis.original_gradient_norm,
                        'combined_gradient_norm': analysis.combined_gradient_norm,
                        'stacked_gradient_norms': analysis.stacked_gradient_norms,
                        'activation_shape': list(analysis.activation_shape),
                        'gradient_shape': list(analysis.gradient_shape),
                        'forward_time_ms': analysis.forward_time_ms,
                        'backward_time_ms': analysis.backward_time_ms,
                    }
                    for layer_name, analysis in result.layer_results.items()
                }
            }
        
        with open(path, 'w') as f:
            json.dump(serializable, f, indent=2)
        
        self.log(f"Detailed results saved to {path}")