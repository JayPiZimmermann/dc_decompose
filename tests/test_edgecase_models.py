"""
Test script for all models from edgecase_test_models.py using the DC test framework.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('/home/jakob/code/src/dc_decompose/Imaginative_Polytopes')

import torch
import torch.nn as nn
from typing import Dict, Callable
import importlib.util

from dc_test_framework import DCTestFramework, TestMode

# Import the edgecase test models
spec = importlib.util.spec_from_file_location(
    "edgecase_test_models", 
    "/home/jakob/code/src/dc_decompose/Imaginative_Polytopes/dc_decomposition/test/edgecase_test_models.py"
)
edgecase_models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(edgecase_models)


def get_test_models() -> Dict[str, Callable[[], nn.Module]]:
    """Get all test models from edgecase_test_models.py"""
    
    models = {
        # Complex networks
        'ComplexNet': lambda: edgecase_models.ComplexNet(),
        'ComplexNetDeeper': lambda: edgecase_models.ComplexNetDeeper(),
        
        # Pooling-heavy models
        'MaxPoolHeavyModel': lambda: edgecase_models.MaxPoolHeavyModel(),
        'AvgPoolChainModel': lambda: edgecase_models.AvgPoolChainModel(),
        'AdaptivePoolMixModel': lambda: edgecase_models.AdaptivePoolMixModel(),
        
        # Dropout models
        'DropoutHeavyModel': lambda: edgecase_models.DropoutHeavyModel(),
        
        # Mixed complex models
        'MixedComplexBModel': lambda: edgecase_models.MixedComplexBModel(),
        'MixedComplexCModel': lambda: edgecase_models.MixedComplexCModel(),
    }
    
    # Add simple layer models if they exist
    try:
        # Single layer models
        models['SimpleLinear'] = lambda: nn.Linear(10, 5)
        models['SimpleConv'] = lambda: nn.Conv2d(3, 16, 3)
        models['SimpleReLU'] = lambda: nn.ReLU()
        models['SimpleBatchNorm'] = lambda: nn.BatchNorm2d(16)
        
        # Sequential models
        models['LinearChain'] = lambda: nn.Sequential(
            nn.Linear(20, 50),
            nn.ReLU(),
            nn.Linear(50, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )
        
        models['ConvChain'] = lambda: nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, 10)
        )
        
        models['ResNetBlock'] = lambda: nn.Sequential(
            edgecase_models.ResBlock(16, 16),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(16, 1)
        )
        
    except AttributeError as e:
        print(f"Warning: Some model classes not found: {e}")
    
    return models


def get_input_shapes() -> Dict[str, tuple]:
    """Get appropriate input shapes for each model."""
    return {
        # Complex networks (2D inputs)
        'ComplexNet': (2,),
        'ComplexNetDeeper': (2,),
        
        # CNN models (image inputs)
        'MaxPoolHeavyModel': (3, 64, 64),
        'AvgPoolChainModel': (3, 64, 64),
        'AdaptivePoolMixModel': (3, 32, 32),
        'DropoutHeavyModel': (3, 32, 32),
        'MixedComplexBModel': (3, 64, 64),
        'MixedComplexCModel': (3, 64, 64),
        'ConvChain': (3, 32, 32),
        'ResNetBlock': (16, 32, 32),
        
        # Linear models (1D inputs)
        'SimpleLinear': (10,),
        'LinearChain': (20,),
        
        # Single layer tests
        'SimpleConv': (3, 32, 32),
        'SimpleReLU': (10,),
        'SimpleBatchNorm': (16, 32, 32),
    }


def run_quick_test():
    """Run a quick test on a few models to verify framework works."""
    print("🚀 Running quick test...")
    
    framework = DCTestFramework(
        tolerance_reconstruction=1e-6,
        tolerance_gradient=1e-5,
        verbose=True
    )
    
    # Test a few simple models
    quick_models = {
        'SimpleLinear': lambda: nn.Linear(10, 5),
        'LinearChain': lambda: nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 4)
        ),
    }
    
    quick_shapes = {
        'SimpleLinear': (10,),
        'LinearChain': (8,),
    }
    
    results = framework.test_multiple_models(quick_models, quick_shapes, batch_size=2)
    
    print("\n" + "="*50)
    print("QUICK TEST RESULTS:")
    print("="*50)
    for name, result in results.items():
        if result.success:
            print(f"✅ {name}: Reconstruction error = {result.max_reconstruction_error:.2e}")
        else:
            print(f"❌ {name}: {result.error_message}")
    
    return all(r.success for r in results.values())


def run_full_test():
    """Run comprehensive test on all models."""
    print("🧪 Running comprehensive test on all edgecase models...")
    
    framework = DCTestFramework(
        tolerance_reconstruction=1e-6,
        tolerance_gradient=1e-5,
        verbose=True
    )
    
    models = get_test_models()
    input_shapes = get_input_shapes()
    
    print(f"Testing {len(models)} models...")
    
    # Run tests
    results = framework.test_multiple_models(models, input_shapes, batch_size=2)
    
    # Generate report
    report = framework.generate_report()
    print("\n" + report)
    
    # Save results
    framework.save_results("test_results.json")
    
    # Summary
    successful = sum(1 for r in results.values() if r.success)
    total = len(results)
    
    print(f"\n🎯 FINAL SUMMARY: {successful}/{total} models passed ({100*successful/total:.1f}%)")
    
    if successful == total:
        print("🎉 All tests passed!")
        return True
    else:
        print("⚠️  Some tests failed. Check the detailed report above.")
        return False


def test_hook_bypass():
    """Test hook bypass functionality."""
    print("🔧 Testing hook bypass functionality...")
    
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2)
    )
    
    input_tensor = torch.randn(2, 4)
    
    # Create decomposer
    from dc_decompose import HookDecomposer
    decomposer = HookDecomposer(model)
    decomposer.initialize(input_tensor)
    
    # Test with hooks enabled
    print("Testing with DC hooks enabled...")
    decomposer.enable_hooks(True)
    output_with_hooks = model(input_tensor)
    
    # Test with hooks disabled
    print("Testing with DC hooks disabled...")
    decomposer.disable_hooks()
    output_without_hooks = model(input_tensor)
    
    # Compare outputs
    error = torch.norm(output_with_hooks - output_without_hooks)
    print(f"Output difference with/without hooks: {error.item():.2e}")
    
    # Test hook status
    print(f"Hooks enabled status: {decomposer.hooks_enabled}")
    
    # Re-enable hooks
    decomposer.enable_hooks(True)
    print(f"Hooks re-enabled status: {decomposer.hooks_enabled}")
    
    print("✅ Hook bypass functionality working!")
    return True


if __name__ == "__main__":
    print("DC Decomposition Edge Case Model Testing")
    print("="*50)
    
    # First test the framework itself
    if not run_quick_test():
        print("❌ Quick test failed! Aborting full test.")
        sys.exit(1)
    
    print("\n" + "="*50)
    
    # Test hook bypass functionality
    if not test_hook_bypass():
        print("❌ Hook bypass test failed!")
        sys.exit(1)
    
    print("\n" + "="*50)
    
    # Run comprehensive test
    success = run_full_test()
    
    if success:
        print("🎉 All tests completed successfully!")
        sys.exit(0)
    else:
        print("❌ Some tests failed!")
        sys.exit(1)