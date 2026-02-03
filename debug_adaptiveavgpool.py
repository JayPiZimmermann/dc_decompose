#!/usr/bin/env python3
"""
Debug script for AdaptiveAvgPool2d backward pass dimensions.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from dc_decompose import HookDecomposer, InputMode, BackwardMode

def debug_adaptive_avgpool():
    """Debug the AdaptiveAvgPool2d dimension issues."""
    print("🔍 Debugging AdaptiveAvgPool2d dimensions...")
    
    # Create the problematic model
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((4, 4)),
        nn.Flatten(),
        nn.Linear(32 * 16, 10)
    )
    
    input_tensor = torch.randn(1, 3, 16, 16, requires_grad=True)
    
    print(f"Input shape: {input_tensor.shape}")
    
    # Forward pass to see intermediate shapes
    x = input_tensor
    for i, layer in enumerate(model):
        x = layer(x)
        print(f"After layer {i} ({type(layer).__name__}): {x.shape}")
    
    # Now test with decomposer
    print("\n🧪 Testing with DC decomposer...")
    model.eval()
    
    decomposer = HookDecomposer(model)
    decomposer.set_input_mode(InputMode.CENTER)
    decomposer.set_backward_mode(BackwardMode.ALPHA)
    decomposer.set_alpha(0.35)
    decomposer.initialize(input_tensor)
    
    # Enable hooks and forward
    decomposer.enable_hooks(True)
    output = model(input_tensor)
    print(f"DC Forward output shape: {output.shape}")
    
    # Try backward
    try:
        print("🧪 Attempting backward pass...")
        loss = output.sum()
        loss.backward()
        print("✅ Backward pass succeeded!")
        
        # Check the AdaptiveAvgPool2d cache
        adaptive_pool_layer = model[5]  # AdaptiveAvgPool2d
        for name, cache in decomposer.caches.items():
            if 'adaptive' in name.lower() or '5' in name:
                print(f"Cache for layer {name}:")
                if hasattr(cache, 'input_pos') and cache.input_pos is not None:
                    print(f"  input_pos shape: {cache.input_pos.shape}")
                if hasattr(cache, 'output_pos') and cache.output_pos is not None:
                    print(f"  output_pos shape: {cache.output_pos.shape}")
        
    except Exception as e:
        print(f"❌ Backward pass failed: {e}")
        
        # Debug the specific AdaptiveAvgPool2d layer
        print("\n🔍 Debugging AdaptiveAvgPool2d layer specifically...")
        
        # Find the AdaptiveAvgPool2d cache
        adaptive_cache = None
        for name, cache in decomposer.caches.items():
            print(f"Layer {name}: {type(cache).__name__}")
            if hasattr(cache, 'input_pos'):
                print(f"  input_pos: {cache.input_pos.shape if cache.input_pos is not None else None}")
            if hasattr(cache, 'output_pos'):
                print(f"  output_pos: {cache.output_pos.shape if cache.output_pos is not None else None}")

if __name__ == "__main__":
    debug_adaptive_avgpool()