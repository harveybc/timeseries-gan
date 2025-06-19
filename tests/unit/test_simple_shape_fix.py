#!/usr/bin/env python3
"""
Simple test to verify training coordinator noise dimension fix.
"""

import sys
sys.path.append('/home/harveybc/Documents/GitHub/timeseries-gan')

from app.config import DEFAULT_VALUES

def test_noise_dimensions():
    """Test that noise dimensions are correct in config."""
    
    print("Testing noise dimension parameters...")
    
    # Get parameters as training coordinator would
    noise_dim = DEFAULT_VALUES.get("noise_dim", 100)
    feeder_noise_dim = DEFAULT_VALUES.get("feeder_noise_dim", 32)
    conditional_features_dim = DEFAULT_VALUES.get("conditional_features_dim", 10)
    context_vector_dim = DEFAULT_VALUES.get("context_vector_dim", 64)
    
    print(f"noise_dim (for generator): {noise_dim}")
    print(f"feeder_noise_dim (for feeder): {feeder_noise_dim}")
    print(f"conditional_features_dim: {conditional_features_dim}")
    print(f"context_vector_dim: {context_vector_dim}")
    
    # Simulate what training coordinator does now
    batch_size = 32
    print(f"\nGenerator input shapes (batch_size={batch_size}):")
    print(f"  noise: ({batch_size}, {noise_dim})")
    print(f"  conditions: ({batch_size}, {conditional_features_dim})")
    print(f"  context: ({batch_size}, {context_vector_dim})")
    
    # Check if noise_dim is correct (should be 100, not 32)
    if noise_dim == 100:
        print("✓ SUCCESS: noise_dim is 100 (correct for generator)")
    else:
        print(f"✗ ERROR: noise_dim is {noise_dim} (should be 100)")
    
    if feeder_noise_dim == 32:
        print("✓ SUCCESS: feeder_noise_dim is 32 (correct for feeder)")
    else:
        print(f"✗ ERROR: feeder_noise_dim is {feeder_noise_dim} (should be 32)")

if __name__ == "__main__":
    test_noise_dimensions()
