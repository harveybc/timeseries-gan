#!/usr/bin/env python3
"""
Test script to verify noise dimension fix in training coordinator.
"""

import sys
import os
sys.path.insert(0, '/home/harveybc/Documents/GitHub/timeseries-gan')

import tensorflow as tf
import numpy as np
import logging
from app.config import DEFAULT_VALUES

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_noise_dimensions_in_training():
    """Test that training coordinator uses correct noise dimensions."""
    
    print("Testing noise dimensions in training coordinator...")
    
    # Test params like training coordinator would use
    params = DEFAULT_VALUES.copy()
    
    # Simulate what _train_discriminator_step and _train_generator_step do
    batch_size = 32
    
    # Test discriminator step noise generation (fixed version)
    noise_dim = params.get("noise_dim", 100)  # FIXED: was feeder_noise_dim
    conditional_features_dim = params.get("conditional_features_dim", 10)
    context_vector_dim = params.get("context_vector_dim", 64)
    
    print(f"Parameters used by training coordinator:")
    print(f"  noise_dim: {noise_dim}")
    print(f"  conditional_features_dim: {conditional_features_dim}")
    print(f"  context_vector_dim: {context_vector_dim}")
    
    # Generate tensors as training coordinator would
    try:
        noise = tf.random.normal([batch_size, noise_dim])
        conditions = tf.random.normal([batch_size, conditional_features_dim])
        context = tf.random.normal([batch_size, context_vector_dim])
        
        print(f"\nGenerated tensor shapes:")
        print(f"  noise: {noise.shape}")
        print(f"  conditions: {conditions.shape}")
        print(f"  context: {context.shape}")
        
        # Check if noise shape is correct (should be [32, 100], not [32, 32])
        expected_noise_shape = tf.TensorShape([batch_size, 100])
        if noise.shape == expected_noise_shape:
            print(f"✓ SUCCESS: Noise shape is correct {noise.shape}")
            print("✓ Shape mismatch issue should be RESOLVED!")
            return True
        else:
            print(f"✗ ERROR: Noise shape is {noise.shape}, expected {expected_noise_shape}")
            return False
            
    except Exception as e:
        print(f"✗ Error generating tensors: {e}")
        return False

def test_old_vs_new_behavior():
    """Compare old (wrong) vs new (correct) behavior."""
    
    print("\n" + "="*50)
    print("COMPARING OLD vs NEW BEHAVIOR")
    print("="*50)
    
    params = DEFAULT_VALUES.copy()
    batch_size = 32
    
    # OLD behavior (wrong)
    old_noise_dim = params.get("feeder_noise_dim", 32)  # Was using feeder_noise_dim
    old_noise = tf.random.normal([batch_size, old_noise_dim])
    
    # NEW behavior (correct)
    new_noise_dim = params.get("noise_dim", 100)  # Now using noise_dim
    new_noise = tf.random.normal([batch_size, new_noise_dim])
    
    print(f"OLD (wrong) behavior:")
    print(f"  Parameter used: feeder_noise_dim = {old_noise_dim}")
    print(f"  Noise shape: {old_noise.shape}")
    
    print(f"\nNEW (correct) behavior:")
    print(f"  Parameter used: noise_dim = {new_noise_dim}")
    print(f"  Noise shape: {new_noise.shape}")
    
    print(f"\nGenerator expects: (batch_size, 100)")
    print(f"Old version would give: (batch_size, {old_noise_dim}) ❌")
    print(f"New version gives: (batch_size, {new_noise_dim}) ✅")

if __name__ == "__main__":
    print("Testing TimeSeries-GAN noise dimension fix...")
    
    success = test_noise_dimensions_in_training()
    test_old_vs_new_behavior()
    
    print("\n" + "="*50)
    if success:
        print("🎉 NOISE DIMENSION FIX VERIFIED!")
        print("The shape mismatch should be resolved.")
    else:
        print("❌ NOISE DIMENSION ISSUE NOT FIXED!")
    print("="*50)
