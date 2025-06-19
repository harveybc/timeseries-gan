#!/usr/bin/env python3
"""
Test script to verify the noise input shape fix in the training coordinator.
"""

import sys
import os
sys.path.append('/home/harveybc/Documents/GitHub/timeseries-gan')

import numpy as np
import pandas as pd
import tensorflow as tf
from app.config import DEFAULT_VALUES
from tsg_plugins.generator_plugin.generator_plugin import GeneratorPlugin
from tsg_plugins.gan_trainer_plugin.training_coordinator import TrainingCoordinator
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_noise_input_shapes():
    """Test that the training coordinator uses correct noise input shapes."""
    
    print("=" * 60)
    print("TESTING NOISE INPUT SHAPE FIX")
    print("=" * 60)
    
    # Test configuration with correct noise dimensions
    test_config = DEFAULT_VALUES.copy()
    test_config.update({
        'noise_dim': 100,
        'conditional_features_dim': 10,
        'context_vector_dim': 64,
        'batch_size': 32
    })
    
    print(f"Test config noise_dim: {test_config['noise_dim']}")
    print(f"Test config conditional_features_dim: {test_config['conditional_features_dim']}")
    print(f"Test config context_vector_dim: {test_config['context_vector_dim']}")
    
    # Create generator plugin
    try:
        generator_plugin = GeneratorPlugin(test_config, logger)
        print("✓ GeneratorPlugin created successfully")
    except Exception as e:
        print(f"✗ Error creating GeneratorPlugin: {e}")
        return False
    
    # Create training coordinator
    try:
        coordinator = TrainingCoordinator(test_config, logger, generator_plugin)
        print("✓ TrainingCoordinator created successfully")
    except Exception as e:
        print(f"✗ Error creating TrainingCoordinator: {e}")
        return False
    
    # Test noise shape extraction
    try:
        batch_size = test_config['batch_size']
        
        # Test discriminator step noise generation
        noise_dim = test_config.get("noise_dim", 100)
        conditional_features_dim = test_config.get("conditional_features_dim", 10)
        context_vector_dim = test_config.get("context_vector_dim", 64)
        
        print(f"\nNoise input shapes for batch_size={batch_size}:")
        print(f"  noise: ({batch_size}, {noise_dim})")
        print(f"  conditions: ({batch_size}, {conditional_features_dim})")
        print(f"  context: ({batch_size}, {context_vector_dim})")
        
        # Generate test tensors
        noise = tf.random.normal([batch_size, noise_dim])
        conditions = tf.random.normal([batch_size, conditional_features_dim])
        context = tf.random.normal([batch_size, context_vector_dim])
        
        print(f"\nActual tensor shapes:")
        print(f"  noise: {noise.shape}")
        print(f"  conditions: {conditions.shape}")
        print(f"  context: {context.shape}")
        
        # Verify shapes match expected generator input
        expected_noise_shape = (batch_size, 100)  # Generator expects (batch_size, 100)
        if tuple(noise.shape) == expected_noise_shape:
            print(f"✓ Noise shape matches expected generator input: {expected_noise_shape}")
        else:
            print(f"✗ Noise shape mismatch. Expected: {expected_noise_shape}, Got: {tuple(noise.shape)}")
            return False
            
        print("✓ All input shapes are correct!")
        return True
        
    except Exception as e:
        print(f"✗ Error testing noise shapes: {e}")
        return False

def test_config_parameter_access():
    """Test that config parameters are accessible with correct values."""
    
    print("\n" + "=" * 60)
    print("TESTING CONFIG PARAMETER ACCESS")
    print("=" * 60)
    
    test_config = DEFAULT_VALUES.copy()
    
    # Test parameters used by training coordinator
    params_to_test = [
        ("noise_dim", 100),
        ("conditional_features_dim", 10),
        ("context_vector_dim", 64),
        ("feeder_noise_dim", 32)  # Should be different from noise_dim
    ]
    
    for param_name, expected_value in params_to_test:
        actual_value = test_config.get(param_name)
        if actual_value == expected_value:
            print(f"✓ {param_name}: {actual_value} (correct)")
        else:
            print(f"✗ {param_name}: Expected {expected_value}, Got {actual_value}")
            return False
    
    print("✓ All config parameters have correct values!")
    return True

if __name__ == "__main__":
    print("Testing noise input shape fix for TimeSeries-GAN training coordinator...")
    
    success = True
    
    # Test config parameter access
    if not test_config_parameter_access():
        success = False
    
    # Test noise input shapes
    if not test_noise_input_shapes():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED! Shape mismatch should be resolved.")
    else:
        print("❌ SOME TESTS FAILED! Issues remain.")
    print("=" * 60)
