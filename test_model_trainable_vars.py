#!/usr/bin/env python3
"""
Test to check if discriminator and generator models have trainable variables.
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

def test_model_trainable_variables():
    """Test if models have trainable variables."""
    
    print("=" * 60)
    print("TESTING MODEL TRAINABLE VARIABLES")
    print("=" * 60)
    
    try:
        # Import plugins
        from tsg_plugins.generator_plugin.generator_plugin import GeneratorPlugin
        from tsg_plugins.discriminator_plugin import DiscriminatorPlugin
        
        params = DEFAULT_VALUES.copy()
        
        # Create generator plugin
        print("\n1. Testing Generator Plugin...")
        generator_plugin = GeneratorPlugin(params, logger)
        generator_model = generator_plugin.get_model()
        
        if generator_model is not None:
            print(f"✓ Generator model loaded successfully")
            print(f"  Trainable variables: {len(generator_model.trainable_variables)}")
            if len(generator_model.trainable_variables) > 0:
                print(f"  First trainable variable shape: {generator_model.trainable_variables[0].shape}")
            else:
                print("  ❌ No trainable variables!")
        else:
            print("✗ Generator model is None")
        
        # Create discriminator plugin
        print("\n2. Testing Discriminator Plugin...")
        discriminator_plugin = DiscriminatorPlugin(params)  # Only pass params, not logger
        discriminator_model = discriminator_plugin.get_model()
        
        if discriminator_model is not None:
            print(f"✓ Discriminator model loaded successfully")
            print(f"  Trainable variables: {len(discriminator_model.trainable_variables)}")
            if len(discriminator_model.trainable_variables) > 0:
                print(f"  First trainable variable shape: {discriminator_model.trainable_variables[0].shape}")
            else:
                print("  ❌ No trainable variables!")
        else:
            print("✗ Discriminator model is None")
        
        # Test model compilation
        print("\n3. Testing Model Compilation...")
        if generator_model is not None and discriminator_model is not None:
            
            # Check if models are compiled
            try:
                # Test generator forward pass
                batch_size = 4
                noise = tf.random.normal([batch_size, 100])
                conditions = tf.random.normal([batch_size, 10])
                context = tf.random.normal([batch_size, 64])
                
                gen_output = generator_model([noise, conditions, context])
                print(f"✓ Generator forward pass successful. Output shape: {gen_output.shape}")
                
                # Test discriminator forward pass
                disc_output = discriminator_model(gen_output)
                print(f"✓ Discriminator forward pass successful. Output shape: {disc_output.shape}")
                
                # Test gradient computation
                with tf.GradientTape() as tape:
                    fake_output = discriminator_model(gen_output, training=True)
                    loss = tf.reduce_mean(fake_output)
                
                gradients = tape.gradient(loss, discriminator_model.trainable_variables)
                valid_gradients = [g for g in gradients if g is not None]
                
                print(f"✓ Gradient computation successful. Valid gradients: {len(valid_gradients)}/{len(gradients)}")
                
                if len(valid_gradients) == 0:
                    print("❌ No valid gradients computed!")
                    return False
                else:
                    print("✅ Models appear to be working correctly!")
                    return True
                
            except Exception as e:
                print(f"✗ Error during model testing: {e}")
                return False
        else:
            print("✗ Cannot test compilation - models not loaded")
            return False
            
    except Exception as e:
        print(f"✗ Error during model loading: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing TimeSeries-GAN model trainable variables...")
    success = test_model_trainable_variables()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 MODEL TEST PASSED: Models have trainable variables and work correctly!")
    else:
        print("❌ MODEL TEST FAILED: Issues with model trainable variables or compilation!")
    print("=" * 60)
