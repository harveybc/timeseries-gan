#!/usr/bin/env python3
"""
Test to simulate the exact training coordinator workflow to identify the gradient issue.
"""

import sys
sys.path.insert(0, '/home/harveybc/Documents/GitHub/timeseries-gan')

import tensorflow as tf
import numpy as np
import pandas as pd
import logging
from app.config import DEFAULT_VALUES

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_training_coordinator_workflow():
    """Test the exact workflow that training coordinator follows."""
    
    print("=" * 60)
    print("TESTING TRAINING COORDINATOR WORKFLOW")
    print("=" * 60)
    
    try:
        # Import required classes
        from tsg_plugins.generator_plugin.generator_plugin import GeneratorPlugin
        from tsg_plugins.discriminator_plugin import DiscriminatorPlugin
        from tsg_plugins.gan_trainer_plugin.training_coordinator import TrainingCoordinator
        
        params = DEFAULT_VALUES.copy()
        
        # Step 1: Create plugins (as GANTrainerPlugin does)
        print("\n1. Creating plugins...")
        generator_plugin = GeneratorPlugin(params, logger)
        discriminator_plugin = DiscriminatorPlugin(params)
        
        # Step 2: Get models (as GANTrainerPlugin does)
        print("\n2. Getting models...")
        generator_model = generator_plugin.get_model()
        discriminator_model = discriminator_plugin.get_model()
        
        print(f"Generator model: {generator_model is not None}")
        print(f"Discriminator model: {discriminator_model is not None}")
        
        if generator_model is None or discriminator_model is None:
            print("❌ Models not available")
            return False
        
        # Step 3: Check trainable variables (before creating training coordinator)
        print(f"\n3. Model trainable variables:")
        print(f"Generator trainable vars: {len(generator_model.trainable_variables)}")
        print(f"Discriminator trainable vars: {len(discriminator_model.trainable_variables)}")
        
        if len(discriminator_model.trainable_variables) == 0:
            print("❌ Discriminator has no trainable variables!")
            return False
        
        # Step 4: Create training coordinator (as GANTrainerPlugin does)
        print("\n4. Creating training coordinator...")
        training_coordinator = TrainingCoordinator(params, logger, generator_plugin)
        
        # Step 5: Create dummy training data
        print("\n5. Creating dummy training data...")
        dummy_data = pd.DataFrame(np.random.randn(1000, 45))  # 45 features as input
        
        # Step 6: Prepare real data (as training coordinator does)
        print("\n6. Preparing real data...")
        real_data = training_coordinator._prepare_real_data(dummy_data)
        print(f"Real data shape: {real_data.shape}")
        
        # Convert to tensor
        real_data_tensor = tf.constant(real_data, dtype=tf.float32)
        print(f"Real data tensor shape: {real_data_tensor.shape}")
        
        # Step 7: Test discriminator step simulation
        print("\n7. Testing discriminator training step...")
        batch_size = 4
        
        # Sample real batch (as training coordinator does)
        batch_indices = tf.random.uniform([batch_size], 0, tf.shape(real_data_tensor)[0], dtype=tf.int32)
        real_batch = tf.gather(real_data_tensor, batch_indices)
        print(f"Real batch shape: {real_batch.shape}")
        
        # Generate fake batch (as training coordinator does)
        noise_dim = params.get("noise_dim", 100)
        conditional_features_dim = params.get("conditional_features_dim", 10)
        context_vector_dim = params.get("context_vector_dim", 64)
        
        noise = tf.random.normal([batch_size, noise_dim])
        conditions = tf.random.normal([batch_size, conditional_features_dim])
        context = tf.random.normal([batch_size, context_vector_dim])
        
        fake_batch = generator_model([noise, conditions, context], training=False)
        print(f"Fake batch shape: {fake_batch.shape}")
        
        # Step 8: Test discriminator forward pass and gradient computation
        print("\n8. Testing discriminator gradient computation...")
        
        print(f"Discriminator trainable vars before gradient: {len(discriminator_model.trainable_variables)}")
        
        with tf.GradientTape() as tape:
            real_pred = discriminator_model(real_batch, training=True)
            fake_pred = discriminator_model(fake_batch, training=True)
            
            print(f"Real pred shape: {real_pred.shape}")
            print(f"Fake pred shape: {fake_pred.shape}")
            
            # Simple loss calculation
            real_loss = tf.keras.losses.binary_crossentropy(tf.ones_like(real_pred), real_pred)
            fake_loss = tf.keras.losses.binary_crossentropy(tf.zeros_like(fake_pred), fake_pred)
            d_loss = tf.reduce_mean(real_loss + fake_loss)
            
            print(f"Discriminator loss: {d_loss.numpy()}")
        
        # Compute gradients
        gradients = tape.gradient(d_loss, discriminator_model.trainable_variables)
        
        print(f"Number of gradients: {len(gradients) if gradients else 0}")
        print(f"Discriminator trainable vars: {len(discriminator_model.trainable_variables)}")
        
        if gradients is None:
            print("❌ Gradients are None!")
            return False
        
        # Count valid gradients
        valid_gradients = [g for g in gradients if g is not None]
        print(f"Valid gradients: {len(valid_gradients)}/{len(gradients)}")
        
        if len(valid_gradients) == 0:
            print("❌ No valid gradients!")
            return False
        
        print("✅ Gradient computation successful!")
        return True
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing Training Coordinator Workflow...")
    success = test_training_coordinator_workflow()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 WORKFLOW TEST PASSED: Training coordinator workflow works correctly!")
    else:
        print("❌ WORKFLOW TEST FAILED: Issues with training coordinator workflow!")
    print("=" * 60)
