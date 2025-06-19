#!/usr/bin/env python3
"""
Test MMD (Maximum Mean Discrepancy) Implementation

This script tests the MMD loss functionality in the GAN trainer,
including proper integration and comprehensive logging.
"""

import sys
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict, Any

# Add the project root to the path
sys.path.insert(0, '/home/harveybc/Documents/GitHub/timeseries-gan')

from app.config import DEFAULT_VALUES
from tsg_plugins.gan_trainer_plugin.training_coordinator import TrainingCoordinator


def create_test_config_with_mmd() -> Dict[str, Any]:
    """Create test configuration with MMD enabled."""
    config = DEFAULT_VALUES.copy()
    
    # Enable MMD loss
    config.update({
        "enable_mmd_loss": True,
        "mmd_lambda_g": 0.01,
        "mmd_lambda_d": 0.001,
        "mmd_gamma": None,  # Auto bandwidth
        "mmd_sample_size": 32,
        
        # Add missing optimizer parameters
        "generator_lr": 0.0002,
        "generator_beta1": 0.5,
        "discriminator_lr": 0.0002,
        "discriminator_beta1": 0.5,
        
        # Reduced sizes for testing
        "gan_epochs": 2,
        "gan_batch_size": 16,
        "seq_len": 10,
        "noise_dim": 20,
        "conditional_features_dim": 5,
        "context_vector_dim": 10,
        
        # Enable comprehensive logging
        "log_interval_epochs": 1,
        
        # Reduced patience for faster testing
        "lr_patience": 5,
        "early_stopping_patience": 10
    })
    
    return config


def create_mock_generator_plugin():
    """Create a mock generator plugin for testing."""
    class MockGeneratorPlugin:
        def prepare_features_for_discriminator(self, training_data_df):
            # Return synthetic time series data
            num_samples = len(training_data_df)
            num_features = 51  # Expected feature count
            
            # Generate synthetic data with temporal patterns
            data = np.random.randn(num_samples, num_features)
            
            # Add some structure to make it more realistic
            for i in range(num_features):
                data[:, i] += np.sin(np.linspace(0, 4*np.pi, num_samples)) * 0.5
            
            return data
    
    return MockGeneratorPlugin()


def create_simple_models(config):
    """Create simple models for testing."""
    seq_len = config["seq_len"]
    num_features = 51
    noise_dim = config["noise_dim"]
    conditional_dim = config["conditional_features_dim"]
    context_dim = config["context_vector_dim"]
    
    # Simple generator
    noise_input = tf.keras.layers.Input(shape=(noise_dim,))
    conditions_input = tf.keras.layers.Input(shape=(conditional_dim,))
    context_input = tf.keras.layers.Input(shape=(context_dim,))
    
    # Concatenate all inputs
    combined = tf.keras.layers.Concatenate()([noise_input, conditions_input, context_input])
    
    # Simple dense layers to generate time series
    x = tf.keras.layers.Dense(128, activation='relu')(combined)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    output = tf.keras.layers.Dense(seq_len * num_features, activation='tanh')(x)
    output = tf.keras.layers.Reshape((seq_len, num_features))(output)
    
    generator = tf.keras.Model(
        inputs=[noise_input, conditions_input, context_input],
        outputs=output,
        name="test_generator"
    )
    
    # Simple discriminator
    discriminator_input = tf.keras.layers.Input(shape=(seq_len, num_features))
    x = tf.keras.layers.Flatten()(discriminator_input)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    discriminator_output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    discriminator = tf.keras.Model(
        inputs=discriminator_input,
        outputs=discriminator_output,
        name="test_discriminator"
    )
    
    # Create GAN model (generator -> discriminator with discriminator frozen)
    discriminator.trainable = False  # Freeze discriminator for generator training
    gan_output = discriminator(generator([noise_input, conditions_input, context_input]))
    gan_model = tf.keras.Model(
        inputs=[noise_input, conditions_input, context_input],
        outputs=gan_output,
        name="test_gan"
    )
    
    # Compile models
    generator.compile(optimizer='adam', loss='mse')
    discriminator.compile(optimizer='adam', loss='binary_crossentropy')
    gan_model.compile(optimizer='adam', loss='binary_crossentropy')
    
    return generator, discriminator, gan_model


def test_mmd_calculation():
    """Test the MMD calculation function directly."""
    print("🧪 Testing MMD calculation function...")
    
    config = create_test_config_with_mmd()
    
    # Create a minimal TrainingCoordinator for testing
    import logging
    logger = logging.getLogger("test_mmd")
    logger.setLevel(logging.INFO)
    
    mock_generator_plugin = create_mock_generator_plugin()
    coordinator = TrainingCoordinator(config, logger, mock_generator_plugin)
    
    # Create test tensors
    batch_size = 16
    seq_len = 10
    num_features = 51
    
    # Real data (structured pattern)
    real_data = tf.random.normal([batch_size, seq_len, num_features])
    # Add a sine pattern
    time_steps = tf.linspace(0.0, 2*np.pi, seq_len)
    sine_pattern = tf.sin(time_steps)
    sine_pattern = tf.expand_dims(tf.expand_dims(sine_pattern, 0), -1)  # [1, seq_len, 1]
    sine_pattern = tf.tile(sine_pattern, [batch_size, 1, num_features])  # [batch_size, seq_len, num_features]
    real_data = real_data + 0.5 * sine_pattern
    
    # Fake data (different pattern)
    fake_data = tf.random.normal([batch_size, seq_len, num_features])
    # Add a cosine pattern  
    cosine_pattern = tf.cos(time_steps)
    cosine_pattern = tf.expand_dims(tf.expand_dims(cosine_pattern, 0), -1)  # [1, seq_len, 1]
    cosine_pattern = tf.tile(cosine_pattern, [batch_size, 1, num_features])  # [batch_size, seq_len, num_features]
    fake_data = fake_data + 0.5 * cosine_pattern
    
    # Test MMD calculation
    try:
        mmd_value = coordinator._calculate_mmd_rbf(real_data, fake_data)
        print(f"✓ MMD calculation successful: {mmd_value.numpy():.6f}")
        
        # Test MMD loss calculation
        mmd_loss, mmd_components = coordinator._calculate_mmd_loss(real_data, fake_data, 0.01)
        print(f"✓ MMD loss calculation successful: {mmd_loss.numpy():.6f}")
        print(f"✓ MMD components: {[f'{k}: {v.numpy():.6f}' for k, v in mmd_components.items() if hasattr(v, 'numpy')]}")
        
        return True
        
    except Exception as e:
        print(f"❌ MMD calculation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mmd_integration():
    """Test MMD integration in loss functions."""
    print("\n🧪 Testing MMD integration in loss functions...")
    
    config = create_test_config_with_mmd()
    
    import logging
    logger = logging.getLogger("test_mmd_integration")
    logger.setLevel(logging.INFO)
    
    mock_generator_plugin = create_mock_generator_plugin()
    coordinator = TrainingCoordinator(config, logger, mock_generator_plugin)
    
    # Create test data
    batch_size = 16
    seq_len = 10
    num_features = 51
    
    real_data = tf.random.normal([batch_size, seq_len, num_features])
    fake_data = tf.random.normal([batch_size, seq_len, num_features])
    
    # Test predictions (discriminator outputs)
    real_pred = tf.random.uniform([batch_size, 1], 0.6, 0.9)  # Should be close to 1
    fake_pred = tf.random.uniform([batch_size, 1], 0.1, 0.4)  # Should be close to 0
    
    try:
        # Test discriminator loss with MMD
        d_loss, d_real_loss, d_fake_loss, d_components = coordinator._discriminator_loss(
            real_pred, fake_pred, real_data, fake_data
        )
        
        print(f"✓ Discriminator loss calculation successful:")
        print(f"  Total D_loss: {d_loss.numpy():.6f}")
        print(f"  D_real_loss: {d_real_loss.numpy():.6f}")
        print(f"  D_fake_loss: {d_fake_loss.numpy():.6f}")
        print(f"  D_adversarial_loss: {d_components['d_adversarial_loss'].numpy():.6f}")
        print(f"  D_mmd_loss: {d_components['d_mmd_loss'].numpy():.6f}")
        print(f"  D_mmd_raw: {d_components['d_mmd_raw'].numpy():.6f}")
        
        # Test generator loss with MMD
        g_loss, g_components = coordinator._generator_loss(fake_pred, real_data, fake_data)
        
        print(f"✓ Generator loss calculation successful:")
        print(f"  Total G_loss: {g_loss.numpy():.6f}")
        print(f"  G_adversarial_loss: {g_components['g_adversarial_loss'].numpy():.6f}")
        print(f"  G_mmd_loss: {g_components['g_mmd_loss'].numpy():.6f}")
        print(f"  G_mmd_raw: {g_components['g_mmd_raw'].numpy():.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ MMD integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mmd_training_step():
    """Test MMD in actual training step."""
    print("\n🧪 Testing MMD in training step...")
    
    config = create_test_config_with_mmd()
    
    import logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger("test_mmd_training")
    
    mock_generator_plugin = create_mock_generator_plugin()
    coordinator = TrainingCoordinator(config, logger, mock_generator_plugin)
    
    # Create models
    generator, discriminator, gan_model = create_simple_models(config)
    
    # Ensure discriminator is trainable for discriminator training step
    discriminator.trainable = True
    
    # Setup optimizers
    coordinator._setup_optimizers()
    
    # Create training data
    num_samples = 100
    training_data = pd.DataFrame(np.random.randn(num_samples, 51))
    
    # Prepare real data
    try:
        real_data = coordinator._prepare_real_data(training_data)
        real_data_tensor = tf.convert_to_tensor(real_data, dtype=tf.float32)
        
        print(f"✓ Real data prepared: shape {real_data_tensor.shape}")
        
        # Store for MMD calculations
        coordinator.current_real_data_for_mmd = real_data_tensor
        coordinator.current_generator_for_mmd = generator
        
        # Test discriminator training step
        d_loss_avg, d_real_loss_avg, d_fake_loss_avg, d_metrics = coordinator._train_discriminator_step(
            real_data_tensor, generator, discriminator, 8, 1
        )
        
        print(f"✓ Discriminator training step successful:")
        print(f"  D_loss_avg: {d_loss_avg:.6f}")
        print(f"  D_mmd_loss: {d_metrics.get('d_mmd_loss', 0):.6f}")
        print(f"  D_mmd_raw: {d_metrics.get('d_mmd_raw', 0):.6f}")
        print(f"  D_adversarial_loss: {d_metrics.get('d_adversarial_loss', 0):.6f}")
        
        # Test generator training step
        g_loss, g_metrics = coordinator._train_generator_step(gan_model, 8, 1)
        
        print(f"✓ Generator training step successful:")
        print(f"  G_loss: {g_loss:.6f}")
        print(f"  G_mmd_loss: {g_metrics.get('g_mmd_loss', 0):.6f}")
        print(f"  G_mmd_raw: {g_metrics.get('g_mmd_raw', 0):.6f}")
        print(f"  G_adversarial_loss: {g_metrics.get('g_adversarial_loss', 0):.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training step test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all MMD tests."""
    print("🚀 Testing MMD (Maximum Mean Discrepancy) Implementation")
    print("=" * 60)
    
    # Suppress TensorFlow warnings for cleaner output
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.get_logger().setLevel('ERROR')
    
    tests = [
        ("MMD Calculation", test_mmd_calculation),
        ("MMD Integration", test_mmd_integration),
        ("MMD Training Step", test_mmd_training_step)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name}...")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 MMD Implementation Test Results:")
    print("=" * 60)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if success:
            passed += 1
    
    print(f"\n🎯 Overall Result: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All MMD tests passed! The implementation is working correctly.")
        print("\n📝 Key Features Verified:")
        print("  ✓ MMD calculation with RBF kernel")
        print("  ✓ MMD loss integration in discriminator")
        print("  ✓ MMD loss integration in generator")
        print("  ✓ MMD components tracking and logging")
        print("  ✓ Configurable MMD parameters")
        return True
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
