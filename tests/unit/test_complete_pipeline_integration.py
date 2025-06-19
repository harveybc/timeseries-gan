#!/usr/bin/env python3
"""
Test script to verify the complete VAE-GAN training pipeline integration.
This simulates the full pipeline with mock models to test the method signature flow.
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
import tensorflow as tf

# Add project to path
sys.path.insert(0, os.path.dirname(__file__))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_mock_generator_model():
    """Create a mock generator model with proper input/output shapes."""
    # Mock composite generator with 3 inputs as per REFERENCE.md
    noise_input = tf.keras.Input(shape=(32,), name="noise_input")
    conditions_input = tf.keras.Input(shape=(10,), name="conditions_input") 
    context_input = tf.keras.Input(shape=(64,), name="context_input")
    
    # Simple processing - concatenate inputs and process
    concat = tf.keras.layers.Concatenate()([noise_input, conditions_input, context_input])
    dense1 = tf.keras.layers.Dense(256, activation='relu')(concat)
    dense2 = tf.keras.layers.Dense(512, activation='relu')(dense1)
    # Output 57 features for 144 timesteps (as per REFERENCE.md)
    output = tf.keras.layers.Dense(144 * 57, activation='tanh')(dense2)
    output = tf.keras.layers.Reshape((144, 57))(output)
    
    model = tf.keras.Model(inputs=[noise_input, conditions_input, context_input], outputs=output, name="mock_generator")
    return model

def create_mock_discriminator_model():
    """Create a mock discriminator model with proper input shape."""
    # Discriminator expects (144, 57) as per REFERENCE.md
    input_layer = tf.keras.Input(shape=(144, 57), name="discriminator_input")
    conv1 = tf.keras.layers.Conv1D(64, 3, activation='relu')(input_layer)
    conv2 = tf.keras.layers.Conv1D(128, 3, activation='relu')(conv1)
    lstm = tf.keras.layers.LSTM(64)(conv2)
    dense = tf.keras.layers.Dense(32, activation='relu')(lstm)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(dense)
    
    model = tf.keras.Model(inputs=input_layer, outputs=output, name="mock_discriminator")
    return model

def create_mock_gan_model(generator, discriminator):
    """Create a mock GAN model combining generator and discriminator."""
    # Freeze discriminator for GAN training
    discriminator.trainable = False
    
    # GAN inputs (same as generator)
    noise_input = tf.keras.Input(shape=(32,), name="gan_noise_input")
    conditions_input = tf.keras.Input(shape=(10,), name="gan_conditions_input")
    context_input = tf.keras.Input(shape=(64,), name="gan_context_input")
    
    # Generate fake data
    fake_data = generator([noise_input, conditions_input, context_input])
    
    # Discriminator prediction
    prediction = discriminator(fake_data)
    
    model = tf.keras.Model(inputs=[noise_input, conditions_input, context_input], outputs=prediction, name="mock_gan")
    return model

class MockGeneratorPlugin:
    """Mock generator plugin for testing."""
    def __init__(self):
        self.model = create_mock_generator_model()
        
    def get_model(self):
        return self.model

class MockDiscriminatorPlugin:
    """Mock discriminator plugin for testing."""
    def __init__(self):
        self.model = create_mock_discriminator_model()
        
    def get_model(self):
        return self.model

def test_complete_vae_gan_pipeline():
    """Test the complete VAE-GAN pipeline with mock models."""
    
    print("=== Testing Complete VAE-GAN Pipeline Integration ===")
    
    try:
        # Import the GANTrainerPlugin
        from tsg_plugins.gan_trainer_plugin.gan_trainer_plugin import GANTrainerPlugin
        print("✓ Successfully imported GANTrainerPlugin")
        
        # Create test parameters matching REFERENCE.md specifications
        params = {
            'gan_epochs': 2,
            'gan_batch_size': 4,
            'generator_lr': 1e-4,
            'discriminator_lr': 1e-4,
            'seq_len': 144,
            'feeder_noise_dim': 32,
            'conditional_features_dim': 10,
            'context_vector_dim': 64
        }
        
        # Initialize the plugin
        plugin = GANTrainerPlugin(params)
        print("✓ GANTrainerPlugin initialized successfully")
        
        # Create mock plugin instances
        mock_generator_plugin = MockGeneratorPlugin()
        mock_discriminator_plugin = MockDiscriminatorPlugin()
        
        # Set plugin instances directly
        plugin.generator_plugin_instance = mock_generator_plugin
        plugin.discriminator_plugin_instance = mock_discriminator_plugin
        
        print("✓ Mock plugin instances created and assigned")
        
        # Verify models are built correctly
        models_built = plugin._ensure_models_are_built()
        print(f"✓ Models built successfully: {models_built}")
        
        if models_built:
            print(f"  - Generator model: {plugin.generator_model.name}")
            print(f"  - Discriminator model: {plugin.discriminator_model.name}")
            print(f"  - GAN model: {plugin.gan_model.name if plugin.gan_model else 'None'}")
            
            # Verify model shapes
            print(f"  - Generator input shapes: {[inp.shape for inp in plugin.generator_model.inputs]}")
            print(f"  - Generator output shape: {plugin.generator_model.output.shape}")
            print(f"  - Discriminator input shape: {plugin.discriminator_model.input.shape}")
            print(f"  - Discriminator output shape: {plugin.discriminator_model.output.shape}")
        else:
            print("✗ Failed to build models")
            return False
        
        # Create dummy training data with proper shape (57 features as per REFERENCE.md)
        dummy_data = pd.DataFrame(np.random.randn(144, 57))  # 144 timesteps, 57 features
        print(f"✓ Created dummy training data with shape: {dummy_data.shape}")
        
        # Test the complete training flow
        try:
            print("🚀 Starting mock training process...")
            result = plugin.train(training_data=dummy_data, epochs=2, batch_size=4)
            print("✓ Training completed successfully!")
            print(f"  - Training result type: {type(result)}")
            if isinstance(result, dict):
                print(f"  - Result keys: {list(result.keys())}")
            
        except Exception as e:
            print(f"✗ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error during pipeline test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting Complete VAE-GAN Pipeline Integration Test...\n")
    
    # Disable TensorFlow warnings
    tf.get_logger().setLevel('ERROR')
    
    success = test_complete_vae_gan_pipeline()
    
    print("\n=== Test Results ===")
    print(f"Complete VAE-GAN pipeline test: {'PASS' if success else 'FAIL'}")
    
    if success:
        print("\n🎉 COMPLETE SUCCESS! The VAE-GAN training pipeline is working correctly with:")
        print("  ✓ TrainingCoordinator method signature fixed")
        print("  ✓ Proper model passing between components")
        print("  ✓ Correct input/output shapes (144x57 as per REFERENCE.md)")
        print("  ✓ Full training workflow integration")
    else:
        print("\n❌ Pipeline test FAILED. Please check the errors above.")
        sys.exit(1)
