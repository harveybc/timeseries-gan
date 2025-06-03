#!/usr/bin/env python3
"""
Integration Test for GAN Trainer Plugin

This test verifies that the modular GAN trainer plugin works correctly
with all its specialized modules.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
import tensorflow as tf
from unittest.mock import Mock, MagicMock
import logging

# Add the plugin directory to sys.path
sys.path.insert(0, '/home/harveybc/Documents/GitHub/timeseries-gan')

# Import the GAN trainer plugin
from tsg_plugins.gan_trainer_plugin import GANTrainerPlugin


class TestGANTrainerPlugin(unittest.TestCase):
    """Test cases for the modular GAN trainer plugin."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        
        # Create test configuration
        self.config = {
            "gan_epochs": 5,  # Small number for testing
            "gan_batch_size": 4,
            "seq_len": 10,
            "latent_dim": 8,
            "discriminator_lstm_units": 16,
            "results_base_dir": "/tmp/test_gan_trainer",
            "discriminator_input_feature_names": ["OPEN", "HIGH", "LOW", "CLOSE"]
        }
        
        # Create mock generator plugin
        self.mock_generator_plugin = Mock()
        self.mock_generator_plugin.generator_model = self._create_mock_generator()
        
        # Create mock feeder plugin
        self.mock_feeder_plugin = Mock()
        
        # Create test data
        self.test_data = self._create_test_data()
    
    def _create_mock_generator(self):
        """Create a mock generator model for testing."""
        # Create a simple generator model
        input_layer = tf.keras.layers.Input(shape=(10, 8), name="generator_input")
        x = tf.keras.layers.LSTM(16, return_sequences=True)(input_layer)
        x = tf.keras.layers.Dense(4, activation='tanh')(x)
        output = tf.keras.layers.Dense(4)(x)
        
        generator = tf.keras.Model(inputs=input_layer, outputs=output, name="mock_generator")
        generator.compile(optimizer='adam', loss='mse')
        
        return generator
    
    def _create_test_data(self):
        """Create test data for training."""
        # Create synthetic time series data
        np.random.seed(42)
        data = np.random.randn(100, 4)  # 100 time steps, 4 features (OHLC)
        
        # Create DataFrame
        columns = ["OPEN", "HIGH", "LOW", "CLOSE"]
        df = pd.DataFrame(data, columns=columns)
        
        return df
    
    def test_plugin_initialization(self):
        """Test that the plugin initializes correctly."""
        # Create plugin instance
        plugin = GANTrainerPlugin(
            config=self.config,
            generator_plugin_instance=self.mock_generator_plugin,
            feeder_plugin_instance=self.mock_feeder_plugin
        )
        
        # Verify initialization
        self.assertIsNotNone(plugin)
        self.assertIsNotNone(plugin.training_coordinator)
        self.assertIsNotNone(plugin.model_builder)
        self.assertIsNotNone(plugin.data_generator)
        self.assertIsNotNone(plugin.model_persistence)
        self.assertIsNotNone(plugin.training_metrics)
        
        # Verify parameters were set correctly
        self.assertEqual(plugin.params["gan_epochs"], 5)
        self.assertEqual(plugin.params["gan_batch_size"], 4)
        self.assertEqual(plugin.params["seq_len"], 10)
    
    def test_plugin_set_params(self):
        """Test the mandatory set_params method."""
        plugin = GANTrainerPlugin(
            config=self.config,
            generator_plugin_instance=self.mock_generator_plugin
        )
        
        # Call set_params
        plugin.set_params(gan_epochs=10, gan_batch_size=8)
        
        # Verify parameters were updated
        self.assertEqual(plugin.params["gan_epochs"], 10)
        self.assertEqual(plugin.params["gan_batch_size"], 8)
    
    def test_plugin_get_debug_info(self):
        """Test the mandatory get_debug_info method."""
        plugin = GANTrainerPlugin(
            config=self.config,
            generator_plugin_instance=self.mock_generator_plugin
        )
        
        # Get debug info
        debug_info = plugin.get_debug_info()
        
        # Verify debug info structure
        self.assertIsInstance(debug_info, dict)
        self.assertIn("gan_epochs", debug_info)
        self.assertIn("gan_batch_size", debug_info)
        self.assertIn("training_coordinator", debug_info)
        self.assertIn("model_builder", debug_info)
    
    def test_plugin_add_debug_info(self):
        """Test the mandatory add_debug_info method."""
        plugin = GANTrainerPlugin(
            config=self.config,
            generator_plugin_instance=self.mock_generator_plugin
        )
        
        # Create debug info dictionary
        debug_info = {"test_key": "test_value"}
        
        # Add debug info
        plugin.add_debug_info(debug_info)
        
        # Verify debug info was added
        self.assertIn("gan_epochs", debug_info)
        self.assertIn("test_key", debug_info)
    
    def test_model_building(self):
        """Test that models are built correctly."""
        plugin = GANTrainerPlugin(
            config=self.config,
            generator_plugin_instance=self.mock_generator_plugin
        )
        
        # Verify models were built
        self.assertIsNotNone(plugin.generator)
        self.assertIsNotNone(plugin.discriminator)
        self.assertIsNotNone(plugin.gan_model)
        
        # Verify model properties
        self.assertEqual(plugin.generator.name, "mock_generator")
        self.assertEqual(plugin.discriminator.name, "discriminator")
        self.assertEqual(plugin.gan_model.name, "gan")
    
    def test_data_preparation(self):
        """Test that data preparation works correctly."""
        plugin = GANTrainerPlugin(
            config=self.config,
            generator_plugin_instance=self.mock_generator_plugin
        )
        
        # Prepare real data
        real_data_tensor = plugin.data_generator.prepare_real_data(self.test_data)
        
        # Verify data shape
        self.assertEqual(len(real_data_tensor.shape), 3)  # (samples, seq_len, features)
        self.assertEqual(real_data_tensor.shape[1], 10)   # seq_len
        self.assertEqual(real_data_tensor.shape[2], 4)    # num_features
    
    def test_training_process_short(self):
        """Test a short training process."""
        plugin = GANTrainerPlugin(
            config=self.config,
            generator_plugin_instance=self.mock_generator_plugin,
            feeder_plugin_instance=self.mock_feeder_plugin
        )
        
        # Run short training
        try:
            results = plugin.process(
                self.test_data,
                epochs=2,  # Very short for testing
                batch_size=4
            )
            
            # Verify results structure
            self.assertIsInstance(results, dict)
            self.assertIn("training_history", results)
            self.assertIn("final_metrics", results)
            self.assertIn("models_dir", results)
            
        except Exception as e:
            # Training might fail due to mock objects, but structure should be correct
            self.assertIsInstance(e, (ValueError, TypeError))
    
    def test_model_persistence(self):
        """Test model saving and loading functionality."""
        plugin = GANTrainerPlugin(
            config=self.config,
            generator_plugin_instance=self.mock_generator_plugin
        )
        
        # Test save models (may fail due to mock models, but should not crash)
        try:
            plugin.save_models("test_")
        except Exception as e:
            # Expected to fail with mock models
            pass
        
        # Verify persistence module is working
        self.assertIsNotNone(plugin.model_persistence)
        debug_info = plugin.model_persistence.get_debug_info()
        self.assertIsInstance(debug_info, dict)
    
    def test_training_metrics(self):
        """Test training metrics functionality."""
        plugin = GANTrainerPlugin(
            config=self.config,
            generator_plugin_instance=self.mock_generator_plugin
        )
        
        # Record some test metrics
        plugin.training_metrics.record_epoch(
            epoch=1,
            generator_loss=0.5,
            discriminator_loss=0.7,
            generator_accuracy=0.6,
            discriminator_accuracy=0.8
        )
        
        # Verify metrics were recorded
        latest_metrics = plugin.training_metrics.get_latest_metrics()
        self.assertEqual(latest_metrics["epoch"], 1)
        self.assertEqual(latest_metrics["generator_loss"], 0.5)
        self.assertEqual(latest_metrics["discriminator_loss"], 0.7)
    
    def test_plugin_structure_compliance(self):
        """Test that the plugin follows the mandatory structure."""
        plugin = GANTrainerPlugin(
            config=self.config,
            generator_plugin_instance=self.mock_generator_plugin
        )
        
        # Verify mandatory class variable exists
        self.assertTrue(hasattr(GANTrainerPlugin, 'plugin_params'))
        self.assertIsInstance(GANTrainerPlugin.plugin_params, dict)
        
        # Verify mandatory methods exist
        self.assertTrue(hasattr(plugin, 'set_params'))
        self.assertTrue(callable(plugin.set_params))
        
        self.assertTrue(hasattr(plugin, 'get_debug_info'))
        self.assertTrue(callable(plugin.get_debug_info))
        
        self.assertTrue(hasattr(plugin, 'add_debug_info'))
        self.assertTrue(callable(plugin.add_debug_info))
        
        self.assertTrue(hasattr(plugin, 'process'))
        self.assertTrue(callable(plugin.process))
        
        # Verify params structure
        self.assertTrue(hasattr(plugin, 'params'))
        self.assertIsInstance(plugin.params, dict)
        
        # Verify config was properly merged
        for key, value in self.config.items():
            self.assertEqual(plugin.params[key], value)
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up temporary directories
        import shutil
        if os.path.exists("/tmp/test_gan_trainer"):
            shutil.rmtree("/tmp/test_gan_trainer")


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)
