#!/usr/bin/env python3
"""
Integration tests for the training pipeline.
Tests the complete plugin loading and GAN training workflow.
"""

import pytest
import sys
import os
import logging
from unittest.mock import Mock, patch
import numpy as np

# Add the project root to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from app.main import load_and_initialize_plugins
from app.config import DEFAULT_VALUES


# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)


@pytest.fixture
def test_config():
    """Create a test configuration."""
    config = DEFAULT_VALUES.copy()
    config.update({
        'feeder': 'default_feeder',
        'generator': 'default_generator', 
        'discriminator': 'default_discriminator',
        'trainer': 'gan_trainer',
        'gan_epochs': 1,
        'gan_batch_size': 4,
        'x_train_file': 'tests/data/sample_data.csv'
    })
    return config


def test_plugin_loading(test_config):
    """Test that all plugins can be loaded successfully."""
    print("Testing plugin loading...")
    
    try:
        plugins = load_and_initialize_plugins(test_config)
        
        # Check that all expected plugins are loaded
        expected_plugins = ['feeder_plugin', 'generator_plugin', 'discriminator_plugin', 'trainer_plugin']
        for plugin_name in expected_plugins:
            assert plugin_name in plugins, f"Plugin {plugin_name} not found in loaded plugins"
            print(f"✓ {plugin_name} loaded successfully")
        
        # Check that plugins have required methods
        if plugins['generator_plugin']:
            assert hasattr(plugins['generator_plugin'], 'get_model'), "GeneratorPlugin missing get_model method"
            print("✓ GeneratorPlugin has get_model method")
            
        if plugins['discriminator_plugin']:
            assert hasattr(plugins['discriminator_plugin'], 'get_model'), "DiscriminatorPlugin missing get_model method"  
            print("✓ DiscriminatorPlugin has get_model method")
            
        print("✓ All plugin loading tests passed")
        
    except Exception as e:
        print(f"❌ Plugin loading failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_plugin_get_model_methods(test_config):
    """Test that plugin get_model methods work correctly."""
    print("Testing plugin get_model methods...")
    
    try:
        plugins = load_and_initialize_plugins(test_config)
        
        # Test GeneratorPlugin get_model
        if plugins['generator_plugin']:
            gen_model = plugins['generator_plugin'].get_model()
            print(f"✓ GeneratorPlugin.get_model() returned: {type(gen_model)}")
            
        # Test DiscriminatorPlugin get_model  
        if plugins['discriminator_plugin']:
            disc_model = plugins['discriminator_plugin'].get_model()
            print(f"✓ DiscriminatorPlugin.get_model() returned: {type(disc_model)}")
            
        print("✓ All plugin get_model tests passed")
        
    except Exception as e:
        print(f"❌ Plugin get_model tests failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_gan_trainer_plugin_interface(test_config):
    """Test that GANTrainerPlugin can access generator and discriminator models."""
    print("Testing GANTrainerPlugin interface...")
    
    try:
        plugins = load_and_initialize_plugins(test_config)
        trainer = plugins.get('trainer_plugin')
        
        if trainer and hasattr(trainer, 'plugin_interface'):
            # Test generator model access
            gen_model = trainer.plugin_interface.get_generator_model()
            print(f"✓ GANTrainer can access generator model: {type(gen_model)}")
            
            # Test discriminator model access
            disc_model = trainer.plugin_interface.get_discriminator_model()
            print(f"✓ GANTrainer can access discriminator model: {type(disc_model)}")
            
        print("✓ GANTrainerPlugin interface tests passed")
        
    except Exception as e:
        print(f"❌ GANTrainerPlugin interface tests failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    # Run tests directly
    test_config = {
        'feeder': 'default_feeder',
        'generator': 'default_generator', 
        'discriminator': 'default_discriminator',
        'trainer': 'gan_trainer',
        'gan_epochs': 1,
        'gan_batch_size': 4,
        'x_train_file': 'tests/data/sample_data.csv'
    }
    
    try:
        test_plugin_loading(test_config)
        test_plugin_get_model_methods(test_config)
        test_gan_trainer_plugin_interface(test_config)
        print("\n🎉 All tests passed!")
    except Exception as e:
        print(f"\n❌ Tests failed: {e}")
        sys.exit(1)
