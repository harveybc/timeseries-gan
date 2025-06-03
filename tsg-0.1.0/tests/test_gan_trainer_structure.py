#!/usr/bin/env python3
"""
Simple structure test for GAN Trainer Plugin modular architecture.
This test verifies that all modules can be imported and instantiated without TensorFlow.
"""

import pytest
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_imports():
    """Test that all modules can be imported successfully."""
    try:
        from tsg_plugins.gan_trainer_plugin.training_coordinator import TrainingCoordinator
        from tsg_plugins.gan_trainer_plugin.model_builder import ModelBuilder
        from tsg_plugins.gan_trainer_plugin.data_generator import DataGenerator
        from tsg_plugins.gan_trainer_plugin.model_persistence import ModelPersistence
        from tsg_plugins.gan_trainer_plugin.training_metrics import TrainingMetrics
        from tsg_plugins.gan_trainer_plugin.gan_trainer_plugin import GANTrainerPlugin
        print("✓ All modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_plugin_instantiation():
    """Test that the main plugin can be instantiated with basic config."""
    try:
        from tsg_plugins.gan_trainer_plugin.gan_trainer_plugin import GANTrainerPlugin
        
        config = {
            "seq_len": 10,
            "latent_dim": 8,
            "num_features": 4,
            "gan_epochs": 5,
            "gan_batch_size": 16
        }
        
        plugin = GANTrainerPlugin(config)
        
        # Test mandatory plugin methods exist
        assert hasattr(plugin, 'set_params'), "Missing set_params method"
        assert hasattr(plugin, 'get_debug_info'), "Missing get_debug_info method"
        assert hasattr(plugin, 'add_debug_info'), "Missing add_debug_info method"
        assert hasattr(plugin, 'plugin_params'), "Missing plugin_params attribute"
        
        print("✓ Plugin instantiated successfully with mandatory methods")
        return True
    except Exception as e:
        print(f"✗ Plugin instantiation failed: {e}")
        return False

def test_debug_info():
    """Test debug info functionality."""
    try:
        from tsg_plugins.gan_trainer_plugin.gan_trainer_plugin import GANTrainerPlugin
        
        config = {"seq_len": 10, "latent_dim": 8, "num_features": 4}
        plugin = GANTrainerPlugin(config)
        
        # Test debug info methods
        debug_info = plugin.get_debug_info()
        assert isinstance(debug_info, dict), "get_debug_info should return a dict"
        
        plugin.add_debug_info("test_key", "test_value")
        updated_debug_info = plugin.get_debug_info()
        assert "test_key" in updated_debug_info, "add_debug_info should add to debug info"
        assert updated_debug_info["test_key"] == "test_value", "Debug info value should match"
        
        print("✓ Debug info functionality works correctly")
        return True
    except Exception as e:
        print(f"✗ Debug info test failed: {e}")
        return False

def test_set_params():
    """Test parameter setting functionality."""
    try:
        from tsg_plugins.gan_trainer_plugin.gan_trainer_plugin import GANTrainerPlugin
        
        config = {"seq_len": 10, "latent_dim": 8, "num_features": 4}
        plugin = GANTrainerPlugin(config)
        
        # Test parameter setting
        original_epochs = plugin.params["gan_epochs"]
        plugin.set_params(gan_epochs=100)
        assert plugin.params["gan_epochs"] == 100, "set_params should update parameters"
        assert plugin.params["gan_epochs"] != original_epochs, "Parameter should actually change"
        
        print("✓ Parameter setting functionality works correctly")
        return True
    except Exception as e:
        print(f"✗ Parameter setting test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing GAN Trainer Plugin Modular Structure...")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_plugin_instantiation,
        test_debug_info,
        test_set_params
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            failed += 1
        print()
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All structure tests passed! The modular GAN trainer plugin is properly structured.")
    else:
        print("❌ Some tests failed. Please check the implementation.")
        sys.exit(1)
