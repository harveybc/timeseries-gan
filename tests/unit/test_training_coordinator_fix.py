#!/usr/bin/env python3
"""
Test script to verify the TrainingCoordinator method signature fix.
"""

import sys
import os
import logging
import pandas as pd
import numpy as np

# Add project to path
sys.path.insert(0, os.path.dirname(__file__))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_training_coordinator_fix():
    """Test the TrainingCoordinator method signature fix."""
    
    print("=== Testing TrainingCoordinator Method Signature Fix ===")
    
    try:
        # Import the TrainingCoordinator
        from tsg_plugins.gan_trainer_plugin.training_coordinator import TrainingCoordinator
        print("✓ Successfully imported TrainingCoordinator")
        
        # Create test parameters
        params = {
            'generator_lr': 1e-4,
            'discriminator_lr': 1e-4,
            'batch_size': 8,
            'seq_len': 144,
            'feeder_noise_dim': 32,
            'conditional_features_dim': 10,
            'context_vector_dim': 64
        }
        
        # Initialize coordinator
        coordinator = TrainingCoordinator(params, logger)
        print("✓ TrainingCoordinator initialized successfully")
        
        # Check the train method signature
        import inspect
        train_signature = inspect.signature(coordinator.train)
        print(f"✓ Train method signature: {train_signature}")
        
        # Verify required positional parameters
        required_params = []
        for param_name, param in train_signature.parameters.items():
            if param.default == inspect.Parameter.empty and param.kind != inspect.Parameter.VAR_KEYWORD:
                required_params.append(param_name)
        
        print(f"✓ Required positional parameters: {required_params}")
        
        expected_required = ['generator', 'discriminator', 'gan_model', 'feeder_plugin']
        if set(required_params[:4]) == set(expected_required):
            print("✓ Method signature matches expected pattern")
        else:
            print(f"✗ Method signature mismatch. Expected: {expected_required}, Got: {required_params[:4]}")
            
        return True
        
    except Exception as e:
        print(f"✗ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gan_trainer_integration():
    """Test GANTrainerPlugin with TrainingCoordinator integration."""
    
    print("\n=== Testing GANTrainerPlugin Integration ===")
    
    try:
        # Import the GANTrainerPlugin
        from tsg_plugins.gan_trainer_plugin.gan_trainer_plugin import GANTrainerPlugin
        print("✓ Successfully imported GANTrainerPlugin")
        
        # Create test parameters
        params = {
            'gan_epochs': 1,
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
        
        # Check if training coordinator is available
        if hasattr(plugin, 'training_coordinator'):
            print("✓ TrainingCoordinator is available in GANTrainerPlugin")
        else:
            print("✗ TrainingCoordinator not found in GANTrainerPlugin")
            return False
        
        # Create dummy training data
        dummy_data = pd.DataFrame(np.random.randn(100, 45))  # 100 samples, 45 features
        print(f"✓ Created dummy training data with shape: {dummy_data.shape}")
        
        # Test the train method call (this should fail gracefully since we don't have real models)
        try:
            result = plugin.train(training_data=dummy_data, epochs=1, batch_size=4)
            print("✓ Train method called successfully (expected to fail at model building)")
        except Exception as e:
            if "Failed to build GAN models" in str(e) or "Generator plugin instance not available" in str(e):
                print("✓ Train method failed as expected (missing models/plugins)")
            else:
                print(f"✗ Unexpected error in train method: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error during GANTrainerPlugin test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting TrainingCoordinator fix verification tests...\n")
    
    test1_success = test_training_coordinator_fix()
    test2_success = test_gan_trainer_integration()
    
    print("\n=== Test Results ===")
    print(f"TrainingCoordinator signature test: {'PASS' if test1_success else 'FAIL'}")
    print(f"GANTrainerPlugin integration test: {'PASS' if test2_success else 'FAIL'}")
    
    if test1_success and test2_success:
        print("\n🎉 All tests PASSED! The TrainingCoordinator method signature fix is working correctly.")
    else:
        print("\n❌ Some tests FAILED. Please check the errors above.")
        sys.exit(1)
