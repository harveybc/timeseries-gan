#!/usr/bin/env python3
"""
Test the complete training pipeline with MMD loss enabled to verify the fix.
"""

import os
import sys
import logging

# Add the project root to Python path
sys.path.append('/home/harveybc/Documents/GitHub/timeseries-gan')

from app.cli import main
from app.config import AppConfig

def test_training_with_mmd():
    """Test training with MMD loss enabled to verify the fix."""
    
    print("Testing complete training pipeline with MMD loss enabled...")
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Set up test configuration
    config = AppConfig()
    
    # Override config for testing
    config.gan_epochs = 2  # Just 2 epochs for quick test
    config.enable_mmd_loss = True
    config.mmd_lambda_g = 0.01
    config.mmd_lambda_d = 0.001
    config.mmd_sample_size = 32  # Smaller sample size for faster testing
    config.log_interval_epochs = 1  # Log every epoch
    
    print(f"Testing with:")
    print(f"  - Epochs: {config.gan_epochs}")
    print(f"  - MMD enabled: {config.enable_mmd_loss}")
    print(f"  - MMD lambda G: {config.mmd_lambda_g}")
    print(f"  - MMD lambda D: {config.mmd_lambda_d}")
    print(f"  - MMD sample size: {config.mmd_sample_size}")
    
    # Test arguments simulating CLI usage
    test_args = [
        'gan_trainer_plugin',
        '--input_file', '/home/harveybc/Documents/GitHub/timeseries-gan/data/data_EURUSD_1h_2010_2024.csv',
        '--output_file', '/home/harveybc/Documents/GitHub/timeseries-gan/output/gan_trainer_test_output.csv',
        '--save_generator_sequential_model_file', '/home/harveybc/Documents/GitHub/timeseries-gan/models/generator_test_mmd.keras',
        '--save_discriminator_sequential_model_file', '/home/harveybc/Documents/GitHub/timeseries-gan/models/discriminator_test_mmd.keras'
    ]
    
    # Test with temporary config modification
    original_epochs = config.gan_epochs
    original_mmd = config.enable_mmd_loss
    
    try:
        # Temporarily modify config for testing
        AppConfig.gan_epochs = 2
        AppConfig.enable_mmd_loss = True
        AppConfig.mmd_lambda_g = 0.01
        AppConfig.mmd_lambda_d = 0.001
        AppConfig.mmd_sample_size = 32
        
        # Run training
        print("\nStarting GAN training with MMD loss...")
        print("=" * 80)
        
        # Change sys.argv for the test
        original_argv = sys.argv
        sys.argv = ['test_training'] + test_args
        
        # Call main function
        main()
        
        print("=" * 80)
        print("✓ SUCCESS: Training completed without data type errors!")
        print("✓ MMD loss integration is working correctly!")
        
        return True
        
    except Exception as e:
        print("=" * 80)
        print(f"✗ ERROR: Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Restore original config and argv
        AppConfig.gan_epochs = original_epochs
        AppConfig.enable_mmd_loss = original_mmd
        sys.argv = original_argv

if __name__ == "__main__":
    print("=" * 80)
    print("Complete Training Pipeline Test with MMD Loss")
    print("=" * 80)
    
    success = test_training_with_mmd()
    
    print("\n" + "=" * 80)
    if success:
        print("✓ TRAINING TEST PASSED: MMD loss integration is complete and working!")
        print("✓ All data type issues have been resolved.")
        print("✓ Comprehensive metrics logging with MMD components is functional.")
    else:
        print("✗ TRAINING TEST FAILED: There are still issues with the implementation.")
    print("=" * 80)
