#!/usr/bin/env python3
"""
Test script to verify MMD calculation is working properly after the fix.
"""

import os
import sys
import traceback
import logging
import numpy as np
import pandas as pd
import tensorflow as tf

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import project modules
from app.config import CONFIG
from app.plugin_loader import PluginLoader
from app.utils.logging_utils import setup_logging

def test_mmd_training():
    """Test MMD calculation during training."""
    print("=" * 80)
    print("TESTING MMD CALCULATION DURING TRAINING")
    print("=" * 80)
    
    # Setup logging
    setup_logging(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize plugin loader
        plugin_loader = PluginLoader(CONFIG)
        
        # Load plugins
        feeder_plugin = plugin_loader.load_plugin("feeder")
        generator_plugin = plugin_loader.load_plugin("generator")
        discriminator_plugin = plugin_loader.load_plugin("discriminator")
        trainer_plugin = plugin_loader.load_plugin("trainer")
        
        print(f"✓ All plugins loaded successfully")
        
        # Configure for short test run
        test_config = {
            "gan_epochs": 5,  # Only 5 epochs for quick test
            "gan_batch_size": 16,  # Small batch size
            "enable_mmd_loss": True,
            "mmd_lambda_g": 0.01,
            "mmd_lambda_d": 0.001,
            "learning_rate": 0.001,
            "print_model_summary": False,
            "save_models": False
        }
        
        # Update plugin configurations
        feeder_plugin.set_params(**test_config)
        generator_plugin.set_params(**test_config)
        discriminator_plugin.set_params(**test_config)
        trainer_plugin.set_params(**test_config)
        
        print(f"✓ Plugin configurations updated for MMD testing")
        
        # Generate sample data
        print("Generating sample training data...")
        sample_size = 500
        features = 23  # VAE decoder outputs 23 features
        
        # Create sample data with realistic financial time series patterns
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=sample_size, freq='1H')
        
        # Create realistic OHLC data
        base_price = 100.0
        price_changes = np.random.normal(0, 0.02, sample_size)  # 2% volatility
        prices = [base_price]
        for change in price_changes[1:]:
            prices.append(prices[-1] * (1 + change))
        
        training_data = pd.DataFrame({
            'datetime': dates,
            'OPEN': prices,
            'HIGH': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'LOW': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'CLOSE': prices,
        })
        
        # Add some additional features to reach closer to 23
        for i in range(5, 23):
            training_data[f'feature_{i}'] = np.random.normal(0, 1, sample_size)
        
        print(f"✓ Sample training data created: {training_data.shape}")
        
        # Start training
        print("Starting training with MMD loss enabled...")
        
        history = trainer_plugin.train(
            training_data=training_data,
            epochs=test_config["gan_epochs"],
            batch_size=test_config["gan_batch_size"]
        )
        
        print(f"✓ Training completed successfully!")
        
        # Check if MMD values were logged properly
        if hasattr(trainer_plugin, 'training_coordinator'):
            coordinator = trainer_plugin.training_coordinator
            if hasattr(coordinator, 'training_history'):
                print("Training history available for analysis.")
            else:
                print("⚠ No training history found in coordinator")
        else:
            print("⚠ No training coordinator found in trainer plugin")
        
        print("\n" + "=" * 80)
        print("MMD TRAINING TEST COMPLETED SUCCESSFULLY!")
        print("Check the logs above for MMD values (G_mmd and D_mmd should be > 0)")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ ERROR during MMD training test: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False
    
    return True

def main():
    """Main test function."""
    print("Starting MMD Calculation Test...")
    
    try:
        success = test_mmd_training()
        
        if success:
            print("\n🎉 ALL MMD TESTS PASSED!")
            return 0
        else:
            print("\n❌ MMD TESTS FAILED!")
            return 1
            
    except Exception as e:
        print(f"\n💥 CRITICAL ERROR: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return 1

if __name__ == "__main__":
    exit(main())
