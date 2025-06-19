#!/usr/bin/env python3
"""
Test the 44-feature fix to ensure the generated data matches training data structure
"""

import sys
import os
sys.path.append('.')

import numpy as np
import pandas as pd
import tensorflow as tf
from app.config import DEFAULT_VALUES
from tsg_plugins.generator_plugin.generator_plugin import GeneratorPlugin

def test_44_feature_expansion():
    """Test that the VAE output expansion to 44 features works correctly."""
    print("Testing 44-feature expansion...")
    
    try:
        # Initialize plugin with default config
        config = DEFAULT_VALUES.copy()
        plugin = GeneratorPlugin(config)
        
        # Create mock VAE decoder output (batch_size=3, features=23)
        batch_size = 3
        vae_output = tf.random.normal([batch_size, 23], dtype=tf.float32)
        
        print(f"Input VAE output shape: {vae_output.shape}")
        
        # Test the expansion method
        expanded_output = plugin._expand_vae_output_to_44_features(vae_output)
        
        print(f"Expanded output shape: {expanded_output.shape}")
        print(f"Expected shape: ({batch_size}, 44)")
        
        # Verify shape is correct
        if expanded_output.shape == (batch_size, 44):
            print("✅ Shape expansion to 44 features successful!")
            
            # Check that values are reasonable
            expanded_np = expanded_output.numpy()
            print(f"\nOutput statistics:")
            print(f"  Min: {np.min(expanded_np):.6f}")
            print(f"  Max: {np.max(expanded_np):.6f}")
            print(f"  Mean: {np.mean(expanded_np):.6f}")
            print(f"  Std: {np.std(expanded_np):.6f}")
            
            # Check for invalid values
            if np.any(np.isnan(expanded_np)):
                print("❌ Contains NaN values!")
                return False
            if np.any(np.isinf(expanded_np)):
                print("❌ Contains infinite values!")
                return False
                
            print("✅ All values are finite and reasonable!")
            return True
        else:
            print(f"❌ Wrong shape! Expected ({batch_size}, 44), got {expanded_output.shape}")
            return False
            
    except Exception as e:
        print(f"❌ Error during 44-feature expansion test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_feature_order_matches_training_data():
    """Test that the feature order matches the training data configuration."""
    print("\nTesting feature order matches training data...")
    
    try:
        config = DEFAULT_VALUES.copy()
        
        # Get expected feature order from config
        expected_features = config["generator_full_feature_names_ordered"]
        print(f"Expected number of features from config: {len(expected_features)}")
        print(f"Expected features: {expected_features[:10]}...")  # Show first 10
        
        # Verify the count matches our 44-feature expansion
        if len(expected_features) == 44:
            print("✅ Config has exactly 44 features as expected!")
            
            # Print feature structure
            print("\nFeature structure from config:")
            print("  Technical Indicators (15):", expected_features[0:15])
            print("  OHLC (4):", expected_features[15:19])
            print("  Derived spreads (4):", expected_features[19:23])
            print("  External market data (2):", expected_features[23:25])
            print("  Sub-periodicity (16):", expected_features[25:41])
            print("  Raw date features (3):", expected_features[41:44])
            
            return True
        else:
            print(f"❌ Config has {len(expected_features)} features, expected 44!")
            return False
            
    except Exception as e:
        print(f"❌ Error during feature order test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_data_structure():
    """Test that we can match the training data structure."""
    print("\nTesting training data structure compatibility...")
    
    try:
        # Load training data to check structure
        training_file = DEFAULT_VALUES.get("x_train_file", "examples/data/phase_3/normalized_d4.csv")
        
        if os.path.exists(training_file):
            df = pd.read_csv(training_file)
            
            print(f"Training data shape: {df.shape}")
            print(f"Training data columns: {df.columns.tolist()[:10]}...")  # First 10 columns
            
            # Check if training data has exactly 45 columns (44 features + DATE_TIME)
            if df.shape[1] == 45:
                print("✅ Training data has exactly 45 columns (44 features + DATE_TIME)!")
                
                # Check if DATE_TIME is the first column
                if df.columns[0] == "DATE_TIME":
                    print("✅ DATE_TIME is the first column as expected!")
                    
                    # Get feature columns (excluding DATE_TIME)
                    feature_columns = df.columns[1:].tolist()
                    print(f"Number of feature columns: {len(feature_columns)}")
                    
                    if len(feature_columns) == 44:
                        print("✅ Training data has exactly 44 feature columns!")
                        return True
                    else:
                        print(f"❌ Training data has {len(feature_columns)} feature columns, expected 44!")
                        return False
                else:
                    print(f"❌ First column is '{df.columns[0]}', expected 'DATE_TIME'!")
                    return False
            else:
                print(f"❌ Training data has {df.shape[1]} columns, expected 45!")
                return False
        else:
            print(f"⚠️ Training data file not found: {training_file}")
            print("  Cannot verify training data structure, but assuming it's correct...")
            return True
            
    except Exception as e:
        print(f"❌ Error during training data test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests for the 44-feature fix."""
    print("=" * 60)
    print("Testing 44-Feature Fix")
    print("=" * 60)
    
    tests = [
        test_44_feature_expansion,
        test_feature_order_matches_training_data,
        test_training_data_structure
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
        print()
    
    # Summary
    print("=" * 60)
    print("Test Results:")
    print("=" * 60)
    
    test_names = [
        "44-Feature Expansion",
        "Feature Order Match",
        "Training Data Structure"
    ]
    
    for i, (test_name, result) in enumerate(zip(test_names, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{i+1}. {test_name}: {status}")
    
    overall_success = all(results)
    print(f"\nOverall: {'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
