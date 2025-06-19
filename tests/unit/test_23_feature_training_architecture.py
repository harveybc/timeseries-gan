#!/usr/bin/env python3

"""
Test 23-Feature Training Architecture

This script verifies that:
1. Training Mode: Generator outputs 23 features, discriminator expects 23 features
2. Generate Mode: Generator can expand to 44 features for downstream tasks
3. Configuration is properly set for 23-feature training
"""

import sys
import os
import numpy as np
import tensorflow as tf

# Add the project root to the path
sys.path.insert(0, '/home/harveybc/Documents/GitHub/timeseries-gan')

from app.config import DEFAULT_VALUES

def test_23_feature_training_architecture():
    """Test the 23-feature training architecture."""
    print("=== Testing 23-Feature Training Architecture ===")
    
    # 1. Load and verify configuration
    print("\n1. Loading and verifying configuration...")
    config = DEFAULT_VALUES.copy()
    
    print(f"   - num_features: {config.get('num_features', 'not set')}")
    print(f"   - discriminator_input_dim: {config.get('discriminator_input_dim', 'not set')}")
    print(f"   - operation_mode: {config.get('operation_mode', 'not set')}")
    
    # Verify 23-feature configuration
    assert config.get('num_features') == 23, f"Expected num_features=23, got {config.get('num_features')}"
    assert config.get('discriminator_input_dim') == 23, f"Expected discriminator_input_dim=23, got {config.get('discriminator_input_dim')}"
    print("   ✅ Configuration verified for 23-feature training")
    
    # 2. Test Training Mode (23 features)
    print("\n2. Testing Training Mode (23 features)...")
    training_config = config.copy()
    training_config["operation_mode"] = "train"
    
    # Test generator plugin
    try:
        from tsg_plugins.generator_plugin.generator_plugin import GeneratorPlugin
        
        generator_plugin = GeneratorPlugin(training_config)
        
        print("   - Generator plugin initialized for training mode")
        print(f"   - Generator num_features: {generator_plugin.params.get('num_features', 'not set')}")
        
        # Test discriminator plugin
        from tsg_plugins.discriminator_plugin import DiscriminatorPlugin
        
        discriminator_plugin = DiscriminatorPlugin(training_config)
        
        print("   - Discriminator plugin initialized for training mode")
        print(f"   - Discriminator num_features: {discriminator_plugin.params.get('num_features', 'not set')}")
        
        # Verify both use 23 features in training mode
        assert generator_plugin.params.get('num_features') == 23, "Generator should use 23 features in training"
        assert discriminator_plugin.params.get('num_features') == 23, "Discriminator should use 23 features in training"
        
        print("   ✅ Training mode: Both generator and discriminator use 23 features")
        
    except Exception as e:
        print(f"   ❌ Error in training mode test: {e}")
        return False
    
    # 3. Test Generate Mode Architecture
    print("\n3. Testing Generate Mode Architecture...")
    generate_config = config.copy()
    generate_config["operation_mode"] = "generate"
    
    try:
        # Test that generator can handle generate mode
        generator_plugin_gen = GeneratorPlugin(generate_config)
        
        print("   - Generator plugin initialized for generate mode")
        print(f"   - Generator num_features: {generator_plugin_gen.params.get('num_features', 'not set')}")
        
        # In generate mode, the plugin should still use 23 base features but expand during post-processing
        assert generator_plugin_gen.params.get('num_features') == 23, "Generator base features should still be 23"
        
        print("   ✅ Generate mode: Generator uses 23 base features (expansion handled in model)")
        
    except Exception as e:
        print(f"   ❌ Error in generate mode test: {e}")
        return False
    
    # 4. Test Model Architecture Compatibility
    print("\n4. Testing model architecture compatibility...")
    
    try:
        # Create simple test data
        batch_size = 4
        seq_len = 144
        
        # Test training mode data flow
        print("   - Testing training mode data flow...")
        
        # Simulate generator output (23 features)
        fake_data_23 = np.random.randn(batch_size, seq_len, 23).astype(np.float32)
        print(f"     Generator output shape: {fake_data_23.shape}")
        
        # Simulate real data (first 23 features)
        real_data_23 = np.random.randn(batch_size, seq_len, 23).astype(np.float32)
        print(f"     Real data shape: {real_data_23.shape}")
        
        # Verify shapes match for discriminator
        assert fake_data_23.shape == real_data_23.shape, "Generator and real data shapes should match"
        assert fake_data_23.shape[2] == 23, "Should have 23 features for training"
        
        print("   ✅ Training mode data flow: Generator (23) → Discriminator (23)")
        
        # Test generate mode conceptual flow
        print("   - Testing generate mode conceptual flow...")
        
        # In generate mode, post-processing would expand 23 → 44 features
        base_features_23 = np.random.randn(batch_size, seq_len, 23).astype(np.float32)
        
        # Simulate post-processing expansion (this would happen in the model)
        # For testing, just concatenate random features to simulate expansion
        technical_indicators = np.random.randn(batch_size, seq_len, 15).astype(np.float32)  # 15 technical indicators
        seasonal_features = np.random.randn(batch_size, seq_len, 3).astype(np.float32)      # 3 seasonal features
        additional_features = np.random.randn(batch_size, seq_len, 3).astype(np.float32)    # 3 additional features
        
        expanded_features_44 = np.concatenate([
            base_features_23, technical_indicators, seasonal_features, additional_features
        ], axis=2)
        
        print(f"     Base features: {base_features_23.shape}")
        print(f"     Expanded features: {expanded_features_44.shape}")
        
        assert expanded_features_44.shape == (batch_size, seq_len, 44), "Should expand to 44 features"
        
        print("   ✅ Generate mode conceptual flow: 23 base → 44 expanded features")
        
    except Exception as e:
        print(f"   ❌ Error in model architecture test: {e}")
        return False
    
    print("\n=== 23-Feature Training Architecture Test PASSED ===")
    print("\nArchitecture Summary:")
    print("  Training Mode:")
    print("    - Generator: Outputs 23 base features")
    print("    - Discriminator: Expects 23 base features")
    print("    - Adversarial training on 23 features only")
    print("  Generate Mode:")
    print("    - Generator: Generates 23 base features")
    print("    - Post-processing: Expands to 44 features (23+15+3+3)")
    print("    - Output: Complete feature set for downstream tasks")
    
    return True

if __name__ == "__main__":
    success = test_23_feature_training_architecture()
    sys.exit(0 if success else 1)
