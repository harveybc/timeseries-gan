#!/usr/bin/env python3
"""
Simple test script for the 23-feature architecture.
"""

import sys
import os
import logging
import numpy as np
import tensorflow as tf

# Add the project root to the path
sys.path.insert(0, os.path.abspath('.'))

def main():
    print("Testing 23-Feature Architecture Implementation")
    print("=" * 60)
    
    try:
        from app.config import DEFAULT_VALUES
        print("✅ Config imported successfully")
        
        # Test configuration
        config = DEFAULT_VALUES.copy()
        print(f"✅ Base config loaded")
        print(f"   - Original num_features: {config.get('num_features', 'not set')}")
        
        # Update for 23-feature architecture
        config.update({
            "num_features": 23,
            "sequence_length": 144,
        })
        print(f"✅ Config updated for 23 features")
        print(f"   - New num_features: {config['num_features']}")
        
        # Test discriminator plugin import
        from tsg_plugins.discriminator_plugin import DiscriminatorPlugin
        print("✅ DiscriminatorPlugin imported successfully")
        
        # Create discriminator with 23 features
        discriminator_plugin = DiscriminatorPlugin(config)
        print("✅ DiscriminatorPlugin created successfully")
        
        # Test model building
        discriminator_plugin.build_model()
        discriminator = discriminator_plugin.get_model()
        
        if discriminator is not None:
            print("✅ Discriminator built successfully!")
            print(f"   - Input shape: {discriminator.input_shape}")
            print(f"   - Output shape: {discriminator.output_shape}")
            print(f"   - Parameters: {discriminator.count_params():,}")
            
            # Test with sample data
            batch_size = 2
            sample_data = np.random.randn(batch_size, 144, 23).astype(np.float32)
            predictions = discriminator.predict(sample_data, verbose=0)
            
            print(f"✅ Discriminator prediction test passed!")
            print(f"   - Input shape: {sample_data.shape}")
            print(f"   - Output shape: {predictions.shape}")
            
            return True
        else:
            print("❌ Failed to build discriminator")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)
