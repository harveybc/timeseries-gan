#!/usr/bin/env python3
"""
Test script to verify generator plugin works with proper configuration
"""

import sys
import os
sys.path.append('.')

from app.config import DEFAULT_VALUES
from tsg_plugins.generator_plugin.generator_plugin import GeneratorPlugin

def test_generator_with_vae_config():
    """Test generator plugin with VAE decoder configuration"""
    print("Testing GeneratorPlugin with VAE decoder configuration...")
    
    # Use the complete default configuration (no need to override feature lists)
    config = DEFAULT_VALUES.copy()
    
    try:
        # Initialize generator plugin
        print("1. Initializing GeneratorPlugin...")
        generator = GeneratorPlugin(config)
        print("✓ GeneratorPlugin initialized successfully")
        
        # Set parameters (mimicking what main.py does) 
        print("2. Setting parameters...")
        generator.set_params(**config)
        print("✓ Parameters set successfully")
        
        # Try to get the model
        print("3. Getting generator model...")
        model = generator.get_model()
        
        if model is not None:
            print(f"✓ Generator model obtained successfully!")
            print(f"  Model type: {type(model)}")
            print(f"  Model name: {model.name}")
            print(f"  Input shapes: {[inp.shape for inp in model.inputs]}")
            print(f"  Output shape: {model.output.shape}")
            print(f"  Total parameters: {model.count_params()}")
            return True
        else:
            print("❌ Generator model is None")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_generator_with_vae_config()
    if success:
        print("\n🎉 Generator plugin test PASSED!")
    else:
        print("\n💥 Generator plugin test FAILED!")
    sys.exit(0 if success else 1)
