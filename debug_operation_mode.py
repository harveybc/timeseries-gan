#!/usr/bin/env python3

"""
Debug Operation Mode Detection

This script specifically tests whether the operation_mode parameter is correctly 
passed to the generator plugin during model building.
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, '/home/harveybc/Documents/GitHub/timeseries-gan')

from app.config import DEFAULT_VALUES

def test_operation_mode_detection():
    """Test if operation_mode is correctly detected in generator plugin."""
    print("=== Testing Operation Mode Detection ===")
    
    # Test 1: Training Mode
    print("\n1. Testing Training Mode Detection...")
    training_config = DEFAULT_VALUES.copy()
    training_config["operation_mode"] = "train"
    
    print(f"   Config operation_mode: {training_config.get('operation_mode')}")
    
    try:
        from tsg_plugins.generator_plugin.generator_plugin import GeneratorPlugin
        
        # Create generator with training config
        generator_plugin = GeneratorPlugin(training_config)
        
        print(f"   Generator params operation_mode: {generator_plugin.params.get('operation_mode')}")
        print(f"   Generator main_config operation_mode: {generator_plugin.main_config.get('operation_mode')}")
        
    except Exception as e:
        print(f"   ❌ Error in training mode test: {e}")
        return False
    
    # Test 2: Generate Mode  
    print("\n2. Testing Generate Mode Detection...")
    generate_config = DEFAULT_VALUES.copy()
    generate_config["operation_mode"] = "generate"
    
    print(f"   Config operation_mode: {generate_config.get('operation_mode')}")
    
    try:
        # Create generator with generate config
        generator_plugin_gen = GeneratorPlugin(generate_config)
        
        print(f"   Generator params operation_mode: {generator_plugin_gen.params.get('operation_mode')}")
        print(f"   Generator main_config operation_mode: {generator_plugin_gen.main_config.get('operation_mode')}")
        
    except Exception as e:
        print(f"   ❌ Error in generate mode test: {e}")
        return False
    
    print("\n=== Operation Mode Detection Test COMPLETED ===")
    return True

if __name__ == "__main__":
    success = test_operation_mode_detection()
    sys.exit(0 if success else 1)
