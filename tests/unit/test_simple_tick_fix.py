#!/usr/bin/env python3
"""
Simple test to verify constraint-based tick generation fix
"""

import sys
import os
sys.path.append('/home/harveybc/Documents/GitHub/timeseries-gan')

import numpy as np
import tensorflow as tf
from app.config import DEFAULT_VALUES

def test_constraint_based_tick_generation():
    """Test the constraint-based tick generation implementation"""
    
    print("🧪 TESTING CONSTRAINT-BASED TICK GENERATION")
    print("=" * 50)
    
    try:
        # Import the FeatureExpansionLayer
        from tsg_plugins.generator_plugin.generator_plugin import FeatureExpansionLayer
        
        print("✅ Successfully imported FeatureExpansionLayer")
        
        # Create a test instance
        expansion_layer = FeatureExpansionLayer()
        
        # Create mock VAE decoder output (batch_size=2, features=23) 
        batch_size = 2
        test_input = tf.random.normal([batch_size, 23], dtype=tf.float32)
        
        print(f"✅ Created test input: {test_input.shape}")
        
        # Test the expansion
        expanded_output = expansion_layer(test_input)
        
        print(f"✅ Expansion successful: {test_input.shape} -> {expanded_output.shape}")
        
        # Verify output shape
        if expanded_output.shape == (batch_size, 44):
            print("✅ Output shape is correct (batch_size, 44)")
            
            # Extract tick columns (positions 28-43: 16 tick features)
            tick_data = expanded_output[:, 28:44]  # Last 16 features are ticks
            
            print(f"✅ Extracted tick data: {tick_data.shape}")
            
            # Check if tick columns are populated (not all zeros)
            tick_populated = not tf.reduce_all(tf.equal(tick_data, 0.0))
            
            if tick_populated:
                print("✅ SUCCESS: Tick columns are populated!")
                print("✅ Constraint-based tick generation is working!")
                
                # Show sample tick values
                sample_ticks = tick_data[0].numpy()
                print(f"Sample tick values: {sample_ticks[:8]}")  # First 8 ticks
                
                return True
            else:
                print("❌ FAILURE: Tick columns are empty!")
                return False
        else:
            print(f"❌ FAILURE: Wrong output shape! Expected ({batch_size}, 44), got {expanded_output.shape}")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ohlc_constraints():
    """Test OHLC constraint satisfaction"""
    
    print("\n🧪 TESTING OHLC CONSTRAINTS")
    print("=" * 30)
    
    try:
        from tsg_plugins.generator_plugin.generator_plugin import FeatureExpansionLayer
        
        expansion_layer = FeatureExpansionLayer()
        
        # Create test data with known OHLC values
        test_input = tf.constant([
            [0.5, 0.3, 0.7, 0.6, 0.1, 0.2, 0.4] + [0.0] * 16  # OPEN, LOW, HIGH, others...
        ], dtype=tf.float32)
        
        print(f"✅ Created test OHLC input: O=0.5, L=0.3, H=0.7")
        
        expanded_output = expansion_layer(test_input)
        
        # Extract OHLC from expanded output (positions 15-18: OHLC after technical indicators)
        output_ohlc = expanded_output[0, 15:19].numpy()  
        
        print(f"✅ Output OHLC: O={output_ohlc[0]:.3f}, H={output_ohlc[1]:.3f}, L={output_ohlc[2]:.3f}, C={output_ohlc[3]:.3f}")
        
        # Verify OHLC constraints: L <= O,C <= H
        open_val, high_val, low_val, close_val = output_ohlc
        
        valid_constraints = (
            low_val <= open_val <= high_val and
            low_val <= close_val <= high_val
        )
        
        if valid_constraints:
            print("✅ SUCCESS: OHLC constraints are satisfied!")
            return True
        else:
            print("❌ FAILURE: OHLC constraints violated!")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

if __name__ == "__main__":
    print("🚀 STARTING CONSTRAINT-BASED TICK GENERATION TESTS")
    print("=" * 60)
    
    success1 = test_constraint_based_tick_generation()
    success2 = test_ohlc_constraints()
    
    print(f"\n🏁 FINAL RESULTS:")
    print("=" * 20)
    
    if success1 and success2:
        print("✅ ALL TESTS PASSED!")
        print("✅ Constraint-based tick generation is working correctly!")
        exit(0)
    else:
        print("❌ SOME TESTS FAILED!")
        print("❌ Issues still need to be resolved!")
        exit(1)
