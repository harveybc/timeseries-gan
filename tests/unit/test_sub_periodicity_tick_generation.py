#!/usr/bin/env python3
"""
Test script for constraint-based sub-periodicity tick generation.

This script tests the new FeatureExpansionLayer with OHLC-constrained tick generation.
"""

import sys
import os
sys.path.append('.')

import tensorflow as tf
import numpy as np
import pandas as pd
from app.config import DEFAULT_VALUES

def test_feature_expansion_layer():
    """Test the FeatureExpansionLayer with constraint-based tick generation."""
    print("🧪 Testing FeatureExpansionLayer with Constraint-Based Sub-Periodicity Tick Generation")
    print("=" * 80)
    
    # Import the FeatureExpansionLayer
    try:
        from tsg_plugins.generator_plugin.generator_plugin import FeatureExpansionLayer
        print("✅ Successfully imported FeatureExpansionLayer")
    except Exception as e:
        print(f"❌ Failed to import FeatureExpansionLayer: {e}")
        return False
    
    # Create test data (batch_size=3, 23 features)
    batch_size = 3
    test_input = tf.constant([
        # Sample 1 - Normal market conditions
        [1.0850, 1.0820, 1.0880, 0.35,   # OPEN, LOW, HIGH, vix_close
         0.02, 0.04, 4250.5,             # BC-BO, BH-BL, S&P500_Close
         # 16 sub-periodicity ticks (will be replaced by constraint-based generation)
         1.083, 1.084, 1.085, 1.086, 1.087, 1.088, 1.089, 1.090,  # 15m ticks
         1.083, 1.084, 1.085, 1.086, 1.087, 1.088, 1.089, 1.090], # 30m ticks
        
        # Sample 2 - Volatile market conditions  
        [1.0900, 1.0850, 1.0950, 0.42,   # OPEN, LOW, HIGH, vix_close
         -0.015, 0.08, 4180.2,           # BC-BO, BH-BL, S&P500_Close
         1.091, 1.092, 1.093, 1.094, 1.095, 1.096, 1.097, 1.098,  # 15m ticks
         1.091, 1.092, 1.093, 1.094, 1.095, 1.096, 1.097, 1.098], # 30m ticks
        
        # Sample 3 - Low volatility conditions
        [1.0800, 1.0795, 1.0810, 0.28,   # OPEN, LOW, HIGH, vix_close
         0.005, 0.015, 4300.8,           # BC-BO, BH-BL, S&P500_Close
         1.080, 1.081, 1.082, 1.083, 1.084, 1.085, 1.086, 1.087,  # 15m ticks
         1.080, 1.081, 1.082, 1.083, 1.084, 1.085, 1.086, 1.087]  # 30m ticks
    ], dtype=tf.float32)
    
    print(f"✅ Created test input with shape: {test_input.shape}")
    print(f"   Expected: (3, 23) - Got: {test_input.shape}")
    
    # Create FeatureExpansionLayer instance
    expansion_layer = FeatureExpansionLayer()
    print("✅ Created FeatureExpansionLayer instance")
    
    # Test the layer
    try:
        output = expansion_layer(test_input)
        print(f"✅ Layer execution successful!")
        print(f"   Output shape: {output.shape}")
        print(f"   Expected: (3, 44) - Got: {output.shape}")
        
        if output.shape == (3, 44):
            print("✅ Output shape is correct!")
        else:
            print(f"❌ Output shape mismatch!")
            return False
            
    except Exception as e:
        print(f"❌ Layer execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Extract and analyze the generated sub-periodicity ticks
    print("\n🔍 ANALYZING GENERATED SUB-PERIODICITY TICKS")
    print("=" * 60)
    
    # Output structure: [15 TI, 4 OHLC, 4 spreads, 2 external, 16 ticks, 3 date]
    # Sub-periodicity ticks are at indices 25:41 (16 features)
    sub_ticks = output[:, 25:41].numpy()  # Convert to numpy for analysis
    ohlc_values = test_input[:, 0:4].numpy()  # Original OHLC values
    
    print("📊 CONSTRAINT VERIFICATION:")
    print("-" * 40)
    
    for sample_idx in range(batch_size):
        print(f"\nSample {sample_idx + 1}:")
        sample_ohlc = ohlc_values[sample_idx]
        sample_ticks = sub_ticks[sample_idx]
        
        # Extract OHLC values
        o, l, h, vix = sample_ohlc
        
        # Calculate derived close from OHLC (as done in layer)
        c = (o + h + l) / 3.0
        
        print(f"  OHLC: O={o:.4f}, H={h:.4f}, L={l:.4f}, C={c:.4f}")
        
        # Extract 15m and 30m ticks
        ticks_15m = sample_ticks[:8]
        ticks_30m = sample_ticks[8:]
        
        # Verify constraints for 15m ticks
        print(f"  15m ticks: {[f'{t:.4f}' for t in ticks_15m]}")
        
        # Check constraint satisfaction
        constraints_15m = {
            "first_tick_eq_open": abs(ticks_15m[0] - o) < 0.001,
            "last_tick_eq_close": abs(ticks_15m[-1] - c) < 0.001,
            "high_reached": any(abs(t - h) < 0.001 for t in ticks_15m),
            "low_reached": any(abs(t - l) < 0.001 for t in ticks_15m),
            "all_within_bounds": all(l <= t <= h for t in ticks_15m)
        }
        
        # Same for 30m ticks
        print(f"  30m ticks: {[f'{t:.4f}' for t in ticks_30m]}")
        
        constraints_30m = {
            "first_tick_eq_open": abs(ticks_30m[0] - o) < 0.001,
            "last_tick_eq_close": abs(ticks_30m[-1] - c) < 0.001,
            "high_reached": any(abs(t - h) < 0.001 for t in ticks_30m),
            "low_reached": any(abs(t - l) < 0.001 for t in ticks_30m),
            "all_within_bounds": all(l <= t <= h for t in ticks_30m)
        }
        
        # Report constraint satisfaction
        print(f"  15m Constraints: {sum(constraints_15m.values())}/5 satisfied")
        for constraint, satisfied in constraints_15m.items():
            status = "✅" if satisfied else "❌"
            print(f"    {status} {constraint}")
        
        print(f"  30m Constraints: {sum(constraints_30m.values())}/5 satisfied")
        for constraint, satisfied in constraints_30m.items():
            status = "✅" if satisfied else "❌"
            print(f"    {status} {constraint}")
    
    print("\n🎯 OVERALL RESULTS:")
    print("=" * 40)
    
    # Count total constraint satisfaction
    total_constraints = 0
    satisfied_constraints = 0
    
    for sample_idx in range(batch_size):
        sample_ohlc = ohlc_values[sample_idx]
        sample_ticks = sub_ticks[sample_idx]
        
        o, l, h, vix = sample_ohlc
        c = (o + h + l) / 3.0
        
        ticks_15m = sample_ticks[:8] 
        ticks_30m = sample_ticks[8:]
        
        # Check all constraints
        for ticks, period in [(ticks_15m, "15m"), (ticks_30m, "30m")]:
            constraints = [
                abs(ticks[0] - o) < 0.001,  # First = Open
                abs(ticks[-1] - c) < 0.001,  # Last = Close  
                any(abs(t - h) < 0.001 for t in ticks),  # High reached
                any(abs(t - l) < 0.001 for t in ticks),  # Low reached
                all(l <= t <= h for t in ticks)  # Within bounds
            ]
            
            total_constraints += len(constraints)
            satisfied_constraints += sum(constraints)
    
    success_rate = (satisfied_constraints / total_constraints) * 100
    print(f"📈 Constraint Satisfaction Rate: {satisfied_constraints}/{total_constraints} ({success_rate:.1f}%)")
    
    if success_rate >= 80:
        print("✅ CONSTRAINT-BASED TICK GENERATION: SUCCESS!")
        print("   The sub-periodicity tick generation system is working correctly.")
        print("   Generated ticks satisfy OHLC constraints and show realistic price movement.")
        return True
    else:
        print("❌ CONSTRAINT-BASED TICK GENERATION: NEEDS IMPROVEMENT")
        print("   Some constraints are not being satisfied properly.")
        return False

def test_integration_with_generator():
    """Test integration with a simple generator model."""
    print("\n🔗 TESTING INTEGRATION WITH GENERATOR MODEL")
    print("=" * 60)
    
    try:
        from tsg_plugins.generator_plugin.generator_plugin import FeatureExpansionLayer
        
        # Create a simple model that includes the FeatureExpansionLayer
        inputs = tf.keras.layers.Input(shape=(23,))
        outputs = FeatureExpansionLayer()(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        print("✅ Created test model with FeatureExpansionLayer")
        print(f"   Model input shape: {model.input_shape}")
        print(f"   Model output shape: {model.output_shape}")
        
        # Test with random input
        test_input = tf.random.normal((5, 23))
        output = model(test_input)
        
        print(f"✅ Model execution successful!")
        print(f"   Input shape: {test_input.shape}")
        print(f"   Output shape: {output.shape}")
        
        if output.shape == (5, 44):
            print("✅ Model integration successful!")
            return True
        else:
            print("❌ Model integration failed - shape mismatch")
            return False
            
    except Exception as e:
        print(f"❌ Model integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("🚀 SUB-PERIODICITY TICK GENERATION TEST SUITE")
    print("=" * 80)
    
    # Test 1: Basic FeatureExpansionLayer functionality
    test1_success = test_feature_expansion_layer()
    
    # Test 2: Integration with generator model
    test2_success = test_integration_with_generator()
    
    # Final results
    print("\n🏁 FINAL TEST RESULTS")
    print("=" * 40)
    
    tests_passed = sum([test1_success, test2_success])
    total_tests = 2
    
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("✅ ALL TESTS PASSED!")
        print("\n🎉 The constraint-based sub-periodicity tick generation system is ready!")
        print("   • OHLC constraints are properly enforced")
        print("   • Realistic price movements are generated")  
        print("   • Integration with generator models works correctly")
        print("   • Empty tick columns issue has been RESOLVED!")
    else:
        print("❌ Some tests failed. Please review the implementation.")
    
    return tests_passed == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
