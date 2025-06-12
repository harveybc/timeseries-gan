#!/usr/bin/env python3
"""
Simple integration test for constraint-based sub-periodicity tick generation
Tests the FeatureExpansionLayer directly without full plugin setup
"""

import numpy as np
import tensorflow as tf
import sys
import os

# Add the project directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_feature_expansion_layer_directly():
    """Test constraint-based tick generation using FeatureExpansionLayer directly"""
    
    print("🚀 TESTING FEATURE EXPANSION LAYER WITH CONSTRAINT-BASED TICKS")
    print("=" * 70)
    
    # Import the layer
    try:
        from tsg_plugins.generator_plugin.generator_plugin import FeatureExpansionLayer
        print("✅ Successfully imported FeatureExpansionLayer")
    except Exception as e:
        print(f"❌ Failed to import FeatureExpansionLayer: {e}")
        return False
    
    # Create test input (23 VAE features)
    batch_size = 5
    test_input = tf.random.normal([batch_size, 23], dtype=tf.float32)
    
    print(f"✅ Created test input: {test_input.shape}")
    
    # Create the feature expansion layer
    try:
        expansion_layer = FeatureExpansionLayer(name="test_expansion")
        print("✅ Created FeatureExpansionLayer instance")
    except Exception as e:
        print(f"❌ Failed to create FeatureExpansionLayer: {e}")
        return False
    
    # Run the expansion
    try:
        print("🔄 Running feature expansion...")
        expanded_output = expansion_layer(test_input)
        print(f"✅ Feature expansion completed: {test_input.shape} -> {expanded_output.shape}")
    except Exception as e:
        print(f"❌ Feature expansion failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Check output shape
    expected_shape = (batch_size, 44)
    actual_shape = tuple(expanded_output.shape)
    
    if actual_shape == expected_shape:
        print(f"✅ Output shape is correct: {actual_shape}")
    else:
        print(f"❌ Output shape mismatch. Expected: {expected_shape}, Got: {actual_shape}")
        return False
    
    # Convert to numpy for detailed analysis
    expanded_data = expanded_output.numpy()
    
    print("\n📊 ANALYZING GENERATED TICK DATA:")
    print("=" * 45)
    
    # Extract OHLC and tick columns (positions based on 44-feature structure)
    # Structure: TI(15) + OHLC(4) + Spreads(4) + External(2) + Ticks(16) + Date(3)
    ohlc_start = 15  # After 15 technical indicators
    tick_start = 15 + 4 + 4 + 2  # After TI + OHLC + Spreads + External = position 25
    
    ohlc_data = expanded_data[:, ohlc_start:ohlc_start+4]  # OHLC (4 features)
    tick_data = expanded_data[:, tick_start:tick_start+16]  # 16 tick columns
    
    print(f"✅ OHLC data shape: {ohlc_data.shape}")
    print(f"✅ Tick data shape: {tick_data.shape}")
    
    # Split ticks into 15m and 30m
    close_15m_ticks = tick_data[:, :8]
    close_30m_ticks = tick_data[:, 8:]
    
    all_populated = True
    all_constraints_satisfied = True
    
    for i in range(batch_size):
        sample_ohlc = ohlc_data[i]
        sample_15m = close_15m_ticks[i]
        sample_30m = close_30m_ticks[i]
        
        open_price, high_price, low_price, close_price = sample_ohlc
        
        # Check if columns are populated (not all zeros)
        empty_15m = np.allclose(sample_15m, 0.0, atol=1e-6)
        empty_30m = np.allclose(sample_30m, 0.0, atol=1e-6)
        
        # Check OHLC constraints for 15m ticks
        first_15m = sample_15m[0]
        last_15m = sample_15m[-1]
        max_15m = np.max(sample_15m)
        min_15m = np.min(sample_15m)
        
        # Check OHLC constraints for 30m ticks
        first_30m = sample_30m[0]
        last_30m = sample_30m[-1]
        max_30m = np.max(sample_30m)
        min_30m = np.min(sample_30m)
        
        print(f"\nSample {i+1}:")
        print(f"  OHLC: O={open_price:.4f}, H={high_price:.4f}, L={low_price:.4f}, C={close_price:.4f}")
        print(f"  15m ticks: [{', '.join([f'{x:.4f}' for x in sample_15m])}]")
        print(f"  30m ticks: [{', '.join([f'{x:.4f}' for x in sample_30m])}]")
        
        if empty_15m or empty_30m:
            print(f"  ❌ EMPTY COLUMNS! 15m_empty={empty_15m}, 30m_empty={empty_30m}")
            all_populated = False
        else:
            print(f"  ✅ All tick columns populated!")
        
        # Check constraints (allowing small tolerance for numerical precision)
        tolerance = 0.01
        
        constraints_ok = True
        
        # Check 15m constraints
        if abs(first_15m - open_price) > tolerance:
            print(f"  ❌ 15m first tick constraint violated: {first_15m:.4f} != {open_price:.4f}")
            constraints_ok = False
        
        if abs(last_15m - close_price) > tolerance:
            print(f"  ❌ 15m last tick constraint violated: {last_15m:.4f} != {close_price:.4f}")
            constraints_ok = False
        
        if max_15m < high_price - tolerance:
            print(f"  ❌ 15m high constraint violated: max={max_15m:.4f} < high={high_price:.4f}")
            constraints_ok = False
        
        if min_15m > low_price + tolerance:
            print(f"  ❌ 15m low constraint violated: min={min_15m:.4f} > low={low_price:.4f}")
            constraints_ok = False
        
        # Check 30m constraints
        if abs(first_30m - open_price) > tolerance:
            print(f"  ❌ 30m first tick constraint violated: {first_30m:.4f} != {open_price:.4f}")
            constraints_ok = False
        
        if abs(last_30m - close_price) > tolerance:
            print(f"  ❌ 30m last tick constraint violated: {last_30m:.4f} != {close_price:.4f}")
            constraints_ok = False
        
        if max_30m < high_price - tolerance:
            print(f"  ❌ 30m high constraint violated: max={max_30m:.4f} < high={high_price:.4f}")
            constraints_ok = False
        
        if min_30m > low_price + tolerance:
            print(f"  ❌ 30m low constraint violated: min={min_30m:.4f} > low={low_price:.4f}")
            constraints_ok = False
        
        if constraints_ok:
            print(f"  ✅ All OHLC constraints satisfied!")
        else:
            all_constraints_satisfied = False
    
    # Summary
    print(f"\n🏁 FEATURE EXPANSION LAYER TEST RESULTS:")
    print("=" * 50)
    
    success = True
    
    if all_populated:
        print("✅ SUCCESS: All tick columns are populated!")
    else:
        print("❌ FAILURE: Some tick columns are still empty!")
        success = False
    
    if all_constraints_satisfied:
        print("✅ SUCCESS: All OHLC constraints are satisfied!")
    else:
        print("❌ FAILURE: Some OHLC constraints are violated!")
        success = False
    
    if success:
        print("🎉 CONSTRAINT-BASED TICK GENERATION IS WORKING CORRECTLY!")
        print("🎉 Empty tick columns issue has been RESOLVED!")
    else:
        print("💔 Issues detected in constraint-based tick generation!")
    
    return success

if __name__ == "__main__":
    success = test_feature_expansion_layer_directly()
    exit(0 if success else 1)
