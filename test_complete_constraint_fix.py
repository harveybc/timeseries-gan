#!/usr/bin/env python3
"""
Test Complete OHLC + Tick Constraint Fix
Tests both OHLC constraint fixing and constraint-based tick generation
"""

import numpy as np
import tensorflow as tf
from tsg_plugins.generator_plugin.generator_plugin import FeatureExpansionLayer


def test_complete_constraint_fix():
    """Test both OHLC and tick constraint fixes together"""
    
    print("🚀 TESTING COMPLETE CONSTRAINT FIX")
    print("=" * 60)
    
    # Create test data with likely OHLC violations
    batch_size = 3
    np.random.seed(42)
    
    # Generate 23 VAE features with potential constraint violations
    test_vae_output = np.random.normal(100, 10, (batch_size, 23)).astype(np.float32)
    
    print("📊 INPUT VAE OUTPUT (23 features):")
    print(f"Shape: {test_vae_output.shape}")
    for i in range(batch_size):
        sample = test_vae_output[i]
        print(f"Sample {i}: O={sample[0]:.3f}, L={sample[1]:.3f}, H={sample[2]:.3f}")
        # Check violations
        if sample[2] < max(sample[0], (sample[0]+sample[1]+sample[2])/3):  # H < max(O,C)
            print(f"  ❌ High constraint violation detected!")
        if sample[1] > min(sample[0], (sample[0]+sample[1]+sample[2])/3):  # L > min(O,C)
            print(f"  ❌ Low constraint violation detected!")
    
    # Test FeatureExpansionLayer with constraint fixing
    print(f"\n🔧 TESTING FEATURE EXPANSION WITH CONSTRAINT FIXING:")
    
    expansion_layer = FeatureExpansionLayer()
    vae_tensor = tf.constant(test_vae_output)
    
    expanded_output = expansion_layer(vae_tensor)
    expanded_result = expanded_output.numpy()
    
    print(f"✅ Expansion completed: {test_vae_output.shape} -> {expanded_result.shape}")
    
    # Extract OHLC from expanded output (positions 15-18)
    ohlc_start = 15  # Technical indicators (15) + OHLC (4)
    ohlc_data = expanded_result[:, ohlc_start:ohlc_start+4]
    
    # Extract tick data (positions 23-38) 
    tick_start = 23  # TI(15) + OHLC(4) + Spreads(4) = 23
    tick_data = expanded_result[:, tick_start:tick_start+16]
    
    print(f"\n📊 ANALYZED FIXED OHLC DATA:")
    print("=" * 40)
    
    all_ohlc_valid = True
    all_ticks_populated = True
    
    for i in range(batch_size):
        sample_ohlc = ohlc_data[i]
        sample_ticks = tick_data[i]
        
        o, h, l, c = sample_ohlc
        
        # Check OHLC constraints
        high_valid = h >= max(o, c)
        low_valid = l <= min(o, c)
        
        # Check tick population
        ticks_populated = not np.allclose(sample_ticks, 0.0, atol=1e-6)
        
        print(f"\nSample {i}:")
        print(f"  OHLC: O={o:.3f}, H={h:.3f}, L={l:.3f}, C={c:.3f}")
        print(f"  High constraint (H >= max(O,C)): {high_valid} ({h:.3f} >= {max(o,c):.3f})")
        print(f"  Low constraint (L <= min(O,C)): {low_valid} ({l:.3f} <= {min(o,c):.3f})")
        print(f"  Ticks populated: {ticks_populated}")
        
        if not high_valid or not low_valid:
            print(f"  ❌ OHLC constraint violation!")
            all_ohlc_valid = False
        else:
            print(f"  ✅ OHLC constraints satisfied!")
            
        if not ticks_populated:
            print(f"  ❌ Tick columns still empty!")
            all_ticks_populated = False
        else:
            print(f"  ✅ Tick columns populated!")
            
        # Show first few ticks for verification
        close_15m = sample_ticks[:8]
        close_30m = sample_ticks[8:]
        print(f"  15m ticks (first 4): [{close_15m[0]:.3f}, {close_15m[1]:.3f}, {close_15m[2]:.3f}, {close_15m[3]:.3f}...]")
        print(f"  30m ticks (first 4): [{close_30m[0]:.3f}, {close_30m[1]:.3f}, {close_30m[2]:.3f}, {close_30m[3]:.3f}...]")
    
    # Final results
    print(f"\n🏁 COMPREHENSIVE TEST RESULTS:")
    print("=" * 45)
    
    success = all_ohlc_valid and all_ticks_populated
    
    if success:
        print("✅ SUCCESS: ALL CONSTRAINTS SATISFIED!")
        print("✅ OHLC constraints: Fixed and validated")
        print("✅ Tick constraints: Generated and populated")
        print("✅ Empty tick columns issue: RESOLVED!")
        return True
    else:
        print("❌ FAILURE: Some constraints still violated!")
        if not all_ohlc_valid:
            print("❌ OHLC constraints still failing")
        if not all_ticks_populated:
            print("❌ Tick columns still empty")
        return False


if __name__ == "__main__":
    success = test_complete_constraint_fix()
    exit(0 if success else 1)
