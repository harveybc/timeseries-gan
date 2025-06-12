#!/usr/bin/env python3
"""
Test the FeatureExpansionLayer constraint-based tick generation
"""

import numpy as np
import tensorflow as tf
from tsg_plugins.generator_plugin.generator_plugin import FeatureExpansionLayer

def test_feature_expansion_layer():
    """Test the FeatureExpansionLayer directly"""
    
    print("🧪 TESTING FEATURE EXPANSION LAYER")
    print("=" * 50)
    
    # Create the layer
    expansion_layer = FeatureExpansionLayer()
    
    # Create test input (batch_size=3, features=23)
    batch_size = 3
    test_input = tf.random.normal((batch_size, 23))
    
    print(f"✅ Input shape: {test_input.shape}")
    print(f"✅ Input sample values: {test_input[0, :5].numpy()}")
    
    # Run the layer
    try:
        expanded_output = expansion_layer(test_input)
        print(f"✅ Output shape: {expanded_output.shape}")
        
        # Extract different sections
        technical_indicators = expanded_output[:, :15]  # First 15 features
        ohlc = expanded_output[:, 15:19]               # Next 4 features  
        spreads = expanded_output[:, 19:23]            # Next 4 features
        external = expanded_output[:, 23:25]           # Next 2 features
        ticks = expanded_output[:, 25:41]              # Next 16 features (tick columns)
        date_features = expanded_output[:, 41:44]      # Last 3 features
        
        print(f"✅ Technical indicators shape: {technical_indicators.shape}")
        print(f"✅ OHLC shape: {ohlc.shape}")
        print(f"✅ Spreads shape: {spreads.shape}")
        print(f"✅ External shape: {external.shape}")
        print(f"✅ Ticks shape: {ticks.shape}")
        print(f"✅ Date features shape: {date_features.shape}")
        
        # Analyze tick generation quality
        print("\n📊 TICK GENERATION ANALYSIS:")
        print("=" * 35)
        
        all_constraints_satisfied = True
        
        for i in range(batch_size):
            sample_ohlc = ohlc[i].numpy()
            sample_ticks = ticks[i].numpy()
            
            # Split into 15m and 30m ticks
            ticks_15m = sample_ticks[:8]
            ticks_30m = sample_ticks[8:]
            
            open_val, high_val, low_val, close_val = sample_ohlc
            
            print(f"\nSample {i+1}:")
            print(f"  OHLC: O={open_val:.4f}, H={high_val:.4f}, L={low_val:.4f}, C={close_val:.4f}")
            print(f"  15m ticks: {[f'{x:.4f}' for x in ticks_15m]}")
            print(f"  30m ticks: {[f'{x:.4f}' for x in ticks_30m]}")
            
            # Check constraints for 15m ticks
            constraints_15m = check_tick_constraints(ticks_15m, open_val, high_val, low_val, close_val)
            constraints_30m = check_tick_constraints(ticks_30m, open_val, high_val, low_val, close_val)
            
            if constraints_15m and constraints_30m:
                print(f"  ✅ All OHLC constraints satisfied!")
            else:
                print(f"  ❌ OHLC constraints violated! 15m_ok={constraints_15m}, 30m_ok={constraints_30m}")
                all_constraints_satisfied = False
        
        # Summary
        print(f"\n🏁 FEATURE EXPANSION TEST RESULTS:")
        print("=" * 40)
        
        if all_constraints_satisfied:
            print("✅ SUCCESS: All OHLC constraints satisfied!")
            print("✅ Constraint-based tick generation is working!")
            return True
        else:
            print("❌ FAILURE: Some OHLC constraints violated!")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def check_tick_constraints(ticks, open_val, high_val, low_val, close_val):
    """Check if tick sequence satisfies OHLC constraints"""
    
    # Check if not all zeros (populated)
    if np.allclose(ticks, 0.0, atol=1e-6):
        return False
    
    # Check bounds (all ticks within [low, high])
    if np.any(ticks < low_val) or np.any(ticks > high_val):
        return False
    
    # Check first tick = open (within tolerance)
    if not np.isclose(ticks[0], open_val, atol=1e-3):
        return False
    
    # Check last tick = close (within tolerance)  
    if not np.isclose(ticks[-1], close_val, atol=1e-3):
        return False
    
    # Check high value reached
    if not np.any(np.isclose(ticks, high_val, atol=1e-3)):
        return False
    
    # Check low value reached
    if not np.any(np.isclose(ticks, low_val, atol=1e-3)):
        return False
    
    return True

if __name__ == "__main__":
    success = test_feature_expansion_layer()
    exit(0 if success else 1)
