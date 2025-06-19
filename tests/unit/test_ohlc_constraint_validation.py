#!/usr/bin/env python3
"""
OHLC Constraint Validation Test

Tests that generated sub-periodicity tick columns (15-min and 30-min) 
satisfy the fundamental OHLC constraints:

1. First tick = Open value (constraint)
2. Last tick = Close value (constraint) 
3. At least one tick must reach High value (constraint)
4. At least one tick must reach Low value (constraint)
5. Open should be closest to previous close value (realistic constraint)
6. All ticks should be between Low and High values (basic constraint)

For 30-min columns: We have 4 OHLC constraint points that the 8 ticks must satisfy
For 15-min columns: We have 2 OHLC constraint points per sequence
"""

import sys
import os
import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.append('/home/harveybc/Documents/GitHub/timeseries-gan')

from app.config import DEFAULT_VALUES
from tsg_plugins.generator_plugin.generator_plugin import GeneratorPlugin

def test_ohlc_constraint_satisfaction():
    """Test OHLC constraint satisfaction in generated tick sequences"""
    
    print("🔍 TESTING OHLC CONSTRAINT SATISFACTION IN TICK SEQUENCES")
    print("=" * 65)
    
    try:
        # Initialize generator plugin with generate mode
        config = DEFAULT_VALUES.copy()
        config["operation_mode"] = "generate"  # Force generate mode for 44-feature output
        
        plugin = GeneratorPlugin(config)
        print("✅ Generator plugin initialized in generate mode")
        
        # Get the built generator model
        generator_model = plugin.get_model()
        
        if generator_model is None:
            print("❌ Failed to build generator model")
            return False
            
        print(f"✅ Generator model built successfully")
        print(f"   Model output shape: {generator_model.output.shape}")
        
        # Generate test data
        batch_size = 10  # Generate multiple samples for thorough testing
        noise_input = np.random.normal(0, 1, (batch_size, 100)).astype(np.float32)
        conditional_input = np.random.normal(0, 1, (batch_size, 10)).astype(np.float32)
        context_input = np.random.normal(0, 1, (batch_size, 64)).astype(np.float32)
        
        print(f"\n🔄 Generating {batch_size} synthetic samples...")
        
        # Generate synthetic data
        synthetic_data = generator_model.predict([noise_input, conditional_input, context_input], verbose=0)
        
        print(f"✅ Generated data shape: {synthetic_data.shape}")
        
        if len(synthetic_data.shape) != 3 or synthetic_data.shape[2] != 44:
            print(f"❌ Unexpected output shape! Expected (batch, seq, 44), got {synthetic_data.shape}")
            return False
            
        # Extract features based on expected order from FeatureExpansionLayer
        # Order: [TI(15), OHLC(4), spreads(4), external(2), sub_periodicity(16), date(3)]
        
        seq_len = synthetic_data.shape[1]
        all_constraints_satisfied = True
        
        print(f"\n📊 ANALYZING OHLC CONSTRAINTS FOR {seq_len} TIMESTEPS")
        print("=" * 55)
        
        for sample_idx in range(min(batch_size, 3)):  # Test first 3 samples thoroughly
            print(f"\n🔬 SAMPLE {sample_idx + 1} ANALYSIS:")
            print("-" * 35)
            
            sample_data = synthetic_data[sample_idx]  # Shape: (seq_len, 44)
            
            # Extract OHLC data (positions 15-18)
            open_vals = sample_data[:, 15]   # OPEN
            high_vals = sample_data[:, 16]   # HIGH  
            low_vals = sample_data[:, 17]    # LOW
            close_vals = sample_data[:, 18]  # CLOSE
            
            # Extract tick data (positions 19-34: sub_periodicity 16 features)
            close_15m_ticks = sample_data[:, 19:27]  # 8 features for 15-min ticks
            close_30m_ticks = sample_data[:, 27:35]  # 8 features for 30-min ticks
            
            print(f"OHLC data shape: {open_vals.shape}")
            print(f"15-min ticks shape: {close_15m_ticks.shape}")
            print(f"30-min ticks shape: {close_30m_ticks.shape}")
            
            sample_violations = 0
            
            # Test each timestep
            for t in range(min(seq_len, 5)):  # Test first 5 timesteps
                print(f"\n  Timestep {t}:")
                
                # Current OHLC values
                open_val = open_vals[t]
                high_val = high_vals[t] 
                low_val = low_vals[t]
                close_val = close_vals[t]
                
                # Current tick sequences
                ticks_15m = close_15m_ticks[t]
                ticks_30m = close_30m_ticks[t]
                
                print(f"    OHLC: O={open_val:.4f}, H={high_val:.4f}, L={low_val:.4f}, C={close_val:.4f}")
                print(f"    15m: [{', '.join([f'{x:.3f}' for x in ticks_15m[:4]])}...]")
                print(f"    30m: [{', '.join([f'{x:.3f}' for x in ticks_30m[:4]])}...]")
                
                # Test constraints for 30-min ticks
                violations_30m = test_tick_constraints(
                    ticks_30m, open_val, high_val, low_val, close_val, 
                    f"Sample {sample_idx+1}, T{t}, 30m"
                )
                
                # Test constraints for 15-min ticks  
                violations_15m = test_tick_constraints(
                    ticks_15m, open_val, high_val, low_val, close_val,
                    f"Sample {sample_idx+1}, T{t}, 15m"
                )
                
                total_violations = violations_30m + violations_15m
                sample_violations += total_violations
                
                if total_violations > 0:
                    print(f"    ❌ {total_violations} constraint violations detected")
                    all_constraints_satisfied = False
                else:
                    print(f"    ✅ All constraints satisfied")
            
            print(f"\n  Sample {sample_idx+1} Summary: {sample_violations} total violations")
        
        # Final summary
        print(f"\n🏁 OHLC CONSTRAINT VALIDATION RESULTS:")
        print("=" * 45)
        
        if all_constraints_satisfied:
            print("✅ SUCCESS: All OHLC constraints are satisfied!")
            print("✅ Constraint-based tick generation is working correctly!")
            print("✅ Generated ticks respect OHLC boundaries and relationships!")
            return True
        else:
            print("❌ FAILURE: Some OHLC constraints are violated!")
            print("❌ Constraint-based tick generation needs refinement!")
            return False
            
    except Exception as e:
        print(f"❌ Error during OHLC constraint validation: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tick_constraints(ticks, open_val, high_val, low_val, close_val, context):
    """
    Test OHLC constraints for a single tick sequence.
    
    Args:
        ticks: Array of tick values
        open_val, high_val, low_val, close_val: OHLC constraint values
        context: String for logging context
        
    Returns:
        Number of constraint violations detected
    """
    violations = 0
    
    # Constraint 1: First tick should equal Open value (with tolerance)
    if not np.isclose(ticks[0], open_val, atol=0.1):
        print(f"      ❌ {context}: First tick ({ticks[0]:.4f}) != Open ({open_val:.4f})")
        violations += 1
    
    # Constraint 2: Last tick should equal Close value (with tolerance)
    if not np.isclose(ticks[-1], close_val, atol=0.1):
        print(f"      ❌ {context}: Last tick ({ticks[-1]:.4f}) != Close ({close_val:.4f})")
        violations += 1
    
    # Constraint 3: At least one tick should reach High value (with tolerance)
    max_tick = np.max(ticks)
    if not np.isclose(max_tick, high_val, atol=0.1):
        print(f"      ❌ {context}: No tick reaches High. Max={max_tick:.4f}, High={high_val:.4f}")
        violations += 1
    
    # Constraint 4: At least one tick should reach Low value (with tolerance)
    min_tick = np.min(ticks)
    if not np.isclose(min_tick, low_val, atol=0.1):
        print(f"      ❌ {context}: No tick reaches Low. Min={min_tick:.4f}, Low={low_val:.4f}")
        violations += 1
    
    # Constraint 5: All ticks should be within Low-High range
    out_of_bounds = (ticks < low_val - 0.1) | (ticks > high_val + 0.1)
    if np.any(out_of_bounds):
        out_count = np.sum(out_of_bounds)
        print(f"      ❌ {context}: {out_count} ticks outside Low-High range")
        violations += 1
    
    # Constraint 6: Basic ordering (High >= Low)
    if high_val < low_val:
        print(f"      ❌ {context}: Invalid OHLC - High ({high_val:.4f}) < Low ({low_val:.4f})")
        violations += 1
        
    return violations

def test_previous_close_continuity():
    """Test that Open values are closest to previous Close values"""
    
    print("\n🔗 TESTING PREVIOUS CLOSE CONTINUITY")
    print("=" * 40)
    
    try:
        # This would require sequence data to test continuity between timesteps
        # For now, we'll implement a basic test
        
        config = DEFAULT_VALUES.copy()
        config["operation_mode"] = "generate"
        
        plugin = GeneratorPlugin(config)
        generator_model = plugin.get_model()
        
        if generator_model is None:
            print("❌ Failed to build generator model")
            return False
        
        # Generate a sequence
        batch_size = 1
        noise_input = np.random.normal(0, 1, (batch_size, 100)).astype(np.float32)
        conditional_input = np.random.normal(0, 1, (batch_size, 10)).astype(np.float32)
        context_input = np.random.normal(0, 1, (batch_size, 64)).astype(np.float32)
        
        synthetic_data = generator_model.predict([noise_input, conditional_input, context_input], verbose=0)
        
        # Extract OHLC sequence
        sample_data = synthetic_data[0]  # First sample
        open_vals = sample_data[:, 15]   # OPEN  
        close_vals = sample_data[:, 18]  # CLOSE
        
        print(f"Testing continuity across {len(open_vals)} timesteps...")
        
        continuity_violations = 0
        
        for t in range(1, min(len(open_vals), 10)):  # Test first 10 timesteps
            prev_close = close_vals[t-1]
            current_open = open_vals[t]
            
            # The current open should be reasonably close to previous close
            # (allowing for some market movement)
            price_change = abs(current_open - prev_close) / prev_close
            
            if price_change > 0.1:  # More than 10% price jump is unrealistic
                print(f"  Timestep {t}: Large gap - Close({prev_close:.4f}) -> Open({current_open:.4f}) = {price_change:.2%}")
                continuity_violations += 1
        
        if continuity_violations == 0:
            print("✅ Price continuity looks reasonable")
            return True
        else:
            print(f"⚠️  {continuity_violations} large price gaps detected")
            return False
            
    except Exception as e:
        print(f"❌ Error testing continuity: {e}")
        return False

if __name__ == "__main__":
    print("🚀 COMPREHENSIVE OHLC CONSTRAINT VALIDATION")
    print("=" * 50)
    
    # Test 1: Basic OHLC constraint satisfaction
    constraints_ok = test_ohlc_constraint_satisfaction()
    
    # Test 2: Previous close continuity  
    continuity_ok = test_previous_close_continuity()
    
    # Overall results
    print(f"\n🏆 FINAL RESULTS:")
    print("=" * 20)
    print(f"OHLC Constraints: {'✅ PASS' if constraints_ok else '❌ FAIL'}")
    print(f"Price Continuity: {'✅ PASS' if continuity_ok else '❌ FAIL'}")
    
    overall_success = constraints_ok and continuity_ok
    
    if overall_success:
        print(f"\n🎉 ALL TESTS PASSED!")
        print("✅ Constraint-based tick generation is working correctly!")
        print("✅ Generated data satisfies OHLC mathematical constraints!")
    else:
        print(f"\n💥 SOME TESTS FAILED!")
        print("❌ Constraint-based tick generation needs improvement!")
    
    sys.exit(0 if overall_success else 1)
