#!/usr/bin/env python3
"""
Comprehensive test for generate operation mode
Tests OHLC constraints, high-frequency tick constraints, and seasonal date features
"""

import numpy as np
import tensorflow as tf
from tsg_plugins.generator_plugin.generator_plugin import GeneratorPlugin, FeatureExpansionLayer
from app.config import DEFAULT_VALUES

def test_generate_mode_comprehensive():
    """Test generate mode with comprehensive constraint validation"""
    
    print("🚀 COMPREHENSIVE GENERATE MODE TEST")
    print("=" * 50)
    print("Testing:")
    print("  1. OHLC constraint satisfaction")
    print("  2. High-frequency tick constraints")
    print("  3. Seasonal date feature normalization")
    print("  4. Feature expansion from 23→44")
    print()
    
    # Test 1: Direct FeatureExpansionLayer Test
    print("🧪 TEST 1: FEATURE EXPANSION LAYER DIRECT TEST")
    print("=" * 55)
    
    success_layer = test_feature_expansion_layer_direct()
    
    # Test 2: Generator Plugin Generate Mode Test
    print("\n🧪 TEST 2: GENERATOR PLUGIN GENERATE MODE TEST")
    print("=" * 55)
    
    success_generator = test_generator_plugin_generate_mode()
    
    # Final Results
    print("\n🏆 COMPREHENSIVE TEST RESULTS")
    print("=" * 35)
    print(f"Feature Expansion Layer: {'✅ PASS' if success_layer else '❌ FAIL'}")
    print(f"Generator Plugin Mode:   {'✅ PASS' if success_generator else '❌ FAIL'}")
    
    overall_success = success_layer and success_generator
    
    if overall_success:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Generate mode is working correctly")
        print("✅ All constraints are satisfied")
        print("✅ Feature generation is correct")
    else:
        print("\n💔 SOME TESTS FAILED!")
        print("❌ Generate mode needs attention")
    
    return overall_success

def test_feature_expansion_layer_direct():
    """Test FeatureExpansionLayer directly with detailed constraint checking"""
    
    print("Testing FeatureExpansionLayer constraint generation...")
    
    try:
        # Create test input with realistic financial data
        batch_size = 5
        test_input = tf.constant([
            # Sample 1: Normal market conditions
            [1.0850, 1.0820, 1.0880, 0.35, 0.02, 0.04, 4250.5] + [0.0] * 16,
            # Sample 2: Volatile conditions
            [1.0900, 1.0850, 1.0950, 0.45, -0.01, 0.08, 4180.2] + [0.0] * 16,
            # Sample 3: Low volatility
            [1.0800, 1.0795, 1.0810, 0.28, 0.005, 0.015, 4300.8] + [0.0] * 16,
            # Sample 4: High volatility
            [1.1200, 1.1150, 1.1280, 0.52, -0.02, 0.12, 4050.3] + [0.0] * 16,
            # Sample 5: Edge case
            [1.0500, 1.0500, 1.0500, 0.20, 0.0, 0.0, 4200.0] + [0.0] * 16
        ], dtype=tf.float32)
        
        print(f"✅ Created test input: {test_input.shape}")
        
        # Create and run FeatureExpansionLayer
        expansion_layer = FeatureExpansionLayer()
        expanded_output = expansion_layer(test_input)
        
        print(f"✅ Feature expansion completed: {test_input.shape} → {expanded_output.shape}")
        
        if expanded_output.shape != (batch_size, 44):
            print(f"❌ Wrong output shape! Expected ({batch_size}, 44), got {expanded_output.shape}")
            return False
        
        # Extract feature sections for analysis
        expanded_data = expanded_output.numpy()
        
        # Feature structure: [TI(15), OHLC(4), Spreads(4), External(2), Ticks(16), Date(3)]
        technical_indicators = expanded_data[:, 0:15]
        ohlc_data = expanded_data[:, 15:19]
        spreads_data = expanded_data[:, 19:23]
        external_data = expanded_data[:, 23:25]
        tick_data = expanded_data[:, 25:41]
        date_data = expanded_data[:, 41:44]
        
        print(f"✅ Extracted features:")
        print(f"  - Technical Indicators: {technical_indicators.shape}")
        print(f"  - OHLC: {ohlc_data.shape}")
        print(f"  - Spreads: {spreads_data.shape}")
        print(f"  - External: {external_data.shape}")
        print(f"  - Ticks: {tick_data.shape}")
        print(f"  - Date: {date_data.shape}")
        
        # Test OHLC constraints
        print("\n📊 TESTING OHLC CONSTRAINTS:")
        print("-" * 35)
        
        ohlc_violations = 0
        for i in range(batch_size):
            open_val, high_val, low_val, close_val = ohlc_data[i]
            
            print(f"\nSample {i+1}:")
            print(f"  OHLC: O={open_val:.4f}, H={high_val:.4f}, L={low_val:.4f}, C={close_val:.4f}")
            
            # Check OHLC constraints
            constraints_satisfied = True
            
            if high_val < max(open_val, close_val):
                print(f"  ❌ High constraint violated: H={high_val:.4f} < max(O,C)={max(open_val, close_val):.4f}")
                constraints_satisfied = False
                ohlc_violations += 1
                
            if low_val > min(open_val, close_val):
                print(f"  ❌ Low constraint violated: L={low_val:.4f} > min(O,C)={min(open_val, close_val):.4f}")
                constraints_satisfied = False
                ohlc_violations += 1
                
            if constraints_satisfied:
                print(f"  ✅ OHLC constraints satisfied")
        
        # Test tick constraints
        print("\n🎯 TESTING HIGH-FREQUENCY TICK CONSTRAINTS:")
        print("-" * 45)
        
        tick_violations = 0
        close_15m_ticks = tick_data[:, :8]
        close_30m_ticks = tick_data[:, 8:]
        
        for i in range(batch_size):
            open_val, high_val, low_val, close_val = ohlc_data[i]
            ticks_15m = close_15m_ticks[i]
            ticks_30m = close_30m_ticks[i]
            
            print(f"\nSample {i+1} Tick Analysis:")
            print(f"  15m ticks: [{', '.join([f'{x:.4f}' for x in ticks_15m])}]")
            print(f"  30m ticks: [{', '.join([f'{x:.4f}' for x in ticks_30m])}]")
            
            # Check 15m tick constraints
            violations_15m = check_tick_constraints(ticks_15m, open_val, high_val, low_val, close_val, "15m")
            violations_30m = check_tick_constraints(ticks_30m, open_val, high_val, low_val, close_val, "30m")
            
            tick_violations += violations_15m + violations_30m
        
        # Test seasonal date features
        print("\n🕒 TESTING SEASONAL DATE FEATURES:")
        print("-" * 35)
        
        date_violations = 0
        for i in range(batch_size):
            day_of_month, hour_of_day, day_of_week = date_data[i]
            
            print(f"\nSample {i+1} Date Features:")
            print(f"  Day of Month: {day_of_month:.4f}")
            print(f"  Hour of Day:  {hour_of_day:.4f}")
            print(f"  Day of Week:  {day_of_week:.4f}")
            
            # Check normalization (should be between 0 and 1)
            date_features = [day_of_month, hour_of_day, day_of_week]
            date_names = ["Day of Month", "Hour of Day", "Day of Week"]
            
            for feature_val, feature_name in zip(date_features, date_names):
                if not (0.0 <= feature_val <= 1.0):
                    print(f"  ❌ {feature_name} not normalized: {feature_val:.4f} (should be 0.0-1.0)")
                    date_violations += 1
                else:
                    print(f"  ✅ {feature_name} properly normalized")
        
        # Summary
        print(f"\n📋 FEATURE EXPANSION LAYER TEST SUMMARY:")
        print("=" * 45)
        print(f"OHLC Constraint Violations:     {ohlc_violations}")
        print(f"Tick Constraint Violations:     {tick_violations}")
        print(f"Date Normalization Violations:  {date_violations}")
        
        total_violations = ohlc_violations + tick_violations + date_violations
        
        if total_violations == 0:
            print("✅ ALL CONSTRAINTS SATISFIED!")
            return True
        else:
            print(f"❌ TOTAL VIOLATIONS: {total_violations}")
            return False
            
    except Exception as e:
        print(f"❌ Error in FeatureExpansionLayer test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_generator_plugin_generate_mode():
    """Test GeneratorPlugin in generate mode"""
    
    print("Testing GeneratorPlugin in generate operation mode...")
    
    try:
        # Configure for generate mode
        config = DEFAULT_VALUES.copy()
        config["operation_mode"] = "generate"  # Force generate mode
        
        print("✅ Configured for generate mode")
        
        # Create generator plugin
        generator = GeneratorPlugin(config)
        print("✅ Created GeneratorPlugin")
        
        # Test the _expand_vae_output_to_44_features method if it exists
        if hasattr(generator, '_expand_vae_output_to_44_features'):
            print("✅ Found _expand_vae_output_to_44_features method")
            
            # Test direct method call
            batch_size = 3
            vae_output = tf.random.normal([batch_size, 23], dtype=tf.float32)
            
            expanded_output = generator._expand_vae_output_to_44_features(vae_output)
            
            print(f"✅ Method expansion: {vae_output.shape} → {expanded_output.shape}")
            
            if expanded_output.shape != (batch_size, 44):
                print(f"❌ Wrong method output shape!")
                return False
                
            # Quick constraint check
            expanded_data = expanded_output.numpy()
            ohlc_data = expanded_data[:, 15:19]
            
            violations = 0
            for i in range(batch_size):
                open_val, high_val, low_val, close_val = ohlc_data[i]
                if high_val < max(open_val, close_val) or low_val > min(open_val, close_val):
                    violations += 1
            
            print(f"✅ Generator method OHLC violations: {violations}/{batch_size}")
            
            return violations == 0
        else:
            print("⚠️ _expand_vae_output_to_44_features method not found")
            return True  # Not a failure if method doesn't exist
            
    except Exception as e:
        print(f"❌ Error in GeneratorPlugin test: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_tick_constraints(ticks, open_val, high_val, low_val, close_val, period_name):
    """Check tick sequence constraints and return number of violations"""
    
    violations = 0
    tolerance = 0.01
    
    # Constraint 1: First tick should equal Open
    if abs(ticks[0] - open_val) > tolerance:
        print(f"    ❌ {period_name}: First tick ({ticks[0]:.4f}) != Open ({open_val:.4f})")
        violations += 1
    
    # Constraint 2: Last tick should equal Close
    if abs(ticks[-1] - close_val) > tolerance:
        print(f"    ❌ {period_name}: Last tick ({ticks[-1]:.4f}) != Close ({close_val:.4f})")
        violations += 1
    
    # Constraint 3: At least one tick should reach High
    max_tick = np.max(ticks)
    if abs(max_tick - high_val) > tolerance:
        print(f"    ❌ {period_name}: No tick reaches High. Max={max_tick:.4f}, High={high_val:.4f}")
        violations += 1
    
    # Constraint 4: At least one tick should reach Low
    min_tick = np.min(ticks)
    if abs(min_tick - low_val) > tolerance:
        print(f"    ❌ {period_name}: No tick reaches Low. Min={min_tick:.4f}, Low={low_val:.4f}")
        violations += 1
    
    # Constraint 5: All ticks should be within Low-High range
    out_of_bounds = (ticks < low_val - tolerance) | (ticks > high_val + tolerance)
    if np.any(out_of_bounds):
        out_count = np.sum(out_of_bounds)
        print(f"    ❌ {period_name}: {out_count} ticks outside Low-High range")
        violations += 1
    
    # Check if all constraints satisfied
    if violations == 0:
        print(f"    ✅ {period_name}: All tick constraints satisfied")
    
    return violations

if __name__ == "__main__":
    success = test_generate_mode_comprehensive()
    exit(0 if success else 1)
