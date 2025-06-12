#!/usr/bin/env python3
"""
Comprehensive test for generate operation mode.
Tests OHLC constraints, high-frequency tick constraints, and seasonal date features.
"""

import numpy as np
import tensorflow as tf
from tsg_plugins.generator_plugin.generator_plugin import GeneratorPlugin
from app.config import DEFAULT_VALUES

def test_generate_operation_mode_comprehensive():
    """Test generate operation mode with comprehensive constraint verification"""
    
    print("🚀 COMPREHENSIVE GENERATE OPERATION MODE TEST")
    print("=" * 55)
    
    try:
        # Create config for generate mode
        config = DEFAULT_VALUES.copy()
        config["operation_mode"] = "generate"  # Force generate mode for 44-feature output
        config["print_model_summary"] = False  # Reduce output verbosity
        
        # Initialize generator plugin
        plugin = GeneratorPlugin(config)
        print("✅ Generator plugin initialized for generate mode")
        
        # Get the built generator model
        generator_model = plugin.get_model()
        
        if generator_model is None:
            print("❌ Failed to build generator model")
            return False
            
        print(f"✅ Generator model built successfully")
        print(f"   Model output shape: {generator_model.output.shape}")
        
        # Verify model outputs 44 features
        expected_output_features = 44
        actual_output_features = generator_model.output.shape[-1]
        
        if actual_output_features != expected_output_features:
            print(f"❌ Wrong output features! Expected {expected_output_features}, got {actual_output_features}")
            return False
        
        print(f"✅ Correct output feature count: {actual_output_features}")
        
        # Generate test data
        batch_size = 10
        seq_len = generator_model.output.shape[1] if len(generator_model.output.shape) == 3 else 1
        
        # Prepare inputs based on model requirements
        if len(generator_model.input.shape) == 2:
            # Single input model
            noise_input = np.random.normal(0, 1, (batch_size, generator_model.input.shape[1])).astype(np.float32)
            print(f"✅ Generated noise input: {noise_input.shape}")
            
            # Generate synthetic data
            synthetic_data = generator_model.predict(noise_input, verbose=0)
            
        else:
            # Multi-input model (composite generator)
            noise_input = np.random.normal(0, 1, (batch_size, 100)).astype(np.float32)
            conditional_input = np.random.normal(0, 1, (batch_size, 10)).astype(np.float32)
            context_input = np.random.normal(0, 1, (batch_size, 64)).astype(np.float32)
            
            print(f"✅ Generated inputs: noise{noise_input.shape}, conditional{conditional_input.shape}, context{context_input.shape}")
            
            # Generate synthetic data
            synthetic_data = generator_model.predict([noise_input, conditional_input, context_input], verbose=0)
        
        print(f"✅ Generated data shape: {synthetic_data.shape}")
        
        # Handle both sequence and single timestep outputs
        if len(synthetic_data.shape) == 3:
            # Sequence output (batch_size, seq_len, features)
            print(f"✅ Sequence output detected: {seq_len} timesteps")
            test_data = synthetic_data  # Use all timesteps
        else:
            # Single timestep output (batch_size, features)
            print("✅ Single timestep output detected")
            test_data = synthetic_data.reshape(batch_size, 1, -1)  # Reshape to (batch_size, 1, features)
            seq_len = 1
        
        # Test comprehensive constraints
        return test_comprehensive_constraints(test_data, batch_size, seq_len)
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_comprehensive_constraints(synthetic_data, batch_size, seq_len):
    """Test all constraints comprehensively"""
    
    print(f"\n🔍 COMPREHENSIVE CONSTRAINT VERIFICATION")
    print("=" * 50)
    
    # Feature structure: [TI(15), OHLC(4), Spreads(4), External(2), Ticks(16), Date(3)]
    all_constraints_satisfied = True
    
    for sample_idx in range(min(batch_size, 5)):  # Test first 5 samples
        print(f"\n📊 SAMPLE {sample_idx + 1} ANALYSIS:")
        print("-" * 35)
        
        sample_data = synthetic_data[sample_idx]  # Shape: (seq_len, 44)
        
        # Test each timestep
        for t in range(min(seq_len, 3)):  # Test first 3 timesteps
            timestep_data = sample_data[t] if seq_len > 1 else sample_data[0]
            
            print(f"\n  Timestep {t+1}:")
            
            # Extract feature sections
            ti_features = timestep_data[0:15]      # Technical indicators
            ohlc_features = timestep_data[15:19]   # OHLC
            spread_features = timestep_data[19:23] # Spreads
            external_features = timestep_data[23:25] # External data
            tick_features = timestep_data[25:41]   # 16 tick columns
            date_features = timestep_data[41:44]   # Date features
            
            # Test 1: OHLC Constraint Satisfaction
            ohlc_ok = test_ohlc_constraints(ohlc_features)
            if not ohlc_ok:
                all_constraints_satisfied = False
            
            # Test 2: High-Frequency Tick Constraints
            ticks_ok = test_tick_constraints(tick_features, ohlc_features)
            if not ticks_ok:
                all_constraints_satisfied = False
            
            # Test 3: Seasonal Date Features
            date_ok = test_seasonal_date_features(date_features)
            if not date_ok:
                all_constraints_satisfied = False
            
            # Test 4: Feature Population (no empty columns)
            population_ok = test_feature_population(timestep_data)
            if not population_ok:
                all_constraints_satisfied = False
    
    # Final results
    print(f"\n🏁 COMPREHENSIVE TEST RESULTS:")
    print("=" * 40)
    
    if all_constraints_satisfied:
        print("✅ SUCCESS: All constraints satisfied!")
        print("✅ OHLC constraints: PASSED")
        print("✅ Tick constraints: PASSED") 
        print("✅ Seasonal date features: PASSED")
        print("✅ Feature population: PASSED")
        print("✅ Generate operation mode is working correctly!")
        return True
    else:
        print("❌ FAILURE: Some constraints violated!")
        print("❌ Generate operation mode needs refinement!")
        return False

def test_ohlc_constraints(ohlc_features):
    """Test OHLC constraint satisfaction"""
    
    open_val, high_val, low_val, close_val = ohlc_features
    
    print(f"    OHLC: O={open_val:.4f}, H={high_val:.4f}, L={low_val:.4f}, C={close_val:.4f}")
    
    # Constraint 1: High >= max(Open, Close)
    max_oc = max(open_val, close_val)
    if high_val < max_oc - 1e-6:
        print(f"    ❌ OHLC violation: High ({high_val:.4f}) < max(O,C) ({max_oc:.4f})")
        return False
    
    # Constraint 2: Low <= min(Open, Close)
    min_oc = min(open_val, close_val)
    if low_val > min_oc + 1e-6:
        print(f"    ❌ OHLC violation: Low ({low_val:.4f}) > min(O,C) ({min_oc:.4f})")
        return False
    
    # Constraint 3: High >= Low (basic sanity)
    if high_val < low_val - 1e-6:
        print(f"    ❌ OHLC violation: High ({high_val:.4f}) < Low ({low_val:.4f})")
        return False
    
    print(f"    ✅ OHLC constraints satisfied")
    return True

def test_tick_constraints(tick_features, ohlc_features):
    """Test high-frequency tick constraint satisfaction"""
    
    open_val, high_val, low_val, close_val = ohlc_features
    
    # Split into 15m and 30m ticks
    ticks_15m = tick_features[0:8]   # First 8 ticks
    ticks_30m = tick_features[8:16]  # Last 8 ticks
    
    print(f"    15m ticks: [{', '.join([f'{x:.3f}' for x in ticks_15m[:4]])}...]")
    print(f"    30m ticks: [{', '.join([f'{x:.3f}' for x in ticks_30m[:4]])}...]")
    
    # Test both tick sequences
    constraints_ok = True
    
    for tick_seq, name in [(ticks_15m, "15m"), (ticks_30m, "30m")]:
        # Check if populated (not all zeros)
        if np.allclose(tick_seq, 0.0, atol=1e-6):
            print(f"    ❌ {name} ticks are empty (all zeros)")
            constraints_ok = False
            continue
        
        # Constraint 1: First tick = Open (with tolerance)
        if not np.isclose(tick_seq[0], open_val, atol=0.1):
            print(f"    ❌ {name}: First tick ({tick_seq[0]:.4f}) != Open ({open_val:.4f})")
            constraints_ok = False
        
        # Constraint 2: Last tick = Close (with tolerance)
        if not np.isclose(tick_seq[-1], close_val, atol=0.1):
            print(f"    ❌ {name}: Last tick ({tick_seq[-1]:.4f}) != Close ({close_val:.4f})")
            constraints_ok = False
        
        # Constraint 3: At least one tick reaches High (with tolerance)
        if not np.any(np.isclose(tick_seq, high_val, atol=0.1)):
            print(f"    ❌ {name}: No tick reaches High ({high_val:.4f})")
            constraints_ok = False
        
        # Constraint 4: At least one tick reaches Low (with tolerance)
        if not np.any(np.isclose(tick_seq, low_val, atol=0.1)):
            print(f"    ❌ {name}: No tick reaches Low ({low_val:.4f})")
            constraints_ok = False
        
        # Constraint 5: All ticks within [Low, High] bounds
        out_of_bounds = (tick_seq < low_val - 0.1) | (tick_seq > high_val + 0.1)
        if np.any(out_of_bounds):
            out_count = np.sum(out_of_bounds)
            print(f"    ❌ {name}: {out_count} ticks outside Low-High range")
            constraints_ok = False
    
    if constraints_ok:
        print(f"    ✅ Tick constraints satisfied")
    
    return constraints_ok

def test_seasonal_date_features(date_features):
    """Test seasonal date feature normalization"""
    
    day_of_month, hour_of_day, day_of_week = date_features
    
    print(f"    Date: day_of_month={day_of_month:.3f}, hour_of_day={hour_of_day:.3f}, day_of_week={day_of_week:.3f}")
    
    # All date features should be normalized between 0 and 1
    constraints_ok = True
    
    for value, name in [(day_of_month, "day_of_month"), (hour_of_day, "hour_of_day"), (day_of_week, "day_of_week")]:
        if not (0.0 <= value <= 1.0):
            print(f"    ❌ {name} ({value:.3f}) not normalized between 0 and 1")
            constraints_ok = False
    
    if constraints_ok:
        print(f"    ✅ Seasonal date features properly normalized")
    
    return constraints_ok

def test_feature_population(timestep_data):
    """Test that all features are populated (no unexpected zeros)"""
    
    # Check for completely zero feature sections (which might indicate problems)
    zero_features = np.sum(np.abs(timestep_data) < 1e-8)
    total_features = len(timestep_data)
    
    # Allow some zero values, but not too many
    zero_ratio = zero_features / total_features
    
    if zero_ratio > 0.3:  # More than 30% zeros might indicate a problem
        print(f"    ⚠️  High zero ratio: {zero_features}/{total_features} ({zero_ratio:.1%}) features are zero")
        return False
    
    print(f"    ✅ Feature population good: {zero_features}/{total_features} ({zero_ratio:.1%}) zeros")
    return True

if __name__ == "__main__":
    success = test_generate_operation_mode_comprehensive()
    exit(0 if success else 1)
