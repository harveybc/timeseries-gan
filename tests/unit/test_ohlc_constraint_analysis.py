#!/usr/bin/env python3
"""
OHLC Constraint Analysis and Fix
Analyzes current OHLC constraint violations and implements fixing strategy
"""

import numpy as np
import tensorflow as tf

def analyze_ohlc_constraints(ohlc_data):
    """
    Analyze OHLC constraint violations in generated data
    
    Args:
        ohlc_data: Array of shape (batch_size, 4) with [Open, High, Low, Close]
        
    Returns:
        Dictionary with constraint violation statistics
    """
    print("🔍 ANALYZING OHLC CONSTRAINTS")
    print("=" * 50)
    
    batch_size = ohlc_data.shape[0]
    violations = {
        'high_too_low': 0,    # High < max(Open, Close)
        'low_too_high': 0,    # Low > min(Open, Close)
        'total_samples': batch_size
    }
    
    for i in range(batch_size):
        open_val, high_val, low_val, close_val = ohlc_data[i]
        
        # Check High constraint: High >= max(Open, Close)
        max_oc = max(open_val, close_val)
        if high_val < max_oc:
            violations['high_too_low'] += 1
            print(f"Sample {i}: High violation - H={high_val:.4f} < max(O={open_val:.4f}, C={close_val:.4f})={max_oc:.4f}")
        
        # Check Low constraint: Low <= min(Open, Close)  
        min_oc = min(open_val, close_val)
        if low_val > min_oc:
            violations['low_too_high'] += 1
            print(f"Sample {i}: Low violation - L={low_val:.4f} > min(O={open_val:.4f}, C={close_val:.4f})={min_oc:.4f}")
    
    print(f"\n📊 CONSTRAINT VIOLATION SUMMARY:")
    print(f"  High violations: {violations['high_too_low']}/{batch_size} ({violations['high_too_low']/batch_size*100:.1f}%)")
    print(f"  Low violations: {violations['low_too_high']}/{batch_size} ({violations['low_too_high']/batch_size*100:.1f}%)")
    
    return violations

def fix_ohlc_constraints(ohlc_data, previous_close=None):
    """
    Fix OHLC constraint violations using intelligent value swapping and adjustment
    
    Strategy:
    1. Choose Open as the value closest to previous Close (from O/C candidates)
    2. Ensure High = max(O, H, L, C) 
    3. Ensure Low = min(O, H, L, C)
    4. Assign remaining value to Close
    
    Args:
        ohlc_data: Array of shape (batch_size, 4) with [Open, High, Low, Close]
        previous_close: Previous close values for continuity (optional)
        
    Returns:
        Fixed OHLC data with constraints satisfied
    """
    print("\n🔧 FIXING OHLC CONSTRAINTS")
    print("=" * 40)
    
    fixed_ohlc = np.copy(ohlc_data)
    batch_size = ohlc_data.shape[0]
    
    for i in range(batch_size):
        original_values = ohlc_data[i].copy()
        sorted_values = np.sort(original_values)
        
        # Step 1: Choose Open value
        if previous_close is not None:
            # Find closest value to previous close (excluding extreme high/low)
            prev_close = previous_close[i] if isinstance(previous_close, np.ndarray) else previous_close
            
            # Consider middle two values for Open (avoid using extreme high/low as Open)
            middle_values = sorted_values[1:3]  # Second lowest and second highest
            distances = np.abs(middle_values - prev_close)
            open_idx = np.argmin(distances)
            open_val = middle_values[open_idx]
        else:
            # Use second lowest value as Open (conservative approach)
            open_val = sorted_values[1]
        
        # Step 2: Assign High and Low
        high_val = sorted_values[3]  # Highest value
        low_val = sorted_values[0]   # Lowest value
        
        # Step 3: Assign Close (remaining value)
        remaining_values = original_values[original_values != high_val]
        remaining_values = remaining_values[remaining_values != low_val]
        remaining_values = remaining_values[remaining_values != open_val]
        
        if len(remaining_values) > 0:
            close_val = remaining_values[0]
        else:
            # If all values are the same, use open_val
            close_val = open_val
        
        # Assign fixed values
        fixed_ohlc[i] = [open_val, high_val, low_val, close_val]
        
        # Verify constraints
        assert high_val >= max(open_val, close_val), f"High constraint failed: {high_val} < {max(open_val, close_val)}"
        assert low_val <= min(open_val, close_val), f"Low constraint failed: {low_val} > {min(open_val, close_val)}"
        
        print(f"Sample {i}: [{original_values[0]:.4f}, {original_values[1]:.4f}, {original_values[2]:.4f}, {original_values[3]:.4f}] "
              f"→ [{open_val:.4f}, {high_val:.4f}, {low_val:.4f}, {close_val:.4f}]")
    
    return fixed_ohlc

def fix_ohlc_constraints_tensorflow(ohlc_tensor, previous_close=None):
    """
    TensorFlow version of OHLC constraint fixing for use in Keras layers
    
    Args:
        ohlc_tensor: Tensor of shape (batch_size, 4) with [Open, High, Low, Close]
        previous_close: Previous close values tensor (optional)
        
    Returns:
        Fixed OHLC tensor with constraints satisfied
    """
    batch_size = tf.shape(ohlc_tensor)[0]
    
    # Sort values for each sample
    sorted_values = tf.sort(ohlc_tensor, axis=1)  # Shape: (batch_size, 4)
    
    # Assign High and Low (extreme values)
    high_val = sorted_values[:, 3:4]  # Highest value, shape: (batch_size, 1)
    low_val = sorted_values[:, 0:1]   # Lowest value, shape: (batch_size, 1)
    
    # Choose Open from middle values (avoid extremes)
    if previous_close is not None:
        # Choose middle value closest to previous close
        middle_values = sorted_values[:, 1:3]  # Shape: (batch_size, 2)
        
        # Expand previous_close to match middle_values shape for broadcasting
        if len(previous_close.shape) == 1:
            prev_close_expanded = tf.expand_dims(previous_close, axis=1)  # (batch_size, 1)
        else:
            prev_close_expanded = previous_close
        
        # Calculate distances
        distances = tf.abs(middle_values - prev_close_expanded)  # (batch_size, 2)
        closest_idx = tf.argmin(distances, axis=1)  # (batch_size,)
        
        # Select closest middle value as Open
        batch_indices = tf.range(batch_size)
        indices = tf.stack([batch_indices, closest_idx], axis=1)
        open_val = tf.expand_dims(tf.gather_nd(middle_values, indices), axis=1)  # (batch_size, 1)
    else:
        # Use second lowest as Open
        open_val = sorted_values[:, 1:2]  # Shape: (batch_size, 1)
    
    # Close gets the remaining middle value
    close_val = sorted_values[:, 2:3]  # Second highest value
    
    # Concatenate fixed OHLC
    fixed_ohlc = tf.concat([open_val, high_val, low_val, close_val], axis=1)
    
    return fixed_ohlc

# Test the constraint fixing
if __name__ == "__main__":
    print("🧪 TESTING OHLC CONSTRAINT FIXING")
    print("=" * 60)
    
    # Generate test data with likely constraint violations
    np.random.seed(42)
    batch_size = 5
    
    # Generate random OHLC data (likely to have violations)
    test_ohlc = np.random.normal(100, 5, (batch_size, 4))
    
    print("📊 ORIGINAL DATA:")
    for i in range(batch_size):
        o, h, l, c = test_ohlc[i]
        print(f"Sample {i}: O={o:.4f}, H={h:.4f}, L={l:.4f}, C={c:.4f}")
    
    # Analyze violations
    violations = analyze_ohlc_constraints(test_ohlc)
    
    # Fix constraints
    previous_closes = np.random.normal(100, 2, batch_size)  # Simulated previous closes
    fixed_ohlc = fix_ohlc_constraints(test_ohlc, previous_closes)
    
    print(f"\n✅ VERIFICATION - Analyzing fixed data:")
    fixed_violations = analyze_ohlc_constraints(fixed_ohlc)
    
    # Test TensorFlow version
    print(f"\n🔧 TESTING TENSORFLOW VERSION:")
    ohlc_tensor = tf.constant(test_ohlc, dtype=tf.float32)
    prev_close_tensor = tf.constant(previous_closes, dtype=tf.float32)
    
    tf_fixed_ohlc = fix_ohlc_constraints_tensorflow(ohlc_tensor, prev_close_tensor)
    
    print("TensorFlow fixed OHLC:")
    tf_result = tf_fixed_ohlc.numpy()
    for i in range(batch_size):
        o, h, l, c = tf_result[i]
        print(f"Sample {i}: O={o:.4f}, H={h:.4f}, L={l:.4f}, C={c:.4f}")
    
    # Verify TensorFlow version
    print(f"\n✅ VERIFICATION - TensorFlow fixed data:")
    tf_violations = analyze_ohlc_constraints(tf_result)
    
    print(f"\n🎉 CONSTRAINT FIXING SUMMARY:")
    print(f"Original violations: {violations['high_too_low'] + violations['low_too_high']}")
    print(f"NumPy fixed violations: {fixed_violations['high_too_low'] + fixed_violations['low_too_high']}")
    print(f"TensorFlow fixed violations: {tf_violations['high_too_low'] + tf_violations['low_too_high']}")
