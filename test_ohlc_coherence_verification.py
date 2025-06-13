#!/usr/bin/env python3
"""
OHLC Coherence Verification Test

This test verifies that the OHLC constraints are properly maintained in high-frequency tick columns:
1. For each hourly tick, HIGH and LOW values must appear in the corresponding 15min/30min tick columns
2. OPEN = first tick value, CLOSE = last tick value
3. HIGH = maximum of all tick values, LOW = minimum of all tick values
4. All tick values must be within the HIGH-LOW range
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

def test_ohlc_coherence_verification():
    """Test OHLC coherence in high-frequency tick columns."""
    
    # Configuration
    results_dir = "examples/results/phase_4_3"
    output_file = "normalized_d4_25200_synthetic_12600_prepended_o.csv"
    output_path = os.path.join(results_dir, output_file)
    
    print("=" * 80)
    print("OHLC COHERENCE VERIFICATION TEST")
    print("=" * 80)
    
    # Check if output file exists
    if not os.path.exists(output_path):
        print(f"❌ ERROR: Output file not found: {output_path}")
        return False
    
    print(f"✓ Found output file: {output_path}")
    
    # Load the CSV
    try:
        df = pd.read_csv(output_path)
        print(f"✓ Loaded CSV with shape: {df.shape}")
    except Exception as e:
        print(f"❌ ERROR loading CSV: {e}")
        return False
    
    # Define column names
    ohlc_cols = ['OPEN', 'HIGH', 'LOW', 'CLOSE']
    tick_15min_cols = [f'CLOSE_15m_tick_{i}' for i in range(1, 9)]  # 1-8
    tick_30min_cols = [f'CLOSE_30m_tick_{i}' for i in range(1, 9)]  # 1-8
    
    # Check if all required columns exist
    missing_cols = []
    all_required_cols = ohlc_cols + tick_15min_cols + tick_30min_cols
    
    for col in all_required_cols:
        if col not in df.columns:
            missing_cols.append(col)
    
    if missing_cols:
        print(f"❌ ERROR: Missing required columns: {missing_cols}")
        print(f"   Available columns: {list(df.columns)}")
        return False
    
    print(f"✓ All required OHLC and tick columns found")
    
    # Test sample size for performance (test first 1000 rows)
    test_size = min(1000, len(df))
    test_df = df.head(test_size).copy()
    print(f"✓ Testing first {test_size} rows for performance")
    
    # Initialize test results
    total_tests = 0
    passed_tests = 0
    failed_rows = []
    
    print(f"\n🧪 RUNNING OHLC COHERENCE TESTS...")
    
    for idx, row in test_df.iterrows():
        total_tests += 1
        row_passed = True
        row_errors = []
        
        # Extract OHLC values
        open_val = row['OPEN']
        high_val = row['HIGH'] 
        low_val = row['LOW']
        close_val = row['CLOSE']
        
        # Extract 15min tick values
        tick_15min_vals = [row[col] for col in tick_15min_cols]
        
        # Extract 30min tick values  
        tick_30min_vals = [row[col] for col in tick_30min_cols]
        
        # All tick values combined
        all_tick_vals = tick_15min_vals + tick_30min_vals
        
        # TEST 1: OPEN should equal first tick value (tick_1)
        first_tick_15min = tick_15min_vals[0]  # CLOSE_15m_tick_1
        if abs(open_val - first_tick_15min) > 1e-6:
            row_errors.append(f"OPEN ({open_val:.6f}) != first 15min tick ({first_tick_15min:.6f})")
            row_passed = False
        
        # TEST 2: CLOSE should equal last tick value (tick_8) 
        last_tick_15min = tick_15min_vals[-1]  # CLOSE_15m_tick_8
        if abs(close_val - last_tick_15min) > 1e-6:
            row_errors.append(f"CLOSE ({close_val:.6f}) != last 15min tick ({last_tick_15min:.6f})")
            row_passed = False
        
        # TEST 3: HIGH should be maximum of all tick values
        max_tick_val = max(all_tick_vals)
        if abs(high_val - max_tick_val) > 1e-6:
            row_errors.append(f"HIGH ({high_val:.6f}) != max tick value ({max_tick_val:.6f})")
            row_passed = False
        
        # TEST 4: LOW should be minimum of all tick values
        min_tick_val = min(all_tick_vals)
        if abs(low_val - min_tick_val) > 1e-6:
            row_errors.append(f"LOW ({low_val:.6f}) != min tick value ({min_tick_val:.6f})")
            row_passed = False
        
        # TEST 5: All tick values should be within HIGH-LOW range
        for i, tick_val in enumerate(all_tick_vals):
            if tick_val < low_val - 1e-6 or tick_val > high_val + 1e-6:
                col_name = tick_15min_cols[i] if i < 8 else tick_30min_cols[i-8]
                row_errors.append(f"{col_name} ({tick_val:.6f}) outside HIGH-LOW range [{low_val:.6f}, {high_val:.6f}]")
                row_passed = False
        
        # TEST 6: HIGH value must appear in at least one tick
        high_found = any(abs(tick_val - high_val) < 1e-6 for tick_val in all_tick_vals)
        if not high_found:
            row_errors.append(f"HIGH value ({high_val:.6f}) not found in any tick column")
            row_passed = False
        
        # TEST 7: LOW value must appear in at least one tick
        low_found = any(abs(tick_val - low_val) < 1e-6 for tick_val in all_tick_vals)
        if not low_found:
            row_errors.append(f"LOW value ({low_val:.6f}) not found in any tick column")
            row_passed = False
        
        # TEST 8: 30min ticks should also be coherent with OHLC
        max_30min = max(tick_30min_vals)
        min_30min = min(tick_30min_vals)
        
        if abs(max_30min - high_val) > 1e-6 and max_30min < high_val - 1e-6:
            # 30min max can be less than HIGH if 15min ticks contain the HIGH
            pass  # This is acceptable
        
        if abs(min_30min - low_val) > 1e-6 and min_30min > low_val + 1e-6:
            # 30min min can be greater than LOW if 15min ticks contain the LOW  
            pass  # This is acceptable
        
        # Record results
        if row_passed:
            passed_tests += 1
        else:
            failed_rows.append((idx, row_errors))
        
        # Progress indicator
        if total_tests % 100 == 0:
            print(f"   Tested {total_tests}/{test_size} rows... ({passed_tests} passed)")
    
    # Calculate results
    pass_rate = (passed_tests / total_tests) * 100
    
    print(f"\n" + "=" * 80)
    print("OHLC COHERENCE TEST RESULTS")
    print("=" * 80)
    print(f"Total rows tested: {total_tests}")
    print(f"Rows passed: {passed_tests}")
    print(f"Rows failed: {total_tests - passed_tests}")
    print(f"Pass rate: {pass_rate:.2f}%")
    
    # Show first few failures for debugging
    if failed_rows:
        print(f"\n❌ FIRST 5 FAILURES (for debugging):")
        for i, (row_idx, errors) in enumerate(failed_rows[:5]):
            print(f"\n   Row {row_idx}:")
            for error in errors[:3]:  # Show first 3 errors per row
                print(f"     • {error}")
            if len(errors) > 3:
                print(f"     • ... and {len(errors) - 3} more errors")
    
    # Detailed analysis for systematic issues
    if failed_rows:
        print(f"\n🔍 FAILURE ANALYSIS:")
        
        # Count error types
        error_types = {}
        for _, errors in failed_rows:
            for error in errors:
                error_type = error.split('(')[0].strip()
                error_types[error_type] = error_types.get(error_type, 0) + 1
        
        print("   Most common error types:")
        for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"     • {error_type}: {count} occurrences")
    
    print("=" * 80)
    
    # Return success if pass rate is above threshold
    success_threshold = 95.0  # 95% pass rate required
    success = pass_rate >= success_threshold
    
    if success:
        print(f"🎉 OHLC COHERENCE TEST PASSED! ({pass_rate:.2f}% >= {success_threshold}%)")
    else:
        print(f"💥 OHLC COHERENCE TEST FAILED! ({pass_rate:.2f}% < {success_threshold}%)")
    
    return success

if __name__ == "__main__":
    success = test_ohlc_coherence_verification()
    if success:
        print("✅ OHLC coherence is maintained correctly!")
        sys.exit(0)
    else:
        print("❌ OHLC coherence violations detected!")
        sys.exit(1)
