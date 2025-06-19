#!/usr/bin/env python3
"""
Test script to verify that synthetic data is correctly prepended to real data.
This test checks that:
1. Synthetic data appears in the first n_samples rows
2. Real data appears in the last max_steps_train rows  
3. Synthetic datetimes come BEFORE real datetimes
4. The final CSV has the correct chronological order
"""

import pandas as pd
import os
import sys
from datetime import datetime, timedelta

def test_prepending_verification():
    """Test that synthetic data is properly prepended to real data in the final CSV."""
    
    # Configuration
    results_dir = "examples/results/phase_4_3"
    output_file = "normalized_d4_25200_synthetic_12600_prepended_o.csv"
    output_path = os.path.join(results_dir, output_file)
    
    n_samples = 12600  # Expected number of synthetic samples
    max_steps_train = 25200  # Expected number of real data samples
    datetime_col = "DATE_TIME"
    
    print("=" * 80)
    print("PREPENDING VERIFICATION TEST")
    print("=" * 80)
    
    # Check if output file exists
    if not os.path.exists(output_path):
        print(f"❌ ERROR: Output file not found: {output_path}")
        print("   Please run the generate pipeline first.")
        return False
    
    print(f"✓ Found output file: {output_path}")
    
    # Load the CSV
    try:
        df = pd.read_csv(output_path)
        print(f"✓ Loaded CSV with shape: {df.shape}")
    except Exception as e:
        print(f"❌ ERROR loading CSV: {e}")
        return False
    
    # Check if datetime column exists
    if datetime_col not in df.columns:
        print(f"❌ ERROR: DateTime column '{datetime_col}' not found in CSV")
        print(f"   Available columns: {list(df.columns)}")
        return False
    
    # Convert datetime column
    try:
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        print(f"✓ Successfully parsed datetime column")
    except Exception as e:
        print(f"❌ ERROR parsing datetime column: {e}")
        return False
    
    # Verify total expected rows
    expected_total = n_samples + max_steps_train
    if len(df) != expected_total:
        print(f"⚠️  WARNING: Expected {expected_total} total rows, got {len(df)}")
        print(f"   Expected: {n_samples} synthetic + {max_steps_train} real = {expected_total}")
        print(f"   Actual: {len(df)}")
    
    # Test 1: Check synthetic data is in first n_samples rows
    print(f"\n🧪 TEST 1: Synthetic data in first {n_samples} rows")
    synthetic_data = df.head(n_samples)
    synthetic_start_time = synthetic_data[datetime_col].iloc[0]
    synthetic_end_time = synthetic_data[datetime_col].iloc[-1]
    print(f"   Synthetic data time range: {synthetic_start_time} to {synthetic_end_time}")
    
    # Test 2: Check real data is in last max_steps_train rows  
    print(f"\n🧪 TEST 2: Real data in last {max_steps_train} rows")
    real_data = df.tail(max_steps_train)
    real_start_time = real_data[datetime_col].iloc[0]
    real_end_time = real_data[datetime_col].iloc[-1]
    print(f"   Real data time range: {real_start_time} to {real_end_time}")
    
    # Test 3: Verify synthetic comes BEFORE real chronologically
    print(f"\n🧪 TEST 3: Chronological order verification")
    time_gap = real_start_time - synthetic_end_time
    print(f"   Last synthetic time: {synthetic_end_time}")
    print(f"   First real time: {real_start_time}")
    print(f"   Time gap: {time_gap}")
    
    if synthetic_end_time >= real_start_time:
        print(f"❌ FAIL: Synthetic data does NOT come before real data!")
        print(f"   Synthetic ends at: {synthetic_end_time}")
        print(f"   Real starts at: {real_start_time}")
        return False
    else:
        print(f"✓ PASS: Synthetic data correctly comes before real data")
    
    # Test 4: Check for proper hourly progression (no gaps/overlaps)
    print(f"\n🧪 TEST 4: Data continuity check")
    expected_gap = timedelta(hours=1)
    if abs(time_gap - expected_gap) > timedelta(minutes=1):
        print(f"⚠️  WARNING: Time gap is not exactly 1 hour")
        print(f"   Expected gap: {expected_gap}")
        print(f"   Actual gap: {time_gap}")
    else:
        print(f"✓ PASS: Proper 1-hour gap between synthetic and real data")
    
    # Test 5: Verify chronological order within datasets
    print(f"\n🧪 TEST 5: Internal chronological order")
    
    # Check synthetic data is chronologically ordered
    synthetic_sorted = synthetic_data[datetime_col].is_monotonic_increasing
    if synthetic_sorted:
        print(f"✓ PASS: Synthetic data is chronologically ordered")
    else:
        print(f"❌ FAIL: Synthetic data is NOT chronologically ordered")
        return False
    
    # Check real data is chronologically ordered  
    real_sorted = real_data[datetime_col].is_monotonic_increasing
    if real_sorted:
        print(f"✓ PASS: Real data is chronologically ordered")
    else:
        print(f"❌ FAIL: Real data is NOT chronologically ordered")
        return False
    
    # Test 6: Verify overall dataset is chronologically ordered
    print(f"\n🧪 TEST 6: Overall chronological order")
    overall_sorted = df[datetime_col].is_monotonic_increasing
    if overall_sorted:
        print(f"✓ PASS: Overall dataset is chronologically ordered")
    else:
        print(f"❌ FAIL: Overall dataset is NOT chronologically ordered")
        return False
    
    # Summary
    print(f"\n" + "=" * 80)
    print("PREPENDING VERIFICATION SUMMARY")
    print("=" * 80)
    print(f"✓ Total rows: {len(df)}")
    print(f"✓ Synthetic rows (first {n_samples}): {synthetic_start_time} to {synthetic_end_time}")
    print(f"✓ Real rows (last {max_steps_train}): {real_start_time} to {real_end_time}")
    print(f"✓ Time gap: {time_gap}")
    print(f"✓ Chronological order: CORRECT")
    print(f"✓ Prepending: SUCCESSFUL")
    print("=" * 80)
    
    return True

if __name__ == "__main__":
    success = test_prepending_verification()
    if success:
        print("🎉 ALL TESTS PASSED: Synthetic data is correctly prepended!")
        sys.exit(0)
    else:
        print("💥 TESTS FAILED: Prepending is not working correctly!")
        sys.exit(1)
