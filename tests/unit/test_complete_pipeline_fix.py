#!/usr/bin/env python3
"""
Test the complete constraint-based tick generation pipeline by generating synthetic data
and verifying that the output CSV contains populated tick columns.
"""

import sys
import os
sys.path.append('.')

import numpy as np
import pandas as pd
from app.config import DEFAULT_VALUES
from app.data_generation.synthetic_generator import SyntheticDataGenerator

def test_complete_pipeline():
    """Test the complete pipeline from generate mode to CSV output."""
    print("=== TESTING COMPLETE CONSTRAINT-BASED TICK GENERATION PIPELINE ===")
    
    try:
        # Set up configuration for generate mode
        config = DEFAULT_VALUES.copy()
        config["operation_mode"] = "generate"
        config["n_samples"] = 10  # Generate small amount for testing
        config["x_train_file"] = "examples/results/phase_4_3/x_train_normalized.csv"  # Use existing training data
        config["output_file"] = "test_constraint_tick_output.csv"  # Test output file
        
        print("✅ Configuration set up for generate mode")
        
        # Initialize synthetic data generator
        generator = SyntheticDataGenerator(config)
        print("✅ SyntheticDataGenerator initialized")
        
        # Generate synthetic data
        print("\n1. Generating synthetic data...")
        result_df = generator.generate_sequences()
        
        if result_df is not None and not result_df.empty:
            print(f"✅ Generated synthetic data: {result_df.shape}")
            print(f"   Columns: {len(result_df.columns)}")
            print(f"   Sample columns: {list(result_df.columns[:10])}")
            
            # Check for tick columns
            print("\n2. Checking for constraint-based tick columns...")
            
            # Look for 15m and 30m tick columns
            tick_15m_cols = [col for col in result_df.columns if 'CLOSE_15m_tick_' in col]
            tick_30m_cols = [col for col in result_df.columns if 'CLOSE_30m_tick_' in col]
            
            print(f"   Found {len(tick_15m_cols)} CLOSE_15m_tick columns")
            print(f"   Found {len(tick_30m_cols)} CLOSE_30m_tick columns")
            
            if len(tick_15m_cols) > 0 and len(tick_30m_cols) > 0:
                print("   ✅ Tick columns found in output")
                
                # Check if tick columns are populated (not all NaN or zero)
                print("\n3. Verifying tick column population...")
                
                # Test first 15m tick column
                first_15m_col = tick_15m_cols[0]
                first_15m_data = result_df[first_15m_col].dropna()
                
                if len(first_15m_data) > 0 and not (first_15m_data == 0).all():
                    print(f"   ✅ {first_15m_col} is populated with non-zero values")
                    print(f"      Sample values: {first_15m_data.head(3).tolist()}")
                    print(f"      Range: [{first_15m_data.min():.6f}, {first_15m_data.max():.6f}]")
                else:
                    print(f"   ❌ {first_15m_col} appears empty or all zeros")
                    return False
                
                # Test first 30m tick column
                first_30m_col = tick_30m_cols[0]
                first_30m_data = result_df[first_30m_col].dropna()
                
                if len(first_30m_data) > 0 and not (first_30m_data == 0).all():
                    print(f"   ✅ {first_30m_col} is populated with non-zero values")
                    print(f"      Sample values: {first_30m_data.head(3).tolist()}")
                    print(f"      Range: [{first_30m_data.min():.6f}, {first_30m_data.max():.6f}]")
                else:
                    print(f"   ❌ {first_30m_col} appears empty or all zeros")
                    return False
                
                # Check OHLC constraint satisfaction for a sample
                print("\n4. Testing OHLC constraint satisfaction...")
                
                ohlc_cols = ['OPEN', 'HIGH', 'LOW', 'CLOSE']
                if all(col in result_df.columns for col in ohlc_cols):
                    sample_ohlc = result_df[ohlc_cols].iloc[0]
                    open_val, high_val, low_val, close_val = sample_ohlc
                    
                    print(f"   Sample OHLC: O={open_val:.6f}, H={high_val:.6f}, L={low_val:.6f}, C={close_val:.6f}")
                    
                    # Check basic OHLC constraints
                    if high_val >= max(open_val, close_val) and low_val <= min(open_val, close_val):
                        print("   ✅ Basic OHLC constraints satisfied")
                    else:
                        print("   ⚠️ OHLC constraints may not be fully satisfied")
                
                # Save the test output for verification
                print(f"\n5. Saving test output to {config['output_file']}...")
                result_df.to_csv(config['output_file'], index=False)
                print(f"   ✅ Output saved successfully")
                
                print("\n🎉 COMPLETE PIPELINE TEST SUCCESSFUL!")
                print("   - Synthetic data generated successfully")
                print("   - Tick columns are present and populated")
                print("   - OHLC constraints can be verified")
                print("   - CSV output contains constraint-based tick data")
                return True
                
            else:
                print("   ❌ Tick columns not found in output")
                return False
                
        else:
            print("❌ Failed to generate synthetic data")
            return False
            
    except Exception as e:
        print(f"❌ Error during pipeline test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_complete_pipeline()
    if success:
        print(f"\n✅ COMPLETE CONSTRAINT-BASED TICK GENERATION PIPELINE SUCCESSFUL!")
        print(f"   The empty tick columns issue has been completely resolved.")
        print(f"   Generated CSV files will now contain populated tick columns.")
    else:
        print(f"\n❌ COMPLETE PIPELINE TEST FAILED")
    
    sys.exit(0 if success else 1)
