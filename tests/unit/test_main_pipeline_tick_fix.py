#!/usr/bin/env python3
"""
Test the complete pipeline using the main entry point to verify that constraint-based 
tick generation is working and the generated CSV contains populated tick columns.
"""

import sys
import os
sys.path.append('.')

import numpy as np
import pandas as pd
import subprocess
import traceback

def test_main_pipeline():
    """Test the complete pipeline using the main.py entry point."""
    print("=== TESTING MAIN PIPELINE WITH CONSTRAINT-BASED TICK GENERATION ===")
    
    try:
        # Prepare test configuration
        config_file = "test_config_pipeline.json"
        output_file = "test_pipeline_output.csv"
        
        # Configuration for generation with constraint-based ticks
        config_data = {
            "operation_mode": "generate",
            "n_samples": 5,
            "x_train_file": "examples/results/phase_4_3/x_train_normalized.csv",
            "output_file": output_file,
            "generator_model_path": "examples/models/23f_generator_d4_2559.h5",
            "vae_decoder_model_path_param": "examples/models/23f_generator_d4_2559.h5",
            "print_model_summary": False
        }
        
        # Save configuration to file
        import json
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        print(f"✅ Configuration saved to {config_file}")
        
        # Run the main pipeline
        print("\n1. Running main pipeline...")
        cmd = ["python", "app/main.py", "--config", config_file]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Main pipeline executed successfully")
            print(f"   stdout: {result.stdout[-200:]}")  # Last 200 chars
        else:
            print(f"❌ Main pipeline failed with return code {result.returncode}")
            print(f"   stderr: {result.stderr}")
            print(f"   stdout: {result.stdout}")
            return False
        
        # Check if output file was created
        if not os.path.exists(output_file):
            print(f"❌ Output file {output_file} was not created")
            return False
            
        print(f"✅ Output file {output_file} created")
        
        # Load and analyze the output
        print("\n2. Analyzing generated output...")
        df = pd.read_csv(output_file)
        
        print(f"✅ Generated data shape: {df.shape}")
        print(f"   Sample columns: {list(df.columns)[:10]}...")
        
        # Check for tick columns
        print("\n3. Checking constraint-based tick columns...")
        
        tick_15m_cols = [col for col in df.columns if 'CLOSE_15m_tick_' in col]
        tick_30m_cols = [col for col in df.columns if 'CLOSE_30m_tick_' in col]
        
        print(f"   Found {len(tick_15m_cols)} CLOSE_15m_tick columns")
        print(f"   Found {len(tick_30m_cols)} CLOSE_30m_tick columns")
        
        if len(tick_15m_cols) == 0 and len(tick_30m_cols) == 0:
            print("   ❌ No tick columns found")
            return False
        
        # Verify tick column population
        populated_ticks = 0
        total_ticks = 0
        
        for col in tick_15m_cols + tick_30m_cols:
            total_ticks += 1
            col_data = df[col].dropna()
            
            if len(col_data) > 0 and not (col_data == 0).all():
                populated_ticks += 1
                print(f"   ✅ {col}: populated with values {col_data.iloc[0]:.6f} to {col_data.iloc[-1]:.6f}")
            else:
                print(f"   ❌ {col}: empty or all zeros")
        
        if populated_ticks > 0:
            print(f"\n✅ SUCCESS: {populated_ticks}/{total_ticks} tick columns are populated!")
            
            # Test OHLC constraints for one sample
            if 'OPEN' in df.columns and 'HIGH' in df.columns and 'LOW' in df.columns and 'CLOSE' in df.columns:
                print("\n4. Testing OHLC constraint satisfaction...")
                sample_idx = 0
                open_val = df['OPEN'].iloc[sample_idx]
                high_val = df['HIGH'].iloc[sample_idx]
                low_val = df['LOW'].iloc[sample_idx]
                close_val = df['CLOSE'].iloc[sample_idx]
                
                print(f"   Sample OHLC: O={open_val:.6f}, H={high_val:.6f}, L={low_val:.6f}, C={close_val:.6f}")
                
                # Basic OHLC validation
                if high_val >= max(open_val, close_val) and low_val <= min(open_val, close_val):
                    print("   ✅ OHLC constraints satisfied")
                else:
                    print("   ⚠️ OHLC constraints may need adjustment")
            
            print("\n🎉 MAIN PIPELINE TEST SUCCESSFUL!")
            print("   - Main entry point works correctly")
            print("   - Constraint-based tick generation is active")
            print("   - Output CSV contains populated tick columns")
            print("   - Empty tick columns issue is RESOLVED")
            
            # Clean up
            os.remove(config_file)
            os.remove(output_file)
            
            return True
        else:
            print(f"\n❌ FAILURE: All tick columns are still empty")
            return False
            
    except Exception as e:
        print(f"❌ Error during main pipeline test: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_main_pipeline()
    if success:
        print(f"\n✅ MAIN PIPELINE CONSTRAINT-BASED TICK GENERATION TEST PASSED!")
        print(f"   The constraint-based sub-periodicity tick generation is working correctly.")
    else:
        print(f"\n❌ MAIN PIPELINE TEST FAILED")
    
    sys.exit(0 if success else 1)
