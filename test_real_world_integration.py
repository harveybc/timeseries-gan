#!/usr/bin/env python3
"""
Test Real-World Generator Integration
Tests the complete generator pipeline with actual CSV generation
"""

import os
import sys
import subprocess
import pandas as pd

def test_real_world_generator():
    """Test the complete generator pipeline with actual CSV output"""
    
    print("🚀 TESTING REAL-WORLD GENERATOR INTEGRATION")
    print("=" * 70)
    
    # Test parameters
    test_command = [
        "python", "timeseries-gan/main.py",
        "--mode", "generate",
        "--config", "examples/config_phase_4_3.py",
        "--num_samples", "100"  # Small test
    ]
    
    print("🔄 Running actual generator pipeline...")
    print(f"Command: {' '.join(test_command)}")
    
    try:
        # Change to the parent directory to run the command
        os.chdir("/home/harveybc/Documents/GitHub")
        
        # Run the generator
        result = subprocess.run(
            test_command,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        print(f"✅ Generator completed with return code: {result.returncode}")
        
        if result.stdout:
            print(f"📄 STDOUT:\n{result.stdout}")
        if result.stderr:
            print(f"❌ STDERR:\n{result.stderr}")
        
        # Check for output files
        output_dir = "/home/harveybc/Documents/GitHub/timeseries-gan/examples/results"
        if os.path.exists(output_dir):
            output_files = [f for f in os.listdir(output_dir) if f.endswith('.csv')]
            print(f"📁 Found {len(output_files)} output CSV files")
            
            # Analyze the most recent output
            if output_files:
                latest_file = max([os.path.join(output_dir, f) for f in output_files], 
                                key=os.path.getctime)
                print(f"📊 Analyzing latest output: {latest_file}")
                
                # Load and check the CSV
                df = pd.read_csv(latest_file)
                print(f"✅ CSV loaded: {df.shape}")
                
                # Check for tick columns
                tick_columns = [col for col in df.columns if 'tick' in col.lower()]
                print(f"🎯 Found {len(tick_columns)} tick columns: {tick_columns}")
                
                if tick_columns:
                    # Check if tick columns are populated
                    populated_ticks = []
                    for col in tick_columns:
                        non_null_count = df[col].notna().sum()
                        non_zero_count = (df[col] != 0).sum()
                        if non_null_count > 0 and non_zero_count > 0:
                            populated_ticks.append(col)
                    
                    print(f"✅ Populated tick columns: {len(populated_ticks)}/{len(tick_columns)}")
                    
                    if len(populated_ticks) == len(tick_columns):
                        print("🎉 SUCCESS: All tick columns are populated!")
                        print("🎉 Empty tick columns issue has been RESOLVED!")
                        return True
                    else:
                        print("❌ Some tick columns are still empty")
                        empty_cols = [col for col in tick_columns if col not in populated_ticks]
                        print(f"❌ Empty columns: {empty_cols}")
                        return False
                else:
                    print("❌ No tick columns found in output")
                    return False
            else:
                print("❌ No CSV output files found")
                return False
        else:
            print(f"❌ Output directory not found: {output_dir}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Generator timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"❌ Error running generator: {e}")
        return False


if __name__ == "__main__":
    success = test_real_world_generator()
    if success:
        print("\n🏆 INTEGRATION TEST PASSED!")
        print("✅ Constraint-based tick generation is working!")
    else:
        print("\n💥 INTEGRATION TEST FAILED!")
        print("❌ Further debugging needed")
    
    exit(0 if success else 1)
