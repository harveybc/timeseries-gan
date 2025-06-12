#!/usr/bin/env python3
"""
Integration test for constraint-based sub-periodicity tick generation
Tests the complete generator pipeline to verify tick columns are populated
"""

import numpy as np
import tensorflow as tf
from tsg_plugins.generator_plugin.generator_plugin import GeneratorPlugin

def test_generator_tick_integration():
    """Test constraint-based tick generation in complete generator pipeline"""
    
    print("🚀 TESTING GENERATOR INTEGRATION WITH CONSTRAINT-BASED TICKS")
    print("=" * 70)
    
    # Create generator plugin
    generator = GeneratorPlugin()
    
    # Configure for 23->44 feature transformation
    generator.config = {
        'input_dim': 23,
        'latent_dim': 16,
        'hidden_units': [128, 64, 32],
        'output_dim': 44,
        'dropout_rate': 0.2
    }
    
    print("✅ Building generator model...")
    generator.build_model()
    
    print(f"✅ Model created: {generator.model.input_shape} -> {generator.model.output_shape}")
    
    # Generate test data
    batch_size = 5
    test_input = np.random.randn(batch_size, 23).astype(np.float32)
    
    print(f"✅ Generated test input: {test_input.shape}")
    
    # Run inference
    print("🔄 Running generator inference...")
    synthetic_output = generator.model(test_input)
    
    print(f"✅ Generator output shape: {synthetic_output.shape}")
    
    # Extract OHLC and tick columns
    # Assuming standard feature order: [other features, OHLC, ticks]
    # Last 20 features: OHLC (4) + 15m_ticks (8) + 30m_ticks (8) = 20
    ohlc_start = 44 - 20  # Position 24
    tick_start = 44 - 16   # Position 28
    
    ohlc_data = synthetic_output[:, ohlc_start:ohlc_start+4]  # OHLC
    tick_data = synthetic_output[:, tick_start:]              # 16 tick columns
    
    print(f"✅ OHLC data shape: {ohlc_data.shape}")
    print(f"✅ Tick data shape: {tick_data.shape}")
    
    # Split ticks into 15m and 30m
    close_15m_ticks = tick_data[:, :8]
    close_30m_ticks = tick_data[:, 8:]
    
    print("\n📊 ANALYZING GENERATED TICK DATA:")
    print("=" * 45)
    
    all_populated = True
    
    for i in range(batch_size):
        sample_ohlc = ohlc_data[i]
        sample_15m = close_15m_ticks[i]
        sample_30m = close_30m_ticks[i]
        
        # Check if columns are populated (not all zeros)
        empty_15m = np.allclose(sample_15m, 0.0, atol=1e-6)
        empty_30m = np.allclose(sample_30m, 0.0, atol=1e-6)
        
        print(f"\nSample {i+1}:")
        print(f"  OHLC: O={sample_ohlc[0]:.4f}, H={sample_ohlc[1]:.4f}, L={sample_ohlc[2]:.4f}, C={sample_ohlc[3]:.4f}")
        print(f"  15m ticks: {[f'{x:.4f}' for x in sample_15m]}")
        print(f"  30m ticks: {[f'{x:.4f}' for x in sample_30m]}")
        
        if empty_15m or empty_30m:
            print(f"  ❌ EMPTY COLUMNS DETECTED! 15m_empty={empty_15m}, 30m_empty={empty_30m}")
            all_populated = False
        else:
            print(f"  ✅ All tick columns populated!")
    
    # Summary
    print(f"\n🏁 INTEGRATION TEST RESULTS:")
    print("=" * 35)
    
    if all_populated:
        print("✅ SUCCESS: All tick columns are populated!")
        print("✅ Empty tick columns issue has been RESOLVED!")
        return True
    else:
        print("❌ FAILURE: Some tick columns are still empty!")
        print("❌ Empty tick columns issue persists!")
        return False

if __name__ == "__main__":
    success = test_generator_tick_integration()
    exit(0 if success else 1)
