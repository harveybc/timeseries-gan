#!/usr/bin/env python3
"""Direct test of FeatureExpansionLayer constraint-based tick generation"""

import sys
sys.path.append('/home/harveybc/Documents/GitHub/timeseries-gan')

try:
    import numpy as np
    import tensorflow as tf
    from tsg_plugins.generator_plugin.generator_plugin import FeatureExpansionLayer
    
    # Set up test
    print("Testing FeatureExpansionLayer...")
    
    # Create simple test input (batch_size=2, features=23)
    test_input = np.random.normal(100, 5, (2, 23)).astype(np.float32)
    
    # Create layer and test
    layer = FeatureExpansionLayer()
    input_tensor = tf.constant(test_input)
    
    print(f"Input shape: {input_tensor.shape}")
    
    # Run expansion
    output = layer(input_tensor)
    result = output.numpy()
    
    print(f"Output shape: {result.shape}")
    print(f"Expected 44 features: {result.shape[1] == 44}")
    
    # Check tick columns (positions 23-38)
    tick_data = result[:, 23:39]  # 16 tick columns
    
    print(f"Tick data shape: {tick_data.shape}")
    
    # Check if populated
    for i in range(2):
        sample_ticks = tick_data[i]
        is_populated = not np.allclose(sample_ticks, 0.0, atol=1e-6)
        print(f"Sample {i} ticks populated: {is_populated}")
        if is_populated:
            print(f"  First 4 ticks: {sample_ticks[:4]}")
    
    print("✅ Test completed successfully!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
