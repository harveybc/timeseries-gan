#!/usr/bin/env python3
"""
Test the simplified generator implementation
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import sys
import traceback

print("Testing simplified generator...")

try:
    # Add the project path
    sys.path.append('/home/harveybc/Documents/GitHub/timeseries-gan')
    
    from tsg_plugins.generator_plugin.generator_plugin import GeneratorPlugin
    
    # Create minimal config for testing
    config = {
        'operation_mode': 'train',
        'vae_decoder_model_path_param': 'examples/results/phase_4_3/phase_4_3_cnn_small_decoder_model.keras',
        'noise_dim': 100,
        'conditional_features_dim': 10,
        'context_vector_dim': 64,
        'print_model_summary': False
    }
    
    pipeline_config = {}
    
    print("Creating GeneratorPlugin...")
    generator = GeneratorPlugin(config, pipeline_config)
    
    print("Building generator model...")
    test_input_shape = (100,)
    test_condition_shape = (10,)
    model = generator.build(test_input_shape, test_condition_shape)
    
    print(f'✅ SUCCESS: Model built successfully!')
    print(f'Model parameters: {model.count_params():,}')
    print(f'Input shapes: {[inp.shape for inp in model.inputs]}')
    print(f'Output shape: {model.output.shape}')
    
    # Test generation
    print("Testing generation...")
    batch_size = 4
    noise = np.random.randn(batch_size, 100).astype(np.float32)
    context = np.random.randn(batch_size, 64).astype(np.float32)
    conditions = np.random.randn(batch_size, 10).astype(np.float32)
    
    output = model.predict([noise, context, conditions], verbose=0)
    print(f'✅ Generation test successful!')
    print(f'Generated output shape: {output.shape}')
    print(f'Output stats: min={output.min():.3f}, max={output.max():.3f}, mean={output.mean():.3f}')
    
    # Compare with old approach parameter count
    old_dense_params = 576 * (100 + 1)  # Dense(576) parameters
    old_bilstm_params = (16 * 4 * (32 + 16 + 1)) * 2  # BiLSTM parameters
    old_conv_params = 32 * (32 + 1)  # Conv1D parameters
    old_total = old_dense_params + old_bilstm_params + old_conv_params
    
    new_params = model.count_params()
    
    print(f"\n📊 PARAMETER COMPARISON:")
    print(f"Old Dense+Reshape approach: ~{old_total:,} parameters")
    print(f"New simplified approach: {new_params:,} parameters")
    print(f"Reduction: {((old_total - new_params) / old_total * 100):.1f}%")
    
    print(f"\n🎉 ALL TESTS PASSED! Simplified generator is ready for training.")

except Exception as e:
    print(f'❌ ERROR: {e}')
    traceback.print_exc()
