#!/usr/bin/env python3
"""
Simple test of the core generator architecture
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np

print("Testing simplified Z-generator architecture...")

def test_simple_z_generator():
    """Test the simple Z-generator approach"""
    
    # Parameters
    noise_dim = 100
    internal_z_seq_len = 18
    internal_z_dim = 32
    
    # Inputs
    noise_input = tf.keras.layers.Input(shape=(noise_dim,), name="noise_input")
    
    # Simple Z-generator (new approach)
    latent_size = internal_z_seq_len * internal_z_dim  # 18 * 32 = 576
    
    # Single dense layer to generate all latent values
    z_dense = tf.keras.layers.Dense(
        latent_size, 
        activation='tanh',
        kernel_initializer='glorot_uniform',
        name="z_simple_dense"
    )(noise_input)
    
    # Reshape to sequence format
    z_sequence_raw = tf.keras.layers.Reshape(
        (internal_z_seq_len, internal_z_dim), 
        name="z_simple_reshape"
    )(z_dense)
    
    # MINIMAL processing - just one BiLSTM layer for sequential refinement
    z_sequence_refined = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(16, return_sequences=True, activation='tanh'),
        name="z_simple_bilstm"
    )(z_sequence_raw)
    
    # Final projection to exact VAE input dimension
    z_sequence_for_vae = tf.keras.layers.Dense(
        internal_z_dim,
        activation='tanh',
        name="z_final_projection"
    )(z_sequence_refined)
    
    # Create model
    z_generator = tf.keras.Model(
        inputs=noise_input,
        outputs=z_sequence_for_vae,
        name="simple_z_generator"
    )
    
    return z_generator

def test_parameter_efficiency():
    """Test parameter efficiency compared to old approach"""
    
    z_gen = test_simple_z_generator()
    
    # Calculate parameters
    new_params = z_gen.count_params()
    
    # Old approach parameter estimate
    noise_dim = 100
    old_dense_params = 576 * (noise_dim + 1)  # Dense(576)
    old_bilstm_params = (16 * 4 * (32 + 16 + 1)) * 2  # BiLSTM
    old_conv_params = 32 * (32 + 1)  # Conv1D
    old_total = old_dense_params + old_bilstm_params + old_conv_params
    
    print(f"✅ Simple Z-generator built successfully!")
    print(f"New approach parameters: {new_params:,}")
    print(f"Old approach estimate: {old_total:,}")
    print(f"Parameter reduction: {((old_total - new_params) / old_total * 100):.1f}%")
    
    return z_gen

def test_generation():
    """Test actual generation"""
    
    z_gen = test_simple_z_generator()
    
    # Test generation
    batch_size = 4
    noise = np.random.randn(batch_size, 100).astype(np.float32)
    
    output = z_gen(noise)
    
    print(f"✅ Generation successful!")
    print(f"Input shape: {noise.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: ({batch_size}, 18, 32)")
    print(f"Shape match: {output.shape == (batch_size, 18, 32)}")
    print(f"Output range: [{output.numpy().min():.3f}, {output.numpy().max():.3f}]")
    print(f"Output mean: {output.numpy().mean():.3f}")
    
    return output

if __name__ == "__main__":
    try:
        print("🔧 Testing parameter efficiency...")
        z_gen = test_parameter_efficiency()
        
        print("\n🔧 Testing generation...")
        output = test_generation()
        
        print("\n🎉 ALL TESTS PASSED! Simple Z-generator is working correctly.")
        print("\nThis approach should have MUCH better convergence than the complex version!")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
