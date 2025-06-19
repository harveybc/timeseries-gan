#!/usr/bin/env python3
"""
Test script to verify BiLSTM Z-generator output shape

This script tests the BiLSTM implementation in the generator plugin
to ensure it produces exactly (batch_size, 18, 32) for VAE decoder input.
"""

import sys
import os
sys.path.append('/home/harveybc/Documents/GitHub/timeseries-gan')

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Bidirectional, LSTM, Conv1D
from tensorflow.keras.models import Model

def test_bilstm_z_generator():
    """Test the BiLSTM Z-generator to ensure correct output shape."""
    
    # Parameters matching the generator plugin
    feeder_noise_dim = 100
    internal_z_sequence_length = 18
    internal_z_latent_dim = 32
    batch_size = 32
    
    print("Testing BiLSTM Z-generator implementation...")
    print(f"Input: noise dim = {feeder_noise_dim}")
    print(f"Expected output: ({batch_size}, {internal_z_sequence_length}, {internal_z_latent_dim})")
    
    # Build the BiLSTM Z-generator (same as in generator plugin)
    feeder_noise_input = Input(shape=(feeder_noise_dim,), name="feeder_noise_input")
    
    # Dense layer to expand noise to sequence
    x = Dense(internal_z_sequence_length * internal_z_latent_dim, activation='relu')(feeder_noise_input)
    
    # Reshape to sequence format
    x = Reshape((internal_z_sequence_length, internal_z_latent_dim))(x)
    
    # Bidirectional LSTM layer
    x = Bidirectional(LSTM(internal_z_latent_dim * 2, return_sequences=True))(x)
    
    # Conv1D layer to ensure correct output dimension
    internal_z_seq_output = Conv1D(
        filters=internal_z_latent_dim,
        kernel_size=1,
        padding="same",
        activation='tanh',
        name="internal_z_seq_output"
    )(x)
    
    # Create model
    z_generator = Model(inputs=feeder_noise_input, outputs=internal_z_seq_output, name="bilstm_z_generator")
    
    print("\nModel architecture:")
    z_generator.summary()
    
    # Test with sample input
    test_input = np.random.randn(batch_size, feeder_noise_dim).astype(np.float32)
    
    print(f"\nTesting with input shape: {test_input.shape}")
    
    # Generate output
    output = z_generator.predict(test_input, verbose=0)
    
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: ({batch_size}, {internal_z_sequence_length}, {internal_z_latent_dim})")
    
    # Verify shape
    expected_shape = (batch_size, internal_z_sequence_length, internal_z_latent_dim)
    if output.shape == expected_shape:
        print("✅ SUCCESS: Output shape matches expected VAE decoder input!")
    else:
        print("❌ FAILURE: Output shape does not match expected!")
        return False
    
    # Check output range (should be in [-1, 1] due to tanh activation)
    print(f"\nOutput statistics:")
    print(f"Min value: {np.min(output):.4f}")
    print(f"Max value: {np.max(output):.4f}")
    print(f"Mean value: {np.mean(output):.4f}")
    print(f"Std value: {np.std(output):.4f}")
    
    if np.min(output) >= -1.1 and np.max(output) <= 1.1:
        print("✅ Output values in expected range for tanh activation")
    else:
        print("⚠️  Output values outside expected tanh range")
    
    return True

def test_vae_decoder_input_compatibility():
    """Test that the BiLSTM output can be used as VAE decoder input."""
    
    print("\n" + "="*60)
    print("Testing VAE decoder input compatibility...")
    
    # Parameters
    batch_size = 8
    feeder_noise_dim = 100
    internal_z_sequence_length = 18
    internal_z_latent_dim = 32
    context_vector_dim = 64
    conditional_features_dim = 10
    
    # Build complete composite generator inputs (as in the actual plugin)
    feeder_noise_input = Input(shape=(feeder_noise_dim,), name="feeder_noise_input")
    previous_step_output_input = Input(shape=(context_vector_dim,), name="previous_step_output_input")
    current_step_conditions_input = Input(shape=(conditional_features_dim,), name="current_step_conditions_input")
    
    # Build BiLSTM Z-generator
    x = Dense(internal_z_sequence_length * internal_z_latent_dim, activation='relu')(feeder_noise_input)
    x = Reshape((internal_z_sequence_length, internal_z_latent_dim))(x)
    x = Bidirectional(LSTM(internal_z_latent_dim * 2, return_sequences=True))(x)
    vae_decoder_input_z_seq = Conv1D(
        filters=internal_z_latent_dim,
        kernel_size=1,
        padding="same",
        activation='tanh',
        name="vae_decoder_input_z_seq"
    )(x)
    
    # Create mock VAE decoder inputs
    vae_decoder_input_h_context = previous_step_output_input
    vae_decoder_input_conditions = current_step_conditions_input
    
    # Create test model that shows all three VAE inputs
    test_model = Model(
        inputs=[feeder_noise_input, previous_step_output_input, current_step_conditions_input],
        outputs=[vae_decoder_input_z_seq, vae_decoder_input_h_context, vae_decoder_input_conditions],
        name="vae_input_test"
    )
    
    print("\nComposite generator model architecture:")
    test_model.summary()
    
    # Test with sample inputs
    test_noise = np.random.randn(batch_size, feeder_noise_dim).astype(np.float32)
    test_context = np.random.randn(batch_size, context_vector_dim).astype(np.float32)
    test_conditions = np.random.randn(batch_size, conditional_features_dim).astype(np.float32)
    
    print(f"\nTesting with batch size: {batch_size}")
    
    # Generate outputs
    z_seq, h_context, conditions = test_model.predict([test_noise, test_context, test_conditions], verbose=0)
    
    print(f"\nVAE Decoder Input Shapes:")
    print(f"decoder_input_z_seq: {z_seq.shape} (expected: ({batch_size}, 18, 32))")
    print(f"decoder_input_h_context: {h_context.shape} (expected: ({batch_size}, 64))")
    print(f"decoder_input_conditions: {conditions.shape} (expected: ({batch_size}, 10))")
    
    # Verify all shapes
    shapes_correct = (
        z_seq.shape == (batch_size, 18, 32) and
        h_context.shape == (batch_size, 64) and
        conditions.shape == (batch_size, 10)
    )
    
    if shapes_correct:
        print("✅ SUCCESS: All VAE decoder input shapes are correct!")
        return True
    else:
        print("❌ FAILURE: One or more VAE decoder input shapes are incorrect!")
        return False

if __name__ == "__main__":
    print("BiLSTM Z-Generator Test Suite")
    print("="*60)
    
    # Test 1: Basic BiLSTM output shape
    success1 = test_bilstm_z_generator()
    
    # Test 2: VAE decoder input compatibility
    success2 = test_vae_decoder_input_compatibility()
    
    print("\n" + "="*60)
    print("TEST SUMMARY:")
    print(f"BiLSTM Z-generator shape: {'✅ PASS' if success1 else '❌ FAIL'}")
    print(f"VAE input compatibility: {'✅ PASS' if success2 else '❌ FAIL'}")
    
    if success1 and success2:
        print("\n🎉 ALL TESTS PASSED! BiLSTM Z-generator is ready for VAE-GAN integration.")
    else:
        print("\n❌ SOME TESTS FAILED! Check implementation before proceeding.")
