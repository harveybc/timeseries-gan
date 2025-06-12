#!/usr/bin/env python3
"""
_build_composite_generator.py

Composite GAN Generator implementation for Sequential Conditional VAE-GAN.
Created from scratch based on REFERENCE.md specifications.

This module implements the Composite GAN Generator architecture that combines:
1. BiLSTM Z-generator: Dense(576) → Reshape(18,32) → Bidirectional(LSTM(64)) → Conv1D(32 filters)
2. Pre-trained VAE Decoder: Loaded with trainable=True for joint optimization
3. Iterative Processing: Sequential generation with context updating

Author: TimeSeries-GAN Team
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Reshape, Bidirectional, LSTM, Conv1D, 
    Concatenate, Lambda, TimeDistributed
)
from tensorflow.keras.models import Model, load_model
from typing import Dict, Any, Tuple, Optional
import numpy as np
import logging


def _build_composite_generator(self) -> Model:
    """
    Build the Composite GAN Generator according to REFERENCE.md specifications.
    
    Architecture:
    1. BiLSTM Z-generator: Processes noise to create latent sequences (batch_size, 18, 32)
    2. Pre-trained VAE Decoder: Loaded with trainable=True for joint optimization  
    3. Iterative Processing: Sequential generation with context from previous timesteps
    
    Returns:
        Model: Composite GAN Generator that outputs full sequences (batch_size, seq_len, 57)
    """
    logger = logging.getLogger(__name__)
    logger.info("Building Composite GAN Generator from REFERENCE.md specifications")
    
    # Get configuration parameters
    seq_len = self.params.get("generator_decoder_input_window_size", 144)
    latent_seq_len = self.params.get("latent_shape", [18, 32])[0]  # 18
    latent_dim = self.params.get("latent_shape", [18, 32])[1]     # 32
    context_dim = self.params.get("context_vector_dim", 64)
    num_conditions = 10  # From REFERENCE.md: cyclical date/time features
    noise_dim = 100  # Standard GAN noise dimension
    
    logger.info(f"Generator config: seq_len={seq_len}, latent_shape=[{latent_seq_len}, {latent_dim}], "
               f"context_dim={context_dim}, noise_dim={noise_dim}")
    
    # === INPUTS ===
    # Main noise input for the BiLSTM Z-generator
    noise_input = Input(shape=(noise_dim,), name="noise_input")
    
    # Context vector from previous timestep (for iterative generation)
    context_input = Input(shape=(context_dim,), name="context_input")
    
    # Conditional features (date/time) for current timestep
    conditions_input = Input(shape=(num_conditions,), name="conditions_input")
    
    # === BILSTM Z-GENERATOR ===
    # As specified in REFERENCE.md: Dense(576) → Reshape(18,32) → Bidirectional(LSTM(64)) → Conv1D(32 filters)
    
    # Dense layer to expand noise
    dense_expand = Dense(576, activation='relu', name="z_gen_dense")(noise_input)
    
    # Reshape to sequence format (18, 32)
    reshaped = Reshape((latent_seq_len, latent_dim), name="z_gen_reshape")(dense_expand)
    
    # Bidirectional LSTM (64 units each direction = 128 total output)
    bilstm_output = Bidirectional(
        LSTM(64, return_sequences=True, name="z_gen_lstm"),
        name="z_gen_bidirectional"
    )(reshaped)
    
    # Conv1D layer (32 filters to match latent_dim)
    latent_sequences = Conv1D(
        filters=latent_dim,
        kernel_size=3,
        padding='same',
        activation='tanh',
        name="z_gen_conv1d"
    )(bilstm_output)
    
    logger.info(f"BiLSTM Z-generator output shape: (batch_size, {latent_seq_len}, {latent_dim})")
    
    # === LOAD PRE-TRAINED VAE DECODER ===
    vae_decoder_path = self.params.get("generator_sequential_model_file")
    if not vae_decoder_path:
        raise ValueError("generator_sequential_model_file not specified for VAE decoder")
    
    logger.info(f"Loading pre-trained VAE decoder from: {vae_decoder_path}")
    try:
        vae_decoder = load_model(vae_decoder_path)
        # Set to trainable=True for joint optimization as specified in REFERENCE.md
        vae_decoder.trainable = True
        logger.info(f"VAE decoder loaded successfully with {vae_decoder.count_params():,} parameters")
        logger.info("VAE decoder set to trainable=True for joint optimization")
    except Exception as e:
        logger.error(f"Failed to load VAE decoder: {e}")
        raise
    
    # === ITERATIVE SEQUENCE GENERATION ===
    # The composite generator needs to produce sequences of length seq_len
    # Each timestep calls the VAE decoder with:
    # - decoder_input_z_seq: latent sequences from BiLSTM Z-generator
    # - decoder_input_h_context: context from previous timestep
    # - decoder_input_conditions: current timestep conditions
    
    def iterative_generation(inputs):
        """
        Iterative generation function that processes timesteps sequentially.
        
        Args:
            inputs: [latent_sequences, context_input, conditions_input]
        
        Returns:
            Generated sequence of shape (batch_size, seq_len, 23)
        """
        latent_seq, initial_context, current_conditions = inputs
        
        # Initialize output list to collect timestep outputs
        timestep_outputs = []
        
        # Current context starts with the initial context
        current_context = initial_context
        
        # Generate each timestep in the sequence
        for t in range(seq_len):
            # Call VAE decoder with current inputs
            # Based on REFERENCE.md, decoder expects: [z_latent_seq, context_input, conditions_input]
            decoder_output = vae_decoder([latent_seq, current_context, current_conditions])
            
            # decoder_output shape: (batch_size, 23) - the 23 base features
            timestep_outputs.append(decoder_output)
            
            # Update context for next timestep using selected features from current output
            # Use a subset of the 23 features as context for next timestep
            current_context = tf.slice(decoder_output, [0, 0], [-1, context_dim])
            
            # For the next timestep, conditions could be updated if needed
            # For now, we use the same conditions for all timesteps
        
        # Stack all timestep outputs into a sequence
        # Shape: (batch_size, seq_len, 23)
        output_sequence = tf.stack(timestep_outputs, axis=1)
        
        return output_sequence
    
    # Apply iterative generation
    base_sequence = Lambda(
        iterative_generation,
        output_shape=(seq_len, 23),
        name="iterative_vae_generation"
    )([latent_sequences, context_input, conditions_input])
    
    logger.info(f"Iterative generation output shape: (batch_size, {seq_len}, 23)")
    
    # === 23-FEATURE ARCHITECTURE ===
    # Use only the 23 base features generated by the VAE decoder
    # Technical indicators and datetime features will be calculated as post-processing
    
    logger.info(f"Final output shape: (batch_size, {seq_len}, 23)")
    
    # === CREATE COMPOSITE MODEL ===
    composite_generator = Model(
        inputs=[noise_input, context_input, conditions_input],
        outputs=base_sequence,
        name="composite_gan_generator"
    )
    
    logger.info(f"Composite GAN Generator created with {composite_generator.count_params():,} total parameters")
    logger.info(f"VAE decoder parameters (trainable): {sum([np.prod(v.shape) for v in vae_decoder.trainable_variables]):,}")
    
    return composite_generator


def _prepare_generator_inputs(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare inputs for the composite generator.
    
    Args:
        batch_size: Number of samples in the batch
    
    Returns:
        Tuple of (noise, context, conditions) arrays
    """
    noise_dim = 100
    context_dim = self.params.get("context_vector_dim", 64)
    num_conditions = 10
    
    # Generate random noise
    noise = np.random.normal(0, 1, (batch_size, noise_dim))
    
    # Generate random initial context
    context = np.random.normal(0, 1, (batch_size, context_dim))
    
    # Generate random conditions (in practice, these would be real date/time features)
    conditions = np.random.uniform(-1, 1, (batch_size, num_conditions))
    
    return noise, context, conditions
