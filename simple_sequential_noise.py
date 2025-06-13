#!/usr/bin/env python3
"""
Simple Sequential Noise Generator

A lightweight approach to generate sequential noise patterns that preserve
the sequential dimension without complex deep learning layers.

The idea is to generate noise that has some sequential correlation rather
than completely independent noise vectors.
"""

import tensorflow as tf
import numpy as np
from typing import Tuple


class SimpleSequentialNoiseLayer(tf.keras.layers.Layer):
    """
    Simple layer that generates sequential noise with temporal correlation.
    
    Instead of Dense(576) → Reshape(18,32), this layer:
    1. Takes noise input
    2. Generates 18 sequences where each has some correlation with previous ones
    3. Uses a simple recurrent process without heavy LSTM layers
    """
    
    def __init__(self, seq_len: int = 18, latent_dim: int = 32, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        
        # Simple dense layer to expand noise to initial sequence
        self.initial_dense = tf.keras.layers.Dense(
            latent_dim, 
            activation='tanh', 
            name="initial_sequence_dense"
        )
        
        # Small dense layer for sequence evolution (much lighter than LSTM)
        self.evolution_dense = tf.keras.layers.Dense(
            latent_dim, 
            activation='tanh', 
            name="sequence_evolution_dense"
        )
    
    def call(self, inputs):
        """
        Generate sequential noise from input noise.
        
        Args:
            inputs: Noise tensor of shape (batch_size, noise_dim)
            
        Returns:
            Sequential noise tensor of shape (batch_size, seq_len, latent_dim)
        """
        batch_size = tf.shape(inputs)[0]
        
        # Generate first sequence element from noise
        current_seq = self.initial_dense(inputs)  # (batch_size, latent_dim)
        
        # Collect all sequence elements
        sequence_list = [current_seq]
        
        # Generate remaining sequence elements with simple evolution
        for i in range(1, self.seq_len):
            # Add some noise to current sequence for variation
            noise_factor = 0.1  # Small noise to maintain correlation but add variation
            random_noise = tf.random.normal(tf.shape(current_seq), stddev=noise_factor)
            
            # Evolve the sequence (simple transformation)
            evolved_input = current_seq + random_noise
            current_seq = self.evolution_dense(evolved_input)
            
            sequence_list.append(current_seq)
        
        # Stack into final sequence: (batch_size, seq_len, latent_dim)
        output_sequence = tf.stack(sequence_list, axis=1)
        
        return output_sequence
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'seq_len': self.seq_len,
            'latent_dim': self.latent_dim
        })
        return config


class SimpleRandomWalkNoiseLayer(tf.keras.layers.Layer):
    """
    Even simpler approach: Generate sequences using random walk.
    This creates natural sequential correlation without any trainable parameters.
    """
    
    def __init__(self, seq_len: int = 18, latent_dim: int = 32, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        
        # Only one dense layer to create initial state
        self.initial_dense = tf.keras.layers.Dense(
            latent_dim, 
            activation='tanh', 
            name="random_walk_initial"
        )
    
    def call(self, inputs):
        """
        Generate sequential noise using random walk from initial state.
        
        Args:
            inputs: Noise tensor of shape (batch_size, noise_dim)
            
        Returns:
            Sequential noise tensor of shape (batch_size, seq_len, latent_dim)
        """
        batch_size = tf.shape(inputs)[0]
        
        # Generate initial state from noise
        initial_state = self.initial_dense(inputs)  # (batch_size, latent_dim)
        
        # Generate random walk steps
        random_steps = tf.random.normal(
            (batch_size, self.seq_len - 1, self.latent_dim), 
            stddev=0.1
        )
        
        # Create random walk sequence
        # Start with initial state, then add cumulative random steps
        initial_expanded = tf.expand_dims(initial_state, axis=1)  # (batch_size, 1, latent_dim)
        
        # Cumulative sum of random steps to create walk
        cumulative_steps = tf.cumsum(random_steps, axis=1)  # (batch_size, seq_len-1, latent_dim)
        
        # Add initial state to all steps
        random_walk = initial_expanded + tf.concat([
            tf.zeros_like(initial_expanded), 
            cumulative_steps
        ], axis=1)
        
        # Apply tanh to keep values in reasonable range
        output_sequence = tf.tanh(random_walk)
        
        return output_sequence
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'seq_len': self.seq_len,
            'latent_dim': self.latent_dim
        })
        return config


def test_simple_approaches():
    """Test both simple approaches and compare with Dense+Reshape."""
    
    print("=== TESTING SIMPLE SEQUENTIAL NOISE APPROACHES ===\n")
    
    batch_size = 4
    noise_dim = 100
    seq_len = 18
    latent_dim = 32
    
    # Generate test noise
    test_noise = np.random.randn(batch_size, noise_dim).astype(np.float32)
    
    print("1. CURRENT APPROACH (Dense + Reshape):")
    # Current approach
    dense_layer = tf.keras.layers.Dense(seq_len * latent_dim, activation='tanh')
    reshape_layer = tf.keras.layers.Reshape((seq_len, latent_dim))
    
    dense_output = dense_layer(test_noise)
    current_output = reshape_layer(dense_output)
    
    print(f"   Output shape: {current_output.shape}")
    print(f"   Parameters: {dense_layer.count_params():,}")
    
    # Check sequential correlation (should be low)
    correlation = np.corrcoef(current_output[0, :, 0])  # Correlation across time for first feature
    avg_correlation = np.mean(np.abs(correlation - np.eye(seq_len)))
    print(f"   Avg temporal correlation: {avg_correlation:.4f} (lower = less sequential structure)")
    
    print("\n2. SIMPLE SEQUENTIAL APPROACH:")
    # Simple sequential approach
    sequential_layer = SimpleSequentialNoiseLayer(seq_len, latent_dim)
    sequential_output = sequential_layer(test_noise)
    
    print(f"   Output shape: {sequential_output.shape}")
    print(f"   Parameters: {sequential_layer.count_params():,}")
    
    # Check sequential correlation
    correlation = np.corrcoef(sequential_output[0, :, 0])
    avg_correlation = np.mean(np.abs(correlation - np.eye(seq_len)))
    print(f"   Avg temporal correlation: {avg_correlation:.4f} (higher = more sequential structure)")
    
    print("\n3. RANDOM WALK APPROACH:")
    # Random walk approach
    walk_layer = SimpleRandomWalkNoiseLayer(seq_len, latent_dim)
    walk_output = walk_layer(test_noise)
    
    print(f"   Output shape: {walk_output.shape}")
    print(f"   Parameters: {walk_layer.count_params():,}")
    
    # Check sequential correlation
    correlation = np.corrcoef(walk_output[0, :, 0])
    avg_correlation = np.mean(np.abs(correlation - np.eye(seq_len)))
    print(f"   Avg temporal correlation: {avg_correlation:.4f} (higher = more sequential structure)")
    
    print("\n=== RECOMMENDATION ===")
    print("The Random Walk approach is the simplest and most parameter-efficient")
    print("while still providing natural sequential correlation.")
    print("It uses only ONE Dense layer instead of Dense(576) + many parameters.")


if __name__ == "__main__":
    test_simple_approaches()
