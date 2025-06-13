#!/usr/bin/env python3
"""
Improved Sequential Latent Generator

This module provides an improved approach for generating latent sequences that better
match the sequential structure learned by the VAE encoder, rather than using a simple
Dense+Reshape transformation.

The key insight is that the VAE encoder learned to compress SEQUENTIAL patterns from
real time series data. Therefore, the latent generator should also produce SEQUENTIAL 
patterns that respect temporal dependencies.
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Bidirectional, Conv1D, Dropout, 
    LayerNormalization, MultiHeadAttention, Add, RepeatVector
)
from tensorflow.keras.models import Model
import numpy as np
from typing import Tuple, Optional


class SequentialLatentGenerator:
    """
    Improved latent sequence generator that creates temporally coherent latent vectors
    by modeling the sequential process that the VAE encoder learned to reverse.
    """
    
    def __init__(self, 
                 noise_dim: int = 100,
                 latent_seq_len: int = 18,
                 latent_dim: int = 32,
                 context_dim: int = 64,
                 num_conditions: int = 10):
        """
        Initialize the sequential latent generator.
        
        Args:
            noise_dim: Dimension of input noise vector
            latent_seq_len: Length of output latent sequence (18)
            latent_dim: Dimension of each latent vector (32)
            context_dim: Dimension of context vector (64)
            num_conditions: Number of conditional features (10)
        """
        self.noise_dim = noise_dim
        self.latent_seq_len = latent_seq_len
        self.latent_dim = latent_dim
        self.context_dim = context_dim
        self.num_conditions = num_conditions
    
    def build_sequential_latent_generator(self) -> Model:
        """
        Build an improved latent generator that creates sequential patterns.
        
        This approach models the INVERSE of what the VAE encoder does:
        - VAE encoder: Sequential data → Compressed latent sequences
        - This generator: Noise + conditions → Sequential latent patterns
        
        Key improvements over Dense+Reshape:
        1. Uses a stack of LSTM layers to generate true sequential patterns
        2. Incorporates conditioning information throughout the sequence
        3. Uses attention mechanisms to model long-range dependencies
        4. Applies progressive refinement through multiple processing stages
        
        Returns:
            Model that generates latent sequences with proper temporal structure
        """
        
        # === INPUTS ===
        noise_input = Input(shape=(self.noise_dim,), name="noise_input")
        context_input = Input(shape=(self.context_dim,), name="context_input")
        conditions_input = Input(shape=(self.num_conditions,), name="conditions_input")
        
        # === STAGE 1: INITIAL SEQUENCE SEED GENERATION ===
        # Create an initial "seed" sequence using noise and conditions
        # This replaces the naive Dense(576) approach
        
        # Combine noise with context and conditions for richer initial state
        combined_input = tf.keras.layers.Concatenate(name="input_combination")([
            noise_input, context_input, conditions_input
        ])
        
        # Create multiple "seed vectors" that will become the initial sequence
        # Instead of one big dense layer, use multiple smaller ones for different aspects
        seed_vectors = []
        for i in range(self.latent_seq_len):
            # Each timestep gets its own specialized seed
            seed = Dense(self.latent_dim, 
                        activation='tanh', 
                        name=f"seed_dense_{i}")(combined_input)
            seed_vectors.append(seed)
        
        # Stack into initial sequence: (batch, latent_seq_len, latent_dim)
        initial_sequence = tf.keras.layers.Lambda(
            lambda x: tf.stack(x, axis=1), 
            name="initial_sequence"
        )(seed_vectors)
        
        # === STAGE 2: SEQUENTIAL REFINEMENT ===
        # Process the initial sequence through LSTM layers to create temporal dependencies
        
        # First LSTM pass: Forward direction
        forward_lstm = LSTM(64, return_sequences=True, name="forward_lstm")
        forward_output = forward_lstm(initial_sequence)
        
        # Second LSTM pass: Backward direction  
        backward_lstm = LSTM(64, return_sequences=True, go_backwards=True, name="backward_lstm")
        backward_output = backward_lstm(initial_sequence)
        backward_output = tf.keras.layers.Lambda(
            lambda x: tf.reverse(x, axis=[1]), 
            name="reverse_backward"
        )(backward_output)  # Reverse back to forward order
        
        # Combine bidirectional information
        bidirectional_output = tf.keras.layers.Concatenate(axis=-1, name="bidirectional_combine")([
            forward_output, backward_output
        ])  # Shape: (batch, 18, 128)
        
        # === STAGE 3: CONDITIONING INTEGRATION ===
        # Integrate conditioning information at each timestep
        
        # Repeat conditions and context for each timestep
        repeated_conditions = RepeatVector(self.latent_seq_len, name="repeated_conditions")(conditions_input)
        repeated_context = RepeatVector(self.latent_seq_len, name="repeated_context")(context_input)
        
        # Combine with bidirectional LSTM output
        conditioned_sequence = tf.keras.layers.Concatenate(axis=-1, name="condition_integration")([
            bidirectional_output,    # (batch, 18, 128)
            repeated_conditions,     # (batch, 18, 10)  
            repeated_context        # (batch, 18, 64)
        ])  # Shape: (batch, 18, 202)
        
        # === STAGE 4: ATTENTION-BASED REFINEMENT ===
        # Use attention to model long-range dependencies (similar to VAE decoder)
        
        # Project to attention-friendly dimension
        attention_proj = Dense(64, activation='relu', name="attention_projection")(conditioned_sequence)
        
        # Multi-head self-attention (similar to VAE decoder's attention mechanism)
        attention_output = MultiHeadAttention(
            num_heads=4,
            key_dim=16,
            name="sequential_attention"
        )(attention_proj, attention_proj)
        
        # Residual connection
        attended_sequence = Add(name="attention_residual")([attention_proj, attention_output])
        
        # Layer normalization
        normalized_sequence = LayerNormalization(name="attention_norm")(attended_sequence)
        
        # === STAGE 5: FINAL PROJECTION ===
        # Project to final latent dimensions
        
        final_latent_sequence = Conv1D(
            filters=self.latent_dim,
            kernel_size=3,
            padding='same',
            activation='tanh',
            name="final_projection"
        )(normalized_sequence)
        
        # === BUILD MODEL ===
        model = Model(
            inputs=[noise_input, context_input, conditions_input],
            outputs=final_latent_sequence,
            name="sequential_latent_generator"
        )
        
        return model
    
    def build_hierarchical_latent_generator(self) -> Model:
        """
        Alternative approach: Hierarchical latent generation.
        
        This approach generates latent sequences at multiple temporal resolutions,
        similar to how the VAE encoder might have learned hierarchical patterns.
        
        Returns:
            Model that generates latent sequences with hierarchical temporal structure
        """
        
        # === INPUTS ===
        noise_input = Input(shape=(self.noise_dim,), name="noise_input")
        context_input = Input(shape=(self.context_dim,), name="context_input") 
        conditions_input = Input(shape=(self.num_conditions,), name="conditions_input")
        
        # === HIERARCHICAL GENERATION ===
        
        # Level 1: Generate coarse temporal structure (fewer timesteps)
        combined_input = tf.keras.layers.Concatenate()([noise_input, context_input, conditions_input])
        
        # Generate 6 coarse timesteps first (18/3 = 6)
        coarse_length = self.latent_seq_len // 3
        coarse_dense = Dense(coarse_length * self.latent_dim, activation='relu')(combined_input)
        coarse_sequence = tf.keras.layers.Reshape((coarse_length, self.latent_dim))(coarse_dense)
        
        # Refine coarse sequence with LSTM
        coarse_refined = Bidirectional(LSTM(32, return_sequences=True))(coarse_sequence)
        coarse_projected = Dense(self.latent_dim, activation='tanh')(coarse_refined)
        
        # Level 2: Upsample to medium resolution (12 timesteps)
        medium_length = (self.latent_seq_len * 2) // 3
        medium_upsampled = tf.keras.layers.UpSampling1D(size=2)(coarse_projected)  # 6 -> 12
        
        # Add medium-scale variations
        medium_noise = Dense(medium_length * 16, activation='relu')(combined_input)
        medium_noise = tf.keras.layers.Reshape((medium_length, 16))(medium_noise)
        medium_variations = Dense(self.latent_dim, activation='tanh')(medium_noise)
        
        medium_combined = Add()([medium_upsampled, medium_variations])
        medium_refined = LSTM(64, return_sequences=True)(medium_combined)
        medium_projected = Dense(self.latent_dim, activation='tanh')(medium_refined)
        
        # Level 3: Final upsampling to full resolution (18 timesteps)
        final_upsampled = tf.keras.layers.UpSampling1D(size=1.5)(medium_projected)  # 12 -> 18
        
        # Add fine-scale details
        fine_noise = Dense(self.latent_seq_len * 8, activation='relu')(combined_input)
        fine_noise = tf.keras.layers.Reshape((self.latent_seq_len, 8))(fine_noise)
        fine_details = Dense(self.latent_dim, activation='tanh')(fine_noise)
        
        final_sequence = Add()([final_upsampled, fine_details])
        
        # Final refinement
        output_sequence = Conv1D(
            filters=self.latent_dim,
            kernel_size=3,
            padding='same',
            activation='tanh'
        )(final_sequence)
        
        # === BUILD MODEL ===
        model = Model(
            inputs=[noise_input, context_input, conditions_input],
            outputs=output_sequence,
            name="hierarchical_latent_generator"
        )
        
        return model


def compare_approaches():
    """
    Compare the different latent generation approaches.
    """
    print("=== LATENT SEQUENCE GENERATION APPROACHES ===\n")
    
    print("1. CURRENT APPROACH (Dense+Reshape):")
    print("   ❌ Dense(576) → Reshape(18,32) → BiLSTM → Conv1D")
    print("   ❌ Problems:")
    print("      - Dense layer creates static, non-sequential pattern")
    print("      - No temporal dependencies in initial latent vectors")
    print("      - BiLSTM tries to add sequence info to already-formed vectors")
    print("      - Doesn't match how VAE encoder learned to compress sequences\n")
    
    print("2. RECOMMENDED: SEQUENTIAL APPROACH:")
    print("   ✅ Multi-stage sequential generation")
    print("   ✅ Benefits:")
    print("      - Creates true sequential patterns from the start")
    print("      - Each timestep has specialized initialization")
    print("      - Bidirectional LSTM creates temporal dependencies")
    print("      - Attention mechanism models long-range dependencies")
    print("      - Conditioning integrated throughout the sequence")
    print("      - Mimics the INVERSE of VAE encoder's compression process\n")
    
    print("3. ALTERNATIVE: HIERARCHICAL APPROACH:")
    print("   ✅ Multi-resolution temporal modeling")
    print("   ✅ Benefits:")
    print("      - Generates patterns at multiple time scales")
    print("      - Coarse → Medium → Fine temporal structure")
    print("      - Captures hierarchical temporal patterns")
    print("      - Similar to how CNNs work with spatial hierarchies\n")
    
    print("=== IMPLEMENTATION RECOMMENDATION ===")
    print("Start with the SEQUENTIAL APPROACH as it most directly addresses")
    print("the fundamental issue: creating latent sequences that have proper")
    print("temporal dependencies from the beginning, rather than trying to")
    print("add them after the fact with Dense+Reshape.")


if __name__ == "__main__":
    # Demonstrate the improved approaches
    generator = SequentialLatentGenerator()
    
    print("Building Sequential Latent Generator...")
    sequential_model = generator.build_sequential_latent_generator()
    print(f"Sequential model parameters: {sequential_model.count_params():,}")
    print("\nSequential Model Architecture:")
    sequential_model.summary()
    
    print("\n" + "="*60)
    compare_approaches()
    
    # Test with sample data
    print("\n=== TESTING OUTPUT SHAPES ===")
    batch_size = 4
    noise = np.random.randn(batch_size, 100)
    context = np.random.randn(batch_size, 64)
    conditions = np.random.randn(batch_size, 10)
    
    sequential_output = sequential_model.predict([noise, context, conditions], verbose=0)
    
    print(f"Sequential output shape: {sequential_output.shape}")
    print(f"Expected shape: ({batch_size}, 18, 32)")
    
    print(f"\nSequential output stats: min={sequential_output.min():.3f}, max={sequential_output.max():.3f}")
    
    # Verify sequential properties
    print("\n=== ANALYZING SEQUENTIAL PROPERTIES ===")
    
    # Test temporal correlation - neighboring timesteps should be more correlated than distant ones
    correlations = []
    for lag in range(1, 10):
        if sequential_output.shape[1] > lag:
            corr_sum = 0
            for batch_idx in range(batch_size):
                for dim in range(32):
                    series = sequential_output[batch_idx, :, dim]
                    lagged_series = np.roll(series, -lag)
                    corr = np.corrcoef(series[:-lag], lagged_series[:-lag])[0,1]
                    if not np.isnan(corr):
                        corr_sum += corr
            avg_corr = corr_sum / (batch_size * 32)
            correlations.append(avg_corr)
            print(f"Lag-{lag} correlation: {avg_corr:.4f}")
    
    print(f"\nTemporal structure analysis:")
    print(f"- Lag-1 correlation: {correlations[0]:.4f} (should be high for sequential data)")
    print(f"- Correlation decay: {correlations[0] - correlations[-1]:.4f} (should be positive)")
    
    if correlations[0] > 0.1 and correlations[0] > correlations[-1]:
        print("✅ Sequential structure detected: neighboring timesteps are correlated")
    else:
        print("❌ Weak sequential structure: may need architecture adjustments")
