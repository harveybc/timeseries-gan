#!/usr/bin/env python3
"""
Generator Plugin - Main Interface

Main plugin interface that orchestrates specialized modules for synthetic data generation.
Maintains mandatory plugin structure while delegating to focused modules.
"""

import sys
import traceback
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Conv1D, Dense, Reshape, Concatenate, ZeroPadding1D, TimeDistributed # Ensure TimeDistributed is imported
from tensorflow.keras.models import Model
from typing import Dict, Any, List, Optional
import logging

from .model_loader import ModelLoader
from .normalization_handler import NormalizationHandler
from .feature_validator import FeatureValidator
from .initial_data_handler import InitialDataHandler
from .data_generator import DataGenerator
from .technical_indicator_calculator import TechnicalIndicatorCalculator
from .sequence_builder import SequenceBuilder
from tensorflow import keras
keras.config.enable_unsafe_deserialization()


class GeneratorPlugin:
    """
    Main generator plugin interface following extreme separation of concerns.
    Orchestrates specialized modules for focused functionality.
    """
    
    plugin_params = {
        "sequential_model_file": None, # This will be the VAE DECODER path
        "decoder_input_window_size": 144, # Output sequence length from VAE Decoder
        "full_feature_names_ordered": [],
        "decoder_output_feature_names": [],
        "ohlc_feature_names": ["OPEN", "HIGH", "LOW", "CLOSE"],
        "ti_feature_names": [],
        "date_conditional_feature_names": [],
        "feeder_conditional_feature_names": [],
        "ti_calculation_min_lookback": 200,
        "ti_params": {},
        # VAE Decoder Input Names (matching the pre-trained model)
        "decoder_input_name_latent": "decoder_input_z_seq", 
        "decoder_input_name_window": "input_x_window", # This VAE decoder input might not be used by our new composite model directly if we generate z_seq
        "decoder_input_name_conditions": "decoder_input_conditions",
        "decoder_input_name_context": "decoder_input_h_context",
        "generator_normalization_params_file": None,
        # New params for our internal Z-generator
        "internal_z_sequence_length": 18, # As per your spec (batch_size, 18, 32)
        "internal_z_latent_dim": 32,    # As per your spec
        "feeder_noise_dim": 100, # Example: dimension of noise from FeederPlugin
        "context_vector_dim": 64, # For decoder_input_h_context
        "conditional_features_dim": 10 # For decoder_input_conditions
    }
    
    plugin_debug_vars = [
        "sequential_model_file", "decoder_input_window_size", "batch_size_inference",
        "full_feature_names_ordered", "decoder_output_feature_names",
        "ti_calculation_min_lookback"
    ]
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the GeneratorPlugin with modular architecture.
        
        Args:
            config: Configuration dictionary
        """
        if config is None:
            raise ValueError("Configuration dictionary ('config') is required.")
        
        # Initialize parameters and main config
        self.params = self.plugin_params.copy()
        self.main_config = config.copy()
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize core attributes
        self.sequential_model: Optional[Model] = None
        self.model: Optional[Model] = None  # Alias for sequential_model
        self.feature_to_idx: Dict[str, int] = {}
        self.num_all_features: int = 0
        
        # Determine initial close file path from config
        initial_close_file_path = config.get("x_train_file", config.get("real_data_file"))
        
        # Initialize specialized modules
        self._initialize_modules()
        
        # Set parameters and validate
        self.set_params(**config)
        
        # Validate configuration
        self._validate_plugin_configuration()
        
        # Load initial close anchor
        if self.initial_data_handler.get_initial_close_anchor() is None:
            self.initial_data_handler.load_initial_close_anchor(initial_close_file_path)
    
    def _initialize_modules(self) -> None:
        """Initialize all specialized modules."""
        # Normalization handler (needs to be first)
        self.normalization_handler = NormalizationHandler(self.params, self.logger)
        
        # Model loader
        self.model_loader = ModelLoader(self.params, self.logger)
        
        # Initial data handler
        self.initial_data_handler = InitialDataHandler(self.normalization_handler)
        
        # Other modules will be initialized after feature validation
        self.feature_validator = None
        self.data_generator = None
        self.ti_calculator = None
        self.sequence_builder = None
    
    def _initialize_feature_dependent_modules(self) -> None:
        """Initialize modules that depend on feature configuration."""
        if not self.params.get("full_feature_names_ordered"):
            return
        
        # Feature validator
        self.feature_validator = FeatureValidator(self.params["full_feature_names_ordered"])
        
        # Create feature mapping
        self.feature_to_idx = self.feature_validator.create_feature_index_mapping()
        self.num_all_features = self.feature_validator.get_num_features()
        
        # Technical indicator calculator
        self.ti_calculator = TechnicalIndicatorCalculator(
            self.params["ti_feature_names"], 
            self.params["ti_params"]
        )
        
        # Data generator
        self.data_generator = DataGenerator(
            self.params, self.feature_to_idx, 
            self.normalization_handler, self.ti_calculator
        )
        self.data_generator.set_main_config(self.main_config)
        
        # Sequence builder
        self.sequence_builder = SequenceBuilder(
            self.params, self.feature_to_idx, self.num_all_features,
            self.normalization_handler, self.ti_calculator
        )
    
    def set_params(self, **kwargs) -> None:
        """
        Update plugin parameters and reload components as needed.
        
        Args:
            **kwargs: Parameter updates
        """
        print(f"GeneratorPlugin.set_params called with kwargs: {list(kwargs.keys())}")
        
        # Store old values for change detection
        old_model_file = self.params.get("sequential_model_file")
        old_norm_file = self.params.get("generator_normalization_params_file")
        old_full_feature_names = self.params.get("full_feature_names_ordered")
        old_initial_close_file_path = self.main_config.get("x_train_file", self.main_config.get("real_data_file"))
        
        # Update main config
        if hasattr(self, 'main_config') and self.main_config is not None:
            self.main_config.update(kwargs)
        else:
            self.main_config = kwargs.copy()
        
        # Update plugin parameters (handle both prefixed and non-prefixed)
        for param_key in self.plugin_params.keys():
            prefixed_key = f"generator_{param_key}"
            
            if prefixed_key in kwargs:
                self.params[param_key] = kwargs[prefixed_key]
            elif param_key in kwargs:
                self.params[param_key] = kwargs[param_key]
        
        # Handle special normalization parameter
        if "generator_normalization_params_file" in kwargs:
            self.params["generator_normalization_params_file"] = kwargs["generator_normalization_params_file"]
        
        # Check for model changes
        new_model_file = self.params.get("sequential_model_file")
        if new_model_file != old_model_file or (new_model_file and self.sequential_model is None):
            self._load_model(new_model_file)
        elif not new_model_file and old_model_file:
            print("GeneratorPlugin: Model path cleared. Cleaning loaded model.")
            self.sequential_model = None
            self.model = None
        
        # Check for normalization parameter changes
        new_norm_file = self.params.get("generator_normalization_params_file")
        if new_norm_file != old_norm_file:
            if new_norm_file:
                self.normalization_handler.load_normalization_params(new_norm_file)
            else:
                self.normalization_handler.normalization_params = None
        
        # Check for feature configuration changes
        if (self.params.get("full_feature_names_ordered") != old_full_feature_names or
            self._features_config_changed(kwargs)):
            if self.params.get("full_feature_names_ordered"):
                self._initialize_feature_dependent_modules()
                if self.feature_validator:
                    self.feature_validator.validate_feature_name_consistency(self.params)
        
        # Handle initial close anchor reload if the relevant file path changed
        new_initial_close_file_path = self.main_config.get("x_train_file", self.main_config.get("real_data_file"))
        if new_initial_close_file_path != old_initial_close_file_path and new_initial_close_file_path:
            self.logger.info(f"Initial close anchor file path changed. Reloading from: {new_initial_close_file_path}")
            self.initial_data_handler.load_initial_close_anchor(new_initial_close_file_path)
    
    def _features_config_changed(self, kwargs: Dict[str, Any]) -> bool:
        """Check if any feature-related configuration parameters have changed."""
        feature_keys = [
            "full_feature_names_ordered", "decoder_output_feature_names",
            "ohlc_feature_names", "ti_feature_names",
            "date_conditional_feature_names", "feeder_conditional_feature_names"
        ]
        for key in feature_keys:
            if self.params.get(key) != self.plugin_params.get(key): # Compare with initial defaults or previous state
                if key in kwargs or f"generator_{key}" in kwargs: # Check if it was in the update
                    return True
        return False

    def _validate_plugin_configuration(self) -> None:
        """Validate essential plugin configurations."""
        if not self.params.get("full_feature_names_ordered"):
            self.logger.warning("GeneratorPlugin: 'full_feature_names_ordered' is not configured.")
        if not self.params.get("decoder_output_feature_names"):
            self.logger.warning("GeneratorPlugin: 'decoder_output_feature_names' is not configured.")

    def _load_model(self, vae_decoder_model_path: str) -> None:
        """
        Load the pre-trained VAE decoder and build the composite generator model.
        
        Based on REFERENCE.md:
        - Load pre-trained VAE decoder from examples/results/phase_4_3/phase_4_3_cnn_small_decoder_model.keras
        - Build BiLSTM Z-generator: Dense(576) → Reshape(18,32) → Bidirectional(LSTM(64)) → Conv1D(32 filters)
        - Combine into composite generator that outputs 57 features
        
        Args:
            vae_decoder_model_path: Path to the pre-trained VAE decoder model
        """
        if not vae_decoder_model_path:
            self.sequential_model = None
            self.model = None
            self.logger.warning("GeneratorPlugin: Attempted to load VAE decoder with empty path.")
            return

        self.logger.info(f"Building composite generator with VAE decoder from: {vae_decoder_model_path}")

        try:
            # Load the pre-trained VAE decoder
            loaded_vae_decoder = self.model_loader.load_model_from_path(vae_decoder_model_path)
            if loaded_vae_decoder is None:
                raise IOError(f"Failed to load VAE decoder from {vae_decoder_model_path}")
            
            # Set the loaded VAE decoder to be trainable for joint GAN optimization
            loaded_vae_decoder.trainable = True
            self.logger.info(f"Loaded VAE decoder '{loaded_vae_decoder.name}'. Set trainable=True.")
            self.logger.info(f"VAE decoder input shapes: {[inp.shape for inp in loaded_vae_decoder.inputs]}")
            self.logger.info(f"VAE decoder output shape: {loaded_vae_decoder.output.shape}")

            # Build the composite generator model
            self.sequential_model = self._build_composite_generator(loaded_vae_decoder)
            self.model = self.sequential_model  # Alias

            self.logger.info("Composite generator model built successfully.")
            self.logger.info(f"Composite generator input shapes: {[inp.shape for inp in self.sequential_model.inputs]}")
            self.logger.info(f"Composite generator output shape: {self.sequential_model.output.shape}")

        except Exception as e:
            self.logger.error(f"Error building composite generator model: {e}")
            self.sequential_model = None
            self.model = None
            raise IOError(f"Failed to build composite generator model: {e}")

    def _build_composite_generator(self, vae_decoder: Model) -> Model:
        """
        Build Composite GAN Generator according to REFERENCE.md specifications.
        
        Architecture:
        1. BiLSTM Z-generator: Dense(576) → Reshape(18,32) → Bidirectional(LSTM(64)) → Conv1D(32 filters)
        2. Pre-trained VAE Decoder: Loaded with trainable=True for joint optimization  
        3. Iterative Processing: Sequential generation with context from previous timesteps
        
        Inputs:
        - noise_input: (batch_size, noise_dim) - single noise vector per sequence
        - context_input: (batch_size, context_dim) - initial context vector
        - conditions_input: (batch_size, num_conditions) - conditional features per timestep
        
        Output:
        - full_sequence: (batch_size, seq_len, 57) - complete feature sequences
        """
        try:
            self.logger.info("Building Composite GAN Generator from REFERENCE.md specifications")
            
            # Get configuration parameters
            seq_len = self.params.get("decoder_input_window_size", 144)
            latent_seq_len = self.params.get("internal_z_sequence_length", 18)
            latent_dim = self.params.get("internal_z_latent_dim", 32)
            context_dim = self.params.get("context_vector_dim", 64)
            num_conditions = self.params.get("conditional_features_dim", 10)
            noise_dim = self.params.get("feeder_noise_dim", 100)
            
            self.logger.info(f"Generator config: seq_len={seq_len}, latent_shape=[{latent_seq_len}, {latent_dim}], "
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
            
            self.logger.info(f"BiLSTM Z-generator output shape: (batch_size, {latent_seq_len}, {latent_dim})")
            
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
                    if context_dim <= 23:
                        current_context = tf.slice(decoder_output, [0, 0], [-1, context_dim])
                    else:
                        # If context_dim > 23, pad with zeros
                        padding = tf.zeros((tf.shape(decoder_output)[0], context_dim - 23))
                        current_context = tf.concat([decoder_output, padding], axis=1)
                    
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
            
            self.logger.info(f"Iterative generation output shape: (batch_size, {seq_len}, 23)")
            
            # === POST-PROCESSING TO FULL 57 FEATURES ===
            # The VAE decoder outputs 23 base features
            # We need to expand this to 57 features by:
            # 1. Adding technical indicators (15 features)
            # 2. Adding cyclical date features (8 features)  
            # 3. Adding other derived features (11 features)
            
            def expand_to_full_features(base_features):
                """
                Expand 23 base features to full 57 features.
                
                Args:
                    base_features: Shape (batch_size, seq_len, 23)
                
                Returns:
                    Full feature tensor of shape (batch_size, seq_len, 57)
                """
                batch_size = tf.shape(base_features)[0]
                
                # For now, we'll create placeholder features for the additional 34 features
                # In a complete implementation, this would calculate actual technical indicators
                additional_features = tf.zeros((batch_size, seq_len, 34), dtype=tf.float32)
                
                # Concatenate base features with additional features
                full_features = tf.concat([base_features, additional_features], axis=-1)
                
                return full_features
            
            # Expand to full feature set
            full_sequence = Lambda(
                expand_to_full_features,
                output_shape=(seq_len, 57),
                name="expand_to_57_features"
            )(base_sequence)
            
            self.logger.info(f"Final output shape: (batch_size, {seq_len}, 57)")
            
            # === CREATE COMPOSITE MODEL ===
            composite_generator = Model(
                inputs=[noise_input, context_input, conditions_input],
                outputs=full_sequence,
                name="composite_gan_generator"
            )
            
            self.logger.info(f"Composite GAN Generator created with {composite_generator.count_params():,} total parameters")
            if hasattr(vae_decoder, 'trainable_variables'):
                self.logger.info(f"VAE decoder parameters (trainable): {sum([tf.size(v).numpy() for v in vae_decoder.trainable_variables]):,}")
            
            return composite_generator

        except Exception as e:
            self.logger.error(f"Error building composite generator: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None 

    def get_model(self) -> Optional[Model]:
        """Returns the built generator model."""
        return self.sequential_model # Or self.model, whichever holds the final composite generator
