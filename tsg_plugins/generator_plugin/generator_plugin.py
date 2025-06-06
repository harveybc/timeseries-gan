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
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Conv1D, Dense, Reshape, Concatenate, ZeroPadding1D, TimeDistributed, Lambda
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
        
        # Set parameters (simplified for initial testing)
        for key, value in config.items():
            if key in self.plugin_params:
                self.params[key] = value
        
        # Validate configuration
        self._validate_plugin_configuration()
        
        # Load initial close anchor (simplified)
        if initial_close_file_path and self.initial_data_handler.get_initial_close_anchor() is None:
            try:
                self.initial_data_handler.load_initial_close_anchor(initial_close_file_path)
            except Exception as e:
                self.logger.warning(f"Failed to load initial close anchor: {e}")
    
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
        if new_model_file and new_model_file != old_model_file:
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

    def get_model(self) -> Optional[Model]:
        """
        Get the composite generator model.
        
        Returns:
            Optional[Model]: The composite generator model if available, None otherwise
        """
        # First check if model was loaded via _load_model()
        if hasattr(self, 'sequential_model') and self.sequential_model is not None:
            return self.sequential_model
            
        # Check if we have a composite model built separately
        if hasattr(self, 'composite_model') and self.composite_model is not None:
            return self.composite_model
            
        # Try to build a simple generator model for testing/fallback
        self.logger.warning("No generator model available. Building fallback generator...")
        try:
            model = self._build_composite_generator()
            return model
        except Exception as e:
            self.logger.error(f"Failed to build fallback generator: {e}")
            return None

    def _build_composite_generator(self, vae_decoder_model=None) -> Optional[Model]:
        """
        Build the composite generator model combining BiLSTM Z-generator + VAE decoder.
        Based on REFERENCE.md Sequential Conditional VAE-GAN Architecture.
        
        The generator must output sequences of shape (batch_size, 144, 57) to match discriminator input.
        
        Args:
            vae_decoder_model: Optional pre-trained VAE decoder model to integrate
            
        Returns:
            Model: The built composite generator model
        """
        try:
            self.logger.info("Building composite generator model...")
            
            # Get configuration parameters
            seq_len = self.params.get("decoder_input_window_size", 144)
            noise_dim = self.params.get("feeder_noise_dim", 32)
            conditional_features_dim = self.params.get("conditional_features_dim", 10)
            context_vector_dim = self.params.get("context_vector_dim", 64)
            
            self.logger.info(f"Building generator with seq_len={seq_len}, noise_dim={noise_dim}")
            
            # Build generator inputs - per REFERENCE.md, VAE decoder only needs 2 inputs
            noise_input = Input(shape=(noise_dim,), name="noise_input")
            conditions_input = Input(shape=(conditional_features_dim,), name="conditions_input")
            
            if vae_decoder_model is not None:
                # Use pre-trained VAE decoder - implement BiLSTM Z-generator as per REFERENCE.md
                self.logger.info("Building composite model with pre-trained VAE decoder")
                self.logger.info(f"VAE decoder expects {len(vae_decoder_model.inputs)} inputs")
                
                # BiLSTM Z-generator architecture: Dense(576) → Reshape(18,32) → Bidirectional(LSTM(64)) → Conv1D(32)
                z_dense = Dense(576, activation='relu', name="z_dense")(noise_input)
                z_reshape = Reshape((18, 32), name="z_reshape")(z_dense)
                z_bilstm = Bidirectional(LSTM(64, return_sequences=True), name="z_bilstm")(z_reshape)
                z_latent_seq = Conv1D(32, kernel_size=3, padding='same', activation='relu', name="z_conv")(z_bilstm)
                
                # Generate full sequence by iteratively calling VAE decoder
                def generate_sequence(inputs):
                    """Generate full sequence using VAE decoder iteratively."""
                    z_seq, conditions = inputs
                    batch_size = tf.shape(z_seq)[0]
                    
                    # Initialize output sequence
                    output_seq = []
                    
                    # Generate each timestep
                    for t in range(seq_len):
                        # Use latent sequence as input to VAE decoder
                        # VAE decoder expects: [z_seq, conditions] as per REFERENCE.md
                        vae_out = vae_decoder_model([z_seq, conditions])  # Shape: (batch, 23)
                        
                        # Expand 23 features to 57 features using a dense layer
                        expanded_features = tf.keras.layers.Dense(57, activation='linear', name=f"expand_{t}")(vae_out)
                        output_seq.append(expanded_features)
                    
                    # Stack timesteps to create sequence
                    return tf.stack(output_seq, axis=1)  # Shape: (batch, seq_len, 57)
                
                # Apply sequence generation with explicit output shape
                expanded_output = tf.keras.layers.Lambda(
                    generate_sequence,
                    output_shape=(seq_len, 57),
                    name="sequence_generator"
                )([z_latent_seq, conditions_input])
                    
            else:
                # Build simple generator from scratch for testing
                self.logger.info("Building simple generator from scratch")
                
                # Combine inputs (only noise and conditions)
                combined_inputs = Concatenate(name="combined_inputs")([noise_input, conditions_input])
                
                # Generate sequence directly
                hidden1 = Dense(256, activation='relu', name="hidden1")(combined_inputs)
                hidden2 = Dense(512, activation='relu', name="hidden2")(hidden1)
                hidden3 = Dense(1024, activation='relu', name="hidden3")(hidden2)
                
                # Output full sequence: seq_len * 57 features
                sequence_flat = Dense(seq_len * 57, activation='tanh', name="sequence_flat")(hidden3)
                
                # Reshape to sequence format: (batch_size, seq_len, 57)
                expanded_output = Reshape((seq_len, 57), name="output_reshape")(sequence_flat)
            
            # Create composite model with only 2 inputs as per REFERENCE.md
            composite_model = Model(
                inputs=[noise_input, conditions_input],
                outputs=expanded_output,
                name="composite_generator"
            )
            
            self.logger.info(f"Composite generator built with {composite_model.count_params()} parameters")
            self.logger.info(f"Generator output shape: {composite_model.output.shape}")
            
            # Store the model
            self.composite_model = composite_model
            
            return composite_model
            
        except Exception as e:
            self.logger.error(f"Error building composite generator: {e}")
            self.logger.error(traceback.format_exc())
            self.composite_model = None
            return None

    def build_model(self) -> None:
        """Public interface for building the composite generator model."""
        self._build_composite_generator()
