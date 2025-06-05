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
        Build composite generator model combining BiLSTM Z-generator + VAE decoder.
        This version is modified to output sequences of length `decoder_input_window_size`.
        
        Inputs:
        - noise_input_sequence: (batch_size, seq_len, feeder_noise_dim)
        - conditions_input_sequence: (batch_size, seq_len, conditional_features_dim)
        - context_input_sequence: (batch_size, seq_len, context_vector_dim)
        
        Output:
        - expanded_feature_sequences: (batch_size, seq_len, 57)
        """
        try:
            self.logger.info("Building composite generator model (sequence output)...")
            
            seq_len = self.params["decoder_input_window_size"] # Should be 144

            # Input layers for the composite generator (now sequences)
            noise_input_sequence = Input(shape=(seq_len, self.params["feeder_noise_dim"],), name="noise_input_sequence")
            conditions_input_sequence = Input(shape=(seq_len, self.params["conditional_features_dim"],), name="conditions_input_sequence")
            context_input_sequence = Input(shape=(seq_len, self.params["context_vector_dim"],), name="context_input_sequence")

            # --- Define the BiLSTM Z-generator as a sub-model ---
            # This sub-model takes a single noise vector and produces a (18, 32) latent sequence
            single_noise_input_for_z = Input(shape=(self.params["feeder_noise_dim"],), name="single_noise_input_for_z")
            z_dense_layer = Dense(576, activation='relu', name="z_dense_sub")(single_noise_input_for_z)
            z_reshape_layer = Reshape((self.params["internal_z_sequence_length"], self.params["internal_z_latent_dim"]), name="z_reshape_sub")(z_dense_layer)
            z_bilstm_layer = Bidirectional(LSTM(64, return_sequences=True), name="z_bilstm_sub")(z_reshape_layer)
            z_latent_seq_output = Conv1D(self.params["internal_z_latent_dim"], kernel_size=3, padding='same', activation='relu', name="z_conv_sub")(z_bilstm_layer)
            
            z_generator_submodel = Model(inputs=single_noise_input_for_z, outputs=z_latent_seq_output, name="z_generator_submodel")
            
            # Apply the Z-generator sub-model to each step of the noise_input_sequence
            # Input: (batch, seq_len, feeder_noise_dim) -> Output: (batch, seq_len, 18, 32)
            all_z_latent_sequences = TimeDistributed(z_generator_submodel, name="td_z_generator")(noise_input_sequence)

            # Apply VAE decoder to each time step
            # VAE decoder inputs per step: z_latent_seq (18,32), context (64), conditions (10)
            # TimeDistributed inputs: 
            #   all_z_latent_sequences (batch, seq_len, 18, 32)
            #   context_input_sequence (batch, seq_len, 64)
            #   conditions_input_sequence (batch, seq_len, 10)
            # Output: (batch, seq_len, 23)
            self.logger.info(f"Passing inputs to TimeDistributed VAE decoder in order: all_z_latent_sequences, context_input_sequence, conditions_input_sequence")
            
            # Ensure vae_decoder is set to trainable if not already done (it is done in _load_model)
            # vae_decoder.trainable = True 
            
            decoder_output_sequences = TimeDistributed(vae_decoder, name="td_vae_decoder")([all_z_latent_sequences, context_input_sequence, conditions_input_sequence])

            # Expand from 23 features to 57 features for each time step
            # Input: (batch, seq_len, 23) -> Output: (batch, seq_len, 57)
            feature_expansion_layer = Dense(57, activation='linear', name="feature_expansion_dense")
            expanded_feature_sequences = TimeDistributed(feature_expansion_layer, name="td_feature_expansion")(decoder_output_sequences)

            # Create the composite model
            composite_generator = Model(
                inputs=[noise_input_sequence, conditions_input_sequence, context_input_sequence],
                outputs=expanded_feature_sequences,
                name="composite_sequence_generator"
            )

            # Compile the model (optional here, as GANTrainerPlugin will compile the combined GAN)
            # composite_generator.compile(
            #     optimizer='adam',
            #     loss='mse' # Placeholder loss
            # )

            self.logger.info(f"Composite sequence generator built with {composite_generator.count_params()} parameters.")
            # composite_generator.summary(print_fn=self.logger.info)
            return composite_generator

        except Exception as e:
            self.logger.error(f"Error building composite sequence generator: {e}")
            self.logger.error(traceback.format_exc())
            return None 

    def get_model(self) -> Optional[Model]:
        """Returns the built generator model."""
        return self.sequential_model # Or self.model, whichever holds the final composite generator
