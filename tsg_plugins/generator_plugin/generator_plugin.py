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
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Conv1D, Dense, Reshape, Concatenate, ZeroPadding1D, TimeDistributed
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
        
        # Handle initial close anchor reload
        self._handle_initial_close_anchor_reload()
    
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
        
        According to REFERENCE.md architecture:
        1. BiLSTM Z-generator: Dense(576) → Reshape(18,32) → Bidirectional(LSTM(64)) → Conv1D(32 filters)
        2. VAE decoder inputs: decoder_input_z_seq (18,32), decoder_input_conditions (10)
        3. VAE decoder outputs: 23 features from reconstruction_out layer
        4. Post-processing: Expand to 57 features (23 base + 34 technical indicators/other features)
        
        Args:
            vae_decoder: Pre-trained VAE decoder model
            
        Returns:
            Composite generator model that outputs 57 features
        """
        try:
            self.logger.info("Building composite generator model...")

            # Input layers for the composite generator
            noise_input = Input(shape=(self.params["feeder_noise_dim"],), name="noise_input")
            conditions_input = Input(shape=(self.params["conditional_features_dim"],), name="conditions_input")

            # BiLSTM Z-generator: Dense(576) → Reshape(18,32) → Bidirectional(LSTM(64)) → Conv1D(32 filters)
            z_dense = Dense(576, activation='relu', name="z_dense")(noise_input)
            z_reshaped = Reshape((18, 32), name="z_reshape")(z_dense)
            z_bilstm = Bidirectional(LSTM(64, return_sequences=True), name="z_bilstm")(z_reshaped)
            z_latent_seq = Conv1D(32, kernel_size=3, padding='same', activation='relu', name="z_conv")(z_bilstm)

            # Get VAE decoder output (23 features)
            # VAE decoder expects: decoder_input_z_seq and decoder_input_conditions
            decoder_output = vae_decoder([z_latent_seq, conditions_input])

            # Expand from 23 features to 57 features
            # This simulates the post-processing that adds technical indicators and other features
            expanded_features = Dense(57, activation='linear', name="feature_expansion")(decoder_output)

            # Create the composite model
            composite_generator = Model(
                inputs=[noise_input, conditions_input],
                outputs=expanded_features,
                name="composite_generator"
            )

            # Compile the model
            composite_generator.compile(
                optimizer='adam',
                loss='mse'
            )

            self.logger.info(f"Composite generator built with {composite_generator.count_params()} parameters")
            return composite_generator

        except Exception as e:
            self.logger.error(f"Error building composite generator: {e}")
            # Consider re-raising the exception or returning a specific error indicator.
            # If this path is taken, the function will implicitly return None.
            # This might conflict with the type hint '-> Model'.
            # Adding an explicit return None for clarity if an error is caught and not re-raised.
            return None 

        # NOTE: The following block of code appears to be unreachable if the 'try' block above succeeds,
        # as that block contains 'return composite_generator'.
        # If the 'try' block fails and the 'except' block is executed (as modified above to return None),
        # this code is also not reached.
        # If the 'except' block does not return, then this code would be reached after an error in the 'try' block.
        # This section also references 'feeder_noise_input' and 'loaded_vae_decoder'
        # which might not be in scope here, potentially leading to NameErrors.
        # Please review the logical structure of this method.

        # Assuming 'feeder_noise_input' should be defined for this path if it were reachable.
        # For example:
        # feeder_noise_input = Input(shape=(self.params["feeder_noise_dim"],), name="feeder_noise_input_alternate_path")


        # For simplicity, let's assume its dimensionality is fixed or derived from decoder_output_feature_names
        # For now, let's use context_vector_dim directly, assuming it's prepared correctly.
        previous_step_output_input = Input(
            shape=(self.params["context_vector_dim"],), # This will feed h_context
            name="composite_generator_previous_step_output_input"
        )
        # These are the date/time conditions for the current step
        current_step_conditions_input = Input(
            shape=(self.params["conditional_features_dim"],), # This will feed decoder_input_conditions
            name="composite_generator_current_step_conditions_input"
        )

        # --- 3. Build the internal Z-sequence generator sub-model ---
        # This sub-model takes feeder_noise_input and generates decoder_input_z_seq
        
        # The 'feeder_noise_input' used below would need to be defined in this execution path.
        # If this code block is intended to be functional, ensure 'feeder_noise_input' is correctly sourced.
        # x = Dense(self.params["internal_z_sequence_length"] * self.params["internal_z_latent_dim"], activation='relu')(feeder_noise_input) # Increased units for better representation
        # x = Reshape((self.params["internal_z_sequence_length"], self.params["internal_z_latent_dim"]))(x) # Reshape to (batch, 18, 32) directly if Dense output matches
        
        # # Bidirectional LSTM layer
        # x = Bidirectional(LSTM(self.params["internal_z_latent_dim"] * 2, return_sequences=True))(x) # More units in LSTM
        
        # # Conv1D layer to refine features per timestep and ensure correct output dimension
        # internal_z_seq_output = Conv1D(
        #     filters=self.params["internal_z_latent_dim"], 
        #     kernel_size=1, # Pointwise convolution to adjust feature dimension
        #     padding="same", 
        #     activation='tanh', # tanh is common for latent vectors
        #     name="internal_z_seq_output"
        # )(x)
        
        # # --- 4. Prepare inputs for the loaded VAE Decoder ---
        # # 'loaded_vae_decoder' is not in scope here. This would cause a NameError.
        # vae_decoder_input_z_seq = internal_z_seq_output 
        # vae_decoder_input_h_context = previous_step_output_input
        # vae_decoder_input_conditions = current_step_conditions_input

        # # Log the names of the input layers of the loaded VAE decoder
        # vae_decoder_actual_input_names = [inp.name for inp in loaded_vae_decoder.inputs]
        # self.logger.info(f"Loaded VAE Decoder actual input layer names: {vae_decoder_actual_input_names}")

        # # Prepare the dictionary of inputs for the VAE decoder
        # # The keys in this dictionary MUST match the names of the input layers of the loaded_vae_decoder
        # # or the names provided when the VAE decoder model was originally defined.
        
        # # Expected names from params:
        # # self.params[\"decoder_input_name_latent\"] -> 'decoder_input_z_seq'
        # # self.params[\"decoder_input_name_conditions\"] -> 'decoder_input_conditions'
        # # self.params[\"decoder_input_name_context\"] -> 'decoder_input_h_context'
        # # self.params[\"decoder_input_name_window\"] -> 'input_x_window' (potentially)

        # vae_inputs_for_loaded_decoder = {}
        
        # # Map our generated/provided tensors to the VAE decoder's *actual* input layers
        # # This requires knowing the exact names the VAE decoder expects.
        # # We assume the names in plugin_params are the target names for the VAE decoder's inputs.

        # # Attempt to map by the names defined in params
        # # This is crucial: the names must match what the loaded_vae_decoder expects.
        # # For example, if loaded_vae_decoder.inputs[0] is named 'z_input_for_vae', then
        # # self.params[\"decoder_input_name_latent\"] should be 'z_input_for_vae'.
        
        # # Let's iterate through the VAE decoder's actual inputs and try to map them
        # # from our prepared tensors using the names stored in self.params.
        
        # param_to_tensor_map = {
        #     self.params["decoder_input_name_latent"]: vae_decoder_input_z_seq,
        #     self.params["decoder_input_name_conditions"]: vae_decoder_input_conditions,
        #     self.params["decoder_input_name_context"]: vae_decoder_input_h_context,
        #     # How to handle self.params[\"decoder_input_name_window\"] ('input_x_window')?
        #     # If the VAE decoder requires 'input_x_window', we need to provide it.
        #     # For a GAN generating from noise, this input might not be directly available
        #     # from the composite generator's external inputs.
        #     # It might need to be a learned constant, zeros, or derived differently.
        # }
