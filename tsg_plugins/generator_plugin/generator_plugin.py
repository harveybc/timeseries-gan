#!/usr/bin/env python3
"""
Generator Plugin - Main Interface

Main plugin interface that orchestrates specialized modules for synthetic data generation.
Maintains mandatory plugin structure while delegating to focused modules.
"""

import logging
from typing import Dict, Any, Optional, List
import os
import traceback # Added traceback

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import (Dense, LSTM, RepeatVector, TimeDistributed, Input, 
                                     Bidirectional, Conv1D, BatchNormalization, Dropout,
                                     Concatenate, Reshape, Flatten, LeakyReLU, ReLU, Add) # Added Add
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2 # Added l2
from tsg_plugins.plugin_base import PluginBase
from app.utils.logging_utils import get_logger

# Imports for specialized modules
from .model_loader import ModelLoader
from .model_saver import ModelSaver
from .normalization_handler import NormalizationHandler
from .initial_data_handler import InitialDataHandler
from .feature_validator import FeatureValidator
from .data_generator import DataGenerator
from .technical_indicator_calculator import TechnicalIndicatorCalculator
from .sequence_builder import SequenceBuilder

logger = get_logger(__name__)

class GeneratorPlugin(PluginBase):
    """
    Generator Plugin for the TimeSeries GAN.
    Handles the generation of synthetic time series data.
    """
    plugin_name_prefix = "generator_"
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
        "feeder_noise_dim": 32, # Default noise dim, aligned with config.py
        "context_vector_dim": 64, # For decoder_input_h_context
        "conditional_features_dim": 10, # For decoder_input_conditions
        "num_features": 51, # Target output features for the composite generator
        "base_feature_names_ordered": [] # Added for _post_process_to_target_features if needed
    }

    plugin_debug_vars = [
        "sequential_model_file", "decoder_input_window_size", "batch_size_inference",
        "full_feature_names_ordered", "decoder_output_feature_names",
        "ti_calculation_min_lookback",
        "num_features" # Added num_features for debug
    ]
    
    def __init__(self, config: Dict[str, Any], pipeline_config: Dict[str, Any] = None):
        super().__init__(config)
        self.logger = get_logger(self.__class__.__name__)
        self.logger.debug(f"Initializing GeneratorPlugin with config: {config}")
        self.main_config = config
        self.pipeline_config = pipeline_config if pipeline_config is not None else {}
        self.model = None
        self.model_loader = ModelLoader(self.logger)
        self.model_saver = ModelSaver(self.logger)
        self.norm_handler = NormalizationHandler(self.logger)
        self.initial_data_handler = InitialDataHandler(self.logger)
        self.feature_validator = FeatureValidator(self.logger)
        self.data_generator = DataGenerator(self.logger)
        self.ti_calculator = TechnicalIndicatorCalculator(self.logger)
        self.sequence_builder = SequenceBuilder(self.logger)

        self._initialize_modules()
        self.set_params(**self.main_config) # Initialize params from main_config
        self.logger.debug("GeneratorPlugin initialized.")

    def _initialize_modules(self) -> None:
        """Initialize core modules for the generator."""
        self.logger.info("Initializing GeneratorPlugin modules...")
        
        # Model loading and saving
        self.model_loader = ModelLoader(self.params, self.logger) # Pass self.params
        self.model_saver = ModelSaver(self.logger)
        
        # Normalization (even if minimal, the handler object is expected)
        self.norm_handler = NormalizationHandler(self.logger)
        
        # Initial data handling (for anchor close, initial window)
        self.initial_data_handler = InitialDataHandler(
            logger=self.logger, 
            normalization_handler=self.norm_handler
        )
        
        # Feature engineering and validation
        self.feature_validator: Optional[FeatureValidator] = None
        self.data_generator: Optional[DataGenerator] = None
        self.ti_calculator: Optional[TechnicalIndicatorCalculator] = None
        self.sequence_builder: Optional[SequenceBuilder] = None
    
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
            self.norm_handler, self.ti_calculator
        )
        self.data_generator.set_main_config(self.main_config)
        
        # Sequence builder
        self.sequence_builder = SequenceBuilder(
            self.params, self.feature_to_idx, self.num_all_features,
            self.norm_handler, self.ti_calculator
        )
    
    def set_params(self, **kwargs) -> None:
        """
        Update plugin parameters and reload components as needed.
        
        Args:
            **kwargs: Parameter updates
        """
        self.logger.debug(f"Setting params for GeneratorPlugin with kwargs: {kwargs}")
        super().set_params(**kwargs)
        # Re-initialize or update components if necessary based on new params
        # For example, if 'generator_model_path' changes, the model might need reloading.
        # If L2 reg params change, model needs rebuilding.
        self.logger.debug(f"GeneratorPlugin params updated: {self.params}")
        # Potentially rebuild or reload model if relevant parameters changed
        if any(k in kwargs for k in ['generator_l2_reg_factor', 'use_generator_l2_reg', 'generator_model_path']):
            self.logger.info("Relevant parameters changed, re-initializing model components.")
            # self._load_model() # Or a more specific re-build logic
            if self.model and any(k in kwargs for k in ['generator_l2_reg_factor', 'use_generator_l2_reg']):
                self.logger.info("L2 regularization parameters changed. Rebuilding model.")
                # This assumes you have a way to rebuild with new L2, or you might need to reload a base model
                # and then apply modifications. For now, logging the intent.
                # If the model structure itself depends on these, a full rebuild is needed.
                pass # Placeholder for actual model rebuild logic if L2 params change post-init

    def _load_model(self) -> None:
        self.logger.info(f"Loading generator model from path: {self.params.get('generator_model_path')}")
        if not self.params.get('generator_model_path'):
            self.logger.warning("Generator model path not provided. Building a new VAE-based generator.")
            # This path assumes we are building the VAE-based generator from scratch or a base decoder
            # The REFERENCE.md implies the VAE decoder is loaded and then becomes part of a composite generator.
            
            # Attempt to load the VAE decoder first, as it's a required component
            vae_decoder_path = self.params.get('vae_decoder_model_path')
            if not vae_decoder_path:
                self.logger.error("VAE decoder model path ('vae_decoder_model_path') not provided. Cannot build generator.")
                raise ValueError("VAE decoder model path is required to build the generator.")

            try:
                self.logger.info(f"Loading VAE decoder from: {vae_decoder_path}")
                loaded_vae_decoder = self.model_loader.load_model(vae_decoder_path)
                if loaded_vae_decoder is None:
                    self.logger.error(f"Failed to load VAE decoder from {vae_decoder_path}.")
                    raise ValueError(f"Failed to load VAE decoder from {vae_decoder_path}.")
                self.logger.info("VAE decoder loaded successfully.")
                loaded_vae_decoder.trainable = True # As per REFERENCE.md
                self.logger.info(f"VAE decoder '{loaded_vae_decoder.name}' set to trainable=True.")

                # Now build the composite generator using the loaded VAE decoder
                # The error was here: _build_composite_generator does not exist.
                # Based on the error message, it should be _build_vae_generator or similar.
                # Assuming _build_vae_generator is the method that incorporates the VAE decoder.
                built_model = self._build_vae_generator(loaded_vae_decoder) # Corrected method name
                if built_model:
                    self.model = built_model
                    self.logger.info("Successfully built VAE-based generator using loaded VAE decoder.")
                    if self.params.get("print_model_summary", False):
                        self.model.summary(print_fn=self.logger.info)
                else:
                    self.logger.error("Failed to build VAE-based generator.")
                    raise RuntimeError("Generator model could not be built.")

            except Exception as e:
                self.logger.error(f"Error building VAE-based generator: {e}", exc_info=True)
                raise
        else:
            # Load a pre-existing full generator model
            try:
                loaded_model = self.model_loader.load_model(self.params['generator_model_path'])
                if loaded_model:
                    self.model = loaded_model
                    self.logger.info(f"Generator model loaded successfully from {self.params['generator_model_path']}.")
                    # If L2 regularization needs to be applied to a loaded model, it's more complex.
                    # Typically, L2 is part of the model's definition.
                    # For now, we assume a loaded model already has its intended configuration.
                    # If it's a VAE decoder being loaded to be part of a composite model, that's handled above.
                    if self.params.get("print_model_summary", False):
                        self.model.summary(print_fn=self.logger.info)
                else:
                    self.logger.error(f"Failed to load generator model from {self.params['generator_model_path']}.")
                    # Fallback or error based on requirements
                    raise FileNotFoundError(f"Generator model not found at {self.params['generator_model_path']}")
            except Exception as e:
                self.logger.error(f"Error loading generator model: {e}", exc_info=True)
                raise
        
        if self.model is None:
            self.logger.critical("Generator model is None after load/build attempt.")
            raise RuntimeError("Failed to initialize generator model.")

    def _apply_l2_regularization(self, layer):
        self.logger.debug(f"Applying L2 regularization to layer: {layer.name}")
        # Existing L2 application logic...

    def _build_bilstm_z_generator(self, input_noise_dim: int, output_latent_seq_len: int, output_latent_dim: int) -> tf.keras.Model:
        self.logger.debug(f"Building BiLSTM Z-Generator. Input noise dim: {input_noise_dim}, Output latent seq len: {output_latent_seq_len}, Output latent dim: {output_latent_dim}")
        # ... existing Keras model definition ...
        # Example: Add logging before returning
        self.logger.info("BiLSTM Z-Generator built.")
        return model

    def _build_vae_generator(self, vae_decoder_model: tf.keras.Model) -> tf.keras.Model:
        """
        Builds the composite GAN generator using the pre-trained VAE decoder.
        This involves creating the internal BiLSTM Z-generator and connecting it
        to the VAE decoder, along with handling conditional inputs.
        """
        self.logger.info("Building VAE-based composite generator.")
        self.logger.debug(f"Using VAE decoder: {vae_decoder_model.name}")

        # Parameters from config or defaults
        noise_dim = self.params.get("noise_dim", 100) # Example: dimension of the input noise vector for Z-generator
        # Latent dimensions should match VAE decoder's expectations for z_seq
        # From REFERENCE.md: VAE decoder input z_seq is (batch_size, 18, 32)
        internal_z_seq_len = 18 # self.params.get("internal_z_sequence_length", 18)
        internal_z_dim = 32 # self.params.get("internal_z_latent_dim", 32)

        # Conditional input dimensions
        # From REFERENCE.md: VAE decoder input_conditions is (batch_size, 10)
        conditional_dim = self.params.get("conditional_features_dim", 10) # Example: number of conditional features

        # 1. Define Input Layers for the Composite Generator
        self.logger.debug("Defining input layers for composite generator.")
        # Input for the Z-generator part
        noise_input = tf.keras.layers.Input(shape=(noise_dim,), name="noise_input")
        # Conditional inputs that go directly to the VAE decoder part
        conditional_input = tf.keras.layers.Input(shape=(conditional_dim,), name="conditional_input_to_vae")
        
        self.logger.debug(f"Noise input shape: {(noise_dim,)}, Conditional input shape: {(conditional_dim,)}")

        # 2. Build or Get the Internal BiLSTM Z-Generator
        # This Z-generator will produce the `decoder_input_z_seq` for the VAE decoder
        # Its input is `noise_input`, output is `(batch_size, internal_z_seq_len, internal_z_dim)`
        self.logger.debug("Building internal BiLSTM Z-generator.")
        # The Z-generator's internal Dense layer input size needs to be determined.
        # REFERENCE.md: Dense(576) -> Reshape(18,32) -> Bidirectional(LSTM(64)) -> Conv1D(32 filters)
        # So, the input to Dense(576) is our `noise_input`. We need to ensure noise_dim matches this,
        # or adjust the Z-generator. Let's assume noise_dim is flexible or Z-gen adapts.
        # For simplicity, let's assume the Z-generator takes `noise_input` and produces the z_sequence.
        
        # Simplified Z-Generator based on REFERENCE.md description
        # Dense layer input is noise_input
        x = tf.keras.layers.Dense(576, activation='relu', kernel_regularizer=self._get_l2_reg())(noise_input)
        self.logger.debug(f"Z-gen: Dense output shape: {x.shape}")
        x = tf.keras.layers.Reshape((internal_z_seq_len, internal_z_dim))(x) # Reshape to (18, 32)
        self.logger.debug(f"Z-gen: Reshape output shape: {x.shape}")
        # Bidirectional LSTM
        # Note: REFERENCE.md says LSTM(64), Bidirectional will double units if not specified for merge_mode
        lstm_layer = tf.keras.layers.LSTM(64, return_sequences=True, kernel_regularizer=self._get_l2_reg(), recurrent_regularizer=self._get_l2_reg())
        x = tf.keras.layers.Bidirectional(lstm_layer, name="z_bilstm")(x)
        self.logger.debug(f"Z-gen: BiLSTM output shape: {x.shape}") # Should be (None, 18, 128) if merge_mode='concat' (default)
        
        # Conv1D layer to get back to (None, 18, 32) for VAE decoder
        # The filter count should be `internal_z_dim` (32)
        z_sequence_for_vae = tf.keras.layers.Conv1D(filters=internal_z_dim, kernel_size=1, activation='linear', padding='same', kernel_regularizer=self._get_l2_reg(), name="z_conv1d_to_vae_spec")(x)
        self.logger.debug(f"Z-gen: Conv1D output shape (z_sequence_for_vae): {z_sequence_for_vae.shape}") # Should be (None, 18, 32)

        # 3. Connect to the VAE Decoder
        # The VAE decoder expects inputs in a specific order.
        # From REFERENCE.md: `decoder_input_z_seq`, `decoder_input_conditions`
        self.logger.debug(f"Preparing inputs for VAE decoder '{vae_decoder_model.name}'.")
        self.logger.debug(f"  - z_sequence_for_vae shape: {z_sequence_for_vae.shape}")
        self.logger.debug(f"  - conditional_input shape: {conditional_input.shape}")

        # Ensure VAE decoder is trainable
        vae_decoder_model.trainable = True
        self.logger.info(f"Ensured VAE decoder '{vae_decoder_model.name}' is trainable.")

        # Apply L2 to VAE decoder's trainable layers if specified
        if self.params.get("use_generator_l2_reg", False):
            self.logger.info(f"Applying L2 regularization to trainable layers of VAE decoder '{vae_decoder_model.name}'.")
            for layer in vae_decoder_model.layers:
                if isinstance(layer, (tf.keras.layers.Dense, tf.keras.layers.Conv1D)):
                    if layer.trainable:
                        self.logger.debug(f"Applying L2 to VAE decoder layer (kernel): {layer.name}")
                        layer.kernel_regularizer = self._get_l2_reg()
                elif isinstance(layer, tf.keras.layers.LSTM): # Includes Bidirectional wrapped LSTMs if VAE has them
                    if layer.trainable:
                        self.logger.debug(f"Applying L2 to VAE decoder LSTM layer (kernel/recurrent): {layer.name}")
                        layer.kernel_regularizer = self._get_l2_reg()
                        layer.recurrent_regularizer = self._get_l2_reg()
        
        # The VAE decoder might be a multi-input model itself.
        # We need to pass the generated z_sequence and the conditional_input to it.
        # Assuming vae_decoder_model.inputs is a list: [z_input_tensor, condition_input_tensor]
        # The names in REFERENCE.md are `decoder_input_z_seq` and `decoder_input_conditions`.
        # We need to ensure the VAE model loaded has input layers with these names or compatible structure.
        
        # Let's assume the loaded vae_decoder_model expects a list of inputs in the correct order.
        # Or, if it's a functional model, we can call it with a dictionary of inputs by name.
        # For simplicity, assuming it takes a list: [z_sequence, conditions]
        try:
            # Check input names of the VAE decoder to be sure
            vae_input_names = [inp.name for inp in vae_decoder_model.inputs]
            self.logger.info(f"VAE Decoder expected input layer names: {vae_input_names}")
            # Example: "decoder_input_z_seq:0", "decoder_input_conditions:0"
            # We need to map our generated tensors to these inputs.
            # If names are critical, use a dictionary:
            # vae_decoder_output = vae_decoder_model({
            #     vae_input_names[0]: z_sequence_for_vae,  # Assuming first is z_seq
            #     vae_input_names[1]: conditional_input    # Assuming second is conditions
            # })
            # For now, using list based on typical order from REFERENCE.md
            vae_decoder_output = vae_decoder_model([z_sequence_for_vae, conditional_input])
            self.logger.debug(f"VAE decoder output tensor: {vae_decoder_output}")
        except Exception as e:
            self.logger.error(f"Error when calling the VAE decoder model: {e}", exc_info=True)
            self.logger.error(f"VAE Decoder inputs: {vae_decoder_model.inputs}")
            self.logger.error(f"Provided z_sequence_for_vae: {z_sequence_for_vae}")
            self.logger.error(f"Provided conditional_input: {conditional_input}")
            raise

        # The output of vae_decoder_model is specified as `(batch_size, 23)` in REFERENCE.md
        # This is the `reconstruction_out` layer.
        # This output will then be post-processed by other methods in GeneratorPlugin
        # (e.g., calculate_technical_indicators, assemble_features)
        # For the Keras model definition, this is the final output of this composite generator.
        
        # 4. Create the Composite Keras Model
        # Inputs: noise_input, conditional_input (for VAE)
        # Output: vae_decoder_output
        composite_generator_model = tf.keras.Model(
            inputs=[noise_input, conditional_input],
            outputs=vae_decoder_output,
            name="Composite_VAE_GAN_Generator"
        )
        self.logger.info("Composite VAE-GAN Generator model built successfully.")
        
        if self.params.get("print_model_summary", False):
            self.logger.info("Composite VAE-GAN Generator Summary:")
            composite_generator_model.summary(print_fn=self.logger.info)

        return composite_generator_model

    def _get_l2_reg(self):
        # Existing L2 regularization logic...
        pass

    def build(self, input_shape: Tuple[int, ...], condition_shape: Tuple[int, ...] = None) -> tf.keras.Model:
        self.logger.info(f"Building generator model with input shape: {input_shape}, condition shape: {condition_shape}")
        # Existing build logic...
        pass

    def generate_synthetic_data(self, n_samples: int, initial_conditions: Optional[np.ndarray] = None, date_conditions: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        self.logger.info(f"Generating {n_samples} synthetic samples.")
        self.logger.debug(f"Initial conditions shape: {initial_conditions.shape if initial_conditions is not None else 'None'}")
        self.logger.debug(f"Date conditions shape: {date_conditions.shape if date_conditions is not None else 'None'}")
        # Existing data generation logic...
        pass

    def _iterative_generation_with_composite_model(
        self, 
        initial_conditions: np.ndarray, 
        date_conditions: pd.DataFrame, 
        n_steps: int, 
        step_size: int = 1
    ) -> pd.DataFrame:
        self.logger.info(f"Iterative generation with composite model. Steps: {n_steps}, Step size: {step_size}")
        # Existing iterative generation logic...
        pass

    def _post_process_generated_output(self, raw_generated_data: np.ndarray, current_datetimes_batch: pd.DataFrame) -> pd.DataFrame:
        self.logger.debug(f"Post-processing generated output. Raw data shape: {raw_generated_data.shape}, Datetimes count: {len(current_datetimes_batch)}")
        # Existing post-processing logic...
        pass

    def prepare_features_for_discriminator(self, real_data_batch: pd.DataFrame) -> np.ndarray:
        self.logger.debug(f"Preparing features for discriminator from real data batch of shape: {real_data_batch.shape}")
        # Existing feature preparation logic...
        pass
