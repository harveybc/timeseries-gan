#!/usr/bin/env python3
"""
Generator Plugin - Main Interface

Main plugin interface that orchestrates specialized modules for synthetic data generation.
Maintains mandatory plugin structure while delegating to focused modules.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple # Added Tuple
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
        self.pipeline_config = pipeline_config if pipeline_config is not None else {}
        
        self.model: Optional[Model] = None
        self.model_loader: Optional[ModelLoader] = None
        self.model_saver: Optional[ModelSaver] = None
        self.initial_data_handler: Optional[InitialDataHandler] = None
        self.feature_validator: Optional[FeatureValidator] = None
        self.data_generator: Optional[DataGenerator] = None
        self.ti_calculator: Optional[TechnicalIndicatorCalculator] = None
        self.sequence_builder: Optional[SequenceBuilder] = None
        self.feature_to_idx: Optional[Dict[str, int]] = None
        self.num_all_features: Optional[int] = None

        self._initialize_core_modules()

        if hasattr(self, 'main_config') and self.main_config:
            self.set_params(**self.main_config.copy())
        else:
            self.logger.warning("main_config not found after super().__init__, set_params might not have all initial values.")
            self.set_params()

        self.logger.debug("GeneratorPlugin initialized.")

    def _initialize_core_modules(self) -> None:
        """Initialize core modules for the generator that do not depend on feature names."""
        self.logger.info("Initializing GeneratorPlugin core modules...")
        
        if not hasattr(self, 'params') or not self.params:
            self.logger.error("Params not initialized before _initialize_core_modules call.")
            if not hasattr(self, 'params'): # Ensure self.params exists
                self.params = {} 
            
        self.model_loader = ModelLoader(self.params, self.logger)
        self.model_saver = ModelSaver(self.logger)
        
        self.initial_data_handler = InitialDataHandler(
            params=self.params, # Added missing params argument
            logger=self.logger
        )
        
        if self.params.get("full_feature_names_ordered") and not self.feature_validator:
            self.logger.info("Feature names found in initial params, initializing feature-dependent modules.")
            self._initialize_feature_dependent_modules()

    def _initialize_feature_dependent_modules(self) -> None:
        """Initialize modules that depend on feature configuration."""
        self.logger.info("Initializing feature-dependent modules...")
        if not self.params.get("full_feature_names_ordered"):
            self.logger.warning("Cannot initialize feature-dependent modules: 'full_feature_names_ordered' not in params.")
            return
        
        self.feature_validator = FeatureValidator(self.params["full_feature_names_ordered"], self.logger) # Pass self.logger
        
        self.feature_to_idx = self.feature_validator.create_feature_index_mapping()
        self.num_all_features = self.feature_validator.get_num_features()
        
        self.ti_calculator = TechnicalIndicatorCalculator(
            ti_feature_names=self.params.get("ti_feature_names", []), 
            ti_params=self.params.get("ti_params", {}),
            logger=self.logger
        )

        self.data_generator = DataGenerator(
            params=self.params, 
            feature_to_idx=self.feature_to_idx, 
            ti_calculator=self.ti_calculator,
            logger=self.logger
        )

        self.sequence_builder = SequenceBuilder(
            params=self.params, 
            feature_to_idx=self.feature_to_idx, 
            num_all_features=self.num_all_features,
            ti_calculator=self.ti_calculator,
            logger=self.logger
        )
        self.logger.info("Feature-dependent modules initialized.")
    
    def set_params(self, **kwargs) -> None:
        """
        Update plugin parameters and reload/rebuild components as needed.
        """
        self.logger.debug(f"Setting params for GeneratorPlugin with kwargs: {kwargs}")
        super().set_params(**kwargs)

        if ("full_feature_names_ordered" in kwargs or 
            (self.params.get("full_feature_names_ordered") and not self.feature_validator)):
            self.logger.info("Feature names updated or now available via set_params, (re-)initializing feature-dependent modules.")
            self._initialize_feature_dependent_modules()
        
        self.logger.debug(f"GeneratorPlugin params after super().set_params and potential feature module init: {self.params}")
        
        model_relevant_params_changed = any(k in kwargs for k in [
            'generator_l2_reg', 'use_generator_l2_reg', 
            'generator_model_path', 'sequential_model_file', 
            'internal_z_sequence_length', 'internal_z_latent_dim',
            'noise_dim', 'conditional_features_dim'
        ])

        if model_relevant_params_changed:
            self.logger.info("Relevant model parameters changed in set_params, attempting to reload/rebuild model.")
            if not self.model_loader:
                self.logger.warning("ModelLoader not initialized. Attempting to initialize core modules first.")
                self._initialize_core_modules()

            if self.model_loader:
                try:
                    self._load_model() 
                except Exception as e:
                    self.logger.error(f"Failed to reload/rebuild model in set_params: {e}", exc_info=True)
            else:
                self.logger.error("ModelLoader still not available after check in set_params. Cannot load model.")
        elif not self.model and (self.params.get('sequential_model_file') or self.params.get('generator_model_path')):
            self.logger.info("Initial model load triggered from set_params as model is None and paths are available.")
            if not self.model_loader:
                self.logger.warning("ModelLoader not initialized before initial model load. Initializing core modules.")
                self._initialize_core_modules()
            
            if self.model_loader:
                try:
                    self._load_model()
                except Exception as e:
                    self.logger.error(f"Failed initial model load in set_params: {e}", exc_info=True)
            else:
                self.logger.error("ModelLoader not available for initial model load in set_params.")

    def _load_model(self) -> None:
        self.logger.info(f"Loading generator model from path: {self.params.get('generator_model_path')}")
        if not self.params.get('generator_model_path'):
            self.logger.warning("Generator model path not provided. Building a new VAE-based generator.")
            vae_decoder_path = self.params.get('sequential_model_file')
            if not vae_decoder_path:
                self.logger.error("VAE decoder model path ('sequential_model_file') not provided. Cannot build generator.")
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
            try:
                loaded_model = self.model_loader.load_model(self.params['generator_model_path'])
                if loaded_model:
                    self.model = loaded_model
                    self.logger.info(f"Generator model loaded successfully from {self.params['generator_model_path']}.")
                    if self.params.get("print_model_summary", False):
                        self.model.summary(print_fn=self.logger.info)
                else:
                    self.logger.error(f"Failed to load generator model from {self.params['generator_model_path']}.")
                    raise FileNotFoundError(f"Generator model not found at {self.params['generator_model_path']}")
            except Exception as e:
                self.logger.error(f"Error loading generator model: {e}", exc_info=True)
                raise
        
        if self.model is None:
            self.logger.critical("Generator model is None after load/build attempt.")
            raise RuntimeError("Failed to initialize generator model.")

    def _apply_l2_regularization(self, layer):
        self.logger.debug(f"Applying L2 regularization to layer: {layer.name}")
        if self.params.get("use_generator_l2_reg", False):
            l2_factor = self.params.get("generator_l2_reg_factor", 0.01)
            if isinstance(layer, (tf.keras.layers.Dense, tf.keras.layers.Conv1D)):
                if hasattr(layer, 'kernel_regularizer'):
                    layer.kernel_regularizer = l2(l2_factor)
                    self.logger.debug(f"Applied L2 (factor: {l2_factor}) to kernel of layer: {layer.name}")
            elif isinstance(layer, tf.keras.layers.LSTM):
                if hasattr(layer, 'kernel_regularizer'):
                    layer.kernel_regularizer = l2(l2_factor)
                    self.logger.debug(f"Applied L2 (factor: {l2_factor}) to kernel of LSTM layer: {layer.name}")
                if hasattr(layer, 'recurrent_regularizer'):
                    layer.recurrent_regularizer = l2(l2_factor)
                    self.logger.debug(f"Applied L2 (factor: {l2_factor}) to recurrent kernel of LSTM layer: {layer.name}")
            elif isinstance(layer, tf.keras.layers.Bidirectional):
                if hasattr(layer.forward_layer, 'kernel_regularizer'):
                    layer.forward_layer.kernel_regularizer = l2(l2_factor)
                    layer.backward_layer.kernel_regularizer = l2(l2_factor)
                    self.logger.debug(f"Applied L2 (factor: {l2_factor}) to kernel of Bidirectional LSTM (forward/backward): {layer.name}")
                if hasattr(layer.forward_layer, 'recurrent_regularizer'):
                    layer.forward_layer.recurrent_regularizer = l2(l2_factor)
                    layer.backward_layer.recurrent_regularizer = l2(l2_factor)
                    self.logger.debug(f"Applied L2 (factor: {l2_factor}) to recurrent kernel of Bidirectional LSTM (forward/backward): {layer.name}")

    def _build_bilstm_z_generator(self, input_noise_dim: int, output_latent_seq_len: int, output_latent_dim: int) -> tf.keras.Model:
        self.logger.debug(f"Building BiLSTM Z-Generator. Input noise dim: {input_noise_dim}, Output latent seq len: {output_latent_seq_len}, Output latent dim: {output_latent_dim}")
        
        noise_input = tf.keras.layers.Input(shape=(input_noise_dim,), name="z_generator_noise_input")
        
        dense_units = output_latent_seq_len * output_latent_dim
        x = tf.keras.layers.Dense(dense_units, activation='relu', kernel_regularizer=self._get_l2_reg())(noise_input)
        x = tf.keras.layers.Reshape((output_latent_seq_len, output_latent_dim))(x)
        
        lstm_layer = tf.keras.layers.LSTM(64, return_sequences=True, kernel_regularizer=self._get_l2_reg(), recurrent_regularizer=self._get_l2_reg())
        x = tf.keras.layers.Bidirectional(lstm_layer, name="z_bilstm_internal")(x)
        
        z_sequence_output = tf.keras.layers.Conv1D(filters=output_latent_dim, kernel_size=1, activation='linear', padding='same', kernel_regularizer=self._get_l2_reg(), name="z_conv1d_to_target_dim")(x)
        
        model = tf.keras.Model(inputs=noise_input, outputs=z_sequence_output, name="Internal_BiLSTM_Z_Generator")
        
        self.logger.info("Internal BiLSTM Z-Generator built.")
        if self.params.get("print_model_summary", False):
            model.summary(print_fn=self.logger.info)
        return model

    def _build_vae_generator(self, vae_decoder_model: tf.keras.Model) -> tf.keras.Model:
        """
        Builds the composite GAN generator using the pre-trained VAE decoder.
        This involves creating the internal BiLSTM Z-generator and connecting it
        to the VAE decoder, along with handling conditional inputs.
        """
        self.logger.info("Building VAE-based composite generator.")
        self.logger.debug(f"Using VAE decoder: {vae_decoder_model.name}")

        noise_dim = self.params.get("noise_dim", 100) # Example: dimension of the input noise vector for Z-generator
        internal_z_seq_len = 18 # self.params.get("internal_z_sequence_length", 18)
        internal_z_dim = 32 # self.params.get("internal_z_latent_dim", 32)

        conditional_dim = self.params.get("conditional_features_dim", 10) # Example: number of conditional features

        # 1. Define Input Layers for the Composite Generator
        self.logger.debug("Defining input layers for composite generator.")
        noise_input = tf.keras.layers.Input(shape=(noise_dim,), name="noise_input")
        conditional_input = tf.keras.layers.Input(shape=(conditional_dim,), name="conditional_input_to_vae")
        
        self.logger.debug(f"Noise input shape: {(noise_dim,)}, Conditional input shape: {(conditional_dim,)}")

        # 2. Build or Get the Internal BiLSTM Z-Generator
        self.logger.debug("Building internal BiLSTM Z-generator.")
        
        x = tf.keras.layers.Dense(576, activation='relu', kernel_regularizer=self._get_l2_reg())(noise_input)
        self.logger.debug(f"Z-gen: Dense output shape: {x.shape}")
        x = tf.keras.layers.Reshape((internal_z_seq_len, internal_z_dim))(x) # Reshape to (18, 32)
        self.logger.debug(f"Z-gen: Reshape output shape: {x.shape}")
        lstm_layer = tf.keras.layers.LSTM(64, return_sequences=True, kernel_regularizer=self._get_l2_reg(), recurrent_regularizer=self._get_l2_reg())
        x = tf.keras.layers.Bidirectional(lstm_layer, name="z_bilstm")(x)
        self.logger.debug(f"Z-gen: BiLSTM output shape: {x.shape}") # Should be (None, 18, 128) if merge_mode='concat' (default)
        
        z_sequence_for_vae = tf.keras.layers.Conv1D(filters=internal_z_dim, kernel_size=1, activation='linear', padding='same', kernel_regularizer=self._get_l2_reg(), name="z_conv1d_to_vae_spec")(x)
        self.logger.debug(f"Z-gen: Conv1D output shape (z_sequence_for_vae): {z_sequence_for_vae.shape}") # Should be (None, 18, 32)

        # 3. Connect to the VAE Decoder
        self.logger.debug(f"Preparing inputs for VAE decoder '{vae_decoder_model.name}'.")
        self.logger.debug(f"  - z_sequence_for_vae shape: {z_sequence_for_vae.shape}")
        self.logger.debug(f"  - conditional_input shape: {conditional_input.shape}")

        vae_decoder_model.trainable = True
        self.logger.info(f"Ensured VAE decoder '{vae_decoder_model.name}' is trainable.")

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
        
        try:
            vae_input_names = [inp.name for inp in vae_decoder_model.inputs]
            self.logger.info(f"VAE Decoder expected input layer names: {vae_input_names}")
            vae_decoder_output = vae_decoder_model([z_sequence_for_vae, conditional_input])
            self.logger.debug(f"VAE decoder output tensor: {vae_decoder_output}")
        except Exception as e:
            self.logger.error(f"Error when calling the VAE decoder model: {e}", exc_info=True)
            self.logger.error(f"VAE Decoder inputs: {vae_decoder_model.inputs}")
            self.logger.error(f"Provided z_sequence_for_vae: {z_sequence_for_vae}")
            self.logger.error(f"Provided conditional_input: {conditional_input}")
            raise

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
        """Get L2 regularizer if enabled, otherwise return None."""
        if self.params.get("use_generator_l2_reg", False):
            l2_factor = self.params.get("generator_l2_reg", 0.01)
            return l2(l2_factor)
        return None

    def build(self, input_shape: Tuple[int, ...], condition_shape: Tuple[int, ...] = None) -> tf.keras.Model:
        """
        Build generator model with specified input and condition shapes.
        
        Args:
            input_shape: Shape of the noise input
            condition_shape: Shape of conditional inputs
            
        Returns:
            Built Keras model
        """
        self.logger.info(f"Building generator model with input shape: {input_shape}, condition shape: {condition_shape}")
        
        # If model is already loaded/built, return it
        if self.model is not None:
            self.logger.info("Generator model already exists, returning existing model")
            return self.model
            
        # Try to load/build the model
        try:
            self._load_model()
            return self.model
        except Exception as e:
            self.logger.error(f"Failed to build generator model: {e}")
            raise RuntimeError(f"Generator model build failed: {e}")

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

    @property
    def model(self):
        """Property to access the generator model."""
        return self._model if hasattr(self, '_model') else None

    @model.setter
    def model(self, value):
        """Property setter for the generator model."""
        self._model = value

    def get_model(self):
        """Get the generator model for training."""
        if self.model is None:
            self.logger.warning("Generator model is None. Attempting to load/build model.")
            try:
                self._load_model()
            except Exception as e:
                self.logger.error(f"Failed to load/build model in get_model: {e}")
                return None
        return self.model
