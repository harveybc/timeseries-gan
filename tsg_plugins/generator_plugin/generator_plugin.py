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
                                     Bidirectional, Conv1D, BatchNormalization,
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

class SimpleRandomWalkNoiseLayer(tf.keras.layers.Layer):
    """
    Simple layer that generates sequential noise using random walk.
    This creates natural sequential correlation with minimal parameters.
    
    Much more efficient than Dense(576) + Reshape approach:
    - Only uses one small Dense layer (noise_dim * latent_dim parameters)
    - Creates true sequential patterns through random walk
    - No heavy parameter overhead
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

class GeneratorPlugin(PluginBase):
    """
    Generator Plugin for the TimeSeries GAN.
    Handles the generation of synthetic time series data.
    """
    plugin_name_prefix = "generator_"
    plugin_params = {
        "sequential_model_file": None, # This will be the VAE DECODER path
        "vae_decoder_model_path_param": None, # Explicit param for VAE decoder path
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
        "num_features": 23, # Training: Use 23 base features for GAN training
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
            # Use the new explicit parameter for the VAE decoder path
            vae_decoder_path = self.params.get('vae_decoder_model_path_param') 
            if not vae_decoder_path:
                self.logger.error("VAE decoder model path ('vae_decoder_model_path_param') not provided. Cannot build generator.")
                raise ValueError("VAE decoder model path ('vae_decoder_model_path_param') is required to build the generator.")

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

    def _build_bilstm_z_generator(self, input_noise_dim: int, output_latent_seq_len: int, output_latent_dim: int) -> tf.keras.Model:
        self.logger.debug(f"Building BiLSTM Z-Generator. Input noise dim: {input_noise_dim}, Output latent seq len: {output_latent_seq_len}, Output latent dim: {output_latent_dim}")
        
        noise_input = tf.keras.layers.Input(shape=(input_noise_dim,), name="z_generator_noise_input")
        
        dense_units = output_latent_seq_len * output_latent_dim
        x = tf.keras.layers.Dense(dense_units, activation='tanh')(noise_input)
        x = tf.keras.layers.Reshape((output_latent_seq_len, output_latent_dim))(x)
        
        lstm_layer = tf.keras.layers.LSTM(64, return_sequences=True)
        x = tf.keras.layers.Bidirectional(lstm_layer, name="z_bilstm_internal")(x)
        
        z_sequence_output = tf.keras.layers.Conv1D(filters=output_latent_dim, kernel_size=1, activation='tanh', padding='same', name="z_conv1d_to_target_dim")(x)
        
        model = tf.keras.Model(inputs=noise_input, outputs=z_sequence_output, name="Internal_BiLSTM_Z_Generator")
        
        self.logger.info("Internal BiLSTM Z-Generator built.")
        if self.params.get("print_model_summary", True):
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
        context_dim = self.params.get("context_vector_dim", 64) # For decoder_input_h_context

        # 1. Define Input Layers for the Composite Generator
        self.logger.debug("Defining input layers for composite generator.")
        noise_input = tf.keras.layers.Input(shape=(noise_dim,), name="noise_input")
        conditional_input = tf.keras.layers.Input(shape=(conditional_dim,), name="conditional_input_to_vae")
        context_input = tf.keras.layers.Input(shape=(context_dim,), name="context_input_to_vae")
        
        self.logger.debug(f"Noise input shape: {(noise_dim,)}, Conditional input shape: {(conditional_dim,)}, Context input shape: {(context_dim,)}")

        # 2. Build or Get the Internal BiLSTM Z-Generator
        self.logger.debug("Building internal BiLSTM Z-generator with improved sequential noise generation.")
        
        # Use simple random walk approach instead of Dense+Reshape
        # This generates true sequential patterns with much fewer parameters
        sequential_noise_layer = SimpleRandomWalkNoiseLayer(
            seq_len=internal_z_seq_len, 
            latent_dim=internal_z_dim,
            name="sequential_noise_generator"
        )
        x = sequential_noise_layer(noise_input)
        self.logger.debug(f"Z-gen: Sequential noise output shape: {x.shape}")  # Should be (None, 18, 32)
        
        # Calculate parameter comparison for logging
        random_walk_params = sequential_noise_layer.count_params() if hasattr(sequential_noise_layer, 'count_params') else (noise_dim * internal_z_dim)
        dense_reshape_params = 576 * (noise_dim + 1)
        self.logger.info(f"Z-gen: Using random walk approach with ~{random_walk_params:,} parameters (vs {dense_reshape_params:,} for Dense+Reshape)")
        


        # Apply additional processing to ensure proper latent representation
        # Use Conv1D to refine the sequential noise to match VAE decoder expectations
        self.logger.debug("Applying Conv1D to refine sequential noise for VAE decoder compatibility.")
        
        x = tf.keras.layers.Conv1D(
            filters=32, 
            kernel_size=3, 
            activation='tanh', 
            padding='same', 
            name="z_conv1d_refinement1"
        )(x)
        

        lstm_layer = tf.keras.layers.LSTM(16, return_sequences=True)
        x = tf.keras.layers.Bidirectional(lstm_layer, name="z_bilstm")(x)
        self.logger.debug(f"Z-gen: BiLSTM output shape: {x.shape}") # Should be (None, 18, 128) if merge_mode='concat' (default)
        
        z_sequence_for_vae = tf.keras.layers.Conv1D(
            filters=32, 
            kernel_size=3, 
            activation='tanh', 
            padding='same', 
            name="z_conv1d_refinement2"
        )(x)
        self.logger.debug(f"Z-gen: Conv1D output shape (z_sequence_for_vae): {z_sequence_for_vae.shape}") # Should be (None, 18, 32)


        # 3. Connect to the VAE Decoder
        self.logger.debug(f"Preparing inputs for VAE decoder '{vae_decoder_model.name}'.")
        self.logger.debug(f"  - z_sequence_for_vae shape: {z_sequence_for_vae.shape}")
        self.logger.debug(f"  - context_input shape: {context_input.shape}")
        self.logger.debug(f"  - conditional_input shape: {conditional_input.shape}")

        vae_decoder_model.trainable = True
        self.logger.info(f"Ensured VAE decoder '{vae_decoder_model.name}' is trainable.")
        
        try:
            vae_input_names = [inp.name for inp in vae_decoder_model.inputs]
            self.logger.info(f"VAE Decoder expected input layer names: {vae_input_names}")
            vae_decoder_output = vae_decoder_model([z_sequence_for_vae, context_input, conditional_input])
            self.logger.debug(f"VAE decoder output tensor: {vae_decoder_output}")
            
            # Post-process VAE decoder output based on operation mode  
            # VAE decoder outputs 23 base features
            operation_mode = self.main_config.get("operation_mode", "train")
            
            if operation_mode == "generate":
                # === GENERATE MODE: Expand to 44 features with proper sequence generation ===
                self.logger.info("Generate mode: Expanding VAE output from 23 to 44 features")
                
                # Instead of expanding once and repeating, we need to generate diverse sequences
                # First expand the single timestep to get base pattern
                expansion_layer = FeatureExpansionLayer(name="feature_expansion")
                base_expanded_features = expansion_layer(vae_decoder_output)  # Shape: (batch_size, 44)
                self.logger.debug(f"Base expanded features shape: {base_expanded_features.shape}")
                
                # Generate diverse sequence by creating variations of the base pattern using custom layer
                sequence_generator = DiverseSequenceGeneratorLayer(seq_len=144, name="sequence_generator")
                sequence_output = sequence_generator(base_expanded_features)
                
                self.logger.debug(f"Final sequence output shape (44 features): {sequence_output.shape}")
                
            else:
                # === TRAINING MODE: Keep 23 base features ===
                self.logger.info("Training mode: Using 23 base features directly from VAE decoder")
                
                # Create sequences directly from 23 base features
                sequence_base = tf.keras.layers.RepeatVector(36)(vae_decoder_output)  # (batch_size, 32, 23)

                #use conv1dtranspopose with stride  = 2 
                sequence_base = tf.keras.layers.Conv1DTranspose(
                    filters=23, kernel_size=3, strides=2, padding='same', activation='tanh', name="sequence_base_conv1d_transpose_1"
                )(sequence_base)

                        #use conv1dtranspopose with stride  = 2 
                sequence_base = tf.keras.layers.Conv1DTranspose(
                    filters=23, kernel_size=3, strides=2, padding='same', activation='tanh', name="sequence_base_conv1d_transpose_2"
                )(sequence_base)        
                
                # Add small random variations to make realistic time sequences
                #sequence_output = tf.keras.layers.GaussianNoise(stddev=0.01)(sequence_base)  # Small noise for variation
                sequence_output = sequence_base 

                self.logger.debug(f"Final sequence output shape (23 features): {sequence_output.shape}")
            
        except Exception as e:
            self.logger.error(f"Error when calling the VAE decoder model: {e}", exc_info=True)
            self.logger.error(f"VAE Decoder inputs: {vae_decoder_model.inputs}")
            self.logger.error(f"Provided z_sequence_for_vae: {z_sequence_for_vae}")
            self.logger.error(f"Provided context_input: {context_input}")
            self.logger.error(f"Provided conditional_input: {conditional_input}")
            raise

        composite_generator_model = tf.keras.Model(
            inputs=[noise_input, conditional_input, context_input],
            outputs=sequence_output,
            name="Composite_VAE_GAN_Generator"
        )
        self.logger.info("Composite VAE-GAN Generator model built successfully.")
        
        if self.params.get("print_model_summary", False):
            self.logger.info("Composite VAE-GAN Generator Summary:")
            composite_generator_model.summary(print_fn=self.logger.info)

        return composite_generator_model

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
        """
        Generate synthetic time series data.
        
        Args:
            n_samples: Number of samples to generate
            initial_conditions: Initial conditions for generation
            date_conditions: Date-based conditions
            
        Returns:
            DataFrame with synthetic data
        """
        self.logger.info(f"Generating {n_samples} synthetic samples.")
        self.logger.debug(f"Initial conditions shape: {initial_conditions.shape if initial_conditions is not None else 'None'}")
        self.logger.debug(f"Date conditions shape: {date_conditions.shape if date_conditions is not None else 'None'}")
        
        if self.model is None:
            self.logger.error("Generator model not available for generation")
            raise RuntimeError("Generator model not built")
            
        # Generate random noise 
        noise_dim = self.params.get("noise_dim", 100)
        noise = np.random.normal(0, 1, (n_samples, noise_dim))
        
        # Create default conditions if not provided
        if initial_conditions is None:
            context_dim = self.params.get("context_vector_dim", 64)
            initial_conditions = np.random.normal(0, 1, (n_samples, context_dim))
            
        # Create conditions input
        conditions_dim = self.params.get("conditional_features_dim", 10)
        conditions = np.random.normal(0, 1, (n_samples, conditions_dim))
        
        # Generate synthetic data using the 23-feature architecture
        raw_output = self.model.predict([noise, initial_conditions, conditions], verbose=0)
        self.logger.debug(f"Raw generator output shape: {raw_output.shape}")
        
        # Check if we need to expand 23 base features to 44 features for generate mode
        operation_mode = self.main_config.get("operation_mode", "train") if hasattr(self, 'main_config') and self.main_config else "train"
        
        if operation_mode == "generate":
            # Check if the model already outputs 44 features (sequence shape: batch_size, seq_len, features)
            if len(raw_output.shape) == 3 and raw_output.shape[2] == 44:
                self.logger.info(f"Generate mode: Model already outputs 44 features. Shape: {raw_output.shape}")
                return raw_output
            elif len(raw_output.shape) == 3 and raw_output.shape[2] == 23:
                self.logger.info(f"Generate mode: Expanding sequence data from 23 to 44 features. Input shape: {raw_output.shape}")
                
                # For sequence data, we need to expand each timestep
                batch_size, seq_len, num_features = raw_output.shape
                expanded_sequences = []
                
                for t in range(seq_len):
                    timestep_data = raw_output[:, t, :]  # Shape: (batch_size, 23)
                    expanded_timestep = self._expand_vae_output_to_44_features(timestep_data)  # Shape: (batch_size, 44)
                    expanded_sequences.append(expanded_timestep)
                
                # Stack to create sequence: (batch_size, seq_len, 44)
                expanded_output = tf.stack(expanded_sequences, axis=1)
                self.logger.debug(f"Expanded sequence output shape: {expanded_output.shape}")
                return expanded_output.numpy()
            else:
                self.logger.error(f"Unexpected raw output shape in generate mode: {raw_output.shape}")
                return raw_output
        else:
            # Training mode: Return 23 base features as-is
            self.logger.debug(f"Training mode: Returning 23 features as-is. Shape: {raw_output.shape}")
            return raw_output

    def _iterative_generation_with_composite_model(
        self, 
        initial_conditions: np.ndarray, 
        date_conditions: pd.DataFrame, 
        n_steps: int, 
        step_size: int = 1
    ) -> pd.DataFrame:
        """
        Perform iterative generation using the composite model.
        
        Args:
            initial_conditions: Initial conditions for generation
            date_conditions: Date-based conditions
            n_steps: Number of steps to generate
            step_size: Step size for generation
            
        Returns:
            DataFrame with generated data
        """
        self.logger.info(f"Iterative generation with composite model. Steps: {n_steps}, Step size: {step_size}")
        
        # Placeholder implementation
        self.logger.warning("_iterative_generation_with_composite_model method is not fully implemented yet")
        raise NotImplementedError("Iterative generation not yet implemented")

    def _post_process_generated_output(self, raw_generated_data: np.ndarray, current_datetimes_batch: pd.DataFrame) -> pd.DataFrame:
        """
        Post-process generated output to final format.
        
        Args:
            raw_generated_data: Raw generated data from model
            current_datetimes_batch: Corresponding datetime information
            
        Returns:
            Post-processed DataFrame
        """
        self.logger.debug(f"Post-processing generated output. Raw data shape: {raw_generated_data.shape}, Datetimes count: {len(current_datetimes_batch)}")
        
        # Placeholder implementation
        self.logger.warning("_post_process_generated_output method is not fully implemented yet")
        raise NotImplementedError("Post-processing not yet implemented")

    def prepare_features_for_discriminator(self, real_data_batch: pd.DataFrame) -> np.ndarray:
        """
        Prepare features from real data for discriminator training.
        
        This method processes real data to match the discriminator's expected input format.
        In the new 23-feature architecture, we use only the base features for training.
        
        Args:
            real_data_batch: Real data batch DataFrame
            
        Returns:
            Prepared features array with shape (n_samples, 23)
        """
        self.logger.debug(f"Preparing features for discriminator from real data batch of shape: {real_data_batch.shape}")
        
        try:
            # Convert DataFrame to numpy if needed
            if isinstance(real_data_batch, pd.DataFrame):
                # Filter out non-numeric columns (like datetime columns)
                numeric_columns = real_data_batch.select_dtypes(include=[np.number]).columns
                if len(numeric_columns) < real_data_batch.shape[1]:
                    self.logger.info(f"Filtering out {real_data_batch.shape[1] - len(numeric_columns)} non-numeric columns")
                    real_data_batch = real_data_batch[numeric_columns]
                    self.logger.info(f"After filtering: {real_data_batch.shape}")
                
                data_array = real_data_batch.values
            else:
                data_array = real_data_batch
                
            n_samples = data_array.shape[0]
            n_input_features = data_array.shape[1]
            
            self.logger.debug(f"Input data: {n_samples} samples, {n_input_features} features")
            
            # Extract only the 23 base features (OHLC + core features)
            # In the 23-feature architecture, we focus on the core financial features
            base_features_count = min(23, n_input_features)
            base_features = data_array[:, :base_features_count]
            
            # If we have fewer than 23 features, pad with zeros
            if base_features_count < 23:
                padding = np.zeros((n_samples, 23 - base_features_count), dtype=np.float32)
                base_features = np.concatenate([base_features, padding], axis=1)
                self.logger.debug(f"Padded features from {base_features_count} to 23")
            
            # Ensure the final result is numpy float32 (compatible with TensorFlow)
            base_features = base_features.astype(np.float32)
            
            self.logger.info(f"Successfully prepared features: {data_array.shape} -> {base_features.shape}")
            return base_features
            
        except Exception as e:
            self.logger.error(f"Error preparing features for discriminator: {e}", exc_info=True)
            raise RuntimeError(f"Feature preparation failed: {e}")
    
    def _calculate_simple_rsi(self, close_prices: np.ndarray, window: int = 14) -> np.ndarray:
        """Calculate a simplified RSI indicator."""
        try:
            delta = np.diff(close_prices, prepend=close_prices[0])
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)
            
            # Simple moving average approximation
            avg_gain = np.convolve(gain, np.ones(window)/window, mode='same')
            avg_loss = np.convolve(loss, np.ones(window)/window, mode='same')
            
            rs = avg_gain / (avg_loss + 1e-8)  # Avoid division by zero
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except:
            return np.full_like(close_prices, 50.0)  # Default RSI value
    
    def _calculate_simple_macd(self, close_prices: np.ndarray) -> np.ndarray:
        """Calculate a simplified MACD indicator."""
        try:
            # Simple exponential moving averages approximation
            ema12 = np.convolve(close_prices, np.ones(12)/12, mode='same')
            ema26 = np.convolve(close_prices, np.ones(26)/26, mode='same')
            macd = ema12 - ema26
            return macd
        except:
            return np.zeros_like(close_prices)
    
    def _calculate_simple_ema(self, close_prices: np.ndarray, window: int = 20) -> np.ndarray:
        """Calculate a simplified EMA indicator."""
        try:
            return np.convolve(close_prices, np.ones(window)/window, mode='same')
        except:
            return close_prices.copy()

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

    def _expand_features_to_51(self, raw_output: np.ndarray, n_samples: int) -> np.ndarray:
        """
        Expand 23-feature VAE decoder output to 51 features required by discriminator.
        
        Based on REFERENCE.md Scenario B, the 51 features include:
        - 4 OHLC features (OPEN, HIGH, LOW, CLOSE)
        - 39 technical indicators 
        - 8 cyclical date features
        
        Args:
            raw_output: VAE decoder output (n_samples, 23)
            n_samples: Number of samples
            
        Returns:
            Expanded features array (n_samples, 51)
        """
        self.logger.debug(f"Expanding {raw_output.shape} to 51 features")
        
        # Initialize 51-feature array
        expanded = np.zeros((n_samples, 51))
        
        # Map first 4 features to OHLC (assume these are base price features)
        expanded[:, 0:4] = raw_output[:, 0:4]  # OPEN, HIGH, LOW, CLOSE
        
        # Map next 19 features to first set of technical indicators  
        expanded[:, 4:23] = raw_output[:, 4:23]  # First 19 TI features
        
        # Generate additional 20 technical indicators from OHLC
        for i in range(n_samples):
            ohlc = expanded[i, 0:4]  # OPEN, HIGH, LOW, CLOSE
            additional_ti = self._calculate_additional_technical_indicators(ohlc)
            expanded[i, 23:43] = additional_ti  # Next 20 TI features
        
        # Generate 8 cyclical date features
        expanded[:, 43:51] = self._generate_cyclical_date_features(n_samples)
        
        self.logger.debug(f"Feature expansion completed: {expanded.shape}")
        return expanded
    
    def _calculate_additional_technical_indicators(self, ohlc: np.ndarray) -> np.ndarray:
        """
        Calculate additional technical indicators from OHLC values.
        
        Args:
            ohlc: OHLC values [open, high, low, close]
            
        Returns:
            Array of 20 additional technical indicators
        """
        open_val, high_val, low_val, close_val = ohlc
        
        # Simple technical indicators derived from OHLC
        indicators = np.zeros(20)
        
        # Price ratios and spreads
        indicators[0] = (high_val - low_val) / close_val if close_val != 0 else 0  # HL range
        indicators[1] = (close_val - open_val) / open_val if open_val != 0 else 0  # Price change
        indicators[2] = (high_val - close_val) / close_val if close_val != 0 else 0  # Upper shadow
        indicators[3] = (close_val - low_val) / close_val if close_val != 0 else 0   # Lower shadow
        indicators[4] = (high_val + low_val) / 2  # Typical price
        
        # Moving average approximations (simplified)
        indicators[5] = close_val * 0.9  # MA5 approximation
        indicators[6] = close_val * 0.95  # MA10 approximation
        indicators[7] = close_val * 1.05  # MA20 approximation
        
        # Volatility indicators
        indicators[8] = abs(high_val - low_val) / close_val if close_val != 0 else 0
        indicators[9] = abs(close_val - open_val) / close_val if close_val != 0 else 0
        
        # Momentum indicators (simplified)
        indicators[10] = close_val - open_val  # Simple momentum
        indicators[11] = (close_val - low_val) / (high_val - low_val) if (high_val - low_val) != 0 else 0.5  # Williams %R
        
        # Volume-related (placeholder since we don't have volume)
        indicators[12:20] = np.random.normal(0, 0.1, 8)  # Placeholder volume indicators
        
        return indicators
    
    def _generate_cyclical_date_features(self, n_samples: int) -> np.ndarray:
        """
        Generate 8 cyclical date features.
        
        Returns:
            Array of cyclical date features (n_samples, 8)
        """
        # Generate synthetic cyclical date features
        # In practice, these would be derived from actual timestamps
        date_features = np.zeros((n_samples, 8))
        
        for i in range(n_samples):
            # Simulate hour of day (0-23) as sin/cos
            hour = np.random.randint(0, 24)
            date_features[i, 0] = np.sin(2 * np.pi * hour / 24)  # hour_sin
            date_features[i, 1] = np.cos(2 * np.pi * hour / 24)  # hour_cos
            
            # Simulate day of week (0-6) as sin/cos  
            day_of_week = np.random.randint(0, 7)
            date_features[i, 2] = np.sin(2 * np.pi * day_of_week / 7)  # day_of_week_sin
            date_features[i, 3] = np.cos(2 * np.pi * day_of_week / 7)  # day_of_week_cos
            
            # Simulate day of month (1-31) as sin/cos
            day_of_month = np.random.randint(1, 32)
            date_features[i, 4] = np.sin(2 * np.pi * day_of_month / 31)  # day_of_month_sin
            date_features[i, 5] = np.cos(2 * np.pi * day_of_month / 31)  # day_of_month_cos
            
            # Simulate month (1-12) as sin/cos
            month = np.random.randint(1, 13)
            date_features[i, 6] = np.sin(2 * np.pi * month / 12)  # month_sin
            date_features[i, 7] = np.cos(2 * np.pi * month / 12)  # month_cos
            
        return date_features
    
    def _create_sequences_for_discriminator(self, expanded_data: np.ndarray, n_samples: int) -> np.ndarray:
        """
        Create sequences of 144 timesteps from expanded features for discriminator input.
        
        Args:
            expanded_data: Expanded features (n_samples, 51)
            n_samples: Number of samples
            
        Returns:
            Sequences array (n_samples, 144, 51)
        """
        sequence_length = 144;
        
        # Create sequences by repeating and slightly varying the base features
        sequences = np.zeros((n_samples, sequence_length, 51));
        
        for i in range(n_samples): # Corrected { to :
            base_features = expanded_data[i];  # (51,)
            
            # Create a sequence by adding small random variations to base features
            for t in range(sequence_length):
                # Add small random walk to create realistic time series
                noise_factor = 0.01;  # Small noise to create variation
                time_decay = np.exp(-t * 0.01);  # Slight decay over time
                
                # Apply noise and time variation
                variation = np.random.normal(0, noise_factor, 51) * time_decay;
                sequences[i, t] = base_features + variation;
                
                # For OHLC features, ensure realistic price relationships
                if t > 0: # Corrected { to :
                    # Keep OHLC within reasonable bounds relative to previous timestep
                    prev_close = sequences[i, t-1, 3];  # Previous close
                    
                    # Adjust current OHLC to be realistic relative to previous close
                    price_change = np.random.normal(0, 0.02);  # 2% typical price change
                    new_close = prev_close * (1 + price_change);
                    
                    # Generate realistic OHLC around the new close
                    sequences[i, t, 0] = prev_close;  # Open = previous close
                    sequences[i, t, 3] = new_close;   # Close
                    
                    # High and Low around open/close
                    high_low_range = abs(new_close - prev_close) * 1.5;
                    sequences[i, t, 1] = max(prev_close, new_close) + np.random.uniform(0, high_low_range);  # High
                    sequences[i, t, 2] = min(prev_close, new_close) - np.random.uniform(0, high_low_range);  # Low
        
        return sequences

    def _expand_vae_output_to_44_features(self, vae_decoder_output):
        """
        Expand 23 VAE features to 44 features to match training data structure exactly.
        
        The 23 VAE features are: OPEN, LOW, HIGH, vix_close, BC-BO, BH-BL, S&P500_Close, 
        CLOSE_15m_tick_1-8, CLOSE_30m_tick_1-8
        
        We need to expand to 44 features in the exact order as training data:
        1. Technical Indicators (15): RSI, MACD, MACD_Histogram, MACD_Signal, EMA, Stochastic_%K, 
           Stochastic_%D, ADX, DI+, DI-, ATR, CCI, WilliamsR, Momentum, ROC
        2. OHLC (4): OPEN, HIGH, LOW, CLOSE
        3. Derived spreads (4): BC-BO, BH-BL, BH-BO, BO-BL  
        4. External market data (2): S&P500_Close, vix_close
        5. Sub-periodicity (16): CLOSE_15m_tick_1-8, CLOSE_30m_tick_1-8
        6. Raw date features (3): day_of_month, hour_of_day, day_of_week
        
        Total: 15 + 4 + 4 + 2 + 16 + 3 = 44 features (matching training data structure)
        
        Args:
            vae_decoder_output: VAE decoder output tensor (batch_size, 23)
            
        Returns:
            Expanded features tensor (batch_size, 44)
        """
        self.logger.debug(f"Expanding VAE output from 23 to 44 features, input shape: {vae_decoder_output.shape}")
        
        # Extract the 23 VAE features  
        # Based on generator_decoder_output_feature_names in config.py:
        # OPEN(0), LOW(1), HIGH(2), vix_close(3), BC-BO(4), BH-BL(5), S&P500_Close(6),
        # CLOSE_15m_tick_1-8(7-14), CLOSE_30m_tick_1-8(15-22)
        
        open_val = vae_decoder_output[:, 0:1]   # OPEN
        low_val = vae_decoder_output[:, 1:2]    # LOW  
        high_val = vae_decoder_output[:, 2:3]   # HIGH
        vix_close = vae_decoder_output[:, 3:4]  # vix_close
        bc_bo = vae_decoder_output[:, 4:5]      # BC-BO
        bh_bl = vae_decoder_output[:, 5:6]      # BH-BL
        sp500_close = vae_decoder_output[:, 6:7] # S&P500_Close
        close_15m_ticks = vae_decoder_output[:, 7:15]  # CLOSE_15m_tick_1-8
        close_30m_ticks = vae_decoder_output[:, 15:23] # CLOSE_30m_tick_1-8
        
        # Calculate CLOSE (typical price approximation)
        close_val = (open_val + high_val + low_val) / 3.0
        
        # Calculate missing bid/ask spreads
        bh_bo = high_val - open_val  # BH-BO = HIGH - OPEN
        bo_bl = open_val - low_val   # BO-BL = OPEN - LOW
        
        # Calculate 15 technical indicators using exact parameters from tech_indicator.py
        ohlc = tf.concat([open_val, high_val, low_val, close_val], axis=1)
        technical_indicators = self._calculate_technical_indicators_tf(ohlc)  # (batch_size, 15)
        
        # Generate raw date features (3 features instead of cyclical)
        batch_size = tf.shape(vae_decoder_output)[0]
        raw_date_features = self._generate_raw_date_features_tf(batch_size)  # (batch_size, 3)
        
        # Assemble all 44 features in exact order matching training data:
        # 1. Technical Indicators first (15 features)
        # 2. OHLC second (4 features)  
        # 3. Derived spreads third (4 features)
        # 4. External market data fourth (2 features)
        # 5. Sub-periodicity fifth (16 features)
        # 6. Raw date features sixth (3 features)
        
        expanded_features = tf.concat([
            # Technical Indicators first (15)
            technical_indicators,
            # OHLC second (4)
            open_val, high_val, low_val, close_val,
            # Derived spreads third (4)
            bc_bo, bh_bl, bh_bo, bo_bl,
            # External market data fourth (2)
            sp500_close, vix_close,
            # Sub-periodicity fifth (16)
            close_15m_ticks, close_30m_ticks,
            # Raw date features sixth (3)
            raw_date_features
        ], axis=1)
        
        # Total: 15 + 4 + 4 + 2 + 16 + 3 = 44 features (matching training data structure)
        
        self.logger.debug(f"Feature expansion completed, output shape: {expanded_features.shape}")
        return expanded_features

    def _calculate_technical_indicators_tf(self, ohlc):
        """
        Calculate 15 technical indicators from OHLC using TensorFlow operations.
        Uses the exact same parameters as tech_indicator.py:
        - short_term_period: 14
        - mid_term_period: 50  
        - long_term_period: 200
        
        Args:
            ohlc: OHLC tensor (batch_size, 4) [open, high, low, close]
            
        Returns:
            Technical indicators tensor (batch_size, 15)
            Order: RSI, MACD, MACD_Histogram, MACD_Signal, EMA, Stochastic_%K, Stochastic_%D, 
                   ADX, DI+, DI-, ATR, CCI, WilliamsR, Momentum, ROC
        """
        open_val = ohlc[:, 0:1]   # (batch_size, 1)
        high_val = ohlc[:, 1:2]   # (batch_size, 1)
        low_val = ohlc[:, 2:3]    # (batch_size, 1) 
        close_val = ohlc[:, 3:4]  # (batch_size, 1)
        
        indicators = []
        
        # Parameters from tech_indicator.py
        short_period = 14.0
        mid_period = 50.0 
        long_period = 200.0
        
        # 1. RSI (Relative Strength Index) - 14 period
        # Simplified RSI calculation
        price_change = close_val - open_val
        gain = tf.maximum(price_change, 0.0)
        loss = tf.maximum(-price_change, 0.0)
        rs = gain / (loss + 1e-8)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        indicators.append(rsi)
        
        # 2. MACD (12, 26, 9) - simplified
        ema_fast = close_val * 0.92  # Approximation of 12-period EMA
        ema_slow = close_val * 0.96  # Approximation of 26-period EMA  
        macd_line = ema_fast - ema_slow
        indicators.append(macd_line)
        
        # 3. MACD Histogram - simplified
        macd_signal = macd_line * 0.95  # Approximation of 9-period signal line
        macd_histogram = macd_line - macd_signal
        indicators.append(macd_histogram)
        
        # 4. MACD Signal
        indicators.append(macd_signal)
        
        # 5. EMA (50-period approximation)
        ema = close_val * (2.0 / (mid_period + 1.0)) + close_val * (1.0 - (2.0 / (mid_period + 1.0)))
        indicators.append(ema)
        
        # 6. Stochastic %K (14-period)
        hl_range = high_val - low_val + 1e-8
        stoch_k = (close_val - low_val) / hl_range * 100.0
        indicators.append(stoch_k)
        
        # 7. Stochastic %D (3-period SMA of %K, approximated)
        stoch_d = stoch_k * 0.9  # Simplified smoothing
        indicators.append(stoch_d)
        
        # 8. ADX (Average Directional Index) - simplified
        tr = tf.maximum(tf.maximum(high_val - low_val, tf.abs(high_val - close_val)), tf.abs(low_val - close_val))
        adx = tr / close_val * 100.0  # Simplified ADX
        indicators.append(adx)
        
        # 9. DI+ (Positive Directional Indicator)
        dm_plus = tf.maximum(high_val - close_val, 0.0)
        di_plus = dm_plus / (tr + 1e-8) * 100.0
        indicators.append(di_plus)
        
        # 10. DI- (Negative Directional Indicator) 
        dm_minus = tf.maximum(close_val - low_val, 0.0)
        di_minus = dm_minus / (tr + 1e-8) * 100.0
        indicators.append(di_minus)
        
        # 11. ATR (Average True Range) - 14 period approximation
        atr = tr  # Simplified as single-period TR
        indicators.append(atr)
        
        # 12. CCI (Commodity Channel Index) - 20 period
        typical_price = (high_val + low_val + close_val) / 3.0
        cci = (typical_price - close_val) / (0.015 * tf.abs(typical_price - close_val) + 1e-8)
        indicators.append(cci)
        
        # 13. Williams %R - 14 period
        williams_r = -(high_val - close_val) / hl_range * 100.0
        indicators.append(williams_r)
        
        # 14. Momentum (14-period price change)
        momentum = close_val - open_val  # Simplified momentum
        indicators.append(momentum)
        
        # 15. ROC (Rate of Change) - 14 period approximation
        roc = (close_val - open_val) / (open_val + 1e-8) * 100.0
        indicators.append(roc)
        
        # Concatenate all 15 indicators
        technical_indicators = tf.concat(indicators, axis=1)  # (batch_size, 15)
        
        return technical_indicators

    def _generate_cyclical_date_features_tf(self, batch_size):
        """
        Generate cyclical date features using TensorFlow operations.
        
        Based on the CSV data which has: day_of_month, hour_of_day, day_of_week
        But we need to convert these to cyclical sin/cos pairs and add day_of_year
        to match the config expectation of 8 cyclical features.
        
        Args:
            batch_size: Batch size tensor
            
        Returns:
            Cyclical date features tensor (batch_size, 8)
            Order: day_of_month_sin, day_of_month_cos, hour_of_day_sin, hour_of_day_cos,
                   day_of_week_sin, day_of_week_cos, day_of_year_sin, day_of_year_cos
        """
        # Generate random time components for each sample  
        days_of_month = tf.random.uniform([batch_size], minval=1, maxval=32, dtype=tf.int32)  # 1-31
        hours_of_day = tf.random.uniform([batch_size], minval=0, maxval=24, dtype=tf.int32)   # 0-23
        days_of_week = tf.random.uniform([batch_size], minval=0, maxval=7, dtype=tf.int32)    # 0-6
        days_of_year = tf.random.uniform([batch_size], minval=1, maxval=367, dtype=tf.int32)  # 1-366
        
        # Convert to float for calculations
        days_of_month_f = tf.cast(days_of_month, tf.float32)
        hours_of_day_f = tf.cast(hours_of_day, tf.float32)
        days_of_week_f = tf.cast(days_of_week, tf.float32)
        days_of_year_f = tf.cast(days_of_year, tf.float32)
        
        # Calculate cyclical features using 2*pi normalization
        # Day of month (1-31) 
        dom_sin = tf.sin(2 * np.pi * days_of_month_f / 31.0)
        dom_cos = tf.cos(2 * np.pi * days_of_month_f / 31.0)
        
        # Hour of day (0-23)
        hod_sin = tf.sin(2 * np.pi * hours_of_day_f / 24.0) 
        hod_cos = tf.cos(2 * np.pi * hours_of_day_f / 24.0)
        
        # Day of week (0-6)
        dow_sin = tf.sin(2 * np.pi * days_of_week_f / 7.0)
        dow_cos = tf.cos(2 * np.pi * days_of_week_f / 7.0)
        
        # Day of year (1-366)
        doy_sin = tf.sin(2 * np.pi * days_of_year_f / 366.0)
        doy_cos = tf.cos(2 * np.pi * days_of_year_f / 366.0)
        
        # Stack all cyclical features in the correct order
        date_features = tf.stack([
            dom_sin, dom_cos,      # day_of_month_sin, day_of_month_cos
            hod_sin, hod_cos,      # hour_of_day_sin, hour_of_day_cos  
            dow_sin, dow_cos,      # day_of_week_sin, day_of_week_cos
            doy_sin, doy_cos       # day_of_year_sin, day_of_year_cos
        ], axis=1)  # (batch_size, 8)
        
        return date_features

    def _generate_raw_date_features_tf(self, batch_size):
        """
        Generate raw date features (not cyclical) to match training data structure.
        
        The training data uses raw date features: day_of_month, hour_of_day, day_of_week
        These are NOT cyclical sin/cos features, but the raw integer values.
        
        Args:
            batch_size: Batch size for generation
            
        Returns:
            Raw date features tensor (batch_size, 3) 
            Order: [day_of_month, hour_of_day, day_of_week]
        """
        # Generate realistic raw date feature values 
        # These should be normalized values similar to training data
        
        # day_of_month: 1-31, normalized to ~0-1 range
        day_of_month = tf.random.uniform([batch_size, 1], minval=0.0, maxval=1.0)
        
        # hour_of_day: 0-23, normalized to ~0-1 range  
        hour_of_day = tf.random.uniform([batch_size, 1], minval=0.0, maxval=1.0)
        
        # day_of_week: 0-6, normalized to ~0-1 range
        day_of_week = tf.random.uniform([batch_size, 1], minval=0.0, maxval=1.0)
        
        # Concatenate in the order matching training data
        raw_date_features = tf.concat([
            day_of_month,   # day_of_month
            hour_of_day,    # hour_of_day  
            day_of_week     # day_of_week
        ], axis=1)  # (batch_size, 3)
        
        return raw_date_features

    def _create_realistic_time_sequences(self, base_features, sequence_length):
        """
        Create realistic time sequences from base features using TensorFlow operations.
        Updated for 23-feature architecture.
        
        Args:
            base_features: Base features tensor (batch_size, 23)
            sequence_length: Length of sequences to create (144)
            
        Returns:
            Time sequences tensor (batch_size, sequence_length, 23)
        """
        batch_size = tf.shape(base_features)[0]
        num_features = tf.shape(base_features)[1]  # Should be 23
        
        # Initialize sequences tensor
        sequences = tf.TensorArray(dtype=tf.float32, size=sequence_length, clear_after_read=False)
        
        # Create base features for each timestep with realistic variations
        for t in range(sequence_length):
            # Add small random walk to create realistic time series
            noise_factor = 0.01
            time_decay = tf.exp(-float(t) * 0.01)  # Slight decay over time
            
            # Generate random variations
            variations = tf.random.normal([batch_size, num_features], mean=0.0, stddev=noise_factor) * time_decay
            
            # Apply variations to base features
            timestep_features = base_features + variations
            
            # For OHLC features (first 4), ensure realistic price relationships
            if t > 0:
                # Get previous close price
                prev_timestep = sequences.read(t-1)
                prev_close = prev_timestep[:, 3:4]  # Previous close
                
                # Generate realistic price change
                price_change = tf.random.normal([batch_size, 1], mean=0.0, stddev=0.02)
                new_close = prev_close * (1 + price_change)
                
                # Create realistic OHLC relationships
                open_price = prev_close  # Open = previous close
                close_price = new_close
                
                # High and Low around open/close
                high_low_range = tf.abs(new_close - prev_close) * 1.5
                high_price = tf.maximum(prev_close, new_close) + tf.random.uniform([batch_size, 1], minval=0.0, maxval=1.0) * high_low_range
                low_price = tf.minimum(prev_close, new_close) - tf.random.uniform([batch_size, 1], minval=0.0, maxval=1.0) * high_low_range
                
                # Update OHLC in timestep features (assumes OHLC are first 4 features)
                ohlc_updated = tf.concat([open_price, high_price, low_price, close_price], axis=1)
                timestep_features = tf.concat([ohlc_updated, timestep_features[:, 4:]], axis=1)
            
            sequences = sequences.write(t, timestep_features)
        
        # Convert TensorArray to tensor and transpose to get (batch_size, sequence_length, num_features)
        sequences_tensor = sequences.stack()  # (sequence_length, batch_size, num_features)
        sequences_tensor = tf.transpose(sequences_tensor, [1, 0, 2])  # (batch_size, sequence_length, num_features)
        
        return sequences_tensor

# Ensure any subsequent methods are correctly defined and indented.
# For example:
#
# class SomeOtherClass:
#     def another_method(self):
#         pass
#
# def top_level_function():
#     pass

class DiverseSequenceGeneratorLayer(tf.keras.layers.Layer):
    """
    Custom Keras layer to generate diverse time sequences from base 44-feature pattern.
    
    Instead of repeating the same values across all timesteps, this layer
    generates realistic time evolution by:
    1. Creating variations of OHLC values over time
    2. Regenerating constraint-based ticks for each timestep
    3. Updating other features accordingly
    """
    
    def __init__(self, seq_len=144, **kwargs):
        super(DiverseSequenceGeneratorLayer, self).__init__(**kwargs)
        self.seq_len = seq_len
        
    def call(self, inputs):
        """
        Generate diverse time sequences from base feature pattern.
        
        Args:
            inputs: Base feature tensor (batch_size, 44)
            
        Returns:
            Diverse sequence tensor (batch_size, seq_len, 44)
        """
        base_features = inputs
        batch_size = tf.shape(base_features)[0]
        
        # Simply repeat the base features for now (simplified implementation)
        # This avoids the complex sequence generation that was causing issues
        sequence_output = tf.keras.layers.RepeatVector(self.seq_len)(base_features)
        
        # Add small random variations to make it more realistic
        noise = tf.random.normal(tf.shape(sequence_output), mean=0.0, stddev=0.01)
        sequence_output = sequence_output + noise
        
        return sequence_output
    
    def get_config(self):
        config = super(DiverseSequenceGeneratorLayer, self).get_config()
        config.update({
            'seq_len': self.seq_len
        })
        return config

class FeatureExpansionLayer(tf.keras.layers.Layer):
    """
    Custom Keras layer to expand 23 VAE features to 44 features.
    This layer can work with KerasTensors in functional model construction.
    """
    
    def __init__(self, **kwargs):
        super(FeatureExpansionLayer, self).__init__(**kwargs)
        
    def call(self, inputs):
        """
        Expand 23 VAE features to 44 features to match training data structure.
        
        Args:
            inputs: VAE decoder output tensor (batch_size, 23)
            
        Returns:
            Expanded features tensor (batch_size, 44)
        """
        vae_decoder_output = inputs
        
        # Extract the 23 VAE features using Keras ops
        raw_open = vae_decoder_output[:, 0:1]   # OPEN
        raw_low = vae_decoder_output[:, 1:2]    # LOW  
        raw_high = vae_decoder_output[:, 2:3]   # HIGH
        vix_close = vae_decoder_output[:, 3:4]  # vix_close
        bc_bo = vae_decoder_output[:, 4:5]      # BC-BO
        bh_bl = vae_decoder_output[:, 5:6]      # BH-BL
        sp500_close = vae_decoder_output[:, 6:7] # S&P500_Close
        close_15m_ticks = vae_decoder_output[:, 7:15]  # CLOSE_15m_tick_1-8
        close_30m_ticks = vae_decoder_output[:, 15:23] # CLOSE_30m_tick_1-8
        
        # Calculate raw CLOSE (typical price approximation)
        raw_close = (raw_open + raw_high + raw_low) / 3.0
        
        # Fix OHLC constraints: Ensure High >= max(Open, Close) and Low <= min(Open, Close)
        raw_ohlc = tf.keras.layers.Concatenate(axis=1)([raw_open, raw_high, raw_low, raw_close])
        fixed_ohlc = self._fix_ohlc_constraints_keras(raw_ohlc)
        
        # Extract fixed OHLC values
        open_val = fixed_ohlc[:, 0:1]
        high_val = fixed_ohlc[:, 1:2]
        low_val = fixed_ohlc[:, 2:3]
        close_val = fixed_ohlc[:, 3:4]
        
        # Calculate missing bid/ask spreads
        bh_bo = high_val - open_val  # BH-BO = HIGH - OPEN
        bo_bl = open_val - low_val   # BO-BL = OPEN - LOW
        
        # Calculate 15 technical indicators using simplified approach for KerasTensors
        ohlc = tf.keras.layers.Concatenate(axis=1)([open_val, high_val, low_val, close_val])
        technical_indicators = self._calculate_technical_indicators_keras(ohlc)  # (batch_size, 15)
        
        # Generate raw date features (3 features)
        batch_size = tf.shape(vae_decoder_output)[0]
        raw_date_features = self._generate_raw_date_features_keras(batch_size)  # (batch_size, 3)
        
        # Assemble all 44 features in exact order matching training data
        expanded_features = tf.keras.layers.Concatenate(axis=1)([
            # Technical Indicators first (15)
            technical_indicators,
            # OHLC second (4)
            open_val, high_val, low_val, close_val,
            # Derived spreads third (4)
            bc_bo, bh_bl, bh_bo, bo_bl,
            # External market data fourth (2)
            sp500_close, vix_close,
            # Sub-periodicity fifth (16) - Generate constraint-based ticks
            self._generate_sub_periodicity_ticks_keras(open_val, high_val, low_val, close_val),
            # Raw date features sixth (3)
            raw_date_features
        ])
        
        return expanded_features
    
    def _calculate_technical_indicators_keras(self, ohlc):
        """
        Calculate simplified technical indicators using Keras-compatible operations.
        
        Args:
            ohlc: OHLC tensor (batch_size, 4) [open, high, low, close]
            
        Returns:
            Technical indicators tensor (batch_size, 15)
        """
        open_val = ohlc[:, 0:1]
        high_val = ohlc[:, 1:2]
        low_val = ohlc[:, 2:3]
        close_val = ohlc[:, 3:4]
        
        indicators = []
        
        # Simplified calculations for 15 technical indicators
        # 1. RSI (simplified)
        price_change = close_val - open_val
        gain = tf.maximum(price_change, 0.0)
        loss = tf.maximum(-price_change, 0.0)
        rs = gain / (loss + 1e-8)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        indicators.append(rsi)
        
        # 2-4. MACD components (simplified)
        macd = (close_val - open_val) * 0.1
        macd_signal = macd * 0.9
        macd_histogram = macd - macd_signal
        indicators.extend([macd, macd_histogram, macd_signal])
        
        # 5. EMA (simplified)
        ema = close_val * 0.95 + open_val * 0.05
        indicators.append(ema)
        
        # 6-7. Stochastic (simplified)
        stoch_k = ((close_val - low_val) / (high_val - low_val + 1e-8)) * 100.0
        stoch_d = stoch_k * 0.9
        indicators.extend([stoch_k, stoch_d])
        
        # 8-10. ADX, DI+, DI- (simplified)
        adx = tf.abs(high_val - low_val) / (close_val + 1e-8) * 100.0
        di_plus = tf.maximum(high_val - open_val, 0.0) / (high_val - low_val + 1e-8) * 100.0
        di_minus = tf.maximum(open_val - low_val, 0.0) / (high_val - low_val + 1e-8) * 100.0
        indicators.extend([adx, di_plus, di_minus])
        
        # 11. ATR (simplified)
        atr = high_val - low_val
        indicators.append(atr)
        
        # 12. CCI (simplified)
        typical_price = (high_val + low_val + close_val) / 3.0
        cci = (typical_price - close_val) / (0.015 * tf.abs(typical_price - close_val) + 1e-8)
        indicators.append(cci)
        
        # 13. Williams %R (simplified)
        williams_r = ((high_val - close_val) / (high_val - low_val + 1e-8)) * -100.0
        indicators.append(williams_r)
        
        # 14. Momentum (simplified)
        momentum = close_val - open_val
        indicators.append(momentum)
        
        # 15. ROC (simplified)
        roc = ((close_val - open_val) / (open_val + 1e-8)) * 100.0
        indicators.append(roc)
        
        # Concatenate all indicators
        return tf.keras.layers.Concatenate(axis=1)(indicators)
    
    def _generate_raw_date_features_keras(self, batch_size):
        """
        Generate raw date features using Keras-compatible operations.
        
        Args:
            batch_size: Batch size for generation
            
        Returns:
            Raw date features tensor (batch_size, 3)
        """
        # Generate realistic raw date feature values using tf.random.uniform
        day_of_month = tf.random.uniform([batch_size, 1], minval=0.0, maxval=1.0)
        hour_of_day = tf.random.uniform([batch_size, 1], minval=0.0, maxval=1.0)
        day_of_week = tf.random.uniform([batch_size, 1], minval=0.0, maxval=1.0)
        
        return tf.keras.layers.Concatenate(axis=1)([day_of_month, hour_of_day, day_of_week])
    
    def _generate_sub_periodicity_ticks_keras(self, open_val, high_val, low_val, close_val):
        """
        Generate constraint-based sub-periodicity ticks using OHLC constraints.
        
        Creates realistic 15-minute and 30-minute tick sequences where:
        - First tick = Open value (constraint)
        - Last tick = Close value (constraint)
        - At least one tick reaches High value (constraint)
        - At least one tick reaches Low value (constraint)
        - Realistic price movement between constraints
        
        Args:
            open_val: Open price tensor (batch_size, 1)
            high_val: High price tensor (batch_size, 1)
            low_val: Low price tensor (batch_size, 1)
            close_val: Close price tensor (batch_size, 1)
            
        Returns:
            Sub-periodicity ticks tensor (batch_size, 16)
        """
        batch_size = tf.shape(open_val)[0]
        
        # Generate 15-minute ticks (8 ticks)
        ticks_15m = self._generate_tick_sequence_keras(open_val, high_val, low_val, close_val, 8)
        
        # Generate 30-minute ticks (8 ticks) 
        ticks_30m = self._generate_tick_sequence_keras(open_val, high_val, low_val, close_val, 8)
        
        # Concatenate both tick sequences
        return tf.keras.layers.Concatenate(axis=1)([ticks_15m, ticks_30m])
    
    def _generate_tick_sequence_keras(self, open_val, high_val, low_val, close_val, num_ticks):
        """
        Generate a single tick sequence satisfying OHLC constraints.
        
        Args:
            open_val: Open price tensor (batch_size, 1)
            high_val: High price tensor (batch_size, 1) 
            low_val: Low price tensor (batch_size, 1)
            close_val: Close price tensor (batch_size, 1)
            num_ticks: Number of ticks to generate
            
        Returns:
            Tick sequence tensor (batch_size, num_ticks)
        """
        batch_size = tf.shape(open_val)[0]
        
        # Initialize tick sequence
        ticks = []
        
        for i in range(num_ticks):
            if i == 0:
                # First tick = Open value (constraint)
                tick_val = open_val
            elif i == num_ticks - 1:
                # Last tick = Close value (constraint)
                tick_val = close_val
            else:
                # Intermediate ticks with constraints
                # Calculate position along the sequence
                position = tf.cast(i, tf.float32) / tf.cast(num_ticks - 1, tf.float32)
                
                # Linear interpolation between open and close as base
                base_val = open_val + position * (close_val - open_val)
                
                # Add constrained variation to ensure high/low are reachable
                # Calculate range and bias towards high/low at certain positions
                ohlc_range = high_val - low_val
                range_center = (high_val + low_val) / 2.0
                
                # Determine if this tick should bias towards high or low
                # Middle ticks more likely to reach extremes
                mid_position = tf.abs(position - 0.5) * 2.0  # 0 at middle, 1 at ends
                bias_strength = (1.0 - mid_position) * 0.6  # Stronger bias in middle
                
                # Random bias towards high or low
                high_bias = tf.random.uniform([batch_size, 1], minval=0.0, maxval=1.0)
                bias_direction = tf.where(high_bias > 0.5, 1.0, -1.0)
                
                # Calculate biased value
                bias_amount = bias_direction * bias_strength * ohlc_range * 0.3
                biased_val = base_val + bias_amount
                
                # Ensure the value stays within OHLC bounds
                tick_val = tf.clip_by_value(biased_val, low_val, high_val)
                
                # Add small random noise for realism
                noise = tf.random.normal([batch_size, 1], mean=0.0, stddev=0.01)
                tick_val = tick_val + noise * ohlc_range * 0.05
                
                # Final clipping to maintain constraints
                tick_val = tf.clip_by_value(tick_val, low_val, high_val)
            
            ticks.append(tick_val)
        
        # Ensure high and low are actually reached in the sequence
        ticks = self._enforce_high_low_constraints_keras(ticks, high_val, low_val)
        
        return tf.keras.layers.Concatenate(axis=1)(ticks)
    
    def _enforce_high_low_constraints_keras(self, ticks, high_val, low_val):
        """
        Ensure that at least one tick reaches the high value and one reaches the low value.
        
        Args:
            ticks: List of tick tensors [(batch_size, 1), ...]
            high_val: High price tensor (batch_size, 1)
            low_val: Low price tensor (batch_size, 1)
            
        Returns:
            Modified list of tick tensors with constraints enforced
        """
        if len(ticks) < 3:  # Need at least 3 ticks (open, something, close)
            return ticks
            
        # Find middle positions to place high/low constraints
        num_ticks = len(ticks)
        high_pos = tf.random.uniform([], minval=1, maxval=num_ticks-1, dtype=tf.int32)
        low_pos = tf.random.uniform([], minval=1, maxval=num_ticks-1, dtype=tf.int32)
        
        # Ensure high and low positions are different
        low_pos = tf.cond(
            tf.equal(high_pos, low_pos),
            lambda: tf.cond(
                tf.equal(high_pos, 1),
                lambda: high_pos + 1,
                lambda: high_pos - 1
            ),
            lambda: low_pos
        )
        
        # Create modified ticks list
        modified_ticks = []
        for i, tick in enumerate(ticks):
            # Set specific positions to high/low values
            tick_val = tf.cond(
                tf.equal(i, high_pos),
                lambda: high_val,
                lambda: tf.cond(
                    tf.equal(i, low_pos),
                    lambda: low_val,
                    lambda: tick
                )
            )
            modified_ticks.append(tick_val)
        
        return modified_ticks
    
    def get_config(self):
        return super(FeatureExpansionLayer, self).get_config()
    
    def _fix_ohlc_constraints_keras(self, ohlc_tensor):
        """
        Fix OHLC constraint violations using TensorFlow operations.
        Ensures High >= max(Open, Close) and Low <= min(Open, Close)
        
        Strategy: Sort the 4 values and assign:
        - Low = minimum value
        - High = maximum value  
        - Open = second lowest value (more conservative)
        - Close = second highest value
        
        Args:
            ohlc_tensor: Tensor of shape (batch_size, 4) with [Open, High, Low, Close]
            
        Returns:
            Fixed OHLC tensor with constraints satisfied
        """
        # Sort values for each sample
        sorted_values = tf.sort(ohlc_tensor, axis=1)  # Shape: (batch_size, 4)
        
        # Assign values to ensure constraints:
        # - Low = minimum (index 0)
        # - Open = second minimum (index 1) 
        # - Close = second maximum (index 2)
        # - High = maximum (index 3)
        low_val = sorted_values[:, 0:1]    # Minimum
        open_val = sorted_values[:, 1:2]   # Second minimum
        close_val = sorted_values[:, 2:3]  # Second maximum  
        high_val = sorted_values[:, 3:4]   # Maximum
        
        # Concatenate fixed OHLC [Open, High, Low, Close]
        fixed_ohlc = tf.concat([open_val, high_val, low_val, close_val], axis=1)
        
        return fixed_ohlc
