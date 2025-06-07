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
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Input, Dense, LSTM, Bidirectional, Conv1D, Reshape, Concatenate, Lambda # type: ignore

# Import modularized components
from .model_loader import ModelLoader
from .model_saver import ModelSaver # Added import for ModelSaver
from .normalization_handler import NormalizationHandler
from .initial_data_handler import InitialDataHandler
from .feature_validator import FeatureValidator
from .data_generator import DataGenerator
from .sequence_builder import SequenceBuilder
from .technical_indicator_calculator import TechnicalIndicatorCalculator


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
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the GeneratorPlugin with modular architecture.
        
        Args:
            config: Configuration dictionary
        """
        # Setup logging first
        self.logger = logging.getLogger(__name__) # Moved up to be available immediately

        if config is None:
            self.logger.error("GeneratorPlugin initialized with None config. Cannot proceed.")
            raise ValueError("Configuration cannot be None for GeneratorPlugin.")
        
        # Initialize parameters and main config
        self.params = self.plugin_params.copy()
        self.main_config = config.copy() # Store the full config
        
        # self.logger is already set up
        
        # Initialize core attributes
        self.sequential_model: Optional[Model] = None
        self.model: Optional[Model] = None  # Alias for sequential_model
        self.composite_model: Optional[Model] = None # Ensure this is initialized
        self.feature_to_idx: Dict[str, int] = {}
        self.num_all_features: int = 0
        
        # Initialize specialized modules (those not dependent on full config first)
        self._initialize_modules()
        
        # Fully set parameters and initialize feature-dependent modules using the provided config
        # This ensures ti_calculator, data_generator etc. are ready.
        # Note: set_params itself might call _initialize_feature_dependent_modules if features change.
        # We call it here to ensure modules are set up based on the initial config.
        # However, the original code calls self.set_params(**config) which then might call _initialize_feature_dependent_modules.
        # Let's stick to the original call order for set_params.
        self.set_params(**config) 
        
        # Load initial close anchor. This was potentially problematic if initial_data_handler wasn't fully set up.
        # initial_data_handler is initialized in _initialize_modules(), which is called before set_params.
        initial_close_file_path = self.main_config.get("x_train_file", self.main_config.get("real_data_file"))
        if initial_close_file_path and hasattr(self, 'initial_data_handler') and self.initial_data_handler.get_initial_close_anchor() is None:
            try:
                self.logger.info(f"Attempting to load initial close anchor from: {initial_close_file_path}")
                self.initial_data_handler.load_initial_close_anchor(initial_close_file_path)
                self.logger.info(f"Successfully loaded initial close anchor from: {initial_close_file_path}")
            except Exception as e:
                self.logger.error(f"Failed to load initial close anchor from {initial_close_file_path}: {e}", exc_info=True)
                # Depending on application requirements, this might be a critical error.
                # For now, it logs the error and continues.

    def _initialize_modules(self) -> None:
        """Initialize core modules for the generator."""
        self.logger.info("Initializing GeneratorPlugin modules...")
        
        # Model loading and saving
        self.model_loader = ModelLoader(self.params, self.logger) # Pass self.params
        self.model_saver = ModelSaver(self.logger)
        
        # Normalization (even if minimal, the handler object is expected)
        self.normalization_handler = NormalizationHandler(self.params, self.logger)
        
        # Initial data handling (for anchor close, initial window)
        self.initial_data_handler = InitialDataHandler(
            params=self.params, 
            logger=self.logger, 
            normalization_handler=self.normalization_handler
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
        print(f"GeneratorPlugin.set_params called with kwargs: {list(kwargs.keys())}") # Existing print
        
        # Store old values for change detection
        old_model_file = self.params.get("sequential_model_file")
        old_norm_file = self.params.get("generator_normalization_params_file")
        old_full_feature_names = list(self.params.get("full_feature_names_ordered", [])) # Ensure it's a list for comparison
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
        # Compare current self.params["full_feature_names_ordered"] (which might have been updated by kwargs)
        # with its old value.
        current_full_feature_names = self.params.get("full_feature_names_ordered")
        if current_full_feature_names is None: current_full_feature_names = [] # Ensure list for comparison

        if list(current_full_feature_names) != old_full_feature_names or \
           self._features_config_changed(kwargs):
            if self.params.get("full_feature_names_ordered"): # Check if there are still features configured
                self.logger.info("Feature configuration changed. Re-initializing feature-dependent modules.")
                self._initialize_feature_dependent_modules() # This will set up validator, ti_calc, etc.
                if hasattr(self, 'feature_validator') and self.feature_validator:
                    self.logger.info("Feature validator available. Running plugin configuration validation.")
                    self._validate_plugin_configuration() 
            else:
                self.logger.warning("Feature configuration resulted in empty 'full_feature_names_ordered'. Clearing feature-dependent modules.")
                self.feature_validator = None
                self.feature_to_idx = {}
                self.num_all_features = 0
                if hasattr(self, 'ti_calculator'): self.ti_calculator = None
                if hasattr(self, 'data_generator'): self.data_generator = None
                if hasattr(self, 'sequence_builder'): self.sequence_builder = None
        
        # Handle initial close anchor reload if the relevant file path changed
        new_initial_close_file_path = self.main_config.get("x_train_file", self.main_config.get("real_data_file"))
        if new_initial_close_file_path != old_initial_close_file_path and new_initial_close_file_path:
            self.logger.info(f"Initial close anchor file path changed. Reloading from: {new_initial_close_file_path}")
            if hasattr(self, 'initial_data_handler'): # Ensure handler exists
                self.initial_data_handler.load_initial_close_anchor(new_initial_close_file_path)
    
    def _features_config_changed(self, kwargs: Dict[str, Any]) -> bool:
        """
        Check if any feature-related configuration parameters, when provided in kwargs,
        lead to a change from their initial default values.
        self.params is assumed to be already updated with kwargs at this point.
        """
        feature_keys = [
            "full_feature_names_ordered", "decoder_output_feature_names",
            "ohlc_feature_names", "ti_feature_names",
            "date_conditional_feature_names", "feeder_conditional_feature_names"
        ]
        for key in feature_keys:
            param_prefix = "generator_"
            key_was_in_kwargs = False

            if f"{param_prefix}{key}" in kwargs:
                key_was_in_kwargs = True
            elif key in kwargs:
                key_was_in_kwargs = True
            
            if key_was_in_kwargs:
                # self.params.get(key) reflects the value after update from kwargs.
                # self.plugin_params.get(key) is the initial default.
                current_value = self.params.get(key)
                default_value = self.plugin_params.get(key)
                
                # Handle cases where values might be lists or other mutable types for comparison
                if isinstance(current_value, list) and isinstance(default_value, list):
                    if set(current_value) != set(default_value): # Order-agnostic for lists of simple items
                        self.logger.info(f"Feature configuration for list '{key}' changed via kwargs and differs from default.")
                        return True
                elif current_value != default_value:
                    self.logger.info(f"Feature configuration for '{key}' changed via kwargs and differs from default.")
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
        """
        if not vae_decoder_model_path:
            self.sequential_model = None
            self.model = None
            self.composite_model = None # Clear composite model as well
            self.logger.warning("GeneratorPlugin: Attempted to load VAE decoder with empty path.")
            return

        self.logger.info(f"Building composite generator with VAE decoder from: {vae_decoder_model_path}")

        try:
            loaded_vae_decoder = self.model_loader.load_model_from_path(vae_decoder_model_path)
            if loaded_vae_decoder is None:
                raise IOError(f"Failed to load VAE decoder from {vae_decoder_model_path}")
            
            loaded_vae_decoder.trainable = True
            self.logger.info(f"Loaded VAE decoder '{loaded_vae_decoder.name}'. Set trainable=True.")
            if hasattr(loaded_vae_decoder, 'inputs') and loaded_vae_decoder.inputs is not None: # Check against None
                self.logger.info(f"VAE decoder input shapes: {[inp.shape for inp in loaded_vae_decoder.inputs]}")
            if hasattr(loaded_vae_decoder, 'output') and loaded_vae_decoder.output is not None: # Check against None
                self.logger.info(f"VAE decoder output shape: {loaded_vae_decoder.output.shape}")

            # Build the composite generator model
            # _build_composite_generator will set self.composite_model
            built_model = self._build_composite_generator(loaded_vae_decoder) 
            
            if built_model is None: # Check if building failed
                self.logger.error("Failed to build composite generator model (_build_composite_generator returned None).")
                self.sequential_model = None
                self.model = None
                self.composite_model = None # Ensure it's None
                raise IOError("Composite generator could not be built.")

            self.sequential_model = built_model # For compatibility if anything still uses it
            self.model = built_model  # Alias
            # self.composite_model is already set by _build_composite_generator

            self.logger.info("Composite generator model built successfully.")
            if hasattr(self.model, 'inputs') and self.model.inputs is not None: # Check against None
                self.logger.info(f"Composite generator input shapes: {[inp.shape for inp in self.model.inputs]}")
            if hasattr(self.model, 'output') and self.model.output is not None: # Check against None
                self.logger.info(f"Composite generator output shape: {self.model.output.shape}")

        except Exception as e:
            self.logger.error(f"Error during _load_model (building composite generator): {e}", exc_info=True)
            self.sequential_model = None
            self.model = None
            self.composite_model = None # Ensure it's None on any exception
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
        self.logger.warning("No pre-loaded or pre-built composite model available. Attempting to build fallback generator...")
        try:
            # Attempt to build using _build_composite_generator
            # It might load VAE if path is in params and vae_decoder_model is None
            vae_path = self.params.get("sequential_model_file")
            vae_decoder_for_fallback = None
            if vae_path and os.path.exists(vae_path): 
                self.logger.info(f"Found VAE decoder path for fallback: {vae_path}")
                vae_decoder_for_fallback = self.model_loader.load_model_from_path(vae_path)
                if vae_decoder_for_fallback:
                    vae_decoder_for_fallback.trainable = True 
            
            # _build_composite_generator sets self.composite_model
            self._build_composite_generator(vae_decoder_model=vae_decoder_for_fallback) 
            return self.composite_model # Return the potentially newly built model
        except Exception as e:
            self.logger.error(f"Failed to build fallback generator: {e}", exc_info=True) # Use exc_info
            return None

    def _post_process_to_target_features(self, vae_output_features: tf.Tensor, target_num_features: int) -> tf.Tensor:
        """
        Transforms VAE output features (e.g., 23 from VAE decoder) to the target 
        number of features for the GAN (e.g., 51).
        This uses a Dense layer for the transformation.
        
        Args:
            vae_output_features: Tensor from VAE decoder (shape: (batch, num_vae_features)).
            target_num_features: The desired number of output features (e.g., 51).
            
        Returns:
            Tensor with shape (batch, target_num_features).
        """
        current_features = vae_output_features.shape[-1]
        self.logger.info(f"Post-processing VAE output from {current_features} to {target_num_features} features.")
        
        if current_features == target_num_features:
            self.logger.info("VAE output features already match target. No expansion needed.")
            return vae_output_features
        
        # Use a Dense layer to expand or contract features.
        # Activation can be None (linear) or 'tanh' if outputs are normalized to [-1, 1].
        # Using None for now, assuming normalization is handled elsewhere or features are raw.
        expanded_features = Dense(target_num_features, activation=None, name="feature_expansion_layer")(vae_output_features)
        self.logger.info(f"Feature expansion layer output shape: {expanded_features.shape}")
        return expanded_features

    def _build_composite_generator(self, vae_decoder_model=None) -> Optional[Model]:
        """
        Build the composite generator model combining BiLSTM Z-generator + VAE decoder.
        Based on REFERENCE.md Sequential Conditional VAE-GAN Architecture.
        
        The generator must output sequences of shape (batch_size, 144, self.params.get("num_features", 51)) to match discriminator input.
        
        Args:
            vae_decoder_model: Optional pre-trained VAE decoder model to integrate
            
        Returns:
            Model: The built composite generator model
        """
        try:
            self.logger.info("Building composite generator model...")
            
            # Get configuration parameters
            seq_len = self.params.get("decoder_input_window_size", 144)
            # Use the num_features param from plugin_params, defaulting to 51
            num_output_features = self.params.get("num_features", 51) 
            noise_dim = self.params.get("feeder_noise_dim", 32) # Aligned with plugin_params
            conditional_features_dim = self.params.get("conditional_features_dim", 10)
            context_vector_dim = self.params.get("context_vector_dim", 64)
            
            self.logger.info(f"Building generator with seq_len={seq_len}, num_output_features={num_output_features}, noise_dim={noise_dim}")
            
            # Build generator inputs - per REFERENCE.md, VAE decoder needs 3 inputs
            noise_input = Input(shape=(noise_dim,), name="noise_input")
            conditions_input = Input(shape=(conditional_features_dim,), name="conditions_input")
            context_input = Input(shape=(context_vector_dim,), name="context_input")
            
            if vae_decoder_model is not None:
                # Use pre-trained VAE decoder - implement BiLSTM Z-generator as per REFERENCE.md
                self.logger.info("Building composite model with pre-trained VAE decoder")
                self.logger.info(f"VAE decoder expects {len(vae_decoder_model.inputs)} inputs")
                
                # BiLSTM Z-generator architecture: Dense(576) → Reshape(18,32) → Bidirectional(LSTM(64)) → Conv1D(32)
                z_dense = Dense(576, activation='relu', name="z_dense")(noise_input)
                z_reshape = Reshape((self.params.get("internal_z_sequence_length", 18), 
                                     self.params.get("internal_z_latent_dim", 32)), name="z_reshape")(z_dense)
                z_bilstm = Bidirectional(LSTM(64, return_sequences=True), name="z_bilstm")(z_reshape)
                z_latent_seq = Conv1D(self.params.get("internal_z_latent_dim", 32), kernel_size=3, padding='same', activation='relu', name="z_conv")(z_bilstm)
                
                # Call VAE decoder once to get base features (batch, 23)
                # VAE decoder expects: [z_latent_seq, context_input, conditions_input] as per REFERENCE.md  
                # Ensure vae_decoder_model.inputs matches this expectation
                if not hasattr(vae_decoder_model, 'inputs') or vae_decoder_model.inputs is None: # Check against None
                    self.logger.error("Provided VAE decoder model has no inputs defined or inputs is None.")
                    self.composite_model = None # Ensure reset before returning
                    return None

                if len(vae_decoder_model.inputs) == 3:
                    vae_base_features = vae_decoder_model([z_latent_seq, context_input, conditions_input]) 
                elif len(vae_decoder_model.inputs) == 2: 
                    self.logger.warning("VAE decoder expects 2 inputs, providing z_latent_seq and conditions_input.")
                    vae_base_features = vae_decoder_model([z_latent_seq, conditions_input])
                elif len(vae_decoder_model.inputs) == 1: 
                    self.logger.warning("VAE decoder expects 1 input, providing z_latent_seq.")
                    vae_base_features = vae_decoder_model(z_latent_seq)
                else:
                    self.logger.error(f"VAE decoder has an unexpected number of inputs: {len(vae_decoder_model.inputs)}. Expected 1, 2 or 3.")
                    self.composite_model = None # Ensure reset
                    return None
                
                # Post-process VAE base features to target number of output features
                expanded_features = self._post_process_to_target_features(vae_base_features, num_output_features)
                
                # Replicate single timestep across sequence length to create (batch, seq_len, num_output_features)
                def replicate_across_time(features_input_tuple): # Modified to accept tuple
                    features, target_seq_len, target_num_features = features_input_tuple
                    """Replicate features across time dimension."""
                    # Expand dims to add time dimension: (batch, target_num_features) -> (batch, 1, target_num_features)
                    expanded = tf.expand_dims(features, axis=1)
                    # Tile across time dimension: (batch, 1, target_num_features) -> (batch, target_seq_len, target_num_features)
                    return tf.tile(expanded, [1, target_seq_len, 1])
                
                # Apply replication with explicit output shape
                expanded_output = tf.keras.layers.Lambda(
                    lambda x: replicate_across_time((x, seq_len, num_output_features)), # Pass seq_len and num_output_features
                    output_shape=(seq_len, num_output_features),
                    name="sequence_replicator"
                )(expanded_features)
                    
            else:
                # Build simple generator from scratch for testing
                self.logger.info("Building simple generator from scratch")
                
                # Combine inputs (noise, conditions, and context)
                combined_inputs = Concatenate(name="combined_inputs")([noise_input, conditions_input, context_input])
                
                # Generate sequence directly
                hidden1 = Dense(256, activation='relu', name="hidden1")(combined_inputs)
                hidden2 = Dense(512, activation='relu', name="hidden2")(hidden1)
                hidden3 = Dense(1024, activation='relu', name="hidden3")(hidden2)
                
                # Output full sequence: seq_len * num_output_features features
                sequence_flat = Dense(seq_len * num_output_features, activation='tanh', name="sequence_flat")(hidden3)
                
                # Reshape to sequence format: (batch_size, seq_len, num_output_features)
                expanded_output = Reshape((seq_len, num_output_features), name="output_reshape")(sequence_flat)
            
            # Create composite model with 3 inputs as per REFERENCE.md
            composite_model = Model(
                inputs=[noise_input, conditions_input, context_input],
                outputs=expanded_output,
                name="composite_generator"
            )
            
            self.logger.info(f"Composite generator built with {composite_model.count_params()} parameters")
            if hasattr(composite_model, 'output') and composite_model.output is not None: # Check against None
                self.logger.info(f"Generator output shape: {composite_model.output.shape}")
            else:
                self.logger.warning("Built composite model has no output attribute or output is None.")
            
            # Store the model internally
            self.composite_model = composite_model # Set the class attribute
            
            return composite_model # Return the built model
            
        except Exception as e:
            self.logger.error(f"Error building composite generator: {e}")
            self.logger.error(traceback.format_exc()) # Log full traceback
            self.composite_model = None # Ensure it's None on failure
            return None

    def prepare_features_for_discriminator(self, data_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepares a raw DataFrame to have the features expected by the discriminator.
        Calculates technical indicators and cyclical date/time features.
        Orders features according to 'full_feature_names_ordered'.
        The number of output features should align with self.params.get("num_features", 51).
        """
        self.logger.info("Preparing real data features for discriminator input...")
        processed_df = data_df.copy()
        
        target_num_features = self.params.get("num_features", 51)
        datetime_col_name = self.main_config.get("datetime_col_name", "DATE_TIME")

        if datetime_col_name not in processed_df.columns:
            raise ValueError(f"Datetime column '{datetime_col_name}' not found in input data for feature preparation.")
        processed_df[datetime_col_name] = pd.to_datetime(processed_df[datetime_col_name])

        # 1. Calculate Technical Indicators
        if not hasattr(self, 'ti_calculator') or self.ti_calculator is None:
            self.logger.warning("TI calculator not found, attempting to initialize feature-dependent modules.")
            self._initialize_feature_dependent_modules()
        if not hasattr(self, 'ti_calculator') or self.ti_calculator is None:
            raise RuntimeError("TechnicalIndicatorCalculator not initialized in GeneratorPlugin.")
        
        ohlc_features = self.params.get("ohlc_feature_names", ["OPEN", "HIGH", "LOW", "CLOSE"])
        missing_ohlc = [f for f in ohlc_features if f not in processed_df.columns]
        if missing_ohlc:
            raise ValueError(f"Missing OHLC columns for TI calculation: {missing_ohlc}. Available: {processed_df.columns.tolist()}")

        self.logger.info(f"Calculating TIs using ohlc_features: {ohlc_features}")
        # Pass a copy of processed_df to ensure ti_calculator doesn't modify it in a way that affects original data if it operates in-place on parts.
        ti_df = self.ti_calculator.calculate_technical_indicators(
            processed_df.copy(), 
            ohlc_feature_names=ohlc_features,
            return_last_row_only=False
        )
        
        self.logger.info(f"Shape of processed_df before TI merge: {processed_df.shape}")
        self.logger.info(f"Columns of processed_df before TI merge: {processed_df.columns.tolist()}")
        self.logger.info(f"Shape of ti_df from calculator: {ti_df.shape}")
        ti_cols_produced_by_calculator = ti_df.columns.tolist()
        self.logger.info(f"Columns of ti_df from calculator: {ti_cols_produced_by_calculator}")

        # Before merging, remove any columns from processed_df that have the same names
        # as the TIs we are about to merge from ti_df.
        # This ensures the newly calculated TIs from ti_df take precedence and avoids suffixed columns.
        cols_to_drop_from_processed_df = [col for col in ti_cols_produced_by_calculator if col in processed_df.columns]
        if cols_to_drop_from_processed_df:
            self.logger.info(f"Dropping pre-existing columns from input data to avoid TI merge conflicts: {cols_to_drop_from_processed_df}")
            processed_df.drop(columns=cols_to_drop_from_processed_df, inplace=True)
        
        # Merge the newly calculated TIs.
        processed_df = pd.merge(processed_df, ti_df, left_index=True, right_index=True, how='left')
        self.logger.info(f"Columns after TI merge: {processed_df.columns.tolist()}")
        self.logger.info(f"Shape of processed_df after TI merge: {processed_df.shape}")
        
        # Check for NaNs introduced by TIs (especially at the beginning of the series)
        # Use ti_cols_produced_by_calculator for checking NaNs in the TIs that were just added.
        ti_cols_in_processed_after_merge = [col for col in ti_cols_produced_by_calculator if col in processed_df.columns]
        if ti_cols_in_processed_after_merge:
            nan_counts_in_tis = processed_df[ti_cols_in_processed_after_merge].isnull().sum()
            if nan_counts_in_tis.any():
                 self.logger.info(f"NaN counts in TI columns after merge (expected at start of series):\\n{nan_counts_in_tis[nan_counts_in_tis > 0]}")
        else:
            self.logger.warning("No TI columns (as produced by calculator) found in processed_df after merge for NaN check.")


        # 2. Calculate Cyclical Date/Time Features
        if not hasattr(self, 'data_generator') or self.data_generator is None:
            self.logger.warning("Data generator not found, attempting to initialize feature-dependent modules.")
            self._initialize_feature_dependent_modules()
        if not hasattr(self, 'data_generator') or self.data_generator is None:
            raise RuntimeError("DataGenerator not initialized in GeneratorPlugin.")

        # Use 'feeder_date_features_for_conditioning' from main_config to determine which cyclical features to generate.
        # This ensures alignment with what the FeederPlugin might provide for conditions.
        # The desired final list of features should NOT include day_of_year_sin/cos if not intended.
        date_features_to_make_cyclical = self.main_config.get('feeder_date_features_for_conditioning', [])
        # Example: if date_features_to_make_cyclical = ["day_of_month", "hour_of_day", "day_of_week"]
        # then only sin/cos for these three will be generated.
        
        self.logger.info(f"Base date features for cyclical generation: {date_features_to_make_cyclical}")

        cyclical_feature_specs = []
        # Ensure default_max_map covers all features in feeder_date_features_for_conditioning from config
        default_max_map = {
            'day_of_month': 31, 
            'hour_of_day': 23, # Max value is 23 for 0-23 hours
            'day_of_week': 6,  # Max value is 6 for 0-6 days
            # 'day_of_year': 365, # Removed as per Scenario B - day_of_year cyclical features are not desired
            'month_of_year': 12, 
            'week_of_year': 52
        }
        
        for base_feature_name in date_features_to_make_cyclical:
            if base_feature_name in default_max_map:
                cyclical_feature_specs.append({
                    "feature_name": base_feature_name,
                    "max_value": default_max_map[base_feature_name]
                })
            else:
                self.logger.warning(f"Max value for date feature '{base_feature_name}' not defined in default_max_map. Skipping cyclical generation for it.")
        
        if cyclical_feature_specs:
            self.logger.info(f"Generating cyclical features using specs: {cyclical_feature_specs}")
            processed_df = self.data_generator.add_cyclical_date_features(processed_df, datetime_col_name, cyclical_feature_specs)
            self.logger.info(f"Columns after cyclical feature generation: {processed_df.columns.tolist()}")
        else:
            self.logger.info("No cyclical features specified or to be generated based on 'feeder_date_features_for_conditioning'.")

        # 3. Final Feature Selection and Ordering
        all_expected_features_ordered = self.main_config.get("generator_full_feature_names_ordered", [])
        if not all_expected_features_ordered:
            self.logger.error("'generator_full_feature_names_ordered' not found in config. Cannot finalize features.")
            raise ValueError("'generator_full_feature_names_ordered' is crucial and not found in configuration.")

        numeric_features_ordered = [f for f in all_expected_features_ordered if f != datetime_col_name]
        self.logger.info(f"Target numeric features ordered ({len(numeric_features_ordered)}): {numeric_features_ordered}")


        missing_features = [f for f in numeric_features_ordered if f not in processed_df.columns]
        if missing_features:
            self.logger.error(f"Missing expected numeric features after all processing steps: {missing_features}")
            self.logger.error(f"Available columns in processed_df: {processed_df.columns.tolist()}")
            raise ValueError(f"Could not produce all expected numeric features. Missing: {missing_features}. Please check TI and cyclical feature generation.")

        try:
            final_df = processed_df[numeric_features_ordered].copy()
        except KeyError as e:
            self.logger.error(f"KeyError during final numeric feature selection: {e}. One or more expected features not found.")
            self.logger.error(f"Expected numeric features: {numeric_features_ordered}")
            self.logger.error(f"Available columns: {processed_df.columns.tolist()}")
            raise

        if final_df.shape[1] != target_num_features:
            self.logger.error(
                f"Final numeric feature count ({final_df.shape[1]}) does not match target_num_features ({target_num_features})."
            )
            self.logger.error(f"Selected features: {final_df.columns.tolist()}")
            self.logger.error(f"Expected from config (numeric part of generator_full_feature_names_ordered): {numeric_features_ordered}")
            raise ValueError(f"Final numeric feature count mismatch: expected {target_num_features}, got {final_df.shape[1]}. Check 'generator_full_feature_names_ordered' and feature generation steps.")
        
        self.logger.info(f"Successfully prepared {final_df.shape[1]} numeric features for discriminator. Shape: {final_df.shape}")
        self.logger.info(f"Final numeric feature columns for discriminator: {final_df.columns.tolist()}")
        
        if final_df.isnull().values.any():
            nan_counts_final = final_df.isnull().sum()
            self.logger.warning(f"NaNs found in final numeric feature set for discriminator:\\n{nan_counts_final[nan_counts_final > 0]}")
            final_df = final_df.fillna(0) 
            self.logger.warning("Filled NaNs with 0 in the final numeric feature set. Review if this is the desired strategy.")
            
        return final_df

    def sample_noise_for_model(self, batch_size: int) -> Dict[str, np.ndarray]:
        """
        Generates a batch of noise, conditions, and context vectors for the generator model.
        This method is intended to be called by the GAN training loop.

        Args:
            batch_size: The number of samples to generate inputs for.

        Returns:
            A dictionary containing 'noise_input', 'conditions_input', and 'context_input'
            compatible with the generator model's inputs.
        """
        noise_dim = self.params.get("feeder_noise_dim", 32)
        conditional_features_dim = self.params.get("conditional_features_dim", 10) # Example, ensure this matches model
        context_vector_dim = self.params.get("context_vector_dim", 64) # Example, ensure this matches model

        self.logger.debug(f"Sampling inputs for generator: batch_size={batch_size}, noise_dim={noise_dim}, cond_dim={conditional_features_dim}, ctx_dim={context_vector_dim}")

        noise = np.random.normal(0, 1, (batch_size, noise_dim))
        
        # Placeholder for actual conditional features and context vectors
        # These should ideally come from the FeederPlugin or be generated based on some strategy
        # For now, using random data as a placeholder.
        # TODO: Integrate with FeederPlugin to get meaningful conditional/context data if available for the batch.
        conditions = np.random.rand(batch_size, conditional_features_dim) 
        context = np.random.rand(batch_size, context_vector_dim)

        return {
            "noise_input": noise,
            "conditions_input": conditions,
            "context_input": context
        }
    
    def save_model(self, model_path: Optional[str] = None) -> str:
        """
        Save the current generator model to the specified path.

        This method uses the ModelSaver component to handle the actual saving process.
        It ensures that the model is saved in a way that is compatible with TensorFlow/Keras
        and can be easily reloaded later.

        Args:
            model_path: Optional path where the model should be saved.
                         If not provided, a default path will be used.

        Returns:
            str: The path where the model was saved.
        """
        self.logger.info(f"Saving generator model to path: {model_path}")
        effective_model_path = self.model_saver.save_model(self.model, model_path)
        self.logger.info(f"Model saved. Effective path: {effective_model_path}")
        return effective_model_path

    def generate(self,
                 n_samples: int,
                 conditional_features: Optional[np.ndarray] = None,
                 initial_context_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Generates synthetic time series data.

        This method delegates the generation task to the internal DataGenerator component.
        It ensures that the model and its components are initialized before generation.

        Args:
            n_samples: The number of samples (time steps) to generate.
            conditional_features: An optional numpy array of conditional features
                                  (e.g., date/time features) for each sample to be generated.
                                  Shape should be (n_samples, num_conditional_features).
            initial_context_data: Optional DataFrame providing an initial window of
                                  real data to seed the generation process. This is used
                                  for iterative generation where the previous step's output
                                  (or real data) becomes context for the next.

        Returns:
            pd.DataFrame: A DataFrame containing the generated synthetic time series data,
                          with features as columns and time steps as rows.
        
        Raises:
            RuntimeError: If the DataGenerator component is not initialized or the model is not available.
        """
        self.logger.info(f"GeneratorPlugin: Received request to generate {n_samples} samples.")

        if not hasattr(self, 'data_generator_component') or self.data_generator_component is None:
            self.logger.error("DataGenerator component is not initialized in GeneratorPlugin. "
                              "Ensure plugin.set_params() which calls initialize_model_components() has been successfully executed.")
            raise RuntimeError("GeneratorPlugin's DataGenerator component is not initialized.")

        # self.model here refers to self.generator_model_composite, which should be built/loaded.
        if self.model is None:
            self.logger.error("Generator model (self.model / self.generator_model_composite) is not built or loaded in GeneratorPlugin.")
            raise RuntimeError("GeneratorPlugin's model is not available for generation.")

        self.logger.debug(f"Number of samples to generate: {n_samples}")
        self.logger.debug(f"Conditional features shape: {conditional_features.shape if conditional_features is not None else 'None'}")
        self.logger.debug(f"Initial context data shape: {initial_context_data.shape if initial_context_data is not None else 'None'}")

        try:
            synthetic_data_df = self.data_generator_component.generate_synthetic_data(
                n_samples=n_samples,
                conditional_features=conditional_features,
                initial_context_data=initial_context_data
            )
            self.logger.info(f"GeneratorPlugin: Successfully generated {synthetic_data_df.shape[0]} samples with {synthetic_data_df.shape[1]} features.")
            return synthetic_data_df
        except Exception as e:
            self.logger.error(f"GeneratorPlugin: Error during synthetic data generation via DataGenerator component: {e}", exc_info=True)
            # Re-raise as a RuntimeError to be caught by the caller
            raise RuntimeError(f"GeneratorPlugin failed to generate data: {e}")

# Example usage (for testing or direct invocation, not part of the plugin's core flow in the app)
if __name__ == '__main__':
    # Basic testing or demonstration of the GeneratorPlugin usage
    # This block can be used to quickly test the plugin's functionality
    # without needing to run the entire application.
    
    # Example config - in practice, this would be more detailed and loaded from a file or other source
    example_config = {
        "sequential_model_file": "path/to/vae_decoder_model", # Update with a valid model path for testing
        "decoder_input_window_size": 144,
        "full_feature_names_ordered": ["OPEN", "HIGH", "LOW", "CLOSE", "TARGET_FEATURE"],
        "decoder_output_feature_names": ["TARGET_FEATURE"],
        "ohlc_feature_names": ["OPEN", "HIGH", "LOW", "CLOSE"],
        "ti_feature_names": ["SMA", "EMA"], # Example TIs
        "date_conditional_feature_names": ["year", "month", "day"],
        "feeder_conditional_feature_names": ["noise"],
        "ti_calculation_min_lookback": 200,
        "ti_params": {},
        "generator_normalization_params_file": "path/to/normalization_params.json",
        "internal_z_sequence_length": 18,
        "internal_z_latent_dim": 32,
        "feeder_noise_dim": 32,
        "context_vector_dim": 64,
        "conditional_features_dim": 10,
        "num_features": 51,
        "base_feature_names_ordered": []
    }
    
    # Initialize the plugin
    generator_plugin = GeneratorPlugin(example_config)
    
    # Generate some synthetic data
    try:
        synthetic_data = generator_plugin.generate(
            n_samples=10, 
            conditional_features=np.random.rand(10, 10), # Example conditional features
            initial_context_data=None # No initial context for now
        )
        print("Generated synthetic data:")
        print(synthetic_data)
    except Exception as e:
        print(f"Error during synthetic data generation: {e}")
