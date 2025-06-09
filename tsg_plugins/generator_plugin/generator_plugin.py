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

logger = get_logger(__name__)

class GeneratorPlugin(PluginBase):
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
                self.composite_model = None # Ensure it's None
                raise IOError("Composite generator could not be built.")

            self.sequential_model = built_model # For compatibility if anything still uses it
            # self.composite_model is already set by _build_composite_generator

            self.logger.info("Composite generator model built successfully.")
            if hasattr(self.model, 'inputs') and self.model.inputs is not None: # Check against None
                self.logger.info(f"Composite generator input shapes: {[inp.shape for inp in self.model.inputs]}")
            if hasattr(self.model, 'output') and self.model.output is not None: # Check against None
                self.logger.info(f"Composite generator output shape: {self.model.output.shape}")

        except Exception as e:
            self.logger.error(f"Error during _load_model (building composite generator): {e}", exc_info=True)
            self.sequential_model = None
            self.composite_model = None # Ensure it's None on any exception
            raise IOError(f"Failed to build composite generator model: {e}")

    def _build_bilstm_z_generator(self, input_dim_z, input_dim_c, output_dim, seq_len):
        logger.info(f"Building BiLSTM Z-Generator with Z_input_dim={input_dim_z}, C_input_dim={input_dim_c}, output_dim={output_dim}, seq_len={seq_len}")
        
        l2_reg = self.config.get("generator_l2_reg", 0.01) if self.config.get("use_generator_l2_reg", False) else None
        if l2_reg is not None:
            logger.info(f"Applying L2 regularization with lambda={l2_reg} to Dense, Conv1D, LSTM layers in BiLSTM Z-Generator.")
        else:
            logger.info("No L2 regularization will be applied to BiLSTM Z-Generator.")

        # Latent space Z input
        z_input = Input(shape=(input_dim_z,), name="generator_z_input")
        
        # Conditional input C
        c_input = Input(shape=(input_dim_c,), name="generator_c_input")

        # Process Z
        z_processed = Dense(128, activation='relu', kernel_regularizer=l2(l2_reg) if l2_reg else None)(z_input)
        z_processed = RepeatVector(seq_len)(z_processed)

        # Process C (make it compatible for concatenation with Z sequence)
        # Assuming c_input might represent static features or a summary that needs to be expanded
        c_processed = Dense(64, activation='relu', kernel_regularizer=l2(l2_reg) if l2_reg else None)(c_input) # Apply L2 if configured
        c_processed = RepeatVector(seq_len)(c_processed)

        # Concatenate processed Z and C
        merged_input = Concatenate(axis=-1)([z_processed, c_processed])
        
        # Bidirectional LSTM layers
        # First BiLSTM layer
        lstm_out1 = Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=l2(l2_reg) if l2_reg else None, recurrent_regularizer=l2(l2_reg) if l2_reg else None))(merged_input)
        # Apply BatchNormalization and Dropout if configured (currently not, as per requirements)
        # lstm_out1 = BatchNormalization()(lstm_out1) # Example if needed
        # lstm_out1 = Dropout(0.2)(lstm_out1)      # Example if needed

        # Second BiLSTM layer (optional, depending on complexity needed)
        lstm_out2 = Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(l2_reg) if l2_reg else None, recurrent_regularizer=l2(l2_reg) if l2_reg else None))(lstm_out1)
        # lstm_out2 = BatchNormalization()(lstm_out2) # Example if needed
        # lstm_out2 = Dropout(0.2)(lstm_out2)      # Example if needed
        
        # TimeDistributed Dense layer to get the desired output_dim for each time step
        output = TimeDistributed(Dense(output_dim, activation='sigmoid', kernel_regularizer=l2(l2_reg) if l2_reg else None))(lstm_out2) # Sigmoid for normalized data [0,1]

        model = Model(inputs=[z_input, c_input], outputs=output, name="BiLSTM_Z_Generator")
        logger.info("BiLSTM Z-Generator built successfully.")
        return model

    def _build_vae_generator(self, vae_model_path, latent_dim, condition_dim, output_dim, seq_len):
        logger.info(f"Building VAE-based Generator. Loading VAE from: {vae_model_path}")
        try:
            vae = tf.keras.models.load_model(vae_model_path, compile=False) # Set compile=False
            decoder = vae.decoder 
            logger.info("VAE model and decoder loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading VAE model from {vae_model_path}: {e}")
            raise

        # Apply L2 regularization to the decoder's layers if configured
        l2_reg = self.config.get("generator_l2_reg", 0.01) if self.config.get("use_generator_l2_reg", False) else None
        if l2_reg:
            logger.info(f"Applying L2 regularization with lambda={l2_reg} to VAE decoder's Dense, Conv1D, LSTM layers.")
            for layer in decoder.layers:
                if isinstance(layer, (Dense, Conv1D, LSTM)):
                    logger.info(f"Applying L2 to VAE decoder layer: {layer.name}")
                    layer.kernel_regularizer = l2(l2_reg)
                    if hasattr(layer, 'recurrent_regularizer') and isinstance(layer, LSTM): # Apply to recurrent kernel for LSTMs
                         layer.recurrent_regularizer = l2(l2_reg)
                elif isinstance(layer, Bidirectional):
                    logger.info(f"Applying L2 to VAE decoder Bidirectional layer: {layer.name}")
                    # For Bidirectional, apply to the wrapped LSTM layer
                    if isinstance(layer.forward_layer, LSTM):
                        layer.forward_layer.kernel_regularizer = l2(l2_reg)
                        layer.forward_layer.recurrent_regularizer = l2(l2_reg)
                    if isinstance(layer.backward_layer, LSTM):
                        layer.backward_layer.kernel_regularizer = l2(l2_reg)
                        layer.backward_layer.recurrent_regularizer = l2(l2_reg)
            
            # Re-instantiate the decoder model with the modified layers if necessary.
            # This is crucial if the regularizers are not applied by modifying layers in-place effectively.
            # For functional models, rebuilding the model with new layer configs is safer.
            # However, Keras often allows direct modification of layer attributes before model compilation/use.
            # Let's assume direct modification works for now. If issues arise, model cloning/rebuilding is the way.
            logger.info("L2 regularization applied to VAE decoder layers. Note: This assumes direct modification of layer attributes is effective. If not, model re-cloning might be needed.")


        # Define inputs for the new generator model
        z_input = Input(shape=(latent_dim,), name="generator_z_input") # Latent vector from GAN's noise
        c_input = Input(shape=(condition_dim,), name="generator_c_input") # Conditional input

        # How to use c_input with the VAE decoder?
        # Option 1: Concatenate c_input with z_input if decoder's first layer can handle it.
        # Option 2: Process c_input and merge it at a deeper layer (more complex).
        # Option 3: If VAE was conditional, its decoder might already expect conditional input.
        # Assuming the VAE decoder takes only a latent vector. We might need to adapt.
        # For now, let's try concatenating them and adding a Dense layer to match decoder's expected input.
        
        # This part is speculative and depends heavily on the VAE decoder's architecture.
        # If the VAE decoder expects input shape (decoder_latent_dim,), we need to map [z_input, c_input] to it.
        
        # Example: Combine z and c, then pass to a Dense layer to match decoder's expected input dim
        combined_input = Concatenate()([z_input, c_input])
        # The size of this Dense layer should match the input dimension expected by the VAE's decoder first layer
        # This requires knowing the VAE decoder's architecture.
        # Let's assume decoder.input_shape is (None, decoder_latent_dim)
        try:
            decoder_input_dim = decoder.input_shape[-1]
            logger.info(f"VAE Decoder expected input dimension: {decoder_input_dim}")
        except Exception as e:
            logger.warning(f"Could not infer VAE decoder input dimension: {e}. Assuming a default or direct pass-through.")
            # Fallback or raise error if this is critical
            decoder_input_dim = latent_dim # Defaulting, this might be wrong

        processed_for_decoder = Dense(decoder_input_dim, activation='relu', kernel_regularizer=l2(l2_reg) if l2_reg else None)(combined_input) # Apply L2 if configured
        
        # Get output from the VAE decoder
        decoder_output = decoder(processed_for_decoder) # This is the generated sequence part

        # The VAE decoder output might be, for example, (batch_size, seq_len, num_features_in_vae)
        # We need to ensure it matches the GAN's `output_dim` and `seq_len`.
        # If `output_dim` (from GAN perspective) is different from `num_features_in_vae`,
        # a final Dense layer (TimeDistributed) might be needed.

        # Let's assume VAE output is (batch_size, seq_len, vae_output_features)
        # And we need (batch_size, seq_len, output_dim) where output_dim is for the GAN
        
        # If the VAE's output_dim already matches the GAN's required output_dim, no extra layer needed.
        # Otherwise, add a TimeDistributed Dense layer to adjust the feature dimension.
        current_output_features = decoder_output.shape[-1]
        if current_output_features != output_dim:
            logger.info(f"VAE decoder output features ({current_output_features}) != GAN output_dim ({output_dim}). Adding TimeDistributed Dense layer.")
            final_output = TimeDistributed(Dense(output_dim, activation='sigmoid', kernel_regularizer=l2(l2_reg) if l2_reg else None), name="gan_output_projection")(decoder_output) # Sigmoid for [0,1]
        else:
            # If shapes match, but activation is not 'sigmoid', we might need to add it.
            # However, usually, the decoder's last layer would have the appropriate activation.
            # For now, assume it's compatible or the VAE is designed for [0,1] outputs.
            # If VAE output is e.g. tanh, an additional activation layer or rescaling might be needed.
            logger.info(f"VAE decoder output features ({current_output_features}) == GAN output_dim ({output_dim}). Using decoder output directly.")
            final_output = decoder_output 
            # Consider adding a final activation if decoder doesn't ensure [0,1] range, e.g.
            # final_output = tf.keras.layers.Activation('sigmoid')(decoder_output)


        model = Model(inputs=[z_input, c_input], outputs=final_output, name="VAE_GAN_Generator")
        logger.info("VAE-based Generator built successfully.")
        # model.summary(print_fn=logger.info) # Optional: print summary here or in trainer
        return model

    @property
    def model(self):
        # This property should ideally return the primary model interface used by the GAN,
        # which is the composite_model if built, or the sequential_model (VAE decoder) otherwise.
        # Or, it could be specifically self._model if that's the convention for GANs.
        # Given the GAN trainer will interact with this, it should be the generator model
        # that takes (noise, conditions) and outputs sequences.
        if hasattr(self, '_model') and self._model is not None:
            return self._model
        if hasattr(self, 'composite_model') and self.composite_model is not None:
            # This was the VAE-based composite model, which might be what's intended
            # if generator_type was 'vae' and it was successfully built.
            return self.composite_model 
        # Fallback or initial state before a specific GAN generator is built by self.build()
        # self.logger.warning("GeneratorPlugin.model accessed but self._model is not set. Check build process.")
        return None # Or raise an error if a model is always expected after init/build

    @model.setter
    def model(self, value: Optional[Model]):
        # This setter should align with what self.model property returns.
        # It's likely intended to set self._model, which is the common pattern for the GAN's generator.
        self.logger.info(f"GeneratorPlugin.model being set with: {type(value)}")
        self._model = value
        # If self.composite_model was the VAE-based one, and self.model is now the GAN's generator,
        # this implies a shift in what 'model' refers to, or that self._model is the primary interface.

    def build(self):
        """
        Builds the generator model based on the configuration.
        This method should set self._model.
        """
        self.logger.info("GeneratorPlugin: Initiating build process for the generator model...")
        try:
            # Ensure essential configurations are present
            if not self.config.get("num_features") or not self.config.get("seq_len"):
                self.logger.error("GeneratorPlugin: Missing 'num_features' or 'seq_len' in config for building model.")
                raise ValueError("Essential configuration for generator model building is missing.")

            # Determine which generator to build based on config
            generator_type = self.config.get("generator_type", "bilstm_z") # Default to bilstm_z
            self.logger.info(f"Selected generator type: {generator_type}")

            if generator_type == "bilstm_z":
                self._model = self._build_bilstm_z_generator(
                    input_dim_z=self.config.get("latent_dim_z"),
                    input_dim_c=self.config.get("latent_dim_c"),
                    output_dim=self.config.get("num_features"), # num_features from data config
                    seq_len=self.config.get("seq_len")
                )
            elif generator_type == "vae":
                # This implies the VAE decoder itself, or a wrapper around it, will be the generator.
                # The _build_vae_generator method should return a Keras model.
                vae_model_path = self.config.get("vae_model_path")
                if not vae_model_path:
                    self.logger.error("GeneratorPlugin: 'vae_model_path' not provided for 'vae' generator type.")
                    raise ValueError("'vae_model_path' is required for VAE generator type.")
                
                self._model = self._build_vae_generator(
                    vae_model_path=vae_model_path,
                    latent_dim=self.config.get("latent_dim_z"), # GAN's latent space for Z
                    condition_dim=self.config.get("latent_dim_c"), # GAN's conditional input C
                    output_dim=self.config.get("num_features"), # Target num_features for GAN output
                    seq_len=self.config.get("seq_len")
                )
            else:
                error_msg = f"Unsupported generator type: {generator_type}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)

            if self._model is None:
                self.logger.error(f"GeneratorPlugin: Model building returned None for type '{generator_type}'.")
                raise RuntimeError(f"Failed to build generator model of type '{generator_type}'.")
            
            self.logger.info(f"GeneratorPlugin: Successfully built generator model of type '{generator_type}'.")
            # self._model.summary(print_fn=self.logger.info) # Optional: Log summary after build

        except Exception as e:
            self.logger.error(f"GeneratorPlugin: Error during build process: {e}", exc_info=True)
            self._model = None # Ensure model is None if build fails
            raise # Re-raise the exception to signal failure

        return self._model # Return the built model
