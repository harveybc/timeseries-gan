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
        "conditional_features_dim": 10, # For decoder_input_conditions
        "num_features": 51 # ADDED: To align with discriminator and overall architecture
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
        self.main_config = config.copy() # Store the full config
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize core attributes
        self.sequential_model: Optional[Model] = None
        self.model: Optional[Model] = None  # Alias for sequential_model
        self.feature_to_idx: Dict[str, int] = {}
        self.num_all_features: int = 0
        
        # Initialize specialized modules (those not dependent on full config first)
        self._initialize_modules()
        
        # Fully set parameters and initialize feature-dependent modules using the provided config
        # This ensures ti_calculator, data_generator etc. are ready.
        self.set_params(**config) 
        
        # Validate configuration (set_params might call parts of this, ensure it's covered)
        # self._validate_plugin_configuration() # Often called within set_params logic

        # Load initial close anchor (set_params might handle this if x_train_file is in config)
        initial_close_file_path = self.main_config.get("x_train_file", self.main_config.get("real_data_file"))
        if initial_close_file_path and self.initial_data_handler.get_initial_close_anchor() is None:
            try:
                self.initial_data_handler.load_initial_close_anchor(initial_close_file_path)
            except Exception as e:
                self.logger.warning(f"Failed to load initial close anchor in __init__: {e}")

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
        
        The generator must output sequences of shape (batch_size, 144, 51) to match discriminator input.
        
        Args:
            vae_decoder_model: Optional pre-trained VAE decoder model to integrate
            
        Returns:
            Model: The built composite generator model
        """
        try:
            self.logger.info("Building composite generator model...")
            
            # Get configuration parameters
            seq_len = self.params.get("decoder_input_window_size", 144)
            # Use the new num_features param, defaulting to 51
            num_output_features = self.params.get("num_features", 51) 
            noise_dim = self.params.get("feeder_noise_dim", 32) # Corrected from 100 to 32 based on recent config
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
                vae_base_features = vae_decoder_model([z_latent_seq, context_input, conditions_input])  # Shape: (batch, 23)
                
                # Post-process 23 base features to 51 final features (num_output_features)
                expanded_features = self._post_process_to_target_features(vae_base_features, num_output_features)  # Shape: (batch, num_output_features)
                
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
            self.logger.info(f"Generator output shape: {composite_model.output.shape}")
            
            # Store the model
            self.composite_model = composite_model
            
            return composite_model
            
        except Exception as e:
            self.logger.error(f"Error building composite generator: {e}")
            self.logger.error(traceback.format_exc())
            self.composite_model = None
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
        
        target_num_features = self.params.get("num_features", 51) # Target 51 features

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
        ti_df = self.ti_calculator.calculate_technical_indicators(
            processed_df, 
            ohlc_feature_names=ohlc_features,
            return_last_row_only=False # Ensure all rows are processed
        )
        
        self.logger.info(f"Shape of processed_df before TI merge: {processed_df.shape}")
        self.logger.info(f"Columns of processed_df before TI merge: {processed_df.columns.tolist()}")
        self.logger.info(f"Shape of ti_df: {ti_df.shape}")
        self.logger.info(f"Columns of ti_df: {ti_df.columns.tolist()}")

        processed_df = pd.merge(processed_df, ti_df, left_index=True, right_index=True, how='left')
        self.logger.info(f"Columns after TI merge: {processed_df.columns.tolist()}")
        self.logger.info(f"Shape of processed_df after TI merge: {processed_df.shape}")
        
        ti_cols_in_processed = [col for col in self.params.get("ti_feature_names", []) if col in processed_df.columns]
        if ti_cols_in_processed:
            nan_counts_in_tis = processed_df[ti_cols_in_processed].isnull().sum()
            self.logger.info(f"NaN counts in TI columns after merge:\\n{nan_counts_in_tis[nan_counts_in_tis > 0]}")


        # 2. Calculate Cyclical Date/Time Features
        if not hasattr(self, 'data_generator') or self.data_generator is None:
            self.logger.warning("Data generator not found, attempting to initialize feature-dependent modules.")
            self._initialize_feature_dependent_modules()
        if not hasattr(self, 'data_generator') or self.data_generator is None:
            raise RuntimeError("DataGenerator not initialized in GeneratorPlugin.")

        date_features_to_generate = self.main_config.get('feeder_date_features_for_conditioning', [])
        cyclical_feature_specs = []
        default_max_map = {'day_of_month': 31, 'hour_of_day': 23, 'day_of_week': 6, 'day_of_year': 365}
        
        for base_feature_name in date_features_to_generate:
            max_val_key_1 = f'feeder_max_{base_feature_name}'
            max_val_key_2 = f'max_{base_feature_name}'
            max_val = self.main_config.get(max_val_key_1, self.main_config.get(max_val_key_2, default_max_map.get(base_feature_name)))
            if max_val is None:
                self.logger.warning(f"Max value for cyclical feature {base_feature_name} not found. Skipping.")
                continue
            cyclical_feature_specs.append((base_feature_name, max_val))

        if cyclical_feature_specs:
            self.logger.info(f"Calculating cyclical features for: {cyclical_feature_specs}")
            cyclical_df = self.data_generator.generate_cyclical_features_for_df(
                data_df=processed_df, 
                datetime_col_name=datetime_col_name,
                feature_specs=cyclical_feature_specs
            )
            processed_df = pd.merge(processed_df, cyclical_df, left_index=True, right_index=True, how='left')
            self.logger.info(f"Columns after cyclical features merge: {processed_df.columns.tolist()}")

        # 3. Ensure all features from full_feature_names_ordered are present and in order
        raw_full_feature_names = self.params.get("full_feature_names_ordered", [])
        self.logger.info(f"RAW full_feature_names_ordered from params ({len(raw_full_feature_names)}): {raw_full_feature_names}")

        if not raw_full_feature_names:
            raise ValueError("GeneratorPlugin: 'full_feature_names_ordered' is not configured or empty.")
        
        expected_numeric_features = [f for f in raw_full_feature_names if f != datetime_col_name]
        self.logger.info(f"Expected numeric features AFTER filtering datetime_col '{datetime_col_name}' ({len(expected_numeric_features)}): {expected_numeric_features}")
        
        # Ensure expected_numeric_features matches target_num_features
        if len(expected_numeric_features) != target_num_features:
            self.logger.warning(
                f"Mismatch: 'full_feature_names_ordered' (numeric part) has {len(expected_numeric_features)} features, "
                f"but 'num_features' param is {target_num_features}. Using 'full_feature_names_ordered'."
            )
            # If you want to strictly enforce num_features, you might adjust expected_numeric_features here,
            # but it's usually better if full_feature_names_ordered is the source of truth.

        self.logger.info(f"Columns in processed_df before final selection ({len(processed_df.columns.tolist())}): {processed_df.columns.tolist()}")

        missing_cols = [col for col in expected_numeric_features if col not in processed_df.columns]
        if missing_cols:
            self.logger.warning(f"Missing columns that were expected in 'full_feature_names_ordered' (numeric part): {missing_cols}. They will be filled with 0.0.")
            for col in missing_cols:
                processed_df[col] = 0.0 # Ensure new columns are added with the correct type if possible, e.g., float
        
        # Ensure no NaN values in the final feature set (TIs can produce NaNs at the beginning)
        # Only select columns that are actually in processed_df to avoid KeyErrors if a feature in expected_numeric_features was never created
        final_expected_cols_present = [col for col in expected_numeric_features if col in processed_df.columns]
        processed_df[final_expected_cols_present] = processed_df[final_expected_cols_present].fillna(0.0)

        # Select the final set of features
        final_df = processed_df[final_expected_cols_present]
        
        # If some expected_numeric_features were still missing and not created by the loop above, reindex to ensure all are present, filled with 0.0
        if len(final_df.columns) != len(expected_numeric_features):
            self.logger.warning(f"Re-indexing to ensure all {len(expected_numeric_features)} expected numeric features are present.")
            final_df = final_df.reindex(columns=expected_numeric_features, fill_value=0.0)

        self.logger.info(f"Real data prepared with {len(final_df.columns)} features: {final_df.columns.tolist()}")
        
        if len(final_df.columns) != target_num_features:
             self.logger.error(
                 f"CRITICAL MISMATCH: Final prepared data has {len(final_df.columns)} features, "
                 f"but system is configured for {target_num_features} features. This will likely cause errors downstream."
             )
        
        return final_df

    def _post_process_to_target_features(self, base_features, target_num_features: int):
        """
        Post-process base features from VAE decoder to the target number of final features.
        This is a placeholder and should ideally involve actual calculation of TIs and cyclical features
        if the VAE output doesn't include them directly or in a form that can be expanded.
        Currently uses Dense layers for expansion.
        
        Args:
            base_features: Tensor of shape (batch, num_base_features) from VAE decoder (e.g., 23 features)
            target_num_features: The desired number of output features (e.g., 51)
            
        Returns:
            Tensor of shape (batch, target_num_features) with expanded features
        """
        num_base_features = base_features.shape[-1]
        self.logger.info(f"Post-processing VAE output from {num_base_features} to {target_num_features} features using Dense layers.")

        if num_base_features == target_num_features:
            self.logger.info("Base features already match target feature count. No expansion needed.")
            return base_features
        elif num_base_features > target_num_features:
            self.logger.warning(f"Base features ({num_base_features}) > target ({target_num_features}). Truncating with a Dense layer.")
            # Use a single Dense layer to reduce dimensions
            final_features = Dense(target_num_features, activation='tanh', name="feature_reduction")(base_features)
        else:
            # Expand features using Dense layers if base < target
            # Example: 23 -> 37 -> 51 (if target is 51)
            # Adjust intermediate layer size based on difference
            intermediate_dim = max(num_base_features, (num_base_features + target_num_features) // 2)
            
            expanded_intermediate = Dense(intermediate_dim, activation='tanh', name="feature_expansion_intermediate")(base_features)
            final_features = Dense(target_num_features, activation='tanh', name="final_feature_expansion")(expanded_intermediate)
        
        return final_features

    def build_model(self) -> None:
        """Public interface for building the composite generator model."""
        # Ensure VAE decoder model path is available if needed
        vae_decoder_path = self.params.get("sequential_model_file")
        if vae_decoder_path:
            self._load_model(vae_decoder_path) # This builds the composite generator with VAE
        else:
            # Build a simple generator if no VAE decoder is specified (or handle as error)
            self.logger.warning("No VAE decoder model path specified. Building fallback or simple generator.")
            self._build_composite_generator() # Builds a simple one if vae_decoder_model is None
