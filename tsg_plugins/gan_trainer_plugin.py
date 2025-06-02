# optimizer/plugins/gan_plugin.py

from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, GRU, Bidirectional, Concatenate, LayerNormalization, Dropout, LeakyReLU, Reshape, Flatten, Activation, Add, Multiply, Attention, Embedding, Conv1D, MaxPooling1D, GlobalAveragePooling1D, TimeDistributed, BatchNormalization, SpatialDropout1D, RepeatVector
from tensorflow.keras.optimizers import Adam, RMSprop, SGD # Keep SGD if used, otherwise can remove
from tensorflow.keras.metrics import Precision, Recall, AUC, MeanSquaredError, MeanAbsoluteError
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard, CSVLogger
from tensorflow.keras.utils import plot_model
from tensorflow.keras.regularizers import l1, l2, l1_l2

import logging # Ensure logging is imported
import numpy as np
import pandas as pd
import os # For path operations
import json # For saving metrics
import time # For timing epochs
import matplotlib.pyplot as plt # For plotting losses

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

from copy import deepcopy
import tensorflow as tf # Added for tf.Tensor usage in TensorFlowTALayer if not already there

# Initialize logger for this module (if not configured at a higher level)
# logger = logging.getLogger(__name__) # This can be removed if each class instance has its own logger.

# ... TensorFlowTALayer class definition ...
# (Assuming TensorFlowTALayer class is defined above GANTrainerPlugin as in the provided file structure)
# If TensorFlowTALayer is not in this file, this comment can be removed.
# For brevity, TensorFlowTALayer is not repeated here if it's unchanged and already in the file.
# If it needs to be included or is part of the changes, it should be here.
# class TensorFlowTALayer(keras.layers.Layer):
#     ... (full class code) ...


class TensorFlowTALayer(keras.layers.Layer):
    pass # ADDED - Make it a valid empty class for now
    

class GANTrainerPlugin:
    plugin_params: Dict[str, Any] = {
        "gan_epochs": 10000,
        "gan_batch_size": 32,
        "generator_lr": 1e-4,
        "generator_beta1": 0.5,
        "discriminator_lr": 1e-4,
        "discriminator_beta1": 0.5,
        "gan_save_interval": 500,
        "discriminator_conv_filters": [64, 128],
        "discriminator_conv_kernel_size": 3,
        "discriminator_lstm_units": 64,
        "discriminator_dropout_rate": 0.3,
        "enable_reduce_lr_on_plateau": True,
        "lr_reduction_factor": 0.5,
        "lr_patience": 50,
        "lr_min_delta": 0.001,
        "min_lr_g": 1e-7,
        "min_lr_d": 1e-7,
        "lr_monitor_metric": "g_loss",
        "enable_early_stopping": True,
        "es_patience": 200,
        "es_min_delta": 0.001,
        "es_monitor_metric": "g_loss",
        "results_base_dir": "examples/results/gan_training",
        "save_model_dir": "models",
        "save_plot_dir": "plots",
        "save_metrics_dir": "metrics",
        "generator_model_plot_file": "generator_architecture.png",
        "discriminator_model_plot_file": "discriminator_architecture.png",
        "gan_model_plot_file": "gan_architecture.png",
        "model_plot_dpi": 300,
        "save_generator_epoch_template": "generator_epoch_{epoch}.keras",
        "final_generator_model_filename": "generator_final.keras",
        "save_discriminator_epoch_template": "discriminator_epoch_{epoch}.keras",
        "final_discriminator_model_filename": "discriminator_final.keras",
        "save_gan_epoch_template": "gan_epoch_{epoch}.keras",
        "final_gan_model_filename": "gan_final.keras",
        "loss_plot_epoch_template": "loss_plot_epoch_{epoch}.png",
        "final_loss_plot_filename": "loss_plot_final.png",
        "loss_plot_dpi": 300,
        "training_metrics_filename": "training_metrics.json",
        "discriminator_input_feature_names": ["OPEN", "HIGH", "LOW", "CLOSE"], # Base features D expects if no TIs
        "ti_names_for_discriminator": [], # List of TI names to calculate and append for D
        "base_feature_names_ordered_for_ti_calc": [], # Order of base features for TI layer
        "feeder_date_feature_names_for_conditioning": ["day_of_month", "hour_of_day", "day_of_week"],
        "feeder_max_day_of_month": 31.0,
        "feeder_max_hour_of_day": 23.0, # Corrected from 24.0 if 0-23
        "feeder_max_day_of_week": 6.0,  # Corrected from 7.0 if 0-6
        "conditional_fundamental_feature_names": ["S&P500_Close", "vix_close"],
        "num_conditional_prev_tick_features": 0, # Default to 0 if not used
        "datetime_col_name_in_x_real_df": "DATE_TIME",
        "generator_model_path": None, # Path to a pre-trained generator Keras model
        "latent_dim": 100, # Default latent_dim, will be used by _initialize_core_parameters...
        "seq_len": None, # Default seq_len, will be derived or taken from config
        "gan_trainer_log_level": "INFO",
        "feeder_key_name_noise": "noise_vector",
        "feeder_key_name_conditional": "conditional_data",
        "feeder_key_name_context": "context_vector",
        "feeder_key_name_window": "window_data", # For generator input if it uses windowed real data
        "use_ti_layer_in_discriminator": False, # Controls if TensorFlowTALayer is added to D
        "plot_sample_generated_data_interval": 0 # 0 means disabled, >0 means plot N samples every X epochs
    }

    def __init__(self, config: Dict[str, Any], 
                 generator_plugin_instance: Optional[Any] = None, 
                 feeder_plugin_instance: Optional[Any] = None, 
                 preprocessor_plugin_instance: Optional[Any] = None):
        
        # Combine provided config with plugin defaults. Provided config takes precedence.
        # self.params = {**self.plugin_params, **deepcopy(config)} # Python 3.5+
        # For broader compatibility or if plugin_params should be the base:
        temp_params = deepcopy(self.plugin_params)
        temp_params.update(deepcopy(config))
        self.params = temp_params

        self.generator_plugin_instance = generator_plugin_instance
        self.feeder_plugin_instance = feeder_plugin_instance
        self.preprocessor_plugin_instance = preprocessor_plugin_instance

        # Initialize logger for this instance
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        if not self.logger.handlers: # Configure only if no handlers are already set
            # handler = logging.StreamHandler() # Default to console
            # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            # handler.setFormatter(formatter)
            # self.logger.addHandler(handler)
            # self.logger.propagate = False # Avoid double logging if root logger is configured
            pass # Assuming logging is configured at a higher level

        log_level_str = self.params.get("gan_trainer_log_level", "INFO").upper()
        if hasattr(logging, log_level_str):
            self.logger.setLevel(getattr(logging, log_level_str))
        else:
            self.logger.setLevel(logging.INFO)
            self.logger.warning(f"Invalid log level \\'{log_level_str}\\' in config. Defaulting to INFO.")

        self.logger.info("GANTrainerPlugin: Initializing...")
        
        self.generator: Optional[Model] = None 
        if self.generator_plugin_instance and hasattr(self.generator_plugin_instance, 'get_model'):
            self.generator = self.generator_plugin_instance.get_model()
            if self.generator:
                self.logger.info("Successfully retrieved generator model from generator_plugin_instance.")
            else:
                self.logger.warning("generator_plugin_instance.get_model() returned None.")
        
        if not self.generator and self.params.get("generator_model_path"):
            try:
                gen_path = self.params["generator_model_path"]
                if os.path.exists(gen_path):
                    self.generator = keras.models.load_model(gen_path, compile=False)
                    self.logger.info(f"Successfully loaded generator model from path: {gen_path}")
                else:
                    self.logger.warning(f"Generator model path does not exist: {gen_path}.")
            except Exception as e:
                self.logger.error(f"Failed to load generator model from path {self.params.get('generator_model_path')}: {e}.")
        
        if not self.generator:
             self.logger.warning("Generator model not available at __init__. It must be loaded or provided before training, or built if GANTrainerPlugin is responsible.")


        self.discriminator: Optional[Model] = None
        self.gan_model: Optional[Model] = None

        # These will be properly set in _initialize_core_parameters_from_config
        self.seq_len: Optional[int] = None
        self.num_base_features: Optional[int] = None # Features output by Generator (e.g., OHLC)
        self.num_features_for_discriminator: Optional[int] = None # Total features D sees (base + TIs)
        
        self.latent_dim_for_generator: Optional[int] = None
        self.conditional_dim_for_generator: Optional[int] = None
        self.context_dim_for_generator: Optional[int] = None
        self.window_dim_for_generator: Optional[int] = None # If generator uses windowed input
        self.gen_input_seq_len: Optional[int] = None # Seq len for latent if it's sequential

        self.conditional_dim_for_discriminator: Optional[int] = None
        self.context_dim_for_discriminator: Optional[int] = None
        
        self.generator_actual_input_names_ordered: List[str] = []
        self.gan_latent_input_keras_name_hint: Optional[str] = None
        self.gan_conditional_input_keras_name_hint: Optional[str] = None
        self.gan_context_input_keras_name_hint: Optional[str] = None
        self.gan_window_input_keras_name_hint: Optional[str] = None
        
        self.feeder_key_name_noise = self.params.get("feeder_key_name_noise", "noise_vector")
        self.feeder_key_name_conditional = self.params.get("feeder_key_name_conditional", "conditional_data")
        self.feeder_key_name_context = self.params.get("feeder_key_name_context", "context_vector")
        self.feeder_key_name_window = self.params.get("feeder_key_name_window", "window_data")

        self.results_base_dir: str = self.params.get("results_base_dir", "results/gan_trainer")
        self.models_dir: str = os.path.join(self.results_base_dir, self.params.get("save_model_dir", "models"))
        self.plots_dir: str = os.path.join(self.results_base_dir, self.params.get("save_plot_dir", "plots"))
        self.metrics_dir: str = os.path.join(self.results_base_dir, self.params.get("save_metrics_dir", "metrics"))
        
        self.model_plot_dpi: int = self.params.get("model_plot_dpi", 300)
        
        self.logger.info("GANTrainerPlugin: Basic initialization complete. Call set_params() to finalize setup and build models.")

    def _get_config_param(self, key: str, default: Any = None) -> Any:
        """Helper method to get a parameter from self.params with a default value."""
        return self.params.get(key, default)

    def _initialize_core_parameters_from_config(self):
        self.logger.info("Starting _initialize_core_parameters_from_config...")

        # --- 1. Generator Output Dimensions (for Discriminator input shape) ---
        self.seq_len = None
        self.num_base_features = None 

        if self.generator:
            if hasattr(self.generator, 'output_shape') and self.generator.output_shape:
                self.logger.info(f"Attempting to derive dimensions from generator.output_shape: {self.generator.output_shape}")
                shape = self.generator.output_shape
                if isinstance(shape, tuple) and len(shape) == 3: # (batch, seq_len, features)
                    self.seq_len = shape[1]
                    self.num_base_features = shape[2]
                    self.logger.info(f"  Derived from 3D generator.output_shape: seq_len={self.seq_len}, num_base_features={self.num_base_features}")
                elif isinstance(shape, tuple) and len(shape) == 2: # (batch, features) -> implies seq_len=1 or needs config
                    self.num_base_features = shape[1]
                    self.logger.info(f"  Derived num_base_features={self.num_base_features} from 2D generator.output_shape. Seq_len will be sought from config or default to 1.")
                    # Seq_len needs to be found from config or defaults later
                else:
                    self.logger.warning(f"  Generator output_shape {shape} has unexpected format. Cannot reliably derive seq_len and num_base_features.")
            elif hasattr(self.generator, 'output') and hasattr(self.generator.output, 'shape'): # Fallback for functional models
                self.logger.info(f"Attempting to derive dimensions from generator.output.shape: {self.generator.output.shape}")
                shape = self.generator.output.shape
                if len(shape) == 3: # (None, seq_len, features)
                    self.seq_len = shape[1].value if hasattr(shape[1], 'value') else shape[1]
                    self.num_base_features = shape[2].value if hasattr(shape[2], 'value') else shape[2]
                    self.logger.info(f"  Derived from 3D generator.output.shape: seq_len={self.seq_len}, num_base_features={self.num_base_features}")
                elif len(shape) == 2: # (None, features)
                    self.num_base_features = shape[1].value if hasattr(shape[1], 'value') else shape[1]
                    self.logger.info(f"  Derived num_base_features={self.num_base_features} from 2D generator.output.shape. Seq_len will be sought from config.")
                else:
                    self.logger.warning(f"  Generator output.shape {shape} has unexpected format.")
        else:
            self.logger.info("Generator model not available. Output dimensions will rely on config parameters.")

        # Fallback/Override for seq_len from config
        if self.seq_len is None:
            self.seq_len = self._get_config_param('gan_generator_output_actual_seq_len', self._get_config_param('seq_len'))
            self.logger.info(f"Seq_len (from model or config): {self.seq_len}")
        
        # Fallback/Override for num_base_features from config
        if self.num_base_features is None:
            self.num_base_features = self._get_config_param('gan_generator_output_actual_features', 
                                    self._get_config_param('num_base_features_generated', 
                                    self._get_config_param('feature_dim'))) # 'feature_dim' as last resort
            self.logger.info(f"Num_base_features (from model or config): {self.num_base_features}")

        # Fallback for num_base_features from generator_plugin_instance (if GANTrainer didn't get it from model)
        if self.num_base_features is None and self.generator_plugin_instance:
            self.logger.info("Attempting to derive num_base_features from generator_plugin_instance.params['decoder_output_feature_names']")
            gen_plugin_decoder_outputs = self.generator_plugin_instance.params.get('decoder_output_feature_names')
            if gen_plugin_decoder_outputs and isinstance(gen_plugin_decoder_outputs, list):
                self.num_base_features = len(gen_plugin_decoder_outputs)
                self.logger.info(f"  Derived num_base_features from len(decoder_output_feature_names): {self.num_base_features}")
            else:
                self.logger.info(f"  'decoder_output_feature_names' not found or not a list in generator_plugin_instance.params.")
        
        # Convert to int if found, handle potential errors
        for attr_name in ['seq_len', 'num_base_features']:
            val = getattr(self, attr_name)
            if val is not None:
                try:
                    setattr(self, attr_name, int(val))
                except ValueError:
                    self.logger.error(f"Could not convert derived/configured {attr_name} ('{val}') to int. Resetting to None.")
                    setattr(self, attr_name, None)
        
        # If seq_len derived from 2D output shape and still None from config, default to 1
        if self.generator and hasattr(self.generator, 'output_shape') and isinstance(self.generator.output_shape, tuple) and len(self.generator.output_shape) == 2 and self.seq_len is None:
            self.logger.info("Generator output is 2D and seq_len not found in config. Defaulting seq_len to 1.")
            self.seq_len = 1

        # Critical check for generator output dimensions
        error_messages = []
        if self.seq_len is None: error_messages.append("Generator output sequence length (seq_len)")
        if self.num_base_features is None: error_messages.append("Generator output features (num_base_features)")
        if error_messages:
            msg = ("Critical: Could not determine: " + ", ".join(error_messages) +
                   ". Check generator model structure and/or provide in config: " +
                   "'gan_generator_output_actual_seq_len' or 'seq_len', and " +
                   "'gan_generator_output_actual_features' or 'num_base_features_generated' or 'feature_dim'.")
            self.logger.error(msg)
            raise ValueError(msg)
        self.logger.info(f"Final Generator Output Dimensions: seq_len={self.seq_len}, num_base_features={self.num_base_features}")

        # --- 2. Generator Input Dimensions (for GAN model construction) ---
        self.latent_dim_for_generator = self._get_config_param('latent_dim', 100)
        self.conditional_dim_for_generator = self._get_config_param('conditional_dim') # Can be None
        self.context_dim_for_generator = self._get_config_param('context_dim')       # Can be None
        self.window_dim_for_generator = self._get_config_param('window_dim')         # Can be None (features for windowed input)
        self.gen_input_seq_len = self._get_config_param('gen_input_seq_len') # Seq len for latent if sequential, can be None

        self.logger.info(f"Initial Generator Input Dims (from config): latent={self.latent_dim_for_generator}, conditional={self.conditional_dim_for_generator}, context={self.context_dim_for_generator}, window_features={self.window_dim_for_generator}, latent_seq_len={self.gen_input_seq_len}")

        self.gan_latent_input_keras_name_hint = None
        self.gan_conditional_input_keras_name_hint = None
        self.gan_context_input_keras_name_hint = None
        self.gan_window_input_keras_name_hint = None
        self.generator_actual_input_names_ordered = []

        if self.generator and hasattr(self.generator, 'inputs') and self.generator.inputs:
            self.logger.info(f"Analyzing {len(self.generator.inputs)} inputs of the loaded generator model...")
            # Store all input names from Keras model to preserve order for GAN construction if hints are used
            # self.generator_actual_input_names_ordered = [inp.name for inp in self.generator.inputs] # This might be too early if names are generic

            # Try to identify inputs by common names or configured feeder keys
            # These feeder keys are from FeederPlugin and should match how Generator expects them
            # (e.g., generator's Keras Input layers might be named after these keys)
            
            # Convert Keras input shapes to dims, handling None for batch size
            def get_dim_from_shape(shape_val):
                return shape_val.value if hasattr(shape_val, 'value') else shape_val

            for i, keras_input_tensor in enumerate(self.generator.inputs):
                name = keras_input_tensor.name.lower()
                shape = keras_input_tensor.shape
                self.logger.info(f"  Gen Input {i}: Keras name='{keras_input_tensor.name}', Keras shape={shape}")

                # Check for Latent
                if any(k in name for k in ["latent", "noise", "z_input", self.feeder_key_name_noise.lower()]):
                    if len(shape) == 2: # (batch, latent_dim)
                        self.latent_dim_for_generator = get_dim_from_shape(shape[1])
                        self.gen_input_seq_len = None # Flat latent vector
                        self.logger.info(f"    Matched LATENT (2D): dim={self.latent_dim_for_generator}")
                    elif len(shape) == 3: # (batch, seq_len, latent_dim_per_step)
                        self.gen_input_seq_len = get_dim_from_shape(shape[1])
                        self.latent_dim_for_generator = get_dim_from_shape(shape[2])
                        self.logger.info(f"    Matched LATENT (3D): seq_len={self.gen_input_seq_len}, dim_per_step={self.latent_dim_for_generator}")
                    self.gan_latent_input_keras_name_hint = keras_input_tensor.name
                # Check for Conditional
                elif self.feeder_key_name_conditional.lower() in name or "condition" in name:
                    self.conditional_dim_for_generator = get_dim_from_shape(shape[-1]) # Assume features are last dim
                    self.logger.info(f"    Matched CONDITIONAL: dim={self.conditional_dim_for_generator} (from shape {shape})")
                    self.gan_conditional_input_keras_name_hint = keras_input_tensor.name
                # Check for Context
                elif self.feeder_key_name_context.lower() in name or "context" in name:
                    self.context_dim_for_generator = get_dim_from_shape(shape[-1]) # Assume features are last dim
                    self.logger.info(f"    Matched CONTEXT: dim={self.context_dim_for_generator} (from shape {shape})")
                    self.gan_context_input_keras_name_hint = keras_input_tensor.name
                # Check for Windowed Input (e.g. from VAE encoder for sequential generation)
                elif self.feeder_key_name_window.lower() in name or "window" in name:
                    if len(shape) == 3: # (batch, window_seq_len, window_features)
                         # self.window_seq_len_for_generator = get_dim_from_shape(shape[1]) # If needed
                        self.window_dim_for_generator = get_dim_from_shape(shape[2])
                        self.logger.info(f"    Matched WINDOW: features_dim={self.window_dim_for_generator} (from shape {shape})")
                    elif len(shape) == 2: # (batch, window_features_flat) - less common for seq gen
                        self.window_dim_for_generator = get_dim_from_shape(shape[1])
                        self.logger.info(f"    Matched WINDOW (2D): features_dim={self.window_dim_for_generator} (from shape {shape})")
                    self.gan_window_input_keras_name_hint = keras_input_tensor.name
                else:
                    self.logger.warning(f"    Input '{keras_input_tensor.name}' did not match known patterns (latent, conditional, context, window). Order might be important if names are generic.")
            
            # If conditional_dim_for_generator is still None (not in config, not model-derived by name)
            if self.conditional_dim_for_generator is None:
                num_date_feats = len(self._get_config_param("feeder_date_feature_names_for_conditioning", []))
                num_fund_feats = len(self._get_config_param("conditional_fundamental_feature_names", []))
                num_prev_tick_feats = self._get_config_param("num_conditional_prev_tick_features", 0)
                calculated_cond_dim = num_date_feats + num_fund_feats + num_prev_tick_feats
                if calculated_cond_dim > 0:
                    self.conditional_dim_for_generator = calculated_cond_dim
                    self.logger.info(f"Derived 'conditional_dim_for_generator' ({self.conditional_dim_for_generator}) from sum of configured feature name list lengths.")
                else:
                    self.logger.info("'conditional_dim_for_generator' is None and could not be derived from feature name lists.")
        else: # No generator model or no inputs
            self.logger.info("Generator model or its inputs not available. GAN input dimensions based purely on config.")
            if self.conditional_dim_for_generator is None: # Calculate if not in config
                num_date_feats = len(self._get_config_param("feeder_date_feature_names_for_conditioning", []))
                num_fund_feats = len(self._get_config_param("conditional_fundamental_feature_names", []))
                num_prev_tick_feats = self._get_config_param("num_conditional_prev_tick_features", 0)
                calculated_cond_dim = num_date_feats + num_fund_feats + num_prev_tick_feats
                if calculated_cond_dim > 0:
                    self.conditional_dim_for_generator = calculated_cond_dim
                    self.logger.info(f"Derived 'conditional_dim_for_generator' ({self.conditional_dim_for_generator}) from sum of configured feature name list lengths (generator not available).")

        # Validate final generator input dims
        if self.latent_dim_for_generator is None or self.latent_dim_for_generator <= 0:
            self.logger.error("Latent dimension for generator is invalid or not set. Check 'latent_dim' in config or generator model structure.")
            raise ValueError("Invalid latent_dim_for_generator.")
        if self.conditional_dim_for_generator is not None and self.conditional_dim_for_generator <= 0:
            self.logger.info("Conditional dimension for generator is <= 0. Setting to None (no conditional input).")
            self.conditional_dim_for_generator = None
        if self.context_dim_for_generator is not None and self.context_dim_for_generator <= 0:
            self.logger.info("Context dimension for generator is <= 0. Setting to None (no context input).")
            self.context_dim_for_generator = None
        if self.window_dim_for_generator is not None and self.window_dim_for_generator <= 0:
            self.logger.info("Window dimension for generator is <= 0. Setting to None (no window input).")
            self.window_dim_for_generator = None

        self.logger.info(f"Final Generator Input Dims (for GAN): latent={self.latent_dim_for_generator} (seq_len={self.gen_input_seq_len}, hint='{self.gan_latent_input_keras_name_hint}'), conditional={self.conditional_dim_for_generator} (hint='{self.gan_conditional_input_keras_name_hint}'), context={self.context_dim_for_generator} (hint='{self.gan_context_input_keras_name_hint}'), window_features={self.window_dim_for_generator} (hint='{self.gan_window_input_keras_name_hint}')")

        # --- 3. Discriminator Input Dimensions ---
        # Discriminator primarily sees generator's output (seq_len, num_base_features)
        # It might also see TIs (if TensorFlowTALayer is used or TIs are pre-calculated)
        # And potentially conditional/context data.

        self.ti_names_for_discriminator = self._get_config_param("ti_names_for_discriminator", [])
        num_ti_features = len(self.ti_names_for_discriminator) if self.ti_names_for_discriminator else 0
        
        if self._get_config_param("use_ti_layer_in_discriminator", False) and num_ti_features > 0:
            # If TI layer is used, it calculates TIs from base features.
            # So, the TI layer's input is num_base_features, and its output is num_base_features + num_ti_features.
            # The discriminator's subsequent layers will see this combined feature set.
            self.num_features_for_discriminator = self.num_base_features + num_ti_features
            self.logger.info(f"TensorFlowTALayer will be used in Discriminator. Input to D's main body: {self.num_features_for_discriminator} features (base={self.num_base_features} + TIs={num_ti_features}).")
        else:
            # If no TI layer, or TIs are already part of generator's output (unlikely for this setup)
            # or TIs are fed as a separate input (not handled by this simple flag)
            self.num_features_for_discriminator = self.num_base_features
            if num_ti_features > 0:
                 self.logger.warning(f"'ti_names_for_discriminator' are specified, but 'use_ti_layer_in_discriminator' is false. TIs will not be calculated by a layer in D. Discriminator will see {self.num_features_for_discriminator} features.")
            else:
                self.logger.info(f"No TI layer in Discriminator. Discriminator will see {self.num_features_for_discriminator} base features from generator.")

        self.conditional_dim_for_discriminator = self._get_config_param('discriminator_conditional_dim', self.conditional_dim_for_generator)
        self.context_dim_for_discriminator = self._get_config_param('discriminator_context_dim', self.context_dim_for_generator)

        if self.conditional_dim_for_discriminator is not None and self.conditional_dim_for_discriminator <= 0:
            self.logger.info("Discriminator conditional_dim is <= 0. Setting to None.")
            self.conditional_dim_for_discriminator = None
        if self.context_dim_for_discriminator is not None and self.context_dim_for_discriminator <= 0:
            self.logger.info("Discriminator context_dim is <= 0. Setting to None.")
            self.context_dim_for_discriminator = None
            
        self.logger.info(f"Discriminator Input Config: main_features={self.num_features_for_discriminator} (at seq_len={self.seq_len}), conditional_dim={self.conditional_dim_for_discriminator}, context_dim={self.context_dim_for_discriminator}")
        self.logger.info("_initialize_core_parameters_from_config completed.")

    def _build_discriminator(self) -> None:
        self.logger.info("Building Discriminator...")
        if self.seq_len is None or self.num_features_for_discriminator is None:
            self.logger.error("Cannot build discriminator: seq_len or num_features_for_discriminator is not set.")
            raise ValueError("Discriminator dimensions not initialized.")

        # Main input for real/fake samples
        # Shape: (seq_len, num_features_for_discriminator)
        # num_features_for_discriminator already accounts for base_features + TIs if TI layer is used
        main_input_shape = (self.seq_len, self.num_features_for_discriminator)
        main_input_layer = Input(shape=main_input_shape, name="discriminator_main_input")
        x = main_input_layer

        # Optional: Add TensorFlowTALayer if configured
        if self._get_config_param("use_ti_layer_in_discriminator", False) and self.ti_names_for_discriminator:
            if not self.num_base_features: # Should have been set if TIs are to be calculated from base
                self.logger.error("Cannot add TI layer: num_base_features is not set.")
                raise ValueError("num_base_features required for TI layer.")
            
            base_feature_names_for_ti = self._get_config_param("base_feature_names_ordered_for_ti_calc", 
                                                               self._get_config_param("discriminator_input_feature_names")) # Fallback
            if not base_feature_names_for_ti:
                self.logger.error("Cannot add TI layer: 'base_feature_names_ordered_for_ti_calc' or 'discriminator_input_feature_names' (for base features) not provided in config.")
                raise ValueError("Base feature names required for TI layer.")

            self.logger.info(f"Adding TensorFlowTALayer to Discriminator. Input to layer: {self.num_base_features} base features. TIs to calculate: {self.ti_names_for_discriminator}")
            # The TI layer expects input of shape (batch, seq_len, num_base_features)
            # If main_input_layer is already (seq_len, num_base_features + TIs), this is wrong.
            # The TI layer should be applied to the *base features only*.
            # This implies that if use_ti_layer_in_discriminator=True, then num_features_for_discriminator
            # should initially be num_base_features for the input layer, and the TI layer expands it.
            # Let's adjust the logic:
            
            # If TI layer is used, the main_input_layer should represent ONLY the base features.
            # The self.num_features_for_discriminator will be the output features of the TI layer.
            ti_layer_input_shape = (self.seq_len, self.num_base_features) # TI layer gets base features
            ti_layer_input = Input(shape=ti_layer_input_shape, name="discriminator_base_features_input_for_ti")
            
            ti_layer = TensorFlowTALayer(
                base_feature_names=base_feature_names_for_ti,
                ti_names_to_calculate=self.ti_names_for_discriminator,
                num_base_features=self.num_base_features, # num_base_features from generator
                num_total_features=self.num_features_for_discriminator, # base + TIs
                seq_len=self.seq_len,
                name="discriminator_ti_layer"
            )
            x = ti_layer(ti_layer_input) # Output is (seq_len, num_base_features + num_TIs)
            # The actual input to the discriminator model will be ti_layer_input
            # And subsequent layers operate on 'x'
            current_input_layers = [ti_layer_input]
        else:
            # No TI layer, main_input_layer is used directly
            x = main_input_layer
            current_input_layers = [main_input_layer]


        # Discriminator Architecture (example using Conv1D and LSTM)
        conv_filters = self._get_config_param("discriminator_conv_filters", [64, 128])
        conv_kernel_size = self._get_config_param("discriminator_conv_kernel_size", 3)
        lstm_units = self._get_config_param("discriminator_lstm_units", 64)
        dropout_rate = self._get_config_param("discriminator_dropout_rate", 0.3)

        for filters in conv_filters:
            x = Conv1D(filters=filters, kernel_size=conv_kernel_size, padding="same")(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = Dropout(dropout_rate)(x)
        
        # x = LSTM(lstm_units, return_sequences=False)(x) # Or True if more layers follow
        # Using Bidirectional LSTM
        x = Bidirectional(LSTM(lstm_units, return_sequences=False))(x)
        x = Dropout(dropout_rate)(x)
        
        # Optional: Concatenate conditional/context data if provided
        additional_inputs_for_d = []
        if self.conditional_dim_for_discriminator and self.conditional_dim_for_discriminator > 0:
            conditional_input_d = Input(shape=(self.conditional_dim_for_discriminator,), name="discriminator_conditional_input")
            current_input_layers.append(conditional_input_d)
            # Repeat conditional vector to match sequence length if needed, or process differently
            # For simplicity, just concatenate to the flattened LSTM output
            # This might need adjustment based on how conditioning is best applied.
            # If x is flattened (return_sequences=False from LSTM):
            x = Concatenate()([x, conditional_input_d])
            self.logger.info(f"Discriminator will receive conditional input of dim {self.conditional_dim_for_discriminator}")
            additional_inputs_for_d.append(conditional_input_d)

        if self.context_dim_for_discriminator and self.context_dim_for_discriminator > 0:
            context_input_d = Input(shape=(self.context_dim_for_discriminator,), name="discriminator_context_input")
            current_input_layers.append(context_input_d)
            x = Concatenate()([x, context_input_d])
            self.logger.info(f"Discriminator will receive context input of dim {self.context_dim_for_discriminator}")
            additional_inputs_for_d.append(context_input_d)

        # Output layer
        output_layer = Dense(1, activation='sigmoid', name="discriminator_output")(x)
        
        self.discriminator = Model(current_input_layers, output_layer, name="Discriminator")
        
        d_optimizer = Adam(learning_rate=self._get_config_param("discriminator_lr", 1e-4), 
                           beta_1=self._get_config_param("discriminator_beta1", 0.5))
        self.discriminator.compile(loss='binary_crossentropy', optimizer=d_optimizer, metrics=['accuracy', Precision(), Recall(), AUC()])
        self.logger.info("Discriminator built and compiled.")
        # self.discriminator.summary(print_fn=self.logger.info) # Summary printed in set_params

    def _build_gan(self) -> None:
        if not self.generator or not self.discriminator:
            self.logger.error("Cannot build GAN: Generator or Discriminator not available.")
            return
        
        self.logger.info("Building GAN model...")
        # Freeze discriminator weights during GAN training
        self.discriminator.trainable = False

        # GAN inputs: Latent vector, and optionally conditional/context data for the generator
        gan_inputs_list = []
        generator_inputs_for_gan = [] # These will be fed to self.generator()

        # 1. Latent Input for Generator
        if self.gen_input_seq_len and self.gen_input_seq_len > 0: # Sequential latent
            latent_input_shape_for_gan = (self.gen_input_seq_len, self.latent_dim_for_generator)
        else: # Flat latent
            latent_input_shape_for_gan = (self.latent_dim_for_generator,)
        
        # Use Keras name hint if available, otherwise default name
        latent_input_name = self.gan_latent_input_keras_name_hint if self.gan_latent_input_keras_name_hint else "gan_latent_input"
        latent_input_for_gan = Input(shape=latent_input_shape_for_gan, name=latent_input_name)
        gan_inputs_list.append(latent_input_for_gan)
        generator_inputs_for_gan.append(latent_input_for_gan)
        self.logger.info(f"GAN using Latent Input: name='{latent_input_name}', shape={latent_input_shape_for_gan}")

        # 2. Conditional Input for Generator (if used)
        if self.conditional_dim_for_generator and self.conditional_dim_for_generator > 0:
            # Determine shape for conditional input (e.g., could be (cond_dim,) or (seq_len, cond_dim_per_step))
            # For now, assume it's (conditional_dim_for_generator,)
            # This needs to match how the generator expects it.
            # If generator's conditional input is (seq_len, cond_features), this needs adjustment.
            # For simplicity, assume flat conditional vector.
            cond_input_shape_for_gan = (self.conditional_dim_for_generator,)
            cond_input_name = self.gan_conditional_input_keras_name_hint if self.gan_conditional_input_keras_name_hint else "gan_conditional_input"
            conditional_input_for_gan = Input(shape=cond_input_shape_for_gan, name=cond_input_name)
            gan_inputs_list.append(conditional_input_for_gan)
            generator_inputs_for_gan.append(conditional_input_for_gan)
            self.logger.info(f"GAN using Conditional Input for Gen: name='{cond_input_name}', shape={cond_input_shape_for_gan}")

        # 3. Context Input for Generator (if used)
        if self.context_dim_for_generator and self.context_dim_for_generator > 0:
            context_input_shape_for_gan = (self.context_dim_for_generator,) # Assume flat
            context_input_name = self.gan_context_input_keras_name_hint if self.gan_context_input_keras_name_hint else "gan_context_input"
            context_input_for_gan = Input(shape=context_input_shape_for_gan, name=context_input_name)
            gan_inputs_list.append(context_input_for_gan)
            generator_inputs_for_gan.append(context_input_for_gan)
            self.logger.info(f"GAN using Context Input for Gen: name='{context_input_name}', shape={context_input_shape_for_gan}")
        
        # 4. Windowed Input for Generator (if used, e.g., for VAE-style sequential generation)
        if self.window_dim_for_generator and self.window_dim_for_generator > 0:
            # Assume generator's window input is (some_seq_len, window_dim_for_generator)
            # The some_seq_len should be known (e.g., self.seq_len or a specific generator config)
            # For now, let's assume it's self.seq_len for simplicity, but this might need refinement.
            # This input is typically real data (e.g. x_seed or x_prev_generated_step)
            window_input_seq_len = self._get_config_param("generator_window_input_seq_len", self.seq_len) # Default to main seq_len
            window_input_shape_for_gan = (window_input_seq_len, self.window_dim_for_generator)
            window_input_name = self.gan_window_input_keras_name_hint if self.gan_window_input_keras_name_hint else "gan_window_input"
            window_input_for_gan = Input(shape=window_input_shape_for_gan, name=window_input_name)
            gan_inputs_list.append(window_input_for_gan)
            generator_inputs_for_gan.append(window_input_for_gan)
            self.logger.info(f"GAN using Window Input for Gen: name='{window_input_name}', shape={window_input_shape_for_gan}")

        # Get generator output
        # The order of inputs in generator_inputs_for_gan must match generator.inputs if Keras name hints were not perfectly aligned or multiple inputs are unnamed.
        # If Keras name hints were used and generator was built with named inputs, Keras handles matching.
        # If generator_actual_input_names_ordered was populated correctly, we could build a dict.
        # For now, rely on the list order matching the generator's expected input order.
        generated_output = self.generator(generator_inputs_for_gan) # This is the fake data

        # Discriminator's inputs for GAN:
        # 1. Main input: the generated_output
        # 2. Optional conditional/context data for Discriminator (these are GAN inputs too)
        discriminator_inputs_for_gan = []
        
        # If TI layer is used in D, D expects base features. Generator output is base features.
        if self._get_config_param("use_ti_layer_in_discriminator", False) and self.ti_names_for_discriminator:
            # The TI layer in D will take `generated_output` (which are base features)
            # and calculate TIs. So, `generated_output` is the first input to D.
            discriminator_inputs_for_gan.append(generated_output)
            self.logger.info("GAN feeding generator's output (base features) to Discriminator's TI layer input.")
        else:
            # No TI layer, D expects features directly from generator.
            discriminator_inputs_for_gan.append(generated_output)
            self.logger.info("GAN feeding generator's output directly to Discriminator's main input.")

        if self.conditional_dim_for_discriminator and self.conditional_dim_for_discriminator > 0:
            # This conditional data is for the discriminator. It must be an input to the GAN.
            # It could be the same as generator's conditional data or different.
            # For simplicity, assume it's the same if not specified otherwise.
            # If it's a new GAN input:
            # cond_input_d_shape = (self.conditional_dim_for_discriminator,)
            # cond_input_d_for_gan = Input(shape=cond_input_d_shape, name="gan_conditional_input_for_discriminator")
            # gan_inputs_list.append(cond_input_d_for_gan)
            # discriminator_inputs_for_gan.append(cond_input_d_for_gan)
            # self.logger.info(f"GAN needs separate Conditional Input for Discriminator: shape={cond_input_d_shape}")
            # For now, assume if D needs conditional, it's the same as G's conditional input if G uses one.
            if conditional_input_for_gan is not None and self.conditional_dim_for_generator == self.conditional_dim_for_discriminator :
                 discriminator_inputs_for_gan.append(conditional_input_for_gan) # Reuse G's conditional input
                 self.logger.info("GAN reusing Generator's conditional input for Discriminator.")
            else: # D needs conditional, but G doesn't or dims mismatch. This needs a new GAN input.
                cond_input_d_shape = (self.conditional_dim_for_discriminator,)
                cond_input_d_for_gan = Input(shape=cond_input_d_shape, name="gan_conditional_input_for_discriminator")
                # Check if this input is already in gan_inputs_list by name to avoid duplicates
                if not any(inp.name == cond_input_d_for_gan.name for inp in gan_inputs_list):
                    gan_inputs_list.append(cond_input_d_for_gan)
                discriminator_inputs_for_gan.append(cond_input_d_for_gan)
                self.logger.info(f"GAN needs separate Conditional Input for Discriminator: name='{cond_input_d_for_gan.name}', shape={cond_input_d_shape}")


        if self.context_dim_for_discriminator and self.context_dim_for_discriminator > 0:
            # Similar logic for context data for D
            if context_input_for_gan is not None and self.context_dim_for_generator == self.context_dim_for_discriminator:
                discriminator_inputs_for_gan.append(context_input_for_gan) # Reuse G's context input
                self.logger.info("GAN reusing Generator's context input for Discriminator.")
            else:
                context_input_d_shape = (self.context_dim_for_discriminator,)
                context_input_d_for_gan = Input(shape=context_input_d_shape, name="gan_context_input_for_discriminator")
                if not any(inp.name == context_input_d_for_gan.name for inp in gan_inputs_list):
                    gan_inputs_list.append(context_input_d_for_gan)
                discriminator_inputs_for_gan.append(context_input_d_for_gan)
                self.logger.info(f"GAN needs separate Context Input for Discriminator: name='{context_input_d_for_gan.name}', shape={context_input_d_shape}")
        
        # Pass inputs to discriminator
        gan_output = self.discriminator(discriminator_inputs_for_gan)
        
        self.gan_model = Model(gan_inputs_list, gan_output, name="GAN")
        
        g_optimizer = Adam(learning_rate=self._get_config_param("generator_lr", 1e-4), 
                           beta_1=self._get_config_param("generator_beta1", 0.5))
        self.gan_model.compile(loss='binary_crossentropy', optimizer=g_optimizer, metrics=['accuracy', Precision(), Recall(), AUC()]) # Metrics on G loss are tricky
        self.logger.info("GAN model built and compiled.")
        # self.gan_model.summary(print_fn=self.logger.info) # Summary printed in set_params

    def set_params(self, **params: Any) -> None:
        self.logger.info(f"set_params called with: {params}")
        # Update self.params, new params override existing ones
        for key, value in params.items():
            self.params[key] = value
        
        # Re-initialize logger level if changed in params
        log_level_str = self.params.get("gan_trainer_log_level", "INFO").upper()
        if hasattr(logging, log_level_str) and self.logger.level != getattr(logging, log_level_str):
            self.logger.setLevel(getattr(logging, log_level_str))
            self.logger.info(f"Log level updated to {log_level_str}")

        # Re-attempt to load generator if not already loaded and path is now available/updated
        if not self.generator and self.params.get("generator_model_path"):
             try:
                gen_path = self.params["generator_model_path"]
                if os.path.exists(gen_path):
                    self.generator = keras.models.load_model(gen_path, compile=False)
                    self.logger.info(f"Successfully loaded generator model from path in set_params: {gen_path}")
                else:
                    self.logger.warning(f"Generator model path from set_params does not exist: {gen_path}.")
             except Exception as e:
                self.logger.error(f"Failed to load generator model from path in set_params {self.params.get('generator_model_path')}: {e}.")
        elif not self.generator and self.generator_plugin_instance and hasattr(self.generator_plugin_instance, 'get_model'):
            # This case might be redundant if __init__ already tried, but good for late binding
            self.generator = self.generator_plugin_instance.get_model()
            if self.generator:
                self.logger.info("Successfully retrieved generator model from generator_plugin_instance in set_params.")
            else: # This could happen if plugin exists but its model isn't ready yet
                self.logger.warning("generator_plugin_instance.get_model() returned None in set_params. Generator might need to be built or loaded by its own plugin first.")

        if not self.generator:
            self.logger.critical("Generator model is NOT available after set_params. Cannot proceed to build D and GAN without a generator.")
            # Depending on workflow, this could be an error or a warning if G is built later by this plugin
            # For now, assume G must be present before D and GAN are built.
            # raise ValueError("Generator model is required but not loaded or provided.") # Uncomment if G is strictly required here

        self._initialize_core_parameters_from_config() # This relies on self.generator being available if model introspection is used

        self._ensure_dir_exists(self.models_dir)
        self._ensure_dir_exists(self.plots_dir)
        self._ensure_dir_exists(self.metrics_dir)

        if self.generator: # Only build D and GAN if G is available
            self._build_discriminator()
            self._build_gan()
        else:
            self.logger.error("Skipping build of Discriminator and GAN because Generator is not available.")


        self.logger.info("set_params complete.")
        if self.generator: self.generator.summary(print_fn=self.logger.info)
        else: self.logger.warning("Generator model summary not available.")
        if self.discriminator: self.discriminator.summary(print_fn=self.logger.info)
        else: self.logger.warning("Discriminator model summary not available.")
        if self.gan_model: self.gan_model.summary(print_fn=self.logger.info)
        else: self.logger.warning("GAN model summary not available.")

    def _ensure_dir_exists(self, dir_path: str):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            self.logger.info(f"Created directory: {dir_path}")

    def _get_scaled_date_features(self, datetimes_series: pd.Series) -> Optional[np.ndarray]:
        """
        Generates scaled date/time features from a pandas Series of datetime objects.
        Features are scaled to be generally within the 0-1 range.
        Uses 'feeder_date_feature_names_for_conditioning' and 'feeder_max_...' from self.params.
        """
        if datetimes_series is None or datetimes_series.empty:
            return None
        
        date_features_list = []
        for dt_obj in datetimes_series:
            features = []
            if "day_of_month" in self.params.get("feeder_date_feature_names_for_conditioning", []):
                features.append(dt_obj.day / self.params.get("feeder_max_day_of_month", 31.0))
            if "hour_of_day" in self.params.get("feeder_date_feature_names_for_conditioning", []):
                features.append(dt_obj.hour / self.params.get("feeder_max_hour_of_day", 23.0)) # 0-23
            if "day_of_week" in self.params.get("feeder_date_feature_names_for_conditioning", []):
                features.append(dt_obj.dayofweek / self.params.get("feeder_max_day_of_week", 6.0)) # 0-6
            if "day_of_year" in self.params.get("feeder_date_feature_names_for_conditioning", []):
                features.append(dt_obj.dayofyear / self.params.get("feeder_max_day_of_year", 366.0))
            # Add more date features as needed
            date_features_list.append(features)
        
        scaled_date_features = np.array(date_features_list)
        self.logger.debug(f"Scaled date features shape: {scaled_date_features.shape}")
        return scaled_date_features

    def _get_scaled_fundamental_features(self, fundamental_features_df_batch: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Extracts and returns specified fundamental features.
        Assumes features in fundamental_features_df_batch are already scaled (e.g., 0-1).
        Uses 'conditional_fundamental_feature_names' from self.params.
        """
        if fundamental_features_df_batch is None or fundamental_features_df_batch.empty:
            return None
        # Placeholder: assumes columns in fundamental_features_df_batch are already scaled or match expected input.
        # Actual scaling logic would depend on how these features are preprocessed.
        # For example, if preprocessor_plugin is available:
        # if self.preprocessor_plugin and hasattr(self.preprocessor_plugin, 'transform'):
        #     return self.preprocessor_plugin.transform(fundamental_features_df_batch, columns_to_transform=self.params.get("conditional_fundamental_feature_names",[]))
        
        # Simple pass-through for now, assuming they are ready or scaled by feeder.
        self.logger.debug(f"Fundamental features batch shape: {fundamental_features_df_batch.values.shape}")
        return fundamental_features_df_batch.values


    def _get_data_generator(self, 
                            x_real_processed_np_epoch: np.ndarray, 
                            real_datetimes_series_epoch: Optional[pd.Series],
                            fundamental_features_df_epoch: Optional[pd.DataFrame],
                            epoch_context_vectors_aligned: Optional[np.ndarray],
                            batch_size: int):
        num_samples = x_real_processed_np_epoch.shape[0]
        indices = np.arange(num_samples)
        
        # This generator yields batches for one epoch
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:i + batch_size]
            actual_batch_size = len(batch_indices) # Handle last batch if smaller
            if actual_batch_size == 0: continue

            real_samples_batch = x_real_processed_np_epoch[batch_indices]
            
            conditional_data_batch = None
            # Prepare conditional data (date-based and fundamental)
            # This part needs to align with how conditional_dim_for_generator was calculated
            date_cond_data = None
            if real_datetimes_series_epoch is not None and self.params.get("feeder_date_feature_names_for_conditioning"):
                batch_datetimes = real_datetimes_series_epoch.iloc[batch_indices]
                date_cond_data = self._get_scaled_date_features(batch_datetimes) # Shape (batch, num_date_features)
            
            fund_cond_data = None
            if fundamental_features_df_epoch is not None and self.params.get("conditional_fundamental_feature_names"):
                batch_fundamentals_df = fundamental_features_df_epoch.iloc[batch_indices]
                fund_cond_data = self._get_scaled_fundamental_features(batch_fundamentals_df) # Shape (batch, num_fund_features)

            # Combine conditional data
            if date_cond_data is not None and fund_cond_data is not None:
                conditional_data_batch = np.concatenate((date_cond_data, fund_cond_data), axis=1)
            elif date_cond_data is not None:
                conditional_data_batch = date_cond_data
            elif fund_cond_data is not None:
                conditional_data_batch = fund_cond_data
            
            # Add prev_tick_features if configured (this part is complex and needs careful handling of how they are sourced)
            # For now, assuming conditional_data_batch from date/fundamental is what's used if num_conditional_prev_tick_features is 0
            # If num_conditional_prev_tick_features > 0, this batching needs to provide those features.
            # This example assumes they are part of fundamental_features_df_epoch or handled by Feeder.

            context_data_batch = None
            if epoch_context_vectors_aligned is not None:
                context_data_batch = epoch_context_vectors_aligned[batch_indices]

            # Windowed data for generator (if needed)
            # This is complex: it might be a sliding window from x_real_processed_np_epoch or from FeederPlugin
            # For now, assuming FeederPlugin provides this aligned with the number of sequences.
            window_data_batch = None # Placeholder
            if self.window_dim_for_generator and self.window_dim_for_generator > 0:
                # Logic to extract/generate window_data_batch
                # Example: if it's just the real samples (e.g., for a VAE-like step)
                # window_data_batch = real_samples_batch # This assumes window_dim matches num_base_features
                # This needs to match the generator's window input spec, e.g., (batch, window_seq_len, window_features)
                self.logger.warning("Window data batching not fully implemented in _get_data_generator. Generator might not get required window input.")
                # A proper implementation would fetch appropriately shaped windowed data.

            yield real_samples_batch, conditional_data_batch, context_data_batch, window_data_batch, actual_batch_size


    def train(self, x_real_df: pd.DataFrame, start_epoch: int = 0):
        self.logger.info("Starting GAN training...")
        if not self.generator or not self.discriminator or not self.gan_model:
            self.logger.error("Models not built. Call set_params() first or ensure generator is loaded.")
            return {"status": "error", "message": "Models not built."}

        epochs = self._get_config_param("gan_epochs", 1000)
        batch_size = self._get_config_param("gan_batch_size", 32)
        save_interval = self._get_config_param("gan_save_interval", 100)
        plot_sample_interval = self._get_config_param("plot_sample_generated_data_interval", 0)


        # Prepare real data: x_real_df should contain the features the generator aims to produce.
        # These are typically normalized.
        # The columns should match `discriminator_input_feature_names` if no TI layer,
        # or `base_feature_names_ordered_for_ti_calc` if TI layer is used.
        # For now, assume x_real_df contains the base features.
        if self.num_base_features is None: # Should be set by _initialize_core_parameters_from_config
            self.logger.error("num_base_features is not set. Cannot prepare real data.")
            return {"status": "error", "message": "num_base_features not set."}

        # Select only the base feature columns that the generator produces
        # These names should come from a config that defines generator's output, e.g., 'decoder_output_feature_names'
        # or if GANTrainer defines them, 'discriminator_input_feature_names' (if they are base features)
        # For this example, let's assume 'discriminator_input_feature_names' from config holds the base feature names.
        
        # This was: self.params.get("discriminator_input_feature_names")
        # It should be the features the GENERATOR outputs, which are then fed to D.
        # Let's assume self.generator_plugin_instance.params['decoder_output_feature_names'] is the source of truth for generated feature names
        # Or, if GANTrainer is standalone, a config like 'generated_feature_names'
        
        # Using a fallback:
        generated_feature_names = []
        if self.generator_plugin_instance and self.generator_plugin_instance.params.get('decoder_output_feature_names'):
            generated_feature_names = self.generator_plugin_instance.params.get('decoder_output_feature_names')
        elif self.params.get('base_feature_names_ordered_for_ti_calc'): # If TI layer uses these as base
             generated_feature_names = self.params.get('base_feature_names_ordered_for_ti_calc')
        elif self.params.get('discriminator_input_feature_names'): # Fallback if these are indeed the base generated ones
             generated_feature_names = self.params.get('discriminator_input_feature_names')

        if not generated_feature_names:
            self.logger.error("Could not determine the names of features generated by the Generator. Check config: 'decoder_output_feature_names' (in gen plugin), 'base_feature_names_ordered_for_ti_calc', or 'discriminator_input_feature_names'.")
            return {"status": "error", "message": "Generated feature names not defined."}
        
        self.logger.info(f"Using generated_feature_names for x_real_df: {generated_feature_names}")
        
        try:
            x_real_processed_df = x_real_df[generated_feature_names]
        except KeyError as e:
            self.logger.error(f"KeyError when selecting features for training: {e}. Available columns in x_real_df: {x_real_df.columns.tolist()}. Expected: {generated_feature_names}")
            return {"status": "error", "message": f"Feature mismatch: {e}"}

        # Reshape real data if seq_len > 1: (num_samples, seq_len, num_base_features)
        # This requires x_real_df to be a 2D table that can be windowed.
        # If x_real_df is already windowed, this logic needs to change.
        # Assuming x_real_df is a flat sequence of records.
        if self.seq_len is None or self.seq_len <=0: # Should be caught by init
            self.logger.error("seq_len is invalid.")
            return {"status": "error", "message": "seq_len invalid."}

        num_total_records = len(x_real_processed_df)
        if num_total_records < self.seq_len:
            self.logger.error(f"Not enough records ({num_total_records}) in x_real_df to form sequences of length {self.seq_len}.")
            return {"status": "error", "message": "Not enough data for sequence length."}
            
        # Create sequences: (num_sequences, seq_len, num_features)
        # x_real_processed_np = np.array([
        #     x_real_processed_df.iloc[i:i+self.seq_len].values 
        #     for i in range(num_total_records - self.seq_len + 1)
        # ])
        # Simpler: assume x_real_df is ALREADY BATCHED with sequences by Feeder or Preprocessor
        # So, x_real_df.values would be (num_samples_already_sequenced, seq_len, num_features)
        # For now, let's assume x_real_df is NOT YET SEQUENCED, and contains base features.
        # The FeederPlugin should provide data in the correct shape.
        # If GANTrainer gets a flat DataFrame, it needs to window it.
        # This part is often handled by a data loader/feeder.
        # For this example, let's assume x_real_df is a DataFrame where each row is a timestep,
        # and we need to create overlapping sequences.
        
        # This windowing logic is a common source of complexity.
        # A robust FeederPlugin would yield batches of (real_samples, conditional_data, context_data)
        # where real_samples is already (batch_size, seq_len, num_base_features).
        # If GANTrainer does its own batching from a large df:
        x_real_processed_np_epoch = []
        for i in range(0, len(x_real_processed_df) - self.seq_len + 1, 1): # Stride of 1 for overlapping windows
            window = x_real_processed_df.iloc[i:i + self.seq_len].values
            x_real_processed_np_epoch.append(window)
        if not x_real_processed_np_epoch:
            self.logger.error("Failed to create any sequences from x_real_df. Check data length and seq_len.")
            return {"status": "error", "message": "Sequence creation failed."}
        x_real_processed_np_epoch = np.array(x_real_processed_np_epoch)
        self.logger.info(f"Prepared real data for training: {x_real_processed_np_epoch.shape}")


        # Prepare conditional and context data (aligned with x_real_processed_np_epoch)
        # This also depends on how Feeder provides data.
        # For simplicity, assume FeederPlugin (if used) or config provides these.
        real_datetimes_series_epoch: Optional[pd.Series] = None
        if self.params.get("datetime_col_name_in_x_real_df") in x_real_df.columns:
            # Extract datetimes corresponding to the START of each sequence if needed for cond data
            # This alignment is tricky. If cond data is per-timestep, it needs to be windowed too.
            # If cond data is per-sequence, extract for each sequence.
            # For now, assume we can get datetimes for each sequence start.
            datetime_col = self.params["datetime_col_name_in_x_real_df"]
            # real_datetimes_series_epoch = x_real_df[datetime_col].iloc[0 : len(x_real_processed_np_epoch)] # Simplistic: take from start
            # A more correct way if sequences are from x_real_df:
            real_datetimes_series_epoch = x_real_df[datetime_col].iloc[np.arange(len(x_real_processed_np_epoch))]


        fundamental_features_df_epoch: Optional[pd.DataFrame] = None
        if self.params.get("conditional_fundamental_feature_names"):
            fund_cols = self.params["conditional_fundamental_feature_names"]
            if all(col in x_real_df.columns for col in fund_cols):
                # fundamental_features_df_epoch = x_real_df[fund_cols].iloc[0 : len(x_real_processed_np_epoch)] # Simplistic
                fundamental_features_df_epoch = x_real_df[fund_cols].iloc[np.arange(len(x_real_processed_np_epoch))]
            else:
                self.logger.warning(f"Not all fundamental feature columns {fund_cols} found in x_real_df.")
        
        epoch_context_vectors_aligned: Optional[np.ndarray] = None
        if self.context_dim_for_generator and self.feeder_plugin_instance and hasattr(self.feeder_plugin_instance, 'get_aligned_context_vectors'):
            # This assumes FeederPlugin can provide context vectors aligned with the sequences in x_real_processed_np_epoch
            # The FeederPlugin would need the datetimes or indices of these sequences.
            # This is a complex dependency.
            # For now, let's assume if context is needed, Feeder provides it aligned with the number of sequences.
            # epoch_context_vectors_aligned = self.feeder_plugin_instance.get_aligned_context_vectors(target_indices_or_datetimes)
            self.logger.warning("Context vector alignment from FeederPlugin not fully implemented in training loop. Context might be missing or misaligned.")
            # Placeholder: generate dummy context vectors if needed by G or D
            if self.context_dim_for_generator > 0 or self.context_dim_for_discriminator > 0:
                 # This is just a dummy for shape, real context should come from feeder
                 dummy_context_dim = self.context_dim_for_generator if self.context_dim_for_generator else self.context_dim_for_discriminator
                 if dummy_context_dim and dummy_context_dim > 0:
                    epoch_context_vectors_aligned = np.random.randn(len(x_real_processed_np_epoch), dummy_context_dim)
                    self.logger.info(f"Using DUMMY context vectors of shape {epoch_context_vectors_aligned.shape}")


        # Training loop
        history = {"d_loss_real": [], "d_loss_fake": [], "d_acc_real": [], "d_acc_fake": [], "g_loss": [], "g_acc": []}
        
        # Callbacks for learning rate reduction
        reduce_lr_g = None
        reduce_lr_d = None
        if self._get_config_param("enable_reduce_lr_on_plateau", False):
            reduce_lr_g = ReduceLROnPlateau(monitor=self._get_config_param("lr_monitor_metric", "g_loss"), 
                                            factor=self._get_config_param("lr_reduction_factor", 0.5), 
                                            patience=self._get_config_param("lr_patience", 20), 
                                            min_delta=self._get_config_param("lr_min_delta", 0.001),
                                            min_lr=self._get_config_param("min_lr_g", 1e-7), verbose=1)
            reduce_lr_d = ReduceLROnPlateau(monitor='d_loss', # D loss is usually d_loss_real + d_loss_fake / 2
                                            factor=self._get_config_param("lr_reduction_factor", 0.5), 
                                            patience=self._get_config_param("lr_patience", 20),
                                            min_delta=self._get_config_param("lr_min_delta", 0.001),
                                            min_lr=self._get_config_param("min_lr_d", 1e-7), verbose=1)
            # These callbacks need to be associated with models, which is tricky for custom loops.
            # We'll manually adjust LR based on their logic if needed, or use them if training Keras model directly.
            # For custom loop, we might need to implement the logic of ReduceLROnPlateau manually.
            self.logger.info("ReduceLROnPlateau configured. Manual application in custom loop might be needed.")


        for epoch in range(start_epoch, epochs):
            epoch_start_time = time.time()
            
            # Shuffle data each epoch (optional, but good practice)
            # If using _get_data_generator, it can handle shuffling internally if indices are shuffled.
            # For now, assume data is used in order or _get_data_generator shuffles.
            
            epoch_d_losses_real_scalar = []
            epoch_d_losses_fake_scalar = []
            epoch_d_accs_real = []
            epoch_d_accs_fake = []
            epoch_g_losses_scalar = []
            epoch_g_accs = []

            data_loader = self._get_data_generator(x_real_processed_np_epoch, 
                                                   real_datetimes_series_epoch,
                                                   fundamental_features_df_epoch,
                                                   epoch_context_vectors_aligned,
                                                   batch_size)
            
            num_batches = (len(x_real_processed_np_epoch) -1) // batch_size + 1

            for batch_idx, (real_samples_batch, conditional_data_batch, context_data_batch_from_feeder, window_data_batch_from_feeder, actual_batch_size) in enumerate(data_loader):
                if actual_batch_size == 0: continue

                # --- Train Discriminator ---
                # 1. On Real Samples
                d_inputs_real = [real_samples_batch]
                if self.conditional_dim_for_discriminator and conditional_data_batch is not None:
                    if conditional_data_batch.shape[0] == actual_batch_size and conditional_data_batch.shape[1] == self.conditional_dim_for_discriminator:
                        d_inputs_real.append(conditional_data_batch)
                    else: # Dimension mismatch, skip or error
                        self.logger.warning(f"Epoch {epoch+1}, Batch {batch_idx+1}: Real cond data dim mismatch for D. Expected {self.conditional_dim_for_discriminator}, got {conditional_data_batch.shape if conditional_data_batch is not None else 'None'}. Skipping D real train for this batch.")
                        # Fill with zeros or skip batch for D real
                        d_loss_real_metrics = [0.0] * len(self.discriminator.metrics_names) # type: ignore
                if self.context_dim_for_discriminator and context_data_batch_from_feeder is not None:
                    if context_data_batch_from_feeder.shape[0] == actual_batch_size and context_data_batch_from_feeder.shape[1] == self.context_dim_for_discriminator:
                        d_inputs_real.append(context_data_batch_from_feeder)
                    else: # Mismatch
                        self.logger.warning(f"Epoch {epoch+1}, Batch {batch_idx+1}: Real context data dim mismatch for D. Expected {self.context_dim_for_discriminator}, got {context_data_batch_from_feeder.shape if context_data_batch_from_feeder is not None else 'None'}. Skipping D real train for this batch.")
                        d_loss_real_metrics = [0.0] * len(self.discriminator.metrics_names) # type: ignore
                
                y_real = np.ones((actual_batch_size, 1)) * 0.9 # Label smoothing for real
                if len(d_inputs_real[0]) == actual_batch_size : # Check if main input is valid for this batch size
                    d_loss_real_metrics = self.discriminator.train_on_batch(d_inputs_real, y_real)
                else: # Skip if inputs are not correctly formed for D for real samples
                    d_loss_real_metrics = [0.0] * len(self.discriminator.metrics_names) # type: ignore
                    self.logger.warning(f"Epoch {epoch+1}, Batch {batch_idx+1}: Skipping D real train due to input formation issues.")


                # 2. On Fake Samples
                # Prepare inputs for Generator
                gen_inputs_for_predict_list = []
                # Latent noise
                if self.gen_input_seq_len and self.gen_input_seq_len > 0:
                    z_noise_shape = (actual_batch_size, self.gen_input_seq_len, self.latent_dim_for_generator)
                else:
                    z_noise_shape = (actual_batch_size, self.latent_dim_for_generator)
                z_noise = np.random.normal(0, 1, z_noise_shape)
                gen_inputs_for_predict_list.append(z_noise)

                # Conditional data for Generator
                if self.conditional_dim_for_generator and conditional_data_batch is not None:
                    if conditional_data_batch.shape[0] == actual_batch_size and conditional_data_batch.shape[1] == self.conditional_dim_for_generator:
                        gen_inputs_for_predict_list.append(conditional_data_batch)
                    else: # Mismatch, cannot generate fakes properly
                         self.logger.error(f"Epoch {epoch+1}, Batch {batch_idx+1}: Cond data dim mismatch for G. Expected {self.conditional_dim_for_generator}, got {conditional_data_batch.shape}. Skipping D fake & G train.")
                         d_loss_fake_metrics = [0.0] * len(self.discriminator.metrics_names) # type: ignore
                         g_loss_metrics = [0.0] * len(self.gan_model.metrics_names) # type: ignore
                         # Add to epoch losses and continue to next batch
                         epoch_d_losses_real_scalar.append(d_loss_real_metrics[0])
                         epoch_d_accs_real.append(d_loss_real_metrics[1]) # Assuming acc is 2nd metric
                         epoch_d_losses_fake_scalar.append(0.0)
                         epoch_d_accs_fake.append(0.0)
                         epoch_g_losses_scalar.append(0.0)
                         epoch_g_accs.append(0.0)
                         continue # Skip to next batch

                # Context data for Generator
                if self.context_dim_for_generator and context_data_batch_from_feeder is not None:
                    if context_data_batch_from_feeder.shape[0] == actual_batch_size and context_data_batch_from_feeder.shape[1] == self.context_dim_for_generator:
                        gen_inputs_for_predict_list.append(context_data_batch_from_feeder)
                    else: # Mismatch
                         self.logger.error(f"Epoch {epoch+1}, Batch {batch_idx+1}: Context data dim mismatch for G. Expected {self.context_dim_for_generator}, got {context_data_batch_from_feeder.shape}. Skipping D fake & G train.")
                         # Similar handling as conditional mismatch
                         d_loss_fake_metrics = [0.0] * len(self.discriminator.metrics_names) # type: ignore
                         g_loss_metrics = [0.0] * len(self.gan_model.metrics_names) # type: ignore
                         epoch_d_losses_real_scalar.append(d_loss_real_metrics[0])
                         epoch_d_accs_real.append(d_loss_real_metrics[1])
                         epoch_d_losses_fake_scalar.append(0.0)
                         epoch_d_accs_fake.append(0.0)
                         epoch_g_losses_scalar.append(0.0)
                         epoch_g_accs.append(0.0)
                         continue
                
                # Window data for Generator
                if self.window_dim_for_generator and window_data_batch_from_feeder is not None:
                    # Add shape check for window_data_batch_from_feeder
                    # Expected: (actual_batch_size, self._get_config_param("generator_window_input_seq_len", self.seq_len), self.window_dim_for_generator)
                    expected_win_seq_len = self._get_config_param("generator_window_input_seq_len", self.seq_len)
                    if (window_data_batch_from_feeder.shape[0] == actual_batch_size and
                        window_data_batch_from_feeder.shape[1] == expected_win_seq_len and
                        window_data_batch_from_feeder.shape[2] == self.window_dim_for_generator):
                        gen_inputs_for_predict_list.append(window_data_batch_from_feeder)
                    else: # Mismatch
                        self.logger.error(f"Epoch {epoch+1}, Batch {batch_idx+1}: Window data dim mismatch for G. Expected ({actual_batch_size}, {expected_win_seq_len}, {self.window_dim_for_generator}), got {window_data_batch_from_feeder.shape}. Skipping D fake & G train.")
                        # Similar handling
                        d_loss_fake_metrics = [0.0] * len(self.discriminator.metrics_names) # type: ignore
                        g_loss_metrics = [0.0] * len(self.gan_model.metrics_names) # type: ignore
                        epoch_d_losses_real_scalar.append(d_loss_real_metrics[0])
                        epoch_d_accs_real.append(d_loss_real_metrics[1])
                        epoch_d_losses_fake_scalar.append(0.0)
                        epoch_d_accs_fake.append(0.0)
                        epoch_g_losses_scalar.append(0.0)
                        epoch_g_accs.append(0.0)
                        continue


                fake_samples = self.generator.predict(gen_inputs_for_predict_list, verbose=0)
                
                d_inputs_fake = [fake_samples]
                if self.conditional_dim_for_discriminator and conditional_data_batch is not None:
                     # Assuming D uses same conditional data as G if dims match
                    if conditional_data_batch.shape[0] == actual_batch_size and conditional_data_batch.shape[1] == self.conditional_dim_for_discriminator:
                         d_inputs_fake.append(conditional_data_batch)
                    # Else: D might not get conditional data if G didn't use it or dims differ, potential issue.
                if self.context_dim_for_discriminator and context_data_batch_from_feeder is not None:
                    if context_data_batch_from_feeder.shape[0] == actual_batch_size and context_data_batch_from_feeder.shape[1] == self.context_dim_for_discriminator:
                        d_inputs_fake.append(context_data_batch_from_feeder)

                y_fake = np.zeros((actual_batch_size, 1)) # No smoothing for fake typically
                d_loss_fake_metrics = self.discriminator.train_on_batch(d_inputs_fake, y_fake)

                # --- Train Generator (via GAN model) ---
                # Generator aims to make discriminator predict 1 (real) for its output
                gan_train_inputs = gen_inputs_for_predict_list # These are the inputs G took (noise, cond, context, window)
                                                              # # These must also be the inputs the GAN model expects for G's part.
                
                # If D also takes cond/context, and these are part of GAN inputs:
                # The _build_gan method needs to ensure gan_model expects all necessary inputs.
                # Current _build_gan tries to reuse G's cond/context for D if dims match,
                # or adds new GAN inputs if D needs them separately.
                # We need to ensure gan_train_inputs matches the full list of GAN inputs defined in _build_gan.
                
                # Example: if GAN expects [latent, g_cond, g_context, d_cond, d_context]
                # We need to provide all of them.
                # This part is tricky and relies on assumptions about the order of inputs if names are not descriptive.
                # The current strategy is: if not found by name, stick to config values. 
                # A more robust fallback might try to infer based on typical dimensionalities if there's a standard order.
                if self.conditional_dim_for_generator and self.context_dim_for_generator:
                    # If both are used, ensure order is latent, cond, context (G's expected order)
                    gan_train_inputs = [gen_inputs_for_predict_list[0], gen_inputs_for_predict_list[1], gen_inputs_for_predict_list[2]]
                elif self.conditional_dim_for_generator:
                    gan_train_inputs = [gen_inputs_for_predict_list[0], gen_inputs_for_predict_list[1]]
                elif self.context_dim_for_generator:
                    gan_train_inputs = [gen_inputs_for_predict_list[0], gen_inputs_for_predict_list[2]]
                else:
                    gan_train_inputs = gen_inputs_for_predict_list # Default: all

                y_gan = np.ones((actual_batch_size, 1)) # Generator wants D to think fakes are real
                g_loss_metrics = self.gan_model.train_on_batch(gan_train_inputs, y_gan)
                
                # Store batch losses (scalar loss is usually the first metric)
                epoch_d_losses_real_scalar.append(d_loss_real_metrics[0])
                epoch_d_accs_real.append(d_loss_real_metrics[1]) # Assuming accuracy is the second metric
                epoch_d_losses_fake_scalar.append(d_loss_fake_metrics[0])
                epoch_d_accs_fake.append(d_loss_fake_metrics[1])
                epoch_g_losses_scalar.append(g_loss_metrics[0])
                epoch_g_accs.append(g_loss_metrics[1]) # GAN 'accuracy'

                if (batch_idx + 1) % 10 == 0: # Log progress every 10 batches
                    self.logger.info(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{num_batches} "
                                     f"DLR: {d_loss_real_metrics[0]:.4f} (AccR: {d_loss_real_metrics[1]:.2f}), "
                                     f"DLF: {d_loss_fake_metrics[0]:.4f} (AccF: {d_loss_fake_metrics[1]:.2f}), "
                                     f"GL: {g_loss_metrics[0]:.4f} (AccG: {g_loss_metrics[1]:.2f})")

            # End of epoch
            avg_d_loss_real = np.mean(epoch_d_losses_real_scalar)
            avg_d_loss_fake = np.mean(epoch_d_losses_fake_scalar)
            avg_d_acc_real = np.mean(epoch_d_accs_real)
            avg_d_acc_fake = np.mean(epoch_d_accs_fake)
            avg_g_loss = np.mean(epoch_g_losses_scalar)
            avg_g_acc = np.mean(epoch_g_accs)

            history["d_loss_real"].append(avg_d_loss_real)
            history["d_loss_fake"].append(avg_d_loss_fake)
            history["d_acc_real"].append(avg_d_acc_real)
            history["d_acc_fake"].append(avg_d_acc_fake)
            history["g_loss"].append(avg_g_loss)
            history["g_acc"].append(avg_g_acc)

            epoch_duration = time.time() - epoch_start_time
            self.logger.info(f"Epoch {epoch+1}/{epochs} Summary - Duration: {epoch_duration:.2f}s")
            self.logger.info(f"  Avg D Loss Real: {avg_d_loss_real:.4f}, Avg D Acc Real: {avg_d_acc_real:.2f}")
            self.logger.info(f"  Avg D Loss Fake: {avg_d_loss_fake:.4f}, Avg D Acc Fake: {avg_d_acc_fake:.2f}")
            self.logger.info(f"  Avg G Loss: {avg_g_loss:.4f}, Avg G Acc (GAN): {avg_g_acc:.2f}")

            # Reduce LR on plateau (manual check for custom loop)
            if reduce_lr_g and self.gan_model: # Check G loss for G's LR
                # This is a simplified check. ReduceLROnPlateau has more state.
                # For a proper custom loop, you'd implement its logic or find a compatible callback.
                # reduce_lr_g.on_epoch_end(epoch, logs={'g_loss': avg_g_loss}) # This is for Keras model.fit
                pass # Manual LR adjustment would go here if needed
            if reduce_lr_d and self.discriminator: # Check D loss for D's LR
                # reduce_lr_d.on_epoch_end(epoch, logs={'d_loss': (avg_d_loss_real + avg_d_loss_fake)/2})
                pass

            if save_interval > 0 and (epoch + 1) % save_interval == 0:
                self._save_models_at_epoch(epoch + 1)
                self._plot_losses(history, epoch + 1)
            
            if plot_sample_interval > 0 and (epoch + 1) % plot_sample_interval == 0:
                self._plot_sample_generated_data(epoch + 1, num_samples_to_plot=5, 
                                                 conditional_data_for_plot=conditional_data_batch, # Use last batch's cond data
                                                 context_data_for_plot=context_data_batch_from_feeder, # Use last batch's context
                                                 window_data_for_plot=window_data_batch_from_feeder) # Use last batch's window

            # Early stopping (manual check)
            if self._get_config_param("enable_early_stopping", False):
                monitor_metric = self._get_config_param("es_monitor_metric", "g_loss")
                min_delta = self._get_config_param("es_min_delta", 0.001)
                patience = self._get_config_param("es_patience", 100)
                # This requires tracking the best metric value and a counter for epochs without improvement.
                # Simplified: if history[monitor_metric] has not improved by min_delta for patience epochs...
                # This logic needs to be implemented fully if used.
                pass


        # End of training
        self._save_final_models()
        self._plot_losses(history, "final")
        self._save_training_metrics(history)
        
        self.logger.info("GAN training finished.")
        return {"status": "success", "history": history, "message": "Training completed."}

    def _save_models_at_epoch(self, epoch: int):
        if self.generator:
            gen_path = os.path.join(self.models_dir, self.params["save_generator_epoch_template"].format(epoch=epoch))
            self.generator.save(gen_path)
            self.logger.info(f"Saved generator model at epoch {epoch} to {gen_path}")
        if self.discriminator:
            disc_path = os.path.join(self.models_dir, self.params["save_discriminator_epoch_template"].format(epoch=epoch))
            self.discriminator.save(disc_path)
            self.logger.info(f"Saved discriminator model at epoch {epoch} to {disc_path}")
        if self.gan_model: # GAN model is mostly for training G, saving G is more important for inference
            gan_m_path = os.path.join(self.models_dir, self.params["save_gan_epoch_template"].format(epoch=epoch))
            self.gan_model.save(gan_m_path) # Save GAN model state
            self.logger.info(f"Saved GAN model state at epoch {epoch} to {gan_m_path}")


    def _save_final_models(self):
        if self.generator:
            gen_path = os.path.join(self.models_dir, self.params["final_generator_model_filename"])
            self.generator.save(gen_path)
            self.logger.info(f"Saved final generator model to {gen_path}")
        if self.discriminator:
            disc_path = os.path.join(self.models_dir, self.params["final_discriminator_model_filename"])
            self.discriminator.save(disc_path)
            self.logger.info(f"Saved final discriminator model to {disc_path}")
        if self.gan_model:
            gan_m_path = os.path.join(self.models_dir, self.params["final_gan_model_filename"])
            self.gan_model.save(gan_m_path)
            self.logger.info(f"Saved final GAN model state to {gan_m_path}")


    def _save_training_metrics(self, metrics_data: Dict):
        metrics_file = os.path.join(self.metrics_dir, self.params["training_metrics_filename"])
        try:
            with open(metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=4)
            self.logger.info(f"Saved training metrics to {metrics_file}")
        except Exception as e:
            self.logger.error(f"Error saving training metrics: {e}")

    def _plot_losses(self, history: Dict, epoch_or_final_str: Union[int, str]):
        plt.figure(figsize=(12, 8))
        
        plt.plot(history['d_loss_real'], label='D Loss Real')
        plt.plot(history['d_loss_fake'], label='D Loss Fake')
        plt.plot(history['g_loss'], label='G Loss')
        plt.title(f'GAN Training Losses (Epoch {epoch_or_final_str})')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plot_filename = self.params["loss_plot_epoch_template"].format(epoch=epoch_or_final_str) if isinstance(epoch_or_final_str, int) else self.params["final_loss_plot_filename"]
        plot_path = os.path.join(self.plots_dir, plot_filename)
        try:
            plt.savefig(plot_path, dpi=self.params.get("loss_plot_dpi", 300))
            self.logger.info(f"Saved loss plot to {plot_path}")
        except Exception as e:
            self.logger.error(f"Error saving loss plot: {e}")
        plt.close() # Close plot to free memory

        # Plot accuracies
        plt.figure(figsize=(12, 8))
        plt.plot(history['d_acc_real'], label='D Acc Real')
        plt.plot(history['d_acc_fake'], label='D Acc Fake')
        if 'g_acc' in history : plt.plot(history['g_acc'], label='G Acc (GAN)') # GAN 'accuracy'
        plt.title(f'GAN Training Accuracies (Epoch {epoch_or_final_str})')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        acc_plot_filename = f"accuracy_plot_epoch_{epoch_or_final_str}.png" if isinstance(epoch_or_final_str, int) else "accuracy_plot_final.png"
        acc_plot_path = os.path.join(self.plots_dir, acc_plot_filename)
        try:
            plt.savefig(acc_plot_path, dpi=self.params.get("loss_plot_dpi", 300))
            self.logger.info(f"Saved accuracy plot to {acc_plot_path}")
        except Exception as e:
            self.logger.error(f"Error saving accuracy plot: {e}")
        plt.close()

    def _plot_sample_generated_data(self, epoch: int, num_samples_to_plot: int = 3, 
                                   conditional_data_for_plot: Optional[np.ndarray]=None, 
                                   context_data_for_plot: Optional[np.ndarray]=None,
                                   window_data_for_plot: Optional[np.ndarray]=None):
        if not self.generator or self.latent_dim_for_generator is None or self.seq_len is None or self.num_base_features is None:
            self.logger.warning("Cannot plot sample data: generator or dimensions not set.")
            return

        self.logger.info(f"Generating sample data for plotting at epoch {epoch}...")
        
        gen_inputs_plot = []
        # Latent noise
        if self.gen_input_seq_len and self.gen_input_seq_len > 0:
            z_noise_plot_shape = (num_samples_to_plot, self.gen_input_seq_len, self.latent_dim_for_generator)
        else:
            z_noise_plot_shape = (num_samples_to_plot, self.latent_dim_for_generator)
        z_noise_plot = np.random.normal(0, 1, z_noise_plot_shape)
        gen_inputs_plot.append(z_noise_plot)

        # Conditional data
        if self.conditional_dim_for_generator and self.conditional_dim_for_generator > 0:
            if conditional_data_for_plot is not None and conditional_data_for_plot.shape[0] >= num_samples_to_plot and conditional_data_for_plot.shape[1] == self.conditional_dim_for_generator:
                gen_inputs_plot.append(conditional_data_for_plot[:num_samples_to_plot])
            else: # Need to generate dummy conditional data if not available or mismatched
                self.logger.warning(f"Plotting: Conditional data not available or mismatched for plotting. Using zeros. Shape expected: ({num_samples_to_plot}, {self.conditional_dim_for_generator})")
                gen_inputs_plot.append(np.zeros((num_samples_to_plot, self.conditional_dim_for_generator)))
        
        # Context data
        if self.context_dim_for_generator and self.context_dim_for_generator > 0:
            if context_data_for_plot is not None and context_data_for_plot.shape[0] >= num_samples_to_plot and context_data_for_plot.shape[1] == self.context_dim_for_generator:
                gen_inputs_plot.append(context_data_for_plot[:num_samples_to_plot])
            else:
                self.logger.warning(f"Plotting: Context data not available or mismatched. Using zeros. Shape expected: ({num_samples_to_plot}, {self.context_dim_for_generator})")
                gen_inputs_plot.append(np.zeros((num_samples_to_plot, self.context_dim_for_generator)))

        # Window data
        if self.window_dim_for_generator and self.window_dim_for_generator > 0:
            expected_win_seq_len = self._get_config_param("generator_window_input_seq_len", self.seq_len)
            if window_data_for_plot is not None and \
               window_data_for_plot.shape[0] >= num_samples_to_plot and \
               window_data_for_plot.shape[1] == expected_win_seq_len and \
               window_data_for_plot.shape[2] == self.window_dim_for_generator:
                gen_inputs_plot.append(window_data_for_plot[:num_samples_to_plot])
            else:
                self.logger.warning(f"Plotting: Window data not available or mismatched. Using zeros. Shape expected: ({num_samples_to_plot}, {expected_win_seq_len}, {self.window_dim_for_generator})")
                gen_inputs_plot.append(np.zeros((num_samples_to_plot, expected_win_seq_len, self.window_dim_for_generator)))


        sample_generated_data = self.generator.predict(gen_inputs_plot, verbose=0) # Shape: (num_samples_to_plot, seq_len, num_base_features)

        # Plot each feature of each sample
        # Determine feature names (e.g. from 'generated_feature_names' used in train)
        generated_feature_names = self.params.get("discriminator_input_feature_names", [f"Feature_{j}" for j in range(self.num_base_features)])
        if self.generator_plugin_instance and self.generator_plugin_instance.params.get('decoder_output_feature_names'):
            generated_feature_names = self.generator_plugin_instance.params.get('decoder_output_feature_names')
        
        if len(generated_feature_names) != self.num_base_features:
            generated_feature_names = [f"Feature_{j}" for j in range(self.num_base_features)] # Fallback

        num_features_to_plot = min(self.num_base_features, 5) # Plot up to 5 features

        fig, axes = plt.subplots(num_features_to_plot, num_samples_to_plot, figsize=(num_samples_to_plot * 4, num_features_to_plot * 3), squeeze=False)
        fig.suptitle(f'Sample Generated Data - Epoch {epoch}', fontsize=16)

        for i in range(num_samples_to_plot): # Iterate over samples
            for j in range(num_features_to_plot): # Iterate over features
                ax = axes[j, i]
                ax.plot(sample_generated_data[i, :, j])
                ax.set_title(f'Sample {i+1} - {generated_feature_names[j]}')
                ax.set_xlabel('Time Step (in sequence)')
                ax.set_ylabel('Value')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
        plot_path = os.path.join(self.plots_dir, f"sample_generated_data_epoch_{epoch}.png")
        try:
            plt.savefig(plot_path, dpi=self.params.get("model_plot_dpi", 150))
            self.logger.info(f"Saved sample generated data plot to {plot_path}")
        except Exception as e:
            self.logger.error(f"Error saving sample data plot: {e}")
        plt.close(fig)

    def plot_model_architectures(self):
        self.logger.info("Plotting model architectures...")
        if self.generator:
            gen_plot_path = os.path.join(self.plots_dir, self.params.get("generator_model_plot_file", "generator.png"))
            plot_model(self.generator, to_file=gen_plot_path, show_shapes=True, show_layer_names=True, dpi=self.model_plot_dpi)
            self.logger.info(f"Generator architecture plot saved to {gen_plot_path}")
        if self.discriminator:
            disc_plot_path = os.path.join(self.plots_dir, self.params.get("discriminator_model_plot_file", "discriminator.png"))
            plot_model(self.discriminator, to_file=disc_plot_path, show_shapes=True, show_layer_names=True, dpi=self.model_plot_dpi)
            self.logger.info(f"Discriminator architecture plot saved to {disc_plot_path}")
        if self.gan_model:
            gan_plot_path = os.path.join(self.plots_dir, self.params.get("gan_model_plot_file", "gan.png"))
            plot_model(self.gan_model, to_file=gan_plot_path, show_shapes=True, show_layer_names=True, dpi=self.model_plot_dpi)
            self.logger.info(f"GAN architecture plot saved to {gan_plot_path}")
