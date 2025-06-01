# optimizer/plugins/gan_plugin.py

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
import pandas as pd
import pandas_ta as ta
import numpy as np
import logging
import os
import time
import matplotlib.pyplot as plt
from copy import deepcopy
from typing import Any, Dict, List, Tuple, Union, Optional

# Assuming your generator plugin (like VAEGeneratorPlugin) and a new discriminator model definition exist
# from .generator_plugin import VAEGeneratorPlugin # Or your specific generator plugin
# from .discriminator_model import build_discriminator # You'll create this

# Initialize logger for this module
logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO) # Set default log level - Assuming this is set elsewhere or defaults appropriately
# ADD INFO MESSAGE ABOUT GENERATOR WARNING
# logger.info("GANTrainerPlugin: The VAE generator is intended to be frozen during GAN training. A \\\\'UserWarning: The model does not have any trainable weights.\\\\' may appear when generator.predict() is called; this is expected for the frozen generator and does not affect discriminator or GAN training.")


# Custom Keras Layer for Technical Indicator Calculation using tf.numpy_function
class TensorFlowTALayer(layers.Layer):
    def __init__(self, base_feature_names: List[str], ti_names_to_calculate: List[str],
                 num_base_features: int, num_total_features: int, seq_len: int, **kwargs):
        super().__init__(**kwargs)
        self.base_feature_names_ordered = base_feature_names
        self.ti_names_to_calculate = ti_names_to_calculate
        self.num_base_features = num_base_features
        self.num_total_features = num_total_features # num_base_features + num_tis
        self.seq_len = seq_len

        # For faster lookups in call()
        self.base_feature_name_to_idx = {name: i for i, name in enumerate(self.base_feature_names_ordered)}
        self.num_ti_features_to_calc = len(self.ti_names_to_calculate)

        # Pre-parse TIs
        self.parsed_tis_info = []
        for ti_full_name in self.ti_names_to_calculate:
            col, ind, params = self._parse_ti_name(ti_full_name)
            self.parsed_tis_info.append({"full_name": ti_full_name, "col": col, "ind": ind, "params": params})
            if not ind:
                 logger.warning(f"TensorFlowTALayer __init__: TI '{ti_full_name}' could not be parsed. Will output zeros for it.")
            elif ind.lower() not in ["ema"]: # Add other supported TF TIs to this list
                 logger.warning(f"TensorFlowTALayer __init__: TI type '{ind}' (from '{ti_full_name}') is not yet implemented with native TF operations. Will output zeros.")

    @staticmethod
    def _parse_ti_name(ti_full_name: str) -> Tuple[Optional[str], Optional[str], Dict[str, Any]]:
        """
        Parses a technical indicator string like "FeatureName_INDICATOR_param1_param2".
        Example: "BidClose_EMA_10" -> ("BidClose", "ema", {"length": 10})
                 "RSI_14" (if 'close' is default) -> (None, "rsi", {"length": 14}) - needs adjustment if column is not part of name
                 "ATR_14" (no specific column, uses HLC) -> (None, "atr", {"length": 14})
        Returns: (column_name, indicator_method_name, params_dict)
        """
        parts = ti_full_name.split('_')
        
        column_name: Optional[str] = None
        indicator_method_name: Optional[str] = None
        params: Dict[str, Any] = {}

        # Pattern 1: Col_Indicator_Param (e.g., BidClose_EMA_10)
        # Assumes indicator names are standard (EMA, SMA, RSI, etc.)
        # This list can be expanded.
        known_indicator_keywords = [
            "EMA", "SMA", "RSI", "BBANDS", "MACD", "ROC", "MOM", "STOCH", 
            "STOCHRSI", "TSI", "UO", "WILLR", "ATR", "ADX", "CCI", "PSAR"
        ]

        if len(parts) >= 2 and parts[1].upper() in known_indicator_keywords:
            # Check if parts[0] is likely a column name (heuristic, might need base_feature_names context for robustness)
            # For now, assume if parts[1] is a known indicator, parts[0] is the column.
            column_name = parts[0]
            indicator_method_name = parts[1].lower()
            param_parts = parts[2:]
        # Pattern 2: Indicator_Param (e.g., RSI_14, assuming default column like 'close')
        elif len(parts) >= 1 and parts[0].upper() in known_indicator_keywords:
            indicator_method_name = parts[0].lower()
            param_parts = parts[1:]
        else: # Cannot parse reliably
            logger.warning(f"TensorFlowTALayer (_parse_ti_name): Could not reliably parse TI name '{ti_full_name}'.")
            return None, None, {}

        # Parameter parsing (simplified, extend as needed)
        if indicator_method_name in ["ema", "sma", "rsi", "atr", "cci", "roc", "mom", "willr"]: # Common length-based
            if len(param_parts) > 0:
                try: params['length'] = int(param_parts[0])
                except ValueError: logger.warning(f"TensorFlowTALayer (_parse_ti_name): Could not parse length from {param_parts[0]} for {ti_full_name}.")
        elif indicator_method_name == "stoch": # STOCHk_14_3_3 -> k,d,smooth_k
             if len(param_parts) >= 1: params['k'] = int(param_parts[0])
             if len(param_parts) >= 2: params['d'] = int(param_parts[1])
             if len(param_parts) >= 3: params['smooth_k'] = int(param_parts[2]) # often called smooth_k or s
        elif indicator_method_name == "bbands":
            if len(param_parts) > 0: # length
                try: params['length'] = int(param_parts[0])
                except ValueError: logger.warning(f"TensorFlowTALayer (_parse_ti_name): Could not parse BBands length from {param_parts[0]} for {ti_full_name}.")
            if len(param_parts) > 1: # std
                try: params['std'] = float(param_parts[1]) # or int
                except ValueError: logger.warning(f"TensorFlowTALayer (_parse_ti_name): Could not parse BBands std from {param_parts[1]} for {ti_full_name}.")
        elif indicator_method_name == "macd":
            # MACD_fast_slow_signal (e.g., MACD_12_26_9)
            if len(param_parts) == 3: 
                try:
                    params['fast'] = int(param_parts[0])
                    params['slow'] = int(param_parts[1])
                    params['signal'] = int(param_parts[2])
                except ValueError: logger.warning(f"TensorFlowTALayer (_parse_ti_name): Could not parse MACD params from {param_parts} for {ti_full_name}.")
            elif len(param_parts) == 2: # fast, slow (signal default)
                 try:
                    params['fast'] = int(param_parts[0])
                    params['slow'] = int(param_parts[1])
                 except ValueError: logger.warning(f"TensorFlowTALayer (_parse_ti_name): Could not parse MACD fast/slow params from {param_parts} for {ti_full_name}.")
        # Add more specific param parsing as needed for other TIs

        # If column_name was parsed (e.g. "BidClose" from "BidClose_EMA_10")
        # and it's not a standard pandas_ta recognized column name for the indicator (like 'high', 'low', 'close', 'volume')
        # then we need to pass it as the 'column' argument to the pandas_ta function.
        # This logic is more for pandas_ta; for TF implementation, we directly use the column index.
        # However, storing 'column': column_name in params if parsed_col_name is set might be useful for consistency if some TIs still use a generic handler.
        if column_name and 'column' not in params:
            params['column_hint'] = column_name # Store the parsed column name if any

        return column_name, indicator_method_name, params

    @staticmethod
    def _tf_calculate_ema_1d(series: tf.Tensor, length: int, smoothing: float = 2.0) -> tf.Tensor:
        """
        Calculates Exponential Moving Average for a 1D tf.Tensor using TensorFlow operations.
        Args:
            series: 1D tf.Tensor of shape (seq_len,).
            length: EMA period (integer).
            smoothing: Smoothing factor, typically 2.0.
        Returns:
            1D tf.Tensor of shape (seq_len,) with EMA values.
        """
        seq_len_tf = tf.shape(series)[0]

        if length <= 0:
            logger.warning(f"TensorFlowTALayer (_tf_calculate_ema_1d): EMA length must be positive, got {length}. Returning zeros.")
            return tf.zeros_like(series, dtype=tf.float32)
        
        if tf.cast(seq_len_tf, tf.int32) == 0: # Handle empty series
            logger.debug("TensorFlowTALayer (_tf_calculate_ema_1d): Input series is empty. Returning empty tensor.")
            return tf.zeros_like(series, dtype=tf.float32) # Or series

        alpha = tf.cast(smoothing, tf.float32) / (tf.cast(length, tf.float32) + 1.0)
        
        # Initialize TensorArray to store EMA values
        ema_values_array = tf.TensorArray(dtype=tf.float32, size=seq_len_tf, dynamic_size=False, clear_after_read=False)

        # First EMA value is the first series value
        # Handle case where series might be shorter than 1
        first_val = tf.cond(seq_len_tf > 0, lambda: series[0], lambda: tf.constant(0.0, dtype=tf.float32))
        ema_values_array = tf.cond(seq_len_tf > 0, lambda: ema_values_array.write(0, first_val), lambda: ema_values_array)

        # Loop from the second element to compute subsequent EMA values
        loop_counter_init = tf.constant(1, dtype=tf.int32)
        prev_ema_init = first_val

        # Condition for the while_loop
        cond = lambda i, prev_ema, arr: i < seq_len_tf

        # Body of the while_loop
        def body(i, prev_ema, arr):
            current_val = series[i]
            new_ema = alpha * current_val + (1.0 - alpha) * prev_ema
            arr = arr.write(i, new_ema)
            return i + 1, new_ema, arr

        # Execute the loop only if seq_len_tf > 1
        _, _, final_ema_values_array = tf.cond(
            seq_len_tf > 1,
            lambda: tf.while_loop(
                cond, body, [loop_counter_init, prev_ema_init, ema_values_array],
                parallel_iterations=1 # Ensure sequential calculation for EMA
            ),
            lambda: (loop_counter_init, prev_ema_init, ema_values_array) # No loop if only one element or empty
        )
        
        return final_ema_values_array.stack()

    def call(self, inputs: tf.Tensor) -> tf.Tensor: # inputs shape: (batch_size, self.seq_len, self.num_base_features)
        batch_size = tf.shape(inputs)[0]
        
        calculated_ti_tensors_list = []

        for ti_info in self.parsed_tis_info:
            ti_full_name = ti_info["full_name"]
            parsed_col_name = ti_info["col"] # This is the column hint, e.g., "BidClose"
            parsed_indicator_name = ti_info["ind"]
            parsed_params = ti_info["params"]

            # Default to zeros for this TI, shape (batch_size, seq_len, 1)
            current_ti_tensor_for_concat = tf.zeros([batch_size, self.seq_len, 1], dtype=tf.float32)

            if parsed_indicator_name:
                indicator_lower = parsed_indicator_name.lower()
                if indicator_lower == "ema":
                    if parsed_col_name in self.base_feature_name_to_idx:
                        col_idx = self.base_feature_name_to_idx[parsed_col_name]
                        
                        # feature_series_batch shape: (batch_size, seq_len)
                        feature_series_batch = inputs[:, :, col_idx]
                        
                        ema_length = parsed_params.get('length')
                        if ema_length is not None and isinstance(ema_length, int) and ema_length > 0:
                            # Apply _tf_calculate_ema_1d to each sample in the batch
                            ema_output_2d_batch = tf.map_fn(
                                lambda s: self._tf_calculate_ema_1d(s, length=ema_length),
                                feature_series_batch,
                                fn_output_signature=tf.float32
                            ) # Output shape: (batch_size, seq_len)
                            current_ti_tensor_for_concat = tf.expand_dims(ema_output_2d_batch, axis=-1) # Shape: (batch_size, seq_len, 1)
                        else:
                            logger.warning(f"TensorFlowTALayer call: Invalid or missing 'length' ({ema_length}) for EMA '{ti_full_name}'. Using zeros.")
                    else:
                        logger.warning(f"TensorFlowTALayer call: Base feature column '{parsed_col_name}' for EMA '{ti_full_name}' not found in base_feature_name_to_idx. Keys: {list(self.base_feature_name_to_idx.keys())}. Using zeros.")
                # Add elif for other TF-implemented TIs here
                # elif indicator_lower == "sma": ...
                else:
                    # This case is for parsed TIs that are not "ema" (and not yet TF-implemented)
                    # Warning for this was already logged in __init__
                    logger.debug(f"TensorFlowTALayer call: TI '{indicator_lower}' (from '{ti_full_name}') not TF-implemented. Using zeros.")
            else:
                # This case is for TIs that failed to parse initially
                # Warning for this was already logged in __init__
                logger.debug(f"TensorFlowTALayer call: TI '{ti_full_name}' was unparsable. Using zeros.")

            calculated_ti_tensors_list.append(current_ti_tensor_for_concat)

        # Concatenate base features (inputs) with all calculated TI tensors
        if calculated_ti_tensors_list: # If there are TIs to add
            output_tensor = tf.concat([inputs] + calculated_ti_tensors_list, axis=-1)
        else: # No TIs to calculate, output is just the input features
            output_tensor = inputs # No TIs were calculated or requested that are TF-ready
            logger.debug(f"TensorFlowTALayer call: No TF-ready TIs calculated. Output tensor shape: {output_tensor.shape}")

        # Ensure the output shape is correctly set for Keras, especially the feature dimension
        # This is crucial if some TIs were skipped (outputting zeros) or if concatenation happened.
        # The number of features should now be self.num_total_features.
        
        # The problematic 'if' condition that caused OperatorNotAllowedInGraphError has been removed.
        # We rely on the concatenation logic to produce the correct number of features
        # (self.num_base_features + self.num_ti_features_to_calc),
        # and assume self.num_total_features (passed in __init__) is consistent with this.
        # The compute_output_shape method and the set_shape call below handle shape inference for Keras.

        # Try to set the static shape. Keras needs this.
        output_tensor.set_shape([None, self.seq_len, self.num_total_features]) # Static shape for batch, seq_len, features
        logger.info(f"TensorFlowTALayer call: Set output_tensor static shape to (None, {self.seq_len}, {self.num_total_features}). Actual symbolic shape after set_shape: {output_tensor.shape}")

        return output_tensor

    def compute_output_shape(self, input_shape):
        """
        Computes the output shape of the layer based on the input shape.
        Args:
            input_shape: Shape of the input tensor.
        Returns:
            Tuple representing the shape of the output tensor.
        """
        # Assuming input_shape is of the form (batch_size, seq_len, num_base_features)
        # We will output (batch_size, seq_len, num_total_features)
        # where num_total_features is self.num_base_features + number of TI features to calculate

        # This is a simplified example. You might need to adjust based on actual logic and attributes.
        num_ti_features_to_calc = self.num_ti_features_to_calc # This should be set in __init__ or build
        new_feature_count = self.num_base_features + num_ti_features_to_calc

        return (input_shape[0], input_shape[1], new_feature_count)

class GANTrainerPlugin:
    plugin_params: Dict[str, Any] = {
        "gan_epochs": 10000,
        "gan_batch_size": 32,
        "generator_lr": 1e-4,
        "generator_beta1": 0.5,
        "discriminator_lr": 1e-4,
        "discriminator_beta1": 0.5,
        "gan_save_interval": 500, # Epochs between saves
        "latent_dim": 32,
        "seq_len": 18, # Should match generator output sequence length if applicable
        # "gan_model_dir": "models/gan_trained", # Replaced by new path structure
        "discriminator_conv_filters": [64, 128],
        "discriminator_conv_kernel_size": 3,
        "discriminator_lstm_units": 64,
        "discriminator_dropout_rate": 0.3,
        "gan_generator_output_actual_seq_len": 1, # Or actual sequence length from generator
        "enable_reduce_lr_on_plateau": True,
        "lr_reduction_factor": 0.5,
        "lr_patience": 50,
        "lr_min_delta": 0.001,
        "min_lr_g": 1e-7,
        "min_lr_d": 1e-7,
        "lr_monitor_metric": "g_loss", # or 'd_loss', or a combined metric
        "enable_early_stopping": True,
        "es_patience": 200,
        "es_min_delta": 0.001,
        "es_monitor_metric": "g_loss", # or 'd_loss'

        # New structured path configuration
        "results_base_dir": "examples/results/phase_4_3",
        "save_model_dir": "models", # Subdirectory for .keras files under results_base_dir
        "save_plot_dir": "plots",   # Subdirectory for .png plots under results_base_dir

        "generator_model_plot_file": "generator_architecture.png",
        "gan_model_plot_file": "gan_architecture.png",
        "model_plot_dpi": 300,

        "save_generator_epoch_template": "generator_epoch_{epoch}.keras",
        "final_generator_model_filename": "generator_final.keras", # Default if not in global config

        "save_discriminator_epoch_template": "discriminator_epoch_{epoch}.keras",
        "final_discriminator_model_filename": "discriminator_final.keras", # Default if not in global config

        "save_gan_epoch_template": "gan_epoch_{epoch}.keras",
        "final_gan_model_filename": "gan_final.keras", # Default, consider global 'save_gan_sequential_model_file'

        "loss_plot_epoch_template": "loss_plot_epoch_{epoch}.png",
        "final_loss_plot_filename": "loss_plot_final.png", # Default if 'gan_loss_plot_file' not in global
        "loss_plot_dpi": 300,
    }

    def __init__(self, config: Dict[str, Any], generator_plugin_instance: Optional[Any] = None, feeder_plugin_instance: Optional[Any] = None, preprocessor_plugin_instance: Optional[Any] = None):
        self.config = config  # Store global config
        self.params = deepcopy(self.plugin_params)
        
        self.logger = logging.getLogger(__name__) # Ensure logger is initialized

        # Initialize attributes that are often derived or set from global config
        # and might not be in plugin_params. This prevents AttributeErrors if they
        # are accessed before being explicitly set by _initialize_core_parameters_from_config
        # or if they are not present in the provided config to set_params.
        self.discriminator_feature_names: List[str] = []
        self.num_features_for_discriminator: int = 0
        self.base_feature_names_ordered: List[str] = []
        self.num_base_features: int = 0
        self.ti_names_to_calculate: List[str] = [] # General TIs list for the TI calculation layer/logic
        # self.tas_strategy_for_discriminator_tis = None # Example if it's an object/dict
        # self.ohlc_feature_map: Dict[str, str] = {} # Example if used

        self.generator_plugin = generator_plugin_instance
        self.feeder_plugin = feeder_plugin_instance
        self.preprocessor_plugin = preprocessor_plugin_instance

        self._initialize_core_parameters_from_config()

        # Try to get the generator model from the plugin instance
        if self.generator_plugin and hasattr(self.generator_plugin, 'get_model'):
            self.generator = self.generator_plugin.get_model()
            if self.generator is None:
                self.logger.error("Failed to get generator model from generator_plugin_instance.")
                # Potentially raise an error or handle appropriately
            else:
                self.logger.info("Successfully retrieved generator model from plugin.")
                 # Plot and save generator model architecture
                try:
                    plot_dir = os.path.join(self.params["results_base_dir"], self.params["save_plot_dir"])
                    os.makedirs(plot_dir, exist_ok=True)
                    plot_file_name = self.params.get("generator_model_plot_file", "generator_architecture.png")
                    plot_path = os.path.join(plot_dir, plot_file_name)
                    dpi = self.params.get("model_plot_dpi", 300)
                    
                    # Check if model is built, otherwise plot_model might fail for some architectures
                    if hasattr(self.generator, 'built') and not self.generator.built:
                        self.logger.warning("Generator model is not built yet. Plot may be incomplete or fail.")
                        # Attempt to build if input shape is known, e.g. self.generator.build(input_shape)
                        # This is complex as input shape might not be readily available here.
                        # For now, we proceed with the plot attempt.

                    plot_model(self.generator, to_file=plot_path, show_shapes=True, dpi=dpi, expand_nested=True)
                    self.logger.info(f"Generator architecture plot saved to {plot_path}")
                except Exception as e:
                    self.logger.error(f"Failed to plot generator model: {e}", exc_info=True)

        else:
            self.generator = None
            self.logger.error("Generator plugin instance not provided or does not have get_model method.")
            # This is likely a critical issue for GAN training

        self.discriminator = self._build_discriminator()
        self.gan = self._build_gan() # This will also plot the GAN architecture

        # Compile discriminator for standalone training (already mentioned as done, ensure it is)
        if self.discriminator:
            d_optimizer = Adam(learning_rate=self.params["discriminator_lr"], beta_1=self.params["discriminator_beta1"])
            self.discriminator.compile(loss='binary_crossentropy', optimizer=d_optimizer, metrics=['accuracy'])
            self.logger.info("Discriminator compiled for standalone training.")

        self.d_losses_history = []
        self.g_losses_history = []
        self.d_lr_history = []
        self.g_lr_history = []
        
        self.logger.info("GANTrainerPlugin initialized.")

    def _initialize_core_parameters_from_config(self):
        """Helper to initialize parameters needed before model building, typically also managed by set_params."""
        self.gen_input_seq_len = self.params.get("seq_len", 18)
        self.gen_input_latent_dim = self.params.get("latent_dim", 32) # Used for GAN latent input
        
        self.base_feature_names_ordered = self.params.get("base_feature_names_ordered", []) # CORRECTED
        self.num_base_features = len(self.base_feature_names_ordered) # CORRECTED
        
        # Derive ti_names_to_calculate from discriminator features and base features
        all_discriminator_features = self.params.get("feature_names_for_discriminator_ordered", [])
        self.discriminator_feature_names = all_discriminator_features # ADDED INITIALIZATION
        if all_discriminator_features and self.base_feature_names_ordered:
            # This assumes base_feature_names_ordered are a subset and appear first in all_discriminator_features
            self.ti_names_to_calculate = [f for f in all_discriminator_features if f not in self.base_feature_names_ordered]
        else:
            self.ti_names_to_calculate = []
            self.logger.warning("_initialize_core_parameters_from_config: Could not derive ti_names_to_calculate due to missing feature lists in params.")


        self.num_tis = len(self.ti_names_to_calculate)
        
        # self.num_features_for_discriminator = self.num_base_features + self.num_tis # OLD LOGIC
        self.num_features_for_discriminator = len(self.discriminator_feature_names) # CORRECTED LOGIC
        self.seq_len = self.params.get("seq_len", 18) # Used for discriminator input and TI layer

        # Initialize TA Strategy for Discriminator TIs
        self.tas_strategy_for_discriminator_tis = None # Initialize attribute
        ti_definitions_for_strategy = []
        
        if self.ti_names_to_calculate:
            processed_complex_indicators = set() # To handle indicators that produce multiple columns but need one strategy entry

            for ti_full_name in self.ti_names_to_calculate:
                parts = ti_full_name.split('_')
                indicator_name_part = parts[0] 
                current_definition = None

                try:
                    if indicator_name_part == "RSI" and len(parts) == 2:
                        current_definition = {"kind": "rsi", "length": int(parts[1])}
                    elif indicator_name_part == "EMA" and len(parts) == 2:
                        current_definition = {"kind": "ema", "length": int(parts[1])}
                    elif indicator_name_part in ["MACD", "MACDh", "MACDs"] and len(parts) == 4:
                        if "macd" not in processed_complex_indicators:
                            current_definition = {"kind": "macd", "fast": int(parts[1]), "slow": int(parts[2]), "signal": int(parts[3])}
                            processed_complex_indicators.add("macd")
                    elif indicator_name_part in ["STOCHk", "STOCHd"] and len(parts) == 4: # e.g., STOCHk_14_3_3
                        if "stoch" not in processed_complex_indicators:
                            current_definition = {"kind": "stoch", "k": int(parts[1]), "d": int(parts[2]), "smooth_k": int(parts[3])}
                            processed_complex_indicators.add("stoch")
                    elif indicator_name_part in ["ADX", "DMP", "DMN"] and len(parts) == 2: # e.g., ADX_14
                        if "adx" not in processed_complex_indicators:
                            current_definition = {"kind": "adx", "length": int(parts[1])}
                            processed_complex_indicators.add("adx")
                    elif indicator_name_part == "ATRr" and len(parts) == 2: # e.g., ATRr_14
                        current_definition = {"kind": "atr", "length": int(parts[1])} # Corrected "atrr" to "atr"
                    elif indicator_name_part == "CCI" and len(parts) >= 2: # e.g., CCI_14 or CCI_14_0.015
                        params_cci = {"length": int(parts[1])}
                        if len(parts) == 3: params_cci["constant"] = float(parts[2])
                        current_definition = {"kind": "cci", **params_cci}
                    elif indicator_name_part == "WILLR" and len(parts) == 2:
                        current_definition = {"kind": "willr", "length": int(parts[1])}
                    elif indicator_name_part == "MOM" and len(parts) == 2:
                        current_definition = {"kind": "mom", "length": int(parts[1])}
                    elif indicator_name_part == "ROC" and len(parts) == 2:
                        current_definition = {"kind": "roc", "length": int(parts[1])}
                    elif indicator_name_part.startswith("BB") and len(parts) == 3: # e.g., BBL_20_2.0
                        if "bbands" not in processed_complex_indicators:
                            current_definition = {"kind": "bbands", "length": int(parts[1]), "std": float(parts[2])}
                            processed_complex_indicators.add("bbands")
                    else:
                        self.logger.warning(f"GANTrainerPlugin (_initialize_core_parameters_from_config): TI name '{ti_full_name}' not parsed into a TA strategy definition.")
                except ValueError as e:
                    self.logger.error(f"GANTrainerPlugin (_initialize_core_parameters_from_config): Error parsing TI name '{ti_full_name}': {e}")
                    continue # Skip this TI if parsing fails

                if current_definition:
                    # Avoid adding duplicate definitions if already processed (e.g. MACD vs MACDh)
                    is_present = any(d == current_definition for d in ti_definitions_for_strategy)
                    if not is_present:
                        ti_definitions_for_strategy.append(current_definition)
            
            if ti_definitions_for_strategy:
                try:
                    self.tas_strategy_for_discriminator_tis = ta.Strategy(
                        name="Discriminator TIs Strategy",
                        description="Auto-generated TA strategy for discriminator features based on ti_names_to_calculate.",
                        ta=ti_definitions_for_strategy
                    )
                    self.logger.info(f"GANTrainerPlugin (_initialize_core_parameters_from_config): pandas_ta.Strategy for discriminator TIs created with {len(ti_definitions_for_strategy)} definitions:")
                    for ti_def in ti_definitions_for_strategy:
                        self.logger.info(f"  - {ti_def}")
                except Exception as e:
                    self.logger.error(f"GANTrainerPlugin (_initialize_core_parameters_from_config): Failed to create pandas_ta.Strategy: {e}")
                    self.tas_strategy_for_discriminator_tis = None # Ensure it's None on failure
            else:
                self.logger.warning("GANTrainerPlugin (_initialize_core_parameters_from_config): No valid TI definitions derived for TA Strategy. self.tas_strategy_for_discriminator_tis will be None.")
        else:
            self.logger.info("GANTrainerPlugin (_initialize_core_parameters_from_config): self.ti_names_to_calculate is empty. No TA Strategy for discriminator TIs will be created.")
            self.tas_strategy_for_discriminator_tis = None # Ensure it's None if no TIs to calculate

        self.generator_output_actual_seq_len = self.params.get("gan_generator_output_actual_seq_len", self.seq_len)

        self.logger.info(
            f"GANTrainerPlugin (_initialize_core_parameters): "
            f"gen_input_seq_len={self.gen_input_seq_len}, gen_input_latent_dim={self.gen_input_latent_dim}, "
            f"num_base_features={self.num_base_features} (from {len(self.base_feature_names_ordered)} names), "
            f"num_tis={self.num_tis} (from {len(self.ti_names_to_calculate)} names), "
            f"num_features_for_discriminator={self.num_features_for_discriminator}, "
            f"seq_len (for D and TI)={self.seq_len}, "
            f"generator_output_actual_seq_len={self.generator_output_actual_seq_len}"
        )
        if self.num_base_features == 0:
            self.logger.warning("_initialize_core_parameters_from_config: num_base_features is 0. This might be problematic for the TI layer and GAN structure.")

    def set_params(self, **params: Any) -> None:
        logger.info(f"GANTrainerPlugin updating parameters: {list(params.keys())}")
        # Update self.config with any new params passed directly to set_params
        self.config.update(params)
        
        # self.params should be the union of plugin_params and self.config.
        # Assuming self.params was initialized with self.plugin_params.copy() in __init__,
        # then updated with self.config in __init__.
        # Here, we ensure self.params reflects the latest self.config.
        current_params = self.plugin_params.copy() # Start with defaults
        current_params.update(self.config) # Override with current config (which includes updates from **params)
        self.params = current_params # Assign the fully updated params

        # Initialize sequence length and latent dimension attributes used by model builders
        self.gen_input_seq_len = self.params.get("seq_len", 18) # Default to 18 if not specified
        self.gen_input_latent_dim = self.params.get("latent_dim", 32) # Default to 32
        
        # Update feature-related attributes based on the latest self.params
        self.base_feature_names_ordered = self.params.get("base_feature_names_ordered", [])
        self.num_base_features = len(self.base_feature_names_ordered)
        
        self.ti_names_to_calculate = self.params.get("ti_names_to_calculate", [])
        self.num_tis = len(self.ti_names_to_calculate)
        
        # This is crucial for discriminator input shape and TI layer
        # self.num_features_for_discriminator = self.num_base_features + self.num_tis # OLD LOGIC
        self.num_features_for_discriminator = len(self.discriminator_feature_names) # CORRECTED LOGIC
        self.seq_len = self.params.get("seq_len", 18) # Also ensure self.seq_len is updated for discriminator

        if self.num_base_features == 0:
            logger.warning(
                "GANTrainerPlugin (set_params): 'base_feature_names_ordered' is empty or not found in parameters. "
                "Number of base features (self.num_base_features) is 0. This will likely cause issues in model building or data processing, "
                "as the generator is expected to produce these base features."
            )
            
        # This is the sequence length of the generator's output, if it's fixed (e.g., 1 for single-step generation)
        # and different from the main seq_len used by discriminator. Often, it might be the same as self.seq_len.
        self.generator_output_actual_seq_len = self.params.get("gan_generator_output_actual_seq_len", self.seq_len) 

        logger.info(f"GANTrainerPlugin (set_params): GAN Input: Generator input seq_len={self.gen_input_seq_len}, latent_dim={self.gen_input_latent_dim}")
        logger.info(f"GANTrainerPlugin (set_params): Features Config: num_base_features={self.num_base_features} (from 'base_feature_names_ordered')")
        logger.info(f"GANTrainerPlugin (set_params): Features Config: num_TIs_to_calculate={self.num_tis} (from 'ti_names_to_calculate')")
        logger.info(f"GANTrainerPlugin (set_params): Discriminator Input: expected total_features={self.num_features_for_discriminator}, seq_len={self.seq_len}")
        logger.info(f"GANTrainerPlugin (set_params): Generator Output: expected base_features={self.num_base_features}, actual_output_seq_len={self.generator_output_actual_seq_len}")

        if self.generator:
            # Ensure generator model is loaded to get its actual output dimension
            if hasattr(self.generator, 'output_shape'):
                self.actual_generator_output_dim = self.generator.output_shape[-1]
                logger.info(f"GANTrainerPlugin (set_params): Loaded generator model's actual output feature dimension: {self.actual_generator_output_dim}")
                
                if self.actual_generator_output_dim != self.num_base_features:
                    logger.warning(
                        f"GANTrainerPlugin (set_params): MISMATCH! Actual generator output feature dimension ({self.actual_generator_output_dim}) "
                        f"differs from configured num_base_features ({self.num_base_features}) derived from 'base_feature_names_ordered'. "
                        f"The GAN's _build_gan method will attempt to slice the generator's output to the first {self.num_base_features} features."
                    )
                else:
                    logger.info(f"GANTrainerPlugin (set_params): Generator output dimension ({self.actual_generator_output_dim}) matches configured num_base_features ({self.num_base_features}).")
            else:
                logger.warning("GANTrainerPlugin (set_params): Generator model is loaded but has no 'output_shape' attribute. Cannot verify output dimension.")
                self.actual_generator_output_dim = self.num_base_features # Assume it matches to avoid downstream errors, but log clearly
        else:
            logger.info("GANTrainerPlugin (set_params): Generator model not loaded yet. Cannot check its output dimension at this stage.")
            # self.actual_generator_output_dim will be checked again in _build_gan or if generator is loaded later.

        # Re-initialize optimizers if learning rates might have changed in self.params
        # Ensure __init__ also initializes these with values from self.params
        if hasattr(self, 'g_optimizer') and self.g_optimizer is not None:
            current_lr_g = self.params.get("generator_lr", 1e-4)
            current_beta1_g = self.params.get("generator_beta1", 0.5)
            if self.g_optimizer.learning_rate != current_lr_g or self.g_optimizer.beta_1 != current_beta1_g:
                self.g_optimizer = Adam(learning_rate=current_lr_g, beta_1=current_beta1_g)
                logger.info(f"GANTrainerPlugin (set_params): Generator optimizer re-initialized with LR={current_lr_g}, Beta1={current_beta1_g}")
        else:
            self.g_optimizer = Adam(learning_rate=self.params.get("generator_lr", 1e-4), beta_1=self.params.get("generator_beta1", 0.5))

        if hasattr(self, 'd_optimizer') and self.d_optimizer is not None:
            current_lr_d = self.params.get("discriminator_lr", 1e-4)
            current_beta1_d = self.params.get("discriminator_beta1", 0.5)
            if self.d_optimizer.learning_rate != current_lr_d or self.d_optimizer.beta_1 != current_beta1_d:
                self.d_optimizer = Adam(learning_rate=current_lr_d, beta_1=current_beta1_d)
                logger.info(f"GANTrainerPlugin (set_params): Discriminator optimizer re-initialized with LR={current_lr_d}, Beta1={current_beta1_d}")
        else:
            self.d_optimizer = Adam(learning_rate=self.params.get("discriminator_lr", 1e-4), beta_1=self.params.get("discriminator_beta1", 0.5))
        
        # Models might need to be rebuilt if critical dimensions (seq_len, feature counts) change
        # For simplicity, we assume they are built in __init__ and might need re-evaluation if params change significantly.
        # A more robust system might explicitly rebuild them here or set a flag.
        logger.info("GANTrainerPlugin (set_params): Parameters updated. Models (Discriminator, GAN) may need to be re-built if structural params changed.")

    def _build_discriminator(self) -> Model:
        logger.info("Building Discriminator...")
        
        # Input shape uses self.seq_len and self.num_features_for_discriminator
        # These are set in __init__ based on self.params
        data_input = layers.Input(shape=(self.seq_len, self.num_features_for_discriminator), name="discriminator_input")
        
        x = data_input
        
        # Conv1D layers
        conv_filters = self.params.get("discriminator_conv_filters", [64, 128])
        kernel_size = self.params.get("discriminator_conv_kernel_size", 3)
        for filters in conv_filters:
            x = layers.Conv1D(filters=filters, kernel_size=kernel_size, padding='causal', activation='relu')(x)
            x = layers.BatchNormalization()(x) # Added BatchNormalization
            x = layers.SpatialDropout1D(self.params.get("discriminator_dropout_rate", 0.3) / 2)(x) # Added SpatialDropout1D

        # Bidirectional LSTM layer
        lstm_units = self.params.get("discriminator_lstm_units", 64)
        # x = layers.Bidirectional(layers.LSTM(units=lstm_units, return_sequences=True))(x) # If another LSTM or TimeDistributed Dense follows
        # x = layers.BatchNormalization()(x) # Added BatchNormalization
        x = layers.Bidirectional(layers.LSTM(units=lstm_units, return_sequences=False))(x) # return_sequences=False for final feature vector
        x = layers.BatchNormalization()(x) # Added BatchNormalization
        
        x = layers.Dropout(self.params.get("discriminator_dropout_rate", 0.3))(x)
        
        # Output layer
        output = layers.Dense(1, activation='sigmoid', name="discriminator_output")(x)
        
        model = Model(data_input, output, name="Discriminator")
        logger.info("Discriminator built successfully.")
        model.summary(print_fn=logger.info)
        # Compilation moved to __init__ and set_params
        return model

    def _build_gan(self) -> Model:
        if not self.generator:
            self.logger.error("Generator model is not available. Cannot build GAN.")
            return None
        if not self.discriminator:
            self.logger.error("Discriminator model is not available. Cannot build GAN.")
            return None

        # Make discriminator non-trainable when part of GAN
        self.discriminator.trainable = False

        # Define GAN model
        # Input to GAN is noise for the generator
        # Generator's TIs layer needs to be part of the GAN graph
        # The generator model from plugin should already include its TI layer if it's internal
        # Or, if TIs are added *after* generator's raw output, that layer needs to be here.
        
        # Assuming self.generator is the complete generator network
        # and its output is what the discriminator expects (after TIs if TIs are external to generator)

        # If TensorFlowTALayer is part of the generator model, this is simpler.
        # If TensorFlowTALayer is applied to generator's output *before* discriminator,
        # it needs to be instantiated here and wrapped with the generator.

        # For now, assume self.generator output is ready for discriminator
        # This might need adjustment based on where TensorFlowTALayer is.
        # If TensorFlowTALayer is *only* for the discriminator's input processing of real/fake data,
        # then it's part of the discriminator or applied to its inputs.
        # The current setup seems to have TensorFlowTALayer in the discriminator.

        # Let's assume the generator (self.generator) produces data that, after passing through
        # a TI calculation layer (if not already part of generator), goes to discriminator.
        # The GAN model should chain: noise -> generator -> (optional external TI layer) -> discriminator
        
        # If generator_plugin_instance.get_model() returns a model that already has TIs:
        gan_input_latent = self.generator.input # Assuming latent vector is the input
        generated_sequence_with_tis = self.generator.output # Generator output (already has TIs)
        
        # If generator output needs TIs applied before discriminator (and TI layer is not in generator):
        # latent_input = layers.Input(shape=(self.params["latent_dim"],)) # Or appropriate latent shape
        # generated_sequence_raw = self.generator(latent_input) # Generator's raw output
        # ti_calculator_for_gan = TensorFlowTALayer(...) # Configure as needed
        # generated_sequence_with_tis = ti_calculator_for_gan(generated_sequence_raw)
        # gan_input_latent = latent_input

        gan_output = self.discriminator(generated_sequence_with_tis)
        
        gan = Model(gan_input_latent, gan_output, name="Generative_Adversarial_Network")
        
        g_optimizer = Adam(learning_rate=self.params["generator_lr"], beta_1=self.params["generator_beta1"])
        gan.compile(loss='binary_crossentropy', optimizer=g_optimizer)
        self.logger.info("GAN model built and compiled.")

        # Plot and save GAN model architecture
        try:
            plot_dir = os.path.join(self.params["results_base_dir"], self.params["save_plot_dir"])
            os.makedirs(plot_dir, exist_ok=True)
            plot_file_name = self.params.get("gan_model_plot_file", "gan_architecture.png")
            plot_path = os.path.join(plot_dir, plot_file_name)
            dpi = self.params.get("model_plot_dpi", 300)
            plot_model(gan, to_file=plot_path, show_shapes=True, dpi=dpi, expand_nested=True)
            self.logger.info(f"GAN architecture plot saved to {plot_path}")
        except Exception as e:
            self.logger.error(f"Failed to plot GAN model: {e}", exc_info=True)
            
        return gan

    def train(self, x_train_file: str, y_train_file: str = None, x_test_file: str = None, y_test_file: str = None):
        self.logger.info(f"GANTrainerPlugin (train): Starting GAN training with x_train_file: {x_train_file}")

        # Load data using pandas for CSV
        try:
            data_df = pd.read_csv(x_train_file)
            self.logger.info(f"GANTrainerPlugin (train): Successfully loaded CSV data from {x_train_file}. Shape: {data_df.shape}")
        except Exception as e:
            self.logger.error(f"GANTrainerPlugin (train): Error loading CSV file {x_train_file}. Error: {e}")
            raise

        # Drop DATE_TIME column if it exists
        if 'DATE_TIME' in data_df.columns:
            data_df = data_df.drop(columns=['DATE_TIME'])
            self.logger.info(f"GANTrainerPlugin (train): Dropped 'DATE_TIME' column. New shape: {data_df.shape}")
        
        # Convert to numpy array
        x_train_data_full = data_df.to_numpy()
        self.logger.info(f"GANTrainerPlugin (train): Converted data to NumPy array. Shape: {x_train_data_full.shape}")

        # Create sequences
        seq_len = self.params.get('seq_len', 18) # Get seq_len from params, default to 18
        if x_train_data_full.shape[0] < seq_len:
            self.logger.error(f"GANTrainerPlugin (train): Data length ({x_train_data_full.shape[0]}) is less than seq_len ({seq_len}). Cannot create sequences.")
            raise ValueError(f"Data length ({x_train_data_full.shape[0]}) is less than seq_len ({seq_len}).")

        num_samples = x_train_data_full.shape[0] - seq_len + 1
        num_features = x_train_data_full.shape[1]
        
        x_train_sequences = np.array([x_train_data_full[i:i+seq_len] for i in range(num_samples)])
        
        if x_train_sequences.ndim == 2 and num_samples > 0 :
            x_train_sequences = x_train_sequences.reshape(num_samples, seq_len, num_features if num_features > 0 else 1)
        elif num_samples == 0:
            if num_features > 0:
                x_train_sequences = np.empty((0, seq_len, num_features))
            else:
                self.logger.error(f"GANTrainerPlugin (train): Data has 0 features after processing.")
                raise ValueError("Data has 0 features after processing.")

        self.logger.info(f"GANTrainerPlugin (train): Created sequences. Shape: {x_train_sequences.shape}")
        x_train_data = x_train_sequences

        if x_train_data.ndim == 2: # Should not happen if previous logic is correct
            x_train_data = np.expand_dims(x_train_data, axis=1)
            self.logger.warning(f"GANTrainerPlugin (train): Expanded dims for 2D data. New shape: {x_train_data.shape}. This might indicate an issue in sequence creation.")

        current_batch_size = self.params.get("gan_batch_size", 32)
        valid = np.ones((current_batch_size, 1))
        fake = np.zeros((current_batch_size, 1))

        d_losses_history = []
        g_losses_history = []
        d_accs_history = [] # For average D accuracy
        d_accs_real_history = []
        d_accs_fake_history = []


        # Initialize learning rates for manual callback tracking
        current_lr_g = float(tf.keras.backend.get_value(self.g_optimizer.learning_rate))
        current_lr_d = float(tf.keras.backend.get_value(self.d_optimizer.learning_rate))

        # Print model summaries
        if hasattr(self, 'generator') and self.generator:
            self.logger.info("Generator Summary (at start of train):")
            self.generator.summary(print_fn=self.logger.info)
        else:
            self.logger.warning("Generator model not available for summary at start of train.")

        if hasattr(self, 'discriminator') and self.discriminator:
            self.logger.info("Discriminator Summary (at start of train):")
            self.discriminator.summary(print_fn=self.logger.info)
        else:
            self.logger.warning("Discriminator model not available for summary at start of train.")
        
        if hasattr(self, 'gan') and self.gan:
            self.logger.info("Combined GAN Model Summary (at start of train):")
            self.gan.summary(print_fn=self.logger.info)
        else:
            self.logger.warning("Combined GAN model not available for summary at start of train.")


        self.logger.info(f"Starting GAN training for {self.params['gan_epochs']} epochs with batch size {self.params['gan_batch_size']}...")
        self.logger.info(f"Initial Generator LR: {current_lr_g:.1e}, Discriminator LR: {current_lr_d:.1e}")
        # Log trainable status more dynamically if it changes, but for now:
        self.logger.info(f"Discriminator status for standalone training: Trainable") # D is compiled and trained standalone
        self.logger.info(f"Generator status for GAN training: Trainable (via GAN model)") # G is part of GAN, trains via GAN
        self.logger.info(f"Discriminator status for GAN training: Frozen (self.discriminator.trainable = False in _build_gan)")


        self.logger.info(f"Expected Discriminator input shape: (batch_size, {self.seq_len}, {self.num_features_for_discriminator})")
        self.logger.info(f"Expected Generator output (base features) shape for TI calc: (batch_size, {self.seq_len}, {self.num_base_features})")


        for epoch in range(self.params["gan_epochs"]):
            start_time_epoch = time.time()
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            if not self.discriminator.trainable: # Ensure D is trainable for this phase
                self.discriminator.trainable = True
                # Re-compile if needed? Keras usually handles this if optimizer is already set.
                # self.discriminator.compile(loss='binary_crossentropy', optimizer=self.d_optimizer, metrics=['accuracy'])
                # self.logger.info("Re-enabled training for Discriminator and re-compiled.")


            idx = np.random.randint(0, x_train_data.shape[0], self.params["gan_batch_size"])
            real_data_batch_raw_full_features = x_train_data[idx]

            if hasattr(self, 'num_base_features') and self.num_base_features > 0 and \
               real_data_batch_raw_full_features.shape[-1] >= self.num_base_features: # Use >= to allow exact match
                real_data_batch_base_features = real_data_batch_raw_full_features[:, :, :self.num_base_features]
            else:
                real_data_batch_base_features = real_data_batch_raw_full_features
            
            real_data_for_discriminator = self._calculate_technical_indicators(real_data_batch_base_features)

            feeder_output = self.feeder_plugin_instance.generate(n_ticks_to_generate=self.params["gan_batch_size"])
            
            generator_inputs = []
            if not hasattr(self, 'generator_actual_input_names_ordered'):
                self.logger.error("GANTrainerPlugin attribute 'generator_actual_input_names_ordered' not found.")
                raise AttributeError("'generator_actual_input_names_ordered' not found.")

            for input_name in self.generator_actual_input_names_ordered:
                if input_name not in feeder_output:
                    self.logger.error(f"KeyError: Feeder output missing '{input_name}'. Available: {list(feeder_output.keys())}")
                    raise KeyError(f"Feeder output missing '{input_name}'.")
                generator_inputs.append(feeder_output[input_name])
            
            # Use verbose=0 for predict to reduce log spam
            generated_base_features_raw = self.generator.predict(generator_inputs, verbose=0)

            if generated_base_features_raw.ndim == 2:
                target_reshape_dim = self.actual_generator_output_dim if hasattr(self, 'actual_generator_output_dim') and self.actual_generator_output_dim > 0 else -1
                gen_output_seq_len = self.params.get('gan_generator_output_actual_seq_len', 1) # Use the GAN-specific param
                
                if gen_output_seq_len > 0:
                    try:
                        generated_base_features_raw = generated_base_features_raw.reshape(
                            (self.params["gan_batch_size"], gen_output_seq_len, target_reshape_dim)
                        )
                    except ValueError as e:
                        self.logger.error(f"Error reshaping 2D gen output. Shape: {generated_base_features_raw.shape}, target_seq_len={gen_output_seq_len}, target_feat_dim={target_reshape_dim}. Error: {e}")
                        raise
                else:
                    self.logger.error("Cannot reshape 2D generator output. 'gan_generator_output_actual_seq_len' is not valid.")
                    raise AttributeError("'gan_generator_output_actual_seq_len' is not valid.")
            
            # If gan_generator_output_actual_seq_len was 1, and self.seq_len > 1, tile it.
            # This ensures generated_base_features_raw matches self.seq_len for TI calculation.
            if generated_base_features_raw.shape[1] == 1 and self.seq_len > 1 and self.params.get('gan_generator_output_actual_seq_len', 1) == 1:
                self.logger.debug(f"Tiling single-step generator output to seq_len {self.seq_len} for D training.")
                generated_base_features_raw = np.tile(generated_base_features_raw, (1, self.seq_len, 1))


            current_feature_dim = generated_base_features_raw.shape[-1]
            if current_feature_dim != self.num_base_features:
                if self.num_base_features > 0 and current_feature_dim > self.num_base_features : # Slice only if G produces more
                    generated_base_features_raw = generated_base_features_raw[:, :, :self.num_base_features]
                elif self.num_base_features > 0 and current_feature_dim < self.num_base_features:
                     self.logger.error(f"Generator output {current_feature_dim} features, but expected {self.num_base_features} base features. Cannot slice. Check G model or config.")
                     raise ValueError("Generator output feature dimension mismatch for base features.")


            generated_features_with_tis = self._calculate_technical_indicators(generated_base_features_raw)
            generated_data_for_discriminator = generated_features_with_tis

            d_loss_real_metrics = self.discriminator.train_on_batch(real_data_for_discriminator, valid, return_dict=True)
            d_loss_fake_metrics = self.discriminator.train_on_batch(generated_data_for_discriminator, fake, return_dict=True)
            
            d_loss = 0.5 * (d_loss_real_metrics['loss'] + d_loss_fake_metrics['loss'])
            d_acc_real = d_loss_real_metrics.get('accuracy', 0.0) # Use .get for safety
            d_acc_fake = d_loss_fake_metrics.get('accuracy', 0.0)
            d_acc = 0.5 * (d_acc_real + d_acc_fake)


            # ---------------------
            #  Train Generator
            # ---------------------
            if self.discriminator.trainable: # Freeze D for GAN training phase
                self.discriminator.trainable = False
                # No re-compile needed here as GAN model manages D's trainability internally for its pass.
                # self.logger.info("Set discriminator.trainable = False for GAN training phase.")

            feeder_outputs_for_g = self.feeder_plugin_instance.generate(n_ticks_to_generate=self.params["gan_batch_size"])
            
            # Prepare inputs for the GAN model (latent, conditional, context)
            # These names are from config and used to build the GAN model's inputs
            # The GAN model itself expects a list of inputs in the order defined in _build_gan
            # [gan_latent_input, gan_conditional_input, gan_context_input]
            
            # We need to fetch the corresponding data from feeder_outputs_for_g using the
            # configured names (e.g., generator_decoder_input_name_latent).
            
            latent_input_key = self.params.get("generator_decoder_input_name_latent", "decoder_input_z_seq")
            conditional_input_key = self.params.get("generator_decoder_input_name_conditions", "decoder_input_conditions")
            context_input_key = self.params.get("generator_decoder_input_name_context", "decoder_input_h_context")

            gan_inputs_dict = {
                "latent": feeder_outputs_for_g.get(latent_input_key),
                "conditional": feeder_outputs_for_g.get(conditional_input_key),
                "context": feeder_outputs_for_g.get(context_input_key)
            }

            # Check for missing keys
            if gan_inputs_dict["latent"] is None:
                raise KeyError(f"Feeder output for GAN training missing latent key '{latent_input_key}'. Available: {list(feeder_outputs_for_g.keys())}")
            if gan_inputs_dict["conditional"] is None:
                raise KeyError(f"Feeder output for GAN training missing conditional key '{conditional_input_key}'. Available: {list(feeder_outputs_for_g.keys())}")
            if gan_inputs_dict["context"] is None:
                raise KeyError(f"Feeder output for GAN training missing context key '{context_input_key}'. Available: {list(feeder_outputs_for_g.keys())}")

            # The GAN model was built with inputs: [gan_latent_input, gan_conditional_input, gan_context_input]
            # So, we pass the data in that order.
            gan_train_inputs = [gan_inputs_dict["latent"], gan_inputs_dict["conditional"], gan_inputs_dict["context"]]
            
            g_loss_metrics = self.gan.train_on_batch(gan_train_inputs, valid, return_dict=True)
            g_loss = g_loss_metrics['loss']


            d_losses_history.append(d_loss)
            g_losses_history.append(g_loss)
            d_accs_real_history.append(d_acc_real)
            d_accs_fake_history.append(d_acc_fake)
            d_accs_history.append(d_acc) # Average

            epoch_duration = time.time() - start_time_epoch
            
            # Update learning rates for logging (they might have changed due to manual LR reduction)
            current_lr_g = float(tf.keras.backend.get_value(self.g_optimizer.learning_rate))
            current_lr_d = float(tf.keras.backend.get_value(self.d_optimizer.learning_rate))

            print(f"Epoch {epoch+1}/{self.params['gan_epochs']} [{epoch_duration:.2f}s] - D_loss: {d_loss:.4f}, D_acc_real: {d_acc_real:.4f}, D_acc_fake: {d_acc_fake:.4f}, D_acc_avg: {d_acc:.4f}, G_loss: {g_loss:.4f} (LR G: {current_lr_g:.2e}, LR D: {current_lr_d:.2e})")

            # Manual ReduceLROnPlateau
            if self.params["enable_reduce_lr_on_plateau"]:
                metric_to_monitor_lr = g_loss if self.params["lr_monitor_metric"] == "g_loss" else d_loss
                if metric_to_monitor_lr < (self.best_lr_metric - self.params["lr_min_delta"]): # Improvement
                    self.best_lr_metric = metric_to_monitor_lr
                    self.lr_patience_counter = 0
                else: # No improvement
                    self.lr_patience_counter += 1

                if self.lr_patience_counter >= self.params["lr_patience"]:
                    old_lr_g = float(tf.keras.backend.get_value(self.g_optimizer.learning_rate))
                    old_lr_d = float(tf.keras.backend.get_value(self.d_optimizer.learning_rate))

                    new_lr_g = max(old_lr_g * self.params["lr_reduction_factor"], self.params["min_lr_g"])
                    new_lr_d = max(old_lr_d * self.params["lr_reduction_factor"], self.params["min_lr_d"])
                    
                    changed_lr = False
                    if new_lr_g < old_lr_g: # Check if change actually happens
                        tf.keras.backend.set_value(self.g_optimizer.learning_rate, new_lr_g)
                        self.logger.info(f"Epoch {epoch+1}: Reduced generator LR from {old_lr_g:.2e} to {new_lr_g:.2e}")
                        changed_lr = True
                    if new_lr_d < old_lr_d: # Check if change actually happens
                        tf.keras.backend.set_value(self.d_optimizer.learning_rate, new_lr_d)
                        self.logger.info(f"Epoch {epoch+1}: Reduced discriminator LR from {old_lr_d:.2e} to {new_lr_d:.2e}")
                        changed_lr = True
                    
                    if changed_lr:
                        self.lr_patience_counter = 0 # Reset counter after reduction
                        self.best_lr_metric = metric_to_monitor_lr # Reset best metric to current after reduction
                # self.logger.debug(f"  ReduceLROnPlateau: Counter {self.lr_patience_counter}/{self.params['lr_patience']}, Best Metric: {self.best_lr_metric:.4f}, Current: {metric_to_monitor_lr:.4f}")


            # Manual EarlyStopping
            if self.params["enable_early_stopping"]:
                metric_to_monitor_es = g_loss if self.params["es_monitor_metric"] == "g_loss" else d_loss
                if metric_to_monitor_es < (self.best_es_metric - self.params["es_min_delta"]): # Improvement
                    self.best_es_metric = metric_to_monitor_es
                    self.es_patience_counter = 0
                else: # No improvement
                    self.es_patience_counter += 1

                if self.es_patience_counter >= self.params["es_patience"]:
                    self.logger.info(f"Early stopping triggered at epoch {epoch+1}.")
                    # print(f"  EarlyStopping: Counter {self.es_patience_counter}/{self.params['es_patience']}, Best Metric: {self.best_es_metric:.4f}. Stopping training.")
                    break 
                # self.logger.debug(f"  EarlyStopping: Counter {self.es_patience_counter}/{self.params['es_patience']}, Best Metric: {self.best_es_metric:.4f}, Current: {metric_to_monitor_es:.4f}")


            if epoch > 0 and epoch % self.params["gan_save_interval"] == 0:
                self.logger.info(f"Saving models and plotting losses at epoch {epoch+1}...")
                self.save_models(epoch=epoch + 1) # Pass current epoch number
                self._plot_losses(d_losses_history, g_losses_history, epoch=epoch + 1)

        self.logger.info("GAN Training finished.")
        self.save_models(epoch="final") # Save final models
        self._plot_losses(d_losses_history, g_losses_history, epoch="final") # Plot final losses
        # Log final learning rates
        final_lr_g = float(tf.keras.backend.get_value(self.g_optimizer.learning_rate))
        final_lr_d = float(tf.keras.backend.get_value(self.d_optimizer.learning_rate))
        self.logger.info(f"Final Generator LR: {final_lr_g:.2e}, Final Discriminator LR: {final_lr_d:.2e}")

    def _calculate_technical_indicators(self, base_features_batch_np: np.ndarray) -> np.ndarray:
        """
        Calculates technical indicators for a batch of base features.
        Input: base_features_batch_np (batch_size, seq_len, num_base_features)
        Output: combined_features_batch_np (batch_size, seq_len, num_features_for_discriminator)
        """
        batch_size, seq_len, num_input_features = base_features_batch_np.shape
        
        logger.debug(f"[_calculate_technical_indicators] Input base_features_batch_np shape: {base_features_batch_np.shape}")
        logger.debug(f"[_calculate_technical_indicators] self.num_base_features: {self.num_base_features}")
        logger.debug(f"[_calculate_technical_indicators] self.base_feature_names_ordered: {self.base_feature_names_ordered}")
        logger.debug(f"[_calculate_technical_indicators] self.ti_names_to_calculate: {self.ti_names_to_calculate}")
        logger.debug(f"[_calculate_technical_indicators] self.tas_strategy_for_discriminator_tis: {self.tas_strategy_for_discriminator_tis}")
        logger.debug(f"[_calculate_technical_indicators] self.discriminator_feature_names (target for reindex): {self.discriminator_feature_names}")
        logger.debug(f"[_calculate_technical_indicators] self.num_features_for_discriminator (target for output): {self.num_features_for_discriminator}")

        if num_input_features != self.num_base_features:
            error_msg = (
                f"Input for TI calculation has {num_input_features} features, "
                f"but expected {self.num_base_features} base features according to self.num_base_features. "
                f"Base feature names expected: {self.base_feature_names_ordered}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Ensure pandas_ta is imported (consider moving to top of file if not already there)
        try:
            import pandas_ta as ta # type: ignore
        except ImportError:
            logger.error("pandas-ta is not installed. Please install it to calculate technical indicators.")
            # Depending on desired behavior, either raise error or return base features
            raise

        all_combined_features_list = []

        for i in range(batch_size):
            sample_base_features = base_features_batch_np[i, :, :] # Shape: (seq_len, num_base_features)
            
            # Create DataFrame for the current sample's base features
            # Ensure self.base_feature_names_ordered has correct number of names
            if len(self.base_feature_names_ordered) != num_input_features:
                err_msg = f"Mismatch: len(self.base_feature_names_ordered)={len(self.base_feature_names_ordered)} but num_input_features={num_input_features}"
                logger.error(err_msg)
                raise ValueError(err_msg)
                
            df_sample_base = pd.DataFrame(sample_base_features, columns=self.base_feature_names_ordered)
            
            # Map OHLCV if necessary (adjust column names as per your actual base feature names)
            # This mapping is crucial for pandas-ta to find the right columns.
            # Example: if your base features are ['open', 'high', 'low', 'close', 'volume']
            # And pandas-ta expects ['Open', 'High', 'Low', 'Close', 'Volume']
            # ohlcv_map = {
            #     self.ohlc_feature_map.get("open", "open"): "Open",
            #     self.ohlc_feature_map.get("high", "high"): "High",
            #     self.ohlc_feature_map.get("low", "low"): "Low",
            #     self.ohlc_feature_map.get("close", "close"): "Close",
            #     self.ohlc_feature_map.get("volume", "volume"): "Volume"
            # }
            # df_ta_input = df_sample_base.rename(columns=ohlcv_map)
            # For simplicity, assuming df_sample_base column names are already what pandas-ta expects
            # or that self.base_feature_names_ordered are directly usable by pandas-ta.
            # If not, a renaming step like above is needed using self.ohlc_feature_map.
            # For now, we directly use df_sample_base.

            # Initialize an empty DataFrame for TIs to ensure clean concatenation
            df_sample_tis = pd.DataFrame(index=df_sample_base.index)

            if self.tas_strategy_for_discriminator_tis and isinstance(self.tas_strategy_for_discriminator_tis, ta.Strategy):
                logger.debug(f"Sample {i}: Applying TA strategy: {self.tas_strategy_for_discriminator_tis.name}")
                # Calculate TIs using the strategy. pandas-ta appends to the DataFrame.
                # Make a copy to avoid modifying the original df_sample_base if it's used later without TIs.
                df_with_appended_tis = df_sample_base.copy()
                df_with_appended_tis.ta.strategy(self.tas_strategy_for_discriminator_tis, append=True)
                
                # Extract only the newly added TI columns.
                # The columns in df_with_appended_tis that are not in df_sample_base.columns are the TIs.
                ti_cols_added = [col for col in df_with_appended_tis.columns if col not in df_sample_base.columns]
                if ti_cols_added:
                    df_sample_tis = pd.concat([df_sample_tis, df_with_appended_tis[ti_cols_added]], axis=1)
                logger.debug(f"Sample {i}: TI columns added by strategy: {ti_cols_added}")
                logger.debug(f"Sample {i}: df_sample_tis shape after strategy: {df_sample_tis.shape}, columns: {list(df_sample_tis.columns)}")

            else:
                logger.warning(f"Sample {i}: No valid TA strategy found or self.ti_names_to_calculate is empty. No TIs will be calculated by strategy.")
            
            # Concatenate base features with calculated TIs
            df_with_tas = pd.concat([df_sample_base, df_sample_tis], axis=1)
            logger.debug(f"Sample {i}: df_with_tas (base + TIs) shape: {df_with_tas.shape}, columns: {list(df_with_tas.columns)}")
            
                       
            # Reindex to ensure all expected features for the discriminator are present and in order
            # This will add any missing columns (e.g., TIs that were expected but not calculated) as NaN
            df_final_sample = df_with_tas.reindex(columns=self.discriminator_feature_names)
            logger.debug(f"Sample {i}: df_final_sample after reindex shape: {df_final_sample.shape}, columns: {list(df_final_sample.columns)}")
            
            # Fill NaNs - common at the start of series due to TA lookback periods or if TIs weren't calculable.
            df_final_sample = df_final_sample.fillna(0) 

            all_combined_features_list.append(df_final_sample.to_numpy())

        combined_batch_np = np.stack(all_combined_features_list, axis=0)
        logger.debug(f"[_calculate_technical_indicators] Final combined_batch_np shape: {combined_batch_np.shape}")
        

        
        if combined_batch_np.shape[-1] != self.num_features_for_discriminator:
            error_msg = (
                f"Output of TI calculation has {combined_batch_np.shape[-1]} features, "
                f"but discriminator expects {self.num_features_for_discriminator}. "
                f"Expected feature names: {self.discriminator_feature_names}. " # Corrected: self.discriminator_feature_names
                f"Resulting columns: {list(df_final_sample.columns) if 'df_final_sample' in locals() else 'Error before final df construction'}."
            )
            logger.error(error_msg) # Corrected indentation
            raise ValueError(error_msg) # Corrected indentation
            
        return combined_batch_np

    def _plot_losses(self, d_losses: List[float], g_losses: List[float], epoch: Union[int, str]):
        if not d_losses or not g_losses:
            self.logger.warning("Loss history is empty, skipping plotting.")
            return

        plt.figure(figsize=(12, 6))
        plt.plot(d_losses, label=f'Discriminator Loss (Avg: {np.mean(d_losses[-self.params["gan_save_interval"]:]):.4f})' if len(d_losses) > 1 else 'Discriminator Loss')
        plt.plot(g_losses, label=f'Generator Loss (Avg: {np.mean(g_losses[-self.params["gan_save_interval"]:]):.4f})' if len(g_losses) > 1 else 'Generator Loss')
        
        title_epoch = f"Epoch {epoch}" if epoch != "final" else "Final"
        plt.title(f'GAN Training Losses ({title_epoch})')
        plt.xlabel('Batch Iteration / Update Step') # Or 'Epoch' if losses are averaged per epoch
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plot_save_dir = os.path.join(self.params["results_base_dir"], self.params["save_plot_dir"])
        os.makedirs(plot_save_dir, exist_ok=True)
        
        dpi = self.params.get("loss_plot_dpi", 300)
        save_path = ""

        try:
            if epoch == "final":
                # Check global config for final loss plot path first
                final_plot_path_override = self.config.get("gan_loss_plot_file") # From global config
                if final_plot_path_override:
                    save_path = final_plot_path_override
                    # Ensure its directory exists
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                else: # Fallback to plugin params
                    plot_filename = self.params.get("final_loss_plot_filename", "gan_loss_plot_final.png")
                    save_path = os.path.join(plot_save_dir, plot_filename)
            else: # Epoch-specific save
                template_key = "loss_plot_epoch_template"
                filename_template = self.params.get(template_key, "gan_loss_plot_epoch_{epoch}.png")
                plot_filename = filename_template.format(epoch=epoch)
                save_path = os.path.join(plot_save_dir, plot_filename)

            if not save_path.endswith(".png"): # Ensure .png
                save_path = os.path.splitext(save_path)[0] + ".png"

            plt.savefig(save_path, dpi=dpi)
            self.logger.info(f"Loss plot for {title_epoch} saved to {save_path}")
        except Exception as e:
            self.logger.error(f"Failed to save loss plot for {title_epoch}: {e}", exc_info=True)
        finally:
            plt.close() # Ensure plot is closed to free memory


    def save_models(self, epoch: Union[int, str]) -> None:
        model_save_base_dir = self.params["results_base_dir"] # Base directory for all results
        model_save_subdir = self.params["save_model_dir"]   # Subdirectory for models
        
        # This is the specific directory like 'examples/results/phase_4_3/models'
        structured_model_save_path = os.path.join(model_save_base_dir, model_save_subdir)
        os.makedirs(structured_model_save_path, exist_ok=True)

        models_to_save = {
            "generator": self.generator,
            "discriminator": self.discriminator,
            "gan": self.gan
        }

        for model_name, model_instance in models_to_save.items():
            if model_instance:
                try:
                    save_path = ""
                    if epoch == "final":
                        final_path_override = None
                        default_final_filename = ""

                        if model_name == "generator":
                            final_path_override = self.config.get("save_generator_sequential_model_file")
                            default_final_filename = self.params["final_generator_model_filename"]
                        elif model_name == "discriminator":
                            final_path_override = self.config.get("save_discriminator_sequential_model_file")
                            default_final_filename = self.params["final_discriminator_model_filename"]
                        elif model_name == "gan":
                            final_path_override = self.config.get("save_gan_sequential_model_file") # Check for global GAN path
                            default_final_filename = self.params["final_gan_model_filename"]
                        
                        if final_path_override:
                            # If global config provides a path, use it directly.
                            # Ensure its directory exists.
                            save_path = final_path_override
                            os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        else:
                            # Use plugin's default filename, placed in the structured directory
                            save_path = os.path.join(structured_model_save_path, default_final_filename)
                    
                    else: # Epoch-specific save, placed in the structured directory
                        epoch_templates = {
                            "generator": self.params["save_generator_epoch_template"],
                            "discriminator": self.params["save_discriminator_epoch_template"],
                            "gan": self.params["save_gan_epoch_template"]
                        }
                        filename_template = epoch_templates[model_name]
                        save_path = os.path.join(structured_model_save_path, filename_template.format(epoch=epoch))

                    # Ensure .keras extension
                    if not save_path.endswith(".keras"):
                        save_path = os.path.splitext(save_path)[0] + ".keras"
                    
                    model_instance.save(save_path) # Default save_format is 'tf', which is .keras directory structure
                    self.logger.info(f"{model_name.capitalize()} model saved to {save_path} (Epoch: {epoch})")
                
                except Exception as e:
                    self.logger.error(f"Failed to save {model_name} model for epoch {epoch}: {e}", exc_info=True)
            else:
                self.logger.warning(f"{model_name.capitalize()} model instance not found, skipping save for epoch {epoch}.")

    def get_generator(self) -> Optional[Model]:
        """Returns the trained generator model."""
        return self.generator

    def get_discriminator(self) -> Optional[Model]: # Corrected Keras Model type
        """Returns the trained discriminator model."""
        return self.discriminator
