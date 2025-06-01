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
import json # Added for saving metrics
import matplotlib.pyplot as plt # Added for plotting
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
            elif ind.lower() not in ["ema", "rsi"]: # Added "rsi"
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

    @staticmethod
    def _tf_calculate_rsi_1d(series: tf.Tensor, length: int) -> tf.Tensor:
        """
        Calculates Relative Strength Index (RSI) for a 1D tf.Tensor using TensorFlow operations.
        Args:
            series: 1D tf.Tensor of shape (seq_len,).
            length: RSI period (integer).
        Returns:
            1D tf.Tensor of shape (seq_len,) with RSI values (0-100).
            Initial values where RSI is not yet defined will be 0.0.
        """
        seq_len_tf = tf.shape(series)[0]

        if length <= 0:
            logger.warning(f"TensorFlowTALayer (_tf_calculate_rsi_1d): RSI length must be positive, got {length}. Returning zeros.")
            return tf.zeros_like(series, dtype=tf.float32)
        
        if tf.cast(seq_len_tf, tf.int32) <= length : # Not enough data for even one RSI value
            logger.debug(f"TensorFlowTALayer (_tf_calculate_rsi_1d): Input series length {seq_len_tf} is less than or equal to RSI period {length}. Returning zeros.")
            return tf.zeros_like(series, dtype=tf.float32)

        # Calculate price differences
        diff = series[1:] - series[:-1]

        # Separate gains and losses
        gains = tf.maximum(diff, 0.0)
        losses = tf.maximum(-diff, 0.0) # Losses are positive values

        # Pad gains and losses to be the same length as the original series for EMA calculation convenience
        # This means the first diff (and thus first gain/loss) corresponds to series[1]
        # So, EMA will effectively start calculating from the (length+1)-th data point of the original series.
        # We will pad the beginning of gains/losses with 0 to align with series length for EMA function.
        gains_padded = tf.concat([tf.zeros([1], dtype=tf.float32), gains], axis=0)
        losses_padded = tf.concat([tf.zeros([1], dtype=tf.float32), losses], axis=0)
        
        # Calculate EMA of gains and losses
        # Note: _tf_calculate_ema_1d returns EMA of the same length as input, with first value being input[0]
        # For RSI, the first 'length' EMAs are typically based on SMA, then switch to EMA.
        # For simplicity here, we use the direct EMA calculation.
        # The effective "first" value for EMA of gains/losses will be based on the first actual gain/loss.
        avg_gain = TensorFlowTALayer._tf_calculate_ema_1d(gains_padded, length)
        avg_loss = TensorFlowTALayer._tf_calculate_ema_1d(losses_padded, length)

        # Calculate RS
        # Add a small epsilon to avg_loss to prevent division by zero
        rs = avg_gain / (avg_loss + tf.keras.backend.epsilon())

        # Calculate RSI
        rsi = 100.0 - (100.0 / (1.0 + rs))
        
        # Handle cases where avg_loss is zero (RS is effectively infinite, RSI is 100)
        rsi = tf.where(avg_loss < tf.keras.backend.epsilon(), tf.ones_like(rsi) * 100.0, rsi)
        # Handle cases where avg_gain is also zero (RSI can be NaN from 0/0, should be neutral e.g. 50, or 0 if following fillna(0) logic)
        # If both avg_gain and avg_loss are near zero, rsi can be 0.0 from 100.0 - (100.0 / 1.0). This is acceptable.

        # The first 'length' values of RSI are typically undefined.
        # Our EMA implementation starts calculation from the first element.
        # For RSI, we should output 0 for the initial period where it's not well-defined.
        # Create a mask for the initial undefined period.
        # The first actual RSI value can be computed after 'length' price changes, which means at index 'length' of the series.
        
        # Create an array of zeros for the initial part
        initial_zeros = tf.zeros([length], dtype=tf.float32)
        # Get the calculated RSI values (which are for indices length to seq_len_tf-1)
        calculated_rsi = rsi[length:]
        
        # Concatenate to form the final RSI series
        final_rsi = tf.concat([initial_zeros, calculated_rsi], axis=0)
        
        # Ensure output shape is correct, should match input series shape
        # If seq_len_tf is exactly length+1, calculated_rsi will have 1 element. initial_zeros has 'length' elements. Total length+1.
        # This logic seems to produce one less element than seq_len_tf if not careful.
        # Let's re-evaluate the padding/slicing for RSI.
        # The `rsi` tensor is already of shape `seq_len_tf`.
        # We just need to zero out the first `length` elements.

        condition = tf.range(seq_len_tf) < length
        final_rsi_corrected = tf.where(condition, tf.zeros_like(series), rsi)

        return final_rsi_corrected


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
                
                elif indicator_lower == "rsi":
                    # RSI typically uses 'close' price. If parsed_col_name is not set, we might default or require it.
                    # For now, assume parsed_col_name gives the correct column (e.g., "Close_RSI_14" -> "Close")
                    # If TI is just "RSI_14", parsed_col_name might be None. Need a default (e.g. last base feature) or error.
                    # Let's assume for now parsed_col_name must be valid for RSI.
                    if parsed_col_name in self.base_feature_name_to_idx:
                        col_idx = self.base_feature_name_to_idx[parsed_col_name]
                        feature_series_batch = inputs[:, :, col_idx] # (batch_size, seq_len)
                        
                        rsi_length = parsed_params.get('length')
                        if rsi_length is not None and isinstance(rsi_length, int) and rsi_length > 0:
                            rsi_output_2d_batch = tf.map_fn(
                                lambda s: self._tf_calculate_rsi_1d(s, length=rsi_length),
                                feature_series_batch,
                                fn_output_signature=tf.float32
                            ) # Output shape: (batch_size, seq_len)
                            current_ti_tensor_for_concat = tf.expand_dims(rsi_output_2d_batch, axis=-1) # Shape: (batch_size, seq_len, 1)
                        else:
                            logger.warning(f"TensorFlowTALayer call: Invalid or missing 'length' ({rsi_length}) for RSI '{ti_full_name}'. Using zeros.")
                    elif not parsed_col_name and 'close' in self.base_feature_name_to_idx: # Default to 'close' if TI is like "RSI_14"
                        logger.info(f"TensorFlowTALayer call: RSI '{ti_full_name}' had no explicit column, defaulting to 'close'.")
                        col_idx = self.base_feature_name_to_idx['close']
                        feature_series_batch = inputs[:, :, col_idx]
                        rsi_length = parsed_params.get('length')
                        if rsi_length is not None and isinstance(rsi_length, int) and rsi_length > 0:
                            rsi_output_2d_batch = tf.map_fn(
                                lambda s: self._tf_calculate_rsi_1d(s, length=rsi_length),
                                feature_series_batch,
                                fn_output_signature=tf.float32
                            )
                            current_ti_tensor_for_concat = tf.expand_dims(rsi_output_2d_batch, axis=-1)
                        else:
                            logger.warning(f"TensorFlowTALayer call: Invalid or missing 'length' ({rsi_length}) for RSI '{ti_full_name}' (defaulted to close). Using zeros.")
                    else:
                        logger.warning(f"TensorFlowTALayer call: Base feature column '{parsed_col_name}' (or default 'close') for RSI '{ti_full_name}' not found. Keys: {list(self.base_feature_name_to_idx.keys())}. Using zeros.")

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
        "gan_save_interval": 500,
        "latent_dim": 32,
        "seq_len": 18,
        # "gan_model_dir": "models/gan_trained", # Replaced by new path structure
        "discriminator_conv_filters": [64, 128],
        "discriminator_conv_kernel_size": 3,
        "discriminator_lstm_units": 64,
        "discriminator_dropout_rate": 0.3,
        "gan_generator_output_actual_seq_len": 1,
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

        # New structured path configuration
        "results_base_dir": "examples/results/phase_4_3",
        "save_model_dir": "models", # Subdirectory for .keras files under results_base_dir
        "save_plot_dir": "plots",   # Subdirectory for .png plots under results_base_dir
        "save_metrics_dir": "metrics", # Subdirectory for .json metrics under results_base_dir

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
        
        "training_metrics_filename": "training_metrics.json"
    }

    def __init__(self, config: Dict[str, Any], generator_plugin_instance: Optional[Any] = None, feeder_plugin_instance: Optional[Any] = None, preprocessor_plugin_instance: Optional[Any] = None):
        self.logger = logger # Ensure logger is accessible
        self.config = config # Store global config
        self.params = deepcopy(self.plugin_params) # Start with plugin defaults
        # self.params.update(self.config) # Initial population - this will be handled by set_params
        
        # Initialize attributes that might be set by set_params or _initialize_core_parameters_from_config
        # to prevent AttributeErrors if accessed early.
        self.base_feature_names_ordered: List[str] = []
        self.num_base_features: int = 0
        self.ti_names_to_calculate: List[str] = []
        self.num_tis: int = 0
        self.discriminator_feature_names: List[str] = []
        self.num_features_for_discriminator: int = 0
        self.seq_len: int = self.params.get("seq_len", 18) # Default from params
        self.gen_input_seq_len: int = self.params.get("seq_len", 18)
        self.gen_input_latent_dim: int = self.params.get("latent_dim", 32)
        self.generator_output_actual_seq_len: int = self.params.get("gan_generator_output_actual_seq_len", 1)
        self.actual_generator_output_dim: int = 0
        self.generator: Optional[Model] = None # Initialize generator attribute
        self.generator_actual_input_names_ordered: List[str] = [] # Initialize attribute

        self.generator_plugin_instance = generator_plugin_instance
        self.feeder_plugin_instance = feeder_plugin_instance
        self.preprocessor_plugin_instance = preprocessor_plugin_instance

        # Set parameters from the global config, overriding plugin defaults
        self.set_params(**self.config) # This will call _initialize_core_parameters_from_config internally if structured so, or we call it after.

        # Try to get the generator model from the plugin instance
        if self.generator_plugin_instance and hasattr(self.generator_plugin_instance, 'get_model'): # Use get_model()
            self.generator: Optional[Model] = self.generator_plugin_instance.get_model()
            if self.generator:
                self.logger.info("Successfully retrieved generator model from plugin via get_model().")
                if hasattr(self.generator, 'inputs') and self.generator.inputs:
                    self.generator_actual_input_names_ordered = [inp.name.split(':')[0] for inp in self.generator.inputs]
                    self.logger.info(f"Generator actual input names ordered: {self.generator_actual_input_names_ordered}")
                else:
                    self.logger.warning("Generator model retrieved but has no inputs or inputs attribute is missing.")
                    self.generator_actual_input_names_ordered = []
                # Plot and save generator model architecture
                try:
                    plot_dir = os.path.join(self.params["results_base_dir"], self.params["save_plot_dir"])
                    os.makedirs(plot_dir, exist_ok=True)
                    plot_file_name = self.params.get("generator_model_plot_file", "generator_architecture.png")
                    plot_path = os.path.join(plot_dir, plot_file_name)
                    dpi = self.params.get("model_plot_dpi", 300)
                    
                    # Ensure model is built if it has a build method and is not built
                    if hasattr(self.generator, 'built') and not self.generator.built and hasattr(self.generator, '_build_input_shape'):
                        try:
                            self.generator.build(self.generator._build_input_shape) # Example, might need specific shape
                            self.logger.info("Attempted to build generator model before plotting.")
                        except Exception as e_build:
                            self.logger.warning(f"Could not explicitly build generator before plotting: {e_build}")

                    plot_model(self.generator, to_file=plot_path, show_shapes=True, dpi=dpi, expand_nested=True, show_layer_activations=True)
                    self.logger.info(f"Generator architecture plot saved to {plot_path}")
                except Exception as e:
                    self.logger.error(f"Failed to plot generator model: {e}", exc_info=True)

                # Update actual_generator_output_dim based on the retrieved model
                if self.generator.output_shape and len(self.generator.output_shape) > 1 and isinstance(self.generator.output_shape[-1], int):
                    self.actual_generator_output_dim = self.generator.output_shape[-1]
                else: # Try output.shape
                    try:
                        self.actual_generator_output_dim = self.generator.output.shape[-1]
                    except Exception:
                        self.logger.warning(f"Could not reliably determine actual_generator_output_dim from generator output shape: {self.generator.output_shape} or .output.shape. Defaulting to 0.")
                        self.actual_generator_output_dim = 0
                
                # Check for mismatch (this logic is also in set_params, good to have it early if G is available)
                configured_num_base_features = len(self.params.get("base_feature_names_ordered", []))
                if self.actual_generator_output_dim != 0 and configured_num_base_features > 0 and self.actual_generator_output_dim != configured_num_base_features:
                     self.logger.warning(
                        f"GANTrainerPlugin (__init__): MISMATCH! Actual generator output feature dimension ({self.actual_generator_output_dim}) "
                        f"differs from configured num_base_features ({configured_num_base_features}). "
                        "_build_gan will attempt to slice."
                    )

            else: # self.generator is None
                self.logger.error("Failed to get generator model from generator_plugin_instance (get_model() returned None).")
                self.generator = None # Ensure it's None
                self.actual_generator_output_dim = 0
                self.generator_actual_input_names_ordered = [] # Ensure it's empty if no generator
        elif self.generator_plugin_instance and hasattr(self.generator_plugin_instance, 'sequential_model'): # Fallback to direct attribute
            self.generator: Optional[Model] = self.generator_plugin_instance.sequential_model
            if self.generator:
                 self.logger.info("Retrieved generator model from plugin via sequential_model attribute.")
                 if hasattr(self.generator, 'inputs') and self.generator.inputs:
                    self.generator_actual_input_names_ordered = [inp.name.split(':')[0] for inp in self.generator.inputs]
                    self.logger.info(f"Generator actual input names ordered (from sequential_model): {self.generator_actual_input_names_ordered}")
                 else:
                    self.logger.warning("Generator model (sequential_model) retrieved but has no inputs or inputs attribute is missing.")
                    self.generator_actual_input_names_ordered = []
                 # (Plotting and dim check logic as above) ...
            else:
                self.logger.error("Generator model (sequential_model) is None in generator_plugin_instance.")
                self.generator = None
                self.actual_generator_output_dim = 0
                self.generator_actual_input_names_ordered = [] # Ensure it's empty if no generator
        else:
            self.logger.error("GeneratorPlugin instance not provided or does not have get_model method or sequential_model attribute.")
            self.generator = None
            self.actual_generator_output_dim = 0
            self.generator_actual_input_names_ordered = [] # Ensure it's empty if no generator
        
        # _initialize_core_parameters_from_config() is called within set_params in my typical structure,
        # or should be called after set_params if it relies on fully merged params.
        # Given self.set_params(**self.config) was called, core params should be initialized.
        # If not, it needs to be called here:
        # self._initialize_core_parameters_from_config() # Ensure this is called after params are set.

        # Initialize dimensions for GAN model inputs (latent is from core_params, conditional/context here)
        self.conditional_dim_for_generator = 0
        if self.feeder_plugin_instance and hasattr(self.feeder_plugin_instance, 'get_conditional_dim'):
            self.conditional_dim_for_generator = self.feeder_plugin_instance.get_conditional_dim()
            self.logger.info(f"GANTrainerPlugin (__init__): conditional_dim_for_generator from feeder: {self.conditional_dim_for_generator}")
        # ... (rest of conditional_dim_for_generator logic from existing code) ...
        elif self.params: 
            date_cond_feats = self.params.get('feeder_date_features_for_conditioning', [])
            fund_cond_feats = self.params.get('feeder_fundamental_features_for_conditioning', [])
            self.conditional_dim_for_generator = (len(date_cond_feats) * 2) + len(fund_cond_feats) 
            self.logger.info(f"GANTrainerPlugin (__init__): conditional_dim_for_generator from params (fallback): {self.conditional_dim_for_generator}")
        else:
            self.logger.warning("GANTrainerPlugin (__init__): Could not determine conditional_dim_for_generator.")


        self.context_dim_for_generator = self.params.get('context_vector_dim', 64)
        self.logger.info(f"GANTrainerPlugin (__init__): context_dim_for_generator set to {self.context_dim_for_generator}")
        
        self.g_optimizer = Adam(learning_rate=self.params.get("generator_lr", 1e-4), beta_1=self.params.get("generator_beta1", 0.5))
        self.d_optimizer = Adam(learning_rate=self.params.get("discriminator_lr", 1e-4), beta_1=self.params.get("discriminator_beta1", 0.5))
        self.logger.info("Optimizers initialized.")

        self.discriminator: Optional[Model] = self._build_discriminator()
        if self.discriminator:
            self.discriminator.compile(loss='binary_crossentropy', optimizer=self.d_optimizer, metrics=['accuracy'])
            self.logger.info("Discriminator compiled.")
        else:
            self.logger.error("Discriminator model building failed in __init__.")

        self.gan: Optional[Model] = self._build_gan()
        if not self.gan:
             self.logger.error("GAN model building failed in __init__.")

        self.best_lr_metric = float('inf')
        self.lr_patience_counter = 0
        self.best_es_metric = float('inf')
        self.es_patience_counter = 0
        
        # For storing all metrics for JSON export
        self.full_metrics_history: List[Dict[str, Any]] = []
        
        self.logger.info("GANTrainerPlugin initialized with new path configurations.")

    def _initialize_core_parameters_from_config(self):
        """Helper to initialize parameters needed before model building, typically also managed by set_params."""
        self.gen_input_seq_len = self.params.get("seq_len", 18)
        self.gen_input_latent_dim = self.params.get("latent_dim", 32)
        
        self.base_feature_names_ordered = self.params.get("base_feature_names_ordered", [])
        self.num_base_features = len(self.base_feature_names_ordered)
        
        all_discriminator_features = self.params.get("feature_names_for_discriminator_ordered", [])
        self.discriminator_feature_names = all_discriminator_features
        
        if all_discriminator_features and self.base_feature_names_ordered:
            self.ti_names_to_calculate = [f for f in all_discriminator_features if f not in self.base_feature_names_ordered]
        else:
            self.ti_names_to_calculate = self.params.get("ti_names_to_calculate", []) # Fallback if derivation fails
            if not self.ti_names_to_calculate: # If still empty, log warning
                 self.logger.warning("_initialize_core_parameters_from_config: Could not derive ti_names_to_calculate and it's not directly in params. TI list is empty.")


        self.num_tis = len(self.ti_names_to_calculate)
        
        # Ensure num_features_for_discriminator is based on the length of the full list
        if self.discriminator_feature_names:
            self.num_features_for_discriminator = len(self.discriminator_feature_names)
        else: # Fallback if discriminator_feature_names is empty for some reason
            self.num_features_for_discriminator = self.num_base_features + self.num_tis
            if not self.discriminator_feature_names: # Log if it was empty
                 self.logger.warning("_initialize_core_parameters_from_config: 'feature_names_for_discriminator_ordered' is empty. num_features_for_discriminator calculated as num_base + num_tis.")


        self.seq_len = self.params.get("seq_len", 18) 

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
            f"num_base_features={self.num_base_features}, num_tis={self.num_tis}, "
            f"num_features_for_discriminator={self.num_features_for_discriminator}, "
            f"seq_len (for D and TI)={self.seq_len}, "
            f"generator_output_actual_seq_len={self.generator_output_actual_seq_len}"
        )
        if self.num_base_features == 0:
            self.logger.warning("_initialize_core_parameters_from_config: num_base_features is 0. This might be problematic for the TI layer and GAN structure.")

    def set_params(self, **params: Any) -> None:
        self.logger.info(f"GANTrainerPlugin updating parameters with keys: {list(params.keys())}")
        # Merge incoming params into self.config first
        self.config.update(params)
        
        # Then, rebuild self.params by taking plugin defaults and overriding with the full self.config
        current_plugin_defaults = deepcopy(self.plugin_params)
        current_plugin_defaults.update(self.config) # self.config now contains initial config + **params
        self.params = current_plugin_defaults

        # After self.params is fully updated, re-initialize core parameters
        self._initialize_core_parameters_from_config()

        # Update generator's actual output dim if generator is available
        if self.generator and hasattr(self.generator, 'output_shape'):
            try: # Try .output_shape first
                if self.generator.output_shape and len(self.generator.output_shape) > 1 and isinstance(self.generator.output_shape[-1], int):
                    self.actual_generator_output_dim = self.generator.output_shape[-1]
                else: # Try .output.shape as a fallback
                    self.actual_generator_output_dim = self.generator.output.shape[-1]
                self.logger.info(f"GANTrainerPlugin (set_params): Updated actual_generator_output_dim: {self.actual_generator_output_dim}")

                if self.actual_generator_output_dim != self.num_base_features:
                    self.logger.warning(
                        f"GANTrainerPlugin (set_params): MISMATCH! Actual generator output dim ({self.actual_generator_output_dim}) "
                        f"vs configured num_base_features ({self.num_base_features}). Slicing will occur in _build_gan."
                    )
            except Exception as e_shape:
                 self.logger.warning(f"GANTrainerPlugin (set_params): Could not get generator output dim: {e_shape}. Using self.num_base_features ({self.num_base_features}) as assumed actual_generator_output_dim.")
                 self.actual_generator_output_dim = self.num_base_features


        # Re-initialize optimizers if learning rates might have changed
        # (This logic is from existing code, seems fine)
        if hasattr(self, 'g_optimizer') and self.g_optimizer is not None:
            current_lr_g = self.params.get("generator_lr", 1e-4); current_beta1_g = self.params.get("generator_beta1", 0.5)
            if self.g_optimizer.learning_rate != current_lr_g or self.g_optimizer.beta_1 != current_beta1_g:
                self.g_optimizer = Adam(learning_rate=current_lr_g, beta_1=current_beta1_g)
                self.logger.info(f"GANTrainerPlugin (set_params): Gen optimizer re-init with LR={current_lr_g:.1e}, Beta1={current_beta1_g}")
        else: self.g_optimizer = Adam(learning_rate=self.params.get("generator_lr", 1e-4), beta_1=self.params.get("generator_beta1", 0.5))

        if hasattr(self, 'd_optimizer') and self.d_optimizer is not None:
            current_lr_d = self.params.get("discriminator_lr", 1e-4); current_beta1_d = self.params.get("discriminator_beta1", 0.5)
            if self.d_optimizer.learning_rate != current_lr_d or self.d_optimizer.beta_1 != current_beta1_d:
                self.d_optimizer = Adam(learning_rate=current_lr_d, beta_1=current_beta1_d)
                self.logger.info(f"GANTrainerPlugin (set_params): Disc optimizer re-init with LR={current_lr_d:.1e}, Beta1={current_beta1_d}")
        else: self.d_optimizer = Adam(learning_rate=self.params.get("discriminator_lr", 1e-4), beta_1=self.params.get("discriminator_beta1", 0.5))
        
        self.logger.info("GANTrainerPlugin (set_params): Parameters updated. Models might need re-build if structural params changed.")
        # Note: The old gan_model_dir is not used with the new path structure.
        # Directories are created on-demand by save_models, _plot_losses etc.

    def _build_discriminator(self) -> Model:
        self.logger.info(f"Building Discriminator with input shape: (None, {self.seq_len}, {self.num_features_for_discriminator})")
        
        data_input = layers.Input(shape=(self.seq_len, self.num_features_for_discriminator), name="discriminator_input")
        x = data_input
        conv_filters = self.params.get("discriminator_conv_filters", [64, 128])
        kernel_size = self.params.get("discriminator_conv_kernel_size", 3)
        for filters_count in conv_filters: # renamed filters to filters_count
            x = layers.Conv1D(filters=filters_count, kernel_size=kernel_size, padding='causal', activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.SpatialDropout1D(self.params.get("discriminator_dropout_rate", 0.3) / 2)(x)
        lstm_units = self.params.get("discriminator_lstm_units", 64)
        x = layers.Bidirectional(layers.LSTM(units=lstm_units, return_sequences=False))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.params.get("discriminator_dropout_rate", 0.3))(x)
        output = layers.Dense(1, activation='sigmoid', name="discriminator_output")(x)
        model = Model(data_input, output, name="Discriminator")
        
        model.summary(print_fn=self.logger.info) # Use self.logger

        try:
            plot_dir = os.path.join(self.params["results_base_dir"], self.params["save_plot_dir"])
            os.makedirs(plot_dir, exist_ok=True)
            plot_file_name = self.params.get("discriminator_model_plot_file", "discriminator_architecture.png")
            plot_path = os.path.join(plot_dir, plot_file_name)
            dpi = self.params.get("model_plot_dpi", 300)
            plot_model(model, to_file=plot_path, show_shapes=True, dpi=dpi, expand_nested=True, show_layer_activations=True)
            self.logger.info(f"Discriminator architecture plot saved to {plot_path}")
        except Exception as e:
            self.logger.warning(f"Could not plot Discriminator model: {e}. Ensure pydot and graphviz are installed.")
        return model

    def _build_gan(self) -> Model:
        if not self.generator or not self.discriminator:
            self.logger.error("Generator or Discriminator not initialized. Cannot build GAN.")
            # Consider raising an error or returning None, consistent with __init__
            return None # Or raise ValueError("Generator or Discriminator not initialized.")
            
        self.logger.info("Building GAN model (Generator + Discriminator)...")
        self.discriminator.trainable = False
        
        gan_latent_input_shape = (self.gen_input_seq_len, self.gen_input_latent_dim)
        gan_latent_input = tf.keras.Input(shape=gan_latent_input_shape, name="gan_input_latent_vector")
        gan_conditional_input = tf.keras.Input(shape=(self.conditional_dim_for_generator,), name="gan_input_conditional_data")
        gan_context_input = tf.keras.Input(shape=(self.context_dim_for_generator,), name="gan_input_context_vector")

        # Order inputs for the VAE Decoder (Generator)
        generator_input_names_from_model = [inp.name.split(':')[0] for inp in self.generator.inputs]
        cfg_latent_name = self.params.get("generator_decoder_input_name_latent")
        cfg_context_name = self.params.get("generator_decoder_input_name_context")
        cfg_conditional_name = self.params.get("generator_decoder_input_name_conditions")

        input_map = {
            cfg_latent_name: gan_latent_input,
            cfg_context_name: gan_context_input,
            cfg_conditional_name: gan_conditional_input,
        }
        generator_feed_inputs_ordered = []
        for model_input_name in generator_input_names_from_model:
            found_match = any(model_input_name.startswith(cfg_name) for cfg_name in input_map if cfg_name)
            if found_match:
                 generator_feed_inputs_ordered.append(next(gan_layer for cfg_name, gan_layer in input_map.items() if cfg_name and model_input_name.startswith(cfg_name)))
            else: self.logger.warning(f"GAN build: Gen input '{model_input_name}' not mapped.")
        if len(generator_feed_inputs_ordered) != len(self.generator.inputs): self.logger.error("GAN build: Mismatch in ordered inputs for gen.")

        generated_data_raw = self.generator(generator_feed_inputs_ordered)
        
        if self.generator_output_actual_seq_len > 0:
            target_reshape_dim = self.actual_generator_output_dim if self.actual_generator_output_dim > 0 else -1
            generated_data_raw = tf.keras.layers.Reshape((self.generator_output_actual_seq_len, target_reshape_dim), name="reshape_generator_output")(generated_data_raw)

        base_features_from_generator = generated_data_raw
        if self.actual_generator_output_dim != self.num_base_features and self.num_base_features > 0 : # Ensure num_base_features is positive
            base_features_from_generator = tf.keras.layers.Lambda(lambda x: x[:, :, :self.num_base_features], name="slice_gen_output")(generated_data_raw)
        
        if self.generator_output_actual_seq_len != self.seq_len:
            if self.generator_output_actual_seq_len == 1 and self.seq_len > 1:
                squeezed_features = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=1), name="squeeze_time_dim")(base_features_from_generator)
                base_features_from_generator = tf.keras.layers.RepeatVector(self.seq_len, name="repeat_vector_for_ti")(squeezed_features)
            else: raise ValueError("GAN build: Mismatch in generator output seq_len vs TI/D seq_len.")

        ti_calculator_layer = TensorFlowTALayer(
            base_feature_names=self.base_feature_names_ordered, 
            ti_names_to_calculate=self.ti_names_to_calculate,
            num_base_features=self.num_base_features,
            num_total_features=self.num_features_for_discriminator,
            seq_len=self.seq_len, name="symbolic_ti_calculator_in_gan" # Unique name
        )
        data_with_tis_for_discriminator = ti_calculator_layer(base_features_from_generator)
        gan_output = self.discriminator(data_with_tis_for_discriminator) 
        actual_gan_model_inputs = [gan_latent_input, gan_conditional_input, gan_context_input]
        gan = tf.keras.Model(inputs=actual_gan_model_inputs, outputs=gan_output, name="gan_combined")
        
        gan.compile(loss='binary_crossentropy', optimizer=self.g_optimizer) # g_optimizer from self
        self.logger.info("GAN model built and compiled.")
        
        gan.summary(print_fn=self.logger.info) # Use self.logger
        try:
            plot_dir = os.path.join(self.params["results_base_dir"], self.params["save_plot_dir"])
            os.makedirs(plot_dir, exist_ok=True)
            plot_file_name = self.params.get("gan_model_plot_file", "gan_architecture.png")
            plot_path = os.path.join(plot_dir, plot_file_name)
            dpi = self.params.get("model_plot_dpi", 300)
            plot_model(gan, to_file=plot_path, show_shapes=True, dpi=dpi, expand_nested=True, show_layer_activations=True)
            self.logger.info(f"GAN architecture plot saved to {plot_path}")

            # Plot generator component as used in GAN (if different from standalone plot, or for context)
            if self.generator:
                 gen_plot_filename = self.params.get("generator_model_plot_file", "generator_architecture.png").replace(".png", "_component_in_gan.png")
                 gen_plot_path = os.path.join(plot_dir, gen_plot_filename)
                 plot_model(self.generator, to_file=gen_plot_path, show_shapes=True, dpi=dpi, expand_nested=True, show_layer_activations=True)
                 self.logger.info(f"GAN's generator component plot saved to {gen_plot_path}")

        except Exception as e:
            self.logger.warning(f"Could not plot GAN model or its generator component: {e}. Ensure pydot and graphviz are installed.")
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
        seq_len_param = self.params.get('seq_len', 18)
        if x_train_data_full.shape[0] < seq_len_param:
            self.logger.error(f"GANTrainerPlugin (train): Data length ({x_train_data_full.shape[0]}) is less than seq_len ({seq_len_param}). Cannot create sequences.")
            raise ValueError(f"Data length ({x_train_data_full.shape[0]}) is less than seq_len ({seq_len_param}).")

        num_samples = x_train_data_full.shape[0] - seq_len_param + 1
        num_features = x_train_data_full.shape[1]
        
        x_train_sequences = np.array([x_train_data_full[i:i+seq_len_param] for i in range(num_samples)])
        
        if x_train_sequences.ndim == 2: 
            if num_samples > 0 :
                 x_train_sequences = x_train_sequences.reshape(num_samples, seq_len_param, num_features if num_features > 0 else 1)
            else: 
                if num_features > 0:
                    x_train_sequences = np.empty((0, seq_len_param, num_features))
                else: 
                    self.logger.error(f"GANTrainerPlugin (train): Data has 0 features after processing. Original shape: {data_df.shape}, after numpy conversion: {x_train_data_full.shape}")
                    raise ValueError("Data has 0 features after processing.")


        self.logger.info(f"GANTrainerPlugin (train): Created sequences. Shape: {x_train_sequences.shape}")
        
        x_train_data = x_train_sequences 

        if x_train_data.ndim == 2:
            x_train_data = np.expand_dims(x_train_data, axis=1) 
            self.logger.info(f"GANTrainerPlugin (train): Expanded dims for 2D data. New shape: {x_train_data.shape}")

        current_batch_size = self.params.get("gan_batch_size", 32) 
        valid_labels = np.ones((current_batch_size, 1)) 
        fake_labels = np.zeros((current_batch_size, 1))  

        d_losses_history = []
        g_losses_history = []

        current_lr_g = float(tf.keras.backend.get_value(self.g_optimizer.learning_rate))
        current_lr_d = float(tf.keras.backend.get_value(self.d_optimizer.learning_rate))

        # Print model summaries before training loop
        self.logger.info("Printing Model Summaries before training:")
        if self.generator:
            self.logger.info("--- Generator Summary ---")
            self.generator.summary(print_fn=self.logger.info)
        else:
            self.logger.warning("Generator model is None, cannot print summary.")
        
        if self.discriminator:
            self.logger.info("--- Discriminator Summary ---")
            self.discriminator.summary(print_fn=self.logger.info)
        else:
            self.logger.warning("Discriminator model is None, cannot print summary.")

        if self.gan:
            self.logger.info("--- GAN Combined Model Summary ---")
            self.gan.summary(print_fn=self.logger.info)
        else:
            self.logger.warning("GAN combined model is None, cannot print summary.")
        self.logger.info("--------------------------------------")

        self.logger.info(f"Starting GAN training for {self.params['gan_epochs']} epochs, batch size {current_batch_size}...")
        self.logger.info(f"Initial Generator LR: {current_lr_g:.1e}, Discriminator LR: {current_lr_d:.1e}")
        self.logger.info(f"Generator is FROZEN. Discriminator is TRAINABLE.")
        self.logger.info(f"Discriminator input shape: (batch_size, {self.seq_len}, {self.num_features_for_discriminator})")
        self.logger.info(f"Generator output (base features) target shape for TI: (batch_size, {self.seq_len}, {self.num_base_features})")


        for epoch in range(self.params["gan_epochs"]):
            start_time_epoch = time.time()
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            
            # Get a random batch of real samples
            idx = np.random.randint(0, x_train_data.shape[0], current_batch_size)
            # real_data_batch_raw = x_train_data[idx] # OLD LINE
            real_data_batch_raw_full_features = x_train_data[idx] # (batch_size, seq_len, features_from_preprocessor)

            # Slice to get only base features for TI calculation
            # Ensure self.num_base_features is valid before slicing
            if hasattr(self, 'num_base_features') and self.num_base_features > 0 and \
               real_data_batch_raw_full_features.shape[-1] > self.num_base_features:
                real_data_batch_base_features = real_data_batch_raw_full_features[:, :, :self.num_base_features]
                if hasattr(self, 'logger') and self.logger:
                    self.logger.debug( # Changed to debug
                        f"GANTrainerPlugin (train): Sliced real_data_batch_raw_full_features from {real_data_batch_raw_full_features.shape[-1]} "
                        f"to {self.num_base_features} features for TI calculation (real data)."
                    )
            else:
                real_data_batch_base_features = real_data_batch_raw_full_features
            
            self.logger.debug(f"Shape of real_data_batch_base_features before TI calc: {real_data_batch_base_features.shape}")
            real_data_for_discriminator = self._calculate_technical_indicators(real_data_batch_base_features)

            # Generate a batch of new series using the FeederPlugin and Generator
            feeder_output_d = self.feeder_plugin_instance.generate(n_ticks_to_generate=current_batch_size) # Renamed from feeder_output
            
            # Prepare inputs for the generator model in the correct order
            # self.generator_actual_input_names_ordered is set in __init__ based on generator.inputs
            generator_inputs_d = []
            for input_name in self.generator_actual_input_names_ordered: # Assumes this is set
                generator_inputs_d.append(feeder_output_d[input_name])
            
            # --- Refactored logic for generated_base_features_raw ---
            generated_output_from_model = self.generator.predict(generator_inputs_d, verbose=0)

            # Step 1: Ensure generated_output_from_model is 3D: (batch, actual_gen_seq_len, actual_gen_features)
            current_actual_gen_seq_len: int
            if generated_output_from_model.ndim == 2:
                current_actual_gen_seq_len = 1
                current_actual_gen_features = generated_output_from_model.shape[1]
                
                # Consistency check with self.actual_generator_output_dim
                if self.actual_generator_output_dim > 0 and current_actual_gen_features != self.actual_generator_output_dim:
                    self.logger.warning(
                        f"GANTrainerPlugin (train): Generator output feature count ({current_actual_gen_features}) "
                        f"differs from expected self.actual_generator_output_dim ({self.actual_generator_output_dim}). "
                        f"Using self.actual_generator_output_dim for reshaping."
                    )
                    current_actual_gen_features = self.actual_generator_output_dim
                elif self.actual_generator_output_dim == 0 and current_actual_gen_features > 0:
                     # If actual_generator_output_dim wasn't set, use the one from predict()
                     self.logger.info(f"GANTrainerPlugin (train): self.actual_generator_output_dim was 0, using features from predict(): {current_actual_gen_features}")
                elif current_actual_gen_features == 0: # Should not happen if generator works
                    self.logger.error(f"GANTrainerPlugin (train): Generator predicted 0 features. Shape: {generated_output_from_model.shape}")
                    # Fallback to self.actual_generator_output_dim if it's non-zero, else raise error or use a default like 1
                    current_actual_gen_features = self.actual_generator_output_dim if self.actual_generator_output_dim > 0 else 1 
                    if current_actual_gen_features == 1 and self.actual_generator_output_dim == 0:
                         self.logger.warning("GANTrainerPlugin (train): Falling back to 1 feature for generator output due to 0 features predicted and 0 actual_generator_output_dim.")


                generated_3d_features = generated_output_from_model.reshape((current_batch_size, current_actual_gen_seq_len, current_actual_gen_features))
            elif generated_output_from_model.ndim == 3:
                generated_3d_features = generated_output_from_model
                current_actual_gen_seq_len = generated_output_from_model.shape[1]
            else:
                self.logger.error(f"GANTrainerPlugin (train): Generator output has unexpected ndim: {generated_output_from_model.ndim}. Shape: {generated_output_from_model.shape}")
                raise ValueError(f"Generator output has unexpected ndim: {generated_output_from_model.ndim}")

            # Step 2: Slice to self.num_base_features on the feature dimension
            if self.num_base_features > 0 and generated_3d_features.shape[-1] != self.num_base_features :
                self.logger.debug(f"Slicing generated features from {generated_3d_features.shape[-1]} to {self.num_base_features} base features.")
                base_features_from_generator = generated_3d_features[:, :, :self.num_base_features]
            else: # num_base_features is 0 (problematic) or matches, or not restrictive
                base_features_from_generator = generated_3d_features
            
            self.logger.debug(f"Shape of base_features_from_generator after potential slice: {base_features_from_generator.shape}")


            # Step 3: Adjust sequence length to self.seq_len (for TIs and Discriminator)
            generated_base_features_raw_final: np.ndarray
            if current_actual_gen_seq_len != self.seq_len:
                if current_actual_gen_seq_len == 1 and self.seq_len > 0: # self.seq_len should be > 0
                    if base_features_from_generator.shape[1] != 1: # Should be (B,1,F)
                         self.logger.error(f"Mismatch: Expected base_features_from_generator to have seq_len 1 for repeat, but got {base_features_from_generator.shape[1]}")
                         # Attempt to take the first step if it's longer than 1, though this indicates a logic flaw earlier
                         squeezed_for_repeat = np.squeeze(base_features_from_generator[:,0:1,:], axis=1) if base_features_from_generator.shape[1] > 0 else np.squeeze(base_features_from_generator, axis=1)
                    else:
                        squeezed_for_repeat = np.squeeze(base_features_from_generator, axis=1)
                    
                    generated_base_features_raw_final = np.repeat(np.expand_dims(squeezed_for_repeat, axis=1), self.seq_len, axis=1)
                    self.logger.debug(f"Repeated generated features from seq_len {current_actual_gen_seq_len} to {self.seq_len}. New shape: {generated_base_features_raw_final.shape}")
                elif self.seq_len > 0 and current_actual_gen_seq_len > self.seq_len : # Generator produced more seq len than D needs
                    generated_base_features_raw_final = base_features_from_generator[:, :self.seq_len, :] 
                    self.logger.debug(f"Sliced generated features from seq_len {current_actual_gen_seq_len} to {self.seq_len}. New shape: {generated_base_features_raw_final.shape}")
                elif self.seq_len == 0: # self.seq_len for discriminator is 0, highly problematic
                    self.logger.error(f"self.seq_len is 0. Cannot prepare generated data for discriminator. current_actual_gen_seq_len: {current_actual_gen_seq_len}")
                    # Fallback: use as is, but this will likely fail in TI or D
                    generated_base_features_raw_final = base_features_from_generator
                else: # Other mismatches e.g. current_actual_gen_seq_len is 0, or current_actual_gen_seq_len > 1 but not 1 and not > self.seq_len
                    self.logger.error(
                        f"Cannot reconcile generator output seq_len {current_actual_gen_seq_len} "
                        f"with discriminator seq_len {self.seq_len}. Using generator output as is."
                    )
                    # Fallback, this might be incorrect for the TI layer / Discriminator
                    generated_base_features_raw_final = base_features_from_generator
            else: # seq_len matches
                generated_base_features_raw_final = base_features_from_generator
            # --- End of refactored logic ---

            self.logger.debug(f"Shape of generated_base_features_raw_final before TI calc: {generated_base_features_raw_final.shape}")
            generated_data_for_discriminator = self._calculate_technical_indicators(generated_base_features_raw_final)
            
            if self.discriminator: self.discriminator.trainable = True

            d_loss_real_metrics = self.discriminator.train_on_batch(real_data_for_discriminator, valid_labels, return_dict=True) if self.discriminator else {'loss':0.0, 'accuracy':0.0}
            d_loss_fake_metrics = self.discriminator.train_on_batch(generated_data_for_discriminator, fake_labels, return_dict=True) if self.discriminator else {'loss':0.0, 'accuracy':0.0}
            
            d_loss = 0.5 * (d_loss_real_metrics['loss'] + d_loss_fake_metrics['loss'])
            d_acc_real = d_loss_real_metrics['accuracy']
            d_acc_fake = d_loss_fake_metrics['accuracy']
            d_acc_avg = 0.5 * (d_acc_real + d_acc_fake)
            
            if self.discriminator: self.discriminator.trainable = False 

            feeder_outputs_for_g = self.feeder_plugin_instance.generate(n_ticks_to_generate=current_batch_size)
            gan_inputs_g = [] 
            latent_input_name = self.params.get("generator_decoder_input_name_latent", "decoder_input_z_seq")
            conditional_input_name = self.params.get("generator_decoder_input_name_conditions", "decoder_input_conditions")
            context_input_name = self.params.get("generator_decoder_input_name_context", "decoder_input_h_context")
            gan_feed_keys = [latent_input_name, conditional_input_name, context_input_name]
            for key_name in gan_feed_keys:
                gan_inputs_g.append(feeder_outputs_for_g[key_name])
            
            g_loss_metrics = self.gan.train_on_batch(gan_inputs_g, valid_labels, return_dict=True) if self.gan else {'loss':0.0}
            g_loss = g_loss_metrics['loss']

            d_losses_history.append(d_loss)
            g_losses_history.append(g_loss)

            epoch_duration = time.time() - start_time_epoch

            epoch_metrics = {
                "epoch": epoch + 1,
                "d_loss": d_loss,
                "d_acc_real": d_acc_real,
                "d_acc_fake": d_acc_fake,
                "d_acc_avg": d_acc_avg,
                "g_loss": g_loss,
                "lr_g": float(tf.keras.backend.get_value(self.g_optimizer.learning_rate)),
                "lr_d": float(tf.keras.backend.get_value(self.d_optimizer.learning_rate)),
                "duration_sec": epoch_duration
            }
            self.full_metrics_history.append(epoch_metrics)

            log_msg = (
                f"Epoch {epoch+1}/{self.params['gan_epochs']} [{epoch_duration:.2f}s] - "
                f"D_loss: {d_loss:.4f} (Real Acc: {d_acc_real*100:.2f}%, Fake Acc: {d_acc_fake*100:.2f}%, Avg Acc: {d_acc_avg*100:.2f}%) - "
                f"G_loss: {g_loss:.4f} - "
                f"LR G: {epoch_metrics['lr_g']:.1e}, LR D: {epoch_metrics['lr_d']:.1e}"
            )
            self.logger.info(log_msg) 

            if self.params["gan_save_interval"] > 0 and (epoch + 1) % self.params["gan_save_interval"] == 0:
                self.save_models(epoch=epoch + 1)
                if d_losses_history and g_losses_history: 
                     self._plot_losses(d_losses_history, g_losses_history, epoch=epoch + 1)

        self.logger.info("GAN Training finished.")
        self.save_models(epoch="final") 
        if d_losses_history and g_losses_history:
            self._plot_losses(d_losses_history, g_losses_history, epoch="final")
        self._save_training_metrics(epoch="final") 

        self.logger.info(f"Final models, plots, and metrics should be in {self.params['results_base_dir']}")

    def _calculate_technical_indicators(self, base_features_batch_np: np.ndarray) -> np.ndarray:
        """
        Calculates technical indicators for a batch of base features.
        Input: base_features_batch_np (batch_size, seq_len, num_base_features)
        Output: combined_features_batch_np (batch_size, seq_len, num_features_for_discriminator)
        """
        batch_size, seq_len_ti, num_input_features_ti = base_features_batch_np.shape # Renamed for clarity
        
        logger.debug(f"[_calculate_technical_indicators] Input base_features_batch_np shape: {base_features_batch_np.shape}")
        logger.debug(f"[_calculate_technical_indicators] self.num_base_features: {self.num_base_features}")
        logger.debug(f"[_calculate_technical_indicators] self.base_feature_names_ordered: {self.base_feature_names_ordered}")
        logger.debug(f"[_calculate_technical_indicators] self.ti_names_to_calculate: {self.ti_names_to_calculate}")
        logger.debug(f"[_calculate_technical_indicators] self.tas_strategy_for_discriminator_tis: {self.tas_strategy_for_discriminator_tis}")
        logger.debug(f"[_calculate_technical_indicators] self.discriminator_feature_names (target for reindex): {self.discriminator_feature_names}")
        logger.debug(f"[_calculate_technical_indicators] self.num_features_for_discriminator (target for output): {self.num_features_for_discriminator}")

        if num_input_features_ti != self.num_base_features:
            error_msg = (
                f"Input for TI calculation has {num_input_features_ti} features, "
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
            if len(self.base_feature_names_ordered) != num_input_features_ti:
                err_msg = f"Mismatch: len(self.base_feature_names_ordered)={len(self.base_feature_names_ordered)} but num_input_features_ti={num_input_features_ti}"
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

        plt.figure(figsize=(12, 6)) # Increased figure size
        plt.plot(d_losses, label=f'Discriminator Loss (Avg Last {self.params.get("gan_save_interval",100)}: {np.mean(d_losses[-self.params.get("gan_save_interval",100):]):.4f})' if len(d_losses) > 1 else 'Discriminator Loss')
        plt.plot(g_losses, label=f'Generator Loss (Avg Last {self.params.get("gan_save_interval",100)}: {np.mean(g_losses[-self.params.get("gan_save_interval",100):]):.4f})' if len(g_losses) > 1 else 'Generator Loss')
        
        title_epoch_suffix = f"Epoch {epoch}" if isinstance(epoch, int) else "Final"
        plt.title(f'GAN Training Losses ({title_epoch_suffix})')
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
                final_plot_path_override = self.config.get("gan_loss_plot_file") # Global config override
                if final_plot_path_override:
                    save_path = final_plot_path_override
                    os.makedirs(os.path.dirname(save_path), exist_ok=True) # Ensure dir for custom path
                else: # Fallback to plugin params structure
                    plot_filename = self.params.get("final_loss_plot_filename", "gan_loss_plot_final.png")
                    save_path = os.path.join(plot_save_dir, plot_filename)
            else: # Epoch-specific save
                template_key = "loss_plot_epoch_template"
                filename_template = self.params.get(template_key, "gan_loss_plot_epoch_{epoch}.png")
                plot_filename = filename_template.format(epoch=epoch)
                save_path = os.path.join(plot_save_dir, plot_filename)

            if not save_path.endswith(".png"): save_path = os.path.splitext(save_path)[0] + ".png"

            plt.savefig(save_path, dpi=dpi)
            self.logger.info(f"Loss plot for {title_epoch_suffix} saved to {save_path}")
        except Exception as e:
            self.logger.error(f"Failed to save loss plot for {title_epoch_suffix}: {e}", exc_info=True)
        finally:
            plt.close() # Ensure plot is closed

    def save_models(self, epoch: Union[int, str]) -> None:
        model_save_base_dir = self.params.get("results_base_dir", "examples/results/phase_4_3")
        model_save_subdir = self.params.get("save_model_dir", "models")
        
        # Specific directory like 'examples/results/phase_4_3/models'
        structured_model_save_path = os.path.join(model_save_base_dir, model_save_subdir)
        os.makedirs(structured_model_save_path, exist_ok=True)

        models_to_save = {
            "generator": self.generator,
            "discriminator": self.discriminator,
            "gan": self.gan # Also save the combined GAN model
        }

        for model_name, model_instance in models_to_save.items():
            if model_instance:
                try:
                    save_path = ""
                    default_epoch_template = f"{model_name}_epoch_{{epoch}}.keras" # Generic default
                    default_final_filename = f"{model_name}_final.keras"

                    if epoch == "final":
                        final_path_override_key = f"save_{model_name}_sequential_model_file" # e.g., save_generator_sequential_model_file
                        final_path_override = self.config.get(final_path_override_key)
                        
                        param_final_filename_key = f"final_{model_name}_model_filename" # e.g., final_generator_model_filename
                        specific_final_filename = self.params.get(param_final_filename_key, default_final_filename)
                        
                        if final_path_override:
                            save_path = final_path_override
                            os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        else:
                            save_path = os.path.join(structured_model_save_path, specific_final_filename)
                    
                    else: # Epoch-specific save
                        param_epoch_template_key = f"save_{model_name}_epoch_template" # e.g., save_generator_epoch_template
                        filename_template = self.params.get(param_epoch_template_key, default_epoch_template)
                        save_path = os.path.join(structured_model_save_path, filename_template.format(epoch=epoch))

                    if not save_path.endswith(".keras"): save_path = os.path.splitext(save_path)[0] + ".keras"
                    
                    model_instance.save(save_path) 
                    self.logger.info(f"{model_name.capitalize()} model saved to {save_path} (Epoch: {epoch})")
                
                except Exception as e:
                    self.logger.error(f"Failed to save {model_name} model for epoch {epoch}: {e}", exc_info=True)
            else:
                self.logger.warning(f"{model_name.capitalize()} model instance not found, skipping save for epoch {epoch}.")

    def _save_training_metrics(self, epoch: Union[int, str]):
        """Saves the collected training metrics to a JSON file."""
        if not self.full_metrics_history:
            self.logger.warning("No metrics history to save.")
            return

        metrics_save_base_dir = self.params.get("results_base_dir", "examples/results/phase_4_3")
        metrics_save_subdir = self.params.get("save_metrics_dir", "metrics")
        
        structured_metrics_save_path = os.path.join(metrics_save_base_dir, metrics_save_subdir)
        os.makedirs(structured_metrics_save_path, exist_ok=True)

        filename = ""
        if epoch == "final":
            filename = self.params.get("training_metrics_filename", "training_metrics_final.json")
        else: # Epoch specific, though typically we save all at the end or one cumulative file
            # For now, let's assume "final" is the main use case for a single file dump.
            # If epoch-specific JSONs are needed, the naming template would be like models/plots.
            # Defaulting to a single final file for now.
            filename = self.params.get("training_metrics_filename", f"training_metrics_epoch_{epoch}.json")


        file_path = os.path.join(structured_metrics_save_path, filename)



        try:
            with open(file_path, 'w') as f:
                json.dump(self.full_metrics_history, f, indent=4)
            self.logger.info(f"Training metrics saved to {file_path}")
        except IOError as e:
            self.logger.error(f"Failed to save training metrics to {file_path}: {e}", exc_info=True)
        except TypeError as e:
            self.logger.error(f"TypeError while saving metrics (possibly non-serializable data): {e}", exc_info=True)


    def get_generator(self) -> Optional[Model]:
        """Returns the trained generator model."""
        return self.generator

    def get_discriminator(self) -> Optional[Model]:
        """Returns the trained discriminator model."""
        return self.discriminator
