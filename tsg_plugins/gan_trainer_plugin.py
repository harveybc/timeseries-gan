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
        self.discriminator: Optional[Model] = None # Explicitly initialize discriminator
        self.gan_model: Optional[Model] = None # Explicitly initialize GAN model
        self.generator_actual_input_names_ordered: List[str] = [] # Initialize attribute
        self.gan_latent_input_keras_name_hint: Optional[str] = None
        self.gan_conditional_input_keras_name_hint: Optional[str] = None
        self.gan_context_input_keras_name_hint: Optional[str] = None

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
                self.generator.summary(print_fn=self.logger.info) # Print generator summary
                if hasattr(self.generator, 'inputs') and self.generator.inputs:
                    self.generator_actual_input_names_ordered = [inp.name.split(':')[0] for inp in self.generator.inputs]
                    self.logger.info(f"Generator actual input names ordered: {self.generator_actual_input_names_ordered}")
                    if self.generator.built:
                        self.logger.info("Generator Model Summary (from get_model path):")
                        self.generator.summary(print_fn=self.logger.info)
                    else:
                        self.logger.info("Generator model (from get_model path) is not built yet, summary not printed in __init__.")
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
                 self.generator.summary(print_fn=self.logger.info) # Print generator summary
                 if hasattr(self.generator, 'inputs') and self.generator.inputs:
                    self.generator_actual_input_names_ordered = [inp.name.split(':')[0] for inp in self.generator.inputs]
                    self.logger.info(f"Generator actual input names ordered (from sequential_model): {self.generator_actual_input_names_ordered}")
                    if self.generator.built:
                        self.logger.info("Generator Model Summary (from sequential_model attribute):")
                        self.generator.summary(print_fn=self.logger.info)
                    else:
                        self.logger.info("Generator model (from sequential_model attribute) is not built yet, summary not printed in __init__.")
                 else:
                    self.logger.warning("Generator model (sequential_model) retrieved but has no inputs or inputs attribute is missing.")
                    self.generator_actual_input_names_ordered = []
                 # Plotting and dim check logic as above
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
        
        # Initialize core parameters from the initial config
        self._initialize_core_parameters_from_config() # This sets up seq_len, features etc.

        # Build models if the generator is available
        # if self.generator:
        #     self._build_discriminator()
        #     self._build_gan()
        # else:
        #     self.logger.warning("Generator model is not available. Discriminator and GAN models will not be built at initialization.")
        # Defer model building to set_params or a dedicated build method after all params are finalized.
        self.logger.info("Deferring Discriminator and GAN model building until parameters are fully set.")

    def _get_config_param(self, key: str, default: Any = None) -> Any:
        """
        Helper method to get a parameter from the config with a default value.
        Args:
            key: Parameter key as string.
            default: Default value to return if the key is not found.
        Returns:
            Value of the parameter from the config, or the default value.
        """
        return self.config.get(key, default)

    def _initialize_core_parameters_from_config(self):
        """Initializes core GAN, Generator, and Discriminator parameters from the config.
        This method focuses on deriving dimensions and input names based on the actual loaded Keras models 
        where possible, falling back to config values if necessary.
        """
        self.logger.info("Initializing core GAN parameters from config and loaded models...")

        # --- Generator Output Parameters (used by Discriminator and GAN construction) ---
        self.generator_output_actual_seq_len = None
        self.generator_output_actual_features = None

        if self.generator and hasattr(self.generator, 'outputs') and self.generator.outputs:
            self.logger.info(f"Attempting to derive output dimensions from generator model: {self.generator.name}")
            self.logger.info(f"Generator model outputs: {self.generator.outputs}")
            # Assuming the primary output for sequence length determination is the first one
            # For a typical timeseries generator, output shape is (batch, seq_len, features)
            try:
                generator_output_shape = self.generator.outputs[0].shape
                self.logger.info(f"Generator model output[0] shape: {generator_output_shape}")
                if len(generator_output_shape) == 3:
                    self.generator_output_actual_seq_len = generator_output_shape[1]
                    self.generator_output_actual_features = generator_output_shape[2]
                    self.logger.info(f"Derived from Generator model output: generator_output_actual_seq_len = {self.generator_output_actual_seq_len}, generator_output_actual_features = {self.generator_output_actual_features}")
                elif len(generator_output_shape) == 2: # Handle cases where generator might output 2D (e.g. if seq_len is 1 and squeezed)
                    self.logger.warning(f"Generator model output shape {generator_output_shape} is 2D. Assuming seq_len=1 and features={generator_output_shape[1]}. This might be unexpected for a sequence generator.")
                    # Try to get seq_len from config as a primary source if model output is ambiguous
                    self.generator_output_actual_seq_len = self.params.get('gan_generator_output_actual_seq_len') or self.params.get('seq_len') or 1
                    self.generator_output_actual_features = generator_output_shape[1]
                    self.logger.info(f"Derived from 2D Generator model output: generator_output_actual_seq_len = {self.generator_output_actual_seq_len} (config/defaulted), generator_output_actual_features = {self.generator_output_actual_features}")
                else:
                    self.logger.warning(f"Generator model output shape {generator_output_shape} is not 3D or 2D. Cannot automatically derive seq_len and features for GAN. Falling back to config.")
            except Exception as e:
                self.logger.error(f"Error accessing generator model outputs or shape: {e}. Falling back to config.")
        else:
            if not self.generator:
                self.logger.warning("Generator Keras model (self.generator) is None. Cannot derive output dimensions from model.")
            elif not hasattr(self.generator, 'outputs') or not self.generator.outputs:
                self.logger.warning(f"Generator Keras model (self.generator: {self.generator.name if self.generator else 'None'}) has no 'outputs' attribute or it's empty. Cannot derive output dimensions from model.")

        # Fallback to config if model-derived values are still None
        if self.generator_output_actual_seq_len is None:
            self.generator_output_actual_seq_len = self.params.get('gan_generator_output_actual_seq_len') or self.params.get('seq_len')
            self.logger.info(f"Using config/fallback for generator_output_actual_seq_len: {self.generator_output_actual_seq_len}")

        if self.generator_output_actual_features is None:
            # Try specific config, then general feature_dim, then from generator_params if available in main_config
            self.generator_output_actual_features = self.params.get('gan_generator_output_actual_features') or \
                                                    self.params.get('num_base_features_generated') or \
                                                    self.params.get('feature_dim')
            
            if not self.generator_output_actual_features and hasattr(self, 'config') and isinstance(self.config, dict):
                generator_params_key = next((key for key in self.config if key.endswith('generator_params')), None)
                if generator_params_key:
                    generator_params = self.config.get(generator_params_key, {})
                    decoder_output_names = generator_params.get('generator_decoder_output_feature_names', 
                                                                generator_params.get('decoder_output_feature_names', [])) # Check both keys
                    if decoder_output_names:
                        self.generator_output_actual_features = len(decoder_output_names)
                        self.logger.info(f"Derived generator_output_actual_features from '{generator_params_key}' sub-config 'decoder_output_feature_names': {self.generator_output_actual_features}")
            
            self.logger.info(f"Using config/fallback for generator_output_actual_features: {self.generator_output_actual_features}")


        # Final check and error if still not determined
        if self.generator_output_actual_seq_len is None or self.generator_output_actual_features is None:
            missing_parts = []
            if self.generator_output_actual_seq_len is None:
                missing_parts.append("sequence length (checked model, params: 'gan_generator_output_actual_seq_len', 'seq_len')")
            if self.generator_output_actual_features is None:
                missing_parts.append("features (checked model, params: 'gan_generator_output_actual_features', 'num_base_features_generated', 'feature_dim', or len of 'decoder_output_feature_names' in generator's own config section)")
            
            error_msg = (f"Critical: Could not determine generator output {', and '.join(missing_parts)}. "
                         "Please ensure the generator Keras model is loaded correctly and has defined outputs, "
                         "OR provide these values in the trainer configuration: \n"
                         "  - 'gan_generator_output_actual_seq_len' or 'seq_len' (for sequence length)\n"
                         "  - 'gan_generator_output_actual_features', 'num_base_features_generated', 'feature_dim', or ensure 'generator_decoder_output_feature_names' is set in the generator's specific config section (for features).")
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        self.logger.info(f"Final determined Generator output: sequence length = {self.generator_output_actual_seq_len}, features = {self.generator_output_actual_features}")

        # --- Base features (subset of generator's output, used for TI calculation) ---
        # This should typically match generator_output_actual_features unless TIs are calculated on a subset
        self.num_base_features = self.params.get('num_base_features_generated', self.generator_output_actual_features)
        if self.num_base_features > self.generator_output_actual_features:
            self.logger.warning(f"Configured 'num_base_features_generated' ({self.num_base_features}) is greater than "
                                f"generator's output features ({self.generator_output_actual_features}). Clamping to generator's output features.")
            self.num_base_features = self.generator_output_actual_features
        self.logger.info(f"Number of base features for TI calculation (num_base_features): {self.num_base_features}")


        # --- Discriminator Input Parameters ---
        # Discriminator's sequence length should match the generator's output sequence length
        self.seq_len = self.generator_output_actual_seq_len 
        self.logger.info(f"Discriminator input sequence length (self.seq_len) set to: {self.seq_len} (from generator output)")
            
        self.base_feature_names_ordered = self.params.get('base_feature_names_ordered', [])
        if not self.base_feature_names_ordered and self.generator_plugin_instance:
            if hasattr(self.generator_plugin_instance, 'params') and isinstance(self.generator_plugin_instance.params, dict):
                gen_plugin_params = self.generator_plugin_instance.params
                self.base_feature_names_ordered = gen_plugin_params.get('generator_decoder_output_feature_names', 
                                                                        gen_plugin_params.get('decoder_output_feature_names', []))
                if self.base_feature_names_ordered:
                    self.logger.info(f"Derived 'base_feature_names_ordered' from GeneratorPlugin's 'decoder_output_feature_names': {self.base_feature_names_ordered}")

        if not self.base_feature_names_ordered:
            self.logger.warning("'base_feature_names_ordered' is not set and could not be derived from GeneratorPlugin. "
                                "This is crucial for naming features for TI calculation and discriminator input. "
                                "Attempting to proceed, but may cause issues. Please configure 'base_feature_names_ordered' in trainer_params.")
            if self.num_base_features > 0: # num_base_features should be determined by now
                self.base_feature_names_ordered = [f"feature_{i}" for i in range(self.num_base_features)]
                self.logger.info(f"Using generic 'base_feature_names_ordered': {self.base_feature_names_ordered}")
        else:
            if len(self.base_feature_names_ordered) != self.num_base_features:
                self.logger.warning(f"Mismatch between length of 'base_feature_names_ordered' ({len(self.base_feature_names_ordered)}) "
                                    f"and 'num_base_features' ({self.num_base_features}). Adjusting 'num_base_features' to {len(self.base_feature_names_ordered)} "
                                    "to match 'base_feature_names_ordered'.")
                self.num_base_features = len(self.base_feature_names_ordered)
        
        self.logger.info(f"Final 'base_feature_names_ordered': {self.base_feature_names_ordered} (Count: {self.num_base_features})")

        self.tas_strategy_for_discriminator_tis = self.params.get('tas_strategy_for_discriminator_tis', [])
        if isinstance(self.tas_strategy_for_discriminator_tis, str):
            try:
                self.tas_strategy_for_discriminator_tis = json.loads(self.tas_strategy_for_discriminator_tis)
            except json.JSONDecodeError:
                self.logger.error(f"Failed to parse 'tas_strategy_for_discriminator_tis' string as JSON: {self.tas_strategy_for_discriminator_tis}. Using empty list.")
                self.tas_strategy_for_discriminator_tis = []
        self.logger.info(f"Technical analysis strategy for discriminator TIs: {self.tas_strategy_for_discriminator_tis}")
        
        # Initialize discriminator_feature_names with base features. TIs will be added later.
        self.discriminator_feature_names = list(self.base_feature_names_ordered)
        
        # num_features_for_discriminator: This will be finalized after a test TI calculation if not explicitly set.
        # For now, we use the configured value or estimate it.
        configured_num_features_disc = self.params.get('num_features_for_discriminator')
        if configured_num_features_disc:
            self.num_features_for_discriminator = configured_num_features_disc
            self.logger.info(f"Using configured 'num_features_for_discriminator': {self.num_features_for_discriminator}. "
                             "Ensure TI strategy and base features combine to this number.")
        else:
            # Estimate based on base features + TIs from strategy (rough estimate for now)
            estimated_ti_features = 0
            for ti_spec in self.tas_strategy_for_discriminator_tis:
                if isinstance(ti_spec, dict) and 'kind' in ti_spec:
                    ti_name_base = ti_spec['kind'].lower()
                    if ti_name_base == "macd": estimated_ti_features += 3
                    elif ti_name_base in ["rsi", "ema", "sma", "roc", "mom", "stochrsi_k", "stochrsi_d"]: estimated_ti_features += 1 
                    elif ti_name_base in ["stoch_k", "stoch_d"]: estimated_ti_features += 1 # typically separate
                    elif ti_name_base == "adx": estimated_ti_features += 3 # ADX, DMP, DMN
                    elif ti_name_base == "bbands": estimated_ti_features += 3 # Lower, Mid, Upper
                    else: estimated_ti_features += 1 # Default assumption for unknown TIs
            
            self.num_features_for_discriminator = self.num_base_features + estimated_ti_features
            self.logger.info(f"Estimated 'num_features_for_discriminator' based on {self.num_base_features} base features and TI strategy: {self.num_features_for_discriminator} (this may be refined after first TI calculation).")


        # --- Generator Input Parameters (used by GAN construction for its inputs) ---
        # These define the shapes of the inputs that the GAN model will pass to the generator internally.
        self.generator_actual_input_names_ordered = []
        self.gen_input_seq_len = None # For latent vector if it's sequential
        self.gen_input_latent_dim = None
        self.conditional_dim_for_generator = None
        self.context_dim_for_generator = None

        # Feeder key names for different generator inputs
        self.feeder_key_name_latent = self.params.get("generator_decoder_input_name_latent", "latent_vector")
        self.feeder_key_name_conditional = self.params.get("generator_decoder_input_name_conditions", "conditional_data")
        self.feeder_key_name_context = self.params.get("generator_decoder_input_name_context", "context_vector")
        self.logger.info(f"Feeder key names for generator inputs: Latent='{self.feeder_key_name_latent}', Conditional='{self.feeder_key_name_conditional}', Context='{self.feeder_key_name_context}'.")

        if self.generator and self.generator.inputs:
            self.logger.info(f"Deriving GAN input dimensions from {len(self.generator.inputs)} generator input layers.")
            # Attempt to map known feeder key names to the generator's input layers by their Keras names
            # This assumes Keras input layers were named according to these feeder keys or a convention.
            # Example Keras input layer names: "input_latent_vector", "input_conditional_data", "input_context_vector"
            # Or, if not named, rely on the order and config params for dimensions.

            # Store the Keras names of the generator inputs for matching and ordering
            # self.generator_actual_input_names_ordered will be populated based on the order of self.generator.inputs
            # and will store the Keras name of each input layer.
            # This list is then used in the train() method to prepare inputs for generator.predict() in the correct order.

            # Default to config values, then try to override with model-derived values.
            self.gen_input_latent_dim = self.params.get('latent_dim')
            self.gen_input_seq_len = self.params.get('seq_len') # Often, latent vector might share seq_len if it's a sequence
            self.conditional_dim_for_generator = self.params.get('conditional_dim')
            self.context_dim_for_generator = self.params.get('context_dim')
            self.logger.info(f"Initial GAN input dims from config: latent_dim={self.gen_input_latent_dim} (seq_len for latent={self.gen_input_seq_len}), conditional_dim={self.conditional_dim_for_generator}, context_dim={self.context_dim_for_generator}")

            # This list will store the Keras names of the generator inputs in their defined order.
            # It is CRUCIAL for preparing inputs for generator.predict() in the train loop.
            self.generator_actual_input_names_ordered = [inp.name for inp in self.generator.inputs]
            self.logger.info(f"Generator Keras input layer names (in order): {self.generator_actual_input_names_ordered}")

            # Now, try to associate these Keras inputs with their roles (latent, conditional, context)
            # to set the dimensions for _build_gan. We use the feeder_key_names for this association.
            # The _build_gan method will then create GAN inputs named based on these roles (e.g., 'gan_input_latent').

            found_latent = False
            found_conditional = False
            found_context = False

            for i, keras_input_tensor in enumerate(self.generator.inputs):
                input_name_from_keras = keras_input_tensor.name # e.g., "input_1" or "latent_input_layer"
                input_shape_from_keras = keras_input_tensor.shape # e.g. TensorShape([None, 100]) or TensorShape([None, 10, 1])
                self.logger.info(f"Processing generator input {i}: Keras name '{input_name_from_keras}', Keras shape {input_shape_from_keras}")

                # Try to match with hint names to assign a role and log more descriptively
                if self.gan_latent_input_keras_name_hint == input_name_from_keras:
                    input_role = "latent"
                    found_latent = True
                elif self.gan_conditional_input_keras_name_hint == input_name_from_keras:
                    input_role = "conditional"
                    found_conditional = True
                elif self.gan_context_input_keras_name_hint == input_name_from_keras:
                    input_role = "context"
                    found_context = True
                else:
                    input_role = "unknown"

                # Create the Keras Input layer for the GAN model
                # Name it distinctively for the GAN graph, possibly incorporating role and original name
                gan_input_layer_name = f"gan_input_{input_role}_{i}_{input_name_from_keras.replace(':', '_').replace('/', '_')}"
                keras_input_layer_for_gan = layers.Input(shape=input_shape_from_keras[1:], name=gan_input_layer_name)
                
                self.logger.info(f"  Created GAN Input Layer: Name='{keras_input_layer_for_gan.name}', Shape=(None, {input_shape_from_keras[1:]}), Role='{input_role}', From Gen Input='{input_name_from_keras}'")

                self.generator_actual_input_names_ordered.append(keras_input_layer_for_gan.name)
                # Add to GAN inputs for model definition
                gan_keras_inputs_for_model_definition.append(keras_input_layer_for_gan)

            # Fallback if roles couldn't be matched by name (e.g., generic Keras input names like "input_1")
            # This part is tricky and relies on assumptions about the order of inputs if names are not descriptive.
            # The current strategy is: if not found by name, stick to config values. 
            # A more robust fallback might try to infer based on typical dimensionalities if there's a standard order.
            if not found_latent:
                self.logger.warning(f"Could not identify LATENT input layer for generator by name matching feeder key '{self.feeder_key_name_latent}'. Will rely on config value for 'latent_dim' ({self.gen_input_latent_dim}) for GAN construction.")
            if not found_conditional:
                self.logger.warning(f"Could not identify CONDITIONAL input layer for generator by name matching feeder key '{self.feeder_key_name_conditional}'. Will rely on config value for 'conditional_dim' ({self.conditional_dim_for_generator}) for GAN construction.")
            if not found_context:
                self.logger.warning(f"Could not identify CONTEXT input layer for generator by name matching feeder key '{self.feeder_key_name_context}'. Will rely on config value for 'context_dim' ({self.context_dim_for_generator}) for GAN construction.")

            # Ensure essential dimensions are set for GAN building
            if self.gen_input_latent_dim is None:
                self.logger.error(f"Critical: gen_input_latent_dim is None after attempting to derive from model and config. This is required for GAN input. Check generator model input for latent vector and 'latent_dim' in config.")
                raise ValueError("Generator latent input dimension (gen_input_latent_dim) could not be determined.")
            # conditional_dim_for_generator and context_dim_for_generator can be None if not used by the GAN, _build_gan should handle this.
            if self.conditional_dim_for_generator is None:
                self.logger.info("conditional_dim_for_generator is None. GAN will be built without a dedicated conditional input if this was not derived from model or set in config.")
            if self.context_dim_for_generator is None:
                self.logger.info("context_dim_for_generator is None. GAN will be built without a dedicated context input if this was not derived from model or set in config.")

        else: # No generator model or no inputs in generator model
            self.logger.warning("Generator model or its inputs are not defined. GAN input dimensions will be based purely on config values.")
            self.gen_input_latent_dim = self.params.get('latent_dim')
            self.gen_input_seq_len = self.params.get('seq_len') # Or specific latent_seq_len if different
            self.conditional_dim_for_generator = self.params.get('conditional_dim')
            self.context_dim_for_generator = self.params.get('context_dim')
            self.generator_actual_input_names_ordered = [] # No generator inputs to list
            # Try to populate generator_actual_input_names_ordered from config if generator is missing but GAN structure is known
            # This is a fallback for cases where generator might be loaded later or is implicit.
            if self.feeder_key_name_latent: self.generator_actual_input_names_ordered.append(self.feeder_key_name_latent)
            if self.feeder_key_name_conditional: self.generator_actual_input_names_ordered.append(self.feeder_key_name_conditional)
            if self.feeder_key_name_context: self.generator_actual_input_names_ordered.append(self.feeder_key_name_context)
            self.logger.info(f"Using config for GAN input dims: latent_dim={self.gen_input_latent_dim} (seq_len for latent={self.gen_input_seq_len}), conditional_dim={self.conditional_dim_for_generator}, context_dim={self.context_dim_for_generator}")
            self.logger.info(f"Generator input names (from config keys as fallback): {self.generator_actual_input_names_ordered}")

        if not self.generator_actual_input_names_ordered and self.generator and self.generator.inputs:
            self.logger.error("Critical: self.generator_actual_input_names_ordered is empty even though generator has inputs. This should not happen.")
            # This implies an issue with the logic above that populates it from self.generator.inputs

        # --- Discriminator Input Parameters ---
        # The discriminator input shape is determined by generator_output_actual_seq_len and num_tis
        self.discriminator_input_seq_len = self.generator_output_actual_seq_len
        self.num_features_for_discriminator = self.params.get('num_features_for_discriminator', 1) # Base features (e.g. 'close')
        self.num_tis = self.params.get('num_tis', 20) # Number of TI features
        # Total features for discriminator = base features + TI features
        self.discriminator_input_feature_dim = self.num_features_for_discriminator + self.num_tis
        self.logger.info(f"Discriminator input parameters: seq_len={self.discriminator_input_seq_len}, feature_dim (base+TIs)={self.discriminator_input_feature_dim} (base={self.num_features_for_discriminator}, TIs={self.num_tis})")

        # Initialize feature names for discriminator (can be overridden by TI calculation)
        self.discriminator_feature_names = [f'feature_{i}' for i in range(self.discriminator_input_feature_dim)]

        self.logger.info("Core parameter initialization complete.")

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

        # Attempt to build/rebuild models if the generator is available
        if self.generator:
            self.logger.info("GANTrainerPlugin (set_params): Generator is available. Building/rebuilding Discriminator and GAN models.")
            try:
                # Ensure core parameters are up-to-date before building
                # self._initialize_core_parameters_from_config() # Already called earlier in set_params

                self.logger.info(f"GANTrainerPlugin (set_params): About to build Discriminator. Current relevant params: self.seq_len={self.seq_len}, self.num_features_for_discriminator={self.num_features_for_discriminator}")
                self.discriminator = self._build_discriminator()
                
                if self.discriminator: # Check if discriminator was built successfully
                    self.logger.info(f"GANTrainerPlugin (set_params): Discriminator built. About to build GAN. Current relevant params for GAN inputs: latent_dim={self.gen_input_latent_dim}, conditional_dim={self.conditional_dim_for_generator}, context_dim={self.context_dim_for_generator}")
                    self.gan_model = self._build_gan() # _build_gan also checks for self.discriminator
                    if self.gan_model:
                        self.logger.info("GANTrainerPlugin (set_params): Discriminator and GAN models built/rebuilt successfully.")
                    else:
                        self.logger.error("GANTrainerPlugin (set_params): GAN model building failed after discriminator was built.")
                else:
                    self.logger.error("GANTrainerPlugin (set_params): Discriminator model building failed. GAN model will not be built.")
            except Exception as e:
                self.logger.error(f"GANTrainerPlugin (set_params): Error during model building: {e}", exc_info=True)
        else:
            self.logger.warning("GANTrainerPlugin (set_params): Generator is not available. Discriminator and GAN models cannot be built/rebuilt at this time.")
            # Ensure models are None if generator is not available
            self.discriminator = None
            self.gan_model = None
            self.logger.info("GANTrainerPlugin (set_params): Discriminator and GAN models set to None as generator is unavailable.")

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
        if not self.generator:
            self.logger.error("GAN Build: Generator model (self.generator) is not initialized. Cannot build GAN.")
            raise ValueError("Generator model is not initialized.")
        if not self.discriminator:
            self.logger.error("GAN Build: Discriminator model (self.discriminator) is not initialized. Cannot build GAN.")
            raise ValueError("Discriminator model is not initialized.")
            
        self.logger.info("Building GAN model (Generator + Discriminator) with new input handling logic...")
        self.discriminator.trainable = False # Freeze discriminator during GAN training phase
        
        gan_keras_inputs_for_model_definition = []  # List of tf.keras.Input layers for the GAN model's definition
        generator_call_inputs_ordered = [] # List of tensors (from gan_keras_inputs_for_model_definition) to be passed to self.generator()

        if not self.generator.inputs: # Generator Keras model has no .inputs (e.g., it samples noise internally)
            self.logger.info("GAN Build: Generator model has no defined Keras inputs. GAN will call generator without explicit inputs.")
            # gan_keras_inputs_for_model_definition and generator_call_inputs_ordered will remain empty.
        else: # Generator has defined Keras inputs
            if not self.generator_actual_input_names_ordered:
                self.logger.error("GAN Build: Generator has Keras inputs, but self.generator_actual_input_names_ordered is empty. "
                                  "This indicates a failure in _initialize_core_parameters_from_config. Critical error.")
                raise ValueError("generator_actual_input_names_ordered is not populated despite generator having inputs.")

            self.logger.info(f"GAN Build: Preparing GAN inputs based on generator's {len(self.generator_actual_input_names_ordered)} actual Keras input layers: {self.generator_actual_input_names_ordered}")
            
            # Create GAN input layers based on the generator's *actual* Keras inputs, in their defined order.
            for i, actual_gen_keras_input_name in enumerate(self.generator_actual_input_names_ordered):
                # Find the corresponding Keras Tensor object from the generator's inputs
                original_gen_input_tensor = next((inp for inp in self.generator.inputs if inp.name == actual_gen_keras_input_name), None)

                if original_gen_input_tensor is None:
                    self.logger.error(f"GAN Build: Could not find Keras input tensor named '{actual_gen_keras_input_name}' in self.generator.inputs, "
                                      f"even though it was listed in self.generator_actual_input_names_ordered. Inconsistency detected.")
                    raise ValueError(f"Inconsistent generator input specification for '{actual_gen_keras_input_name}'.")

                original_shape_tuple = tuple(original_gen_input_tensor.shape[1:]) # Shape without batch dim (e.g., (latent_dim,) or (seq_len, features))
                
                # Determine a descriptive role for logging and naming, using the hints
                input_role_for_naming = "unknown"
                if self.gan_latent_input_keras_name_hint and self.gan_latent_input_keras_name_hint == actual_gen_keras_input_name:
                    input_role_for_naming = "latent"
                elif self.gan_conditional_input_keras_name_hint and self.gan_conditional_input_keras_name_hint == actual_gen_keras_input_name:
                    input_role_for_naming = "conditional"
                elif self.gan_context_input_keras_name_hint and self.gan_context_input_keras_name_hint == actual_gen_keras_input_name:
                    input_role_for_naming = "context"
                
                # Sanitize original Keras name for use in GAN input layer name
                sanitized_gen_input_name = actual_gen_keras_input_name.replace(':', '_').replace('/', '_')
                
                # Create the Keras Input layer for the GAN model
                # This input layer will be part of `gan_model.inputs`
                gan_input_layer = layers.Input(shape=original_shape_tuple, name=f"gan_input_{input_role_for_naming}_{i}_{sanitized_gen_input_name}")
                
                self.logger.info(f"GAN Build: Created GAN Input Layer: Name='{gan_input_layer.name}', Shape=(None, {original_shape_tuple}), Role='{input_role_for_naming}', From Gen Input='{actual_gen_keras_input_name}'")

                gan_keras_inputs_for_model_definition.append(gan_input_layer)
                generator_call_inputs_ordered.append(gan_input_layer) # This tensor will be passed to the generator

        # Verify the number of inputs prepared matches what the generator expects
        expected_num_gen_inputs = len(self.generator.inputs) if self.generator.inputs else 0
        if len(generator_call_inputs_ordered) != expected_num_gen_inputs:
            self.logger.error(
                f"GAN Build: MISMATCH in number of inputs prepared for generator call ({len(generator_call_inputs_ordered)}) "
                f"and number of inputs generator model expects ({expected_num_gen_inputs}). "
                f"GAN Inputs prepared: {[inp.name for inp in generator_call_inputs_ordered]}. "
                f"Generator Keras inputs: {[inp.name for inp in self.generator.inputs] if self.generator.inputs else 'None'}."
            )
            raise ValueError("Mismatch between prepared GAN inputs and generator's expected inputs.")

        # Call the generator with the (ordered) list of input tensors
        if not self.generator.inputs: # Generator expects no inputs
            self.logger.info("GAN Build: Calling generator() with no arguments as it has no defined Keras inputs.")
            generated_data_raw = self.generator()
        else: # Generator expects inputs
            self.logger.info(f"GAN Build: Calling generator with {len(generator_call_inputs_ordered)} input(s): {[inp.name for inp in generator_call_inputs_ordered]}.")
            generated_data_raw = self.generator(generator_call_inputs_ordered) # Pass the list of tensors
        
        self.logger.info(f"GAN Build: Generator output tensor (raw) received. Shape: {generated_data_raw.shape}, Dtype: {generated_data_raw.dtype}")

        # --- Slicing and Reshaping Generator Output ---
        generated_base_features_tensor = generated_data_raw
        static_gen_output_shape = generated_data_raw.shape 
        rank = len(static_gen_output_shape)

        if rank == 3: # (batch, seq, features)
            actual_gen_seq_len_from_shape = static_gen_output_shape[1]
            actual_gen_features_from_shape = static_gen_output_shape[2]
            
            self.logger.info(f"GAN Build (3D Gen Output): Static shape: {static_gen_output_shape}. Target base features: {self.num_base_features}, Target seq_len for D: {self.seq_len}")

            if actual_gen_features_from_shape is not None and self.num_base_features is not None and actual_gen_features_from_shape > self.num_base_features:
                self.logger.info(f"GAN Build (3D output): Slicing generator output features from {actual_gen_features_from_shape} down to {self.num_base_features}.")
                generated_base_features_tensor = generated_base_features_tensor[:, :, :self.num_base_features]
            elif actual_gen_features_from_shape is not None and self.num_base_features is not None and actual_gen_features_from_shape < self.num_base_features:
                self.logger.error(f"GAN Build (3D output): Generator output features ({actual_gen_features_from_shape}) is LESS than num_base_features ({self.num_base_features}). Critical config error.")
                raise ValueError("Generator outputs fewer features than configured num_base_features.")

            if actual_gen_seq_len_from_shape is not None and self.seq_len is not None and actual_gen_seq_len_from_shape != self.seq_len:
                self.logger.warning(f"GAN Build (3D output): Generator output sequence length ({actual_gen_seq_len_from_shape}) "
                                    f"differs from target discriminator sequence length ({self.seq_len}). "
                                    "This might indicate an issue if not handled. Discriminator expects self.seq_len.")
        elif rank == 2: # (batch, features)
            actual_gen_features_from_shape = static_gen_output_shape[1]
            self.logger.info(f"GAN Build (2D Gen Output): Static shape: {static_gen_output_shape}. Target base features: {self.num_base_features}, Target seq_len for D: {self.seq_len}")

            if actual_gen_features_from_shape is not None and self.num_base_features is not None and actual_gen_features_from_shape > self.num_base_features:
                self.logger.info(f"GAN Build (2D output): Slicing generator output features from {actual_gen_features_from_shape} down to {self.num_base_features}.")
                generated_base_features_tensor = generated_base_features_tensor[:, :self.num_base_features]
            elif actual_gen_features_from_shape is not None and self.num_base_features is not None and actual_gen_features_from_shape < self.num_base_features:
                self.logger.error(f"GAN Build (2D output): Generator output features ({actual_gen_features_from_shape}) is LESS than num_base_features ({self.num_base_features}). Critical error.")
                raise ValueError("Generator outputs fewer features than configured num_base_features for 2D output.")

            if self.seq_len is not None: 
                if self.seq_len == 1: # Discriminator expects a sequence of length 1
                    self.logger.info(f"GAN Build (2D output): Generator output was 2D. Expanding dims to make it 3D (batch, 1, features) as D expects seq_len=1. Shape before: {generated_base_features_tensor.shape}")
                    generated_base_features_tensor = tf.expand_dims(generated_base_features_tensor, axis=1)
                    self.logger.info(f"GAN Build: Shape after expand_dims: {generated_base_features_tensor.shape}")
                else: # Discriminator expects seq_len > 1, but generator output is 2D
                     self.logger.error(f"GAN Build: Generator output is 2D, but discriminator expects seq_len {self.seq_len}. Cannot reconcile. Critical error.")
                     raise ValueError(f"Generator output 2D, but discriminator expects sequence length {self.seq_len}.")
            else: # self.seq_len is None (highly unlikely for D)
                 self.logger.warning("GAN Build: Generator output is 2D, and target seq_len for D is None. This is highly unusual for sequence discriminators.")
        
        else: # Unexpected rank
            self.logger.error(f"GAN Build: Generator output has unexpected rank {rank} (shape {static_gen_output_shape}). Expected 2D or 3D. Cannot proceed.")
            raise ValueError(f"Generator output has unexpected rank {rank}, shape {static_gen_output_shape}.")
        
        self.logger.info(f"GAN Build: Processed generator output for base features (generated_base_features_tensor) shape: {generated_base_features_tensor.shape}")
        
        data_for_discriminator_input = generated_base_features_tensor

        # --- Apply Technical Indicators using TensorFlowTALayer if configured ---
        if self.params.get("use_tensorflow_ta_layer", False) and self.ti_names_to_calculate:
            self.logger.info(f"GAN Build: Using TensorFlowTALayer to calculate TIs for generated data. Num TIs: {len(self.ti_names_to_calculate)}")
            self.logger.info(f"  TensorFlowTALayer input (generated_base_features_tensor) shape: {data_for_discriminator_input.shape}") # Log shape before TA layer
            self.logger.info(f"  TensorFlowTALayer params: base_feature_names={self.base_feature_names_ordered}, ti_names={self.ti_names_to_calculate}, "
                             f"num_base_features={self.num_base_features}, num_total_features={self.num_features_for_discriminator}, seq_len={self.seq_len}")
            
            input_to_ta_layer_shape = data_for_discriminator_input.shape # Check shape of tensor being passed
            if len(input_to_ta_layer_shape) != 3:
                self.logger.error(f"GAN Build: Input to TensorFlowTALayer is not 3D (shape: {input_to_ta_layer_shape}). This is required.")
                raise ValueError(f"TensorFlowTALayer requires 3D input, got {input_to_ta_layer_shape}.")
            if input_to_ta_layer_shape[1] is not None and self.seq_len is not None and input_to_ta_layer_shape[1] != self.seq_len:
                self.logger.error(f"GAN Build: Seq length of input to TensorFlowTALayer ({input_to_ta_layer_shape[1]}) "
                                  f"does not match configured seq_len for layer ({self.seq_len}).")
                raise ValueError(f"Sequence length mismatch for TensorFlowTALayer: input has {input_to_ta_layer_shape[1]}, layer expects {self.seq_len}.")
            if input_to_ta_layer_shape[2] is not None and self.num_base_features is not None and input_to_ta_layer_shape[2] != self.num_base_features:
                self.logger.error(f"GAN Build: Feature count of input to TensorFlowTALayer ({input_to_ta_layer_shape[2]}) "
                                  f"does not match configured num_base_features for layer ({self.num_base_features}).")
                raise ValueError(f"Feature count mismatch for TensorFlowTALayer: input has {input_to_ta_layer_shape[2]}, layer expects {self.num_base_features}.")

            tf_ta_layer_instance_for_gan = TensorFlowTALayer(
                base_feature_names=self.base_feature_names_ordered,
                ti_names_to_calculate=self.ti_names_to_calculate,
                num_base_features=self.num_base_features, 
                num_total_features=self.num_features_for_discriminator, # D expects this total
                seq_len=self.seq_len, 
                name="tf_ta_layer_in_gan" 
            )
            data_for_discriminator_input = tf_ta_layer_instance_for_gan(data_for_discriminator_input)
            self.logger.info(f"GAN Build: Output of TensorFlowTALayer (data_for_discriminator_input) shape: {data_for_discriminator_input.shape}")
        else: # Not using TF TA Layer, or no TIs to calculate
            self.logger.info(f"GAN Build: Not using TensorFlowTALayer, or no TIs to calculate. Passing processed generator output directly to discriminator.")
            if self.num_base_features is not None and self.num_features_for_discriminator is not None and self.num_features_for_discriminator != self.num_base_features:
                 self.logger.error(f"GAN Build: TIs are skipped, but num_features_for_discriminator ({self.num_features_for_discriminator}) "
                                   f"differs from num_base_features ({self.num_base_features}). "
                                   f"Discriminator expects {self.num_features_for_discriminator} features, but will receive {self.num_base_features}. Shape mismatch imminent.")
                 raise ValueError("Discriminator feature count mismatch: TIs expected by D but not added in GAN path, and num_features_for_discriminator != num_base_features.")

        final_disc_input_shape = data_for_discriminator_input.shape
        self.logger.info(f"GAN Build: Data shape entering discriminator: {final_disc_input_shape}. Discriminator built for: (None, {self.seq_len}, {self.num_features_for_discriminator})")

        if len(final_disc_input_shape) != 3:
            self.logger.error(f"GAN Build: CRITICAL SHAPE MISMATCH. Data for discriminator is not 3D. Shape: {final_disc_input_shape}.")
            raise ValueError(f"Data for discriminator must be 3D, got shape {final_disc_input_shape}")
        
        if final_disc_input_shape[1] is not None and self.seq_len is not None and final_disc_input_shape[1] != self.seq_len:
            self.logger.error(
                f"GAN Build: CRITICAL SHAPE MISMATCH for Discriminator Input (Sequence Length). "
                f"Data seq_len: {final_disc_input_shape[1]}, Discriminator expects: {self.seq_len}."
            )
            raise ValueError(f"Sequence length mismatch for data entering discriminator in GAN graph: data has {final_disc_input_shape[1]}, D expects {self.seq_len}.")
        
        if final_disc_input_shape[2] is not None and self.num_features_for_discriminator is not None and final_disc_input_shape[2] != self.num_features_for_discriminator:
            self.logger.error(
                f"GAN Build: CRITICAL SHAPE MISMATCH for Discriminator Input (Number of Features). "
                f"Data features: {final_disc_input_shape[2]}, Discriminator expects: {self.num_features_for_discriminator}."
            )
            raise ValueError(f"Feature count mismatch for data entering discriminator in GAN graph: data has {final_disc_input_shape[2]}, D expects {self.num_features_for_discriminator}.")

        discriminator_output_on_gan = self.discriminator(data_for_discriminator_input)
        self.logger.info(f"GAN Build: Discriminator output tensor (on GAN): {discriminator_output_on_gan}")

        # --- Create GAN Model ---
        # gan_keras_inputs_for_model_definition should be an empty list if generator takes no Keras inputs.
        if not gan_keras_inputs_for_model_definition and (self.generator.inputs and len(self.generator.inputs) > 0) : # Should have been caught by earlier check on expected_num_gen_inputs
            self.logger.error("GAN Build: gan_keras_inputs_for_model_definition is empty, but generator has Keras inputs. Cannot define GAN model inputs.")
            raise ValueError("GAN model input list is empty despite generator needing inputs.")

        gan_model = Model(inputs=gan_keras_inputs_for_model_definition, outputs=discriminator_output_on_gan, name="GAN_Model_Refactored")
        self.logger.info("GAN model (refactored) created successfully.")
        gan_model.summary(print_fn=self.logger.info)

        # Plot GAN model
        try:
            plot_dir = os.path.join(self.params["results_base_dir"], self.params["save_plot_dir"])
            os.makedirs(plot_dir, exist_ok=True)
            plot_file_name = self.params.get("gan_model_plot_file", "gan_architecture.png") 
            plot_path = os.path.join(plot_dir, plot_file_name)
            dpi = self.params.get("model_plot_dpi", 300)
            plot_model(gan_model, to_file=plot_path, show_shapes=True, dpi=dpi, expand_nested=True, show_layer_activations=True)
            self.logger.info(f"GAN architecture plot (refactored) saved to {plot_path}")
        except Exception as e:
            self.logger.warning(f"Could not plot GAN model (refactored): {e}. Ensure pydot and graphviz are installed if not already.")

        return gan_model

    def _calculate_technical_indicators(self, df: pd.DataFrame,
```
