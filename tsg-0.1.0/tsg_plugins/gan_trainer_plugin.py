# optimizer/plugins/gan_plugin.py

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, GRU, Bidirectional, Concatenate, LayerNormalization, Dropout, LeakyReLU, Reshape, Flatten, Activation, Add, Multiply, Attention, Embedding, Conv1D, MaxPooling1D, GlobalAveragePooling1D, TimeDistributed, BatchNormalization, SpatialDropout1D, RepeatVector
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.metrics import Precision, Recall, AUC, MeanSquaredError, MeanAbsoluteError
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard, CSVLogger
from tensorflow.keras.utils import plot_model
from tensorflow.keras.regularizers import l1, l2, l1_l2
import tensorflow.keras.backend as K

import numpy as np # ADDED for _get_data_generator
import pandas as pd # ADDED for _get_data_generator and datetime handling
import os
import json
import logging # Ensure logging is imported at the very top
import time # Add time import
import matplotlib.pyplot as plt # Add matplotlib import
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

# ADDED: Ensure numpy is imported if not already globally for the file
import numpy as np
# ADDED: Ensure logging is imported if not already, or use self.logger if setup in __init__
import logging # Assuming logger is configured per class instance as self.logger

from copy import deepcopy

# Initialize logger for this module
logger = logging.getLogger(__name__) # Ensure logger is defined before classes using it
# logger.setLevel(logging.INFO) # Set default log level - Assuming this is set elsewhere or defaults appropriately

# ADD INFO MESSAGE ABOUT GENERATOR WARNING
# logger.info("GANTrainerPlugin: The VAE generator is intended to be frozen during GAN training. A \\\\\\\\\'UserWarning: The model does not have any trainable weights.\\\\\\\\\' may appear when generator.predict() is called; this is expected for the frozen generator and does not affect discriminator or GAN training.")


# Custom Keras Layer for Technical Indicator Calculation using tf.numpy_function
class TensorFlowTALayer(keras.layers.Layer):
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
        # output_tensor.set_shape([None, self.seq_len, self.num_total_features]) # Static shape for batch, seq_len, features # Temporarily comment out if causing issues with symbolic tensor
        # self.logger.info(f"TensorFlowTALayer call: Set output_tensor static shape to (None, {self.seq_len}, {self.num_total_features}). Actual symbolic shape after set_shape: {output_tensor.shape}")

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
    plugin_name_prefix = "gan_trainer"
    plugin_params: Dict[str, Any] = {
        "gan_epochs": 10000,
        "gan_batch_size": 32,
        "generator_lr": 1e-4,
        "generator_beta1": 0.5,
        "discriminator_lr": 1e-4,
        "discriminator_beta1": 0.5,
        "gan_save_interval": 500,
        # "latent_dim": 32, # This will be derived or taken from generator config
        # "seq_len": 18, # This will be derived from generator output
        "discriminator_conv_filters": [64, 128],
        "discriminator_conv_kernel_size": 3,
        "discriminator_lstm_units": 64,
        "discriminator_dropout_rate": 0.3,
        # "gan_generator_output_actual_seq_len": 1, # Derived from generator
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

        "results_base_dir": "examples/results/phase_4_3",
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

        # Configuration for features fed into the discriminator's main architecture
        # If TensorFlowTALayer is used, these are the base features for TA calculation (e.g. OHLC)
        # If no TAlayer, these are all features the discriminator ConvNet sees.
        "discriminator_input_feature_names": ["OPEN", "HIGH", "LOW", "CLOSE"], # Example from normalized_d4.csv
        
        # Configuration for conditional inputs to the Generator
        "feeder_date_feature_names_for_conditioning": ["day_of_month", "hour_of_day", "day_of_week"], # From normalized_d4.csv
        "feeder_max_day_of_month": 31.0,
        "feeder_max_hour_of_day": 24.0, # Assumes 0-23
        "feeder_max_day_of_week": 7.0,  # Assumes 0-6
        # Add other feeder_max_... if other date features like day_of_year, minute_of_hour are used.

        "conditional_fundamental_feature_names": ["S&P500_Close", "vix_close"], # From normalized_d4.csv
        "num_conditional_prev_tick_features": 5, # Number of features from the start of the real sequence to use as conditional input
                                                 # Total conditional dim = len(date_feats) + len(fund_feats) + num_prev_tick_feats
                                                 # e.g. 3 + 2 + 5 = 10
        "datetime_col_name_in_x_real_df": "DATE_TIME" # Column name for datetimes in input x_real_df
    }

    def __init__(self, config: Dict[str, Any], generator_plugin_instance: Optional[Any] = None, feeder_plugin_instance: Optional[Any] = None, preprocessor_plugin_instance: Optional[Any] = None):
        self.ti_names_for_discriminator = [] # Initialize this attribute first
        # Initialize self.params by copying class-level plugin_params and then updating with config
        self.params = deepcopy(self.plugin_params) # Initialize self.params with defaults
        if config:
            self.params.update(config) # Update with instance-specific config

        self.config = deepcopy(config) # Keep a copy of the original config if needed for other purposes
        self.generator_plugin = generator_plugin_instance
        self.feeder_plugin = feeder_plugin_instance
        self.preprocessor_plugin_instance = preprocessor_plugin_instance

        # Initialize logger for this instance
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        # Configure logger if not already configured by a higher-level setup
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO) # Default level

        self.logger.info("GANTrainerPlugin: Initializing...")
        
        # Update self.params with any relevant items from the main config passed to __init__
        # This ensures that parameters set in the main config override plugin defaults if necessary.
        # The previous self.params.update(config) should handle this, but let's be explicit if needed.
        # For example, if 'latent_dim' or 'seq_len' are in the main config and should directly set plugin params:
        # relevant_config_keys_for_params = ['latent_dim', 'seq_len', 'gan_epochs', 'gan_batch_size'] # etc.
        # for key in relevant_config_keys_for_params:
        #     if key in self.config:
        #         self.params[key] = self.config[key]

        # Core model components will be built in set_params or called from there
        self.generator: Optional[Model] = None 
        self.discriminator: Optional[Model] = None
        self.gan_model: Optional[Model] = None

        # Parameters derived from generator and config
        self.seq_len: int = self._get_config_param("seq_len", 18) # Discriminator's sequence length
        self.latent_dim: int = self._get_config_param("latent_dim", 32)
        self.num_base_features: int = 0 # Features from generator (e.g., OHLC)
        self.num_conditional_features: int = 0 # e.g. date/time, external conditions
        self.num_context_features: int = 0 # e.g. VAE context vector h
        self.num_features_for_discriminator: int = 0 # Total features discriminator sees (base + TIs if applicable)
        
        self.generator_actual_input_names_ordered: List[str] = []
        self.gan_latent_input_keras_name_hint: Optional[str] = None
        self.gan_conditional_input_keras_name_hint: Optional[str] = None
        self.gan_context_input_keras_name_hint: Optional[str] = None
        self.gan_window_input_keras_name_hint: Optional[str] = None # ADDED
        self.gan_feeder_input_keras_name_hints: Dict[str, str] = {} # Ensure initialized as a dictionary

        # Paths for saving results - these will be fully resolved in _initialize_core_parameters_from_config
        self.results_base_dir: str = "" 
        self.models_dir: str = ""
        self.plots_dir: str = ""
        self.metrics_dir: str = ""
        self.generator_model_plot_file: str = ""
        self.discriminator_model_plot_file: str = ""
        self.gan_model_plot_file: str = ""
        self.model_plot_dpi: int = self._get_config_param("model_plot_dpi", 300)
        
        self.logger.info("GANTrainerPlugin: Initialization basic params complete.")
        # Full parameter setup and model building is deferred to set_params

    def _get_config_param(self, key: str, default: Any = None) -> Any:
        """
        Helper method to get a parameter from self.params (primary) or self.config (fallback).
        Args:
            key: Parameter key as string.
            default: Default value to return if the key is not found.
        Returns:
            Value of the parameter from self.params or self.config, or the default value.
        """
        # Prioritize self.params as it should contain the merged and processed parameters
        if key in self.params:
            return self.params[key]
        # Fallback to self.config if needed, though ideally self.params is comprehensive
        # self.logger.debug(f"_get_config_param: Key '{key}' not in self.params, checking self.config.")
        return self.config.get(key, default)

    def _initialize_core_parameters_from_config(self):
        """Initializes core GAN, Generator, and Discriminator parameters.
        Derives dimensions and input names from Keras models where possible,
        falling back to config values (from self.params) if necessary.
        """
        self.logger.info("Starting _initialize_core_parameters_from_config...")

        # --- Ensure Feeder Key Names are available (expected to be set in self.params) ---
        if 'feeder_key_name_noise' not in self.params:
            self.logger.warning("Parameter 'feeder_key_name_noise' not in self.params. Defaulting to 'latent'.")
            self.params['feeder_key_name_noise'] = "latent"
        self.feeder_key_name_noise = self.params['feeder_key_name_noise']
        
        if 'feeder_key_name_conditional' not in self.params:
            self.logger.warning("Parameter 'feeder_key_name_conditional' not in self.params. Defaulting to 'conditional'.")
            self.params['feeder_key_name_conditional'] = "conditional"
        self.feeder_key_name_conditional = self.params['feeder_key_name_conditional']

        if 'feeder_key_name_context' not in self.params:
            self.logger.warning("Parameter 'feeder_key_name_context' not in self.params. Defaulting to 'context'.")
            self.params['feeder_key_name_context'] = "context"
        self.feeder_key_name_context = self.params['feeder_key_name_context']

        # --- Generator Output Parameters (used by Discriminator construction) ---
        self.seq_len = None
        self.num_base_features = None

        # 1. Try to get dimensions from the generator model itself
        if self.generator:
            if hasattr(self.generator, 'output_shape') and self.generator.output_shape:
                self.logger.info(f"Attempting to derive dimensions from generator.output_shape: {self.generator.output_shape}")
                output_shape = self.generator.output_shape
                if isinstance(output_shape, tuple) and len(output_shape) >= 2:
                    if len(output_shape) == 3:  # (batch, seq_len, features)
                        self.seq_len = output_shape[1]
                        self.num_base_features = output_shape[2]
                        self.logger.info(f"  Derived from 3D generator.output_shape: seq_len={self.seq_len}, num_base_features={self.num_base_features}")
                    elif len(output_shape) == 2:  # (batch, features)
                        self.num_base_features = output_shape[1]
                        self.logger.info(f"  Derived from 2D generator.output_shape: num_base_features={self.num_base_features}. Seq_len will be sought from config for D.")
                        # Try to get seq_len from config for this case
                        self.seq_len = self.params.get('gan_generator_output_actual_seq_len', self.params.get('seq_len'))
                        if self.seq_len:
                            self.logger.info(f"  seq_len for 2D output case found in config: {self.seq_len}")
                        else:
                            self.logger.warning("  seq_len for 2D output case not found in config. It might be assumed as 1 or needs explicit setting if features are per step.")
                    else:
                        self.logger.warning(f"  Generator output_shape {output_shape} has an unexpected number of dimensions.")
                else:
                    self.logger.warning(f"  Generator output_shape {output_shape} is not a tuple or has too few dimensions.")
            elif hasattr(self.generator, 'output') and hasattr(self.generator.output, 'shape'):
                self.logger.info(f"Attempting to derive dimensions from generator.output.shape: {self.generator.output.shape}")
                output_shape = self.generator.output.shape
                if len(output_shape) == 3:  # (None, seq_len, features)
                    self.seq_len = output_shape[1].value if hasattr(output_shape[1], 'value') else output_shape[1]
                    self.num_base_features = output_shape[2].value if hasattr(output_shape[2], 'value') else output_shape[2]
                    self.logger.info(f"  Derived from 3D generator.output.shape: seq_len={self.seq_len}, num_base_features={self.num_base_features}")
                elif len(output_shape) == 2:  # (None, features)
                    self.num_base_features = output_shape[1].value if hasattr(output_shape[1], 'value') else output_shape[1]
                    self.logger.info(f"  Derived from 2D generator.output.shape: num_base_features={self.num_base_features}. Seq_len will be sought from config for D.")
                    self.seq_len = self.params.get('gan_generator_output_actual_seq_len', self.params.get('seq_len'))
                    if self.seq_len:
                        self.logger.info(f"  seq_len for 2D output case found in config: {self.seq_len}")
                    else:
                        self.logger.warning("  seq_len for 2D output case not found in config.")
                else:
                    self.logger.warning(f"  Generator output.shape {output_shape} has an unexpected number of dimensions.")
            else:
                self.logger.info("Generator model does not have 'output_shape' or 'output.shape'. Will rely on config parameters.")
        else:
            self.logger.info("Generator model not available at this point. Will rely on config parameters for output dimensions.")

        # 2. Fallback to config for seq_len
        if self.seq_len is None:
            self.seq_len = self.params.get('gan_generator_output_actual_seq_len')
            self.logger.info(f"Attempting to get 'gan_generator_output_actual_seq_len' from params: {self.seq_len}")
            if self.seq_len is None:
                self.seq_len = self.params.get('seq_len')
                self.logger.info(f"Attempting to get 'seq_len' from params: {self.seq_len}")

        # 3. Fallback to config for num_base_features
        if self.num_base_features is None:
            self.num_base_features = self.params.get('gan_generator_output_actual_features')
            self.logger.info(f"Attempting to get 'gan_generator_output_actual_features' from params: {self.num_base_features}")
            if self.num_base_features is None:
                self.num_base_features = self.params.get('num_base_features_generated')
                self.logger.info(f"Attempting to get 'num_base_features_generated' from params: {self.num_base_features}")
                if self.num_base_features is None:
                    self.num_base_features = self.params.get('feature_dim')
                    self.logger.info(f"Attempting to get 'feature_dim' from params: {self.num_base_features}")

        # 4. Fallback for num_base_features using generator_plugin_instance
        if self.num_base_features is None and hasattr(self, 'generator_plugin_instance') and self.generator_plugin_instance:
            self.logger.info("Attempting to derive num_base_features from generator_plugin_instance.params['decoder_output_feature_names']")
            gen_plugin_decoder_outputs = self.generator_plugin_instance.params.get('decoder_output_feature_names')
            if gen_plugin_decoder_outputs and isinstance(gen_plugin_decoder_outputs, list):
                self.num_base_features = len(gen_plugin_decoder_outputs)
                self.logger.info(f"  Derived num_base_features from len(decoder_output_feature_names): {self.num_base_features}")
            else:
                self.logger.warning("  generator_plugin_instance.params['decoder_output_feature_names'] is not a list or is empty.")

        if self.num_base_features is None:
            self.logger.error("GANTrainerPlugin: CRITICAL - num_base_features could not be determined. This is essential for building the discriminator.")
            # Potentially raise an error or set a default if appropriate, though error is safer.
            raise ValueError("Failed to determine num_base_features for the GAN.")


        # 5. Determine latent_dim for GAN input layer (used by _build_gan)
        # This is the dimension of the noise vector fed to the generator *within the GAN model*
        # It should match the generator's primary latent input dimension.
        if hasattr(self, 'generator_plugin_instance') and self.generator_plugin_instance:
            # Try to get it from generator_plugin's latent_shape if it's a simple list/tuple of one dim
            gen_plugin_latent_shape = self.generator_plugin_instance.params.get('latent_shape')
            if isinstance(gen_plugin_latent_shape, (list, tuple)) and len(gen_plugin_latent_shape) == 1:
                self.latent_dim_for_generator = gen_plugin_latent_shape[0]
                self.logger.info(f"Derived latent_dim_for_generator from generator_plugin.params['latent_shape']: {self.latent_dim_for_generator}")
            elif isinstance(gen_plugin_latent_shape, (list, tuple)) and len(gen_plugin_latent_shape) == 2: # e.g. (seq_len, features)
                 # This case might mean the generator expects a sequence as latent input.
                 # The GAN's noise input layer might still be flat, then reshaped.
                 # For now, let's assume the GAN's input layer will be flat, matching the product or a specific config.
                 # The self.latent_input_shape_for_gan_input_layer will handle the actual shape for GAN's Input layer.
                 # Let's prioritize params['latent_dim'] if available.
                pass


        if self.latent_dim_for_generator is None:
            self.latent_dim_for_generator = self.params.get('latent_dim') # Standard config name
            self.logger.info(f"Attempting to get 'latent_dim' from GANTrainerPlugin params: {self.latent_dim_for_generator}")

        if self.latent_dim_for_generator is None and self.generator:
            # If generator has a single input or a named 'latent_input'
            try:
                if isinstance(self.generator.input, list): # Multiple inputs
                    for inp in self.generator.inputs:
                        if 'latent' in inp.name.lower(): # Common naming convention
                            self.latent_dim_for_generator = inp.shape[-1]
                            self.logger.info(f"Derived latent_dim_for_generator from generator model input '{inp.name}': {self.latent_dim_for_generator}")
                            break
                    if self.latent_dim_for_generator is None: # Fallback if no named latent input found
                        # This case is tricky if multiple inputs don't have clear names.
                        # Defaulting to the first input's last dimension if it's a plausible latent dim.
                        # This is a heuristic.
                        if self.generator.inputs[0].shape[-1] and len(self.generator.inputs[0].shape) == 2:
                             self.latent_dim_for_generator = self.generator.inputs[0].shape[-1]
                             self.logger.warning(f"Heuristically derived latent_dim_for_generator from first generator input shape: {self.latent_dim_for_generator}")

                elif hasattr(self.generator.input, 'shape'): # Single input
                    if len(self.generator.input.shape) == 2: # (batch, latent_dim)
                        self.latent_dim_for_generator = self.generator.input.shape[-1]
                        self.logger.info(f"Derived latent_dim_for_generator from single generator input shape: {self.latent_dim_for_generator}")
                    # If generator input is already sequential, e.g. (batch, seq, feat_latent),
                    # latent_dim_for_generator might be feat_latent, or product for GAN input layer.
                    # This part is complex and depends on generator architecture.
                    # For now, we assume a flat latent vector for the GAN's noise input.
            except Exception as e:
                self.logger.warning(f"Could not derive latent_dim_for_generator from generator model inputs: {e}")


        if self.latent_dim_for_generator is None:
            self.logger.error("GANTrainerPlugin: CRITICAL - latent_dim_for_generator could not be determined. This is essential for building the GAN.")
            raise ValueError("Failed to determine latent_dim_for_generator for the GAN.")
        
        # Set the shape for the GAN's main latent noise input layer
        # This is distinct from generator's internal latent processing if it reshapes etc.
        # Typically, this is (latent_dim_for_generator,)
        self.latent_input_shape_for_gan_input_layer = (self.latent_dim_for_generator,)
        self.logger.info(f"Set latent_input_shape_for_gan_input_layer: {self.latent_input_shape_for_gan_input_layer}")


        # 6. Initialize conditional_dim and context_dim (from FeederPlugin)
        if hasattr(self, 'feeder_plugin_instance') and self.feeder_plugin_instance:
            try:
                self.conditional_dim = self.feeder_plugin_instance.get_conditional_dim()
                self.logger.info(f"Set conditional_dim from feeder_plugin_instance: {self.conditional_dim}")
            except Exception as e:
                self.logger.error(f"Failed to get conditional_dim from feeder_plugin_instance: {e}")
                self.conditional_dim = self.params.get('conditional_dim', 0) # Fallback
                self.logger.info(f"Using conditional_dim from GANTrainerPlugin params as fallback: {self.conditional_dim}")

            try:
                self.context_dim = self.feeder_plugin_instance.get_context_vector_dim()
                self.logger.info(f"Set context_dim from feeder_plugin_instance: {self.context_dim}")
            except Exception as e:
                self.logger.error(f"Failed to get context_dim from feeder_plugin_instance: {e}")
                self.context_dim = self.params.get('context_dim', 0) # Fallback
                self.logger.info(f"Using context_dim from GANTrainerPlugin params as fallback: {self.context_dim}")
        else:
            self.logger.warning("FeederPlugin instance not available. Falling back to GANTrainerPlugin params for conditional_dim and context_dim.")
            self.conditional_dim = self.params.get('conditional_dim', 0)
            self.context_dim = self.params.get('context_dim', 0)
            self.logger.info(f"Set conditional_dim from GANTrainerPlugin params: {self.conditional_dim}")
            self.logger.info(f"Set context_dim from GANTrainerPlugin params: {self.context_dim}")
            
        # Ensure they are at least 0
        self.conditional_dim = self.conditional_dim if self.conditional_dim is not None else 0
        self.context_dim = self.context_dim if self.context_dim is not None else 0

        self.logger.info(f"--- Final Core Parameters Initialized ---")
        self.logger.info(f"  seq_len: {self.seq_len}")
        self.logger.info(f"  num_base_features: {self.num_base_features}")
        self.logger.info(f"  latent_dim_for_generator (used for GAN input layer shape): {self.latent_dim_for_generator}")
        self.logger.info(f"  latent_input_shape_for_gan_input_layer: {self.latent_input_shape_for_gan_input_layer}")
        self.logger.info(f"  conditional_dim: {self.conditional_dim}")
        self.logger.info(f"  context_dim: {self.context_dim}")
        self.logger.info(f"--------------------------------------")

        if not all([self.seq_len is not None, self.num_base_features is not None, self.latent_dim_for_generator is not None]):
            missing_params_msg = "One or more critical parameters (seq_len, num_base_features, latent_dim_for_generator) could not be initialized."
            self.logger.error(missing_params_msg)
            raise ValueError(missing_params_msg)


    def _build_discriminator(self) -> Model:
        """
        Builds the Discriminator model.
        The discriminator takes real data (sequence of features) and optionally conditional/context data.
        It outputs a scalar probability indicating whether the input is real or fake.
        """
        if not all([hasattr(self, attr) and getattr(self, attr) is not None for attr in ['seq_len', 'num_base_features', 'conditional_dim', 'context_dim']]):
            self.logger.error("Discriminator cannot be built: one or more core parameters (seq_len, num_base_features, conditional_dim, context_dim) are not initialized.")
            raise ValueError("Core parameters for discriminator not initialized.")

        self.logger.info(f"Building Discriminator with: seq_len={self.seq_len}, num_base_features={self.num_base_features}, conditional_dim={self.conditional_dim}, context_dim={self.context_dim}")

        # Input for the main sequence data
        main_input = Input(shape=(self.seq_len, self.num_base_features), name='discriminator_main_input')
        d_inputs = [main_input]
        
        # Current processed features start with main_input
        processed_features = main_input

        # Conditional Input Handling
        if self.conditional_dim > 0:
            conditional_input = Input(shape=(self.conditional_dim,), name='discriminator_conditional_input')
            d_inputs.append(conditional_input)
            
            # Process conditional input: Dense layer then repeat to match sequence length
            # Projection dimension for conditional features, e.g., 16 or adapt based on num_base_features
            cond_projection_dim = self.params.get('d_conditional_projection_dim', 16)
            conditional_dense = Dense(cond_projection_dim, activation=self.params.get('d_activation', 'relu'), name='d_conditional_dense')(conditional_input)
            conditional_repeated = RepeatVector(self.seq_len, name='d_conditional_repeated')(conditional_dense)
            processed_features = Concatenate(axis=-1, name='d_concat_main_conditional')([processed_features, conditional_repeated])
            self.logger.info(f"Discriminator: Added conditional input processing. Merged shape: {processed_features.shape}")


        # Context Input Handling
        if self.context_dim > 0:
            context_input = Input(shape=(self.context_dim,), name='discriminator_context_input')
            d_inputs.append(context_input)

            # Process context input: Dense layer then repeat to match sequence length
            context_projection_dim = self.params.get('d_context_projection_dim', 16)
            context_dense = Dense(context_projection_dim, activation=self.params.get('d_activation', 'relu'), name='d_context_dense')(context_input)
            context_repeated = RepeatVector(self.seq_len, name='d_context_repeated')(context_dense)
            processed_features = Concatenate(axis=-1, name='d_concat_all_features')([processed_features, context_repeated])
            self.logger.info(f"Discriminator: Added context input processing. Merged shape: {processed_features.shape}")

        # Discriminator Core Architecture (example using Conv1D)
        # Parameters for Conv1D layers
        d_filters_1 = self.params.get('d_filters_1', 64)
        d_kernel_size_1 = self.params.get('d_kernel_size_1', 5)
        d_strides_1 = self.params.get('d_strides_1', 1) # Changed from 2 to 1 to preserve seq_len more
        d_dropout_1 = self.params.get('d_dropout_1', 0.3)
        
        d_filters_2 = self.params.get('d_filters_2', 128)
        d_kernel_size_2 = self.params.get('d_kernel_size_2', 5)
        d_strides_2 = self.params.get('d_strides_2', 2) # Can use strides for downsampling
        d_dropout_2 = self.params.get('d_dropout_2', 0.3)

        d_filters_3 = self.params.get('d_filters_3', 256) # Added another layer
        d_kernel_size_3 = self.params.get('d_kernel_size_3', 3)
        d_strides_3 = self.params.get('d_strides_3', 2)
        d_dropout_3 = self.params.get('d_dropout_3', 0.3)

        d_dense_units = self.params.get('d_dense_units', 256) # Dense layer before output
        d_activation_conv = self.params.get('d_activation_conv', 'leaky_relu') # 'leaky_relu' or 'relu'
        
        # Use LeakyReLU if specified, else relu
        activation_func = LeakyReLU(alpha=0.2) if d_activation_conv == 'leaky_relu' else Activation('relu')

        self.logger.info(f"Discriminator core using Conv1D with activation: {d_activation_conv}")

        x = Conv1D(filters=d_filters_1, kernel_size=d_kernel_size_1, strides=d_strides_1, padding='same', name='d_conv1')(processed_features)
        # x = BatchNormalization(name='d_bn1')(x) # Optional Batch Norm
        x = activation_func(x)
        x = Dropout(d_dropout_1, name='d_dropout1')(x)
        self.logger.info(f"Discriminator: After Conv1 layer 1 shape: {x.shape}")

        x = Conv1D(filters=d_filters_2, kernel_size=d_kernel_size_2, strides=d_strides_2, padding='same', name='d_conv2')(x)
        # x = BatchNormalization(name='d_bn2')(x) # Optional Batch Norm
        x = activation_func(x)
        x = Dropout(d_dropout_2, name='d_dropout2')(x)
        self.logger.info(f"Discriminator: After Conv1 layer 2 shape: {x.shape}")
        
        x = Conv1D(filters=d_filters_3, kernel_size=d_kernel_size_3, strides=d_strides_3, padding='same', name='d_conv3')(x)
        # x = BatchNormalization(name='d_bn3')(x) # Optional Batch Norm
        x = activation_func(x)
        x = Dropout(d_dropout_3, name='d_dropout3')(x)
        self.logger.info(f"Discriminator: After Conv1 layer 3 shape: {x.shape}")

        x = Flatten(name='d_flatten')(x)
        self.logger.info(f"Discriminator: After Flatten shape: {x.shape}")
        
        x = Dense(d_dense_units, name='d_dense_hidden')(x)
        # x = BatchNormalization(name='d_bn_dense')(x) # Optional Batch Norm
        x = activation_func(x)
        x = Dropout(self.params.get('d_dropout_dense', 0.4), name='d_dropout_dense')(x) # Dropout for dense layer
        self.logger.info(f"Discriminator: After Dense hidden layer shape: {x.shape}")

        # Output layer: A single neuron with sigmoid activation for binary classification (real/fake)
        validity_output = Dense(1, activation='sigmoid', name='d_output_validity')(x)
        self.logger.info(f"Discriminator: Output layer shape: {validity_output.shape}")

        # Create and compile the Discriminator model
        discriminator_model = Model(inputs=d_inputs, outputs=validity_output, name="Discriminator")
        
        # Optimizer for the discriminator
        d_optimizer_type = self.params.get('d_optimizer', 'adam').lower()
        d_learning_rate = self.params.get('d_learning_rate', 0.0002) # Common GAN LR
        d_beta_1 = self.params.get('d_beta_1', 0.5) # Common for Adam in GANs
        
        if d_optimizer_type == 'adam':
            optimizer = Adam(learning_rate=d_learning_rate, beta_1=d_beta_1)
        elif d_optimizer_type == 'rmsprop':
            optimizer = RMSprop(learning_rate=d_learning_rate)
        elif d_optimizer_type == 'sgd':
            optimizer = SGD(learning_rate=d_learning_rate)
        else:
            self.logger.warning(f"Unsupported discriminator optimizer: {d_optimizer_type}. Defaulting to Adam.")
            optimizer = Adam(learning_rate=d_learning_rate, beta_1=d_beta_1)
            
        self.logger.info(f"Discriminator optimizer: {d_optimizer_type}, LR: {d_learning_rate}")

        discriminator_model.compile(loss=self.params.get('d_loss', 'binary_crossentropy'),
                                    optimizer=optimizer,
                                    metrics=['accuracy', Precision(name='precision'), Recall(name='recall'), AUC(name='auc')])
        
        self.logger.info("Discriminator model built and compiled successfully.")
        if self.params.get("d_print_summary", True):
            discriminator_model.summary(print_fn=self.logger.info)
            
        return discriminator_model

    def _build_gan(self) -> Model:
        """
        Builds the GAN model by combining the generator and discriminator.
        The GAN model is trained to generate realistic sequences that can fool the discriminator.
        """
        self.logger.info("Building GAN model...")

        # Ensure the generator is not trainable during combined model training
        self.generator.trainable = False
        self.logger.info(f"Generator set to trainable=False for GAN model training.")

        # GAN inputs:
        # 1. Latent vector for the generator
        # The shape of this latent_input must match what the generator expects as its primary noise input.
        # self.latent_input_shape_for_gan_input_layer is (latent_dim,)
        gan_latent_input = Input(shape=self.latent_input_shape_for_gan_input_layer, name="gan_latent_input")
        current_gan_inputs = [gan_latent_input]
        generator_inputs_for_gan = [gan_latent_input] # Start with the main latent input

        # 2. Conditional input for the generator (if applicable)
        if self.conditional_dim > 0:
            # This input is passed to both generator and discriminator (if discriminator uses it)
            # For the GAN model, this conditional input is fed directly to the generator part.
            gan_conditional_input = Input(shape=(self.conditional_dim,), name="gan_conditional_input")
            current_gan_inputs.append(gan_conditional_input)
            generator_inputs_for_gan.append(gan_conditional_input)
            self.logger.info(f"GAN model: Added conditional input of dim {self.conditional_dim} for generator.")

        # 3. Context input for the generator (if applicable)
        if self.context_dim > 0:
            # Similar to conditional input, for the generator within the GAN.
            gan_context_input = Input(shape=(self.context_dim,), name="gan_context_input")
            current_gan_inputs.append(gan_context_input)
            generator_inputs_for_gan.append(gan_context_input)
            self.logger.info(f"GAN model: Added context input of dim {self.context_dim} for generator.")
            
        # Get generated data from the generator
        # The generator_inputs_for_gan list should match the order and number of inputs the self.generator model expects.
        self.logger.info(f"Generator inputs for GAN: {generator_inputs_for_gan}")
        try:
            # If generator_inputs_for_gan has only one element and generator expects a single tensor (not a list)
            if len(generator_inputs_for_gan) == 1 and not isinstance(self.generator.input, list):
                 generated_data = self.generator(generator_inputs_for_gan[0])
            else:
                 generated_data = self.generator(generator_inputs_for_gan)
        except Exception as e:
            self.logger.error(f"Error when trying to call generator with inputs {generator_inputs_for_gan}: {e}")
            self.logger.error(f"Generator model expected inputs: {self.generator.inputs}")
            self.logger.error(f"Generator model summary:")
            self.generator.summary(print_fn=self.logger.error)
            raise

        # Discriminator part of the GAN
        # The discriminator's inputs for the GAN model:
        # 1. The generated_data from the generator
        # 2. Conditional input (if discriminator uses it and it's part of GAN inputs)
        # 3. Context input (if discriminator uses it and it's part of GAN inputs)
        
        discriminator_inputs_for_gan = [generated_data]
        if self.conditional_dim > 0 and 'gan_conditional_input' in [inp.name for inp in current_gan_inputs]:
            # Find the gan_conditional_input from current_gan_inputs
            cond_inp_for_d = next(inp for inp in current_gan_inputs if inp.name == "gan_conditional_input")
            discriminator_inputs_for_gan.append(cond_inp_for_d)
            self.logger.info(f"GAN model: Passing conditional input to discriminator.")
            
        if self.context_dim > 0 and 'gan_context_input' in [inp.name for inp in current_gan_inputs]:
            # Find the gan_context_input from current_gan_inputs
            context_inp_for_d = next(inp for inp in current_gan_inputs if inp.name == "gan_context_input")
            discriminator_inputs_for_gan.append(context_inp_for_d)
            self.logger.info(f"GAN model: Passing context input to discriminator.")

        self.logger.info(f"Discriminator inputs for GAN: {discriminator_inputs_for_gan}")
        try:
            if len(discriminator_inputs_for_gan) == 1 and not isinstance(self.discriminator.input, list):
                gan_output_validity = self.discriminator(discriminator_inputs_for_gan[0])
            else:
                gan_output_validity = self.discriminator(discriminator_inputs_for_gan)
        except Exception as e:
            self.logger.error(f"Error when trying to call discriminator with inputs {discriminator_inputs_for_gan}: {e}")
            self.logger.error(f"Discriminator model expected inputs: {self.discriminator.inputs}")
            self.logger.error(f"Discriminator model summary:")
            self.discriminator.summary(print_fn=self.logger.error)
            raise

        # Create and compile the GAN model
        # The GAN model takes latent noise (and optional conditions/context) and outputs discriminator's decision
        gan_model = Model(inputs=current_gan_inputs, outputs=gan_output_validity, name="GAN")
        
        # Optimizer for the GAN (trains only the generator weights)
        gan_optimizer_type = self.params.get('gan_optimizer', 'adam').lower()
        gan_learning_rate = self.params.get('gan_learning_rate', 0.0002) # Often same as D
        gan_beta_1 = self.params.get('gan_beta_1', 0.5) # Common for Adam in GANs

        if gan_optimizer_type == 'adam':
            optimizer = Adam(learning_rate=gan_learning_rate, beta_1=gan_beta_1)
        elif gan_optimizer_type == 'rmsprop':
            optimizer = RMSprop(learning_rate=gan_learning_rate)
        elif gan_optimizer_type == 'sgd':
            optimizer = SGD(learning_rate=gan_learning_rate)
        else:
            self.logger.warning(f"Unsupported GAN optimizer: {gan_optimizer_type}. Defaulting to Adam.")
            optimizer = Adam(learning_rate=gan_learning_rate, beta_1=gan_beta_1)
        
        self.logger.info(f"GAN optimizer: {gan_optimizer_type}, LR: {gan_learning_rate}")

        gan_model.compile(loss=self.params.get('gan_loss', 'binary_crossentropy'), 
                          optimizer=optimizer,
                          metrics=['accuracy']) # Metrics here are on GAN output, less critical than D's

        self.logger.info("GAN model built and compiled successfully.")
        if self.params.get("gan_print_summary", True):
            gan_model.summary(print_fn=self.logger.info)
            
        return gan_model

    def _get_data_generator(self, x_real_df: pd.DataFrame, batch_size: int) -> Any:
        # Ensure x_real_df is not empty and has enough data
        if x_real_df is None or x_real_df.empty:
            self.logger.error("x_real_df is None or empty in _get_data_generator.")
            raise ValueError("Input data (x_real_df) cannot be None or empty.")
        if len(x_real_df) < self.seq_len:
            self.logger.error(f"Not enough data in x_real_df ({len(x_real_df)}) to form a sequence of length {self.seq_len}.")
            raise ValueError("Not enough data to form sequences.")

        self.logger.info(f"_get_data_generator called with x_real_df shape: {x_real_df.shape}, batch_size: {batch_size}, seq_len: {self.seq_len}")

        # Convert DataFrame to numpy array if it's not already
        if isinstance(x_real_df, pd.DataFrame):
            x_real_np = x_real_df.values
        elif isinstance(x_real_df, np.ndarray):
            x_real_np = x_real_df
        else:
            self.logger.error(f"x_real_df is of unsupported type: {type(x_real_df)}")
            raise TypeError("x_real_df must be a pandas DataFrame or numpy array.")

        if x_real_np.shape[1] != self.num_base_features:
            self.logger.warning(f"Mismatch between x_real_df features ({x_real_np.shape[1]}) and num_base_features ({self.num_base_features}). Adjusting num_base_features for data generator.")
            # This implies num_base_features might not have been correctly derived or data changed.
            # For the generator, we must use the actual feature count from data.
            # However, this could cause issues if the model was built with a different num_base_features.
            # For now, we'll proceed but this is a potential inconsistency.
            # Consider if self.num_base_features should be updated here, or if an error should be raised.

        # Prepare sequences from the real data
        num_samples = len(x_real_np) - self.seq_len + 1
        if num_samples <= 0:
            self.logger.error(f"Cannot create any sequences from data with length {len(x_real_np)} and seq_len {self.seq_len}")
            # This case should ideally be caught earlier
            return iter([]) # Return an empty iterator

        all_sequences = np.array([x_real_np[i:i + self.seq_len] for i in range(num_samples)])
        self.logger.info(f"Created {num_samples} sequences of shape {all_sequences.shape}")

        # Prepare conditional data if present
        epoch_conditional_data_aligned = None
        if self.conditional_dim > 0:
            self.logger.info("Preparing conditional data for epoch.")
            if hasattr(self, 'feeder_plugin_instance') and self.feeder_plugin_instance and \
               hasattr(self.feeder_plugin_instance, 'get_conditional_data_for_batch_indices'): # Assuming a method that takes indices
                # This is a more robust way: get conditional data based on the indices of the sequences
                # We need to align conditional data with `all_sequences`.
                # This requires careful handling based on how conditional data is structured.
                # For simplicity, if `self.conditional_vectors` is already aligned with `x_real_df` (the raw data),
                # we might need to slice it or use a specific method from feeder.
                # Let's assume `self.conditional_vectors` are aligned with the *start* of each sequence.
                if hasattr(self, 'conditional_vectors') and self.conditional_vectors is not None:
                    if len(self.conditional_vectors) >= num_samples: # Check if enough conditional vectors
                        # Assuming conditional_vectors[i] corresponds to sequence starting at x_real_np[i]
                        epoch_conditional_data_aligned = self.conditional_vectors[:num_samples]
                        self.logger.info(f"Using self.conditional_vectors, aligned to {num_samples} sequences. Shape: {epoch_conditional_data_aligned.shape}")
                        if epoch_conditional_data_aligned.shape[1] != self.conditional_dim:
                             self.logger.error(f"Conditional data feature mismatch. Expected {self.conditional_dim}, got {epoch_conditional_data_aligned.shape[1]}")
                             epoch_conditional_data_aligned = None # Invalidate
                    else:
                        self.logger.warning(f"Not enough conditional vectors ({len(self.conditional_vectors)}) for {num_samples} sequences.")
                else:
                    self.logger.warning("self.conditional_vectors not available or not set.")
            else: # Fallback or alternative logic
                 self.logger.info("Feeder plugin does not have 'get_conditional_data_for_batch_indices' or not available.")
            
            if epoch_conditional_data_aligned is None:
                 self.logger.warning("Conditional data (dim > 0) requested but could not be prepared/aligned for the epoch.")
        
        # Prepare context data if present
        epoch_context_vectors_aligned = None # Initialize here
        if self.context_dim > 0:
            self.logger.info("Preparing context data for epoch.")
            # Similar to conditional data
            if hasattr(self, 'feeder_plugin_instance') and self.feeder_plugin_instance and \
               hasattr(self.feeder_plugin_instance, 'get_context_data_for_batch_indices'): # Hypothetical method
                if hasattr(self, 'context_vectors') and self.context_vectors is not None:
                    if len(self.context_vectors) >= num_samples:
                        epoch_context_vectors_aligned = self.context_vectors[:num_samples]
                        self.logger.info(f"Using self.context_vectors, aligned to {num_samples} sequences. Shape: {epoch_context_vectors_aligned.shape}")
                        if epoch_context_vectors_aligned.shape[1] != self.context_dim:
                             self.logger.error(f"Context data feature mismatch. Expected {self.context_dim}, got {epoch_context_vectors_aligned.shape[1]}")
                             epoch_context_vectors_aligned = None # Invalidate
                    else:
                        self.logger.warning(f"Not enough context vectors ({len(self.context_vectors)}) for {num_samples} sequences.")
                else:
                    self.logger.warning("self.context_vectors not available or not set for context data.")
            else:
                self.logger.info("Feeder plugin does not have 'get_context_data_for_batch_indices' or not available for context.")

            if epoch_context_vectors_aligned is None:
                self.logger.warning("Context data (dim > 0) requested but could not be prepared/aligned for the epoch.")


        indices = np.arange(num_samples)
        
        current_index = 0
        while True:
            if current_index == 0: # Shuffle at the beginning of each virtual epoch pass
                self.logger.debug(f"Shuffling data indices for generator. Num samples: {num_samples}")
                np.random.shuffle(indices)

            # Determine batch indices
            start_idx = current_index
            end_idx = current_index + batch_size
            
            if end_idx > num_samples:
                # Not enough samples for a full batch, take what's left and reshuffle for next pass
                batch_indices = indices[start_idx:]
                current_index = 0 # Reset for reshuffle on next call for this generator instance
                if len(batch_indices) == 0: continue # No more batches to process
                if len(batch_indices) < batch_size:
                    self.logger.info(f"Last batch with {len(batch_indices)} samples, reshuffling for next epoch.")
                else:
                    self.logger.debug(f"Yielding batch from {start_idx} to end of data ({num_samples}).")
            else:
                batch_indices = indices[start_idx:end_idx]
                current_index = end_idx
            
            if len(batch_indices) == 0:
                # This can happen if batch_size > num_samples and we've yielded the < batch_size chunk
                # Or if num_samples was 0 initially (though caught earlier)
                self.logger.debug("No batch indices to yield, likely end of data or small dataset. Reshuffling.")
                current_index = 0
                np.random.shuffle(indices)
                # Attempt to form a batch again
                start_idx = 0
                end_idx = min(batch_size, num_samples) # Ensure end_idx doesn't exceed num_samples
                if end_idx == 0 : # Still no data
                    yield # Yield nothing or break, depends on Keras expectations for empty generator
                    continue
                batch_indices = indices[start_idx:end_idx]
                current_index = end_idx


            x_batch_sequences = all_sequences[batch_indices]

            # Yielding logic
            batch_yield = [x_batch_sequences] # Start with main data

            if self.conditional_dim > 0:
                if epoch_conditional_data_aligned is not None:
                    conditional_batch_for_yield = epoch_conditional_data_aligned[batch_indices]
                    if conditional_batch_for_yield.ndim == 1 and self.conditional_dim == 1:
                        conditional_batch_for_yield = conditional_batch_for_yield.reshape(-1, 1)
                    elif conditional_batch_for_yield.ndim > 2 and conditional_batch_for_yield.shape[1] == 1: # e.g. (batch, 1, dim)
                        conditional_batch_for_yield = np.squeeze(conditional_batch_for_yield, axis=1)
                    
                    if conditional_batch_for_yield.shape[1] != self.conditional_dim:
                        self.logger.error(f"CRITICAL: Conditional batch shape mismatch. Expected dim {self.conditional_dim}, got {conditional_batch_for_yield.shape}. Indices length: {len(batch_indices)}")
                        # Handle error: skip this batch or raise? For now, don't append if problematic.
                    else:
                        batch_yield.append(conditional_batch_for_yield)
                else:
                    # If conditional_dim > 0 but data is None, this indicates an issue.
                    # The model will expect this input. We might need to yield zeros or raise error.
                    # For now, this means the input list to the model might be shorter than expected.
                    self.logger.warning("Conditional data expected but not available for batch. Model input structure might be incorrect.")


            if self.context_dim > 0: # Check if context data is expected
                if epoch_context_vectors_aligned is not None: # Check if it was successfully prepared
                    context_batch_for_yield = epoch_context_vectors_aligned[batch_indices]
                    if context_batch_for_yield.ndim == 1 and self.context_dim == 1:
                        context_batch_for_yield = context_batch_for_yield.reshape(-1, 1)
                    elif context_batch_for_yield.ndim > 2 and context_batch_for_yield.shape[1] == 1: # e.g. (batch, 1, dim)
                         context_batch_for_yield = np.squeeze(context_batch_for_yield, axis=1)

                    if context_batch_for_yield.shape[1] != self.context_dim:
                         self.logger.error(f"CRITICAL: Context batch shape mismatch. Expected dim {self.context_dim}, got {context_batch_for_yield.shape}. Indices length: {len(batch_indices)}")
                         # Handle error: skip or raise
                    else:
                        batch_yield.append(context_batch_for_yield)
                else:
                    self.logger.warning("Context data expected but not available for batch. Model input structure might be incorrect.")
            
            # self.logger.debug(f"Yielding batch. Main data shape: {x_batch_sequences.shape}")
            # if self.conditional_dim > 0 and epoch_conditional_data_aligned is not None and len(batch_yield) > 1:
            #     self.logger.debug(f"Conditional data shape: {batch_yield[1].shape if len(batch_yield) > 1 else 'N/A'}")
            # if self.context_dim > 0 and epoch_context_vectors_aligned is not None and len(batch_yield) > (1 + (1 if self.conditional_dim > 0 and epoch_conditional_data_aligned is not None else 0)) :
            #    idx = 1 + (1 if self.conditional_dim > 0 and epoch_conditional_data_aligned is not None else 0)
            #    self.logger.debug(f"Context data shape: {batch_yield[idx].shape if len(batch_yield) > idx else 'N/A'}")


            if len(batch_yield) == 1:
                yield batch_yield[0] # Single input (main data only)
            elif len(batch_yield) > 1:
                yield tuple(batch_yield) # Multiple inputs (main, conditional, context)

    def train(self, x_real_df: pd.DataFrame, epochs: int, batch_size: int, train_discriminator_n_times: int = 1, train_generator_n_times: int = 1):
        # ... (method content as before) ...
        pass

    def save_models(self, path_prefix: str = "gan_model_"):
        # ... (method content as before) ...
        pass

    def load_models(self, path_prefix: str = "gan_model_"):
        # ... (method content as before) ...
        pass

    def set_params(self, **kwargs):
        """
        Updates plugin parameters and rebuilds models if necessary.
        """
        self.logger.info(f"GANTrainerPlugin: Setting parameters with kwargs: {list(kwargs.keys())}")

        # Update self.params with new kwargs. This is the primary parameter store.
        if not hasattr(self, 'params') or self.params is None:
            self.params = deepcopy(self.plugin_params) # Ensure params is initialized
            self.logger.info("GANTrainerPlugin.set_params: self.params was not initialized. Initialized with plugin_params defaults.")

        for key, value in kwargs.items():
            self.params[key] = value
            # Also update self.config if these kwargs are meant to be part of the main config scope
            # This depends on how self.config is used elsewhere. Generally, self.params should be sufficient.
            if hasattr(self, 'config') and self.config is not None:
                self.config[key] = value 
            else:
                self.config = {key: value} # Initialize if it doesn't exist

        self.logger.info(f"GANTrainerPlugin: Updated self.params. Keys: {list(self.params.keys())}")

        # Update instance attributes that are direct copies of params for convenience
        # Example: self.epochs = self.params.get('gan_epochs', 10000)
        # self.batch_size = self.params.get('gan_batch_size', 32)

        # Resolve paths for saving results based on updated params
        self.results_base_dir = self.params.get("results_base_dir", "examples/results/gan_training")
        self.models_dir = os.path.join(self.results_base_dir, self.params.get("save_model_dir", "models"))
        self.plots_dir = os.path.join(self.results_base_dir, self.params.get("save_plot_dir", "plots"))
        self.metrics_dir = os.path.join(self.results_base_dir, self.params.get("save_metrics_dir", "metrics"))

        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
        self.logger.info(f"GANTrainerPlugin: Results will be saved under {self.results_base_dir}")

        # Model plot file paths
        self.generator_model_plot_file = os.path.join(self.plots_dir, self.params.get("generator_model_plot_file", "generator_architecture.png"))
        self.discriminator_model_plot_file = os.path.join(self.plots_dir, self.params.get("discriminator_model_plot_file", "discriminator_architecture.png"))
        self.gan_model_plot_file = os.path.join(self.plots_dir, self.params.get("gan_model_plot_file", "gan_architecture.png"))
        self.model_plot_dpi = self.params.get("model_plot_dpi", 300)

        # Retrieve the generator model from the generator_plugin_instance
        if self.generator_plugin:
            if hasattr(self.generator_plugin, 'generator_model') and self.generator_plugin.generator_model:
                self.generator = self.generator_plugin.generator_model
                self.logger.info("GANTrainerPlugin: Generator model retrieved from generator_plugin.generator_model.")
            elif hasattr(self.generator_plugin, 'model') and self.generator_plugin.model: # Fallback
                self.generator = self.generator_plugin.model
                self.logger.info("GANTrainerPlugin: Generator model retrieved from generator_plugin.model.")
            elif hasattr(self.generator_plugin, 'get_model') and callable(self.generator_plugin.get_model):
                self.generator = self.generator_plugin.get_model()
                self.logger.info("GANTrainerPlugin: Generator model retrieved via generator_plugin.get_model().")
            else:
                self.logger.error("GANTrainerPlugin: Could not retrieve generator model from generator_plugin_instance.")
                # raise ValueError("Generator model not found in generator_plugin_instance.") # Consider if this should be fatal
        else:
            self.logger.warning("GANTrainerPlugin: generator_plugin_instance is not provided. Generator model cannot be set.")

        if self.generator is None:
            self.logger.error("GANTrainerPlugin: CRITICAL - Generator model is None after attempting to retrieve it. Cannot proceed with GAN setup.")
            # Depending on workflow, this might be acceptable if models are loaded later, but typically fatal for GAN building.
            # For now, we allow it to proceed, but _initialize_core_parameters_from_config and model building will likely fail.

        # Initialize core parameters (like seq_len, latent_dim, feature_dims) based on the (potentially new) generator and config
        # This must be called AFTER self.generator is set and self.params is updated.
        self._initialize_core_parameters_from_config()
        self.logger.info("GANTrainerPlugin: Core parameters initialized after set_params.")

        # Build or re-build models if they don't exist or if critical parameters changed
        # For simplicity, let's assume we always rebuild if set_params is called with a generator.
        # More sophisticated logic could check if a rebuild is truly necessary.
        if self.generator: # Only build if we have a generator
            self.logger.info("GANTrainerPlugin: Building/Rebuilding Discriminator and GAN models...")
            if self.discriminator:
                self.logger.info("GANTrainerPlugin: Existing discriminator found, will be replaced.")
            self.discriminator = self._build_discriminator()
            
            if self.gan_model:
                self.logger.info("GANTrainerPlugin: Existing GAN model found, will be replaced.")
            self.gan_model = self._build_gan()
            self.logger.info("GANTrainerPlugin: Discriminator and GAN models built/rebuilt.")
        else:
            self.logger.warning("GANTrainerPlugin: Generator model not available in set_params. Discriminator and GAN models not built/rebuilt.")

        self.logger.info("GANTrainerPlugin: set_params complete.")

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        Gets parameters for this plugin.
        Args:
            deep: If True, will return a deep copy of the parameters.
        Returns:
            A dictionary of parameters.
        """
        if not hasattr(self, 'params') or self.params is None:
            # This case should ideally not happen if __init__ and set_params are correctly called.
            self.logger.warning("GANTrainerPlugin.get_params: self.params is not initialized. Returning a copy of class-level plugin_params.")
            return deepcopy(self.plugin_params) if deep else self.plugin_params.copy()
        
        return deepcopy(self.params) if deep else self.params.copy()

# ...existing code...
