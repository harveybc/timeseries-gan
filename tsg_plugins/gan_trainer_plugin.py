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
import logging # Ensure logging is imported
import time # Add time import
import matplotlib.pyplot as plt # Add matplotlib import
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

from copy import deepcopy

# Initialize logger for this module
logger = logging.getLogger(__name__)
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
        "feeder_max_hour_of_day": 24.0 # Assumes 0-23
        "feeder_max_day_of_week": 7.0,  # Assumes 0-6
        # Add other feeder_max_... if other date features like day_of_year, minute_of_hour are used.

        "conditional_fundamental_feature_names": ["S&P500_Close", "vix_close"], # From normalized_d4.csv
        "num_conditional_prev_tick_features": 5, # Number of features from the start of the real sequence to use as conditional input
                                                 # Total conditional dim = len(date_feats) + len(fund_feats) + num_prev_tick_feats
                                                 # e.g. 3 + 2 + 5 = 10
        "datetime_col_name_in_x_real_df": "DATE_TIME" # Column name for datetimes in input x_real_df
    }

    def __init__(self, config: Dict[str, Any], generator_plugin_instance: Optional[Any] = None, feeder_plugin_instance: Optional[Any] = None, preprocessor_plugin_instance: Optional[Any] = None):
        self.ti_names_for_discriminator = []
        self.config = deepcopy(config)
        self.generator_plugin = generator_plugin_instance
        self.feeder_plugin = feeder_plugin_instance
        self.preprocessor_plugin = preprocessor_plugin_instance

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
        
        # Core model components will be built in set_params
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
        self.logger.info("Starting _initialize_core_parameters_from_config...")

        # --- Generator Output Parameters (used by Discriminator construction) ---
        self.seq_len = None
        self.num_base_features = None # This is the number of features in the *generated* sequence (e.g., just 'close_price')

        # 1. Try to get dimensions from the generator model itself
        if self.generator and hasattr(self.generator, 'output_shape') and self.generator.output_shape:
            self.logger.info(f"Attempting to derive dimensions from generator.output_shape: {self.generator.output_shape}")
            # Example: (None, 60, 1) -> seq_len=60, num_base_features=1
            # Example: (None, 1) for a model that outputs a single value per feature (e.g. if seq_len is 1)
            # Example: (None, 5) for a model that outputs 5 features (if seq_len is implicitly 1 or not applicable)
            output_shape = self.generator.output_shape
            if isinstance(output_shape, tuple) and len(output_shape) >= 2: # Should be at least (batch, features) or (batch, seq, features)
                if len(output_shape) == 3: # (batch, seq_len, features)
                    self.seq_len = output_shape[1]
                    self.num_base_features = output_shape[2]
                    self.logger.info(f"  Derived from 3D generator.output_shape: seq_len={self.seq_len}, num_base_features={self.num_base_features}")
                elif len(output_shape) == 2: # (batch, features) - implies seq_len might be 1 or handled differently
                    # If seq_len is expected to be > 1, this might not be the right place, or model structure is different.
                    # For now, assume this means num_base_features is output_shape[1] and seq_len might be set from config or is 1.
                    self.num_base_features = output_shape[1]
                    self.logger.info(f"  Derived from 2D generator.output_shape: num_base_features={self.num_base_features}. seq_len will be sought from config or assumed 1 if not found.")
                    # We still need seq_len. Try to get it from config.
                    self.seq_len = self.params.get('gan_generator_output_actual_seq_len', self.params.get('seq_len'))
                    if self.seq_len:
                        self.logger.info(f"  seq_len for 2D output case found in config: {self.seq_len}")
                    else:
                        self.logger.warning("  seq_len for 2D output case not found in config. It might be assumed as 1 or needs explicit setting.")
                else:
                    self.logger.warning(f"  Generator output_shape {output_shape} has an unexpected number of dimensions. Cannot reliably derive seq_len and num_base_features.")
            else:
                self.logger.warning(f"  Generator output_shape {output_shape} is not a tuple or has too few dimensions. Cannot derive seq_len and num_base_features.")

        elif self.generator and hasattr(self.generator, 'output') and hasattr(self.generator.output, 'shape'):
            self.logger.info(f"Attempting to derive dimensions from generator.output.shape: {self.generator.output.shape}")
            output_shape = self.generator.output.shape 
            if len(output_shape) == 3: # (None, seq_len, features)
                self.seq_len = output_shape[1].value if hasattr(output_shape[1], 'value') else output_shape[1]
                self.num_base_features = output_shape[2].value if hasattr(output_shape[2], 'value') else output_shape[2]
                self.logger.info(f"  Derived from 3D generator.output.shape: seq_len={self.seq_len}, num_base_features={self.num_base_features}")
            elif len(output_shape) == 2: # (None, features)
                self.num_base_features = output_shape[1].value if hasattr(output_shape[1], 'value') else output_shape[1]
                self.logger.info(f"  Derived from 2D generator.output.shape: num_base_features={self.num_base_features}. seq_len will be sought from config.")
                self.seq_len = self.params.get('gan_generator_output_actual_seq_len', self.params.get('seq_len'))
                if self.seq_len:
                    self.logger.info(f"  seq_len for 2D output case found in config: {self.seq_len}")
                else:
                    self.logger.warning("  seq_len for 2D output case not found in config. It might be assumed as 1 or needs explicit setting.")
            else:
                self.logger.warning(f"  Generator output.shape {output_shape} has an unexpected number of dimensions.")
        else:
            self.logger.info("Generator model or its output shape not available at this point. Will rely on config parameters.")

        # 2. Fallback to config for seq_len if not derived from model
        if self.seq_len is None:
            self.seq_len = self.params.get('gan_generator_output_actual_seq_len')
            self.logger.info(f"Attempting to get 'gan_generator_output_actual_seq_len' from params: {self.seq_len}")
            if self.seq_len is None:
                self.seq_len = self.params.get('seq_len') # General seq_len
                self.logger.info(f"Attempting to get 'seq_len' from params: {self.seq_len}")
        
        # 3. Fallback to config for num_base_features if not derived from model
        if self.num_base_features is None:
            self.num_base_features = self.params.get('gan_generator_output_actual_features')
            self.logger.info(f"Attempting to get 'gan_generator_output_actual_features' from params: {self.num_base_features}")
            if self.num_base_features is None:
                self.num_base_features = self.params.get('num_base_features_generated')
                self.logger.info(f"Attempting to get 'num_base_features_generated' from params: {self.num_base_features}")
                if self.num_base_features is None:
                    self.num_base_features = self.params.get('feature_dim') # More generic, might be used if others are missing
                    self.logger.info(f"Attempting to get 'feature_dim' from params: {self.num_base_features}")

        # 4. Fallback for num_base_features using generator_plugin_instance's decoder_output_feature_names
        if self.num_base_features is None and self.generator_plugin_instance:
            self.logger.info("Attempting to derive num_base_features from generator_plugin_instance.params['decoder_output_feature_names']")
            gen_plugin_decoder_outputs = self.generator_plugin_instance.params.get('decoder_output_feature_names')
            if gen_plugin_decoder_outputs and isinstance(gen_plugin_decoder_outputs, list):
                self.num_base_features = len(gen_plugin_decoder_outputs)
                self.logger.info(f"  Derived num_base_features from len(decoder_output_feature_names): {self.num_base_features}")
            else:
                self.logger.info(f"  'decoder_output_feature_names' not found or not a list in generator_plugin_instance.params: {gen_plugin_decoder_outputs}")
        
        # Convert to int if they are found and are strings
        if self.seq_len is not None:
            try:
                self.seq_len = int(self.seq_len)
            except ValueError:
                self.logger.error(f"Could not convert derived/configured seq_len ('{self.seq_len}') to int. Resetting to None.")
                self.seq_len = None
        
        if self.num_base_features is not None:
            try:
                self.num_base_features = int(self.num_base_features)
            except ValueError:
                self.logger.error(f"Could not convert derived/configured num_base_features ('{self.num_base_features}') to int. Resetting to None.")
                self.num_base_features = None

        # Critical check: Ensure these are now set.
        error_messages = []
        if self.seq_len is None:
            error_messages.append("Could not determine generator output sequence length (checked model, params: 'gan_generator_output_actual_seq_len', 'seq_len').")
        if self.num_base_features is None:
            error_messages.append("Could not determine generator output features (checked model, params: 'gan_generator_output_actual_features', 'num_base_features_generated', 'feature_dim', or 'generator_decoder_output_feature_names' from GeneratorPlugin).")

        if error_messages:
            full_error_msg = (
                "Critical: " + " ".join(error_messages) +
                " Please ensure the generator Keras model is loaded correctly and has defined outputs, OR provide these values in the trainer configuration: \\n" +
                "  - 'gan_generator_output_actual_seq_len' or 'seq_len' (for sequence length)\\n" +
                "  - 'gan_generator_output_actual_features', 'num_base_features_generated', 'feature_dim', or ensure 'generator_decoder_output_feature_names' is set in the generator's specific config section (for features)."
            )
            self.logger.error(full_error_msg)
            raise ValueError(full_error_msg)

        self.logger.info(f"Final generator output dimensions determined: self.seq_len={self.seq_len}, self.num_base_features={self.num_base_features}")

        # --- Generator Input Parameters (used by GAN construction for its inputs) ---
        # These define what the GAN model will expect as inputs to feed the generator part
        self.latent_dim_for_generator = self.params.get('latent_dim', 100) # Default if not specified
        self.conditional_dim_for_generator = self.params.get('conditional_dim') # Can be None if not conditional
        self.context_dim_for_generator = self.params.get('context_dim') # Can be None if no context input

        self.logger.info(f"Initial generator input dims from config: latent_dim={self.latent_dim_for_generator}, conditional_dim={self.conditional_dim_for_generator}, context_dim={self.context_dim_for_generator}")

        # Store Keras input layer names if available, to help match them later when building GAN
        self.gan_latent_input_keras_name_hint = None
        self.gan_conditional_input_keras_name_hint = None
        self.gan_context_input_keras_name_hint = None
        
        # Try to get actual input dimensions and names from the generator model if it's loaded
        if self.generator and hasattr(self.generator, 'inputs') and self.generator.inputs:
            self.logger.info(f"Generator has {len(self.generator.inputs)} inputs. Analyzing them...")
            found_latent = False
            found_conditional = False
            found_context = False

            # Try to match inputs based on common naming conventions or configured feeder keys
            for i, keras_input_tensor in enumerate(self.generator.inputs):
                input_name_from_keras = keras_input_tensor.name # e.g., "input_1" or "latent_input_layer"
                input_shape_from_keras = keras_input_tensor.shape # e.g. TensorShape([None, 100]) or TensorShape([None, 10, 1])
                self.logger.info(f"Processing generator input {i}: Keras name '{input_name_from_keras}', Keras shape {input_shape_from_keras}")

                # Check for Latent Vector (often contains "latent", "noise", "z")
                if "latent" in input_name_from_keras.lower() or "noise" in input_name_from_keras.lower() or "z_input" in input_name_from_keras.lower() or (self.feeder_key_name_noise and self.feeder_key_name_noise in input_name_from_keras.lower()):
                    self.logger.info(f"  Input '{input_name_from_keras}' matched as LATENT.")
                    if len(input_shape_from_keras) == 2 and input_shape_from_keras[1] is not None:
                        self.latent_dim_for_generator = input_shape_from_keras[1]
                        self.gan_latent_input_keras_name_hint = input_name_from_keras # Store Keras name
                        self.logger.info(f"    Set for GAN (from Gen): latent_dim_for_generator = {self.latent_dim_for_generator}. Keras name hint: {self.gan_latent_input_keras_name_hint}")
                    else:
                        self.logger.warning(f"  Latent input '{input_name_from_keras}' has unexpected shape {input_shape_from_keras} or undefined feature dim. Using config/default for latent_dim.")
                    found_latent = True
                
                # Check for Conditional Vector
                elif self.feeder_key_name_conditional and (self.feeder_key_name_conditional in input_name_from_keras.lower() or f"input_{self.feeder_key_name_conditional}" in input_name_from_keras.lower()):
                    self.logger.info(f"  Input '{input_name_from_keras}' matched as CONDITIONAL based on feeder key '{self.feeder_key_name_conditional}'.")
                    # Shape can be (None, cond_dim) or (None, seq_len, cond_dim_per_step)
                    if len(input_shape_from_keras) >= 2 and input_shape_from_keras[-1] is not None:
                        self.conditional_dim_for_generator = input_shape_from_keras[-1] # Last dim is feature dim
                        self.gan_conditional_input_keras_name_hint = input_name_from_keras # Store Keras name
                        self.logger.info(f"    Set for GAN (from Gen): conditional_dim_for_generator = {self.conditional_dim_for_generator} (from shape {input_shape_from_keras}). Keras name hint: {self.gan_conditional_input_keras_name_hint}")
                    else:
                        self.logger.warning(f"  Conditional input '{input_name_from_keras}' has unexpected shape {input_shape_from_keras} or undefined feature dim. Using config/default for conditional_dim.")
                    found_conditional = True
                
                # Check for Context Vector
                elif self.feeder_key_name_context and (self.feeder_key_name_context in input_name_from_keras.lower() or f"input_{self.feeder_key_name_context}" in input_name_from_keras.lower()):
                    self.logger.info(f"  Input '{input_name_from_keras}' matched as CONTEXT based on feeder key '{self.feeder_key_name_context}'.")
                    if len(input_shape_from_keras) == 2 and input_shape_from_keras[1] is not None: # Expect (None, context_dim)
                        self.context_dim_for_generator = input_shape_from_keras[1]
                        self.gan_context_input_keras_name_hint = input_name_from_keras # Store Keras name
                        self.logger.info(f"    Set for GAN (from Gen): context_dim_for_generator = {self.context_dim_for_generator} (2D). Keras name hint: {self.gan_context_input_keras_name_hint}")
                    # Add handling for 3D context data if necessary, e.g. (None, seq_len, context_features_per_step)
                    elif len(input_shape_from_keras) == 3 and input_shape_from_keras[-1] is not None: # Expect (None, seq_len, context_features_per_step)
                        self.context_dim_for_generator = input_shape_from_keras[-1] # Features per step
                        # self.context_seq_len_for_generator = input_shape_from_keras[1] # If context also has a sequence length
                        self.gan_context_input_keras_name_hint = input_name_from_keras
                        self.logger.info(f"    Set for GAN (from Gen): context_dim_for_generator = {self.context_dim_for_generator} (from 3D shape {input_shape_from_keras}). Keras name hint: {self.gan_context_input_keras_name_hint}")
                    else:
                        self.logger.warning(f"  Context input '{input_name_from_keras}' has unexpected shape {input_shape_from_keras} or undefined feature dim. Using config/default for context_dim.")
                    found_context = True
            
            # Fallback if roles couldn't be matched by name (e.g., generic Keras input names like "input_1")
            # This is more heuristic and depends on the order and number of inputs.
            # Assumes a common order: latent, then conditional (if any), then context (if any)
            # This part needs careful consideration based on typical generator structures.
            if not found_latent and len(self.generator.inputs) > 0:
                # Assume first input is latent if not otherwise identified
                keras_input_tensor = self.generator.inputs[0]
                input_name_from_keras = keras_input_tensor.name
                input_shape_from_keras = keras_input_tensor.shape
                if len(input_shape_from_keras) == 2 and input_shape_from_keras[1] is not None:
                    self.latent_dim_for_generator = input_shape_from_keras[1]
                    self.gan_latent_input_keras_name_hint = input_name_from_keras
                    self.logger.info(f"Fallback: Assuming first input '{input_name_from_keras}' is LATENT. Set latent_dim_for_generator = {self.latent_dim_for_generator}. Keras name hint: {self.gan_latent_input_keras_name_hint}")
                    found_latent = True # Mark as found for subsequent checks
                else:
                    self.logger.warning(f"Fallback: First input '{input_name_from_keras}' (shape {input_shape_from_keras}) does not look like a typical latent vector. Latent dim might be incorrect.")
            
            # Example for conditional if not found by name and more than one input exists
            if not found_conditional and self.params.get('conditional_dim') is not None and len(self.generator.inputs) > (1 if found_latent else 0) :
                # Try to find an input that matches the configured conditional_dim
                # This is tricky because multiple inputs could have the same dimension.
                # This logic might need to be more robust or rely on stricter naming/ordering.
                self.logger.info(f"Attempting fallback for CONDITIONAL input based on configured 'conditional_dim': {self.params.get('conditional_dim')}")
                # This part is complex and might need more sophisticated matching.
                # For now, if conditional_dim is configured, we assume it's correct and will be handled by GAN builder.
                # We might not be able to get a Keras name hint here reliably without more info.
                pass # Rely on configured conditional_dim if not found by name

            if not found_context and self.params.get('context_dim') is not None and len(self.generator.inputs) > ( (1 if found_latent else 0) + (1 if found_conditional and self.params.get('conditional_dim') is not None else 0) ):
                self.logger.info(f"Attempting fallback for CONTEXT input based on configured 'context_dim': {self.params.get('context_dim')}")
                # Similar to conditional, relying on config if not found by name.
                pass

        else: # No generator model or no inputs
            self.logger.info("Generator model or its inputs not available. GAN input dimensions will be based purely on config (latent_dim, conditional_dim, context_dim).")
            # If conditional_dim is not directly in params, try to calculate it
            if self.conditional_dim_for_generator is None:
                num_date_feats = len(self.params.get("feeder_date_feature_names_for_conditioning", []))
                num_fund_feats = len(self.params.get("conditional_fundamental_feature_names", []))
                num_prev_tick_feats = self.params.get("num_conditional_prev_tick_features", 0)
                calculated_cond_dim_from_parts = num_date_feats + num_fund_feats + num_prev_tick_feats
                if calculated_cond_dim_from_parts > 0:
                    self.conditional_dim_for_generator = calculated_cond_dim_from_parts
                    self.logger.info(f"Derived 'conditional_dim_for_generator' ({self.conditional_dim_for_generator}) from sum of configured feature name list lengths (generator not available).")
                else:
                    self.logger.info("'conditional_dim_for_generator' is None and could not be derived from feature name lists (generator not available).")


        # Final check and logging for generator inputs for the GAN
        self.logger.info(f"Final generator input parameters for GAN construction: latent_dim={self.latent_dim_for_generator} (Keras hint: {self.gan_latent_input_keras_name_hint}), conditional_dim={self.conditional_dim_for_generator} (Keras hint: {self.gan_conditional_input_keras_name_hint}), context_dim={self.context_dim_for_generator} (Keras hint: {self.gan_context_input_keras_name_hint})")
        if self.conditional_dim_for_generator is not None and self.conditional_dim_for_generator <=0:
            self.logger.warning(f"conditional_dim_for_generator is {self.conditional_dim_for_generator}. If conditioning is intended, this should be > 0.")
            # self.conditional_dim_for_generator = None # Optionally reset if 0 or less and that implies no conditioning

        if self.context_dim_for_generator is not None and self.context_dim_for_generator <=0:
            self.logger.warning(f"context_dim_for_generator is {self.context_dim_for_generator}. If context is intended, this should be > 0.")
            # self.context_dim_for_generator = None # Optionally reset

        # --- Discriminator Input Parameters ---
        # Discriminator primarily takes the generator's output (seq_len, num_base_features)
        # And potentially conditional data (conditional_dim_for_discriminator)
        # And potentially context data (context_dim_for_discriminator)

        self.conditional_dim_for_discriminator = self.params.get('discriminator_conditional_dim', self.conditional_dim_for_generator)
        self.logger.info(f"Discriminator conditional_dim set to: {self.conditional_dim_for_discriminator} (from 'discriminator_conditional_dim' or defaulted to generator's conditional_dim)")
        
        self.context_dim_for_discriminator = self.params.get('discriminator_context_dim', self.context_dim_for_generator) # Discriminator might also use context
        self.logger.info(f"Discriminator context_dim set to: {self.context_dim_for_discriminator} (from 'discriminator_context_dim' or defaulted to generator's context_dim)")

        if self.conditional_dim_for_discriminator is not None and self.conditional_dim_for_discriminator <= 0:
            self.logger.info("discriminator_conditional_dim is <= 0. Setting to None, implying no explicit conditioning for discriminator beyond what's in its main input.")
            self.conditional_dim_for_discriminator = None
        
        if self.context_dim_for_discriminator is not None and self.context_dim_for_discriminator <= 0:
            self.logger.info("discriminator_context_dim is <= 0. Setting to None, implying no explicit context input for discriminator beyond what's in its main input.")
            self.context_dim_for_discriminator = None

        self.logger.info(f"Core parameters initialized: seq_len={self.seq_len}, num_base_features={self.num_base_features}, latent_dim={self.latent_dim_for_generator}, cond_dim_gen={self.conditional_dim_for_generator}, ctx_dim_gen={self.context_dim_for_generator}, cond_dim_disc={self.conditional_dim_for_discriminator}, ctx_dim_disc={self.context_dim_for_discriminator}")
        self.logger.info("_initialize_core_parameters_from_config completed.")

    def _build_generator(self):
