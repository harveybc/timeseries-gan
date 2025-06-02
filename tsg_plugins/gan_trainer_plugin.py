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
        self.logger.info("Initializing core GAN parameters from config and loaded models...")

        # --- Generator Output Parameters (used by Discriminator construction) ---
        self.generator_output_actual_seq_len = None
        self.generator_output_actual_features = None
        self.logger.info("Attempting to determine generator output dimensions...")

        if self.generator:
            self.logger.info(f"Generator model is present. Type: {type(self.generator)}")
            output_shape_from_model = None
            output_features_from_model = None
            
            # Try direct .output_shape attribute
            if hasattr(self.generator, 'output_shape'):
                self.logger.info(f"Generator has 'output_shape' attribute: {self.generator.output_shape}")
                if isinstance(self.generator.output_shape, tuple) and len(self.generator.output_shape) == 3:
                    if isinstance(self.generator.output_shape[1], int) and isinstance(self.generator.output_shape[2], int):
                        output_shape_from_model = self.generator.output_shape[1]
                        output_features_from_model = self.generator.output_shape[2]
                        self.logger.info(f"Derived from generator.output_shape (tuple): seq_len={output_shape_from_model}, features={output_features_from_model}")
                elif isinstance(self.generator.output_shape, list) and \
                     self.generator.output_shape and \
                     isinstance(self.generator.output_shape[0], tuple) and \
                     len(self.generator.output_shape[0]) == 3:
                    if isinstance(self.generator.output_shape[0][1], int) and isinstance(self.generator.output_shape[0][2], int):
                        output_shape_from_model = self.generator.output_shape[0][1]
                        output_features_from_model = self.generator.output_shape[0][2]
                        self.logger.info(f"Derived from generator.output_shape (list of tuples): seq_len={output_shape_from_model}, features={output_features_from_model}")
                else:
                    self.logger.info("Generator 'output_shape' is not a 3D tuple or list of 3D tuples of integers.")

            # If not found, try .output.shape (Keras functional API tensor)
            if output_shape_from_model is None and hasattr(self.generator, 'output'):
                self.logger.info("Generator has 'output' attribute.")
                if hasattr(self.generator.output, 'shape') and len(self.generator.output.shape) == 3:
                    # Ensure dimensions are integers, not None or other types
                    if isinstance(self.generator.output.shape[1], int) and isinstance(self.generator.output.shape[2], int):
                        output_shape_from_model = self.generator.output.shape[1]
                        output_features_from_model = self.generator.output.shape[2]
                        self.logger.info(f"Derived from generator.output.shape: seq_len={output_shape_from_model}, features={output_features_from_model}")
                    else:
                        self.logger.info(f"Generator 'output.shape' is 3D but dimensions are not integers: {self.generator.output.shape}")
                else:
                    self.logger.info(f"Generator 'output.shape' is not 3D: {getattr(self.generator.output, 'shape', 'N/A')}")
            elif output_shape_from_model is None and not hasattr(self.generator, 'output_shape'):
                 self.logger.info("Generator does not have 'output_shape' attribute.")
            elif output_shape_from_model is None : # Already checked output_shape, now checking if 'output' attr exists
                self.logger.info("Generator does not have 'output' attribute or 'output_shape' was not suitable.")


            if output_shape_from_model is not None and output_features_from_model is not None:
                self.generator_output_actual_seq_len = output_shape_from_model
                self.generator_output_actual_features = output_features_from_model
                self.logger.info(f"Successfully set generator output dims from model: seq_len={self.generator_output_actual_seq_len}, features={self.generator_output_actual_features}")
            else:
                self.logger.warning("Could not determine generator output seq_len/features from the model structure.")
        else:
            self.logger.warning("Generator model (self.generator) is None. Cannot derive output dimensions from model.")

        # Fallback to config if model-derived values are not available
        config_seq_len = self.params.get('gan_generator_output_actual_seq_len', self.params.get('seq_len'))
        self.logger.info(f"Config fallback for seq_len: 'gan_generator_output_actual_seq_len'={self.params.get('gan_generator_output_actual_seq_len')}, 'seq_len'={self.params.get('seq_len')}. Chosen: {config_seq_len}")
        if self.generator_output_actual_seq_len is None and config_seq_len is not None:
            self.generator_output_actual_seq_len = int(config_seq_len) # Ensure it's an int
            self.logger.info(f"Using config for generator_output_actual_seq_len: {self.generator_output_actual_seq_len}")

        config_features_gan = self.params.get('gan_generator_output_actual_features')
        config_features_base = self.params.get('num_base_features_generated')
        config_features_dim = self.params.get('feature_dim')
        self.logger.info(f"Config fallbacks for features: 'gan_generator_output_actual_features'={config_features_gan}, 'num_base_features_generated'={config_features_base}, 'feature_dim'={config_features_dim}")

        chosen_config_features = None
        if config_features_gan is not None:
            chosen_config_features = config_features_gan
        elif config_features_base is not None:
            chosen_config_features = config_features_base
        elif config_features_dim is not None:
            chosen_config_features = config_features_dim
        
        self.logger.info(f"Chosen config features (before plugin check): {chosen_config_features}")

        if self.generator_output_actual_features is None and chosen_config_features is not None:
            self.generator_output_actual_features = int(chosen_config_features) # Ensure it's an int
            self.logger.info(f"Using config for generator_output_actual_features: {self.generator_output_actual_features}")
        elif self.generator_output_actual_features is None: # If still None, try GeneratorPlugin's params
            self.logger.info("Attempting to derive features from GeneratorPlugin's 'decoder_output_feature_names'.")
            if self.generator_plugin_instance and hasattr(self.generator_plugin_instance, 'params'):
                # Ensure params is a dict before calling .get()
                gen_plugin_params = self.generator_plugin_instance.params if isinstance(self.generator_plugin_instance.params, dict) else {}
                
                decoder_output_names = gen_plugin_params.get('generator_decoder_output_feature_names',
                                                             gen_plugin_params.get('decoder_output_feature_names', []))
                self.logger.info(f"GeneratorPlugin 'decoder_output_feature_names' (or alias): {decoder_output_names}")
                if decoder_output_names and isinstance(decoder_output_names, list) and len(decoder_output_names) > 0:
                    self.generator_output_actual_features = len(decoder_output_names)
                    self.logger.info(f"Derived generator_output_actual_features ({self.generator_output_actual_features}) from GeneratorPlugin's 'decoder_output_feature_names'.")
                else:
                    self.logger.info("GeneratorPlugin's 'decoder_output_feature_names' is empty, not a list, or not found.")
            else:
                self.logger.info("No generator_plugin_instance or it lacks 'params' attribute or params is not a dict.")
        
        # Critical check: Ensure these are now set, otherwise GAN/Discriminator cannot be built.
        error_messages = []
        if self.generator_output_actual_seq_len is None:
            error_messages.append("Could not determine generator output sequence length (checked model, params: 'gan_generator_output_actual_seq_len', 'seq_len').")
        if self.generator_output_actual_features is None:
            error_messages.append("Could not determine generator output features (checked model, params: 'gan_generator_output_actual_features', 'num_base_features_generated', 'feature_dim', or 'generator_decoder_output_feature_names' from GeneratorPlugin).")

        if error_messages:
            full_error_msg = "Critical: " + " ".join(error_messages) + \
                             " Please ensure the generator Keras model is loaded correctly and has defined outputs, OR provide these values in the trainer configuration: \n" + \
                             "  - 'gan_generator_output_actual_seq_len' or 'seq_len' (for sequence length)\n" + \
                             "  - 'gan_generator_output_actual_features', 'num_base_features_generated', 'feature_dim', or ensure 'generator_decoder_output_feature_names' is set in the generator's specific config section (for features)."
            self.logger.error(full_error_msg)
            raise ValueError(full_error_msg)

        # Assign to self.seq_len and self.num_base_features as these are used elsewhere
        self.seq_len = self.generator_output_actual_seq_len 
        self.num_base_features = self.generator_output_actual_features
        self.logger.info(f"Final generator output dimensions set: self.seq_len={self.seq_len}, self.num_base_features={self.num_base_features}")

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

                # Try to match based on configured feeder key names to the generator's input layers by their Keras names
                # This assumes Keras input layers were named according to these feeder keys or a convention.
                # Example Keras input layer names: "input_latent_vector", "input_conditional_data", "input_context_vector"
                # Or, if not named, rely on the order and config params for dimensions.

                # Check for Latent Vector
                # The feeder_key_name_latent (e.g., "latent_vector") is what the FeederPlugin provides.
                # The Keras input layer might be named "input_latent_vector", "latent_input", or just self.feeder_key_name_latent.
                if self.feeder_key_name_latent and (self.feeder_key_name_latent in input_name_from_keras.lower() or f"input_{self.feeder_key_name_latent}" in input_name_from_keras.lower()):
                    self.logger.info(f"  Input '{input_name_from_keras}' matched as LATENT based on feeder key '{self.feeder_key_name_latent}'.")
                    if len(input_shape_from_keras) == 2: # (batch, features)
                        self.gen_input_latent_dim = input_shape_from_keras[1]
                        self.gen_input_seq_len = None # Latent vector is not sequential
                        self.gan_latent_input_keras_name_hint = input_name_from_keras # Store Keras name
                        self.logger.info(f"    Set for GAN (from Gen): gen_input_latent_dim = {self.gen_input_latent_dim}, gen_input_seq_len = None (2D). Keras name hint: {self.gan_latent_input_keras_name_hint}")
                    elif len(input_shape_from_keras) == 3: # (batch, seq_len, features)
                        self.gen_input_seq_len = input_shape_from_keras[1]
                        self.gen_input_latent_dim = input_shape_from_keras[2]
                        self.gan_latent_input_keras_name_hint = input_name_from_keras # Store Keras name
                        self.logger.info(f"    Set for GAN (from Gen): gen_input_latent_dim = {self.gen_input_latent_dim}, gen_input_seq_len = {self.gen_input_seq_len} (3D). Keras name hint: {self.gan_latent_input_keras_name_hint}")
                    else:
                        self.logger.warning(f"  Latent input '{input_name_from_keras}' has unexpected shape {input_shape_from_keras}. Using config value for latent_dim.")
                    found_latent = True
                
                # Check for Conditional Data
                elif self.feeder_key_name_conditional and (self.feeder_key_name_conditional in input_name_from_keras.lower() or f"input_{self.feeder_key_name_conditional}" in input_name_from_keras.lower()):
                    self.logger.info(f"  Input '{input_name_from_keras}' matched as CONDITIONAL based on feeder key '{self.feeder_key_name_conditional}'.")
                    if len(input_shape_from_keras) == 2 and input_shape_from_keras[1] is not None:
                        self.conditional_dim_for_generator = input_shape_from_keras[1]
                        self.gan_conditional_input_keras_name_hint = input_name_from_keras # Store Keras name
                        self.logger.info(f"    Set for GAN (from Gen): conditional_dim_for_generator = {self.conditional_dim_for_generator} (2D). Keras name hint: {self.gan_conditional_input_keras_name_hint}")
                    # Add handling for 3D conditional data if necessary, e.g. if it has a sequence length
                    # elif len(input_shape_from_keras) == 3 and input_shape_from_keras[1] is not None and input_shape_from_keras[2] is not None:
                    #     self.conditional_seq_len_for_generator = input_shape_from_keras[1]
                    #     self.conditional_dim_for_generator = input_shape_from_keras[2]
                    #     self.logger.info(f"    Set for GAN (from Gen): conditional_dim_for_generator = {self.conditional_dim_for_generator}, conditional_seq_len = {self.conditional_seq_len_for_generator} (3D)")
                    else:
                        self.logger.warning(f"  Conditional input '{input_name_from_keras}' has unexpected shape {input_shape_from_keras} or undefined feature dim. Using config/default for conditional_dim.")
                    found_conditional = True

                # Check for Context Vector
                elif self.feeder_key_name_context and (self.feeder_key_name_context in input_name_from_keras.lower() or f"input_{self.feeder_key_name_context}" in input_name_from_keras.lower()):
                    self.logger.info(f"  Input '{input_name_from_keras}' matched as CONTEXT based on feeder key '{self.feeder_key_name_context}'.")
                    if len(input_shape_from_keras) == 2 and input_shape_from_keras[1] is not None:
                        self.context_dim_for_generator = input_shape_from_keras[1]
                        self.gan_context_input_keras_name_hint = input_name_from_keras # Store Keras name
                        self.logger.info(f"    Set for GAN (from Gen): context_dim_for_generator = {self.context_dim_for_generator} (2D). Keras name hint: {self.gan_context_input_keras_name_hint}")
                    # Add handling for 3D context data if necessary
                    else:
                        self.logger.warning(f"  Context input '{input_name_from_keras}' has unexpected shape {input_shape_from_keras} or undefined feature dim. Using config/default for context_dim.")
                    found_context = True
            
            # Fallback if roles couldn't be matched by name (e.g., generic Keras input names like "input_1")
            # This part is tricky and relies on assumptions about the order of inputs if names are not descriptive.
            # The current strategy is: if not found by name, stick to config values. 
            # A more robust fallback might try to infer based on typical dimensionalities if there's a standard order.
            if not found_latent:
                self.logger.warning(f"Could not identify LATENT input layer for generator by name matching feeder key '{self.feeder_key_name_latent}'. Will rely on config value for 'latent_dim' ({self.gen_input_latent_dim}) for GAN construction.")
            
            if not found_conditional:
                self.logger.warning(f"Could not identify CONDITIONAL input layer for generator by name matching feeder key '{self.feeder_key_name_conditional}'.")
                if self.conditional_dim_for_generator is None: # Check if it's still None
                    num_date_feats = len(self.params.get("feeder_date_feature_names_for_conditioning", []))
                    num_fund_feats = len(self.params.get("conditional_fundamental_feature_names", []))
                    num_prev_tick_feats = self.params.get("num_conditional_prev_tick_features", 0)
                    calculated_cond_dim_from_parts = num_date_feats + num_fund_feats + num_prev_tick_feats
                    
                    if calculated_cond_dim_from_parts > 0:
                        self.conditional_dim_for_generator = calculated_cond_dim_from_parts
                        self.logger.info(f"Set 'conditional_dim_for_generator' to {self.conditional_dim_for_generator} based on the sum of configured conditional feature parts, as 'conditional_dim' was not explicitly in params and not model-derived.")
            
            if not found_context:
                self.logger.warning(f"Could not identify CONTEXT input layer for generator by name matching feeder key '{self.feeder_key_name_context}'. Will rely on config value for 'context_dim' ({self.context_dim_for_generator}) for GAN construction.")
                # Similar fallback for context_dim could be added here if needed and if self.context_dim_for_generator is None

            # Ensure essential dimensions are set for GAN building
            if self.gen_input_latent_dim is None:
                self.logger.error(f"Critical: gen_input_latent_dim is None after attempting to derive from model and config. This is required for GAN input. Check generator model input for latent vector and 'latent_dim' in config.")
                raise ValueError("Generator latent input dimension (gen_input_latent_dim) could not be determined.")
            
            if self.conditional_dim_for_generator is None:
                self.logger.info("conditional_dim_for_generator is None. GAN will be built without a dedicated conditional input (either not configured via 'conditional_dim', model-derived, or sum of parts is 0).")
            else:
                self.logger.info(f"Final 'conditional_dim_for_generator': {self.conditional_dim_for_generator}")

            if self.context_dim_for_generator is None:
                self.logger.info("context_dim_for_generator is None. GAN will be built without a dedicated context input (not configured via 'context_dim' or model-derived).")
            else:
                self.logger.info(f"Final 'context_dim_for_generator': {self.context_dim_for_generator}")

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
        self.discriminator_input_seq_len = self.generator_output_actual_seq_len # This is self.seq_len
        
        # self.num_features_for_discriminator was determined earlier based on config or estimation
        # self.num_tis = self.params.get('num_tis', 0) # Number of TI features if added separately, not by TensorFlowTALayer
        # If TensorFlowTALayer is used, self.num_features_for_discriminator should be the total (base + TIs)
        # If TIs are pre-calculated and part of input data, then self.num_features_for_discriminator is that total.
        # If TIs are calculated *inside* the discriminator by a layer, then discriminator input is just base features.
        # The current TensorFlowTALayer is designed to be ADDED to the model, so discriminator input should be base_features,
        # and the layer outputs base_features + TIs.
        # For now, let's assume self.num_features_for_discriminator is the final number of features the Conv1D in discriminator sees.

        # self.discriminator_input_feature_dim = self.num_features_for_discriminator + self.num_tis # This was potentially double counting if TIs are included in num_features_for_discriminator
        self.discriminator_input_feature_dim = self.num_features_for_discriminator # This should be the total number of features the discriminator's main layers process.
        
        self.logger.info(f"Discriminator input parameters: seq_len={self.discriminator_input_seq_len}, feature_dim={self.discriminator_input_feature_dim}")


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

    def _get_scaled_date_features(self, datetimes_series: pd.Series) -> np.ndarray:
        """
        Generates scaled date/time features from a pandas Series of datetime objects.
        Features are scaled to be generally within the 0-1 range.
        Uses 'feeder_date_feature_names_for_conditioning' and 'feeder_max_...' from self.params.
        """
        if datetimes_series is None or datetimes_series.empty:
            # Determine expected number of features to return an empty array of correct width
            # Ensure param name matches what's used in plugin_params
            feature_names_for_conditioning = self.params.get("feeder_date_feature_names_for_conditioning", [])
            num_expected_features = len(feature_names_for_conditioning)
            return np.empty((0, num_expected_features), dtype=float)

        # Ensure param name matches what's used in plugin_params
        feature_names_for_conditioning = self.params.get("feeder_date_feature_names_for_conditioning", [])
        if not feature_names_for_conditioning:
            return np.array([]).reshape(len(datetimes_series), 0)

        all_scaled_features = []
        num_samples = len(datetimes_series)

        # Ensure dt accessor is available (i.e., series contains datetime-like objects)
        if not pd.api.types.is_datetime64_any_dtype(datetimes_series):
            try:
                datetimes_series = pd.to_datetime(datetimes_series)
            except Exception as e:
                self.logger.error(f"_get_scaled_date_features: Could not convert series to datetime: {e}. Returning empty.")
                return np.empty((num_samples, len(feature_names_for_conditioning) if feature_names_for_conditioning else 0), dtype=float)


        if "day_of_week" in feature_names_for_conditioning:
            # pandas dayofweek is 0 (Mon) to 6 (Sun). Max value for scaling should be 6.0 if 0-6.
            # If config "feeder_max_day_of_week" is 7.0, it implies input is 1-7 or 0-6 and we want to scale to [0, 6/6=1] or [0, (7-1)/(7-1)=1]
            max_val = self.params.get("feeder_max_day_of_week", 7.0) # Default to 7 (e.g. for 1-7 days)
            # If pandas dayofweek (0-6) is used, scaling should be against 6.
            # Let's assume max_val in config is the number of unique values (e.g., 7 for days of week).
            # So, if values are 0-6, then (value) / (max_val - 1)
            actual_max_for_0_based = max_val - 1.0
            scaled_val = datetimes_series.dt.dayofweek.values / actual_max_for_0_based if actual_max_for_0_based > 0 else np.zeros(num_samples)
            all_scaled_features.append(scaled_val)

        if "day_of_month" in feature_names_for_conditioning:
            # pandas day is 1-31. Max value for scaling should be 31.0.
            # If config "feeder_max_day_of_month" is 31.0.
            # Scaled: (day - 1) / (max_val - 1)
            max_val = self.params.get("feeder_max_day_of_month", 31.0)
            actual_max_for_1_based = max_val -1.0 # (31-1) = 30 for range 0-30
            scaled_val = (datetimes_series.dt.day.values - 1.0) / actual_max_for_1_based if actual_max_for_1_based > 0 else np.zeros(num_samples)
            all_scaled_features.append(scaled_val)

        if "hour_of_day" in feature_names_for_conditioning:
            # pandas hour is 0-23. Max value for scaling should be 23.0.
            # If config "feeder_max_hour_of_day" is 24.0 (number of unique values).
            # Scaled: hour / (max_val - 1)
            max_val = self.params.get("feeder_max_hour_of_day", 24.0)
            actual_max_for_0_based = max_val - 1.0 # (24-1) = 23 for range 0-23
            scaled_val = datetimes_series.dt.hour.values / actual_max_for_0_based if actual_max_for_0_based > 0 else np.zeros(num_samples)
            all_scaled_features.append(scaled_val)

        if "minute_of_hour" in feature_names_for_conditioning:
            max_val = self.params.get("feeder_max_minute_of_hour", 60.0)
            actual_max_for_0_based = max_val - 1.0
            scaled_val = datetimes_series.dt.minute.values / actual_max_for_0_based if actual_max_for_0_based > 0 else np.zeros(num_samples)
            all_scaled_features.append(scaled_val)

        if "second_of_minute" in feature_names_for_conditioning:
            max_val = self.params.get("feeder_max_second_of_minute", 60.0)
            actual_max_for_0_based = max_val - 1.0
            scaled_val = datetimes_series.dt.second.values / actual_max_for_0_based if actual_max_for_0_based > 0 else np.zeros(num_samples)
            all_scaled_features.append(scaled_val)
            
        if "day_of_year" in feature_names_for_conditioning: # Added
            max_val = self.params.get("feeder_max_day_of_year", 366.0) # For 1-366
            actual_max_for_1_based = max_val - 1.0
            scaled_val = (datetimes_series.dt.dayofyear.values - 1.0) / actual_max_for_1_based if actual_max_for_1_based > 0 else np.zeros(num_samples)
            all_scaled_features.append(scaled_val)

        if "is_weekend" in feature_names_for_conditioning:
            is_weekend_val = (datetimes_series.dt.dayofweek >= 5).astype(float) # Saturday=5, Sunday=6
            all_scaled_features.append(is_weekend_val)

        if not all_scaled_features:
            return np.array([]).reshape(num_samples, 0)
        
        stacked_features = np.stack(all_scaled_features, axis=-1)
        
        # Ensure the number of generated features matches the length of feature_names_for_conditioning
        if stacked_features.shape[1] != len(feature_names_for_conditioning):
            self.logger.warning(f"_get_scaled_date_features: Mismatch in generated date features ({stacked_features.shape[1]}) vs. requested ({len(feature_names_for_conditioning)}). This may be due to some requested features not being implemented. Padding/truncating.")
            # Create a correctly shaped array of zeros
            correctly_shaped_features = np.zeros((num_samples, len(feature_names_for_conditioning)))
            # Fill with generated features, truncating or leaving zeros if mismatch
            num_to_copy = min(stacked_features.shape[1], len(feature_names_for_conditioning))
            if num_to_copy > 0:
                correctly_shaped_features[:, :num_to_copy] = stacked_features[:, :num_to_copy]
            return correctly_shaped_features
            
        return stacked_features

    def _get_scaled_fundamental_features(self, fundamental_features_df_batch: pd.DataFrame) -> np.ndarray:
        """
        Extracts and returns specified fundamental features.
        Assumes features in fundamental_features_df_batch are already scaled (e.g., 0-1).
        Uses 'conditional_fundamental_feature_names' from self.params.
        """
        if fundamental_features_df_batch is None or fundamental_features_df_batch.empty:
            num_expected_features = len(self.params.get("conditional_fundamental_feature_names", []))
            return np.empty((0, num_expected_features), dtype=float)

        feature_names = self.params.get("conditional_fundamental_feature_names", [])
        if not feature_names:
            return np.array([]).reshape(len(fundamental_features_df_batch), 0)

        # Select the specified columns. Ensure they exist.
        missing_cols = [col for col in feature_names if col not in fundamental_features_df_batch.columns]
        if missing_cols:
            self.logger.warning(f"_get_scaled_fundamental_features: Missing fundamental columns: {missing_cols}. Returning zeros for them.")
            # Create a result array with zeros
            result_array = np.zeros((len(fundamental_features_df_batch), len(feature_names)))
            # Fill in existing columns
            for i, name in enumerate(feature_names):
                if name in fundamental_features_df_batch.columns:
                    result_array[:, i] = fundamental_features_df_batch[name].values
            return result_array
            
        return fundamental_features_df_batch[feature_names].values


    def _get_data_generator(self, 
                            x_real_processed_np_epoch: np.ndarray, 
                            real_datetimes_series_epoch: Optional[pd.Series],
                            fundamental_features_df_epoch: Optional[pd.DataFrame],
                            epoch_context_vectors_aligned: Optional[np.ndarray], # NEW ARGUMENT
                            batch_size: int):
        """
        Data generator that yields batches of real samples, corresponding conditional data,
        and corresponding context data (if available/configured).
        Assumes input data is already shuffled for the epoch.
        """
        num_samples = x_real_processed_np_epoch.shape[0]
        indices = np.arange(num_samples)

        date_feature_names = self.params.get("feeder_date_feature_names_for_conditioning", [])
        fund_feature_names = self.params.get("conditional_fundamental_feature_names", [])
        num_prev_tick_features = self.params.get("num_conditional_prev_tick_features", 0)
        
        expected_total_conditional_features = self.conditional_dim_for_generator if self.conditional_dim_for_generator is not None else 0

        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:i + batch_size]
            actual_batch_size = len(batch_indices)
            if actual_batch_size == 0: continue

            real_samples_batch = x_real_processed_np_epoch[batch_indices]

            all_conditional_components = []
            num_actual_date_features = 0
            if date_feature_names:
                if real_datetimes_series_epoch is not None and not real_datetimes_series_epoch.empty:
                    batch_datetimes = real_datetimes_series_epoch.iloc[batch_indices]
                    if not batch_datetimes.empty:
                        scaled_date_features_batch = self._get_scaled_date_features(batch_datetimes)
                        if scaled_date_features_batch is not None and scaled_date_features_batch.shape[0] == actual_batch_size:
                            all_conditional_components.append(scaled_date_features_batch)
                            num_actual_date_features = scaled_date_features_batch.shape[1]
                        else:
                            self.logger.warning(f"_get_data_generator: _get_scaled_date_features issue. Shape: {scaled_date_features_batch.shape if scaled_date_features_batch is not None else 'None'}")
                if num_actual_date_features != len(date_feature_names):
                    all_conditional_components.append(np.zeros((actual_batch_size, len(date_feature_names))))
                    num_actual_date_features = len(date_feature_names)

            num_actual_fund_features = 0
            if fund_feature_names:
                if fundamental_features_df_epoch is not None and not fundamental_features_df_epoch.empty:
                    batch_fundamentals_df = fundamental_features_df_epoch.iloc[batch_indices]
                    if not batch_fundamentals_df.empty:
                        scaled_fund_features_batch = self._get_scaled_fundamental_features(batch_fundamentals_df)
                        if scaled_fund_features_batch is not None and scaled_fund_features_batch.shape[0] == actual_batch_size:
                            all_conditional_components.append(scaled_fund_features_batch)
                            num_actual_fund_features = scaled_fund_features_batch.shape[1]
                        else:
                             self.logger.warning(f"_get_data_generator: _get_scaled_fundamental_features issue. Shape: {scaled_fund_features_batch.shape if scaled_fund_features_batch is not None else 'None'}")
                if num_actual_fund_features != len(fund_feature_names):
                    all_conditional_components.append(np.zeros((actual_batch_size, len(fund_feature_names))))
                    num_actual_fund_features = len(fund_feature_names)

            num_actual_prev_tick_features = 0
            if num_prev_tick_features > 0:
                if real_samples_batch.shape[2] >= num_prev_tick_features:
                    prev_tick_features_batch = real_samples_batch[:, 0, :num_prev_tick_features]
                    all_conditional_components.append(prev_tick_features_batch)
                    num_actual_prev_tick_features = prev_tick_features_batch.shape[1]
                else:
                    self.logger.warning(f"_get_data_generator: Not enough features in real_samples_batch ({real_samples_batch.shape[2]}) to extract {num_prev_tick_features} prev_tick features. Padding.")
                    all_conditional_components.append(np.zeros((actual_batch_size, num_prev_tick_features)))
                    num_actual_prev_tick_features = num_prev_tick_features

            if all_conditional_components:
                final_conditional_batch = np.concatenate(all_conditional_components, axis=1)
            else: 
                final_conditional_batch = np.zeros((actual_batch_size, 0))

            current_total_cond_features = final_conditional_batch.shape[1]
            if current_total_cond_features != expected_total_conditional_features:
                self.logger.debug(f"_get_data_generator: Current conditional features {current_total_cond_features}, expected {expected_total_conditional_features}. Adjusting.")
                adjusted_batch = np.zeros((actual_batch_size, expected_total_conditional_features))
                copy_cols = min(current_total_cond_features, expected_total_conditional_features)
                if copy_cols > 0:
                    adjusted_batch[:, :copy_cols] = final_conditional_batch[:, :copy_cols]
                final_conditional_batch = adjusted_batch
            
            context_batch_for_yield: Optional[np.ndarray] = None
            if self.context_dim_for_generator is not None and self.context_dim_for_generator > 0:
                if epoch_context_vectors_aligned is not None:
                    context_batch_for_yield = epoch_context_vectors_aligned[batch_indices]
                    if (context_batch_for_yield.shape[0] != actual_batch_size or
                        context_batch_for_yield.shape[1] != self.context_dim_for_generator):
                        self.logger.warning(f"_get_data_generator: Context vector batch shape mismatch. Expected ({actual_batch_size}, {self.context_dim_for_generator}), got {context_batch_for_yield.shape}. Using zeros for this batch.")
                        context_batch_for_yield = np.zeros((actual_batch_size, self.context_dim_for_generator))
                else:
                    context_batch_for_yield = np.zeros((actual_batch_size, self.context_dim_for_generator))
            
            yield real_samples_batch, final_conditional_batch, context_batch_for_yield

    def train(self, x_real_df: pd.DataFrame, start_epoch: int = 0):
        self.logger.info("Starting GAN training process...")

        epochs = self.params.get("gan_epochs", 10000)
        batch_size = self.params.get("gan_batch_size", 32)
        save_interval = self.params.get("gan_save_interval", 500)
        datetime_col_name = self.params.get("datetime_col_name_in_x_real_df", "DATE_TIME")

        if self.generator is None or self.discriminator is None or self.gan_model is None:
            self.logger.error("train: Models not initialized. Call set_params first.")
            return {"status": "error", "message": "Models not initialized."}

        if x_real_df is None or x_real_df.empty:
            self.logger.error("train: Real data (x_real_df) is None or empty.")
            return {"status": "error", "message": "Input data is missing."}

        # --- Data Preparation ---
        # Extract datetimes first
        if datetime_col_name not in x_real_df.columns:
            self.logger.error(f"train: Datetime column '{datetime_col_name}' not found in x_real_df.")
            return {"status": "error", "message": f"Datetime column '{datetime_col_name}' missing."}
        
        x_real_df_datetimes_original = pd.to_datetime(x_real_df[datetime_col_name])

        # Extract features for discriminator input
        discriminator_input_cols = self.params.get("discriminator_input_feature_names", [])
        if not discriminator_input_cols:
            self.logger.error("train: 'discriminator_input_feature_names' not configured.")
            return {"status": "error", "message": "'discriminator_input_feature_names' missing."}
        
        missing_disc_cols = [col for col in discriminator_input_cols if col not in x_real_df.columns]
        if missing_disc_cols:
            self.logger.error(f"train: Missing discriminator input columns in x_real_df: {missing_disc_cols}")
            return {"status": "error", "message": f"Missing discriminator input columns: {missing_disc_cols}"}
            
        real_data_df_values = x_real_df[discriminator_input_cols].fillna(0).values # For discriminator sequences

        # Extract fundamental features for conditional input
        conditional_fundamental_cols = self.params.get("conditional_fundamental_feature_names", [])
        conditional_fundamental_df = None
        if conditional_fundamental_cols:
            missing_fund_cols = [col for col in conditional_fundamental_cols if col not in x_real_df.columns]
            if missing_fund_cols:
                self.logger.warning(f"train: Missing conditional fundamental columns in x_real_df: {missing_fund_cols}. They will be zero-padded in generator.")
                # Create a placeholder DataFrame with zeros if columns are missing, to maintain structure
                conditional_fundamental_df = pd.DataFrame(np.zeros((len(x_real_df), len(conditional_fundamental_cols))), columns=conditional_fundamental_cols)
            else:
                conditional_fundamental_df = x_real_df[conditional_fundamental_cols].fillna(0)
        
        num_total_timesteps = real_data_df_values.shape[0]
        
        if num_total_timesteps < self.seq_len:
            self.logger.error(f"Not enough data ({num_total_timesteps} timesteps) for sequences of length {self.seq_len}.")
            return {"status": "error", "message": "Not enough data for sequences."}

        x_real_sequences = []
        real_datetimes_sequences_list = []
        conditional_fundamental_sequences_list = []

        for i in range(num_total_timesteps - self.seq_len + 1):
            x_real_sequences.append(real_data_df_values[i : i + self.seq_len, :])
            # Datetime corresponds to the *last* observation in the sequence window
            real_datetimes_sequences_list.append(x_real_df_datetimes_original.iloc[i + self.seq_len - 1])
            if conditional_fundamental_df is not None:
                # Fundamental features also correspond to the *last* observation in the sequence window
                conditional_fundamental_sequences_list.append(conditional_fundamental_df.iloc[i + self.seq_len - 1].to_dict())
        
        if not x_real_sequences:
            self.logger.error("No sequences created. Cannot train.")
            return {"status": "error", "message": "No sequences after windowing."}

        x_real_processed_np = np.array(x_real_sequences)
        real_datetimes_series_for_generator = pd.Series(real_datetimes_sequences_list, dtype='datetime64[ns]')
        
        fundamental_features_df_for_generator: Optional[pd.DataFrame] = None
        if conditional_fundamental_sequences_list:
            fundamental_features_df_for_generator = pd.DataFrame(conditional_fundamental_sequences_list)
            # Ensure column order matches config if created from dicts
            if conditional_fundamental_cols:
                 fundamental_features_df_for_generator = fundamental_features_df_for_generator[conditional_fundamental_cols]


        num_samples = x_real_processed_np.shape[0]
        self.logger.info(f"Prepared {num_samples} sequences of length {self.seq_len} for training.")

        # History tracking for losses (optional, for plotting)
        history_d_loss_real, history_d_loss_fake, history_g_loss = [], [], []

        # --- GAN Training Loop ---
        for epoch in range(start_epoch, epochs):
            epoch_start_time = time.time()
            
            # Shuffle data indices for each epoch
            shuffled_indices = np.arange(num_samples)
            np.random.shuffle(shuffled_indices)
            
            x_real_shuffled_epoch = x_real_processed_np[shuffled_indices]
            real_datetimes_shuffled_epoch = real_datetimes_series_for_generator.iloc[shuffled_indices].reset_index(drop=True)
            
            fundamental_features_shuffled_epoch: Optional[pd.DataFrame] = None
            if fundamental_features_df_for_generator is not None:
                fundamental_features_shuffled_epoch = fundamental_features_df_for_generator.iloc[shuffled_indices].reset_index(drop=True)

            epoch_context_vectors_np: Optional[np.ndarray] = None
            if self.context_dim_for_generator is not None and self.context_dim_for_generator > 0:
                if self.feeder_plugin and hasattr(self.feeder_plugin, 'get_aligned_context_vectors'):
                    try:
                        self.logger.info(f"Attempting to get context vectors from FeederPlugin for {num_samples} samples.")
                        epoch_context_vectors_np = self.feeder_plugin.get_aligned_context_vectors(real_datetimes_shuffled_epoch)
                        
                        if epoch_context_vectors_np is not None:
                            if epoch_context_vectors_np.shape[0] != num_samples:
                                self.logger.error(f"Context vectors shape mismatch from Feeder: got {epoch_context_vectors_np.shape[0]} samples, expected {num_samples}. Using zeros for epoch.")
                                epoch_context_vectors_np = None 
                            elif epoch_context_vectors_np.shape[1] != self.context_dim_for_generator:
                                self.logger.error(f"Context vectors dim mismatch from Feeder: got {epoch_context_vectors_np.shape[1]} dims, expected {self.context_dim_for_generator}. Using zeros for epoch.")
                                epoch_context_vectors_np = None
                            else:
                                self.logger.info(f"Successfully retrieved context vectors of shape {epoch_context_vectors_np.shape} from FeederPlugin for the epoch.")
                        else:
                            self.logger.warning("FeederPlugin.get_aligned_context_vectors returned None. Context vectors will be zeros if required by generator.")
                    except Exception as e:
                        self.logger.error(f"Error getting context vectors from FeederPlugin: {e}. Context vectors will be zeros if required by generator.", exc_info=True)
                        epoch_context_vectors_np = None
                else:
                    self.logger.info("Context vectors may be required (context_dim > 0) but FeederPlugin or 'get_aligned_context_vectors' method not available. Context vectors will be zeros if required by generator.")

            # Create data generator for this epoch with shuffled data
            data_gen_epoch = self._get_data_generator(
                x_real_shuffled_epoch, 
                real_datetimes_shuffled_epoch, 
                fundamental_features_shuffled_epoch,
                epoch_context_vectors_np, 
                batch_size
            )
            
            num_batches = (num_samples + batch_size - 1) // batch_size
            epoch_d_losses_real, epoch_d_losses_fake, epoch_g_losses = [], [], []

            for batch_idx in range(num_batches):
                try:
                    real_samples_batch, conditional_data_batch, context_data_batch_from_gen = next(data_gen_epoch)
                except StopIteration:
                    self.logger.warning(f"Epoch {epoch+1}, batch {batch_idx+1}: Data generator exhausted.")
                    break 
                
                actual_batch_size = real_samples_batch.shape[0]
                if actual_batch_size == 0: continue

                # --- Train Discriminator ---
                y_real = np.ones((actual_batch_size, 1)) * 0.9
                d_loss_real_metrics = self.discriminator.train_on_batch(real_samples_batch, y_real) # Removed verbose=0

                # Generate fake samples
                if self.gen_input_seq_len and self.gen_input_seq_len > 0: # For sequential latent vector
                    z_noise_shape = (actual_batch_size, self.gen_input_seq_len, self.gen_input_latent_dim)
                else: # For flat latent vector
                    z_noise_shape = (actual_batch_size, self.gen_input_latent_dim)
                z_noise = np.random.normal(0, 1, z_noise_shape)

                gen_inputs_for_predict_list = []
                if self.gen_input_latent_dim is not None and self.gen_input_latent_dim > 0:
                    gen_inputs_for_predict_list.append(z_noise)
                
                if self.conditional_dim_for_generator is not None and self.conditional_dim_for_generator > 0:
                    if conditional_data_batch.shape[1] != self.conditional_dim_for_generator:
                         self.logger.error(f"Batch {batch_idx+1}: Cond data batch dim {conditional_data_batch.shape[1]} vs expected {self.conditional_dim_for_generator}. Skipping D fake.")
                         d_loss_fake_metrics = [0.0] * len(self.discriminator.metrics_names) # type: ignore
                         epoch_d_losses_fake.append(d_loss_fake_metrics[0])
                         epoch_g_losses.append(0.0) 
                         continue 
                    gen_inputs_for_predict_list.append(conditional_data_batch)
                
                if self.context_dim_for_generator is not None and self.context_dim_for_generator > 0:
                    if context_data_batch_from_gen is not None and context_data_batch_from_gen.shape[0] == actual_batch_size and context_data_batch_from_gen.shape[1] == self.context_dim_for_generator:
                        gen_inputs_for_predict_list.append(context_data_batch_from_gen)
                    else: # Fallback if context_data_batch_from_gen is None or wrong shape (though _get_data_generator should provide zeros)
                        self.logger.warning(f"Epoch {epoch+1}, Batch {batch_idx+1}: Context data from generator is None or has wrong shape for G.predict(). Expected ({actual_batch_size}, {self.context_dim_for_generator}), got {context_data_batch_from_gen.shape if context_data_batch_from_gen is not None else 'None'}. Using zeros.")
                        gen_inputs_for_predict_list.append(np.zeros((actual_batch_size, self.context_dim_for_generator)))

                generated_samples = None
                if len(gen_inputs_for_predict_list) == len(self.generator.inputs):
                    predict_input_arg = gen_inputs_for_predict_list[0] if len(gen_inputs_for_predict_list) == 1 else gen_inputs_for_predict_list
                    if not self.generator.inputs and not gen_inputs_for_predict_list : # No inputs for generator and none provided
                         generated_samples = self.generator.predict(None, verbose=0)
                    elif self.generator.inputs and gen_inputs_for_predict_list:
                         generated_samples = self.generator.predict(predict_input_arg, verbose=0)
                    elif not self.generator.inputs and gen_inputs_for_predict_list:
                         self.logger.warning(f"Epoch {epoch+1}, Batch {batch_idx+1}: Generator expects no inputs, but inputs were provided. Calling with None.")
                         generated_samples = self.generator.predict(None, verbose=0)
                    else: # Generator expects inputs, but none were assembled (e.g. all dims were 0)
                         self.logger.error(f"Epoch {epoch+1}, Batch {batch_idx+1}: Generator expects inputs, but assembled list is empty. Skipping D fake training.")
                         d_loss_fake_metrics = [0.0] * len(self.discriminator.metrics_names) # type: ignore
                else:
                    self.logger.error(f"Epoch {epoch+1}, Batch {batch_idx+1}: Mismatch inputs for G.predict(). Expected {len(self.generator.inputs)}, got {len(gen_inputs_for_predict_list)}. Skipping D fake.")
                

                if generated_samples is not None:
                    y_fake = np.zeros((actual_batch_size, 1))
                    d_loss_fake_metrics = self.discriminator.train_on_batch(generated_samples, y_fake) # Removed verbose=0
                else: # generated_samples is None
                    # d_loss_fake_metrics already set to dummy if error occurred
                     if 'd_loss_fake_metrics' not in locals(): # Ensure it exists
                        d_loss_fake_metrics = [0.0] * len(self.discriminator.metrics_names) # type: ignore


                # --- Train Generator (via GAN model) ---
                gan_inputs_list = [] # Inputs for self.gan_model.train_on_batch
                # These must match the Keras Input layers defined in _build_gan for the GAN model
                # _build_gan creates inputs named like 'gan_input_latent', 'gan_input_conditional', 'gan_input_context'
                # The order here should correspond to how these were added to gan_keras_inputs_for_model_definition in _build_gan

                # Assuming the order in _build_gan is: latent, conditional, context (if they exist)
                if self.gen_input_latent_dim is not None and self.gen_input_latent_dim > 0:
                    gan_inputs_list.append(z_noise)
                if self.conditional_dim_for_generator is not None and self.conditional_dim_for_generator > 0:
                    gan_inputs_list.append(conditional_data_batch) 
                
                if self.context_dim_for_generator is not None and self.context_dim_for_generator > 0:
                    if context_data_batch_from_gen is not None and context_data_batch_from_gen.shape[0] == actual_batch_size and context_data_batch_from_gen.shape[1] == self.context_dim_for_generator:
                        gan_inputs_list.append(context_data_batch_from_gen)
                    else: # Fallback
                        self.logger.warning(f"Epoch {epoch+1}, Batch {batch_idx+1}: Context data from generator is None or has wrong shape for GAN input. Expected ({actual_batch_size}, {self.context_dim_for_generator}), got {context_data_batch_from_gen.shape if context_data_batch_from_gen is not None else 'None'}. Using zeros.")
                        gan_inputs_list.append(np.zeros((actual_batch_size, self.context_dim_for_generator)))
                
                g_loss_metrics_val = [0.0] * len(self.gan_model.metrics_names) # type: ignore # Default
                if len(gan_inputs_list) == len(self.gan_model.inputs):
                    y_gan = np.ones((actual_batch_size, 1))
                    
                    train_input_arg_gan = gan_inputs_list[0] if len(gan_inputs_list) == 1 else gan_inputs_list
                    if not self.gan_model.inputs and not gan_inputs_list:
                         g_loss_metrics_val = self.gan_model.train_on_batch(None, y_gan) # Removed verbose=0
                    elif self.gan_model.inputs and gan_inputs_list:
                         g_loss_metrics_val = self.gan_model.train_on_batch(train_input_arg_gan, y_gan) # Removed verbose=0
                    elif not self.gan_model.inputs and gan_inputs_list:
                         self.logger.warning(f"Epoch {epoch+1}, Batch {batch_idx+1}: GAN model expects no inputs, but inputs were provided. Calling with None.")
                         g_loss_metrics_val = self.gan_model.train_on_batch(None, y_gan)
                    else: # GAN expects inputs, but list is empty
                         self.logger.error(f"Epoch {epoch+1}, Batch {batch_idx+1}: GAN model expects inputs, but assembled list is empty. Skipping G.")

                else:
                    self.logger.error(f"Epoch {epoch+1}, Batch {batch_idx+1}: Mismatch inputs for GAN.train(). Expected {len(self.gan_model.inputs)}, got {len(gan_inputs_list)}. Skipping G.")
                
                # Store batch losses
                # d_loss_real_metrics and d_loss_fake_metrics are lists: [loss, acc, prec, recall, auc]
                # g_loss_metrics_val is also a list if GAN has metrics, or just a float for loss
                epoch_d_losses_real.append(d_loss_real_metrics[0] if isinstance(d_loss_real_metrics, list) else d_loss_real_metrics)
                epoch_d_losses_fake.append(d_loss_fake_metrics[0] if isinstance(d_loss_fake_metrics, list) else d_loss_fake_metrics)
                epoch_g_losses.append(g_loss_metrics_val[0] if isinstance(g_loss_metrics_val, list) else g_loss_metrics_val)


            avg_d_loss_real = np.mean(epoch_d_losses_real) if epoch_d_losses_real else 0
            avg_d_loss_fake = np.mean(epoch_d_losses_fake) if epoch_d_losses_fake else 0
            avg_g_loss = np.mean(epoch_g_losses) if epoch_g_losses else 0
            
            history_d_loss_real.append(avg_d_loss_real)
            history_d_loss_fake.append(avg_d_loss_fake)
            history_g_loss.append(avg_g_loss)

            epoch_duration = time.time() - epoch_start_time
            
            # Log epoch summary (Corrected logging)
            log_msg = (
                f"Epoch {epoch+1}/{epochs} [{epoch_duration:.2f}s] - "
                f"D_Loss_Real: {avg_d_loss_real:.4f}, D_Loss_Fake: {avg_d_loss_fake:.4f}, "
                f"G_Loss: {avg_g_loss:.4f}"
            )
            # Add main metrics if available (e.g., accuracy for D, specific metric for G)
            # Example: D accuracy (metric index 1 for 'accuracy')
            if len(d_loss_real_metrics) > 1 and self.discriminator.metrics_names[1]:
                 avg_d_acc_real = np.mean([m[1] for m in [d_loss_real_metrics] if len(m)>1]) # Simplified, proper avg needed
                 # log_msg += f" | D Acc: {avg_d_acc_real:.2f}" # Needs proper averaging over batches
            self.logger.info(log_msg)

            # Save models at interval
            if (epoch + 1) % save_interval == 0 or (epoch + 1) == epochs:
                self.logger.info(f"Saving models at epoch {epoch+1}...")
                self._save_models_at_epoch(epoch + 1)
                # self._plot_losses_at_epoch(epoch + 1, history_d_loss_real, history_d_loss_fake, history_g_loss) # Implement if needed

        # --- End of Epoch Loop ---
        self.logger.info("GAN Training finished.")
        self._save_final_models() # Save final models
        # self._plot_final_losses(history_d_loss_real, history_d_loss_fake, history_g_loss) # Implement if needed
        
        # Store metrics
        metrics_data = {
            "d_loss_real": history_d_loss_real,
            "d_loss_fake": history_d_loss_fake,
            "g_loss": history_g_loss,
            "epochs_trained": epochs
        }
        self._save_training_metrics(metrics_data)

        return {"status": "completed", "epochs_trained": epochs, "final_metrics": metrics_data}

    def _ensure_dir_exists(self, dir_path: str):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            self.logger.info(f"Created directory: {dir_path}")

    def _save_models_at_epoch(self, epoch: int):
        """Saves G, D, and GAN models at a specific epoch."""
        if not self.params.get("results_base_dir") or not self.params.get("save_model_dir"):
            self.logger.warning("Cannot save models: results_base_dir or save_model_dir not configured.")
            return

        models_path = os.path.join(self.params["results_base_dir"], self.params["save_model_dir"])
        self._ensure_dir_exists(models_path)

        if self.generator:
            gen_filename = self.params.get("save_generator_epoch_template", "generator_epoch_{epoch}.keras").format(epoch=epoch)
            self.generator.save(os.path.join(models_path, gen_filename))
            self.logger.info(f"Saved generator model to {os.path.join(models_path, gen_filename)}")
        if self.discriminator:
            disc_filename = self.params.get("save_discriminator_epoch_template", "discriminator_epoch_{epoch}.keras").format(epoch=epoch)
            self.discriminator.save(os.path.join(models_path, disc_filename))
            self.logger.info(f"Saved discriminator model to {os.path.join(models_path, disc_filename)}")
        if self.gan_model:
            gan_filename = self.params.get("save_gan_epoch_template", "gan_epoch_{epoch}.keras").format(epoch=epoch)
            # GAN model often just orchestrates, saving G and D might be enough.
            # If GAN itself has trainable weights or specific state, save it.
            # self.gan_model.save(os.path.join(models_path, gan_filename)) 
            # self.logger.info(f"Saved GAN model to {os.path.join(models_path, gan_filename)}")
            pass # Typically G and D are the ones to save for inference/reuse.


    def _save_final_models(self):
        """Saves the final G and D models."""
        if not self.params.get("results_base_dir") or not self.params.get("save_model_dir"):
            self.logger.warning("Cannot save final models: results_base_dir or save_model_dir not configured.")
            return
            
        models_path = os.path.join(self.params["results_base_dir"], self.params["save_model_dir"])
        self._ensure_dir_exists(models_path)

        if self.generator:
            gen_filename = self.params.get("final_generator_model_filename", "generator_final.keras")
            self.generator.save(os.path.join(models_path, gen_filename))
            self.logger.info(f"Saved final generator model to {os.path.join(models_path, gen_filename)}")
        if self.discriminator:
            disc_filename = self.params.get("final_discriminator_model_filename", "discriminator_final.keras")
            self.discriminator.save(os.path.join(models_path, disc_filename))
            self.logger.info(f"Saved final discriminator model to {os.path.join(models_path, disc_filename)}")
        # if self.gan_model:
        #     gan_filename = self.params.get("final_gan_model_filename", "gan_final.keras")
        #     self.gan_model.save(os.path.join(models_path, gan_filename))


    def _save_training_metrics(self, metrics_data: Dict):
        """Saves training metrics to a JSON file."""
        if not self.params.get("results_base_dir") or not self.params.get("save_metrics_dir"):
            self.logger.warning("Cannot save metrics: results_base_dir or save_metrics_dir not configured.")
            return

        metrics_path = os.path.join(self.params["results_base_dir"], self.params["save_metrics_dir"])
        self._ensure_dir_exists(metrics_path)
        
        metrics_filename = self.params.get("training_metrics_filename", "training_metrics.json")
        filepath = os.path.join(metrics_path, metrics_filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(metrics_data, f, indent=4)
            self.logger.info(f"Training metrics saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving training metrics to {filepath}: {e}")

    # Placeholder for plotting, if needed later
    # def _plot_losses_at_epoch(self, epoch, d_loss_real, d_loss_fake, g_loss):
    #     pass

    # def _plot_final_losses(self, d_loss_real, d_loss_fake, g_loss):
    #     pass

# Example usage (conceptual, not part of the plugin class itself)
# if __name__ == '__main__':
#     # This block is for illustration if you were to run this file directly for testing
#     # Setup basic logging
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     logger_main = logging.getLogger(__name__)

#     # --- Mock Generator Model ---
#     latent_dim_mock = 32
#     seq_len_mock = 18
#     features_mock = 4 # e.g., OHLC
#     conditional_dim_mock = 10 
    
#     # Generator Inputs
#     gen_input_latent = Input(shape=(latent_dim_mock,), name="mock_latent_input")
#     gen_input_conditional = Input(shape=(conditional_dim_mock,), name="mock_conditional_input")
    
#     # Simple Generator Body (example)
#     x = Concatenate()([Dense(64)(gen_input_latent), Dense(64)(gen_input_conditional)])
#     x = Reshape((1, 128))(x) # Make it somewhat sequential for LSTM
#     x = LSTM(128, return_sequences=True)(x)
#     # RepeatVector to match seq_len_mock if outputting a sequence
#     x = TimeDistributed(Dense(features_mock))(RepeatVector(seq_len_mock)(LSTM(features_mock * 2)(x))) # Output (batch, seq_len, features)
    
#     mock_generator = Model([gen_input_latent, gen_input_conditional], x, name="MockGenerator")
#     mock_generator.summary()

#     # --- Configuration ---
#     test_config = {
#         "gan_epochs": 5, # Short for testing
#         "gan_batch_size": 16,
#         "latent_dim": latent_dim_mock, # Must match generator
#         "seq_len": seq_len_mock, # Discriminator seq_len, should match generator output seq_len
#         "conditional_dim": conditional_dim_mock, # Must match generator
#         "context_dim": 0, # Assuming no context vector for this mock
        
#         "num_base_features_generated": features_mock, # Output features of generator
#         "base_feature_names_ordered": [f"feat_{i}" for i in range(features_mock)],

#         # For _get_scaled_date_features
#         "feeder_date_feature_names_for_conditioning": ["day_of_week", "hour_of_day"],
#         "feeder_max_day_of_week": 7.0,
#         "feeder_max_hour_of_day": 24.0,

#         # For Discriminator (assuming no TensorFlowTALayer for this simple test, D sees raw generator output)
#         "num_features_for_discriminator": features_mock, 

#         # Paths
#         "results_base_dir": "temp_gan_results",
#         "save_model_dir": "models",
#         "save_metrics_dir": "metrics",

#         # Generator input name hints (must match Keras Input layer names in mock_generator)
#         "generator_decoder_input_name_latent": "mock_latent_input", # Matches name in Input()
#         "generator_decoder_input_name_conditions": "mock_conditional_input", # Matches name in Input()
#         # "generator_decoder_input_name_context": "mock_context_input" # Not used here
#     }

#     # --- Instantiate GANTrainerPlugin ---
#     # No FeederPlugin or PreprocessorPlugin for this simple test
#     gan_trainer = GANTrainerPlugin(config=test_config, generator_plugin_instance=None) # Pass config
    
#     # Manually assign the mock generator and call set_params to build D and GAN
#     gan_trainer.generator = mock_generator
#     gan_trainer.set_params(**test_config) # This will build D and GAN

#     if gan_trainer.discriminator and gan_trainer.gan_model:
#         logger_main.info("Mock D and GAN built successfully.")
#         # --- Mock Data ---
#         num_mock_samples = 100
#         # Real data for discriminator: (num_samples, seq_len, features_mock)
#         mock_x_real_df = pd.DataFrame(np.random.rand(num_mock_samples * seq_len_mock, features_mock), 
#                                       columns=[f"feat_{i}" for i in range(features_mock)])
        
#         # Datetimes for conditional features
#         mock_datetimes = pd.to_datetime(pd.date_range(start="2023-01-01", periods=num_mock_samples * seq_len_mock, freq="H"))
        
#         logger_main.info(f"Mock x_real_df shape: {mock_x_real_df.shape}")
#         logger_main.info(f"Mock datetimes length: {len(mock_datetimes)}")

#         # --- Train ---
#         training_results = gan_trainer.train(x_real_df=mock_x_real_df, x_real_df_datetimes=mock_datetimes)
#         logger_main.info(f"Training finished. Results: {training_results}")
#     else:
#         logger_main.error("Failed to build Discriminator or GAN model with mock generator.")
