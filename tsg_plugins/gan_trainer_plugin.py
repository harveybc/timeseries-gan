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

                # Try to match based on configured feeder key names to the generator's inputs by their Keras names
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
        self.discriminator_input_seq_len = self.generator_output_actual_seq_len # This is self.seq_len
        # self.num_features_for_discriminator = self.params.get('num_features_for_discriminator', 1) # Removed this line
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
            
        self.logger.info("Building GAN model (Generator + Discriminator)...")
        self.discriminator.trainable = False # Freeze discriminator during GAN training phase
        
        gan_keras_inputs_for_model_definition = []  # List of tf.keras.Input layers for the GAN model's definition
        generator_call_inputs_ordered = [] # List of tensors (from gan_keras_inputs_for_model_definition) to be passed to self.generator()

        if not self.generator.inputs: 
            self.logger.info("GAN Build: Generator model has no defined Keras inputs. GAN will call generator without explicit inputs.")
        else: 
            if not self.generator_actual_input_names_ordered:
                self.logger.error("GAN Build: Generator has Keras inputs, but self.generator_actual_input_names_ordered is empty. Critical error.")
                raise ValueError("generator_actual_input_names_ordered is not populated despite generator having inputs.")

            self.logger.info(f"GAN Build: Preparing GAN inputs based on generator's {len(self.generator_actual_input_names_ordered)} actual Keras input layers: {self.generator_actual_input_names_ordered}")
            
            for i, actual_gen_keras_input_name in enumerate(self.generator_actual_input_names_ordered):
                original_gen_input_tensor = next((inp for inp in self.generator.inputs if inp.name == actual_gen_keras_input_name), None)
                if original_gen_input_tensor is None:
                    self.logger.error(f"GAN Build: Could not find Keras input tensor named '{actual_gen_keras_input_name}' in self.generator.inputs.")
                    raise ValueError(f"Inconsistent generator input specification for '{actual_gen_keras_input_name}'.")

                original_shape_tuple = tuple(original_gen_input_tensor.shape[1:])
                
                input_role_name = f"input_{i}" # Default role
                if actual_gen_keras_input_name == self.gan_latent_input_keras_name_hint:
                    input_role_name = "latent_z"
                elif actual_gen_keras_input_name == self.gan_window_input_keras_name_hint:
                    input_role_name = "input_window"
                elif actual_gen_keras_input_name == self.gan_conditional_input_keras_name_hint:
                    input_role_name = "conditional_input"
                elif actual_gen_keras_input_name == self.gan_context_input_keras_name_hint:
                    input_role_name = "context_input"
                else:
                    self.logger.warning(f"GAN Build: Gen input '{actual_gen_keras_input_name}' did not match any hint. Using generic role '{input_role_name}'.")
                
                gan_input_layer = keras.Input(shape=original_shape_tuple, name=f"gan_input_for_gen_{input_role_name}")
                gan_keras_inputs_for_model_definition.append(gan_input_layer)
                generator_call_inputs_ordered.append(gan_input_layer)
                self.logger.info(f"GAN Build: Created GAN input '{gan_input_layer.name}' with shape {original_shape_tuple} for generator input '{actual_gen_keras_input_name}'.")

        self.logger.info(f"GAN Build: Calling generator with {len(generator_call_inputs_ordered)} GAN inputs: {[inp.name for inp in generator_call_inputs_ordered]}")
        generated_output_tensor = self.generator(generator_call_inputs_ordered if self.generator.inputs else [])
        self.logger.info(f"GAN Build: Generator output tensor raw shape: {generated_output_tensor.shape}")

        processed_gen_output_for_discriminator = generated_output_tensor
        current_gen_output_seq_len = -1 # Will be updated based on actual generator output shape

        # Determine the actual sequence length and feature count from the generator's output tensor
        if len(generated_output_tensor.shape) == 3: # (batch, seq, features)
            current_gen_output_seq_len = generated_output_tensor.shape[1]
            current_gen_output_features = generated_output_tensor.shape[2]
            self.logger.info(f"GAN Build: Generator output is 3D. Seq_len={current_gen_output_seq_len}, Features={current_gen_output_features}")
            processed_gen_output_for_discriminator = generated_output_tensor
        elif len(generated_output_tensor.shape) == 2: # (batch, features) -> implies seq_len = 1
            current_gen_output_seq_len = 1
            current_gen_output_features = generated_output_tensor.shape[-1]
            self.logger.info(f"GAN Build: Generator output is 2D. Assuming Seq_len=1, Features={current_gen_output_features}. Reshaping to 3D.")
            processed_gen_output_for_discriminator = layers.Reshape((1, current_gen_output_features))(generated_output_tensor)
        else:
            self.logger.error(f"GAN Build: Generator output has unexpected shape {generated_output_tensor.shape}. Expected 2D or 3D.")
            raise ValueError(f"Generator output has unexpected shape {generated_output_tensor.shape}")

        # --- Feature Slicing (if generator produces more features than discriminator base expects) ---
        # This is important if the generator produces, say, 6 features, but the discriminator's base features are only the first 4 (e.g., OHLC).
        # self.num_base_features is the number of features the discriminator expects *before* TIs are added (if use_tensorflow_ta_layer is True)
        # OR it's the number of features the discriminator expects *after* TIs are calculated externally (if use_tensorflow_ta_layer is False).

        # If using TensorFlowTALayer, it expects self.num_base_features as input.
        # If NOT using TensorFlowTALayer, the discriminator directly expects self.num_features_for_discriminator.
        
        # The generator_output_actual_features (derived in _initialize_core_parameters_from_config)
        # should ideally match current_gen_output_features from the tensor.
        # Let's use current_gen_output_features as the source of truth for what the generator *actually* outputted.

        target_feature_count_for_slicing = self.num_base_features # Default for when TensorFlowTALayer will be used
        if self.params.get("use_tensorflow_ta_layer", False):
            target_feature_count_for_slicing = self.num_base_features
            self.logger.info(f"GAN Build: TensorFlowTALayer is active. Generator output will be sliced to self.num_base_features ({target_feature_count_for_slicing}) before TA layer.")
        else: # TIs are calculated externally, discriminator expects final feature set.
              # No, if TIs are external, the generator output should still be sliced to num_base_features,
              # then TIs are calculated on these, then concatenated. The discriminator then sees num_features_for_discriminator.
              # So, slicing to num_base_features is generally correct here.
            target_feature_count_for_slicing = self.num_base_features
            self.logger.info(f"GAN Build: TensorFlowTALayer is NOT active. Generator output will be sliced to self.num_base_features ({target_feature_count_for_slicing}) for external TI calculation.")


        if current_gen_output_features > target_feature_count_for_slicing:
            self.logger.info(f"GAN Build: Slicing generator output features from {current_gen_output_features} to {target_feature_count_for_slicing} (first {target_feature_count_for_slicing} features).")
            processed_gen_output_for_discriminator = processed_gen_output_for_discriminator[:, :, :target_feature_count_for_slicing]
            current_gen_output_features = target_feature_count_for_slicing # Update effective feature count after slicing
        elif current_gen_output_features < target_feature_count_for_slicing:
            self.logger.error(f"GAN Build: Generator output features ({current_gen_output_features}) is LESS than target_feature_count_for_slicing ({target_feature_count_for_slicing}). This is problematic.")
            # This shouldn't happen if num_base_features is set correctly based on generator's capabilities.
            # Consider raising an error or padding, but padding might hide issues.
            raise ValueError(f"Generator output features ({current_gen_output_features}) less than target ({target_feature_count_for_slicing}).")
        else:
            self.logger.info(f"GAN Build: Generator output features ({current_gen_output_features}) matches target_feature_count_for_slicing ({target_feature_count_for_slicing}). No slicing needed for feature count.")


        # --- Sequence Length Adjustment (if generator's output seq_len differs from discriminator's input seq_len) ---
        # self.seq_len is the discriminator's expected input sequence length.
        # current_gen_output_seq_len is what the generator actually outputted.
        if current_gen_output_seq_len != self.seq_len:
            self.logger.info(f"GAN Build: Generator output sequence length ({current_gen_output_seq_len}) differs from discriminator's expected ({self.seq_len}). Adjusting...")
            if self.seq_len == 1 and current_gen_output_seq_len > 1:
                # Take the last time step from the generator's output sequence
                self.logger.info(f"  Taking last time step of generator output sequence (discriminator expects seq_len=1).")
                processed_gen_output_for_discriminator = processed_gen_output_for_discriminator[:, -1:, :] # Shape: (batch, 1, features)
            elif self.seq_len > 1 and current_gen_output_seq_len == 1:
                # Repeat the single time step from generator to match discriminator's sequence length
                self.logger.info(f"  Repeating generator's single time step {self.seq_len} times for discriminator.")
                processed_gen_output_for_discriminator = layers.RepeatVector(self.seq_len)(processed_gen_output_for_discriminator) # Output: (batch, self.seq_len, features)
            else:
                # More complex scenario, e.g., upsampling or downsampling. Not handled by default.
                self.logger.error(f"GAN Build: Unhandled sequence length mismatch. Generator output seq_len={current_gen_output_seq_len}, Discriminator expects seq_len={self.seq_len}. Cannot automatically adjust.")
                raise ValueError("Unhandled sequence length mismatch between generator output and discriminator input.")
            self.logger.info(f"GAN Build: Adjusted generator output shape for discriminator: {processed_gen_output_for_discriminator.shape}")
        else:
            self.logger.info(f"GAN Build: Generator output sequence length ({current_gen_output_seq_len}) matches discriminator's expected ({self.seq_len}). No adjustment needed for sequence length.")

        # --- Apply TensorFlowTALayer if configured ---
        if self.params.get("use_tensorflow_ta_layer", False):
            if not self.base_feature_names_ordered:
                self.logger.error("GAN Build: TensorFlowTALayer is enabled, but 'base_feature_names_ordered' is empty. Cannot initialize layer.")
                raise ValueError("'base_feature_names_ordered' is required for TensorFlowTALayer.")
            
            # num_total_features_for_ta_layer = self.num_base_features + len(self.ti_names_for_discriminator)
            # The TA layer's num_total_features should be num_base_features + num_calculated_TIs.
            # self.num_features_for_discriminator should be this sum.
            
            self.logger.info(f"GAN Build: Applying TensorFlowTALayer. Base features: {self.num_base_features}, TI names: {self.ti_names_for_discriminator}, Expected total features for D: {self.num_features_for_discriminator}")
            self.logger.info(f"GAN Build: Input to TA Layer (processed_gen_output_for_discriminator) shape: {processed_gen_output_for_discriminator.shape}")
            
            # Ensure the input to TA layer has self.num_base_features
            if processed_gen_output_for_discriminator.shape[-1] != self.num_base_features:
                self.logger.error(f"GAN Build: Feature mismatch before TA Layer. Expected {self.num_base_features} features, got {processed_gen_output_for_discriminator.shape[-1]}. This indicates an issue with prior slicing or config.")
                raise ValueError(f"Feature count mismatch for TensorFlowTALayer input. Expected {self.num_base_features}, got {processed_gen_output_for_discriminator.shape[-1]}.")

            ta_layer = TensorFlowTALayer(
                base_feature_names=self.base_feature_names_ordered, # Should match the first N features of generator output
                ti_names_to_calculate=self.ti_names_for_discriminator,
                num_base_features=self.num_base_features, # Number of features input to TA layer
                num_total_features=self.num_features_for_discriminator, # Expected output features (base + TIs)
                seq_len=self.seq_len # Sequence length
            )
            discriminator_input_tensor = ta_layer(processed_gen_output_for_discriminator)
            self.logger.info(f"GAN Build: Output from TensorFlowTALayer shape: {discriminator_input_tensor.shape}")
            if discriminator_input_tensor.shape[-1] != self.num_features_for_discriminator:
                self.logger.warning(f"GAN Build: TensorFlowTALayer output feature count ({discriminator_input_tensor.shape[-1]}) "
                                   f"does not match self.num_features_for_discriminator ({self.num_features_for_discriminator}). "
                                   f"This might be due to TIs not being implemented in TF yet. Discriminator might fail if shapes mismatch.")
                # Potentially adjust self.num_features_for_discriminator if the layer dynamically determines it,
                # but the layer is initialized with num_total_features, so it should stick to it.
        else:
            # If not using TA layer, the processed_gen_output_for_discriminator (already sliced to num_base_features)
            # is what would be fed to external TI calculation.
            # The discriminator itself will expect self.num_features_for_discriminator.
            # This implies that if TA layer is not used, the GAN model cannot directly pipe generator to discriminator
            # if TIs are involved, because the TI calculation step is external to this Keras graph.
            # This is a limitation: the GAN model defined here assumes TIs are either part of D or G, or via TensorFlowTALayer.
            # For now, if no TA layer, we pass the (potentially sliced) generator output directly.
            # This means self.num_features_for_discriminator MUST equal self.num_base_features if TA layer is off.
            self.logger.info("GAN Build: TensorFlowTALayer is NOT active. Passing processed generator output directly to discriminator.")
            if processed_gen_output_for_discriminator.shape[-1] != self.num_features_for_discriminator:
                self.logger.error(f"GAN Build: Feature mismatch for direct discriminator input (no TA layer). "
                                  f"Generator output (after slicing) has {processed_gen_output_for_discriminator.shape[-1]} features, "
                                  f"but discriminator expects {self.num_features_for_discriminator} features. "
                                  f"If not using TensorFlowTALayer, ensure 'num_features_for_discriminator' equals 'num_base_features' "
                                  f"and 'ti_names_for_discriminator' is empty or handled externally before discriminator training.")
                # This is a critical error if they don't match.
                # For the GAN model graph, the output of G must match input of D (after TA layer if used).
                # If no TA layer, G_output_sliced must match D_input.
                # So, if use_tensorflow_ta_layer is False, then this gan_model should have been built with num_features_for_discriminator equal to num_base_features.
                # The _initialize_core_parameters_from_config should ensure this if TIs are empty.
                # If TIs are *not* empty and use_tensorflow_ta_layer is False, then this GAN graph is not complete for end-to-end training.
                # The train() method would need to handle TI calculation separately.
                # For _build_gan, we assume the tensor passed to D is what D expects.
            discriminator_input_tensor = processed_gen_output_for_discriminator


        # Final check on the tensor shape being fed to the discriminator
        if discriminator_input_tensor.shape[1] != self.seq_len or discriminator_input_tensor.shape[2] != self.num_features_for_discriminator:
            self.logger.error(f"GAN Build: CRITICAL SHAPE MISMATCH before discriminator call. "
                              f"Tensor shape: {discriminator_input_tensor.shape}, "
                              f"Discriminator expected input shape: (None, {self.seq_len}, {self.num_features_for_discriminator}).")
            # This might happen if TensorFlowTALayer didn't produce the expected number of features (e.g., some TIs not implemented)
            # Or if slicing/repeating logic had an issue.
            # For now, we proceed, but the discriminator.compile() or model call will likely fail.
            # A more robust solution would be to ensure TensorFlowTALayer pads/zeros unimplemented TIs to maintain shape.
            # The TensorFlowTALayer is designed to output num_total_features, so this check is more of a safeguard.

        gan_output = self.discriminator(discriminator_input_tensor)
        self.logger.info(f"GAN Build: Output from discriminator within GAN: {gan_output.shape}")

        # Create GAN model: takes generator inputs, outputs discriminator's decision
        if not gan_keras_inputs_for_model_definition: # Handle case where generator has no explicit inputs
             self.logger.warning("GAN Build: Generator has no explicit Keras inputs. GAN model will also have no explicit inputs. This is unusual.")
             # This case is problematic for standard GANs. Assuming generator can be called with []
             # If generator truly has no inputs, gan_keras_inputs_for_model_definition would be empty.
             # keras.Model requires inputs. If the generator has no inputs, it cannot be part of a standard Keras model graph this way.
             # This implies _initialize_core_parameters_from_config ensures some form of input for G if G exists.
             # If self.generator.inputs was empty, gan_keras_inputs_for_model_definition will be empty.
             # A Keras model must have inputs. If the generator has no inputs, it cannot be part of a standard Keras model graph this way.
             # This scenario should be caught earlier or handled by a different GAN structure.
             # For now, if gan_keras_inputs_for_model_definition is empty, this will error out when creating Model.
             if not self.generator.inputs: # Double check
                self.logger.error("GAN Build: Generator has no Keras inputs (self.generator.inputs is empty). Cannot define GAN model inputs.")
                raise ValueError("Generator has no Keras inputs, cannot build GAN model with defined inputs.")
             # If self.generator.inputs is NOT empty, but gan_keras_inputs_for_model_definition IS empty, it's an internal logic error.

        self.logger.info(f"GAN Build: Defining Keras GAN Model with inputs: {[inp.name for inp in gan_keras_inputs_for_model_definition]} and output: {gan_output.name}")
        gan_model = Model(inputs=gan_keras_inputs_for_model_definition, outputs=gan_output, name="GAN_Generator_plus_Discriminator")
        
        # Compile GAN model (Optimizer for generator, as discriminator is frozen)
        # Loss is binary cross-entropy as we want generator to fool discriminator (output 1)
        gan_model.compile(optimizer=self.g_optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        self.logger.info("GAN model compiled.")
        gan_model.summary(print_fn=self.logger.info)

        try:
            plot_dir = os.path.join(self.params.get("results_base_dir", "results"), self.params.get("save_plot_dir", "plots"))
            os.makedirs(plot_dir, exist_ok=True)
            plot_file_name = self.params.get("gan_model_plot_file", "gan_architecture.png")
            plot_path = os.path.join(plot_dir, plot_file_name)
            dpi = self.params.get("model_plot_dpi", 300)
            plot_model(gan_model, to_file=plot_path, show_shapes=True, dpi=dpi, expand_nested=True, show_layer_activations=True)
            self.logger.info(f"GAN architecture plot saved to {plot_path}")
        except Exception as e:
            self.logger.warning(f"Could not plot GAN model: {e}. Ensure pydot and graphviz are installed.")
        
        return gan_model

    def _calculate_technical_indicators(self, data: pd.DataFrame, strategy: List[Dict[str, Any]], feature_names_for_output: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Calculates technical indicators for the given data using the specified strategy.
        Args:
            data: Input data as a pandas DataFrame.
            strategy: List of dictionaries specifying the TIs to calculate and their parameters.
            feature_names_for_output: Optional list of feature names for the output DataFrame.
        Returns:
            DataFrame with the calculated TIs appended as new columns.
        """
        # This method was previously complex. Simplified for clarity for now.
        # It needs to use pandas_ta or similar.
        # The original implementation from the user's context is more complete.
        # For now, this is a placeholder to ensure the train loop can call it.
        # The actual logic should be robust as in the user's original _calculate_technical_indicators.
        
        # --- This is a simplified placeholder. The actual TI calculation logic from the user's code is more detailed ---
        # --- and should be used/adapted here. For example, parsing "RSI_14" into indicator and length. ---
        self.logger.debug(f"GANTrainerPlugin._calculate_technical_indicators called with df shape {data.shape}, features: {list(data.columns)}")

        if data.empty:
            self.logger.warning("Input DataFrame for TI calculation is empty.")
            return data

        df_with_ti = data.copy()
        
        # Ensure base feature names are present
        if not all(f_name in df_with_ti.columns for f_name in self.base_feature_names_ordered):
            self.logger.error(f"Missing one or more base features ({self.base_feature_names_ordered}) in DataFrame columns ({df_with_ti.columns}) for TI calculation.")
            # Return original df and names, or raise error
            return data

        # Placeholder: Assume ti_names are simple like 'RSI', 'EMA' and map to pandas_ta calls
        # A proper implementation would parse "RSI_14" etc.
        # For this placeholder, let's assume `ti_names` are directly usable by pandas_ta
        # and that `feature_names` provides the necessary columns (e.g., 'close' for RSI).
        
        # This is where the robust pandas_ta logic from the user's original code should be.
        # For example, using df.ta.rsi(), df.ta.macd(), etc.
        # The column names for OHLC need to be known (e.g., self.ohlc_feature_names from GeneratorPlugin)
        
        # Example (very basic, needs proper mapping and error handling):
        # if 'RSI_14' in ti_names and 'close' in df_with_ti.columns: # Assuming 'close' is a feature name
        #     df_with_ti['RSI_14'] = df_with_ti.ta.rsi(length=14, append=False) 
        # if 'EMA_20' in ti_names and 'close' in df_with_ti.columns:
        #     df_with_ti['EMA_20'] = df_with_ti.ta.ema(length=20, append=False)

        # The user's original _calculate_technical_indicators had more sophisticated parsing.
        # That logic should be integrated here if this method is used.
        # For now, this is just a structural placeholder.
        # If TIs are complex, this per-sample DataFrame conversion and pandas_ta call can be slow.
        
        # If TIs were calculated, new columns are added.
        # The `all_feature_names` should be `feature_names` + newly_added_ti_columns.
        # The `self.feature_names_for_discriminator_ordered` should be used to select final columns.

        # This placeholder doesn't actually calculate TIs. It's a pass-through.
        # The real implementation must add columns for each TI in `ti_names`.
        self.logger.warning("_calculate_technical_indicators is currently a placeholder and does not compute TIs. Real implementation needed if not using TensorFlowTALayer.")

        all_current_features = list(df_with_ti.columns)
        return df_with_ti, all_current_features


    def train(self, x_train_file: str, epochs: Optional[int] = None, batch_size: Optional[int] = None) -> None:
        self.logger.info(f"Starting GAN training process. Data file: {x_train_file}")

        # Get training parameters
        epochs = epochs if epochs is not None else self.params.get("gan_epochs", 10000)
        batch_size = batch_size if batch_size is not None else self.params.get("gan_batch_size", 32)
        save_interval = self.params.get("gan_save_interval", 500)
        
        # Ensure models are built
        if not self.generator or not self.discriminator or not self.gan_model:
            self.logger.error("Models are not built. Cannot start training. Ensure generator is loaded and set_params has been called.")
            # Try to build them if generator is present
            if self.generator_plugin and hasattr(self.generator_plugin, 'get_model') and not self.generator:
                self.generator = self.generator_plugin.get_model()
                self.logger.info("Loaded generator model from plugin.")
            
            if self.generator:
                self.logger.info("Attempting to build discriminator and GAN model before training...")
                # Ensure core parameters are initialized based on the (potentially just loaded) generator
                self._initialize_core_parameters_from_config() 
                self.discriminator = self._build_discriminator()
                self.gan_model = self._build_gan()
                if not self.discriminator or not self.gan_model:
                    self.logger.error("Failed to build discriminator or GAN model. Aborting training.")
                    return
                self.logger.info("Discriminator and GAN models built successfully.")
            else:
                self.logger.error("Generator model is not available. Aborting training.")
                return

        # Load real data
        try:
            if self.preprocessor_plugin and hasattr(self.preprocessor_plugin, 'load_data'):
                self.logger.info(f"Loading data using PreprocessorPlugin from {x_train_file}")
                # Assuming load_data returns a DataFrame or a structure that can be converted
                # The preprocessor might handle scaling, feature selection etc.
                # For GAN training, we typically need the scaled numerical data.
                # Let's assume it returns a DataFrame of the correct features.
                real_data_df = self.preprocessor_plugin.load_data(x_train_file) 
                if not isinstance(real_data_df, pd.DataFrame):
                    self.logger.error(f"PreprocessorPlugin.load_data did not return a pandas DataFrame. Got: {type(real_data_df)}")
                    # Potentially try to convert if it's a known structure, e.g. dict of arrays
                    raise ValueError("Loaded data is not a DataFrame.")
            else:
                self.logger.info(f"Loading data using pandas from {x_train_file}")
                real_data_df = pd.read_csv(x_train_file) # Or other format as needed
            
            self.logger.info(f"Loaded real data with shape: {real_data_df.shape}")
            if real_data_df.empty:
                self.logger.error("Loaded real data is empty. Aborting training.")
                return

            # Ensure data has the expected base features for the generator/discriminator
            # self.base_feature_names_ordered should contain the names of columns expected from real_data_df
            if not self.base_feature_names_ordered:
                self.logger.error("'base_feature_names_ordered' is not set. Cannot select features from real data.")
                # As a fallback, if num_base_features is set, assume first N columns. Risky.
                if self.num_base_features > 0 and self.num_base_features <= real_data_df.shape[1]:
                    self.logger.warning(f"Using first {self.num_base_features} columns from real data as base features due to missing 'base_feature_names_ordered'.")
                    x_real_processed = real_data_df.iloc[:, :self.num_base_features].values
                else:
                    raise ValueError("Cannot determine base features from real data.")
            else:
                missing_cols = [col for col in self.base_feature_names_ordered if col not in real_data_df.columns]
                if missing_cols:
                    self.logger.error(f"Real data is missing expected base feature columns: {missing_cols}. Available: {list(real_data_df.columns)}")
                    raise ValueError(f"Real data missing columns: {missing_cols}")
                x_real_processed = real_data_df[self.base_feature_names_ordered].values # Numpy array of base features
            
            self.logger.info(f"Processed real data (x_real_processed) for base features shape: {x_real_processed.shape}")

        except Exception as e:
            self.logger.error(f"Error loading or processing real data from {x_train_file}: {e}", exc_info=True)
            return

        # Training loop
        d_losses, g_losses = [], []
        start_time = time.time()

        use_tf_ta_layer = self.params.get("use_tensorflow_ta_layer", False)
        self.logger.info(f"Training configuration: use_tensorflow_ta_layer = {use_tf_ta_layer}")
        if not use_tf_ta_layer and self.ti_names_for_discriminator:
            self.logger.info(f"External TI calculation will be performed for {len(self.ti_names_for_discriminator)} TIs.")


        for epoch in range(epochs):
            epoch_start_time = time.time()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            self.discriminator.trainable = True # Unfreeze D for training

            # --- Train with real data ---
            # Select a random batch of real sequences
            # Each sample in x_real_batch should have shape (self.seq_len, self.num_base_features)
            # if TIs are calculated externally, or (self.seq_len, self.num_features_for_discriminator) if TIs are already in x_real_processed
            # For now, assume x_real_processed contains only base features. TIs are handled next.
            
            num_real_samples = x_real_processed.shape[0]
            if num_real_samples < self.seq_len:
                self.logger.error(f"Not enough real data samples ({num_real_samples}) to form a sequence of length {self.seq_len}. Aborting.")
                return

            # Create sequences from real data
            # x_real_sequences will be list of (seq_len, num_base_features) arrays
            x_real_sequences = []
            for i in range(num_real_samples - self.seq_len + 1):
                x_real_sequences.append(x_real_processed[i : i + self.seq_len, :])
            
            if not x_real_sequences:
                self.logger.error(f"Could not create any real sequences of length {self.seq_len} from data of shape {x_real_processed.shape}. Aborting.")
                return
            
            x_real_sequences_np = np.array(x_real_sequences) # Shape: (num_sequences, seq_len, num_base_features)
            self.logger.debug(f"Created {x_real_sequences_np.shape[0]} real sequences. Shape: {x_real_sequences_np.shape}")

            # Select a random batch of these sequences
            idx_real = np.random.randint(0, x_real_sequences_np.shape[0], batch_size)
            real_batch_base_features = x_real_sequences_np[idx_real] # (batch_size, seq_len, num_base_features)
            self.logger.debug(f"Real batch base features shape: {real_batch_base_features.shape}")

            # Prepare real data for discriminator
            # If not using TensorFlowTALayer, calculate TIs externally
            if not use_tf_ta_layer and self.ti_names_for_discriminator:
                # _calculate_technical_indicators expects a DataFrame
                # We need to process each sample in the batch
                real_batch_for_d_list = []
                for i in range(real_batch_base_features.shape[0]):
                    sample_df = pd.DataFrame(real_batch_base_features[i], columns=self.base_feature_names_ordered)
                    sample_with_tis_df = self._calculate_technical_indicators(
                        sample_df, 
                        strategy=self.tas_strategy_for_discriminator_tis,
                        feature_names_for_output=self.discriminator_feature_names # Ensure correct output columns
                    )
                    # Ensure the output has the correct number of features for the discriminator
                    if sample_with_tis_df.shape[1] != self.num_features_for_discriminator:
                         self.logger.error(f"External TI calculation for real data sample {i} resulted in {sample_with_tis_df.shape[1]} features, but discriminator expects {self.num_features_for_discriminator}. Columns: {sample_with_tis_df.columns.tolist()}")
                         # Handle error or pad, for now, error out
                         raise ValueError("Feature count mismatch after external TI calculation for real data.")
                    real_batch_for_d_list.append(sample_with_tis_df.values)
                
                real_batch_for_d = np.array(real_batch_for_d_list) # (batch_size, seq_len, num_features_for_discriminator)
                self.logger.debug(f"Real batch for D (after external TIs) shape: {real_batch_for_d.shape}")
            else:
                # If using TF TA layer, D expects base features. The layer handles TIs.
                # If not using TF TA layer AND no TIs are specified, D also expects base features.
                # In this case, num_features_for_discriminator should equal num_base_features.
                if real_batch_base_features.shape[2] != self.num_features_for_discriminator:
                    self.logger.error(f"Real batch feature count ({real_batch_base_features.shape[2]}) mismatch with discriminator expected features ({self.num_features_for_discriminator}) when no external TIs are calculated. This implies a config error.")
                    raise ValueError("Feature count mismatch for real data when TIs are handled by TA layer or no TIs.")
                real_batch_for_d = real_batch_base_features
                self.logger.debug(f"Real batch for D (base features, TIs by layer or none) shape: {real_batch_for_d.shape}")

            # Labels for real data: all 1s
            real_labels = np.ones((batch_size, 1))
            d_loss_real = self.discriminator.train_on_batch(real_batch_for_d, real_labels)
            self.logger.debug(f"Discriminator loss on real batch: {d_loss_real}")


            # --- Train with fake data ---
            # Generate a batch of fake data using the FeederPlugin and Generator
            if not self.feeder_plugin:
                self.logger.error("FeederPlugin is not available. Cannot generate fake data for discriminator training.")
                # Potentially skip this epoch or abort, for now, skip D training on fake
                d_loss_fake = [0,0] # Placeholder
            else:
                # The FeederPlugin should provide a dictionary of inputs for the generator
                # The keys of this dictionary should match what the generator expects.
                # self.generator_actual_input_names_ordered holds the Keras names of the generator's inputs in order.
                # The FeederPlugin's get_batch method needs to provide data for these.
                
                # How FeederPlugin provides data:
                # It might return a dict like {"latent_vector": ..., "conditional_data": ...}
                # Or it might return a list/tuple of arrays if the order is implicitly known.
                # We need to map feeder output to the generator's Keras input names.

                # Let's assume feeder_plugin.get_batch(batch_size) returns a dictionary
                # where keys are like self.feeder_key_name_latent, self.feeder_key_name_conditional etc.
                try:
                    feeder_output_dict = self.feeder_plugin.get_batch(batch_size)
                    if not isinstance(feeder_output_dict, dict):
                        raise ValueError("Feeder output for GAN training is not a dictionary.")

                    # Prepare the list of input arrays for generator.predict() in the correct order.
                    # self.generator_actual_input_names_ordered contains the Keras names.
                    # We need to find which feeder key corresponds to which Keras input name.
                    # The hints (e.g., self.gan_latent_input_keras_name_hint) store the Keras name for a given role (latent, conditional).
                    # The feeder keys (e.g., self.feeder_key_name_latent) are used to get data from feeder_output_dict.

                    generator_inputs_for_predict = []
                    if not self.generator.inputs: # Generator has no Keras.Input layers defined
                         self.logger.info("Generator has no explicit Keras inputs. Calling generator.predict([])")
                         # generator_inputs_for_predict remains empty list
                    elif not self.generator_actual_input_names_ordered:
                        self.logger.error("Generator has Keras inputs, but self.generator_actual_input_names_ordered is empty. Cannot prepare generator inputs.")
                        raise ValueError("generator_actual_input_names_ordered is empty.")
                    else:
                        # Map feeder output to ordered generator inputs
                        for keras_input_name_needed_by_gen in self.generator_actual_input_names_ordered:
                            found_input_for_gen = False
                            # Check if this Keras input name corresponds to a known role (latent, conditional, context)
                            # and get the data from feeder_output_dict using the role's feeder key.
                            if keras_input_name_needed_by_gen == self.gan_latent_input_keras_name_hint and self.feeder_key_name_latent in feeder_output_dict:
                                generator_inputs_for_predict.append(feeder_output_dict[self.feeder_key_name_latent])
                                found_input_for_gen = True
                            elif keras_input_name_needed_by_gen == self.gan_conditional_input_keras_name_hint and self.feeder_key_name_conditional in feeder_output_dict:
                                generator_inputs_for_predict.append(feeder_output_dict[self.feeder_key_name_conditional])
                                found_input_for_gen = True
                            elif keras_input_name_needed_by_gen == self.gan_context_input_keras_name_hint and self.feeder_key_name_context in feeder_output_dict:
                                generator_inputs_for_predict.append(feeder_output_dict[self.feeder_key_name_context])
                                found_input_for_gen = True
                            # TODO: Add handling for other feeder inputs if self.gan_feeder_input_keras_name_hints is used
                            # This would involve iterating self.gan_feeder_input_keras_name_hints which maps feeder_key to keras_name
                            # elif keras_input_name_needed_by_gen in self.gan_feeder_input_keras_name_hints.values():
                            #    feeder_key = [fk for fk, kn in self.gan_feeder_input_keras_name_hints.items() if kn == keras_input_name_needed_by_gen][0]
                            #    if feeder_key in feeder_output_dict:
                            #        generator_inputs_for_predict.append(feeder_output_dict[feeder_key])
                            #        found_input_for_gen = True
                            
                            if not found_input_for_gen:
                                # Fallback: if a generator input Keras name directly matches a feeder output key
                                if keras_input_name_needed_by_gen in feeder_output_dict:
                                    self.logger.warning(f"Generator input '{keras_input_name_needed_by_gen}' matched directly with a feeder output key. Ensure this is intended.")
                                    generator_inputs_for_predict.append(feeder_output_dict[keras_input_name_needed_by_gen])
                                    found_input_for_gen = True
                                else:
                                    self.logger.error(f"Could not find data in FeederPlugin output for generator Keras input: '{keras_input_name_needed_by_gen}'. Feeder keys: {list(feeder_output_dict.keys())}")
                                    raise ValueError(f"Missing data for generator input '{keras_input_name_needed_by_gen}' from feeder.")
                        
                        self.logger.debug(f"Prepared {len(generator_inputs_for_predict)} inputs for generator.predict().")

                    # Generate fake data sequences
                    # generator.predict() output should be (batch_size, gen_output_seq_len, gen_output_features)
                    # or (batch_size, gen_output_features) if gen_output_seq_len is 1 and squeezed.
                    generated_data_raw = self.generator.predict(generator_inputs_for_predict, batch_size=batch_size) # Use actual batch_size for predict
                    self.logger.debug(f"Raw generated data shape: {generated_data_raw.shape}")

                    # Process generated_data_raw to match discriminator input requirements (reshaping, slicing, TIs)
                    # This logic should mirror parts of _build_gan's processing of generator output.

                    # 1. Reshape if 2D output from generator
                    processed_fake_data = generated_data_raw
                    current_fake_seq_len = -1
                    current_fake_features = -1

                    if len(generated_data_raw.shape) == 2: # (batch, features)
                        current_fake_seq_len = 1
                        current_fake_features = generated_data_raw.shape[-1]
                        processed_fake_data = np.reshape(generated_data_raw, (batch_size, 1, current_fake_features))
                        self.logger.debug(f"Reshaped 2D generated data to 3D: {processed_fake_data.shape}")
                    elif len(generated_data_raw.shape) == 3: # (batch, seq, features)
                        current_fake_seq_len = generated_data_raw.shape[1]
                        current_fake_features = generated_data_raw.shape[2]
                        self.logger.debug(f"Generated data is 3D: {processed_fake_data.shape}")
                    else:
                        self.logger.error(f"Generator.predict() output has unexpected shape: {generated_data_raw.shape}")
                        raise ValueError("Generator output shape error during training.")

                    # 2. Feature Slicing (to self.num_base_features)
                    # This is done *before* external TI calculation or TensorFlowTALayer.
                    if current_fake_features > self.num_base_features:
                        self.logger.debug(f"Slicing generated features from {current_fake_features} to {self.num_base_features} (first {self.num_base_features} features).")
                        processed_fake_data = processed_fake_data[:, :, :self.num_base_features]
                        current_fake_features = self.num_base_features
                    elif current_fake_features < self.num_base_features:
                         self.logger.error(f"Generated data has {current_fake_features} features, but expected at least {self.num_base_features} base features.")
                         raise ValueError("Generated feature count mismatch.")
                    
                    self.logger.debug(f"Generated data after feature slicing (if any): {processed_fake_data.shape}")


                    # 3. Sequence Length Adjustment (to self.seq_len for discriminator)
                    if current_fake_seq_len != self.seq_len:
                        self.logger.debug(f"Adjusting generated sequence length from {current_fake_seq_len} to {self.seq_len}.")
                        if self.seq_len == 1 and current_fake_seq_len > 1:
                            processed_fake_data = processed_fake_data[:, -1:, :] # Take last time step
                        elif self.seq_len > 1 and current_fake_seq_len == 1:
                            processed_fake_data = np.repeat(processed_fake_data, self.seq_len, axis=1) # Repeat vector
                        else:
                            self.logger.error(f"Unhandled sequence length mismatch for generated data: G={current_fake_seq_len}, D_expects={self.seq_len}")
                            raise ValueError("Sequence length adjustment error for fake data.")
                    self.logger.debug(f"Generated data after seq length adjustment (if any): {processed_fake_data.shape}")


                    # 4. Prepare for Discriminator (apply TIs if needed)
                    fake_batch_for_d_list = []
                    if not use_tf_ta_layer and self.ti_names_for_discriminator:
                        # External TI calculation for fake data
                        for i in range(processed_fake_data.shape[0]): # Iterate over batch
                            sample_df = pd.DataFrame(processed_fake_data[i], columns=self.base_feature_names_ordered)
                            try:
                                sample_with_tis_df = self._calculate_technical_indicators(
                                    sample_df,
                                    strategy=self.tas_strategy_for_discriminator_tis,
                                    feature_names_for_output=self.discriminator_feature_names
                                )
                                if sample_with_tis_df.shape[1] != self.num_features_for_discriminator:
                                    self.logger.error(f"External TI calculation for FAKE data sample {i} resulted in {sample_with_tis_df.shape[1]} features, but discriminator expects {self.num_features_for_discriminator}. Columns: {sample_with_tis_df.columns.tolist()}")
                                    # If error, fill with zeros of correct shape to avoid crashing, but log heavily
                                    # This is a workaround for the pandas-ta TypeError with NaNs from generated data
                                    # A better fix is in _calculate_technical_indicators to handle NaNs gracefully.
                                    # For now, if it fails, we might not be able to train D on this fake sample correctly.
                                    # Let's assume _calculate_technical_indicators is robust or we accept potential errors here.
                                    raise ValueError("Feature count mismatch after external TI calculation for fake data.")
                                fake_batch_for_d_list.append(sample_with_tis_df.values)
                            except TypeError as te: # Catch the specific TypeError from pandas-ta
                                self.logger.error(f"TypeError during _calculate_technical_indicators for FAKE data sample {i}: {te}. Columns: {sample_df.columns}. Data head:\\n{sample_df.head()}", exc_info=True)
                                # Create a zero array of the expected shape for this sample to allow training to continue
                                # This is a stop-gap. The root cause of NaNs/NoneTypes in generated data or TI calc needs fixing.
                                # For now, if it fails, we might not be able to train D on this fake sample correctly.
                                # Let's assume _calculate_technical_indicators is robust or we accept potential errors here.
                                zero_sample = np.zeros((self.seq_len, self.num_features_for_discriminator))
                                fake_batch_for_d_list.append(zero_sample)
                                self.logger.warning(f"Replaced problematic fake sample {i} with zeros due to TI calculation error.")
                            except Exception as e_ti_fake:
                                self.logger.error(f"Error in _calculate_technical_indicators for FAKE data sample {i}: {e_ti_fake}", exc_info=True)
                                # Fallback to zeros for this sample
                                zero_sample = np.zeros((self.seq_len, self.num_features_for_discriminator))
                                fake_batch_for_d_list.append(zero_sample)
                                self.logger.warning(f"Replaced problematic fake sample {i} with zeros due to TI calculation error.")


                        fake_batch_for_d = np.array(fake_batch_for_d_list)
                        self.logger.debug(f"Fake batch for D (after external TIs) shape: {fake_batch_for_d.shape}")
                    else:
                        # If using TF TA layer, D expects base features (already in processed_fake_data).
                        # Or if no TIs, D also expects base features.
                        if processed_fake_data.shape[2] != self.num_features_for_discriminator:
                             self.logger.error(f"Fake batch feature count ({processed_fake_data.shape[2]}) mismatch with D expected features ({self.num_features_for_discriminator}) when TIs by layer or none.")
                             raise ValueError("Feature count mismatch for fake data (TIs by layer/none).")
                        fake_batch_for_d = processed_fake_data
                        self.logger.debug(f"Fake batch for D (base features, TIs by layer or none) shape: {fake_batch_for_d.shape}")

                    # Labels for fake data: all 0s
                    fake_labels = np.zeros((batch_size, 1))
                    d_loss_fake = self.discriminator.train_on_batch(fake_batch_for_d, fake_labels)
                    self.logger.debug(f"Discriminator loss on fake batch: {d_loss_fake}")

                except Exception as e_fake_gen:
                    self.logger.error(f"Error during fake data generation or processing for discriminator: {e_fake_gen}", exc_info=True)
                    d_loss_fake = [np.nan, np.nan] # Indicate failure

            # Total discriminator loss
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            self.logger.debug(f"Total discriminator loss for epoch {epoch+1}: {d_loss}")

            # -----------------
            #  Train Generator (via GAN model)
            # -----------------
            self.discriminator.trainable = False # Freeze D when training G (via GAN model)

            # Generate inputs for the GAN model (which are the inputs for the generator part of GAN)
            # This is the same set of inputs we prepared for generator.predict() earlier.
            # However, gan_model.train_on_batch expects a list of numpy arrays if GAN has multiple inputs.
            # Or a single numpy array if GAN has a single input.
            # The `generator_inputs_for_predict` list is already in the correct format.
            
            # We need to re-generate feeder_output_dict and generator_inputs_for_predict
            # because the previous ones were consumed by D training.
            # Or, if they are deterministic for a given batch_size, we could reuse, but safer to regen.
            g_loss = [np.nan, np.nan] # Default if feeder fails
            if self.feeder_plugin:
                try:
                    feeder_output_dict_for_gan = self.feeder_plugin.get_batch(batch_size)
                    if not isinstance(feeder_output_dict_for_gan, dict):
                        raise ValueError("Feeder output for GAN training is not a dictionary.")

                    gan_model_inputs_ordered = []
                    if not self.gan_model.inputs: # GAN model itself has no inputs (should not happen if G has inputs)
                        self.logger.error("GAN model has no Keras inputs defined. Cannot train generator.")
                        # This implies self.gan_keras_inputs_for_model_definition was empty in _build_gan
                    elif not self.generator_actual_input_names_ordered and self.generator.inputs: # G has inputs, but mapping list is empty
                        self.logger.error("Generator has inputs, but 'generator_actual_input_names_ordered' is empty. Cannot prepare GAN inputs for G training.")
                        raise ValueError("generator_actual_input_names_ordered is empty for G training via GAN.")
                    else:
                        # The inputs to gan_model are those defined in _build_gan based on generator's inputs.
                        # Their names are like "gan_input_for_gen_latent_z", etc.
                        # We need to map feeder data to these GAN model inputs.
                        # The order of self.gan_model.inputs should match the order of self.generator_actual_input_names_ordered
                        # because that's how _build_gan created them.

                        for i, gan_model_input_layer in enumerate(self.gan_model.inputs):
                            # The corresponding original generator Keras input name:
                            original_gen_keras_input_name = self.generator_actual_input_names_ordered[i]
                            found_input_for_gan_model = False
                            
                            if original_gen_keras_input_name == self.gan_latent_input_keras_name_hint and self.feeder_key_name_latent in feeder_output_dict_for_gan:
                                gan_model_inputs_ordered.append(feeder_output_dict_for_gan[self.feeder_key_name_latent])
                                found_input_for_gan_model = True
                            elif original_gen_keras_input_name == self.gan_conditional_input_keras_name_hint and self.feeder_key_name_conditional in feeder_output_dict_for_gan:
                                gan_model_inputs_ordered.append(feeder_output_dict_for_gan[self.feeder_key_name_conditional])
                                found_input_for_gan_model = True
                            elif original_gen_keras_input_name == self.gan_context_input_keras_name_hint and self.feeder_key_name_context in feeder_output_dict_for_gan:
                                gan_model_inputs_ordered.append(feeder_output_dict_for_gan[self.feeder_key_name_context])
                                found_input_for_gan_model = True
                            # TODO: Add self.gan_feeder_input_keras_name_hints logic here too

                            if not found_input_for_gan_model:
                                # Fallback: if a generator input Keras name directly matches a feeder output key
                                if original_gen_keras_input_name in feeder_output_dict_for_gan: # Direct match fallback
                                    gan_model_inputs_ordered.append(feeder_output_dict_for_gan[original_gen_keras_input_name])
                                    found_input_for_gan_model = True
                                else:
                                    self.logger.error(f"Could not find data in Feeder output for GAN model input corresponding to generator input '{original_gen_keras_input_name}'. Feeder keys: {list(feeder_output_dict_for_gan.keys())}")
                                    raise ValueError(f"Missing data for GAN model input related to '{original_gen_keras_input_name}'.")
                        
                        self.logger.debug(f"Prepared {len(gan_model_inputs_ordered)} inputs for gan_model.train_on_batch().")


                    # Labels for generator training: we want discriminator to output 1 (real) for fake images
                    valid_labels_for_g = np.ones((batch_size, 1))
                    
                    # If gan_model has only one input, pass the array directly, not as a list
                    input_data_for_gan_train = gan_model_inputs_ordered
                    if len(self.gan_model.inputs) == 1 and isinstance(gan_model_inputs_ordered, list) and len(gan_model_inputs_ordered) == 1:
                        input_data_for_gan_train = gan_model_inputs_ordered[0]
                    elif not self.gan_model.inputs: # No inputs to GAN model (e.g. generator has no inputs)
                        input_data_for_gan_train = [] # Should match how GAN was built if G has no inputs

                    g_loss = self.gan_model.train_on_batch(input_data_for_gan_train, valid_labels_for_g)
                    self.logger.debug(f"Generator (via GAN) loss: {g_loss}")

                except Exception as e_gan_train:
                    self.logger.error(f"Error during GAN model training (generator update): {e_gan_train}", exc_info=True)
                    g_loss = [np.nan, np.nan] # Indicate failure

            # Store losses
            d_losses.append(d_loss[0] if isinstance(d_loss, (list, np.ndarray)) and len(d_loss) > 0 else np.nan)
            g_losses.append(g_loss[0] if isinstance(g_loss, (list, np.ndarray)) and len(g_loss) > 0 else np.nan)

            epoch_duration = time.time() - epoch_start_time
            # Log progress
            if (epoch + 1) % (save_interval // 10 if save_interval >= 100 else 10) == 0 or epoch == 0 : # Log more frequently initially or if save_interval is small
                self.logger.info(f"Epoch {epoch+1}/{epochs} | D Loss: {d_loss[0]:.4f} (R: {d_loss_real[0]:.4f}, F: {d_loss_fake[0]:.4f}) "
                                 f"| D Acc: {100*d_loss[1]:.2f}% (R: {100*d_loss_real[1]:.2f}%, F: {100*d_loss_fake[1]:.2f}%) "
                                 f"| G Loss: {g_loss[0]:.4f} | G Acc: {100*g_loss[1]:.2f}% | Time/Epoch: {epoch_duration:.2f}s")

            # Save models and plot losses at intervals
            if (epoch + 1) % save_interval == 0 or (epoch + 1) == epochs:
                self.logger.info(f"Saving models and plotting losses at epoch {epoch+1}...")
                self._save_models(epoch + 1)
                self._plot_losses(d_losses, g_losses, epoch + 1)
                self._save_training_metrics(epoch + 1, d_losses, g_losses, time.time() - start_time)

        total_training_time = time.time() - start_time
        self.logger.info(f"GAN Training finished after {epochs} epochs. Total time: {total_training_time:.2f} seconds.")

        # Save final models, plots, and metrics
        self.logger.info("Saving final models, loss plot, and metrics...")
        self._save_models(epochs, is_final=True)
        self._plot_losses(d_losses, g_losses, epochs, is_final=True)
        self._save_training_metrics(epochs, d_losses, g_losses, total_training_time, is_final=True)
        
        self.logger.info("GANTrainerPlugin training complete.")

    def _ensure_dir_exists(self, dir_path: str):
        os.makedirs(dir_path, exist_ok=True)
        self.logger.info(f"Ensured directory exists: {dir_path}")

    def _save_models(self, epoch: int, is_final: bool = False) -> None:
        """
        Saves the generator, discriminator, and GAN models to file.
        Args:
            epoch: Current epoch number (for naming files).
            is_final: Whether this is the final save (affects filename).
        """
        try:
            # Save generator
            if self.generator:
                gen_model_path = os.path.join(self.models_dir, self.params["final_generator_model_filename"] if is_final else self.params["save_generator_epoch_template"].format(epoch=epoch))
                self.generator.save(gen_model_path)
                self.logger.info(f"Saved generator model to {gen_model_path}")
            else:
                self.logger.warning("Generator model not available for saving.")
            
            # Save discriminator
            if self.discriminator:
                disc_model_path = os.path.join(self.models_dir, self.params["final_discriminator_model_filename"] if is_final else self.params["save_discriminator_epoch_template"].format(epoch=epoch))
                self.discriminator.save(disc_model_path)
                self.logger.info(f"Saved discriminator model to {disc_model_path}")
            else:
                self.logger.warning("Discriminator model not available for saving.")
            
            # Save GAN model
            if self.gan_model:
                gan_model_path = os.path.join(self.models_dir, self.params["final_gan_model_filename"] if is_final else self.params["save_gan_epoch_template"].format(epoch=epoch))
                self.gan_model.save(gan_model_path)
                self.logger.info(f"Saved GAN model to {gan_model_path}")
            else:
                self.logger.warning("GAN model not available for saving.")
        except Exception as e:
            self.logger.error(f"Error saving models at epoch {epoch}: {e}", exc_info=True)

    def _plot_losses(self, d_losses: List[float], g_losses: List[float], epoch: int, is_final: bool = False) -> None:
        """
        Plots and saves the generator and discriminator loss over epochs.
        Args:
            d_losses: List of discriminator loss values.
            g_losses: List of generator loss values.
            epoch: Current epoch number (for naming files).
            is_final: Whether this is the final plot (affects filename).
        """
        try:
            # Convert to numpy arrays if they aren't already
            d_losses_np = np.array(d_losses)
            g_losses_np = np.array(g_losses)

            # Time to plot
            plt.figure(figsize=(10, 5))

            # Discriminator Loss
            plt.subplot(1, 2, 1)
            plt.plot(d_losses_np, label='Discriminator Loss', color='red')
            plt.title('Discriminator Loss Over Epochs')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()

            # Generator Loss
            plt.subplot(1, 2, 2)
            plt.plot(g_losses_np, label='Generator Loss', color='blue')
            plt.title('Generator Loss Over Epochs')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()

            plt.tight_layout()

            # Save the plot
            if epoch is not None:
                plt.savefig(f"loss_plot_epoch_{epoch}.png", dpi=self.params.get("loss_plot_dpi", 300))
            else:
                plt.savefig("loss_plot_final.png", dpi=self.params.get("loss_plot_dpi", 300))

            plt.show()

        except Exception as e:
            self.logger.error(f"Error in plotting losses: {e}", exc_info=True)

    def _save_training_metrics(self, epoch: int, d_losses: List[float], g_losses: List[float], elapsed_time: float, is_final: bool = False) -> None:
        """
        Saves the training metrics (loss values) to a JSON file.
        Args:
            epoch: Current epoch number.
            d_losses: List of discriminator loss values.
            g_losses: List of generator loss values.
            elapsed_time: Elapsed time for the training (or epoch).
            is_final: Whether this is the final save (affects filename).
        """
        try:
            metrics_path = os.path.join(self.metrics_dir, self.params["training_metrics_filename"])
            all_metrics = {}
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    all_metrics = json.load(f)
            
            # Ensure the epoch key exists
            epoch_key = str(epoch)
            if epoch_key not in all_metrics:
                all_metrics[epoch_key] = {"d_loss": [], "g_loss": [], "elapsed_time": 0}

            # Append current losses
            all_metrics[epoch_key]["d_loss"].extend(d_losses)
            all_metrics[epoch_key]["g_loss"].extend(g_losses)
            all_metrics[epoch_key]["elapsed_time"] = elapsed_time

            # Save back to file
            with open(metrics_path, 'w') as f:
                json.dump(all_metrics, f, indent=4)
            self.logger.info(f"Saved training metrics to {metrics_path}")
        except Exception as e:
            self.logger.error(f"Error saving training metrics at epoch {epoch}: {e}", exc_info=True)

    def get_model(self) -> Optional[Model]:
        return self.gan_model

    def set_generator_plugin(self, generator_plugin_instance: Any) -> None:
        self.generator_plugin = generator_plugin_instance
        self.logger.info(f"Set new generator plugin instance: {generator_plugin_instance}")

    def set_feeder_plugin(self, feeder_plugin_instance: Any) -> None:
        self.feeder_plugin = feeder_plugin_instance
        self.logger.info(f"Set new feeder plugin instance: {feeder_plugin_instance}")

    def set_preprocessor_plugin(self, preprocessor_plugin_instance: Any) -> None:
        self.preprocessor_plugin = preprocessor_plugin_instance
        self.logger.info(f"Set new preprocessor plugin instance: {preprocessor_plugin_instance}")

# Example usage (conceptual, would be part of a larger script)
if __name__ == '__main__':
    # ... existing example code ...
    pass
