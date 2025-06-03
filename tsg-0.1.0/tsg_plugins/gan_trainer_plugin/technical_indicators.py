#!/usr/bin/env python3
"""
technical_indicators.py

TensorFlow-based technical indicator calculation layer for GAN training.
Provides custom Keras layer for real-time technical indicator computation
during discriminator training.

This module encapsulates all technical indicator calculation logic following
single responsibility principle and extreme separation of concerns.

Author: TimeSeries-GAN Team
"""

import tensorflow as tf
from tensorflow import keras
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class TensorFlowTALayer(keras.layers.Layer):
    """
    Custom Keras layer for technical indicator calculation using TensorFlow operations.
    
    This layer dynamically calculates technical indicators during training/inference,
    allowing the discriminator to work with enhanced feature sets including both
    base features (OHLC) and computed technical indicators.
    
    Attributes:
        base_feature_names_ordered: List of base feature names in order
        ti_names_to_calculate: List of technical indicator names to compute
        num_base_features: Number of base features in input
        num_total_features: Total features in output (base + TIs)
        seq_len: Sequence length for input data
    """
    
    def __init__(self, base_feature_names: List[str], ti_names_to_calculate: List[str],
                 num_base_features: int, num_total_features: int, seq_len: int, **kwargs):
        """
        Initialize TensorFlow technical indicator layer.
        
        Args:
            base_feature_names: Ordered list of base feature names (e.g., OHLC)
            ti_names_to_calculate: List of TI names to calculate
            num_base_features: Number of base features in input tensor
            num_total_features: Expected total features in output (base + TIs)
            seq_len: Sequence length for input sequences
            **kwargs: Additional Keras layer arguments
        """
        super().__init__(**kwargs)
        self.base_feature_names_ordered = base_feature_names
        self.ti_names_to_calculate = ti_names_to_calculate
        self.num_base_features = num_base_features
        self.num_total_features = num_total_features
        self.seq_len = seq_len
        
        # For faster lookups during computation
        self.base_feature_name_to_idx = {name: i for i, name in enumerate(self.base_feature_names_ordered)}
        self.num_ti_features_to_calc = len(self.ti_names_to_calculate)
        
        # Pre-parse technical indicator information
        self.parsed_tis_info = []
        for ti_full_name in self.ti_names_to_calculate:
            col, ind, params = self._parse_ti_name(ti_full_name)
            self.parsed_tis_info.append({
                "full_name": ti_full_name, 
                "col": col, 
                "ind": ind, 
                "params": params
            })
            
            if not ind:
                logger.warning(f"TensorFlowTALayer: TI '{ti_full_name}' could not be parsed. Will output zeros.")
            elif ind.lower() not in ["ema", "rsi"]:
                logger.warning(f"TensorFlowTALayer: TI type '{ind}' not implemented with TF operations. Will output zeros.")
    
    @staticmethod
    def _parse_ti_name(ti_full_name: str) -> Tuple[Optional[str], Optional[str], Dict[str, Any]]:
        """
        Parse technical indicator string into components.
        
        Handles patterns like:
        - "FeatureName_INDICATOR_param1_param2" (e.g., "BidClose_EMA_10")
        - "INDICATOR_param" (e.g., "RSI_14")
        
        Args:
            ti_full_name: Full technical indicator name string
            
        Returns:
            Tuple of (column_name, indicator_method_name, params_dict)
        """
        parts = ti_full_name.split('_')
        
        column_name: Optional[str] = None
        indicator_method_name: Optional[str] = None
        params: Dict[str, Any] = {}
        
        # Known indicator keywords for pattern recognition
        known_indicator_keywords = [
            "EMA", "SMA", "RSI", "BBANDS", "MACD", "ROC", "MOM", "STOCH", 
            "STOCHRSI", "TSI", "UO", "WILLR", "ATR", "ADX", "CCI", "PSAR"
        ]
        
        # Pattern 1: Col_Indicator_Param (e.g., BidClose_EMA_10)
        if len(parts) >= 2 and parts[1].upper() in known_indicator_keywords:
            column_name = parts[0]
            indicator_method_name = parts[1].lower()
            param_parts = parts[2:]
        # Pattern 2: Indicator_Param (e.g., RSI_14, assuming default column)
        elif len(parts) >= 1 and parts[0].upper() in known_indicator_keywords:
            indicator_method_name = parts[0].lower()
            param_parts = parts[1:]
        else:
            logger.warning(f"TensorFlowTALayer: Could not parse TI name '{ti_full_name}'.")
            return None, None, {}
        
        # Parse parameters based on indicator type
        if indicator_method_name in ["ema", "sma", "rsi", "atr", "cci", "roc", "mom", "willr"]:
            if len(param_parts) > 0:
                try:
                    params['length'] = int(param_parts[0])
                except ValueError:
                    logger.warning(f"TensorFlowTALayer: Could not parse length from {param_parts[0]} for {ti_full_name}.")
        
        elif indicator_method_name == "stoch":
            if len(param_parts) >= 1: params['k'] = int(param_parts[0])
            if len(param_parts) >= 2: params['d'] = int(param_parts[1])
            if len(param_parts) >= 3: params['smooth_k'] = int(param_parts[2])
        
        elif indicator_method_name == "bbands":
            if len(param_parts) > 0:
                try: params['length'] = int(param_parts[0])
                except ValueError: pass
            if len(param_parts) > 1:
                try: params['std'] = float(param_parts[1])
                except ValueError: pass
        
        elif indicator_method_name == "macd":
            if len(param_parts) == 3:
                try:
                    params['fast'] = int(param_parts[0])
                    params['slow'] = int(param_parts[1])
                    params['signal'] = int(param_parts[2])
                except ValueError: pass
            elif len(param_parts) == 2:
                try:
                    params['fast'] = int(param_parts[0])
                    params['slow'] = int(param_parts[1])
                except ValueError: pass
        
        # Store column hint if parsed
        if column_name and 'column' not in params:
            params['column_hint'] = column_name
        
        return column_name, indicator_method_name, params
    
    @staticmethod
    def _tf_calculate_ema_1d(series: tf.Tensor, length: int, smoothing: float = 2.0) -> tf.Tensor:
        """
        Calculate Exponential Moving Average using TensorFlow operations.
        
        Args:
            series: 1D tensor of shape (seq_len,)
            length: EMA period
            smoothing: Smoothing factor (typically 2.0)
            
        Returns:
            1D tensor with EMA values
        """
        seq_len_tf = tf.shape(series)[0]
        
        if length <= 0:
            logger.warning(f"TensorFlowTALayer: EMA length must be positive, got {length}. Returning zeros.")
            return tf.zeros_like(series, dtype=tf.float32)
        
        if tf.cast(seq_len_tf, tf.int32) == 0:
            return tf.zeros_like(series, dtype=tf.float32)
        
        alpha = tf.cast(smoothing, tf.float32) / (tf.cast(length, tf.float32) + 1.0)
        
        # Initialize TensorArray for EMA values
        ema_values_array = tf.TensorArray(dtype=tf.float32, size=seq_len_tf, dynamic_size=False, clear_after_read=False)
        
        # First EMA value is the first series value
        first_val = tf.cond(seq_len_tf > 0, lambda: series[0], lambda: tf.constant(0.0, dtype=tf.float32))
        ema_values_array = tf.cond(seq_len_tf > 0, lambda: ema_values_array.write(0, first_val), lambda: ema_values_array)
        
        # Compute subsequent EMA values
        loop_counter_init = tf.constant(1, dtype=tf.int32)
        prev_ema_init = first_val
        
        def cond(i, prev_ema, arr):
            return i < seq_len_tf
        
        def body(i, prev_ema, arr):
            current_val = series[i]
            new_ema = alpha * current_val + (1.0 - alpha) * prev_ema
            arr = arr.write(i, new_ema)
            return i + 1, new_ema, arr
        
        # Execute loop only if sequence length > 1
        _, _, final_ema_values_array = tf.cond(
            seq_len_tf > 1,
            lambda: tf.while_loop(
                cond, body, [loop_counter_init, prev_ema_init, ema_values_array],
                parallel_iterations=1
            ),
            lambda: (loop_counter_init, prev_ema_init, ema_values_array)
        )
        
        return final_ema_values_array.stack()
    
    @staticmethod
    def _tf_calculate_rsi_1d(series: tf.Tensor, length: int) -> tf.Tensor:
        """
        Calculate Relative Strength Index using TensorFlow operations.
        
        Args:
            series: 1D tensor of shape (seq_len,)
            length: RSI period
            
        Returns:
            1D tensor with RSI values
        """
        seq_len_tf = tf.shape(series)[0]
        
        if length <= 0:
            logger.warning(f"TensorFlowTALayer: RSI length must be positive, got {length}. Returning zeros.")
            return tf.zeros_like(series, dtype=tf.float32)
        
        if tf.cast(seq_len_tf, tf.int32) <= 1:
            return tf.zeros_like(series, dtype=tf.float32)
        
        # Calculate price changes
        price_changes = series[1:] - series[:-1]
        
        # Separate gains and losses
        gains = tf.maximum(price_changes, 0.0)
        losses = tf.maximum(-price_changes, 0.0)
        
        # Pad with zero for first element (no change)
        gains = tf.concat([tf.constant([0.0], dtype=tf.float32), gains], axis=0)
        losses = tf.concat([tf.constant([0.0], dtype=tf.float32), losses], axis=0)
        
        # Calculate average gains and losses using EMA
        avg_gains = TensorFlowTALayer._tf_calculate_ema_1d(gains, length)
        avg_losses = TensorFlowTALayer._tf_calculate_ema_1d(losses, length)
        
        # Calculate RSI
        rs = tf.where(avg_losses > 0, avg_gains / avg_losses, tf.ones_like(avg_gains) * 100.0)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        
        return rsi
    
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Forward pass - calculate technical indicators and concatenate with base features.
        
        Args:
            inputs: Input tensor of shape (batch_size, seq_len, num_base_features)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, num_total_features)
        """
        calculated_ti_tensors_list = []
        
        # Process each technical indicator
        for ti_info in self.parsed_tis_info:
            ti_full_name = ti_info["full_name"]
            parsed_col_name = ti_info["col"]
            indicator_lower = ti_info["ind"]
            parsed_params = ti_info["params"]
            
            # Initialize tensor for current TI (zeros by default)
            current_ti_tensor_for_concat = tf.zeros((tf.shape(inputs)[0], tf.shape(inputs)[1], 1), dtype=tf.float32)
            
            if indicator_lower:
                if indicator_lower == "ema":
                    if parsed_col_name in self.base_feature_name_to_idx:
                        col_idx = self.base_feature_name_to_idx[parsed_col_name]
                        feature_series_batch = inputs[:, :, col_idx]
                        
                        ema_length = parsed_params.get('length')
                        if ema_length and isinstance(ema_length, int) and ema_length > 0:
                            ema_output_2d_batch = tf.map_fn(
                                lambda s: self._tf_calculate_ema_1d(s, length=ema_length),
                                feature_series_batch,
                                fn_output_signature=tf.float32
                            )
                            current_ti_tensor_for_concat = tf.expand_dims(ema_output_2d_batch, axis=-1)
                        else:
                            logger.warning(f"TensorFlowTALayer: Invalid EMA length for '{ti_full_name}'. Using zeros.")
                    else:
                        logger.warning(f"TensorFlowTALayer: Column '{parsed_col_name}' not found for EMA '{ti_full_name}'. Using zeros.")
                
                elif indicator_lower == "rsi":
                    col_to_use = parsed_col_name
                    if not col_to_use and 'close' in self.base_feature_name_to_idx:
                        col_to_use = 'close'
                        logger.info(f"TensorFlowTALayer: RSI '{ti_full_name}' using default 'close' column.")
                    
                    if col_to_use and col_to_use in self.base_feature_name_to_idx:
                        col_idx = self.base_feature_name_to_idx[col_to_use]
                        feature_series_batch = inputs[:, :, col_idx]
                        
                        rsi_length = parsed_params.get('length')
                        if rsi_length and isinstance(rsi_length, int) and rsi_length > 0:
                            rsi_output_2d_batch = tf.map_fn(
                                lambda s: self._tf_calculate_rsi_1d(s, length=rsi_length),
                                feature_series_batch,
                                fn_output_signature=tf.float32
                            )
                            current_ti_tensor_for_concat = tf.expand_dims(rsi_output_2d_batch, axis=-1)
                        else:
                            logger.warning(f"TensorFlowTALayer: Invalid RSI length for '{ti_full_name}'. Using zeros.")
                    else:
                        logger.warning(f"TensorFlowTALayer: No valid column found for RSI '{ti_full_name}'. Using zeros.")
                
                else:
                    logger.debug(f"TensorFlowTALayer: TI '{indicator_lower}' not implemented. Using zeros.")
            else:
                logger.debug(f"TensorFlowTALayer: TI '{ti_full_name}' unparsable. Using zeros.")
            
            calculated_ti_tensors_list.append(current_ti_tensor_for_concat)
        
        # Concatenate base features with calculated technical indicators
        if calculated_ti_tensors_list:
            output_tensor = tf.concat([inputs] + calculated_ti_tensors_list, axis=-1)
        else:
            output_tensor = inputs
            logger.debug("TensorFlowTALayer: No TI features calculated. Output is input features only.")
        
        return output_tensor
    
    def compute_output_shape(self, input_shape):
        """
        Compute output shape based on input shape.
        
        Args:
            input_shape: Input tensor shape
            
        Returns:
            Output tensor shape tuple
        """
        new_feature_count = self.num_base_features + self.num_ti_features_to_calc
        return (input_shape[0], input_shape[1], new_feature_count)
