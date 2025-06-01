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
from copy import deepcopy
from typing import Any, Dict, List, Tuple, Union, Optional

# Assuming your generator plugin (like VAEGeneratorPlugin) and a new discriminator model definition exist
# from .generator_plugin import VAEGeneratorPlugin # Or your specific generator plugin
# from .discriminator_model import build_discriminator # You'll create this

# Initialize logger for this module
logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO) # Set default log level - Assuming this is set elsewhere or defaults appropriately
# ADD INFO MESSAGE ABOUT GENERATOR WARNING
# logger.info("GANTrainerPlugin: The VAE generator is intended to be frozen during GAN training. A \\'UserWarning: The model does not have any trainable weights.\\' may appear when generator.predict() is called; this is expected for the frozen generator and does not affect discriminator or GAN training.")


# Custom Keras Layer for Technical Indicator Calculation using tf.numpy_function
class TensorFlowTALayer(layers.Layer):
    def __init__(self, base_feature_names: List[str], ti_names_to_calculate: List[str],
                 num_base_features: int, num_total_features: int, seq_len: int, **kwargs):
        super().__init__(**kwargs)
        self.base_feature_names = base_feature_names
        self.ti_names_to_calculate = ti_names_to_calculate
        self.num_base_features = num_base_features
        self.num_total_features = num_total_features # num_base_features + num_tis
        self.seq_len = seq_len

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
        
        # Heuristic: If the first part matches a base feature name, it's likely the target column.
        # This is not a perfect heuristic and depends on naming conventions.
        # A more robust system might require explicit TI definitions (base_col, type, params).
        
        # Try to identify if the TI implies a specific column from base_feature_names
        # This part is tricky and relies on conventions.
        # For "BidClose_EMA_10", parts = ["BidClose", "EMA", "10"]
        # For "EMA_10" (expecting 'close'), parts = ["EMA", "10"]
        # For "ATR_14" (uses H,L,C), parts = ["ATR", "14"]

        # Let's assume a convention:
        # 1. If TI name starts with a base_feature_name + '_', that's the column. E.g. "BidClose_EMA_10"
        # 2. Else, the TI might be column-agnostic or use pandas_ta defaults (like 'close').
        
        column_name: Optional[str] = None
        indicator_method_name: Optional[str] = None
        params: Dict[str, Any] = {}

        # Tentative parsing logic - this may need refinement based on actual ti_names_to_calculate format
        potential_col_name = parts[0]
        # A simple check: if potential_col_name is a known base feature, assume it is.
        # This requires passing base_feature_names to the parser or having it accessible.
        # For now, let's assume if parts[1] is a common indicator, parts[0] is the column.
        
        # Simplified parsing based on common patterns observed in _calculate_technical_indicators
        # This needs to be robust for all TIs used.
        
        # Pattern 1: Col_Indicator_Param (e.g., BidClose_EMA_10)
        if len(parts) >= 2 and parts[1].upper() in ["EMA", "SMA", "RSI", "BBANDS", "MACD", "ROC", "MOM", "STOCH", "STOCHRSI", "TSI", "UO", "WILLR", "ATR", "ADX", "CCI", "PSAR"]: # Common indicators
            column_name = parts[0]
            indicator_method_name = parts[1].lower()
            param_parts = parts[2:]
        # Pattern 2: Indicator_Param (e.g., RSI_14, assuming default column like 'close')
        elif len(parts) >= 1 and parts[0].upper() in ["EMA", "SMA", "RSI", "BBANDS", "MACD", "ROC", "MOM", "STOCH", "STOCHRSI", "TSI", "UO", "WILLR", "ATR", "ADX", "CCI", "PSAR"]:
            indicator_method_name = parts[0].lower()
            param_parts = parts[1:]
        else: # Cannot parse reliably
            logger.warning(f"TensorFlowTALayer: Could not reliably parse TI name '{ti_full_name}'. Skipping this TI in symbolic graph.")
            return None, None, {}

        # Parameter parsing (simplified)
        if indicator_method_name in ["ema", "sma", "rsi", "atr", "cci", "roc", "mom", "stoch", "stochrsi", "tsi", "uo", "willr", "psar"]: # Common length-based
            if len(param_parts) > 0:
                try: params['length'] = int(param_parts[0])
                except ValueError: logger.warning(f"TensorFlowTALayer: Could not parse length from {param_parts[0]} for {ti_full_name}.")
        elif indicator_method_name == "bbands":
            if len(param_parts) > 0: # length
                try: params['length'] = int(param_parts[0])
                except ValueError: logger.warning(f"TensorFlowTALayer: Could not parse BBands length from {param_parts[0]} for {ti_full_name}.")
            if len(param_parts) > 1: # std
                try: params['std'] = float(param_parts[1]) # or int
                except ValueError: logger.warning(f"TensorFlowTALayer: Could not parse BBands std from {param_parts[1]} for {ti_full_name}.")
        elif indicator_method_name == "macd":
            if len(param_parts) == 3: # fast, slow, signal
                try:
                    params['fast'] = int(param_parts[0])
                    params['slow'] = int(param_parts[1])
                    params['signal'] = int(param_parts[2])
                except ValueError: logger.warning(f"TensorFlowTALayer: Could not parse MACD params from {param_parts} for {ti_full_name}.")
            elif len(param_parts) == 2: # fast, slow (signal default)
                 try:
                    params['fast'] = int(param_parts[0])
                    params['slow'] = int(param_parts[1])
                 except ValueError: logger.warning(f"TensorFlowTALayer: Could not parse MACD fast/slow params from {param_parts} for {ti_full_name}.")

        # If column_name was parsed as part of the TI name (e.g. "BidClose" from "BidClose_EMA_10")
        # and it's not a standard pandas_ta recognized column name for the indicator (like 'high', 'low', 'close', 'volume')
        # then we need to pass it as the 'column' argument to the pandas_ta function.
        if column_name and indicator_method_name not in ['atr', 'adx', 'psar']: # These TIs might use HLCV implicitly
             # For TIs like EMA, RSI, etc., if a column was parsed, use it.
            if 'column' not in params: # Avoid overwriting if already parsed differently
                params['column'] = column_name
        
        # For TIs like ATR, ADX, PSAR, they expect 'high', 'low', 'close' columns.
        # If base_feature_names provides these (e.g. "BidHigh", "BidLow", "BidClose"),
        # we might need to rename them temporarily in the DataFrame for pandas_ta.
        # This adds complexity. For now, assume pandas_ta handles it or TIs are chosen carefully.

        return column_name, indicator_method_name, params

    @staticmethod
    def _calculate_tis_numpy_batch(inputs_numpy: np.ndarray, 
                                   base_feature_names_py: List[str], 
                                   ti_names_to_calculate_py: List[str], 
                                   seq_len_py: int,
                                   num_total_features_py: int):
        # inputs_numpy shape: (batch_size, seq_len, num_base_features)
        batch_size = inputs_numpy.shape[0]
        all_samples_processed = []

        for i in range(batch_size):
            sample_base_features_np = inputs_numpy[i, :, :] # (seq_len, num_base_features)
            df_for_tis = pd.DataFrame(sample_base_features_np, columns=base_feature_names_py)
            
            # Store original columns to select base features later
            original_base_df = df_for_tis[base_feature_names_py].copy()

            for ti_full_name in ti_names_to_calculate_py:
                # Use the refined parser
                # Note: _parse_ti_name is static, so it doesn't have access to self.base_feature_names
                # This implies base_feature_names needs to be passed to it, or parsing logic simplified.
                # For now, the parser makes some assumptions.
                # A better parser might need access to the list of base_feature_names to confirm if parts[0] is a column.
                
                # Simplified parsing for now, assuming ti_full_name is like "BidClose_EMA_10" or "RSI_14"
                # This parsing logic should ideally be robust and centralized if used elsewhere.
                parsed_col, parsed_indicator, parsed_params = TensorFlowTALayer._parse_ti_name(ti_full_name)

                if not parsed_indicator:
                    logger.warning(f"TensorFlowTALayer: Skipping TI {ti_full_name} due to parsing failure in batch.")
                    continue
                
                try:
                    indicator_func = getattr(df_for_tis.ta, parsed_indicator)
                    # Ensure 'column' param is correctly handled if parsed_col is set
                    if parsed_col and 'column' not in parsed_params and parsed_indicator not in ['atr', 'adx', 'psar', 'obv', 'cmf', 'efi', 'cmo', 'mfi', 'chop', 'vortex', 'aroon', 'donchian', 'kc', 'ichimoku', 'thermo', 'squeeze', 'sqzpro', 'inertia', 'trendflex', 'ttm_trend', 'vhf', 'vwap', 'vwma']: # Indicators that might not take a single 'column' or use OHLCV
                        # If parsed_col is a valid column in df_for_tis, use it.
                        if parsed_col in df_for_tis.columns:
                             parsed_params['column'] = parsed_col
                        else:
                            logger.warning(f"TensorFlowTALayer: Parsed column '{parsed_col}' for TI '{ti_full_name}' not in base features. Indicator may fail or use defaults.")
                    
                    # For TIs that need OHLCV, ensure columns are named appropriately or pandas_ta can find them.
                    # E.g., if base features are "BHigh", "BLow", "BClose", "BOpen", "BVolume"
                    # and pandas_ta expects "high", "low", "close", "open", "volume".
                    # A temporary rename might be needed:
                    # rename_map = {"BHigh": "high", "BLow": "low", ...}
                    # df_for_tis.rename(columns=rename_map, inplace=True)
                    # ... call indicator ...
                    # df_for_tis.rename(columns={v: k for k, v in rename_map.items()}, inplace=True) # Rename back
                    # This is complex. For now, assume names are compatible or TIs don't require strict OHLCV names.

                    indicator_func(**parsed_params, append=True, col_names=(ti_full_name,))
                except AttributeError:
                    logger.error(f"TensorFlowTALayer: TI method '{parsed_indicator}' not found in pandas_ta for {ti_full_name}.")
                except Exception as e:
                    logger.error(f"TensorFlowTALayer: Error calculating TI {ti_full_name} with params {parsed_params} (parsed_col: {parsed_col}): {e}")

            # Consolidate features: base features + calculated TIs
            final_ordered_columns = base_feature_names_py + ti_names_to_calculate_py
            
            # Start with original base features
            combined_df = original_base_df.copy()
            
            for ti_name in ti_names_to_calculate_py:
                if ti_name in df_for_tis.columns:
                    combined_df[ti_name] = df_for_tis[ti_name]
                else:
                    logger.warning(f"TensorFlowTALayer: TI column '{ti_name}' not found after calculation. Filling with zeros for this sample.")
                    combined_df[ti_name] = np.zeros(seq_len_py) 
            
            # Ensure the final DataFrame has the exact columns in the expected order and count
            # If a TI failed to compute, it's filled with zeros.
            # We need to ensure combined_df has all columns from final_ordered_columns.
            for col in final_ordered_columns:
                if col not in combined_df.columns:
                    logger.warning(f"TensorFlowTALayer: Column '{col}' was expected but not found. Adding as zeros.")
                    combined_df[col] = np.zeros(seq_len_py)
            
            combined_df = combined_df[final_ordered_columns]
            
            # Verify shape before converting to values
            if combined_df.shape[1] != num_total_features_py:
                logger.error(f"TensorFlowTALayer: Shape mismatch in combined_df. Expected {num_total_features_py} features, got {combined_df.shape[1]}. Columns: {combined_df.columns}")
                # Fallback: create a zero array of the correct shape to avoid downstream errors
                # This indicates a serious issue in TI calculation or column management.
                all_samples_processed.append(np.zeros((seq_len_py, num_total_features_py), dtype=np.float32))
            else:
                all_samples_processed.append(combined_df.values)

        processed_batch_np = np.array(all_samples_processed, dtype=np.float32)
        processed_batch_np = np.nan_to_num(processed_batch_np, nan=0.0) # Final NaN check
        
        # Final shape check
        if processed_batch_np.shape != (batch_size, seq_len_py, num_total_features_py):
            logger.error(f"TensorFlowTALayer: Output shape mismatch. Expected {(batch_size, seq_len_py, num_total_features_py)}, got {processed_batch_np.shape}. Forcing reshape if possible.")
            # Attempt to reshape or pad if feature count is off, though this is risky
            if processed_batch_np.shape[0] == batch_size and processed_batch_np.shape[1] == seq_len_py:
                 # If only feature count is wrong, pad with zeros or truncate (last resort)
                if processed_batch_np.shape[2] < num_total_features_py:
                    padding = np.zeros((batch_size, seq_len_py, num_total_features_py - processed_batch_np.shape[2]), dtype=np.float32)
                    processed_batch_np = np.concatenate([processed_batch_np, padding], axis=2)
                elif processed_batch_np.shape[2] > num_total_features_py:
                    processed_batch_np = processed_batch_np[:, :, :num_total_features_py]
            else: # More severe shape mismatch, return zeros of expected shape
                 processed_batch_np = np.zeros((batch_size, seq_len_py, num_total_features_py), dtype=np.float32)


        return processed_batch_np

    def call(self, inputs): # inputs is a KerasTensor (symbolic)
        # inputs shape: (batch_size, self.seq_len, self.num_base_features)
        
        # tf.numpy_function expects Python native types for non-tensor arguments.
        # self.base_feature_names, self.ti_names_to_calculate are already List[str]
        # self.seq_len, self.num_total_features are Python int
        
        y = tf.numpy_function(
            TensorFlowTALayer._calculate_tis_numpy_batch,
            [
                inputs, 
                self.base_feature_names, 
                self.ti_names_to_calculate,
                self.seq_len,
                self.num_total_features 
            ],
            tf.float32 # Output type
        )
        
        # Set the shape of the output tensor because tf.numpy_function loses shape information.
        # The batch size can be dynamic (None or tf.shape(inputs)[0]).
        output_shape = [None, self.seq_len, self.num_total_features] # MODIFIED: Use None for symbolic batch size
        y.set_shape(output_shape)
        return y

    def get_config(self):
        config = super().get_config()
        config.update({
            "base_feature_names": self.base_feature_names,
            "ti_names_to_calculate": self.ti_names_to_calculate,
            "num_base_features": self.num_base_features,
            "num_total_features": self.num_total_features,
            "seq_len": self.seq_len,
        })
        return config

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
        "gan_model_dir": "models/gan_trained",
        # Discriminator architecture params
        "discriminator_conv_filters": [64, 128], # List of filters for Conv1D layers
        "discriminator_conv_kernel_size": 3,      # Kernel size for Conv1D layers
        "discriminator_lstm_units": 64,
        "discriminator_dropout_rate": 0.3,
        # "discriminator_dense_units": 128, # REMOVED - Replaced by Conv1D/LSTM focus
        "gan_generator_output_actual_seq_len": 1,
        # Callbacks parameters
        "enable_reduce_lr_on_plateau": True,
        "lr_reduction_factor": 0.5, # Factor by which LR is reduced
        "lr_patience": 50,          # Epochs to wait for improvement before reducing LR
        "lr_min_delta": 0.001,      # Minimum change to qualify as improvement for LR reduction
        "min_lr_g": 1e-7,           # Minimum LR for generator
        "min_lr_d": 1e-7,           # Minimum LR for discriminator
        "lr_monitor_metric": "g_loss", # "g_loss" or "d_loss"
        "enable_early_stopping": True,
        "es_patience": 200,         # Epochs to wait for improvement before stopping
        "es_min_delta": 0.001,      # Minimum change to qualify as improvement for early stopping
        "es_monitor_metric": "g_loss", # "g_loss" or "d_loss"
    }

    def __init__(self, config: Dict[str, Any], generator_plugin_instance: Optional[Any] = None, feeder_plugin_instance: Optional[Any] = None, preprocessor_plugin_instance: Optional[Any] = None):
        self.logger = logger # Ensure logger is accessible, e.g., global or from a utility
        self.config = config
        self.params = self.plugin_params.copy()
        self.params.update(self.config) # Initial population of self.params

        self.generator_plugin_instance = generator_plugin_instance
        self.feeder_plugin_instance = feeder_plugin_instance
        self.preprocessor_plugin_instance = preprocessor_plugin_instance

        # Initialize self.generator and related attributes
        if self.generator_plugin_instance and hasattr(self.generator_plugin_instance, 'sequential_model'):
            self.generator: Optional[Model] = self.generator_plugin_instance.sequential_model
            if self.generator:
                self.generator_actual_input_names_ordered = [inp.name.split(':')[0] for inp in self.generator.inputs]
                if self.generator.output_shape and len(self.generator.output_shape) > 1 and isinstance(self.generator.output_shape[-1], int):
                    self.actual_generator_output_dim = self.generator.output_shape[-1]
                else:
                    self.logger.warning(f"Could not reliably determine actual_generator_output_dim from generator output shape: {self.generator.output_shape}. Defaulting to 0.")
                    self.actual_generator_output_dim = 0
                
                configured_num_base_features = len(self.params.get("base_feature_names_ordered", []))
                if self.actual_generator_output_dim != 0 and configured_num_base_features > 0 and self.actual_generator_output_dim != configured_num_base_features:
                    self.logger.warning(
                        f"GANTrainerPlugin (__init__): MISMATCH! Actual generator output feature dimension ({self.actual_generator_output_dim}) "
                        f"differs from configured num_base_features ({configured_num_base_features}). "
                        "_build_gan will attempt to slice."
                    )
            else:
                self.logger.error("Generator model (sequential_model) is None in generator_plugin_instance.")
                self.generator = None
                self.generator_actual_input_names_ordered = []
                self.actual_generator_output_dim = 0
        else:
            self.logger.error("GeneratorPlugin instance not provided or does not have 'sequential_model'.")
            self.generator = None
            self.generator_actual_input_names_ordered = []
            self.actual_generator_output_dim = 0

        # Initialize core parameters from config needed for model building
        self._initialize_core_parameters_from_config()

        # Initialize dimensions for GAN model inputs (latent is from core_params, conditional/context here)
        self.conditional_dim_for_generator = 0
        if self.feeder_plugin_instance and hasattr(self.feeder_plugin_instance, 'get_conditional_dim'): # CORRECTED
            self.conditional_dim_for_generator = self.feeder_plugin_instance.get_conditional_dim() # CORRECTED
            self.logger.info(f"GANTrainerPlugin (__init__): conditional_dim_for_generator from feeder: {self.conditional_dim_for_generator}")
        elif self.params: # Fallback to params if feeder doesn't provide it directly
            date_cond_feats = self.params.get('feeder_date_features_for_conditioning', [])
            fund_cond_feats = self.params.get('feeder_fundamental_features_for_conditioning', [])
            # This logic assumes simple concatenation. Adjust if FeederPlugin structures conditions differently.
            self.conditional_dim_for_generator = (len(date_cond_feats) * 2) + len(fund_cond_feats) 
            self.logger.info(f"GANTrainerPlugin (__init__): conditional_dim_for_generator from params (fallback): {self.conditional_dim_for_generator} (date_feats*2: {len(date_cond_feats)*2}, fund_feats: {len(fund_cond_feats)})")
        else:
            self.logger.warning("GANTrainerPlugin (__init__): Could not determine conditional_dim_for_generator.")

        self.context_dim_for_generator = self.params.get('context_vector_dim', 64) # from config
        self.logger.info(f"GANTrainerPlugin (__init__): context_dim_for_generator set to {self.context_dim_for_generator}")
        
        # Initialize optimizers
        self.g_optimizer = Adam(learning_rate=self.params.get("generator_lr", 1e-4), beta_1=self.params.get("generator_beta1", 0.5))
        self.d_optimizer = Adam(learning_rate=self.params.get("discriminator_lr", 1e-4), beta_1=self.params.get("discriminator_beta1", 0.5))
        self.logger.info("Optimizers initialized.")

        # Build models
        self.discriminator: Optional[Model] = self._build_discriminator()
        if self.discriminator:
            self.discriminator.compile(loss='binary_crossentropy', optimizer=self.d_optimizer, metrics=['accuracy'])
            self.logger.info("Discriminator compiled.")
        else:
            self.logger.error("Discriminator model building failed in __init__.")

        self.gan: Optional[Model] = self._build_gan() # _build_gan also compiles the GAN model with self.g_optimizer
        if not self.gan:
             self.logger.error("GAN model building failed in __init__.")

        # For manual callbacks
        self.best_lr_metric = float('inf')
        self.lr_patience_counter = 0
        self.best_es_metric = float('inf')
        self.es_patience_counter = 0
        self.logger.info("GANTrainerPlugin initialized.")

    def _initialize_core_parameters_from_config(self):
        """Helper to initialize parameters needed before model building, typically also managed by set_params."""
        self.gen_input_seq_len = self.params.get("seq_len", 18)
        self.gen_input_latent_dim = self.params.get("latent_dim", 32) # Used for GAN latent input
        
        self.base_feature_names = self.params.get("base_feature_names_ordered", [])
        self.num_base_features = len(self.base_feature_names)
        
        # Derive ti_names_to_calculate from discriminator features and base features
        all_discriminator_features = self.params.get("feature_names_for_discriminator_ordered", [])
        if all_discriminator_features and self.base_feature_names:
            # This assumes base_feature_names are a subset and appear first in all_discriminator_features
            self.ti_names_to_calculate = [f for f in all_discriminator_features if f not in self.base_feature_names]
        else:
            self.ti_names_to_calculate = []
            self.logger.warning("_initialize_core_parameters_from_config: Could not derive ti_names_to_calculate due to missing feature lists in params.")


        self.num_tis = len(self.ti_names_to_calculate)
        
        self.num_features_for_discriminator = self.num_base_features + self.num_tis
        self.seq_len = self.params.get("seq_len", 18) # Used for discriminator input and TI layer

        self.generator_output_actual_seq_len = self.params.get("gan_generator_output_actual_seq_len", self.seq_len)

        self.logger.info(
            f"GANTrainerPlugin (_initialize_core_parameters): "
            f"gen_input_seq_len={self.gen_input_seq_len}, gen_input_latent_dim={self.gen_input_latent_dim}, "
            f"num_base_features={self.num_base_features} (from {len(self.base_feature_names)} names), "
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
        self.base_feature_names = self.params.get("base_feature_names_ordered", [])
        self.num_base_features = len(self.base_feature_names)
        
        self.ti_names_to_calculate = self.params.get("ti_names_to_calculate", [])
        self.num_tis = len(self.ti_names_to_calculate)
        
        # This is crucial for discriminator input shape and TI layer
        self.num_features_for_discriminator = self.num_base_features + self.num_tis
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

        self.gan_model_dir = self.params.get("gan_model_dir", "models/gan_trained")
        os.makedirs(self.gan_model_dir, exist_ok=True)

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
        if not self.generator or not self.discriminator:
            logger.error("Generator or Discriminator not initialized. Cannot build GAN.")
            raise ValueError("Generator or Discriminator not initialized.")
            
        logger.info("Building GAN model (Generator + Discriminator)...")
        self.discriminator.trainable = False # Ensure discriminator is frozen for GAN
        
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
            found_match = False
            for cfg_name, gan_layer in input_map.items():
                if cfg_name and model_input_name.startswith(cfg_name):
                    generator_feed_inputs_ordered.append(gan_layer)
                    found_match = True
                    break
            if not found_match:
                logger.warning(f"GAN build: Generator model input '{model_input_name}' not mapped from GAN inputs. Check config names: latent='{cfg_latent_name}', context='{cfg_context_name}', conditional='{cfg_conditional_name}'.")
        
        if len(generator_feed_inputs_ordered) != len(self.generator.inputs):
             logger.error(f"GAN build: Mismatch in ordered inputs for generator. Expected {len(self.generator.inputs)}, got {len(generator_feed_inputs_ordered)}. This is critical.")

        generated_data_raw = self.generator(generator_feed_inputs_ordered) # Shape: (batch, seq_len_gen_output, actual_generator_output_dim)
        
        # Ensure generator output is 3D for the TI layer and discriminator
        # self.generator_output_actual_seq_len should reflect the generator\\'s intended output sequence length (e.g., 1 or self.seq_len)
        # self.actual_generator_output_dim is the number of features from the generator model
        
        # If the generator might output 2D tensors (e.g., shape (batch, features) for seq_len=1)
        # and it needs to be 3D (batch, seq_len, features) for subsequent layers.
        # We use self.generator_output_actual_seq_len to guide this.
        if self.generator_output_actual_seq_len > 0:
            logger.info(f"GAN build: Ensuring generator output is 3D. Configured actual_seq_len: {self.generator_output_actual_seq_len}, actual_dim: {self.actual_generator_output_dim if hasattr(self, 'actual_generator_output_dim') else 'Not Set'}.")
            target_reshape_dim = -1
            if hasattr(self, 'actual_generator_output_dim') and self.actual_generator_output_dim > 0:
                target_reshape_dim = self.actual_generator_output_dim
            
            generated_data_raw = tf.keras.layers.Reshape(
                (self.generator_output_actual_seq_len, target_reshape_dim), 
                name="reshape_generator_output"
            )(generated_data_raw)
            logger.info(f"GAN build: Reshaped generator output to symbolic shape (None, {self.generator_output_actual_seq_len}, {target_reshape_dim}). Actual shape after reshape: {generated_data_raw.shape}")

        base_features_from_generator = generated_data_raw
        if hasattr(self, 'actual_generator_output_dim') and self.actual_generator_output_dim != self.num_base_features:
            logger.info(f"GAN build: Slicing generator output\\'s feature dimension from {self.actual_generator_output_dim} to {self.num_base_features} features.")
            base_features_from_generator = tf.keras.layers.Lambda(
                lambda x: x[:, :, :self.num_base_features], name="slice_gen_output_to_base_features"
            )(generated_data_raw)
            logger.info(f"GAN build: Sliced base_features_from_generator shape: {base_features_from_generator.shape}.")
        
        if self.generator_output_actual_seq_len != self.seq_len:
            if self.generator_output_actual_seq_len == 1 and self.seq_len > 1:
                logger.info(f"GAN build: Generator output seq_len ({self.generator_output_actual_seq_len}) differs from TI/Discriminator seq_len ({self.seq_len}). Repeating generator output.")
                # base_features_from_generator shape: (None, 1, self.num_base_features)
                # Squeeze the middle dimension (axis=1) before RepeatVector
                squeezed_features = tf.keras.layers.Lambda(
                    lambda x: tf.squeeze(x, axis=1),
                    name="squeeze_time_dim_for_repeat"
                )(base_features_from_generator)
                # Now squeezed_features shape is (None, self.num_base_features)
                base_features_from_generator = tf.keras.layers.RepeatVector(
                    self.seq_len,
                    name="repeat_vector_for_ti_layer"
                )(squeezed_features) # Apply RepeatVector to the 2D tensor
                logger.info(f"GAN build: Repeated generator output to shape: {base_features_from_generator.shape}")
            else:
                error_msg = (
                    f"GAN build: Mismatch between effective generator output sequence length ({self.generator_output_actual_seq_len}) "
                    f"and TI layer / Discriminator expected sequence length ({self.seq_len}). "
                    f"Cannot automatically reconcile unless generator_output_actual_seq_len is 1."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

        # Now, base_features_from_generator should be (None, self.seq_len, self.num_base_features)
        # e.g., (None, 18, 21)

        ti_calculator_layer = TensorFlowTALayer(
            base_feature_names=self.base_feature_names,
            ti_names_to_calculate=self.ti_names_to_calculate,
            num_base_features=self.num_base_features,
            num_total_features=self.num_features_for_discriminator, # This is num_base_features + num_ti_features
            seq_len=self.seq_len, # This must match the input seq_len to this layer
            name="symbolic_ti_calculator"
        )
        
        data_with_tis_for_discriminator = ti_calculator_layer(base_features_from_generator)
        # Expected shape: (None, self.seq_len, self.num_features_for_discriminator) -> (None, 18, 41)
        
        gan_output = self.discriminator(data_with_tis_for_discriminator) 
        
        actual_gan_model_inputs = [gan_latent_input, gan_conditional_input, gan_context_input]
        gan = tf.keras.Model(inputs=actual_gan_model_inputs, outputs=gan_output, name="gan_combined")
        gan.compile(loss='binary_crossentropy', optimizer=self.g_optimizer)
        logger.info("GAN model built and compiled.")
        gan.summary(print_fn=logger.info)
        try:
            plot_model(gan, to_file=os.path.join(self.gan_model_dir, 'gan_model_plot.png'), show_shapes=True, show_layer_names=True, expand_nested=True)
            logger.info(f"GAN model plot saved to {os.path.join(self.gan_model_dir, 'gan_model_plot.png')}")
            if self.generator:
                 plot_model(self.generator, to_file=os.path.join(self.gan_model_dir, 'gan_generator_component_plot.png'), show_shapes=True, show_layer_names=True, expand_nested=True)
                 logger.info(f"GAN's generator component plot saved to {os.path.join(self.gan_model_dir, 'gan_generator_component_plot.png')}")

        except Exception as e:
            logger.warning(f"Could not plot GAN model: {e}. Ensure pydot and graphviz are installed.")
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
        
        if x_train_sequences.ndim == 2: # Should be 3D, but if num_features is 1 and seq_len is also 1, it might become 2D.
            # This case might indicate an issue or a very specific scenario.
            # For now, let's ensure it's 3D if num_samples > 0
            if num_samples > 0 :
                 x_train_sequences = x_train_sequences.reshape(num_samples, seq_len, num_features if num_features > 0 else 1)
            else: # if num_samples is 0, it means not enough data to form even one sequence
                self.logger.error(f"GANTrainerPlugin (train): Not enough data to form any sequences. num_samples: {num_samples}, seq_len: {seq_len}, data_shape: {x_train_data_full.shape}")
                # Decide how to handle this: raise error, return, or proceed with empty data if the rest of the code can handle it.
                # For now, let's make it an empty array with the correct number of dimensions if possible, or raise error.
                if num_features > 0:
                    x_train_sequences = np.empty((0, seq_len, num_features))
                else: # This case (0 features) is highly problematic
                    self.logger.error(f"GANTrainerPlugin (train): Data has 0 features after processing. Original shape: {data_df.shape}, after numpy conversion: {x_train_data_full.shape}")
                    raise ValueError("Data has 0 features after processing.")


        self.logger.info(f"GANTrainerPlugin (train): Created sequences. Shape: {x_train_sequences.shape}")
        
        # x_train_data = np.load(x_train_file, allow_pickle=True) # Old way, for .npy files
        x_train_data = x_train_sequences # Use the processed sequences

        if x_train_data.ndim == 2:
            x_train_data = np.expand_dims(x_train_data, axis=1) # Add sequence length dimension
            self.logger.info(f"GANTrainerPlugin (train): Expanded dims for 2D data. New shape: {x_train_data.shape}")

        # Adversarial ground truths
        valid = np.ones((x_train_data.shape[0], 1))
        fake = np.zeros((x_train_data.shape[0], 1))

        # For storing losses
        d_losses_history = []
        g_losses_history = []
        d_accs_history = []

        # Initialize learning rates for manual callback tracking
        current_lr_g = float(tf.keras.backend.get_value(self.g_optimizer.learning_rate))
        current_lr_d = float(tf.keras.backend.get_value(self.d_optimizer.learning_rate))

        logger.info(f"Starting GAN training for {self.params['gan_epochs']} epochs with batch size {self.params['gan_batch_size']}...")
        logger.info(f"Initial Generator LR: {current_lr_g:.1e}, Discriminator LR: {current_lr_d:.1e}")
        logger.info(f"Generator is FROZEN. Discriminator is TRAINABLE.")
        logger.info(f"Discriminator input shape: (batch_size, {self.seq_len}, {self.num_features_for_discriminator})")
        logger.info(f"Generator output (base features) shape: (batch_size, {self.seq_len}, {self.num_base_features})")


        for epoch in range(self.params["gan_epochs"]):
            start_time_epoch = time.time()
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            
            # Get a random batch of real samples
            idx = np.random.randint(0, x_train_data.shape[0], self.params["gan_batch_size"])
            real_data_batch_raw = x_train_data[idx] # (batch_size, seq_len, features_from_preprocessor)
            
            real_data_for_discriminator = real_data_batch_raw[:, :, :self.num_features_for_discriminator]

            # Generate a batch of new series using the FeederPlugin and Generator
            feeder_output = self.feeder_plugin_instance.generate(n_ticks_to_generate=self.params["gan_batch_size"])
            
            # Prepare inputs for the generator model in the correct order
            # self.generator_actual_input_names_ordered is set in __init__ based on generator.inputs
            generator_inputs = []
            if not hasattr(self, 'generator_actual_input_names_ordered'):
                self.logger.error("GANTrainerPlugin attribute 'generator_actual_input_names_ordered' not found. Was __init__ properly called?")
                raise AttributeError("'generator_actual_input_names_ordered' not found.")

            for input_name in self.generator_actual_input_names_ordered:
                if input_name not in feeder_output:
                    self.logger.error(f"KeyError: Feeder output missing expected key '{input_name}' for generator input. Available keys: {list(feeder_output.keys())}")
                    raise KeyError(f"Feeder output missing expected key '{input_name}' for generator input.")
                generator_inputs.append(feeder_output[input_name])
            
            generated_base_features_raw = self.generator.predict(generator_inputs, verbose=0)

            # Ensure generated_base_features_raw is 3D before TI calculation
            # This mirrors the Reshape layer in _build_gan for the standalone generator's output
            if generated_base_features_raw.ndim == 2:
                # Infer target_reshape_dim which is self.actual_generator_output_dim
                # self.actual_generator_output_dim should be set correctly in __init__
                target_reshape_dim = -1 # Default, works if only one dimension is unspecified
                if hasattr(self, 'actual_generator_output_dim') and self.actual_generator_output_dim > 0:
                    target_reshape_dim = self.actual_generator_output_dim
                
                # self.generator_output_actual_seq_len should also be correctly set
                # It defines the sequence length dimension for the reshape operation
                if hasattr(self, 'generator_output_actual_seq_len') and self.generator_output_actual_seq_len > 0:
                    try:
                        generated_base_features_raw = generated_base_features_raw.reshape(
                            (self.params["gan_batch_size"], self.generator_output_actual_seq_len, target_reshape_dim)
                        )
                        self.logger.info(f"GANTrainerPlugin (train): Reshaped generator output from 2D to 3D. New shape: {generated_base_features_raw.shape}")
                    except ValueError as e:
                        self.logger.error(f"GANTrainerPlugin (train): Error reshaping 2D generator output to 3D. Shape was {generated_base_features_raw.shape}, target_reshape_dim={target_reshape_dim}, generator_output_actual_seq_len={self.generator_output_actual_seq_len}. Error: {e}")
                        raise
                else:
                    self.logger.error("GANTrainerPlugin (train): Cannot reshape 2D generator output. 'generator_output_actual_seq_len' is not properly set.")
                    # Potentially raise an error or handle as critical failure
                    raise AttributeError("'generator_output_actual_seq_len' not set, cannot reshape generator output.")


            # Calculate TIs for generated data
            generated_features_with_tis = self._calculate_technical_indicators(generated_base_features_raw) # (batch, seq_len, num_features_for_discriminator)
            generated_data_for_discriminator = generated_features_with_tis

            # Train the discriminator
            # Note: tf.GradientTape is used for custom training loops.
            # If using model.compile() and model.train_on_batch(), this is handled internally.
            # The current structure seems to use train_on_batch.
            
            # d_loss_real = self.discriminator.train_on_batch(real_data_for_discriminator, valid)
            # d_loss_fake = self.discriminator.train_on_batch(generated_data_for_discriminator, fake)
            # d_loss = 0.5 * np.add(d_loss_real[0], d_loss_fake[0]) # Loss is typically the first element
            # d_acc = 0.5 * np.add(d_loss_real[1], d_loss_fake[1]) # Accuracy is typically the second

            d_loss_real_metrics = self.discriminator.train_on_batch(real_data_for_discriminator, valid)
            d_loss_fake_metrics = self.discriminator.train_on_batch(generated_data_for_discriminator, fake)
            d_loss = 0.5 * (d_loss_real_metrics[0] + d_loss_fake_metrics[0])
            d_acc = 0.5 * (d_loss_real_metrics[1] + d_loss_fake_metrics[1]) # Assuming metric at index 1 is accuracy

            # ---------------------
            #  Train Generator
            # ---------------------
            
            feeder_outputs_for_g = self.feeder_plugin_instance.generate(n_ticks_to_generate=self.params["gan_batch_size"])
            
            # Prepare inputs for the GAN model (latent, conditional, context)
            # These names are from config and used to build the GAN model's inputs
            latent_input_name = self.params.get("generator_decoder_input_name_latent", "decoder_input_z_seq")
            conditional_input_name = self.params.get("generator_decoder_input_name_conditions", "decoder_input_conditions")
            context_input_name = self.params.get("generator_decoder_input_name_context", "decoder_input_h_context")

            gan_feed_keys = [latent_input_name, conditional_input_name, context_input_name]
            gan_inputs = []
            for key_name in gan_feed_keys:
                if key_name not in feeder_outputs_for_g:
                    self.logger.error(f"KeyError: Feeder output for GAN training missing expected key '{key_name}'. Available keys: {list(feeder_outputs_for_g.keys())}")
                    raise KeyError(f"Feeder output for GAN training missing expected key '{key_name}'.")
                gan_inputs.append(feeder_outputs_for_g[key_name])
            
            g_loss = self.gan.train_on_batch(gan_inputs, valid)

            # Store losses
            d_losses_history.append(d_loss)
            g_losses_history.append(g_loss) # g_loss from gan.train_on_batch is a single value (or list if multiple losses)
            d_accs_history.append(d_acc)

            epoch_duration = time.time() - start_time_epoch

            # Print the progress
            print(f"Epoch {epoch+1}/{self.params['gan_epochs']} [{epoch_duration:.2f}s] - D_loss: {d_loss:.4f}, D_acc: {d_acc:.4f}, G_loss: {g_loss:.4f} (LR G: {current_lr_g:.1e}, LR D: {current_lr_d:.1e})")

            # Manual ReduceLROnPlateau
            if self.params["enable_reduce_lr_on_plateau"]:
                metric_to_monitor_lr = g_loss if self.params["lr_monitor_metric"] == "g_loss" else d_loss
                if metric_to_monitor_lr < (self.best_lr_metric - self.params["lr_min_delta"]):
                    self.best_lr_metric = metric_to_monitor_lr
                    self.lr_patience_counter = 0
                else:
                    self.lr_patience_counter += 1

                if self.lr_patience_counter >= self.params["lr_patience"]:
                    # Get current LRs before changing
                    old_lr_g = float(tf.keras.backend.get_value(self.g_optimizer.learning_rate))
                    old_lr_d = float(tf.keras.backend.get_value(self.d_optimizer.learning_rate))

                    new_lr_g = max(old_lr_g * self.params["lr_reduction_factor"], self.params["min_lr_g"])
                    new_lr_d = max(old_lr_d * self.params["lr_reduction_factor"], self.params["min_lr_d"])
                    
                    changed_lr = False
                    if new_lr_g < old_lr_g:
                        tf.keras.backend.set_value(self.g_optimizer.learning_rate, new_lr_g)
                        current_lr_g = new_lr_g # Update for logging
                        logger.info(f"Reduced generator LR to {current_lr_g:.1e}")
                        changed_lr = True
                    if new_lr_d < old_lr_d:
                        tf.keras.backend.set_value(self.d_optimizer.learning_rate, new_lr_d)
                        current_lr_d = new_lr_d # Update for logging
                        logger.info(f"Reduced discriminator LR to {current_lr_d:.1e}")
                        changed_lr = True
                    
                    if changed_lr:
                        self.lr_patience_counter = 0 # Reset counter after reduction
                print(f"  ReduceLROnPlateau: Counter {self.lr_patience_counter}/{self.params['lr_patience']}, Best Metric: {self.best_lr_metric:.4f}")


            # Manual EarlyStopping
            if self.params["enable_early_stopping"]:
                metric_to_monitor_es = g_loss if self.params["es_monitor_metric"] == "g_loss" else d_loss # Adjust
                if metric_to_monitor_es < (self.best_es_metric - self.params["es_min_delta"]):
                    self.best_es_metric = metric_to_monitor_es
                    self.es_patience_counter = 0
                else:
                    self.es_patience_counter += 1

                if self.es_patience_counter >= self.params["es_patience"]:
                    logger.info(f"Early stopping triggered at epoch {epoch+1}.")
                    print(f"  EarlyStopping: Counter {self.es_patience_counter}/{self.params['es_patience']}, Best Metric: {self.best_es_metric:.4f}. Stopping training.")
                    break 
                print(f"  EarlyStopping: Counter {self.es_patience_counter}/{self.params['es_patience']}, Best Metric: {self.best_es_metric:.4f}")


            if epoch % self.params["gan_save_interval"] == 0 and epoch > 0:
                self.save_models(epoch)
                self._plot_losses(d_losses_history, g_losses_history) # Plot intermediate losses

        logger.info("GAN Training finished.")
        self.save_models(self.params['gan_epochs']) # Save final models
        self._plot_losses(d_losses_history, g_losses_history) # Plot final losses

    def _calculate_technical_indicators(self, base_features_batch_np: np.ndarray) -> np.ndarray:
        """
        Calculates technical indicators on the generated base features.
        Input: base_features_batch_np (batch_size, seq_len, num_base_features)
        Output: combined_features_batch_np (batch_size, seq_len, num_features_for_discriminator)
        """
        batch_size, seq_len, num_base_feat_input = base_features_batch_np.shape

        if num_base_feat_input != self.num_base_features:
            error_msg = (
                f"Input for TI calculation has {num_base_feat_input} features, "
                f"expected {self.num_base_features} base features based on 'base_feature_names_ordered': {self.base_feature_names}."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        if not self.ti_names_to_calculate:
            if self.num_base_features == self.num_features_for_discriminator:
                logger.info("No TIs to calculate, and base features count matches discriminator's expected features count. Skipping TI calculation.")
                return base_features_batch_np
            else:
                error_msg = (
                    f"TI Calculation: Configuration Error. No TIs are specified in 'ti_names_to_calculate', "
                    f"but the number of base features ({self.num_base_features} from 'base_feature_names_ordered') "
                    f"does not match the total number of features expected by the discriminator "
                    f"({self.num_features_for_discriminator} from 'feature_names_for_discriminator_ordered'). "
                    "This implies 'feature_names_for_discriminator_ordered' is not simply 'base_feature_names_ordered' + TIs, "
                    "or there's a mismatch in counts."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

        all_combined_features_list = []

        for i in range(batch_size):
            sample_base_features_df = pd.DataFrame(base_features_batch_np[i, :, :], columns=self.base_feature_names)
            df_with_tas = sample_base_features_df.copy()

            # --- Define OHLCV column mapping ---
            # Use direct mapping if names exist in base_feature_names, otherwise fallback or log warning.
            base_cols_set = set(self.base_feature_names)
            col_map = {}

            # Define preferred exact names
            ohlc_preferred_names = {
                'open': 'OPEN', 'high': 'HIGH', 'low': 'LOW', 'close': 'CLOSE', 'volume': 'VOLUME' # Case-sensitive
            }

            # Map preferred names if they exist in base_feature_names
            for key, preferred_name in ohlc_preferred_names.items():
                if preferred_name in base_cols_set:
                    col_map[key] = preferred_name
                else:
                    col_map[key] = None # Explicitly set to None if not found

            # Fallback for CLOSE if not found by preferred name (critical for most TIs)
            if not col_map['close']:
                if 'CLOSE' in base_cols_set: col_map['close'] = 'CLOSE'
                elif 'Close' in base_cols_set: col_map['close'] = 'Close'
                elif 'close' in base_cols_set: col_map['close'] = 'close'
                elif self.base_feature_names: col_map['close'] = self.base_feature_names[0] # Fallback to first column
                else: raise ValueError("TI Calc: 'close' column cannot be determined and base_feature_names is empty.")
                logger.info(f"TI Calc: 'close' mapped to '{col_map['close']}' (fallback or specific case).")

            # Fallbacks for other OHLC if not found by preferred name (less critical, used by some TIs)
            # These will try to find any case variation or use close as a last resort.
            for ohlc_key in ['open', 'high', 'low']:
                if not col_map[ohlc_key]: # If preferred name wasn't found
                    found_alternative = False
                    for base_col_name in self.base_feature_names:
                        if base_col_name.lower() == ohlc_key:
                            col_map[ohlc_key] = base_col_name
                            found_alternative = True
                            break
                    if not found_alternative:
                        col_map[ohlc_key] = col_map['close'] # Default to 'close' if not found
                        logger.warning(f"TI Calc: '{ohlc_key}' not found. Defaulting to '{col_map['close']}'. Some TIs might be inaccurate.")
            
            if not col_map['volume']: # If preferred 'VOLUME' wasn't found
                found_vol_alt = False
                for base_col_name in self.base_feature_names:
                    if base_col_name.lower() == 'volume':
                        col_map['volume'] = base_col_name
                        found_vol_alt = True
                        break
                if not found_vol_alt:
                     # Volume is optional for many TIs, so can be None if not found.
                     # Some TIs (like OBV) will fail if it's None and they are requested.
                    col_map['volume'] = None
                    logger.info("TI Calc: 'volume' column not found. TIs requiring volume may not be calculated or may error.")
            
            logger.debug(f"TI Calc OHLCV Mapping for sample {i}: {col_map}")


            processed_indicator_calls = set() # Stores (indicator_type_normalised, params_tuple_str)

            # --- Technical Indicator Calculation ---

            # RSI (e.g., RSI_14)
            rsi_configs = set()
            for ti_name in self.ti_names_to_calculate:
                if ti_name.upper().startswith("RSI_"):
                    parts = ti_name.split('_')
                    if len(parts) == 2:
                        try: rsi_configs.add(int(parts[1]))
                        except ValueError: logger.warning(f"Could not parse RSI param from {ti_name}")
            for length in rsi_configs:
                call_key = ('rsi', str((length,)))
                if call_key not in processed_indicator_calls:
                    try:
                        df_with_tas.ta.rsi(close=df_with_tas[col_map['close']], length=length, append=True)
                        processed_indicator_calls.add(call_key)
                        logger.debug(f"Calculated RSI_{length}")
                    except Exception as e: logger.warning(f"Error calculating RSI_{length}: {e}")

            # EMA (e.g., EMA_14)
            ema_configs = set()
            for ti_name in self.ti_names_to_calculate:
                if ti_name.upper().startswith("EMA_"):
                    parts = ti_name.split('_')
                    if len(parts) == 2:
                        try: ema_configs.add(int(parts[1]))
                        except ValueError: logger.warning(f"Could not parse EMA param from {ti_name}")
            for length in ema_configs:
                call_key = ('ema', str((length,)))
                if call_key not in processed_indicator_calls:
                    try:
                        df_with_tas.ta.ema(close=df_with_tas[col_map['close']], length=length, append=True)
                        processed_indicator_calls.add(call_key)
                        logger.debug(f"Calculated EMA_{length}")
                    except Exception as e: logger.warning(f"Error calculating EMA_{length}: {e}")
                
            # MACD (e.g., MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9)
            macd_configs = set() # Stores tuples of (fast, slow, signal)
            for ti_name in self.ti_names_to_calculate:
                if ti_name.upper().startswith("MACD"): # Catches MACD_, MACDh_, MACDs_
                    parts = ti_name.split('_')
                    # Expect MACD_F_S_SIG, MACDh_F_S_SIG, MACDs_F_S_SIG
                    if len(parts) == 4:
                        try: macd_configs.add((int(parts[1]), int(parts[2]), int(parts[3])))
                        except ValueError: logger.warning(f"Could not parse MACD params from ti_name {ti_name}")
            for f,s,sig in macd_configs:
                call_key = ('macd', str((f,s,sig)))
                if call_key not in processed_indicator_calls:
                    try:
                        df_with_tas.ta.macd(close=df_with_tas[col_map['close']], fast=f, slow=s, signal=sig, append=True)
                        processed_indicator_calls.add(call_key) # Marks this combination as processed
                        logger.debug(f"Calculated MACD family for {f}_{s}_{sig}")
                    except Exception as e: logger.warning(f"Error calculating MACD({f},{s},{sig}): {e}")

            # Stochastic Oscillator (e.g., STOCHk_14_3_3, STOCHd_14_3_3)
            stoch_configs = set() # Stores (k, d, smooth_k)
            for ti_name in self.ti_names_to_calculate:
                if ti_name.upper().startswith("STOCH"):
                    parts = ti_name.split('_') # STOCHk_K_D_SmoothK or STOCHd_K_D_SmoothK
                    if len(parts) == 4:
                        try: stoch_configs.add((int(parts[1]), int(parts[2]), int(parts[3])))
                        except ValueError: logger.warning(f"Could not parse STOCH params from {ti_name}")
            for k,d,smooth_k in stoch_configs: # k,d,smooth_k from pandas-ta are k,d,smooth_k
                call_key = ('stoch', str((k,d,smooth_k)))
                if call_key not in processed_indicator_calls:
                    if col_map['high'] and col_map['low'] and col_map['close']:
                        try:
                            df_with_tas.ta.stoch(high=df_with_tas[col_map['high']], low=df_with_tas[col_map['low']], close=df_with_tas[col_map['close']], k=k, d=d, smooth_k=smooth_k, append=True)
                            processed_indicator_calls.add(call_key)
                            logger.debug(f"Calculated STOCH family for {k}_{d}_{smooth_k}")
                        except Exception as e: logger.warning(f"Error calculating STOCH({k},{d},{smooth_k}): {e}")
                    else: logger.warning(f"Skipping STOCH({k},{d},{smooth_k}) due to missing HLC columns.")

            # ADX (e.g., ADX_14, DMP_14, DMN_14)
            adx_configs = set() # Stores (length)
            for ti_name in self.ti_names_to_calculate:
                if ti_name.upper().startswith("ADX_") or ti_name.upper().startswith("DMP_") or ti_name.upper().startswith("DMN_"):
                    parts = ti_name.split('_')
                    if len(parts) == 2: # ADX_L, DMP_L, DMN_L
                        try: adx_configs.add(int(parts[1]))
                        except ValueError: logger.warning(f"Could not parse ADX/DMP/DMN param from {ti_name}")
            for length in adx_configs:
                call_key = ('adx', str((length,)))
                if call_key not in processed_indicator_calls:
                    if col_map['high'] and col_map['low'] and col_map['close']:
                        try:
                            df_with_tas.ta.adx(high=df_with_tas[col_map['high']], low=df_with_tas[col_map['low']], close=df_with_tas[col_map['close']], length=length, append=True)
                            processed_indicator_calls.add(call_key)
                            logger.debug(f"Calculated ADX family for {length}")
                        except Exception as e: logger.warning(f"Error calculating ADX({length}): {e}")
                    else: logger.warning(f"Skipping ADX({length}) due to missing HLC columns.")
            
            # ATR (e.g., ATRr_14) - pandas-ta typically generates ATRr_length or ATR_length
            atr_configs = set() # Stores (length)
            for ti_name in self.ti_names_to_calculate:
                # pandas-ta might produce ATRr_L or ATR_L. Config uses ATRr_L.
                if ti_name.upper().startswith("ATRR_") or ti_name.upper().startswith("ATR_"):
                    parts = ti_name.split('_')
                    if len(parts) == 2:
                        try: atr_configs.add(int(parts[1]))
                        except ValueError: logger.warning(f"Could not parse ATR param from {ti_name}")
            for length in atr_configs:
                call_key = ('atr', str((length,)))
                if call_key not in processed_indicator_calls:
                    if col_map['high'] and col_map['low'] and col_map['close']:
                        try:
                            # mamode="rma" is often default for "true range" ATR.
                            # pandas-ta default is mamode="sma". ATRr uses rma.
                            df_with_tas.ta.atr(high=df_with_tas[col_map['high']], low=df_with_tas[col_map['low']], close=df_with_tas[col_map['close']], length=length, mamode="rma", append=True)
                            processed_indicator_calls.add(call_key)
                            logger.debug(f"Calculated ATRr_{length}")
                        except Exception as e: logger.warning(f"Error calculating ATRr_{length}: {e}")
                    else: logger.warning(f"Skipping ATRr_{length} due to missing HLC columns.")

            # CCI (e.g., CCI_14_0.015)
            cci_configs = set() # Stores (length, constant)
            for ti_name in self.ti_names_to_calculate:
                if ti_name.upper().startswith("CCI_"):
                    parts = ti_name.split('_') # CCI_L_C
                    if len(parts) == 3:
                        try: cci_configs.add((int(parts[1]), float(parts[2])))
                        except ValueError: logger.warning(f"Could not parse CCI params from {ti_name}")
            for length, constant in cci_configs:
                call_key = ('cci', str((length, constant)))
                if call_key not in processed_indicator_calls:
                    if col_map['high'] and col_map['low'] and col_map['close']:
                        try:
                            df_with_tas.ta.cci(high=df_with_tas[col_map['high']], low=df_with_tas[col_map['low']], close=df_with_tas[col_map['close']], length=length, c=constant, append=True)
                            processed_indicator_calls.add(call_key)
                            logger.debug(f"Calculated CCI_{length}_{constant}")
                        except Exception as e: logger.warning(f"Error calculating CCI({length},{constant}): {e}")
                    else: logger.warning(f"Skipping CCI({length},{constant}) due to missing HLC columns.")

            # Williams %R (e.g., WILLR_14)
            willr_configs = set() # Stores (length)
            for ti_name in self.ti_names_to_calculate:
                if ti_name.upper().startswith("WILLR_"):
                    parts = ti_name.split('_')
                    if len(parts) == 2:
                        try: 
                            willr_configs.add(int(parts[1]))
                        except ValueError: 
                            logger.warning(f"Could not parse WILLR param from {ti_name}") # Corrected variable
            for length in willr_configs:
                call_key = ('willr', str((length,)))
                if call_key not in processed_indicator_calls:
                    if col_map['high'] and col_map['low'] and col_map['close']:
                        try:
                            df_with_tas.ta.willr(high=df_with_tas[col_map['high']], low=df_with_tas[col_map['low']], close=df_with_tas[col_map['close']], length=length, append=True)
                            processed_indicator_calls.add(call_key)
                            logger.debug(f"Calculated WILLR_{length}")
                        except Exception as e: logger.warning(f"Error calculating WILLR_{length}: {e}")
                    else: logger.warning(f"Skipping WILLR_{length} due to missing HLC columns.")

            # Momentum (e.g., MOM_14)
            mom_configs = set() # Stores (length)
            for ti_name in self.ti_names_to_calculate:
                if ti_name.upper().startswith("MOM_"):
                    parts = ti_name.split('_')
                    if len(parts) == 2:
                        try: 
                            mom_configs.add(int(parts[1]))
                        except ValueError: 
                            logger.warning(f"Could not parse MOM param from {ti_name}") # Corrected variable
            for length in mom_configs:
                call_key = ('mom', str((length,)))
                if call_key not in processed_indicator_calls:
                    try:
                        df_with_tas.ta.mom(close=df_with_tas[col_map['close']], length=length, append=True)
                        processed_indicator_calls.add(call_key)
                        logger.debug(f"Calculated MOM_{length}")
                    except Exception as e: logger.warning(f"Error calculating MOM_{length}: {e}")

           

            # ROC (Rate of Change) (e.g., ROC_14)
            roc_configs = set() # Stores (length)
            for ti_name in self.ti_names_to_calculate:
                if ti_name.upper().startswith("ROC_"): # ROC, not ROC_
                    parts = ti_name.split('_')
                    if len(parts) == 2: # ROC_L
                        try: 
                            roc_configs.add(int(parts[1]))
                        except ValueError: 
                            logger.warning(f"Could not parse ROC param from {ti_name}") # Corrected variable
            for length in roc_configs:
                call_key = ('roc', str((length,)))
                if call_key not in processed_indicator_calls:
                    try:
                        df_with_tas.ta.roc(close=df_with_tas[col_map['close']], length=length, append=True)
                        processed_indicator_calls.add(call_key)
                        logger.debug(f"Calculated ROC_{length}")
                    except Exception as e: logger.warning(f"Error calculating ROC_{length}: {e}")

            # Bollinger Bands (e.g., BBL_20_2.0, BBM_20_2.0, BBU_20_2.0, BBB_20_2.0, BBP_20_2.0)
            bbands_configs = set() # Stores (length, std_dev)
            for ti_name in self.ti_names_to_calculate:
                if ti_name.upper().startswith("BBL_") or ti_name.upper().startswith("BBM_") or \
                   ti_name.upper().startswith("BBU_") or ti_name.upper().startswith("BBB_") or \
                   ti_name.upper().startswith("BBP_"):
                    parts = ti_name.split('_') # e.g., BBL_L_STD
                    if len(parts) == 3:
                        try: bbands_configs.add((int(parts[1]), float(parts[2])))
                        except ValueError: logger.warning(f"Could not parse BBANDS params from {ti_name}")
            for length, std in bbands_configs:
                call_key = ('bbands', str((length, std)))
                if call_key not in processed_indicator_calls:
                    try:
                        df_with_tas.ta.bbands(close=df_with_tas[col_map['close']], length=length, std=std, append=True)
                        processed_indicator_calls.add(call_key)
                        logger.debug(f"Calculated BBANDS family for {length}_{std}")
                    except Exception as e: logger.warning(f"Error calculating BBANDS({length},{std}): {e}")
            
            # --- End Technical Indicator Calculation ---

            # Ensure all columns required by discriminator are present, fill NaNs, and order correctly.
            # Verify that pandas_ta generated the columns with the exact names expected in self.discriminator_feature_names
            missing_cols = [col for col in self.discriminator_feature_names if col not in df_with_tas.columns]
            if missing_cols:
                logger.warning(f"Sample {i}: After TA calculation, the following expected columns are MISSING: {missing_cols}. They will be added with NaNs (then 0). This might indicate a mismatch between 'feature_names_for_discriminator_ordered' and pandas_ta output names.")
            
            # Reindex to ensure correct order and presence of all expected columns (base + TIs)
            # Columns not generated by pandas_ta (or base features if they were somehow dropped) will be added as NaN.
            df_final_sample = df_with_tas.reindex(columns=self.discriminator_feature_names) # Corrected: self.discriminator_feature_names
            
            # Fill NaNs - common at the start of series due to TA lookback periods.
            # Using 0 for now. Consider ffill() then bfill() then 0, or other strategies.
            df_final_sample = df_final_sample.fillna(0) 

            all_combined_features_list.append(df_final_sample.to_numpy())

        combined_batch_np = np.stack(all_combined_features_list, axis=0)
        
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

    def _plot_losses(self, d_losses: List[float], g_losses: List[float]):
        """Plots generator and discriminator loss."""
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 5))
            plt.plot(d_losses, label='Discriminator Loss')
            plt.plot(g_losses, label='Generator Loss')
            plt.title('GAN Training Losses')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            loss_plot_file = self.params.get("loss_plot_file", os.path.join(self.gan_model_dir, "gan_loss_plot.png"))
            plt.savefig(loss_plot_file)
            plt.close()
            logger.info(f"GAN loss plot saved to {loss_plot_file}")
        except ImportError:
            logger.warning("Matplotlib not installed. Skipping loss plotting.")
        except Exception as e:
            logger.error(f"Error plotting losses: {e}")


    def save_models(self, epoch: None) -> None:
        """Saves the generator and discriminator models."""
        if self.generator:
            # Change .h5 to .keras
            g_path = os.path.join(self.gan_model_dir, f"generator_epoch_{epoch}.keras")
            self.generator.save(g_path)
            logger.info(f"Saved generator model to {g_path}")
        if self.discriminator:
            # Change .h5 to .keras
            d_path = os.path.join(self.gan_model_dir, f"discriminator_epoch_{epoch}.keras")
            self.discriminator.save(d_path)
            logger.info(f"Saved discriminator model to {d_path}")
        # Also, if you save the combined GAN model, update its extension too.
        # Example if you were to save self.gan:
        # if self.gan:
        #     gan_path = os.path.join(self.gan_model_dir, f"gan_epoch_{epoch}.keras")
        #     self.gan.save(gan_path)
        #     logger.info(f"Saved GAN model to {gan_path}")

    def get_generator(self) -> Optional[Model]: # Corrected Keras Model type
        """Returns the trained generator model."""
        return self.generator

    def get_discriminator(self) -> Optional[Model]: # Corrected Keras Model type
        """Returns the trained discriminator model."""
        return self.discriminator
