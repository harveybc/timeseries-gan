#!/usr/bin/env python3
"""
Initial Data Handler Module

Handles loading of initial close anchor from data files and manages
the initial window data setup for the generation process.
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional, Dict, Any, List


class InitialDataHandler:
    DEFAULT_ANCHOR_VALUE = 1.0
    MIN_ROWS_FOR_ANCHOR = 1 # Minimum rows to read to find the anchor

    """Manages initial data loading and close anchor initialization."""
    
    def __init__(self, params: Dict[str, Any], logger: logging.Logger, normalization_handler: Optional[Any] = None):
        """
        Initialize the data handler.
        
        Args:
            normalization_handler: Instance of NormalizationHandler for denormalization
        """
        self.params = params
        self.logger = logger
        # The normalization_handler is passed, but we will avoid calling denormalize_value
        # based on the policy of not using explicit denormalization here.
        self.normalization_handler = normalization_handler 
        self.initial_close_anchor: float = self.DEFAULT_ANCHOR_VALUE
        self.initial_window_df: Optional[pd.DataFrame] = None
        
        self.target_column = self.params.get("target_column", "CLOSE")
        self.datetime_col_name = self.params.get("datetime_col_name", "DATE_TIME")
        
        # Attempt to load anchor immediately if file path is available in initial params
        # This might be called again if set_params updates the file path
        initial_file_path = self.params.get("x_train_file", self.params.get("real_data_file"))
        if initial_file_path:
            self.load_initial_close_anchor(initial_file_path)

    def _set_default_anchor(self):
        """Sets the initial close anchor to the default value and logs a warning."""
        self.initial_close_anchor = self.DEFAULT_ANCHOR_VALUE
        self.logger.warning(f"Defaulting initial_close_anchor to {self.initial_close_anchor}.")

    def load_initial_close_anchor(self, file_path: str) -> None:
        """
        Loads the initial CLOSE (or target_column) value from the specified file.
        Uses the value directly as per the "no explicit denormalization" policy.
        """
        self.logger.info(f"Attempting to load initial CLOSE anchor from: {file_path}")
        try:
            # Read only enough rows to get the first value of the target column
            # Ensure pandas is imported if not already at the top of the file
            df = pd.read_csv(file_path, nrows=self.MIN_ROWS_FOR_ANCHOR) 

            if df.empty:
                self.logger.warning(f"Initial data file '{file_path}' is empty. Defaulting anchor.")
                self._set_default_anchor()
                return

            if self.target_column not in df.columns:
                self.logger.error(f"Target column '{self.target_column}' not found in '{file_path}'. Defaulting anchor.")
                self._set_default_anchor()
                return

            initial_value_from_file = df[self.target_column].iloc[0]

            if pd.isna(initial_value_from_file):
                self.logger.warning(f"Initial value for '{self.target_column}' in '{file_path}' is NaN. Defaulting anchor.")
                self._set_default_anchor()
                return
            
            # Policy: "we are not using any normalization (except maybe batch normalization layers in models)"
            # Therefore, use the value directly from the CSV file.
            # The file 'normalized_d4.csv' implies the value is already in a processed scale.
            self.initial_close_anchor = float(initial_value_from_file)
            self.logger.info(f"Successfully loaded initial CLOSE anchor: {self.initial_close_anchor:.6f} "
                             f"directly from '{file_path}' (column: '{self.target_column}', row: 0). "
                             "This value is used as-is, without explicit denormalization.")

        except FileNotFoundError:
            self.logger.error(f"Initial data file not found: '{file_path}'. Defaulting anchor.")
            self._set_default_anchor()
        except pd.errors.EmptyDataError:
            self.logger.error(f"Initial data file '{file_path}' is empty or unreadable. Defaulting anchor.")
            self._set_default_anchor()
        except Exception as e:
            # Catching the broad exception to prevent crashes and log the specific error.
            # This replaces the original error message that mentioned the AttributeError.
            self.logger.error(f"Error loading initial CLOSE anchor from '{file_path}': {type(e).__name__} - {e}. Defaulting anchor.")
            self.logger.debug("Traceback for error in load_initial_close_anchor:", exc_info=True)
            self._set_default_anchor()

    def get_initial_close_anchor(self) -> float:
        return self.initial_close_anchor

    def load_initial_window(self, file_path: str, window_size: int) -> Optional[pd.DataFrame]:
        # This method might also need review if it uses normalization_handler
        self.logger.info(f"Loading initial window of size {window_size} from {file_path}")
        try:
            df = pd.read_csv(file_path, nrows=window_size)
            if len(df) < window_size:
                self.logger.warning(f"File {file_path} has less than {window_size} rows. Loaded {len(df)} rows.")
                # Decide if this is acceptable or should return None or raise error
            
            # Assuming all necessary columns are present. Add validation if needed.
            # If this data needs to be "denormalized" for some reason, that logic would be here.
            # But based on "no normalization" policy, we'd use it as is.
            self.initial_window_df = df
            self.logger.info(f"Initial window loaded with shape: {self.initial_window_df.shape}")
            return self.initial_window_df
        except Exception as e:
            self.logger.error(f"Failed to load initial window from {file_path}: {e}")
            self.initial_window_df = None
            return None

    def get_initial_window(self) -> Optional[pd.DataFrame]:
        return self.initial_window_df

    def set_params(self, **kwargs) -> None:
        """
        Update parameters and reload initial anchor if relevant file path changes.
        """
        # Example: if x_train_file changes, reload anchor
        old_file_path = self.params.get("x_train_file")
        
        # Update internal params with relevant keys from kwargs or main_config
        for key in ["x_train_file", "real_data_file", "target_column", "datetime_col_name"]:
            if key in kwargs: # Prioritize direct kwargs
                self.params[key] = kwargs[key]
            elif f"feeder_{key}" in kwargs: # Check for prefixed keys if applicable
                 self.params[key] = kwargs[f"feeder_{key}"]


        self.target_column = self.params.get("target_column", "CLOSE") # Update target column
        
        new_file_path = self.params.get("x_train_file", self.params.get("real_data_file"))

        if new_file_path and new_file_path != old_file_path:
            self.logger.info(f"Initial data file path changed from '{old_file_path}' to '{new_file_path}'. Reloading anchor.")
            self.load_initial_close_anchor(new_file_path)
        elif not new_file_path:
             self.logger.warning("No initial data file path provided in set_params. Anchor might be default.")

    def setup_initial_window_data(self, initial_full_feature_window: Optional[np.ndarray],
                                 decoder_input_window_size: int, 
                                 num_all_features: int) -> np.ndarray:
        """
        Set up the initial window data for generation.
        
        Args:
            initial_full_feature_window: Pre-provided initial window or None
            decoder_input_window_size: Size of the decoder input window
            num_all_features: Total number of features
            
        Returns:
            Initialized feature window array
        """
        current_input_feature_window = np.zeros(
            (decoder_input_window_size, num_all_features), dtype=np.float32
        )
        
        if initial_full_feature_window is not None:
            expected_shape = current_input_feature_window.shape
            if initial_full_feature_window.shape == expected_shape:
                current_input_feature_window = initial_full_feature_window.astype(np.float32).copy()
                print("InitialDataHandler: Using provided initial_full_feature_window")
            else:
                raise ValueError(
                    f"Shape mismatch for initial_full_feature_window. "
                    f"Expected {expected_shape}, got {initial_full_feature_window.shape}"
                )
        else:
            print("InitialDataHandler: Warning - No initial_full_feature_window provided. "
                  "Using zeros. TIs will be NaN initially.")
        
        return current_input_feature_window
    
    def extract_ohlc_history_from_window(self, window: np.ndarray, 
                                       ohlc_feature_names: List[str],
                                       feature_to_idx: Dict[str, int],
                                       min_ohlc_hist_len: int) -> List[Dict[str, float]]:
        """
        Extract OHLC history from the initial window for TI calculation.
        
        Args:
            window: Initial feature window array
            ohlc_feature_names: List of OHLC feature names
            feature_to_idx: Mapping from feature names to indices
            min_ohlc_hist_len: Minimum required history length
            
        Returns:
            List of OHLC dictionaries (denormalized)
        """
        ohlc_history_list = []
        decoder_input_window_size = window.shape[0]
        
        for i in range(decoder_input_window_size):
            row_ohlc_norm_values = {
                name: window[i, feature_to_idx[name]]
                for name in ohlc_feature_names 
                if name in feature_to_idx and pd.notnull(window[i, feature_to_idx[name]])
            }
            
            if len(row_ohlc_norm_values) == len(ohlc_feature_names):
                ohlc_dict_denorm = {
                    name: self.normalization_handler.denormalize_value(
                        row_ohlc_norm_values.get(name, np.nan), name
                    )
                    for name in ohlc_feature_names
                }
                
                if all(pd.notnull(v) for v in ohlc_dict_denorm.values()):
                    ohlc_history_list.append(ohlc_dict_denorm)
        
        # Trim if it became too long
        if len(ohlc_history_list) > min_ohlc_hist_len + 50:
            ohlc_history_list = ohlc_history_list[-(min_ohlc_hist_len + 50):]
        
        return ohlc_history_list

# Example of how NormalizationHandler might be structured if it were a no-op or incomplete
# class NormalizationHandler:
#     def __init__(self, params, logger):
#         self.params = params
#         self.logger = logger
#         self.norm_params = {}
#         self.logger.info("NormalizationHandler initialized (currently minimal/no-op).")

#     def load_normalization_params(self, file_path):
#         self.logger.info(f"Attempting to load normalization params from {file_path} (handler is minimal).")
#         # In a no-op version, this might just log or store the path without processing
#         try:
#             # Placeholder: actual loading would parse JSON and store means/stds
#             # For now, just acknowledge the call.
#             if file_path and isinstance(file_path, str): # Basic check
#                  self.logger.info(f"Normalization parameters file '{file_path}' would be processed here.")
#             else:
#                  self.logger.warning(f"Invalid or no normalization parameters file path provided: {file_path}")
#             # self.norm_params = json.load(open(file_path)) 
#         except Exception as e:
#             self.logger.error(f"Failed to load/process normalization params from {file_path}: {e}")
#             self.norm_params = {}


#     def normalize_value(self, value, feature_name):
#         self.logger.debug(f"Normalize_value called for {feature_name} (handler is minimal, returning original value).")
#         return value # Pass-through

#     # denormalize_value is MISSING, causing the AttributeError
#     # def denormalize_value(self, value, feature_name):
#     #     self.logger.debug(f"Denormalize_value called for {feature_name} (handler is minimal, returning original value).")
#     #     return value # Pass-through

#     def reset(self):
#         self.logger.info("NormalizationHandler reset (handler is minimal).")
#         self.norm_params = {}
