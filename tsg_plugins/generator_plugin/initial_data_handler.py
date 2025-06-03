#!/usr/bin/env python3
"""
Initial Data Handler Module

Handles loading of initial close anchor from data files and manages
the initial window data setup for the generation process.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List


class InitialDataHandler:
    """Manages initial data loading and close anchor initialization."""
    
    def __init__(self, normalization_handler):
        """
        Initialize the data handler.
        
        Args:
            normalization_handler: Instance of NormalizationHandler for denormalization
        """
        self.normalization_handler = normalization_handler
        self.initial_denormalized_close_anchor = None
        self._last_initial_close_file_path = None
    
    def load_initial_close_anchor(self, file_path: Optional[str]) -> None:
        """
        Load the last CLOSE price from the specified data file.
        
        Args:
            file_path: Path to the data file (e.g., x_train_file)
        """
        if not file_path:
            print("InitialDataHandler: Warning - No data file path provided for initial CLOSE. "
                  "Initial CLOSE anchor will default to 1.0.")
            self.initial_denormalized_close_anchor = 1.0
            return

        try:
            df_real = pd.read_csv(file_path)
            if 'CLOSE' in df_real.columns and not df_real['CLOSE'].empty:
                last_close_val_from_file = float(df_real['CLOSE'].iloc[-1])

                # Check if value needs denormalization
                if (self.normalization_handler.normalization_params and 
                    "CLOSE" in self.normalization_handler.normalization_params):
                    # Value in file is normalized, denormalize it
                    self.initial_denormalized_close_anchor = self.normalization_handler.denormalize_value(
                        last_close_val_from_file, "CLOSE"
                    )
                    print(f"InitialDataHandler: Initial CLOSE anchor (denormalized from file) "
                          f"loaded from '{file_path}': {self.initial_denormalized_close_anchor}")
                else:
                    # Value in file is already denormalized
                    self.initial_denormalized_close_anchor = last_close_val_from_file
                    print(f"InitialDataHandler: Initial CLOSE anchor (assumed denormalized) "
                          f"loaded from '{file_path}': {self.initial_denormalized_close_anchor}")
            else:
                print(f"InitialDataHandler: Warning - 'CLOSE' column not found or empty in "
                      f"'{file_path}'. Initial CLOSE anchor defaulting to 1.0.")
                self.initial_denormalized_close_anchor = 1.0
                
        except FileNotFoundError:
            print(f"InitialDataHandler: ERROR - Data file for initial CLOSE not found: "
                  f"{file_path}. Defaulting to 1.0.")
            self.initial_denormalized_close_anchor = 1.0
        except Exception as e:
            print(f"InitialDataHandler: ERROR - Could not load initial CLOSE from "
                  f"'{file_path}': {e}. Defaulting to 1.0.")
            self.initial_denormalized_close_anchor = 1.0
        
        # Final safety check
        if (self.initial_denormalized_close_anchor is None or 
            pd.isna(self.initial_denormalized_close_anchor)):
            self.initial_denormalized_close_anchor = 1.0
            print("InitialDataHandler: Critical - initial_denormalized_close_anchor was "
                  "None/NaN after attempting load. Defaulted to 1.0.")
    
    def should_reload_close_anchor(self, current_file_path: Optional[str]) -> bool:
        """
        Check if the close anchor should be reloaded due to path changes.
        
        Args:
            current_file_path: Current file path from config
            
        Returns:
            True if anchor should be reloaded
        """
        if current_file_path:
            if (self.initial_denormalized_close_anchor is None or 
                self._last_initial_close_file_path != current_file_path):
                return True
        return False
    
    def update_last_loaded_path(self, file_path: Optional[str]) -> None:
        """
        Update the tracking of the last loaded file path.
        
        Args:
            file_path: File path that was just loaded
        """
        self._last_initial_close_file_path = file_path
    
    def get_initial_close_anchor(self) -> Optional[float]:
        """
        Get the initial denormalized close anchor value.
        
        Returns:
            Initial close anchor value or None if not set
        """
        return self.initial_denormalized_close_anchor
    
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
