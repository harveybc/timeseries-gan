#!/usr/bin/env python3
"""
Data Generator Module

Handles the core generation logic including window pre-filling with derived features,
date feature generation, and data assembly for the main generation process.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional


class DataGenerator:
    """Handles data generation and feature assembly for synthetic sequence creation."""
    
    def __init__(self, params: Dict[str, Any], feature_to_idx: Dict[str, int],
                 normalization_handler, ti_calculator):
        """
        Initialize the data generator.
        
        Args:
            params: Plugin parameters
            feature_to_idx: Mapping from feature names to indices  
            normalization_handler: Instance for normalization/denormalization
            ti_calculator: Technical indicator calculator
        """
        self.params = params
        self.feature_to_idx = feature_to_idx
        self.normalization_handler = normalization_handler
        self.ti_calculator = ti_calculator
    
    def pre_fill_derived_features_in_window(self, current_input_feature_window: np.ndarray,
                                          initial_datetimes_for_window: pd.Series,
                                          ohlc_history_for_ti_list: List[Dict[str, float]],
                                          true_prev_close_for_log_return: Optional[float] = None) -> None:
        """
        Pre-fill derived features in the initial window including date features,
        technical indicators, and log returns.
        
        Args:
            current_input_feature_window: The initial feature window to pre-fill
            initial_datetimes_for_window: DateTime series for the window
            ohlc_history_for_ti_list: OHLC history for TI calculations
            true_prev_close_for_log_return: Previous close for log return calculation
        """
        if len(initial_datetimes_for_window) != current_input_feature_window.shape[0]:
            print("DataGenerator: Warning - datetime series length mismatch. Skipping pre-fill.")
            return
        
        print("DataGenerator: Pre-filling derived features in initial window...")
        
        decoder_input_window_size = current_input_feature_window.shape[0]
        local_prev_norm_close = self._initialize_prev_close_for_prefill(
            current_input_feature_window, true_prev_close_for_log_return
        )
        
        for i in range(decoder_input_window_size):
            dt_obj = initial_datetimes_for_window.iloc[i]
            
            # Fill raw date features
            self._fill_raw_date_features_prefill(current_input_feature_window, i, dt_obj)
            
            # Fill sin/cos date features
            self._fill_sincos_date_features_prefill(current_input_feature_window, i, dt_obj)
            
            # Fill technical indicators
            self._fill_technical_indicators_prefill(
                current_input_feature_window, i, ohlc_history_for_ti_list
            )
            
            # Fill log return
            local_prev_norm_close = self._fill_log_return_prefill(
                current_input_feature_window, i, local_prev_norm_close
            )
        
        print("DataGenerator: Finished pre-filling derived features in initial window.")
    
    def _initialize_prev_close_for_prefill(self, window: np.ndarray,
                                         true_prev_close: Optional[float]) -> Optional[float]:
        """Initialize the previous close value for log return calculation."""
        if true_prev_close is not None and pd.notnull(true_prev_close):
            print(f"DataGenerator: Using true_prev_close_for_log_return: {true_prev_close}")
            return true_prev_close
        elif ('CLOSE' in self.feature_to_idx and 
              pd.notnull(window[0, self.feature_to_idx['CLOSE']])):
            prev_close = window[0, self.feature_to_idx['CLOSE']]
            print(f"DataGenerator: Using fallback prev_close (first CLOSE in window): {prev_close}")
            return prev_close
        else:
            print("DataGenerator: Could not initialize prev_close for prefill")
            return None
    
    def _fill_raw_date_features_prefill(self, window: np.ndarray, i: int, dt_obj: pd.Timestamp) -> None:
        """Fill raw date features for a single window position."""
        raw_date_map = {
            "day_of_month": dt_obj.day,
            "hour_of_day": dt_obj.hour,
            "day_of_week": dt_obj.dayofweek
        }
        
        for raw_feat_name, raw_val in raw_date_map.items():
            if raw_feat_name in self.feature_to_idx:
                normalized_val = self.normalization_handler.normalize_value(raw_val, raw_feat_name)
                window[i, self.feature_to_idx[raw_feat_name]] = normalized_val
    
    def _fill_sincos_date_features_prefill(self, window: np.ndarray, i: int, dt_obj: pd.Timestamp) -> None:
        """Fill sin/cos transformed date features for a single window position."""
        scaled_date_features_arr = self.get_scaled_date_features(dt_obj)
        sincos_idx_counter = 0
        
        for original_date_feat_name in self.params.get("date_conditional_feature_names", []):
            for suffix in ["_sin", "_cos"]:
                feat_name_transformed = f"{original_date_feat_name}{suffix}"
                if feat_name_transformed in self.feature_to_idx:
                    if sincos_idx_counter < len(scaled_date_features_arr):
                        window[i, self.feature_to_idx[feat_name_transformed]] = scaled_date_features_arr[sincos_idx_counter]
                    else:
                        window[i, self.feature_to_idx[feat_name_transformed]] = 0.0
                    sincos_idx_counter += 1
    
    def _fill_technical_indicators_prefill(self, window: np.ndarray, i: int,
                                         ohlc_history_list: List[Dict[str, float]]) -> None:
        """Fill technical indicators for a single window position."""
        if len(ohlc_history_list) > i:
            history_slice_df = pd.DataFrame(ohlc_history_list[:i+1])
            if not history_slice_df.empty:
                calculated_ti_df = self.ti_calculator.calculate_technical_indicators(
                    history_slice_df, self.params["ohlc_feature_names"]
                )
                
                if not calculated_ti_df.empty:
                    for ti_name in self.params["ti_feature_names"]:
                        if ti_name in self.feature_to_idx and ti_name in calculated_ti_df.columns:
                            val_ti_denorm = calculated_ti_df.iloc[0][ti_name]
                            if pd.notnull(val_ti_denorm):
                                normalized_val = self.normalization_handler.normalize_value(val_ti_denorm, ti_name)
                                window[i, self.feature_to_idx[ti_name]] = normalized_val
                            else:
                                window[i, self.feature_to_idx[ti_name]] = np.nan
    
    def _fill_log_return_prefill(self, window: np.ndarray, i: int,
                               prev_norm_close: Optional[float]) -> Optional[float]:
        """Fill log return for a single window position and return updated previous close."""
        # Get current close
        norm_close_current = np.nan
        if ('CLOSE' in self.feature_to_idx and 
            pd.notnull(window[i, self.feature_to_idx['CLOSE']])):
            norm_close_current = window[i, self.feature_to_idx['CLOSE']]
        elif ('OPEN' in self.feature_to_idx and 
              pd.notnull(window[i, self.feature_to_idx['OPEN']])):
            norm_close_current = window[i, self.feature_to_idx['OPEN']]  # Fallback
        
        # Calculate log return
        if "log_return" in self.feature_to_idx:
            log_return_val = 0.0
            if (prev_norm_close is not None and pd.notnull(norm_close_current) and
                prev_norm_close > 1e-9 and norm_close_current > 1e-9):
                log_return_val = np.log(norm_close_current / prev_norm_close)
            
            normalized_log_return = self.normalization_handler.normalize_value(log_return_val, "log_return")
            window[i, self.feature_to_idx["log_return"]] = normalized_log_return
        
        # Update previous close for next iteration
        if pd.notnull(norm_close_current):
            return norm_close_current
        return prev_norm_close
    
    def get_scaled_date_features(self, datetime_obj: pd.Timestamp) -> np.ndarray:
        """
        Generate scaled (sin/cos) date features for a given datetime.
        Uses main_config for max values (compatible with FeederPlugin approach).
        
        Args:
            datetime_obj: Timestamp object to extract features from
            
        Returns:
            Array of sin/cos transformed date features
        """
        date_features = []
        
        # Get main config (should be available from plugin initialization)
        main_cfg = getattr(self, 'main_config', {})
        date_conditional_names = self.params.get("date_conditional_feature_names", [])

        if "day_of_month" in date_conditional_names:
            dom = datetime_obj.day
            max_dom = main_cfg.get("feeder_max_day_of_month", 31)
            date_features.extend([
                np.sin(2 * np.pi * dom / max_dom), 
                np.cos(2 * np.pi * dom / max_dom)
            ])
            
        if "hour_of_day" in date_conditional_names:
            hod = datetime_obj.hour
            max_hod = main_cfg.get("feeder_max_hour_of_day", 23)
            date_features.extend([
                np.sin(2 * np.pi * hod / (max_hod + 1)), 
                np.cos(2 * np.pi * hod / (max_hod + 1))
            ])
            
        if "day_of_week" in date_conditional_names:
            dow = datetime_obj.dayofweek
            max_dow = main_cfg.get("feeder_max_day_of_week", 6)
            date_features.extend([
                np.sin(2 * np.pi * dow / (max_dow + 1)), 
                np.cos(2 * np.pi * dow / (max_dow + 1))
            ])
            
        if "day_of_year" in date_conditional_names:
            doy = datetime_obj.dayofyear
            max_doy = main_cfg.get("feeder_max_day_of_year", 366)
            date_features.extend([
                np.sin(2 * np.pi * doy / max_doy), 
                np.cos(2 * np.pi * doy / max_doy)
            ])
            
        return np.array(date_features, dtype=np.float32)
    
    def set_main_config(self, main_config: Dict[str, Any]) -> None:
        """
        Set the main configuration for date feature scaling.
        
        Args:
            main_config: Main application configuration dictionary
        """
        self.main_config = main_config
