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
        scaled_date_features_arr = self._calculate_single_timestamp_cyclical_features(dt_obj)
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
    
    def _calculate_single_timestamp_cyclical_features(self, datetime_obj: pd.Timestamp) -> np.ndarray:
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
        # Use "date_conditional_feature_names" from params to decide which features to generate
        # This is consistent with how it was used in _fill_sincos_date_features_prefill
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
            # Ensure day_of_year max is fetched correctly, e.g. 366 for leap, 365 otherwise
            # For simplicity, using a common config value or a fixed one.
            # The original code used feeder_max_day_of_year or 366.
            # Let's assume main_cfg provides this, or default to 366 for wider applicability.
            max_doy = main_cfg.get("feeder_max_day_of_year", 366 if datetime_obj.is_leap_year else 365)

            date_features.extend([
                np.sin(2 * np.pi * doy / max_doy), 
                np.cos(2 * np.pi * doy / max_doy)
            ])
            
        return np.array(date_features, dtype=np.float32)

    def add_cyclical_date_features(self, data_df: pd.DataFrame, datetime_col_name: str, cyclical_feature_specs: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Adds cyclical (sin/cos) datetime features to a DataFrame based on provided specifications.
        This method is designed to be called by GeneratorPlugin.prepare_features_for_discriminator.

        Args:
            data_df: Input DataFrame with a datetime column.
            datetime_col_name: Name of the datetime column in data_df.
            cyclical_feature_specs: A list of dictionaries, where each dict contains
                                    'feature_name' (e.g., 'day_of_month') and 'max_value'.
                                    Example: [{"feature_name": "day_of_month", "max_value": 31},
                                              {"feature_name": "hour_of_day", "max_value": 23}]

        Returns:
            The DataFrame with added cyclical features.
        """
        if not cyclical_feature_specs:
            print("DataGenerator: No cyclical feature specifications provided. Returning original DataFrame.")
            return data_df

        # Ensure datetime column is in datetime format
        if datetime_col_name not in data_df.columns:
            raise ValueError(f"Datetime column '{datetime_col_name}' not found in DataFrame.")
        dt_series = pd.to_datetime(data_df[datetime_col_name])

        processed_df = data_df.copy()

        for spec in cyclical_feature_specs:
            base_name = spec.get("feature_name")
            max_val = spec.get("max_value")

            if not base_name or max_val is None:
                print(f"DataGenerator: Warning - Invalid spec: {spec}. Skipping.")
                continue

            values = None
            denominator = None

            if base_name == "day_of_month":
                values = dt_series.dt.day
                denominator = max_val 
            elif base_name == "hour_of_day":
                values = dt_series.dt.hour
                denominator = max_val + 1 # Max value is 23, so 24 distinct values (0-23)
            elif base_name == "day_of_week":
                values = dt_series.dt.dayofweek # Monday=0, Sunday=6
                denominator = max_val + 1 # Max value is 6, so 7 distinct values (0-6)
            elif base_name == "day_of_year":
                values = dt_series.dt.dayofyear
                # max_val should be 365 or 366. If it's 365 for a leap year, it might be slightly off.
                # The GeneratorPlugin is responsible for providing the correct max_val.
                denominator = max_val 
            elif base_name == "month_of_year":
                values = dt_series.dt.month
                denominator = max_val # Max value is 12
            elif base_name == "week_of_year":
                values = dt_series.dt.isocalendar().week.astype(int)
                denominator = max_val # Max value is 52 or 53
            else:
                print(f"DataGenerator: Warning - Unknown base_name '{base_name}' for cyclical feature generation. Skipping.")
                continue
            
            if values is not None and denominator is not None and denominator > 0:
                processed_df[f"{base_name}_sin"] = np.sin(2 * np.pi * values / denominator)
                processed_df[f"{base_name}_cos"] = np.cos(2 * np.pi * values / denominator)
                print(f"DataGenerator: Generated cyclical features for '{base_name}'.")
            else:
                print(f"DataGenerator: Warning - Could not process cyclical feature for base_name '{base_name}'. Values, denominator invalid, or max_val was zero.")

        return processed_df

    def generate_cyclical_features_for_df(self, data_df: pd.DataFrame, datetime_col_name: str, feature_specs: List[tuple[str, Any]]) -> pd.DataFrame:
        """
        Generates cyclical (sin/cos) datetime features for an entire DataFrame.

        Args:
            data_df: Input DataFrame with a datetime column.
            datetime_col_name: Name of the datetime column in data_df.
            feature_specs: A list of tuples, where each tuple is (base_feature_name, max_value).
                           Example: [('day_of_month', 31), ('hour_of_day', 23)]

        Returns:
            A new DataFrame with the generated cyclical features, indexed like data_df.
        """
        cyclical_features_df = pd.DataFrame(index=data_df.index)
        dt_series = pd.to_datetime(data_df[datetime_col_name])

        for base_name, max_val_cfg in feature_specs:
            values = None
            denominator = None

            if base_name == "day_of_month":
                values = dt_series.dt.day
                denominator = max_val_cfg 
            elif base_name == "hour_of_day":
                values = dt_series.dt.hour
                denominator = max_val_cfg + 1
            elif base_name == "day_of_week":
                values = dt_series.dt.dayofweek
                denominator = max_val_cfg + 1
            elif base_name == "day_of_year":
                values = dt_series.dt.dayofyear
                # For day_of_year, max_val_cfg should be 365 or 366.
                # If it's passed as 365 (common year), but it's a leap year, it might be slightly off.
                # However, we will trust max_val_cfg as provided by GeneratorPlugin, which should handle this.
                denominator = max_val_cfg
            else:
                print(f"DataGenerator: Warning - Unknown base_name '{base_name}' for cyclical feature generation. Skipping.")
                continue
            
            if values is not None and denominator is not None:
                cyclical_features_df[f"{base_name}_sin"] = np.sin(2 * np.pi * values / denominator)
                cyclical_features_df[f"{base_name}_cos"] = np.cos(2 * np.pi * values / denominator)
            else:
                print(f"DataGenerator: Warning - Could not process cyclical feature for base_name '{base_name}'. Values or denominator was None.")

        return cyclical_features_df
    
    def set_main_config(self, main_config: Dict[str, Any]) -> None:
        """
        Set the main configuration for date feature scaling.
        
        Args:
            main_config: Main application configuration dictionary
        """
        self.main_config = main_config
    
    def generate_synthetic_data(self, n_samples: int, 
                               conditional_features: Optional[np.ndarray] = None,
                               initial_context_data: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Generate synthetic data using the composite generator model.
        
        Args:
            n_samples: Number of samples to generate
            conditional_features: Optional conditional features for generation
            initial_context_data: Optional initial context data
            
        Returns:
            pd.DataFrame: Generated synthetic data
        """
        # This method should use the main plugin's composite model to generate data
        # For now, we'll create a simple placeholder that generates random data
        # In practice, this would integrate with the GeneratorPlugin's model
        
        # Get feature names from params
        full_feature_names = self.params.get("full_feature_names_ordered", [])
        num_features = len(full_feature_names) if full_feature_names else self.params.get("num_features", 51)
        
        print(f"DataGenerator: Generating {n_samples} samples with {num_features} features")
        
        # Generate random data as placeholder
        # Shape: (n_samples, num_features) - one tick per sample, not sequences
        synthetic_data_2d = np.random.randn(n_samples, num_features)
        
        # Create column names
        if full_feature_names and len(full_feature_names) == num_features:
            columns = full_feature_names
        else:
            columns = [f"feature_{i}" for i in range(num_features)]
        
        df = pd.DataFrame(synthetic_data_2d, columns=columns)
        
        print(f"DataGenerator: Generated synthetic data shape: {df.shape}")
        return df
