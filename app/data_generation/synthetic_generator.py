#!/usr/bin/env python3
"""
synthetic_generator.py

Synthetic data generation module for TimeSeries-GAN.
Handles the complete synthetic data generation workflow including
noise generation, conditioning, and VAE decoder integration.

This module encapsulates synthetic data generation logic following
single responsibility principle and extreme separation of concerns.

Author: TimeSeries-GAN Team
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple


class SyntheticDataGenerator:
    """
    Generator for synthetic time series data using feeder and generator plugins.
    
    This class coordinates the synthetic data generation process:
    - Generates target datetime sequences for synthetic data
    - Uses feeder plugin to generate noise and conditional inputs
    - Uses generator plugin to transform noise into synthetic data
    - Handles initial window preparation for conditional generation
    
    Attributes:
        config: Configuration dictionary containing generation parameters
        feeder_plugin: Plugin instance for noise generation and conditioning
        generator_plugin: Plugin instance for synthetic data generation
    """
    
    def __init__(self, config: Dict[str, Any], feeder_plugin, generator_plugin):
        """
        Initialize synthetic data generator with configuration and plugins.
        
        Args:
            config: Configuration dictionary containing generation parameters
            feeder_plugin: Plugin instance for noise generation and conditioning
            generator_plugin: Plugin instance for synthetic data generation
        """
        self.config = config
        self.feeder_plugin = feeder_plugin
        self.generator_plugin = generator_plugin
    
    def generate(self, n_samples: int, initial_window: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Generate synthetic time series data.
        
        Args:
            n_samples: Number of synthetic samples to generate
            initial_window: Optional initial window data for conditional generation
            
        Returns:
            pd.DataFrame: Generated synthetic data with datetime column
            
        Raises:
            RuntimeError: If synthetic data generation fails
        """
        try:
            print(f"Generating {n_samples} synthetic samples...")
            
            # Generate target datetime sequence
            target_datetimes = self._generate_target_datetimes(n_samples)
            
            # Generate noise and conditional inputs using feeder plugin
            feeder_outputs = self.feeder_plugin.generate(
                n_ticks_to_generate=n_samples,
                target_datetimes=target_datetimes
            )
            
            # Generate synthetic data using generator plugin
            # Prepare initial_context_data from initial_window if available
            initial_context_data = None
            if initial_window and initial_window.get('features') is not None:
                # Convert initial window features to DataFrame for initial_context_data
                features_array = initial_window.get('features')
                if isinstance(features_array, np.ndarray) and features_array.size > 0:
                    # Create a simple DataFrame from the features array
                    # This will serve as initial context for the generator
                    initial_context_data = pd.DataFrame(features_array)
            
            generated_values = self.generator_plugin.generate(
                n_samples=n_samples,
                conditional_features=feeder_outputs,
                initial_context_data=initial_context_data
            )
            
            # Convert to DataFrame
            synthetic_data = self._convert_to_dataframe(generated_values, target_datetimes)
            
            print(f"✓ Synthetic data generation completed. Shape: {synthetic_data.shape}")
            return synthetic_data
            
        except Exception as e:
            raise RuntimeError(f"Synthetic data generation failed: {e}")
    
    def prepare_initial_window(self, preprocessor_plugin) -> Dict[str, Any]:
        """
        Prepare initial window for conditional generation using preprocessor.
        
        Args:
            preprocessor_plugin: Plugin instance for data preprocessing
            
        Returns:
            Dict containing initial window features, datetimes, and previous close
        """
        try:
            print("Preparing initial window for generation...")
            
            # Preprocess training data
            all_processed_datasets = preprocessor_plugin.run_preprocessing(config=self.config)
            X_train_processed = all_processed_datasets.get("x_train")
            datetimes_train_processed = all_processed_datasets.get("x_train_dates")
            
            if X_train_processed is None:
                raise ValueError("Preprocessor did not return 'x_train' data")
            
            # Convert to numpy array
            X_train_np = self._convert_to_numpy(X_train_processed)
            
            # Extract initial window
            decoder_input_window_size = self.generator_plugin.params.get("decoder_input_window_size", 50)
            if X_train_np.shape[0] < decoder_input_window_size:
                raise ValueError(f"Processed data length {X_train_np.shape[0]} < window size {decoder_input_window_size}")
            
            initial_window_features = X_train_np[-decoder_input_window_size:]
            
            # Extract previous close for log returns
            prev_close = self._extract_previous_close(X_train_np, all_processed_datasets, decoder_input_window_size)
            
            # Extract initial datetimes
            initial_datetimes = self._extract_initial_datetimes(datetimes_train_processed, decoder_input_window_size)
            
            return {
                'features': initial_window_features,
                'datetimes': initial_datetimes,
                'prev_close': prev_close
            }
            
        except Exception as e:
            print(f"ERROR: Failed to prepare initial window: {e}")
            return {'features': None, 'datetimes': None, 'prev_close': None}
    
    def _generate_target_datetimes(self, n_samples: int) -> pd.Series:
        """
        Generate target datetime sequence for synthetic data.
        
        Args:
            n_samples: Number of datetime samples to generate
            
        Returns:
            pd.Series: Generated datetime sequence
        """
        try:
            # Get datetime configuration
            dataset_periodicity = self.config.get("dataset_periodicity", "1h")
            start_dt_str = self.config.get("start_datetime", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            
            # Generate datetime sequence
            datetimes = self._generate_datetime_sequence(start_dt_str, n_samples, dataset_periodicity)
            
            return pd.Series(pd.to_datetime(datetimes))
            
        except Exception as e:
            print(f"Warning: Failed to generate target datetimes: {e}")
            # Fallback to simple sequence
            return pd.Series([datetime.now() + timedelta(hours=i) for i in range(n_samples)])
    
    def _generate_datetime_sequence(self, start_datetime_str: str, num_samples: int, 
                                   periodicity_str: str) -> list:
        """
        Generate datetime sequence based on periodicity.
        
        Args:
            start_datetime_str: Starting datetime as string
            num_samples: Number of samples to generate
            periodicity_str: Time period between samples
            
        Returns:
            List of datetime strings
        """
        if num_samples == 0:
            return []
        
        try:
            current_dt = pd.to_datetime(start_datetime_str)
        except Exception:
            current_dt = pd.to_datetime(datetime.now().replace(microsecond=0))
        
        # Define time step mapping
        time_delta_map = {
            "1h": timedelta(hours=1), "1H": timedelta(hours=1),
            "15min": timedelta(minutes=15), "15T": timedelta(minutes=15), "15m": timedelta(minutes=15),
            "1min": timedelta(minutes=1), "1T": timedelta(minutes=1), "1m": timedelta(minutes=1),
            "daily": timedelta(days=1), "1D": timedelta(days=1)
        }
        
        time_step = time_delta_map.get(periodicity_str, timedelta(hours=1))
        datetimes = []
        
        for i in range(num_samples):
            # Skip weekends for non-daily data
            if periodicity_str != "daily":
                while current_dt.weekday() >= 5:  # Saturday or Sunday
                    current_dt += timedelta(days=1)
            
            datetimes.append(current_dt.strftime('%Y-%m-%d %H:%M:%S'))
            current_dt += time_step
        
        return datetimes
    
    def _convert_to_numpy(self, data) -> np.ndarray:
        """
        Convert data to numpy array with proper dtype.
        
        Args:
            data: Data to convert (DataFrame or array)
            
        Returns:
            np.ndarray: Converted data
        """
        if isinstance(data, pd.DataFrame):
            data_np = data.values.astype(np.float32)
        else:
            data_np = data.astype(np.float32)
        
        # Handle 3D to 2D conversion if needed
        if data_np.ndim == 3:
            data_np = data_np[:, -1, :]
        
        return data_np
    
    def _extract_previous_close(self, X_train_np: np.ndarray, all_processed_datasets: Dict, 
                               decoder_input_window_size: int) -> Optional[float]:
        """
        Extract previous close value for log return calculations.
        
        Args:
            X_train_np: Training data array
            all_processed_datasets: Processed datasets dictionary
            decoder_input_window_size: Size of decoder input window
            
        Returns:
            Previous close value or None if not available
        """
        try:
            feature_names = all_processed_datasets.get("feature_names", [])
            if 'CLOSE' in feature_names and X_train_np.shape[0] > decoder_input_window_size:
                close_idx = feature_names.index('CLOSE')
                return float(X_train_np[-(decoder_input_window_size + 1), close_idx])
        except Exception as e:
            print(f"Warning: Could not extract previous close: {e}")
        
        return None
    
    def _extract_initial_datetimes(self, datetimes_train_processed, 
                                  decoder_input_window_size: int) -> Optional[pd.Series]:
        """
        Extract initial datetime sequence for window.
        
        Args:
            datetimes_train_processed: Processed datetime data
            decoder_input_window_size: Size of decoder input window
            
        Returns:
            Initial datetime series or None if not available
        """
        try:
            if datetimes_train_processed is not None:
                dt_series = pd.Series(pd.to_datetime(datetimes_train_processed))
                if len(dt_series) >= decoder_input_window_size:
                    return dt_series.iloc[-decoder_input_window_size:].reset_index(drop=True)
        except Exception as e:
            print(f"Warning: Could not extract initial datetimes: {e}")
        
        return None
    
    def _convert_to_dataframe(self, generated_values, target_datetimes: pd.Series) -> pd.DataFrame:
        """
        Convert generated values to DataFrame with proper column names.
        
        Args:
            generated_values: Generated values from generator plugin
            target_datetimes: Target datetime sequence
            
        Returns:
            pd.DataFrame: Formatted synthetic data
        """
        try:
            # Get feature names from generator configuration
            feature_names = self.generator_plugin.params.get("full_feature_names_ordered", [])
            datetime_col_name = self.config.get("datetime_col_name", "DATE_TIME")
            
            # Handle different generated_values formats
            if isinstance(generated_values, list) and len(generated_values) == 1:
                generated_values = generated_values[0]
            
            if isinstance(generated_values, np.ndarray) and generated_values.ndim == 3:
                generated_values = generated_values[0]
            
            # Create DataFrame
            if feature_names and len(feature_names) > 0:
                synthetic_data = pd.DataFrame(generated_values, columns=feature_names)
            else:
                # Fallback to generic column names
                n_features = generated_values.shape[1] if generated_values.ndim > 1 else 1
                synthetic_data = pd.DataFrame(generated_values, 
                                            columns=[f"feature_{i}" for i in range(n_features)])
            
            # Add datetime column
            synthetic_data[datetime_col_name] = target_datetimes.values
            
            return synthetic_data
            
        except Exception as e:
            print(f"Warning: DataFrame conversion failed, using basic format: {e}")
            # Fallback to simple DataFrame
            return pd.DataFrame({
                'generated_data': generated_values.flatten() if hasattr(generated_values, 'flatten') else [0],
                self.config.get("datetime_col_name", "DATE_TIME"): target_datetimes.values
            })
