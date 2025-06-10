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
        If x_train_file is available, the sequence will lead up to the first datetime in x_train_file.
        Otherwise, it will lead up to the current time.

        Args:
            n_samples: Number of datetime samples to generate
            
        Returns:
            pd.Series: Generated datetime sequence
        """
        try:
            dataset_periodicity = self.config.get("dataset_periodicity", "1h")
            datetime_col_name = self.config.get("feeder_datetime_col_in_real_data", "DATE_TIME")
            x_train_file_path = self.config.get("x_train_file")
            
            end_datetime = None
            if x_train_file_path:
                try:
                    # Read only the first few rows to get the first datetime
                    real_data_first_row = pd.read_csv(x_train_file_path, nrows=1)
                    if datetime_col_name in real_data_first_row.columns:
                        end_datetime = pd.to_datetime(real_data_first_row[datetime_col_name].iloc[0])
                        print(f"Target end datetime for synthetic data (from x_train_file): {end_datetime}")
                    else:
                        print(f"Warning: Datetime column '{datetime_col_name}' not found in '{x_train_file_path}'. Defaulting end time.")
                except Exception as e:
                    print(f"Warning: Could not read first datetime from '{x_train_file_path}': {e}. Defaulting end time.")

            if end_datetime is None:
                end_datetime = pd.to_datetime(datetime.now().replace(microsecond=0))
                print(f"Target end datetime for synthetic data (current time): {end_datetime}")

            # Calculate start_datetime based on n_samples, periodicity, and end_datetime, skipping weekends
            # This is an iterative process because of weekend skipping.
            
            time_delta_map = {
                "1h": timedelta(hours=1), "1H": timedelta(hours=1),
                "15min": timedelta(minutes=15), "15T": timedelta(minutes=15), "15m": timedelta(minutes=15),
                "1min": timedelta(minutes=1), "1T": timedelta(minutes=1), "1m": timedelta(minutes=1),
                "daily": timedelta(days=1), "1D": timedelta(days=1)
            }
            time_step = time_delta_map.get(dataset_periodicity, timedelta(hours=1))

            generated_datetimes = []
            current_dt = end_datetime
            
            # Generate datetimes backwards from end_datetime, then reverse
            # This makes weekend skipping logic more straightforward when calculating the start
            temp_datetimes_reversed = []
            for _ in range(n_samples):
                # Move backwards, skipping weekends if not daily
                current_dt -= time_step
                if dataset_periodicity != "daily" and dataset_periodicity != "1D":
                    while current_dt.weekday() >= 5: # Saturday or Sunday
                        current_dt -= timedelta(days=1) # Move to Friday
                        # Adjust time if necessary, e.g., if time_step is hourly, ensure it's end of Friday
                        if time_step < timedelta(days=1):
                             current_dt = current_dt.replace(hour=23 if time_step.seconds // 3600 == 1 else (23 - (24 - time_step.seconds // 3600)), minute=59 if time_step.seconds % 3600 // 60 > 0 else 0, second=59 if time_step.seconds % 60 > 0 else 0)


                temp_datetimes_reversed.append(current_dt)
            
            # Reverse to get chronological order
            generated_datetimes = [dt.strftime('%Y-%m-%d %H:%M:%S') for dt in reversed(temp_datetimes_reversed)]
            
            # Ensure the first synthetic datetime is exactly one time_step before the end_datetime (or real data start)
            # after accounting for weekends.
            # The logic above should handle this by generating backwards.

            print(f"Generated {len(generated_datetimes)} datetimes. First: {generated_datetimes[0]}, Last: {generated_datetimes[-1]}")
            return pd.Series(pd.to_datetime(generated_datetimes))
            
        except Exception as e:
            print(f"Warning: Failed to generate target datetimes: {e}. Falling back to simple sequence.")
            # Fallback to simple sequence leading up to now (original fallback)
            return pd.Series([datetime.now() - timedelta(hours=(n_samples - 1 - i)) for i in range(n_samples)])
    
    def _generate_datetime_sequence(self, start_datetime_str: str, num_samples: int, 
                                   periodicity_str: str) -> list:
        """
        Generate datetime sequence based on periodicity.
        This method is now primarily a helper or can be deprecated if the logic in 
        _generate_target_datetimes fully covers its use cases.
        The new logic in _generate_target_datetimes is more robust for prepending.

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
        
        time_delta_map = {
            "1h": timedelta(hours=1), "1H": timedelta(hours=1),
            "15min": timedelta(minutes=15), "15T": timedelta(minutes=15), "15m": timedelta(minutes=15),
            "1min": timedelta(minutes=1), "1T": timedelta(minutes=1), "1m": timedelta(minutes=1),
            "daily": timedelta(days=1), "1D": timedelta(days=1)
        }
        
        time_step = time_delta_map.get(periodicity_str, timedelta(hours=1))
        datetimes = []
        
        for _ in range(num_samples):
            # Skip weekends for non-daily data
            if periodicity_str not in ["daily", "1D"]: # Check against "daily" and "1D"
                while current_dt.weekday() >= 5:  # Saturday or Sunday
                    current_dt += timedelta(days=1) # Move to Monday
                    # If hourly, set to start of Monday, e.g. 00:00, or a specific market open time
                    if time_step < timedelta(days=1):
                        current_dt = current_dt.replace(hour=0, minute=0, second=0) 
            
            datetimes.append(current_dt.strftime('%Y-%m-%d %H:%M:%S'))
            current_dt += time_step
        
        return datetimes

    def _convert_to_dataframe(self, generated_values, target_datetimes: pd.Series) -> pd.DataFrame:
        """
        Convert generated values to DataFrame with proper column names and datetime format.
        
        Args:
            generated_values: Generated values from generator plugin
            target_datetimes: Target datetime sequence
            
        Returns:
            pd.DataFrame: Formatted synthetic data
        """
        try:
            # Get feature names from generator configuration
            # Ensure 'DATE_TIME' is NOT in these feature_names if it's handled separately
            feature_names = self.config.get("generator_full_feature_names_ordered", [])
            # Remove 'DATE_TIME' if it exists in the feature list, as it's added separately
            if "DATE_TIME" in feature_names:
                feature_names = [name for name in feature_names if name != "DATE_TIME"]

            datetime_col_name = self.config.get("feeder_datetime_col_in_real_data", "DATE_TIME")
            
            # Handle different generated_values formats
            if isinstance(generated_values, list) and len(generated_values) > 0 and isinstance(generated_values[0], np.ndarray):
                 # Assuming list of arrays, common in some Keras model outputs for multiple outputs
                 # We are interested in the primary output, usually the first one.
                generated_values = generated_values[0] 
            
            if isinstance(generated_values, np.ndarray) and generated_values.ndim == 3:
                # If shape is (1, n_samples, n_features) or (batch_size, n_samples, n_features)
                # and we expect (n_samples, n_features)
                if generated_values.shape[0] == 1 and generated_values.shape[1] == len(target_datetimes):
                    generated_values = generated_values[0] 
                elif generated_values.shape[0] == len(target_datetimes) and generated_values.ndim == 3: # (n_samples, 1, n_features)
                     generated_values = generated_values.reshape(generated_values.shape[0], generated_values.shape[2])


            # Ensure generated_values has the correct number of features
            num_expected_features = len(feature_names)
            if generated_values.shape[1] != num_expected_features:
                print(f"Warning: Number of generated features ({generated_values.shape[1]}) does not match expected ({num_expected_features}). Adjusting columns.")
                if generated_values.shape[1] < num_expected_features:
                    # Pad with NaNs or zeros if fewer features are generated
                    padding = np.full((generated_values.shape[0], num_expected_features - generated_values.shape[1]), np.nan)
                    generated_values = np.hstack((generated_values, padding))
                else:
                    # Truncate if more features are generated
                    generated_values = generated_values[:, :num_expected_features]

            # Create DataFrame
            if feature_names: # Check if feature_names is not empty
                synthetic_data = pd.DataFrame(generated_values, columns=feature_names)
            else:
                # Fallback to generic column names if feature_names is empty
                n_features = generated_values.shape[1] if generated_values.ndim > 1 else 1
                synthetic_data = pd.DataFrame(generated_values, 
                                            columns=[f"feature_{i}" for i in range(n_features)])
            
            # Add datetime column and format it
            synthetic_data[datetime_col_name] = pd.to_datetime(target_datetimes.values).strftime('%Y-%m-%d %H:%M:%S')
            
            # Reorder columns to have datetime_col_name first, then others as per generator_full_feature_names_ordered
            final_ordered_columns = [datetime_col_name] + feature_names
            synthetic_data = synthetic_data[final_ordered_columns]

            return synthetic_data
            
        except Exception as e:
            print(f"Warning: DataFrame conversion failed, using basic format: {e}")
            # Fallback to simple DataFrame
            df = pd.DataFrame()
            df[self.config.get("feeder_datetime_col_in_real_data", "DATE_TIME")] = pd.to_datetime(target_datetimes.values).strftime('%Y-%m-%d %H:%M:%S')
            # Add generated data, ensuring it's 2D
            if hasattr(generated_values, 'ndim'):
                if generated_values.ndim == 1:
                    df['generated_data'] = generated_values
                elif generated_values.ndim > 1:
                    for i in range(generated_values.shape[1]):
                         df[f'feature_{i}'] = generated_values[:, i]
            else: # if it's a list or other non-array type
                 df['generated_data'] = generated_values

            return df
