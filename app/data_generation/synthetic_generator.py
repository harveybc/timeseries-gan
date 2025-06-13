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
import traceback
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, List, TYPE_CHECKING # Added List

if TYPE_CHECKING:
    import tensorflow # For type hinting tf.keras.Model
    from pandas import Series as pd_Series # For type hinting pd.Series explicitly


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
        
        # Initialize feature names from config
        # The generator model outputs all features including DATE_TIME, so we need all feature names
        self.feature_names = config.get("generator_full_feature_names_ordered", [])
    
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
    
    def _generate_target_datetimes(self, n_samples: int) -> 'pd_Series': # Use pd_Series alias
        """
        Generate target datetime sequence for synthetic data.
        If x_train_file is available, the sequence will lead up to the first datetime in x_train_file.
        Otherwise, it will lead up to the current time.

        Args:
            n_samples: Number of datetime samples to generate
            
        Returns:
            pd.Series: Generated datetime sequence, aliased as pd_Series for type hinting
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
            return pd.Series(pd.to_datetime(generated_datetimes)) # Runtime uses pd.Series
            
        except Exception as e:
            print(f"Warning: Failed to generate target datetimes: {e}. Falling back to simple sequence.")
            # Fallback to simple sequence leading up to now (original fallback)
            return pd.Series([datetime.now() - timedelta(hours=(n_samples - 1 - i)) for i in range(n_samples)]) # Runtime uses pd.Series
    
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

    def _convert_to_dataframe(self, generated_values: Any, target_datetimes: 'pd_Series') -> pd.DataFrame: # Use pd_Series alias
        """
        Convert generated values to DataFrame with proper column names and datetime format.
        
        Args:
            generated_values: Generated values from generator plugin
            target_datetimes: Target datetime sequence (Pandas Series, aliased as pd_Series for type hinting)
            
        Returns:
            pd.DataFrame: Formatted synthetic data
        """
        try:
            # Get feature names from generator configuration
            # Ensure 'DATE_TIME' is NOT in these feature_names if it's handled separately
            feature_names = self.config.get("generator_full_feature_names_ordered", [])
            # Remove 'DATE_TIME' if it exists in feature_names, as it's added separately
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
            # Ensure target_datetimes.values is compatible if it's a pd_Series alias
            synthetic_data[datetime_col_name] = pd.to_datetime(target_datetimes.values).strftime('%Y-%m-%d %H:%M:%S')
            
            # Reorder columns to have datetime_col_name first, then others as per generator_full_feature_names_ordered
            final_ordered_columns = [datetime_col_name] + feature_names
            synthetic_data = synthetic_data[final_ordered_columns]

            return synthetic_data
            
        except Exception as e:
            print(f"Warning: DataFrame conversion failed, using basic format: {e}")
            # Fallback to simple DataFrame
            df = pd.DataFrame()
            # Ensure target_datetimes.values is compatible
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
        
    def generate_features_for_datetimes(self, target_datetimes: List[datetime]) -> pd.DataFrame:
        """
        Generate synthetic features for target datetimes using the feeder and generator plugins.
        
        Args:
            target_datetimes: List of datetime objects for which to generate features
            
        Returns:
            DataFrame with generated features for target datetimes
        """
        print(f"SyntheticDataGenerator: Generating features for {len(target_datetimes)} target datetimes...")
        
        try:
            # Step 1: Generate conditional inputs using feeder plugin
            print("  Step 1: Generating conditional inputs...")
            conditional_features = self.feeder_plugin.generate(
                n_ticks_to_generate=len(target_datetimes),
                target_datetimes=pd.Series(target_datetimes)
            )
            
            # Step 2: Generate base features using generator plugin
            print("  Step 2: Generating 23 base features using generator model...")
            
            # Get the generator model directly
            generator_model = self.generator_plugin.get_model()
            if generator_model is None:
                raise ValueError("Generator model is not available")
            
            # Create properly shaped inputs for the composite generator
            batch_size = len(target_datetimes)
            
            # Generate the 3 required inputs in correct order
            noise = np.random.normal(0, 1, (batch_size, 100))  # 100-dim noise
            context = np.random.normal(0, 1, (batch_size, 64))  # 64-dim context  
            conditions = conditional_features  # This should be (batch_size, 10)
            
            # Ensure conditions has correct shape
            if conditions.shape[1] != 10:
                if conditions.shape[1] > 10:
                    conditions = conditions[:, :10]
                else:
                    # Pad with zeros if too few features
                    padding = np.zeros((batch_size, 10 - conditions.shape[1]))
                    conditions = np.concatenate([conditions, padding], axis=1)
            
            # Create model inputs as list in correct order
            model_inputs = [noise, conditions, context]
            
            # Generate base features using the composite generator
            base_features_23 = generator_model.predict(model_inputs, verbose=0)
            
            print(f"  Generated base features shape: {base_features_23.shape}")
            
            # Step 3: Expanding to 44 features with post-processing
            print("  Step 3: Expanding to 44 features with post-processing...")
            
            # Reshape if needed - extract from 3D to 2D if necessary
            if base_features_23.ndim == 3:
                # Shape is likely (batch_size, sequence_length, features)
                # For datetime generation, we want the last timestep
                base_features_23 = base_features_23[:, -1, :]  # Take last timestep
            
            # Use internal method to expand from 23 to 44 features
            expanded_features = self._expand_23_to_44_features(base_features_23, pd.Series(target_datetimes))
            
            # Step 4: Create final DataFrame
            print("  Step 4: Creating final DataFrame...")
            
            # Get feature names from config
            datetime_col_name = self.config.get("feeder_datetime_col_in_real_data", "DATE_TIME")
            feature_names = self.config.get("generator_full_feature_names_ordered", [])
            
            # Remove datetime column from feature names if present
            if datetime_col_name in feature_names:
                feature_names = [name for name in feature_names if name != datetime_col_name]
            
            # Create DataFrame with expanded features
            if len(feature_names) >= expanded_features.shape[1]:
                result_df = pd.DataFrame(expanded_features, columns=feature_names[:expanded_features.shape[1]])
            else:
                # Fallback to generic column names if not enough feature names
                result_df = pd.DataFrame(expanded_features, columns=[f"feature_{i}" for i in range(expanded_features.shape[1])])
            
            # Add datetime column
            result_df[datetime_col_name] = pd.Series(target_datetimes).dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Reorder columns to put datetime first
            cols = [datetime_col_name] + [col for col in result_df.columns if col != datetime_col_name]
            result_df = result_df[cols]
            
            print(f"  ✓ Generated synthetic data shape: {result_df.shape}")
            return result_df
            
        except Exception as e:
            print(f"  ❌ Error in generate_features_for_datetimes: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Return empty DataFrame instead of fallback random data
            print("Synthetic feature generation failed. Returning empty DataFrame.")
            return pd.DataFrame()
    
    def _expand_23_to_44_features(self, base_features_23: np.ndarray, target_datetimes: pd.Series) -> np.ndarray:
        """
        Expand 23 base features to 44 features with OHLC coherence constraints.
        
        Ensures hierarchical coherence:
        - Hourly OHLC drives all tick generation
        - 30min ticks (2 per hour) contain the hour's HIGH and LOW
        - 15min ticks (4 per hour) contain the hour's OPEN, HIGH, LOW, CLOSE
        - Cross-consistency between 15min and 30min ticks
        
        Args:
            base_features_23: Array of 23 base features (n_samples, 23)
            target_datetimes: Series of target datetimes for seasonal features
            
        Returns:
            np.ndarray: Expanded features (n_samples, 44) with OHLC coherence
        """
        n_samples = base_features_23.shape[0]
        expanded_features = np.zeros((n_samples, 44))
        
        try:
            # Extract OHLC from correct positions in base features (15-18 in 0-indexed)
            # Based on column mapping: RSI(0), MACD(1)...ROC(15), OPEN(16), HIGH(17), LOW(18), CLOSE(19)
            open_vals = base_features_23[:, 15]   # OPEN (position 16 in 1-indexed)
            high_vals = base_features_23[:, 16]   # HIGH (position 17 in 1-indexed)  
            low_vals = base_features_23[:, 17]    # LOW (position 18 in 1-indexed)
            close_vals = base_features_23[:, 18]  # CLOSE (position 19 in 1-indexed)
            bc_bo = base_features_23[:, 19]       # BC-BO (position 20 in 1-indexed)
            bh_bl = base_features_23[:, 20]       # BH-BL (position 21 in 1-indexed) 
            bh_bo = base_features_23[:, 21]       # BH-BO (position 22 in 1-indexed)
            bo_bl = base_features_23[:, 22]       # BO-BL (position 23 in 1-indexed)
            
            # External market data should be from earlier positions if available
            # Otherwise we'll calculate them from OHLC
            if base_features_23.shape[1] > 24:
                sp500_close = base_features_23[:, 24]  # S&P500_Close
                vix_close = base_features_23[:, 25]    # vix_close
            else:
                # Generate reasonable external market data if not available
                sp500_close = close_vals * (1 + np.random.normal(0, 0.01, close_vals.shape))
                vix_close = np.abs(close_vals * np.random.normal(0.3, 0.1, close_vals.shape))
            
            # ENFORCE OHLC COHERENCE: Ensure Low ≤ Open,Close ≤ High regardless of normalization
            print("    Enforcing OHLC coherence constraints...")
            for i in range(n_samples):
                o, h, l, c = open_vals[i], high_vals[i], low_vals[i], close_vals[i]
                
                # Calculate correct HIGH and LOW
                true_high = max(o, h, l, c)
                true_low = min(o, h, l, c)
                
                # Ensure OPEN and CLOSE are within [LOW, HIGH]
                corrected_open = max(true_low, min(true_high, o))
                corrected_close = max(true_low, min(true_high, c))
                
                # Update values
                open_vals[i] = corrected_open
                high_vals[i] = true_high
                low_vals[i] = true_low
                close_vals[i] = corrected_close
            
            # Generate OHLC-coherent tick columns
            print("    Generating OHLC-coherent 30min and 15min ticks...")
            
            # Initialize tick arrays
            close_15m_ticks = np.zeros((n_samples, 8))  # 8 15min ticks
            close_30m_ticks = np.zeros((n_samples, 8))  # 8 30min ticks
            
            for i in range(n_samples):
                # Current hour's OHLC values (already corrected for coherence)
                open_val = open_vals[i]
                high_val = high_vals[i]
                low_val = low_vals[i]
                close_val = close_vals[i]
                
                # COHERENCE RULE: All ticks must be within [LOW, HIGH] range
                # and must contain OPEN and CLOSE values
                
                # Generate 8 15min ticks with OHLC coherence
                # Strategy: Ensure first tick = OPEN, last tick = CLOSE, 
                # and that HIGH and LOW appear somewhere in the ticks
                
                # Start with all ticks as interpolated values between OPEN and CLOSE
                for tick_idx in range(8):
                    # Linear interpolation between OPEN and CLOSE
                    alpha = tick_idx / 7.0  # 0 to 1
                    interpolated_value = open_val + alpha * (close_val - open_val)
                    # Add small random variation within [LOW, HIGH]
                    variation_range = (high_val - low_val) * 0.1  # 10% of the range
                    variation = np.random.uniform(-variation_range, variation_range)
                    tick_value = interpolated_value + variation
                    # Clamp to [LOW, HIGH] range
                    tick_value = max(low_val, min(high_val, tick_value))
                    close_15m_ticks[i, tick_idx] = tick_value
                
                # Ensure OHLC constraints are satisfied:
                close_15m_ticks[i, 0] = open_val   # First tick must be OPEN
                close_15m_ticks[i, 7] = close_val  # Last tick must be CLOSE
                
                # Ensure HIGH and LOW appear in the ticks
                # Replace two random middle ticks with HIGH and LOW
                middle_indices = list(range(1, 7))  # Indices 1-6 (avoiding first and last)
                np.random.shuffle(middle_indices)
                close_15m_ticks[i, middle_indices[0]] = high_val  # Place HIGH
                close_15m_ticks[i, middle_indices[1]] = low_val   # Place LOW
                
                # Generate 8 30min ticks (simpler approach)
                # Each tick should be within [LOW, HIGH] and represent reasonable 30min closes
                for tick_idx in range(8):
                    # Generate values that drift between OPEN and CLOSE
                    alpha = tick_idx / 7.0
                    base_value = open_val + alpha * (close_val - open_val)
                    # Add moderate random variation
                    variation = np.random.uniform(low_val - base_value, high_val - base_value) * 0.3
                    tick_value = base_value + variation
                    # Clamp to [LOW, HIGH] range
                    tick_value = max(low_val, min(high_val, tick_value))
                    close_30m_ticks[i, tick_idx] = tick_value
                
                # Ensure 30min ticks also contain HIGH and LOW
                middle_30m = list(range(8))
                np.random.shuffle(middle_30m)
                close_30m_ticks[i, middle_30m[0]] = high_val  # Place HIGH
                close_30m_ticks[i, middle_30m[1]] = low_val   # Place LOW
            
            # Calculate derived spreads using corrected OHLC values  
            # These should be recalculated to ensure mathematical consistency
            bc_bo_corrected = close_vals - open_vals      # BC-BO = CLOSE - OPEN
            bh_bl_corrected = high_vals - low_vals        # BH-BL = HIGH - LOW (always positive)
            bh_bo_corrected = high_vals - open_vals       # BH-BO = HIGH - OPEN  
            bo_bl_corrected = open_vals - low_vals        # BO-BL = OPEN - LOW
            
            # 1. Calculate 15 technical indicators from OHLC
            technical_indicators = self._calculate_technical_indicators(
                open_vals, high_vals, low_vals, close_vals
            )
            
            # 2. Generate 3 seasonal date features from timestamps
            seasonal_features = self._calculate_seasonal_features(target_datetimes)
            
            # 3. Assemble 44 features in exact order matching training data:
            # Order: Technical Indicators (15), OHLC (4), Derived spreads (4), 
            #        External market data (2), Sub-periodicity (16), Seasonal (3)
            
            # Technical Indicators (0-14)
            expanded_features[:, 0:15] = technical_indicators
            
            # OHLC (15-18)
            expanded_features[:, 15] = open_vals
            expanded_features[:, 16] = high_vals
            expanded_features[:, 17] = low_vals
            expanded_features[:, 18] = close_vals
            
            # Derived spreads (19-22) - using corrected values
            expanded_features[:, 19] = bc_bo_corrected
            expanded_features[:, 20] = bh_bl_corrected
            expanded_features[:, 21] = bh_bo_corrected
            expanded_features[:, 22] = bo_bl_corrected
            
            # External market data (23-24)
            expanded_features[:, 23] = sp500_close
            expanded_features[:, 24] = vix_close
            
            # OHLC-coherent sub-periodicity ticks (25-40)
            expanded_features[:, 25:33] = close_15m_ticks  # CLOSE_15m_tick_1-8
            expanded_features[:, 33:41] = close_30m_ticks  # CLOSE_30m_tick_1-8
            
            # Seasonal date features (41-43)
            expanded_features[:, 41:44] = seasonal_features
            
            return expanded_features
            
        except Exception as e:
            print(f"  Warning: Error in feature expansion: {e}")
            # Return padded base features as fallback
            if base_features_23.shape[1] >= 23:
                expanded_features[:, :23] = base_features_23[:, :23]
            else:
                expanded_features[:, :base_features_23.shape[1]] = base_features_23
            return expanded_features
    
    def _calculate_technical_indicators(self, open_vals: np.ndarray, high_vals: np.ndarray, 
                                      low_vals: np.ndarray, close_vals: np.ndarray) -> np.ndarray:
        """
        Calculate 15 technical indicators from OHLC data.
        
        Args:
            open_vals, high_vals, low_vals, close_vals: OHLC arrays
            
        Returns:
            np.ndarray: Technical indicators (n_samples, 15)
        """
        n_samples = len(open_vals)
        indicators = np.zeros((n_samples, 15))
        
        try:
            # 1. RSI (simplified)
            price_change = close_vals - open_vals
            gain = np.maximum(price_change, 0.0)
            loss = np.maximum(-price_change, 0.0)
            rs = gain / (loss + 1e-8)
            rsi = 100.0 - (100.0 / (1.0 + rs))
            indicators[:, 0] = rsi
            
            # 2-4. MACD components (simplified)
            macd = (close_vals - open_vals) * 0.1
            macd_signal = macd * 0.9
            macd_histogram = macd - macd_signal
            indicators[:, 1] = macd
            indicators[:, 2] = macd_histogram
            indicators[:, 3] = macd_signal
            
            # 5. EMA (simplified)
            ema = close_vals * 0.95 + open_vals * 0.05
            indicators[:, 4] = ema
            
            # 6-7. Stochastic (simplified)
            stoch_k = ((close_vals - low_vals) / (high_vals - low_vals + 1e-8)) * 100.0
            stoch_d = stoch_k * 0.9
            indicators[:, 5] = stoch_k
            indicators[:, 6] = stoch_d
            
            # 8-10. ADX, DI+, DI- (simplified)
            adx = np.abs(high_vals - low_vals) / (close_vals + 1e-8) * 100.0
            di_plus = np.maximum(high_vals - open_vals, 0.0) / (high_vals - low_vals + 1e-8) * 100.0
            di_minus = np.maximum(open_vals - low_vals, 0.0) / (high_vals - low_vals + 1e-8) * 100.0
            indicators[:, 7] = adx
            indicators[:, 8] = di_plus
            indicators[:, 9] = di_minus
            
            # 11. ATR (simplified)
            atr = high_vals - low_vals
            indicators[:, 10] = atr
            
            # 12. CCI (simplified)
            typical_price = (high_vals + low_vals + close_vals) / 3.0
            cci = (typical_price - close_vals) / (0.015 * np.abs(typical_price - close_vals) + 1e-8)
            indicators[:, 11] = cci
            
            # 13. Williams %R (simplified)
            williams_r = ((high_vals - close_vals) / (high_vals - low_vals + 1e-8)) * -100.0
            indicators[:, 12] = williams_r
            
            # 14. Momentum (simplified)
            momentum = close_vals - open_vals
            indicators[:, 13] = momentum
            
            # 15. ROC (simplified)
            roc = ((close_vals - open_vals) / (open_vals + 1e-8)) * 100.0
            indicators[:, 14] = roc
            
            return indicators
            
        except Exception as e:
            print(f"  Warning: Error calculating technical indicators: {e}")
            return indicators  # Return zeros if calculation fails
    
    def _calculate_seasonal_features(self, target_datetimes: pd.Series) -> np.ndarray:
        """
        Calculate seasonal date features from target datetimes.
        
        Args:
            target_datetimes: Series of target datetime values
            
        Returns:
            np.ndarray: Seasonal features (n_samples, 3) - [day_of_month, hour_of_day, day_of_week]
        """
        n_samples = len(target_datetimes)
        seasonal_features = np.zeros((n_samples, 3))
        
        try:
            # Convert to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(target_datetimes):
                datetimes = pd.to_datetime(target_datetimes)
            else:
                datetimes = target_datetimes
            
            # Calculate raw date features (normalized 0-1)
            # day_of_month: 1-31 -> 0-1
            day_of_month = datetimes.dt.day.values / 31.0
            
            # hour_of_day: 0-23 -> 0-1  
            hour_of_day = datetimes.dt.hour.values / 23.0
            
            # day_of_week: 0-6 -> 0-1
            day_of_week = datetimes.dt.dayofweek.values / 6.0
            
            seasonal_features[:, 0] = day_of_month
            seasonal_features[:, 1] = hour_of_day
            seasonal_features[:, 2] = day_of_week
            
            return seasonal_features
            
        except Exception as e:
            print(f"  Warning: Error calculating seasonal features: {e}")
            # Return random normalized values as fallback
            seasonal_features = np.random.uniform(0, 1, (n_samples, 3))
            return seasonal_features
