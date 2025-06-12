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
        
    def generate_features_for_datetimes(self, target_datetimes: pd.Series, generator_model: Any) -> pd.DataFrame:
        """
        Generate synthetic features for specified target datetimes using the loaded generator model.
        This implements the 23->44 feature expansion post-processing pipeline.
        
        Args:
            target_datetimes: Series of target datetime values for generation
            generator_model: The loaded Keras generator model 
            
        Returns:
            pd.DataFrame: Generated synthetic data with 44 features + DATE_TIME column
        """
        print(f"SyntheticDataGenerator: Generating features for {len(target_datetimes)} target datetimes...")
        
        try:
            n_samples = len(target_datetimes)
            
            # Step 1: Generate conditional inputs using feeder plugin
            print("  Step 1: Generating conditional inputs...")
            feeder_outputs = self.feeder_plugin.generate(
                n_ticks_to_generate=n_samples,
                target_datetimes=target_datetimes
            )
            
            if feeder_outputs is None:
                raise RuntimeError("Feeder plugin failed to generate conditional inputs")
            
            # Step 2: Use generator model to produce 23 base features
            print("  Step 2: Generating 23 base features using generator model...")
            
            # Extract inputs from feeder outputs
            if isinstance(feeder_outputs, list) and len(feeder_outputs) > 0:
                # Handle list of dictionaries format
                noise_vectors = []
                context_vectors = []
                condition_vectors = []
                
                for output in feeder_outputs:
                    if isinstance(output, dict):
                        noise_vectors.append(output.get("noise", np.random.normal(0, 1, (100,))))
                        context_vectors.append(output.get("context", np.random.normal(0, 1, (64,))))
                        condition_vectors.append(output.get("conditions", np.random.normal(0, 1, (10,))))
                
                noise_input = np.array(noise_vectors)
                context_input = np.array(context_vectors) 
                conditions_input = np.array(condition_vectors)
                
            elif isinstance(feeder_outputs, dict):
                # Handle single dictionary format
                noise_input = feeder_outputs.get("noise", np.random.normal(0, 1, (n_samples, 100)))
                context_input = feeder_outputs.get("context", np.random.normal(0, 1, (n_samples, 64)))
                conditions_input = feeder_outputs.get("conditions", np.random.normal(0, 1, (n_samples, 10)))
            else:
                # Fallback: generate random inputs
                print("  Warning: Using fallback random inputs")
                noise_input = np.random.normal(0, 1, (n_samples, 100))
                context_input = np.random.normal(0, 1, (n_samples, 64))
                conditions_input = np.random.normal(0, 1, (n_samples, 10))
            
            # Generate 23 base features using the model
            model_inputs = [noise_input, context_input, conditions_input]
            base_features_23 = generator_model.predict(model_inputs, verbose=0)
            
            print(f"  Generated base features shape: {base_features_23.shape}")
            
            # Handle different output formats
            if len(base_features_23.shape) == 3:
                # If output is (batch_size, seq_len, features), we want (batch_size, features) 
                if base_features_23.shape[1] == 1:
                    # (batch_size, 1, features) -> (batch_size, features)
                    base_features_23 = base_features_23.squeeze(axis=1)
                else:
                    # Take the last timestep: (batch_size, seq_len, features) -> (batch_size, features)
                    base_features_23 = base_features_23[:, -1, :]
            
            # Verify we have 23 features
            if base_features_23.shape[1] != 23:
                print(f"  Warning: Expected 23 base features, got {base_features_23.shape[1]}. Adjusting...")
                if base_features_23.shape[1] < 23:
                    # Pad with zeros
                    padding = np.zeros((base_features_23.shape[0], 23 - base_features_23.shape[1]))
                    base_features_23 = np.hstack((base_features_23, padding))
                else:
                    # Truncate to 23
                    base_features_23 = base_features_23[:, :23]
            
            # Step 3: Expand 23 base features to 44 features using post-processing
            print("  Step 3: Expanding to 44 features with post-processing...")
            expanded_features_44 = self._expand_23_to_44_features(base_features_23, target_datetimes)
            
            # Step 4: Create DataFrame with proper feature names and DATE_TIME
            print("  Step 4: Creating final DataFrame...")
            feature_names = self.config.get("generator_full_feature_names_ordered", [])
            
            # Ensure we have 44 feature names
            if len(feature_names) < 44:
                # Pad with generic names
                while len(feature_names) < 44:
                    feature_names.append(f"feature_{len(feature_names)}")
            elif len(feature_names) > 44:
                # Truncate to 44
                feature_names = feature_names[:44]
            
            # Create DataFrame
            synthetic_df = pd.DataFrame(expanded_features_44, columns=feature_names)
            
            # Add DATE_TIME column at the beginning
            datetime_col_name = self.config.get("feeder_datetime_col_in_real_data", "DATE_TIME")
            synthetic_df[datetime_col_name] = target_datetimes.values
            
            # Reorder columns to have DATE_TIME first
            cols = [datetime_col_name] + [col for col in synthetic_df.columns if col != datetime_col_name]
            synthetic_df = synthetic_df[cols]
            
            print(f"  ✓ Generated synthetic data shape: {synthetic_df.shape}")
            return synthetic_df
            
        except Exception as e:
            print(f"  ❌ Error in generate_features_for_datetimes: {e}")
            import traceback
            traceback.print_exc()
            
            # Return empty DataFrame as fallback
            return pd.DataFrame()
    
    def _expand_23_to_44_features(self, base_features_23: np.ndarray, target_datetimes: pd.Series) -> np.ndarray:
        """
        Expand 23 base features to 44 features using post-processing.
        
        Based on the 23-feature architecture, this expands:
        23 base features -> 44 features (15 TI + 4 OHLC + 4 spreads + 2 external + 16 ticks + 3 seasonal)
        
        Args:
            base_features_23: Array of 23 base features (n_samples, 23)
            target_datetimes: Series of target datetimes for seasonal features
            
        Returns:
            np.ndarray: Expanded features (n_samples, 44)
        """
        n_samples = base_features_23.shape[0]
        expanded_features = np.zeros((n_samples, 44))
        
        try:
            # Extract 23 base features according to config mapping
            base_feature_names = self.config.get("generator_base_feature_names_ordered", [
                "OPEN", "HIGH", "LOW", "CLOSE", 
                "vix_close", "S&P500_Close", "BC-BO",
                "CLOSE_15m_tick_1", "CLOSE_15m_tick_2", "CLOSE_15m_tick_3", "CLOSE_15m_tick_4",
                "CLOSE_15m_tick_5", "CLOSE_15m_tick_6", "CLOSE_15m_tick_7", "CLOSE_15m_tick_8",
                "CLOSE_30m_tick_1", "CLOSE_30m_tick_2", "CLOSE_30m_tick_3", "CLOSE_30m_tick_4",
                "CLOSE_30m_tick_5", "CLOSE_30m_tick_6", "CLOSE_30m_tick_7", "CLOSE_30m_tick_8"
            ])
            
            # Map base features to their positions
            open_vals = base_features_23[:, 0]    # OPEN
            high_vals = base_features_23[:, 1]    # HIGH  
            low_vals = base_features_23[:, 2]     # LOW
            close_vals = base_features_23[:, 3]   # CLOSE
            vix_close = base_features_23[:, 4]    # vix_close
            sp500_close = base_features_23[:, 5]  # S&P500_Close
            bc_bo = base_features_23[:, 6]        # BC-BO
            
            # Extract sub-periodicity ticks (positions 7-22 in base features)
            close_15m_ticks = base_features_23[:, 7:15]   # CLOSE_15m_tick_1-8
            close_30m_ticks = base_features_23[:, 15:23]  # CLOSE_30m_tick_1-8
            
            # Calculate derived spreads
            bh_bl = high_vals - low_vals          # BH-BL = HIGH - LOW
            bh_bo = high_vals - open_vals         # BH-BO = HIGH - OPEN  
            bo_bl = open_vals - low_vals          # BO-BL = OPEN - LOW
            
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
            
            # Derived spreads (19-22)
            expanded_features[:, 19] = bc_bo
            expanded_features[:, 20] = bh_bl
            expanded_features[:, 21] = bh_bo
            expanded_features[:, 22] = bo_bl
            
            # External market data (23-24)
            expanded_features[:, 23] = sp500_close
            expanded_features[:, 24] = vix_close
            
            # Sub-periodicity ticks (25-40)
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
