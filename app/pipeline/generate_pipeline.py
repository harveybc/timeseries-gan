#!/usr/bin/env python3
"""
generate_pipeline.py

Data generation and evaluation pipeline for TimeSeries-GAN.
Handles synthetic data generation workflow including data synthesis,
real data integration, and optional evaluation against ground truth.

This module encapsulates generation-specific logic following
single responsibility principle and extreme separation of concerns.

Author: TimeSeries-GAN Team
"""

import os
import sys
import traceback
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, TYPE_CHECKING, List
from datetime import datetime, timedelta

# Conditional import for type hinting
if TYPE_CHECKING:
    import tensorflow # For type hinting tensorflow.keras.Model
    from pandas import Series as pd_Series # For type hinting pandas.Series
    KerasModel = tensorflow.keras.Model
    PandasSeries = pd_Series
else:
    # Fallback to Any if imports are not resolvable by linter/type checker
    KerasModel = Any
    PandasSeries = Any

# Actual TensorFlow import, handled with a check before use
try:
    import tensorflow as tf_runtime 
except ImportError:
    tf_runtime = None

from app.data_generation.synthetic_generator import SyntheticDataGenerator
from app.data_generation.real_data_processor import RealDataProcessor
from app.evaluation.metrics_evaluator import MetricsEvaluator
from app.utils.output_manager import OutputManager


class GeneratePipeline:
    """
    Pipeline for generating synthetic data and performing evaluation.
    
    This pipeline coordinates the complete generation workflow:
    - Generates synthetic time series data using feeder and generator plugins
    - Processes and integrates real data segments when configured
    - Combines synthetic and real data for output
    - Performs optional evaluation using evaluator plugin
    - Handles output file management and result persistence
    
    Attributes:
        config: Configuration dictionary containing generation parameters
        feeder_plugin: Plugin instance for noise generation and conditioning
        generator_plugin: Plugin instance for synthetic data generation
        evaluator_plugin: Plugin instance for evaluation metrics (optional)
        generator_model: The loaded Keras generator model.
    """

    def __init__(self, config: Dict[str, Any], feeder_plugin: Any, generator_plugin: Any,
                 evaluator_plugin: Optional[Any]):
        """
        Initialize generation pipeline with configuration and plugin instances.
        
        Args:
            config: Configuration dictionary containing generation parameters
            feeder_plugin: Plugin instance for noise generation and conditioning
            generator_plugin: Plugin instance for synthetic data generation
            evaluator_plugin: Plugin instance for evaluation metrics (optional)
        """
        self.config = config
        self.feeder_plugin = feeder_plugin
        self.generator_plugin = generator_plugin
        self.evaluator_plugin = evaluator_plugin
        self.generator_model: Optional[KerasModel] = None # Use aliased type

        # Initialize component modules
        self.synthetic_generator = SyntheticDataGenerator(config, feeder_plugin, generator_plugin)
        self.real_data_processor = RealDataProcessor(config)
        self.metrics_evaluator = MetricsEvaluator(config)
        self.output_manager = OutputManager(config)

    def execute(self) -> None:
        """
        Execute the full data generation and evaluation pipeline.
        """
        print("Starting data generation and evaluation pipeline...")
        try:
            self._validate_generation_config()
            if tf_runtime is None:
                raise ImportError("TensorFlow is required for loading models in generate mode but is not installed/found.")
            self._load_models()

            evaluation_stage = self._process_evaluation_stage()
            real_data_segment_df = self._process_real_data()
            combined_data_df = self._prepend_and_align_data(real_data_segment_df)

            if combined_data_df.empty:
                print("Combined data generation resulted in an empty DataFrame. Skipping further processing.")
                return

            # Check if evaluation is enabled in config
            use_evaluation = self.config.get("use_evaluation", False)
            if use_evaluation and self.evaluator_plugin and real_data_segment_df is not None and not real_data_segment_df.empty:
                print("✓ Evaluation enabled in config. Running evaluation...")
                self._evaluate_data(combined_data_df, real_data_segment_df, evaluation_stage)
            elif not use_evaluation:
                print("ℹ️ Evaluation disabled in config (use_evaluation=False). Skipping evaluation phase.")
            elif not self.evaluator_plugin:
                print("ℹ️ Evaluator plugin not available, skipping evaluation.")
            else:
                print("ℹ️ Real data segment is empty, skipping evaluation.")

            self._save_outputs(combined_data_df, evaluation_stage)

            print("✓ Data generation and evaluation pipeline completed successfully.")

        except Exception as e:
            print(f"❌ Data generation pipeline failed: {e}")
            traceback.print_exc()
            sys.exit(1)

    def _load_models(self) -> None:
        """Load pre-trained models required for generation mode."""
        print("Loading pre-trained models for generation...")
        if tf_runtime is None:
            raise ImportError("TensorFlow is required for loading models but was not imported successfully.")

        generator_model_path = self.config.get("load_generator_sequential_model_file")
        if not generator_model_path or not os.path.exists(generator_model_path):
            raise FileNotFoundError(f"Generator model file not found or path not specified: {generator_model_path}")

        try:
            # self.generator_model should be Optional[KerasModel]
            loaded_model = tf_runtime.keras.models.load_model(generator_model_path, compile=False)
            self.generator_model = loaded_model # Assign after successful load
            print(f"✓ Generator model loaded successfully from: {generator_model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load Keras generator model: {e}")

    def _validate_generation_config(self) -> None:
        """
        Validate generation configuration parameters.
        
        Checks for required configuration keys and validates data
        file availability.
        
        Raises:
            ValueError: If required configuration is missing
        """
        # Check required configuration keys
        required_keys = [
            "x_train_file", "n_samples", "dataset_periodicity",
            "load_generator_sequential_model_file", "generated_data_file", "output_dir"
        ]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Required configuration key '{key}' missing for generate mode")

        # Validate training data file existence (used by RealDataProcessor and for prepending)
        x_train_file_path = self.config["x_train_file"]
        if not x_train_file_path or not os.path.exists(x_train_file_path):
            print(f"Warning: Training data file '{x_train_file_path}' not found. Real data processing might fail.")
        print("✓ Generation configuration validated")
        if x_train_file_path and os.path.exists(x_train_file_path):
            print(f"✓ Training data file verified: {x_train_file_path}")

    def _process_evaluation_stage(self) -> str:
        """
        Process evaluation stage and update output paths accordingly.
        
        Returns:
            str: The evaluation stage identifier
        """
        evaluation_stage = self.config.get("evaluation_stage", "baseline")
        print(f"Evaluation stage: {evaluation_stage}")
        return evaluation_stage

    def _prepend_and_align_data(self, real_data_segment_df: Optional[pd.DataFrame]) -> pd.DataFrame:
        """
        Generates synthetic data with aligned DATE_TIME, prepends to real_data_segment_df.
        Handles DATE_TIME generation, weekend skipping, and ensures chronological order.
        
        Args:
            real_data_segment_df: The real data segment DataFrame to prepend to
            
        Returns:
            pd.DataFrame: The combined DataFrame with synthetic data prepended
        """
        print("Aligning and prepending synthetic data...")
        n_samples = self.config.get("n_samples", 1000)
        datetime_col_name = self.config.get("feeder_datetime_col_in_real_data", "DATE_TIME")
        dataset_periodicity = self.config.get("dataset_periodicity", "1h")

        if self.generator_model is None:
            raise RuntimeError("Generator Keras model is not loaded. Cannot generate synthetic features.")
        if real_data_segment_df is None or real_data_segment_df.empty:
            raise ValueError("Real data segment (from x_train_file) could not be loaded or is empty.")
        if datetime_col_name not in real_data_segment_df.columns:
            raise ValueError(f"Datetime column '{datetime_col_name}' not found in real_data_segment_df.")

        try:
            real_data_segment_df[datetime_col_name] = pd.to_datetime(real_data_segment_df[datetime_col_name])
            first_real_datetime = real_data_segment_df[datetime_col_name].iloc[0]
        except Exception as e:
            raise ValueError(f"Could not parse datetimes in real data or get first datetime: {e}")

        print(f"First datetime in real data: {first_real_datetime}")

        # Generate target datetimes for synthetic data
        target_datetimes_synthetic_list: List[pd.Timestamp] = self._generate_synthetic_datetimes_before(
            n_samples, first_real_datetime, dataset_periodicity
        )
        
        if not target_datetimes_synthetic_list:
            print("Warning: No synthetic datetimes were generated. Skipping synthetic data production.")
            return real_data_segment_df

        # Convert list of Timestamps to pd.Series for SyntheticDataGenerator
        target_datetimes_synthetic_series: PandasSeries = pd.Series(target_datetimes_synthetic_list, name=datetime_col_name)

        print(f"Generated {len(target_datetimes_synthetic_list)} synthetic datetimes. First: {target_datetimes_synthetic_list[0]}, Last: {target_datetimes_synthetic_list[-1]}")

        if not hasattr(self.synthetic_generator, 'generate_features_for_datetimes'):
            raise NotImplementedError("SyntheticDataGenerator must have a 'generate_features_for_datetimes' method.")

        generated_features_df = self.synthetic_generator.generate_features_for_datetimes(
            target_datetimes=target_datetimes_synthetic_series
        )

        if generated_features_df.empty:
            print("Warning: Synthetic feature generation resulted in an empty DataFrame.")
            return real_data_segment_df

        if datetime_col_name not in generated_features_df.columns:
            generated_features_df[datetime_col_name] = target_datetimes_synthetic_series
        else:
            generated_features_df[datetime_col_name] = pd.to_datetime(generated_features_df[datetime_col_name])

        if "generator_full_feature_names_ordered" in self.config:
            ordered_cols = self.config["generator_full_feature_names_ordered"].copy()
            if datetime_col_name not in ordered_cols:
                ordered_cols.insert(0, datetime_col_name)
            elif ordered_cols[0] != datetime_col_name:
                ordered_cols.remove(datetime_col_name)
                ordered_cols.insert(0, datetime_col_name)
            
            current_synthetic_cols = set(generated_features_df.columns)
            final_ordered_cols_for_synthetic = []
            missing_in_synthetic = []
            for col in ordered_cols:
                if col in current_synthetic_cols:
                    final_ordered_cols_for_synthetic.append(col)
                elif col == datetime_col_name:
                     final_ordered_cols_for_synthetic.append(col)
                else:
                    missing_in_synthetic.append(col)
            if missing_in_synthetic:
                print(f"Warning: Canonical columns missing from generated data, filled with NaN: {missing_in_synthetic}")
                for col in missing_in_synthetic:
                    generated_features_df[col] = np.nan
            generated_features_df = generated_features_df[final_ordered_cols_for_synthetic]

            current_real_cols = set(real_data_segment_df.columns)
            final_ordered_cols_for_real = []
            missing_in_real = []
            for col in ordered_cols:
                if col in current_real_cols:
                    final_ordered_cols_for_real.append(col)
                else:
                    missing_in_real.append(col)
            if missing_in_real:
                print(f"Warning: Canonical columns missing from real data, filled with NaN: {missing_in_real}")
                for col in missing_in_real:
                    real_data_segment_df[col] = np.nan
            real_data_segment_df = real_data_segment_df[final_ordered_cols_for_real]

        print(f"Shape of synthetic data before concat: {generated_features_df.shape}")
        print(f"Shape of real data segment before concat: {real_data_segment_df.shape}")

        # CRITICAL: Prepend synthetic data BEFORE real data (no sorting by datetime)
        # Synthetic data should have earlier datetimes, so it naturally comes first
        combined_data_df = pd.concat([generated_features_df, real_data_segment_df], ignore_index=True)
        # DO NOT sort by datetime - data is already in correct chronological order
        print(f"Shape of combined data: {combined_data_df.shape}")
        print(f"✓ Synthetic data prepended successfully. First datetime: {combined_data_df[datetime_col_name].iloc[0]}")
        print(f"✓ Last synthetic datetime: {generated_features_df[datetime_col_name].iloc[-1]}")
        print(f"✓ First real datetime: {real_data_segment_df[datetime_col_name].iloc[0]}")
        
        return combined_data_df

    def _generate_synthetic_datetimes_before(self, n_samples: int, end_datetime_exclusive: pd.Timestamp, periodicity: str) -> List[pd.Timestamp]:
        """
        Generates `n_samples` datetimes before `end_datetime_exclusive`, skipping weekends.
        Returns datetimes in chronological order.
        The last generated datetime should be exactly one period before end_datetime_exclusive.
        """
        print(f"Generating {n_samples} synthetic datetimes before {end_datetime_exclusive}")
        
        time_delta_map = {
            "1h": timedelta(hours=1), "1H": timedelta(hours=1),
            "15min": timedelta(minutes=15), "15T": timedelta(minutes=15), "15m": timedelta(minutes=15),
            "1min": timedelta(minutes=1), "1T": timedelta(minutes=1), "1m": timedelta(minutes=1),
            "daily": timedelta(days=1), "1D": timedelta(days=1)
        }
        time_step = time_delta_map.get(periodicity.lower(), timedelta(hours=1))

        # Start from exactly one time_step before the end_datetime_exclusive
        last_synthetic_datetime = end_datetime_exclusive - time_step
        
        # Ensure the last synthetic datetime is a weekday
        while last_synthetic_datetime.weekday() >= 5:  # Saturday or Sunday
            last_synthetic_datetime -= timedelta(days=1)
            
        print(f"Last synthetic datetime will be: {last_synthetic_datetime}")
        
        # Generate datetimes backwards from last_synthetic_datetime
        datetimes_list = []
        current_dt = last_synthetic_datetime
        
        for i in range(n_samples):
            # Add current datetime to list
            datetimes_list.append(current_dt)
            
            # Move to previous time step
            current_dt -= time_step
            
            # Skip weekends by moving to previous Friday if needed
            while current_dt.weekday() >= 5:  # Saturday or Sunday
                current_dt -= timedelta(days=1)
                # Adjust time to end of Friday if we moved from weekend
                if time_step < timedelta(days=1):
                    current_dt = current_dt.replace(hour=23, minute=0, second=0)
        
        # Reverse to get chronological order (earliest first)
        datetimes_list.reverse()
        
        print(f"Generated {len(datetimes_list)} synthetic datetimes.")
        print(f"First synthetic datetime: {datetimes_list[0]}")
        print(f"Last synthetic datetime: {datetimes_list[-1]}")
        
        if len(datetimes_list) < n_samples:
            print(f"Warning: Could only generate {len(datetimes_list)}/{n_samples} valid weekday datetimes.")

        return datetimes_list

    def _process_real_data(self) -> Optional[pd.DataFrame]:
        """
        Process real data segment using the configured real_data_processor.
        Loads up to max_steps_train rows from x_train_file.
        """
        print("Processing real data (x_train_file) for combining...")
        real_data_file_to_load = self.config.get("x_train_file")
        if not real_data_file_to_load:
            print("ℹ️ No x_train_file specified. Cannot load real data.")
            return None
        if not os.path.exists(real_data_file_to_load):
            print(f"⚠️ Real data file (x_train_file) not found: {real_data_file_to_load}.")
            return None
            
        max_rows_to_load = self.config.get("max_steps_train")
        real_data_df: Optional[pd.DataFrame] = None
        try:
            if max_rows_to_load is not None and isinstance(max_rows_to_load, int) and max_rows_to_load > 0:
                real_data_df = pd.read_csv(real_data_file_to_load, nrows=max_rows_to_load)
            else:
                real_data_df = pd.read_csv(real_data_file_to_load)
            
            datetime_col_name = self.config.get("feeder_datetime_col_in_real_data", "DATE_TIME")
            if real_data_df is not None and datetime_col_name not in real_data_df.columns:
                print(f"⚠️ Datetime column '{datetime_col_name}' not found in {real_data_file_to_load}.")
            elif real_data_df is not None:
                real_data_df[datetime_col_name] = pd.to_datetime(real_data_df[datetime_col_name])
                print(f"✓ Real data (from x_train_file) processed. Shape: {real_data_df.shape}")
        except Exception as e:
            print(f"❌ Error loading real data from {real_data_file_to_load}: {e}")
            return None
        return real_data_df

    def _evaluate_data(self, combined_data_df: pd.DataFrame, real_data_segment_df: pd.DataFrame, evaluation_stage: str) -> None:
        """Placeholder for evaluation logic if needed beyond what evaluator_plugin does."""
        print(f"Evaluating data for stage: {evaluation_stage}...")
        if self.evaluator_plugin and hasattr(self.evaluator_plugin, 'evaluate'):
            # Assuming evaluator_plugin.evaluate takes synthetic and real data
            # The combined_data_df contains synthetic prepended to real.
            # We need to pass the synthetic part and the original real part.
            
            # Extract synthetic part:
            # This assumes generated_features_df was the synthetic part.
            # The number of synthetic samples is n_samples.
            n_synthetic_samples = self.config.get("n_samples", 0)
            synthetic_part_df = combined_data_df.head(n_synthetic_samples)
            
            # The real_data_segment_df is already the original real part.
            if not synthetic_part_df.empty and not real_data_segment_df.empty:
                try:
                    # Prepare data for evaluation: drop datetime column and convert to numpy
                    datetime_col = self.config.get("feeder_datetime_col_in_real_data", "DATE_TIME")
                    feature_cols = [col for col in synthetic_part_df.columns if col != datetime_col]
                    synthetic_array = synthetic_part_df[feature_cols].to_numpy()
                    real_array = real_data_segment_df[feature_cols].to_numpy()
                    
                    print(f"Evaluation shapes - Synthetic: {synthetic_array.shape}, Real: {real_array.shape}")
                    print(f"Feature columns for evaluation: {len(feature_cols)} features")
                    
                    # Call evaluator with synthetic and real data arrays plus feature names
                    # Note: EvaluatorPlugin.evaluate() signature is (synthetic_data, real_data_processed, feature_names, config=None)
                    evaluation_results = self.evaluator_plugin.evaluate(
                        synthetic_data=synthetic_array,
                        real_data_processed=real_array, 
                        feature_names=feature_cols,
                        config=None
                    )
                    print("✓ Evaluation plugin executed successfully.")
                    print(f"Evaluation results keys: {list(evaluation_results.keys()) if evaluation_results else 'None'}")
                except Exception as e:
                    print(f"Error during evaluation plugin execution: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print("Skipping evaluation as synthetic or real data part is empty.")
        else:
            print("Evaluator plugin or its 'evaluate' method not available.")

    def _save_outputs(self, combined_data_df: pd.DataFrame, evaluation_stage: str) -> None:
        """Save the combined (synthetic + real) data."""
        print("Saving outputs...")
        if combined_data_df.empty:
            print("Combined data is empty, nothing to save.")
            return

        output_file_path = self.config.get("generated_data_file", "generated_synthetic_data.csv")
        
        # Extract directory and filename from the full path
        output_dir = os.path.dirname(output_file_path)
        filename = os.path.basename(output_file_path)
        
        if not output_dir:
            print("Error: Could not determine output directory from 'generated_data_file' path.")
            return

        self.output_manager.save_dataframe(
            combined_data_df, 
            filename,
            output_dir
        )
        # OutputManager should print the full path upon successful save.
        print(f"Output saving initiated by OutputManager. Target directory: {output_dir}, Filename: {filename}")
