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
from typing import Dict, Any, Optional, Tuple

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
    """

    def __init__(self, config: Dict[str, Any], feeder_plugin, generator_plugin,
                 evaluator_plugin):
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

        # Initialize component modules
        self.synthetic_generator = SyntheticDataGenerator(config, feeder_plugin, generator_plugin)
        self.real_data_processor = RealDataProcessor(config)
        self.metrics_evaluator = MetricsEvaluator(config) # Pass only config
        self.output_manager = OutputManager(config)

    def execute(self) -> None:
        """
        Execute the full data generation and evaluation pipeline.
        """
        print("Starting data generation and evaluation pipeline...")
        try:
            self._validate_generation_config()
            evaluation_stage = self._process_evaluation_stage()

            synthetic_data_df = self._generate_synthetic_data()

            if synthetic_data_df.empty:
                print("Synthetic data generation resulted in an empty DataFrame. Skipping further processing.")
                return

            real_data_segment_df = self._process_real_data()
            combined_data_df = self._combine_and_save_data(synthetic_data_df, real_data_segment_df)

            if self.evaluator_plugin:
                self._evaluate_data(combined_data_df, real_data_segment_df, evaluation_stage)
            else:
                print("Evaluator plugin not available, skipping evaluation.")

            self._save_outputs(combined_data_df, evaluation_stage)

            print("✓ Data generation and evaluation pipeline completed successfully.")

        except Exception as e:
            print(f"❌ Data generation pipeline failed: {e}")
            traceback.print_exc()
            sys.exit(1)

    def _validate_generation_config(self) -> None:
        """
        Validate generation configuration parameters.
        
        Checks for required configuration keys and validates data
        file availability.
        
        Raises:
            ValueError: If required configuration is missing
        """
        # Check required configuration keys
        required_keys = ["x_train_file"] # Used by RealDataProcessor for context if needed
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Required configuration key '{key}' missing")

        # Validate training data file existence (used by RealDataProcessor)
        x_train_file_path = self.config["x_train_file"]
        if not x_train_file_path or not os.path.exists(x_train_file_path):
            # This might be optional if no real data processing is done,
            # but RealDataProcessor might expect it.
            print(f"Warning: Training data file '{x_train_file_path}' not found. Real data processing might fail if enabled.")
            # Not raising FileNotFoundError here, as per the target version.

        print(f"✓ Generation configuration validated")
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
        
        # Update output paths based on evaluation stage
        # self.output_manager.update_paths_for_stage(evaluation_stage) # This line was removed as per previous steps
        return evaluation_stage

    def _generate_synthetic_data(self) -> pd.DataFrame:
        """
        Generate synthetic data using the configured synthetic_generator.
        
        Returns:
            pd.DataFrame: The generated synthetic data.
        """
        print("Generating synthetic data...")
        if not hasattr(self, 'synthetic_generator') or self.synthetic_generator is None:
            raise RuntimeError("SyntheticDataGenerator is not initialized.")
        
        # Get the number of samples to generate from config
        n_samples_to_generate = self.config.get("num_synthetic_samples_to_generate", 1000)
        if not isinstance(n_samples_to_generate, int) or n_samples_to_generate <= 0:
            print(f"⚠️ Warning: Invalid 'num_synthetic_samples_to_generate' ({n_samples_to_generate}) in config. Defaulting to 1000.")
            n_samples_to_generate = 1000
        
        # Note: The `initial_window` argument for `self.synthetic_generator.generate()`
        # is optional and defaults to None. If initial window data is required,
        # it would need to be prepared and passed here.
        # For now, we are only addressing the missing `n_samples` argument.
        synthetic_data = self.synthetic_generator.generate(n_samples=n_samples_to_generate)
        if synthetic_data.empty:
            print("⚠️ Synthetic data generation resulted in an empty DataFrame.")
        else:
            print(f"✓ Synthetic data generated with shape: {synthetic_data.shape}")
        return synthetic_data

    def _process_real_data(self) -> Optional[pd.DataFrame]:
        """
        Process real data segment using the configured real_data_processor.
        
        Returns:
            Optional[pd.DataFrame]: The processed real data, or None if not configured.
        """
        print("Processing real data segment...")
        if not hasattr(self, 'real_data_processor') or self.real_data_processor is None:
            print("ℹ️ RealDataProcessor is not initialized. Skipping real data processing.")
            return None
            
        real_data_file = self.config.get("real_data_file_for_generation")
        if not real_data_file:
            print("ℹ️ No real_data_file_for_generation specified. Skipping real data processing.")
            return None

        if not os.path.exists(real_data_file):
            print(f"⚠️ Real data file for generation not found: {real_data_file}. Skipping.")
            return None
            
        real_data = self.real_data_processor.process(real_data_file)
        if real_data is not None and not real_data.empty:
            print(f"✓ Real data processed with shape: {real_data.shape}")
        elif real_data is not None and real_data.empty:
            print("⚠️ Real data processing resulted in an empty DataFrame.")
        else:
            print("ℹ️ No real data processed.")
        return real_data

    def _combine_and_save_data(self, synthetic_data: pd.DataFrame, real_data: Optional[pd.DataFrame]) -> pd.DataFrame:
        """
        Combine synthetic data with real data if available, and save the outputs.
        
        Args:
            synthetic_data: The generated synthetic data.
            real_data: The processed real data (optional).
            
        Returns:
            pd.DataFrame: The combined data (or just synthetic if real_data is None).
        """
        print("Combining and saving data...")
        if not hasattr(self, 'output_manager') or self.output_manager is None:
            raise RuntimeError("OutputManager is not initialized.")

        combined_data = synthetic_data
        if real_data is not None and not real_data.empty:
            # Example combination: append real data to synthetic data
            # Adjust based on actual requirements (e.g., specific merging strategy)
            print(f"Combining synthetic data (shape: {synthetic_data.shape}) with real data (shape: {real_data.shape})")
            combined_data = pd.concat([synthetic_data, real_data], ignore_index=True) # Basic concatenation
            self.output_manager.save_dataframe(real_data, "processed_real_data.csv")
            print(f"✓ Real data segment saved.")

        if not combined_data.empty:
            self.output_manager.save_dataframe(combined_data, "combined_generated_data.csv")
            self.output_manager.save_dataframe(synthetic_data, "synthetic_generated_data.csv")
            print(f"✓ Combined data saved with shape: {combined_data.shape}")
            print(f"✓ Synthetic data saved separately with shape: {synthetic_data.shape}")
        else:
            print("⚠️ Combined data is empty. Nothing to save.")
            
        return combined_data

    def _perform_evaluation(self, synthetic_data: pd.DataFrame, evaluation_stage: str) -> None:
        """
        Perform evaluation of the synthetic data against real data if configured.
        
        Args:
            synthetic_data: The generated synthetic data.
            evaluation_stage: Identifier for the current evaluation stage.
        """
        print(f"Performing evaluation for stage: {evaluation_stage}...")
        if not hasattr(self, 'metrics_evaluator') or self.metrics_evaluator is None:
            print("ℹ️ MetricsEvaluator is not initialized. Skipping evaluation.")
            return

        if synthetic_data.empty:
            print("⚠️ Synthetic data is empty. Skipping evaluation.")
            return
            
        # Assuming real_data_for_evaluation is specified in config for comparison
        real_data_eval_file = self.config.get("real_data_file_for_evaluation") 
        if not real_data_eval_file or not os.path.exists(real_data_eval_file):
            print(f"⚠️ Real data for evaluation not found or not specified ('{real_data_eval_file}'). Evaluation might be limited or skipped.")
            # Decide if evaluation can proceed without real data or with a subset of metrics
            # For now, let's assume the evaluator handles this.
            
        try:
            evaluation_results = self.metrics_evaluator.evaluate(
                synthetic_data=synthetic_data,
                real_data_path=real_data_eval_file, # Pass path, evaluator should load
                stage_name=evaluation_stage
            )
            if evaluation_results:
                self.output_manager.save_results(evaluation_results, f"evaluation_results_{evaluation_stage}.json")
                print(f"✓ Evaluation results saved for stage: {evaluation_stage}")
            else:
                print("⚠️ Evaluation did not produce any results.")
        except Exception as e:
            print(f"❌ Error during evaluation: {e}")
            traceback.print_exc()

    def _evaluate_data(self, combined_data_df: pd.DataFrame, real_data_segment_df: Optional[pd.DataFrame], evaluation_stage: str) -> None:
        """
        Evaluate the generated data using the evaluator plugin.
        
        Args:
            combined_data_df: Combined synthetic and real data
            real_data_segment_df: Real data segment for comparison
            evaluation_stage: Identifier for the current evaluation stage
        """
        try:
            print(f"Evaluating data for stage: {evaluation_stage}...")
            
            if not self.evaluator_plugin:
                print("⚠️ No evaluator plugin available. Skipping evaluation.")
                return
                
            # Use the existing _perform_evaluation method
            self._perform_evaluation(combined_data_df, evaluation_stage)
            
        except Exception as e:
            print(f"❌ Error during data evaluation: {e}")
            traceback.print_exc()

    def _save_outputs(self, combined_data_df: pd.DataFrame, evaluation_stage: str) -> None:
        """
        Save the generated outputs to files.
        
        Args:
            combined_data_df: Combined data to save
            evaluation_stage: Current evaluation stage
        """
        try:
            print("Saving generated outputs...")
            
            # Save combined data
            output_filename = f"generated_data_{evaluation_stage}.csv"
            saved_path = self.output_manager.save_dataframe(combined_data_df, output_filename)
            print(f"✓ Generated data saved to: {saved_path}")
            
        except Exception as e:
            print(f"❌ Error saving outputs: {e}")
            traceback.print_exc()

# Ensure the class definition ends here if it's the end of the file.
# If there's more code after this class, this comment is not needed.
