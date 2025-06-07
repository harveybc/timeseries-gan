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
        self.output_manager = OutputManager(config)
        
        if evaluator_plugin:
            self.metrics_evaluator = MetricsEvaluator(config, evaluator_plugin)
    
    def execute(self) -> None:
        """
        Execute the complete data generation and evaluation pipeline.
        
        Performs the following steps:
        1. Validate generation configuration and data availability
        2. Process evaluation stage and update output paths
        3. Generate synthetic data using feeder and generator plugins
        4. Load and process real data segment if configured
        5. Combine synthetic and real data for output
        6. Perform evaluation if evaluator plugin is available
        7. Save results and handle output management
        
        Raises:
            SystemExit: If required data is unavailable or generation fails
        """
        print("Starting data generation and evaluation pipeline...")
        
        try:
            # Validate generation configuration
            self._validate_generation_config()
            
            # Process evaluation stage configuration
            evaluation_stage = self._process_evaluation_stage()
            
            # Generate synthetic data
            synthetic_data = self._generate_synthetic_data()
            
            # Process real data segment
            real_data = self._process_real_data()
            
            # Combine and save data
            combined_data = self._combine_and_save_data(synthetic_data, real_data)
            
            # Perform evaluation if available
            if self.evaluator_plugin and not synthetic_data.empty:
                self._perform_evaluation(synthetic_data, evaluation_stage)
            
            print("✔ Data generation and evaluation completed successfully.")
            
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
            FileNotFoundError: If required data files are not found
        """
        # Check required configuration keys
        required_keys = ["x_train_file"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Required configuration key '{key}' missing")
        
        # Validate training data file existence
        x_train_file_path = self.config["x_train_file"]
        if not x_train_file_path or not os.path.exists(x_train_file_path):
            raise FileNotFoundError(f"Training data file not found: '{x_train_file_path}'")
        
        print(f"✓ Generation configuration validated")
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
        self.output_manager.update_paths_for_stage(evaluation_stage, self.generator_plugin)
        
        return evaluation_stage
    
    def _generate_synthetic_data(self) -> pd.DataFrame:
        """
        Generate synthetic data using feeder and generator plugins.
        
        Returns:
            pd.DataFrame: Generated synthetic data
            
        Raises:
            RuntimeError: If synthetic data generation fails
        """
        try:
            print("Generating synthetic data...")
            
            # Extract generation parameters
            n_samples_synthetic = self.config.get("n_samples", 
                                                self.config.get("num_synthetic_samples_to_generate", 0))
            
            if n_samples_synthetic <= 0:
                print("No synthetic samples requested, skipping generation")
                return pd.DataFrame()
            
            # Generate synthetic data
            synthetic_data = self.synthetic_generator.generate(
                n_samples=n_samples_synthetic,
                initial_window=None # GeneratorPlugin handles its own initial context
            )
            
            print(f"✓ Synthetic data generated. Shape: {synthetic_data.shape}")
            return synthetic_data
            
        except Exception as e:
            raise RuntimeError(f"Synthetic data generation failed: {e}")
    
    def _process_real_data(self) -> pd.DataFrame:
        """
        Process real data segment for integration with synthetic data.
        
        Returns:
            pd.DataFrame: Processed real data segment
        """
        try:
            print("Processing real data segment...")
            
            max_steps_train_real = self.config.get("max_steps_train", 0)
            
            if max_steps_train_real <= 0:
                print("No real data segment requested, skipping")
                return pd.DataFrame()
            
            # Process real data using real data processor
            real_data = self.real_data_processor.process(max_steps_train_real)
            
            print(f"✓ Real data processed. Shape: {real_data.shape}")
            return real_data
            
        except Exception as e:
            print(f"⚠ Warning: Real data processing failed: {e}")
            return pd.DataFrame()
    
    def _combine_and_save_data(self, synthetic_data: pd.DataFrame, 
                              real_data: pd.DataFrame) -> pd.DataFrame:
        """
        Combine synthetic and real data and save to output file.
        
        Args:
            synthetic_data: Generated synthetic data
            real_data: Processed real data segment
            
        Returns:
            pd.DataFrame: Combined data
        """
        try:
            print("Combining and saving data...")
            
            # Combine data using output manager
            combined_data = self.output_manager.combine_data(synthetic_data, real_data)
            
            # Save combined data
            self.output_manager.save_data(combined_data)
            
            print(f"✓ Data combined and saved. Shape: {combined_data.shape}")
            return combined_data
            
        except Exception as e:
            print(f"⚠ Warning: Data saving failed: {e}")
            return synthetic_data if not synthetic_data.empty else real_data
    
    def _perform_evaluation(self, synthetic_data: pd.DataFrame, evaluation_stage: str) -> None:
        """
        Perform evaluation of synthetic data against real data.
        
        Args:
            synthetic_data: Generated synthetic data for evaluation
            evaluation_stage: Current evaluation stage identifier
        """
        try:
            print(f"Performing evaluation for stage: {evaluation_stage}")
            
            # Execute evaluation using metrics evaluator
            evaluation_results = self.metrics_evaluator.evaluate(
                synthetic_data=synthetic_data,
                evaluation_stage=evaluation_stage
            )
            
            # Save evaluation results
            self.metrics_evaluator.save_results(evaluation_results, evaluation_stage)
            
            print("✓ Evaluation completed successfully")
            
        except Exception as e:
            print(f"⚠ Warning: Evaluation failed: {e}")
            # Don't fail the pipeline for evaluation issues
