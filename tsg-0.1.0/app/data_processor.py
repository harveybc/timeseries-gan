#!/usr/bin/env python3
"""
data_processor.py

Main orchestrator for TimeSeries-GAN data processing pipeline.
Handles operation mode dispatching and high-level workflow coordination.

This module follows extreme separation of concerns by delegating specific
operations to specialized modules while maintaining a clean, simple interface.

Operation Modes:
    - train: Train GAN models using provided training data
    - optimize: Perform hyperparameter optimization using genetic algorithms  
    - generate: Generate synthetic data and optionally evaluate against real data

Author: TimeSeries-GAN Team
"""

import sys
import traceback
from typing import Dict, Any, Optional

from app.pipeline.train_pipeline import TrainPipeline
from app.pipeline.optimize_pipeline import OptimizePipeline  
from app.pipeline.generate_pipeline import GeneratePipeline
from app.utils.latent_shape_inference import infer_and_set_latent_shape


def run_pipeline(config: Dict[str, Any], feeder_plugin=None, generator_plugin=None, 
                evaluator_plugin=None, optimizer_plugin=None, preprocessor_plugin=None, 
                trainer_plugin=None) -> None:
    """
    Main entry point for TimeSeries-GAN data processing pipeline.
    
    Orchestrates the complete workflow based on the configured operation mode:
    - Handles latent shape inference for generator/feeder compatibility
    - Dispatches to appropriate operation pipeline (train/optimize/generate)
    - Provides unified error handling and logging
    
    Args:
        config: Configuration dictionary containing all pipeline parameters
        feeder_plugin: Plugin for generating initial noise and conditional inputs
        generator_plugin: Plugin wrapping the VAE decoder for data generation
        evaluator_plugin: Plugin for computing evaluation metrics on generated data
        optimizer_plugin: Plugin for hyperparameter optimization using genetic algorithms
        preprocessor_plugin: Plugin for optional data preprocessing transformations
        trainer_plugin: Plugin coordinating GAN training (Z-Generator + Discriminator)
        
    Raises:
        SystemExit: On pipeline execution failure or invalid operation mode
    """
    print("Starting TimeSeries-GAN data processing pipeline...")
    
    # Extract operation mode from configuration
    operation_mode = config.get("operation_mode", "generate").lower()
    print(f"▶ Operation mode: {operation_mode}")
    
    # Validate operation mode
    valid_modes = ["train", "optimize", "generate"]
    if operation_mode not in valid_modes:
        print(f"❌ Invalid operation mode '{operation_mode}'. Valid modes: {valid_modes}")
        sys.exit(1)
    
    try:
        # Perform latent shape inference if generator and feeder plugins are available
        if generator_plugin and feeder_plugin:
            print("Performing latent shape inference...")
            infer_and_set_latent_shape(config, generator_plugin, feeder_plugin, trainer_plugin)
        
        # Dispatch to appropriate operation pipeline
        if operation_mode == "train":
            _execute_train_mode(config, trainer_plugin)
            
        elif operation_mode == "optimize":
            _execute_optimize_mode(config, optimizer_plugin, feeder_plugin, 
                                 generator_plugin, evaluator_plugin)
            
        elif operation_mode == "generate":
            _execute_generate_mode(config, feeder_plugin, generator_plugin, 
                                 evaluator_plugin, preprocessor_plugin)
        
        print("✔ Pipeline execution completed successfully.")
        
    except Exception as e:
        print(f"❌ Pipeline execution failed: {e}")
        traceback.print_exc()
        sys.exit(1)


def _execute_train_mode(config: Dict[str, Any], trainer_plugin) -> None:
    """
    Execute training mode pipeline.
    
    Args:
        config: Configuration dictionary
        trainer_plugin: Plugin instance for GAN training coordination
        
    Raises:
        SystemExit: If trainer plugin is unavailable or training fails
    """
    if not trainer_plugin:
        print("❌ Trainer plugin not available for training mode.")
        sys.exit(1)
        
    print("▶ Executing training pipeline...")
    pipeline = TrainPipeline(config, trainer_plugin)
    pipeline.execute()


def _execute_optimize_mode(config: Dict[str, Any], optimizer_plugin, feeder_plugin, 
                          generator_plugin, evaluator_plugin) -> None:
    """
    Execute hyperparameter optimization mode pipeline.
    
    Args:
        config: Configuration dictionary
        optimizer_plugin: Plugin instance for genetic algorithm optimization
        feeder_plugin: Plugin instance for noise generation
        generator_plugin: Plugin instance for data generation
        evaluator_plugin: Plugin instance for fitness evaluation
        
    Raises:
        SystemExit: If required plugins are unavailable or optimization fails
    """
    if not optimizer_plugin:
        print("❌ Optimizer plugin not available for optimization mode.")
        sys.exit(1)
        
    print("▶ Executing optimization pipeline...")
    pipeline = OptimizePipeline(config, optimizer_plugin, feeder_plugin, 
                               generator_plugin, evaluator_plugin)
    pipeline.execute()


def _execute_generate_mode(config: Dict[str, Any], feeder_plugin, generator_plugin, 
                          evaluator_plugin, preprocessor_plugin) -> None:
    """
    Execute data generation and evaluation mode pipeline.
    
    Args:
        config: Configuration dictionary
        feeder_plugin: Plugin instance for noise generation
        generator_plugin: Plugin instance for data generation
        evaluator_plugin: Plugin instance for evaluation metrics
        preprocessor_plugin: Plugin instance for data preprocessing
        
    Raises:
        SystemExit: If required plugins are unavailable or generation fails
    """
    if not feeder_plugin or not generator_plugin:
        print("❌ Feeder and Generator plugins required for generation mode.")
        sys.exit(1)
        
    print("▶ Executing generation pipeline...")
    pipeline = GeneratePipeline(config, feeder_plugin, generator_plugin, 
                               evaluator_plugin, preprocessor_plugin)
    pipeline.execute()


# Legacy function maintained for backward compatibility
def run_pipeline_legacy(config: Dict[str, Any], feeder, z_generator, generator, 
                       discriminator, trainer, evaluator, optimizer) -> None:
    """
    Legacy pipeline interface for backward compatibility.
    
    Maps legacy parameters to new plugin-based interface and delegates
    to the main run_pipeline function.
    
    Args:
        config: Configuration dictionary
        feeder: Legacy feeder instance
        z_generator: Legacy Z-generator instance  
        generator: Legacy generator instance
        discriminator: Legacy discriminator instance
        trainer: Legacy trainer instance
        evaluator: Legacy evaluator instance
        optimizer: Legacy optimizer instance
        
    Note:
        This function is deprecated and will be removed in future versions.
        Use run_pipeline() with proper plugin instances instead.
    """
    print("⚠ WARNING: Using legacy pipeline interface. Consider migrating to plugin-based interface.")
    
    # Map legacy parameters to plugin interface
    run_pipeline(
        config=config,
        feeder_plugin=feeder,
        generator_plugin=generator,
        evaluator_plugin=evaluator,
        optimizer_plugin=optimizer,
        preprocessor_plugin=None,  # Not available in legacy interface
        trainer_plugin=trainer
    )
