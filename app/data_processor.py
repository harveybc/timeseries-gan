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

import logging
from typing import Dict, Any, Optional

# Import specific pipeline classes
from app.pipeline.train_pipeline import TrainPipeline
from app.pipeline.generate_pipeline import GeneratePipeline
from app.pipeline.optimize_pipeline import OptimizePipeline

logger = logging.getLogger(__name__)

def run_pipeline(config: Dict[str, Any],
                 feeder_plugin: Optional[Any] = None,
                 generator_plugin: Optional[Any] = None,
                 discriminator_plugin: Optional[Any] = None,
                 evaluator_plugin: Optional[Any] = None,
                 optimizer_plugin: Optional[Any] = None,
                 trainer_plugin: Optional[Any] = None,
                 preprocessor_plugin: Optional[Any] = None,
                 **kwargs) -> None:
    """
    Run the appropriate pipeline based on the operation mode.
    Passes necessary plugin instances to the specific pipeline constructors.
    Implements "train then generate" workflow if operation_mode is 'train_then_generate'.
    """
    operation_mode = config.get("operation_mode", "generate")  # Default to generate if not specified
    logger.info(f"DataProcessor: Dispatching to pipeline for operation mode: {operation_mode}")

    if operation_mode == "train":
        if not trainer_plugin:
            raise ValueError("Trainer plugin is required for train operation mode")
        if not feeder_plugin:
            raise ValueError("Feeder plugin is required for train operation mode")
        if not generator_plugin:
            raise ValueError("Generator plugin is required for train operation mode")
        if not discriminator_plugin:
            raise ValueError("Discriminator plugin is required for train operation mode")
        train_pipeline = TrainPipeline(config, trainer_plugin, feeder_plugin, generator_plugin, discriminator_plugin)
        logger.info("Executing train pipeline...")
        train_pipeline.execute()
        logger.info("Finished executing train pipeline.")
        
    elif operation_mode == "generate":
        if not generator_plugin or not feeder_plugin:
            raise ValueError("Generator and Feeder plugins are required for generate operation mode")
        generate_pipeline = GeneratePipeline(config, feeder_plugin, generator_plugin, evaluator_plugin)
        logger.info("Executing generate pipeline...")
        generate_pipeline.execute()
        logger.info("Finished executing generate pipeline.")

    elif operation_mode == "train_then_generate":
        # Train first
        if not trainer_plugin:
            raise ValueError("Trainer plugin is required for train operation mode in 'train_then_generate'")
        if not feeder_plugin:
            raise ValueError("Feeder plugin is required for train operation mode in 'train_then_generate'")
        if not generator_plugin:
            raise ValueError("Generator plugin is required for train operation mode in 'train_then_generate'")
        if not discriminator_plugin:
            raise ValueError("Discriminator plugin is required for train operation mode in 'train_then_generate'")
        train_pipeline = TrainPipeline(config, trainer_plugin, feeder_plugin, generator_plugin, discriminator_plugin)
        logger.info("Executing train pipeline (part of train_then_generate)...")
        train_pipeline.execute()
        logger.info("Finished executing train pipeline (part of train_then_generate).")

        # Then generate
        if not generator_plugin or not feeder_plugin:
            raise ValueError("Generator and Feeder plugins are required for generate operation mode in 'train_then_generate'")
        
        # Potentially update config for generation if needed, e.g., load trained model paths
        # This might involve the trainer saving model paths to the config or a shared context
        # For now, assume generator_plugin is aware of trained models or config points to them.

        generate_pipeline = GeneratePipeline(config, feeder_plugin, generator_plugin, evaluator_plugin)
        logger.info("Executing generate pipeline (part of train_then_generate)...")
        generate_pipeline.execute()
        logger.info("Finished executing generate pipeline (part of train_then_generate).")
        
    elif operation_mode == "optimize":
        if not optimizer_plugin:
            raise ValueError("Optimizer plugin is required for optimize operation mode")
        optimize_pipeline = OptimizePipeline(config, optimizer_plugin)
        logger.info(f"Executing {operation_mode} pipeline...")
        optimize_pipeline.execute()
        logger.info(f"Finished executing {operation_mode} pipeline.")
        
    else:
        raise ValueError(f"Unknown operation mode: {operation_mode}")


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
        discriminator_plugin=discriminator,
        evaluator_plugin=evaluator,
        optimizer_plugin=optimizer,
        preprocessor_plugin=None,
        trainer_plugin=trainer
    )
