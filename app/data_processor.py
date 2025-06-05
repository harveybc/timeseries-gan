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
import sys
import traceback
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
                 **kwargs) -> None: # Use **kwargs to catch any other plugins not explicitly listed
    """
    Run the appropriate pipeline based on the operation mode.
    Passes necessary plugin instances to the specific pipeline constructors.
    """
    operation_mode = config.get("operation_mode", "generate") # Default to generate if not specified
    logger.info(f"DataProcessor: Dispatching to pipeline for operation mode: {operation_mode}")

    if operation_mode == "train":
        if not trainer_plugin:
            logger.error("Trainer plugin is required for 'train' mode but was not provided.")
            raise ValueError("Trainer plugin instance not available for training mode.")
        # TrainPipeline might also need feeder_plugin directly for data loading.
        # The trainer_plugin itself should have references to generator, discriminator, feeder.
        pipeline = TrainPipeline(config=config, 
                                 trainer_plugin=trainer_plugin,
                                 feeder_plugin=feeder_plugin) # Pass feeder if TrainPipeline uses it directly
    elif operation_mode == "generate":
        if not generator_plugin or not feeder_plugin:
            logger.error("Generator and Feeder plugins are required for 'generate' mode.")
            raise ValueError("Generator or Feeder plugin instance not available for generate mode.")
        pipeline = GeneratePipeline(config=config,
                                    feeder_plugin=feeder_plugin,
                                    generator_plugin=generator_plugin,
                                    evaluator_plugin=evaluator_plugin,
                                    preprocessor_plugin=preprocessor_plugin)
    elif operation_mode == "optimize":
        if not optimizer_plugin or not feeder_plugin or not generator_plugin or not evaluator_plugin:
            logger.error("Optimizer, Feeder, Generator, and Evaluator plugins are required for 'optimize' mode.")
            raise ValueError("One or more required plugin instances not available for optimize mode.")
        pipeline = OptimizePipeline(config=config,
                                    optimizer_plugin=optimizer_plugin,
                                    feeder_plugin=feeder_plugin,
                                    generator_plugin=generator_plugin,
                                    evaluator_plugin=evaluator_plugin)
    else:
        logger.error(f"Unsupported operation mode: {operation_mode}")
        raise ValueError(f"Unsupported operation mode specified: {operation_mode}")

    logger.info(f"Executing {operation_mode} pipeline...")
    pipeline.execute()
    logger.info(f"Finished executing {operation_mode} pipeline.")


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
