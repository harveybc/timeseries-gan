#!/usr/bin/env python3
"""
train_pipeline.py

Training pipeline for TimeSeries-GAN models.
Handles GAN training workflow including data loading, model training,
and model persistence.

This module encapsulates all training-specific logic in a focused,
reusable component that follows single responsibility principle.

Author: TimeSeries-GAN Team
"""

import os
import sys
import logging
import pandas as pd
import traceback
from typing import Dict, Any


class TrainPipeline:
    """
    Pipeline for training GAN models using provided training data.
    
    This pipeline coordinates the complete training workflow:
    - Validates training data availability
    - Loads and preprocesses training data 
    - Executes GAN training using trainer plugin
    - Handles training result persistence
    
    Attributes:
        config: Configuration dictionary containing training parameters
        trainer_plugin: Plugin instance responsible for GAN training coordination
    """
    
    def __init__(self, config: Dict[str, Any], trainer_plugin):
        """
        Initialize training pipeline with configuration and trainer plugin.
        
        Args:
            config: Configuration dictionary containing training parameters
            trainer_plugin: Plugin instance for GAN training coordination
        """
        self.config = config
        self.trainer_plugin = trainer_plugin
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
    def execute(self) -> None:
        """
        Execute the complete training pipeline.
        
        Performs the following steps:
        1. Validate training configuration and data availability
        2. Load and prepare training data
        3. Execute GAN training using trainer plugin
        4. Handle post-training tasks (model saving, logging)
        
        Raises:
            SystemExit: If training data is unavailable or training fails
        """
        print("Starting GAN training pipeline...")
        
        try:
            # Step 1: Validate training configuration
            self._validate_training_config()
            
            # Step 2: Load training data
            training_data = self._load_training_data()
            
            # Step 3: Execute GAN training
            self._execute_training(training_data)
            
            # Step 4: Handle post-training tasks
            self._handle_post_training()
            
            print("✓ GAN training pipeline completed successfully")
            
        except Exception as e:
            print(f"❌ GAN training pipeline failed: {e}")
            self.logger.error(f"Training pipeline failed: {e}")
            self.logger.error(traceback.format_exc())
            raise
    
    def _validate_training_config(self) -> None:
        """
        Validate training configuration parameters.
        
        Checks for required configuration keys and validates training
        data file availability.
        
        Raises:
            ValueError: If required configuration is missing
            FileNotFoundError: If training data file is not found
        """
        # Check required configuration keys
        required_keys = ["x_train_file", "gan_epochs", "gan_batch_size"]
        for key in required_keys:
            if key not in self.config or self.config[key] is None:
                raise ValueError(f"Required configuration key '{key}' is missing or None")
        
        # Validate training data file existence
        x_train_file_path = self.config["x_train_file"]
        if not x_train_file_path or not os.path.exists(x_train_file_path):
            raise FileNotFoundError(f"Training data file not found: {x_train_file_path}")
        
        print(f"✓ Training configuration validated")
        print(f"✓ Training data file verified: {x_train_file_path}")
    
    def _load_training_data(self) -> pd.DataFrame:
        """
        Load and prepare training data from configured file.
        
        Returns:
            pd.DataFrame: Loaded training data ready for GAN training
            
        Raises:
            ValueError: If training data cannot be loaded or is invalid
        """
        x_train_file_path = self.config["x_train_file"]
        
        try:
            print(f"Loading training data from: {x_train_file_path}")
            
            # Load CSV file
            if x_train_file_path.endswith('.csv'):
                training_data = pd.read_csv(x_train_file_path)
            else:
                raise ValueError(f"Unsupported training data file format: {x_train_file_path}")
            
            # Basic validation
            if training_data.empty:
                raise ValueError("Training data is empty")
            
            if len(training_data.columns) == 0:
                raise ValueError("Training data has no columns")
            
            print(f"✓ Training data loaded successfully. Shape: {training_data.shape}")
            
            return training_data
            
        except Exception as e:
            raise ValueError(f"Failed to load training data from {x_train_file_path}: {e}")
    
    def _execute_training(self, training_data: pd.DataFrame) -> None:
        """
        Execute GAN training using the trainer plugin.
        
        Args:
            training_data: Loaded and validated training data
            
        Raises:
            RuntimeError: If training fails
        """
        try:
            print("Starting GAN model training...")
            
            # Extract training parameters
            epochs = self.config.get("gan_epochs", 1000)
            batch_size = self.config.get("gan_batch_size", 32)
            
            print(f"Training parameters - Epochs: {epochs}, Batch size: {batch_size}")
            
            # Check if trainer plugin is available
            if self.trainer_plugin is None:
                raise ValueError("Trainer plugin not available")
            
            # Check if trainer plugin has train method
            if not hasattr(self.trainer_plugin, 'train'):
                raise ValueError("Trainer plugin does not have 'train' method")
            
            # Execute training via trainer plugin
            # Pass the training data and configuration to the trainer
            self.trainer_plugin.train(
                training_data=training_data,
                epochs=epochs,
                batch_size=batch_size,
                config=self.config
            )
            
            print("✓ GAN training completed successfully")
            
        except Exception as e:
            print(f"❌ GAN training failed: {e}")
            self.logger.error(f"Training execution failed: {e}")
            self.logger.error(traceback.format_exc())
            raise RuntimeError(f"GAN training execution failed: {e}")
    
    def _handle_post_training(self) -> None:
        """
        Handle post-training tasks such as model saving and logging.
        """
        try:
            print("Handling post-training tasks...")
            
            # Save generator model if path specified
            save_generator_path = self.config.get("save_generator_sequential_model_file")
            if save_generator_path and hasattr(self.trainer_plugin, 'save_generator_model'):
                try:
                    self.trainer_plugin.save_generator_model(save_generator_path)
                    print(f"✓ Generator model saved to: {save_generator_path}")
                except Exception as e:
                    print(f"⚠ Failed to save generator model: {e}")
            
            # Save discriminator model if path specified
            save_discriminator_path = self.config.get("save_discriminator_sequential_model_file")
            if save_discriminator_path and hasattr(self.trainer_plugin, 'save_discriminator_model'):
                try:
                    self.trainer_plugin.save_discriminator_model(save_discriminator_path)
                    print(f"✓ Discriminator model saved to: {save_discriminator_path}")
                except Exception as e:
                    print(f"⚠ Failed to save discriminator model: {e}")
            
            # Save training logs if specified
            log_file = self.config.get("gan_loss_plot_file")
            if log_file and hasattr(self.trainer_plugin, 'save_training_logs'):
                try:
                    self.trainer_plugin.save_training_logs(log_file)
                    print(f"✓ Training logs saved to: {log_file}")
                except Exception as e:
                    print(f"⚠ Failed to save training logs: {e}")
            
            print("✓ Post-training tasks completed")
            
        except Exception as e:
            print(f"⚠ Warning: Post-training tasks failed: {e}")
            self.logger.warning(f"Post-training tasks failed: {e}")
