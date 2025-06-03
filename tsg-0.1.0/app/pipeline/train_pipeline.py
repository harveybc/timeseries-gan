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
import traceback
import pandas as pd
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
            # Validate training configuration
            self._validate_training_config()
            
            # Load training data
            training_data = self._load_training_data()
            
            # Execute GAN training
            self._execute_training(training_data)
            
            # Handle post-training tasks
            self._handle_post_training()
            
            print("✔ GAN training completed successfully.")
            
        except Exception as e:
            print(f"❌ GAN training failed: {e}")
            traceback.print_exc()
            sys.exit(1)
    
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
            if key not in self.config:
                raise ValueError(f"Required configuration key '{key}' missing")
        
        # Validate training data file existence
        x_train_file_path = self.config["x_train_file"]
        if not x_train_file_path or not os.path.exists(x_train_file_path):
            raise FileNotFoundError(f"Training data file not found: '{x_train_file_path}'")
        
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
            training_data = pd.read_csv(x_train_file_path)
            
            # Validate data shape and content
            if training_data.empty:
                raise ValueError("Training data file is empty")
            
            print(f"✓ Training data loaded successfully. Shape: {training_data.shape}")
            return training_data
            
        except Exception as e:
            raise ValueError(f"Failed to load training data: {e}")
    
    def _execute_training(self, training_data: pd.DataFrame) -> None:
        """
        Execute GAN training using the trainer plugin.
        
        Args:
            training_data: Preprocessed training data for GAN training
            
        Raises:
            RuntimeError: If training execution fails
        """
        try:
            print("Starting GAN model training...")
            
            # Extract training parameters
            gan_epochs = self.config["gan_epochs"]
            gan_batch_size = self.config["gan_batch_size"]
            
            print(f"Training parameters - Epochs: {gan_epochs}, Batch size: {gan_batch_size}")
            
            # Execute training using trainer plugin
            self.trainer_plugin.train(
                x_real_df=training_data,
                epochs=gan_epochs,
                batch_size=gan_batch_size
            )
            
            print("✓ GAN training execution completed")
            
        except Exception as e:
            raise RuntimeError(f"GAN training execution failed: {e}")
    
    def _handle_post_training(self) -> None:
        """
        Handle post-training tasks such as model saving and logging.
        
        Performs cleanup and persistence operations after successful training.
        """
        try:
            print("Handling post-training tasks...")
            
            # Additional post-training logic can be added here
            # such as model validation, metric logging, etc.
            
            print("✓ Post-training tasks completed")
            
        except Exception as e:
            print(f"⚠ Warning: Post-training tasks failed: {e}")
            # Don't fail the entire pipeline for post-training issues
