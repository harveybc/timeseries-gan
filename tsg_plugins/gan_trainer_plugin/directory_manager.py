#!/usr/bin/env python3
"""
directory_manager.py

Directory Manager module for GAN Trainer Plugin.
Handles creation and management of output directories for models, plots, and metrics.

Part of the extreme separation of concerns approach.
Each module is focused and under 200 lines.

Author: TimeSeries-GAN Team
"""

import os
import logging
from typing import Dict, Any, Tuple


class DirectoryManager:
    """
    Manages output directories for GAN training results.
    
    Handles creation and organization of directories for:
    - Saved models (generators, discriminators, GAN models)
    - Training plots and visualizations
    - Training metrics and logs
    """
    
    def __init__(self, params: Dict[str, Any], logger: logging.Logger):
        """
        Initialize directory manager with parameters and logger.
        
        Args:
            params: Plugin parameters dictionary
            logger: Logger instance for this module
        """
        self.params = params
        self.logger = logger
        
        # Initialize directory paths
        self.results_base_dir = None
        self.models_dir = None
        self.plots_dir = None
        self.metrics_dir = None
        
        self._setup_directories()
    
    def _setup_directories(self):
        """Setup all output directories from parameters."""
        # Extract base directory and subdirectory names
        self.results_base_dir = self.params.get("results_base_dir", "examples/results/gan_training")
        model_subdir = self.params.get("save_model_dir", "models")
        plot_subdir = self.params.get("save_plot_dir", "plots")
        metrics_subdir = self.params.get("save_metrics_dir", "metrics")
        
        # Create full directory paths
        self.models_dir = os.path.join(self.results_base_dir, model_subdir)
        self.plots_dir = os.path.join(self.results_base_dir, plot_subdir)
        self.metrics_dir = os.path.join(self.results_base_dir, metrics_subdir)
        
        # Create directories
        self._create_directories()
        
        self.logger.info(f"Output directories setup under {self.results_base_dir}")
    
    def _create_directories(self):
        """Create all required directories if they don't exist."""
        directories = [
            self.results_base_dir,
            self.models_dir,
            self.plots_dir,
            self.metrics_dir
        ]
        
        for directory in directories:
            try:
                os.makedirs(directory, exist_ok=True)
                self.logger.debug(f"Created/verified directory: {directory}")
            except OSError as e:
                self.logger.error(f"Failed to create directory {directory}: {e}")
                raise
    
    def get_directories(self) -> Tuple[str, str, str, str]:
        """
        Get all directory paths as a tuple.
        
        Returns:
            Tuple containing (results_base_dir, models_dir, plots_dir, metrics_dir)
        """
        return self.results_base_dir, self.models_dir, self.plots_dir, self.metrics_dir
    
    def get_models_dir(self) -> str:
        """Get the models directory path."""
        return self.models_dir
    
    def get_plots_dir(self) -> str:
        """Get the plots directory path."""
        return self.plots_dir
    
    def get_metrics_dir(self) -> str:
        """Get the metrics directory path."""
        return self.metrics_dir
    
    def get_results_base_dir(self) -> str:
        """Get the base results directory path."""
        return self.results_base_dir
    
    def create_subdirectory(self, parent_dir: str, subdir_name: str) -> str:
        """
        Create a subdirectory within a parent directory.
        
        Args:
            parent_dir: Parent directory path
            subdir_name: Name of subdirectory to create
            
        Returns:
            Full path to created subdirectory
        """
        subdir_path = os.path.join(parent_dir, subdir_name)
        
        try:
            os.makedirs(subdir_path, exist_ok=True)
            self.logger.debug(f"Created subdirectory: {subdir_path}")
            return subdir_path
        except OSError as e:
            self.logger.error(f"Failed to create subdirectory {subdir_path}: {e}")
            raise
    
    def get_model_file_path(self, model_type: str, epoch: int = None, is_final: bool = False) -> str:
        """
        Get the full file path for a model file.
        
        Args:
            model_type: Type of model ('generator', 'discriminator', 'gan')
            epoch: Epoch number for epoch-specific saves
            is_final: Whether this is a final model save
            
        Returns:
            Full file path for the model
        """
        if is_final:
            # Use final model filename templates
            if model_type == 'generator':
                filename = self.params.get("final_generator_model_filename", "generator_final.keras")
            elif model_type == 'discriminator':
                filename = self.params.get("final_discriminator_model_filename", "discriminator_final.keras")
            elif model_type == 'gan':
                filename = self.params.get("final_gan_model_filename", "gan_final.keras")
            else:
                filename = f"{model_type}_final.keras"
        elif epoch is not None:
            # Use epoch-specific filename templates
            if model_type == 'generator':
                template = self.params.get("save_generator_epoch_template", "generator_epoch_{epoch}.keras")
            elif model_type == 'discriminator':
                template = self.params.get("save_discriminator_epoch_template", "discriminator_epoch_{epoch}.keras")
            elif model_type == 'gan':
                template = self.params.get("save_gan_epoch_template", "gan_epoch_{epoch}.keras")
            else:
                template = f"{model_type}_epoch_{{epoch}}.keras"
            
            filename = template.format(epoch=epoch)
        else:
            # Default filename
            filename = f"{model_type}.keras"
        
        return os.path.join(self.models_dir, filename)
    
    def get_plot_file_path(self, plot_type: str, epoch: int = None, is_final: bool = False) -> str:
        """
        Get the full file path for a plot file.
        
        Args:
            plot_type: Type of plot ('loss_plot', 'architecture', etc.)
            epoch: Epoch number for epoch-specific saves
            is_final: Whether this is a final plot save
            
        Returns:
            Full file path for the plot
        """
        if is_final and plot_type == 'loss_plot':
            filename = self.params.get("final_loss_plot_filename", "loss_plot_final.png")
        elif epoch is not None and plot_type == 'loss_plot':
            template = self.params.get("loss_plot_epoch_template", "loss_plot_epoch_{epoch}.png")
            filename = template.format(epoch=epoch)
        else:
            # Handle architecture plots and other types
            if plot_type == 'generator_architecture':
                filename = self.params.get("generator_model_plot_file", "generator_architecture.png")
            elif plot_type == 'discriminator_architecture':
                filename = self.params.get("discriminator_model_plot_file", "discriminator_architecture.png")
            elif plot_type == 'gan_architecture':
                filename = self.params.get("gan_model_plot_file", "gan_architecture.png")
            else:
                filename = f"{plot_type}.png"
        
        return os.path.join(self.plots_dir, filename)
    
    def get_metrics_file_path(self, metrics_type: str = "training_metrics") -> str:
        """
        Get the full file path for a metrics file.
        
        Args:
            metrics_type: Type of metrics file
            
        Returns:
            Full file path for the metrics file
        """
        if metrics_type == "training_metrics":
            filename = self.params.get("training_metrics_filename", "training_metrics.json")
        else:
            filename = f"{metrics_type}.json"
        
        return os.path.join(self.metrics_dir, filename)
    
    def get_debug_info(self) -> Dict[str, Any]:
        """
        Get debug information for this module.
        
        Returns:
            Dict containing debug information
        """
        return {
            'results_base_dir': self.results_base_dir,
            'models_dir': self.models_dir,
            'plots_dir': self.plots_dir,
            'metrics_dir': self.metrics_dir,
            'directories_exist': {
                'results_base': os.path.exists(self.results_base_dir),
                'models': os.path.exists(self.models_dir),
                'plots': os.path.exists(self.plots_dir),
                'metrics': os.path.exists(self.metrics_dir)
            }
        }
