#!/usr/bin/env python3
"""
gan_trainer_plugin.py

Main GAN Trainer Plugin with extreme separation of concerns.
This plugin coordinates GAN training by delegating to specialized modules.

Preserves the mandatory plugin structure:
- plugin_params: Class variable with defaults
- __init__: Copies plugin_params to self.params and updates with config
- set_params(): Updates parameters
- get_debug_info(): Returns debug information
- add_debug_info(): Adds debug info to dictionary

Author: TimeSeries-GAN Team
"""

import os
import sys
import logging
import traceback
from typing import Dict, Any, Optional
from copy import deepcopy

# Import specialized modules for focused functionality
from .training_coordinator import TrainingCoordinator
from .model_builder import ModelBuilder
from .data_generator import DataGenerator
from .model_persistence import ModelPersistence
from .training_metrics import TrainingMetrics

# Configure logger
logger = logging.getLogger(__name__)


class GANTrainerPlugin:
    """
    Main GAN Trainer Plugin following extreme separation of concerns.
    
    This plugin coordinates GAN training by delegating specific tasks to
    specialized modules while maintaining the mandatory plugin interface.
    
    The plugin integrates with generator and discriminator plugins to
    build and train GAN models using the configured training data.
    """
    
    # Mandatory plugin_params class variable with defaults
    plugin_params: Dict[str, Any] = {
        # Core training parameters
        "gan_epochs": 10000,
        "gan_batch_size": 32,
        "generator_lr": 1e-4,
        "generator_beta1": 0.5,
        "discriminator_lr": 1e-4,
        "discriminator_beta1": 0.5,
        "gan_save_interval": 500,
        
        # Discriminator architecture parameters
        "discriminator_conv_filters": [64, 128],
        "discriminator_conv_kernel_size": 3,
        "discriminator_lstm_units": 64,
        "discriminator_dropout_rate": 0.3,
        
        # Learning rate scheduling
        "enable_reduce_lr_on_plateau": True,
        "lr_reduction_factor": 0.5,
        "lr_patience": 50,
        "lr_min_delta": 0.001,
        "min_lr_g": 1e-7,
        "min_lr_d": 1e-7,
        "lr_monitor_metric": "g_loss",
        
        # Early stopping
        "enable_early_stopping": True,
        "es_patience": 200,
        "es_min_delta": 0.001,
        "es_monitor_metric": "g_loss",
        
        # Output directories
        "results_base_dir": "examples/results/phase_4_3",
        "save_model_dir": "models",
        "save_plot_dir": "plots",
        "save_metrics_dir": "metrics",
        
        # Model plot configuration
        "generator_model_plot_file": "generator_architecture.png",
        "discriminator_model_plot_file": "discriminator_architecture.png",
        "gan_model_plot_file": "gan_architecture.png",
        "model_plot_dpi": 300,
        
        # Model persistence templates
        "save_generator_epoch_template": "generator_epoch_{epoch}.keras",
        "final_generator_model_filename": "generator_final.keras",
        "save_discriminator_epoch_template": "discriminator_epoch_{epoch}.keras",
        "final_discriminator_model_filename": "discriminator_final.keras",
        "save_gan_epoch_template": "gan_epoch_{epoch}.keras",
        "final_gan_model_filename": "gan_final.keras",
        
        # Training metrics and visualization
        "loss_plot_epoch_template": "loss_plot_epoch_{epoch}.png",
        "final_loss_plot_filename": "loss_plot_final.png",
        "loss_plot_dpi": 300,
        "training_metrics_filename": "training_metrics.json",
        
        # Feature configuration for discriminator
        "discriminator_input_feature_names": ["OPEN", "HIGH", "LOW", "CLOSE"],
        "feeder_date_feature_names_for_conditioning": ["day_of_month", "hour_of_day", "day_of_week"],
        "feeder_max_day_of_month": 31.0,
        "feeder_max_hour_of_day": 24.0,
        "feeder_max_day_of_week": 7.0,
        "conditional_fundamental_feature_names": ["S&P500_Close", "vix_close"],
        "num_conditional_prev_tick_features": 5,
        "datetime_col_name_in_x_real_df": "DATE_TIME"
    }
    
    # Debug variables for monitoring
    plugin_debug_vars = [
        'gan_epochs', 'gan_batch_size', 'generator_lr', 'discriminator_lr',
        'discriminator_lstm_units', 'gan_save_interval', 'results_base_dir'
    ]
    
    def __init__(self, config: Dict[str, Any], generator_plugin_instance: Optional[Any] = None, 
                 feeder_plugin_instance: Optional[Any] = None, 
                 preprocessor_plugin_instance: Optional[Any] = None):
        """
        Initialize GAN trainer plugin with configuration and other plugin instances.
        
        Follows mandatory plugin structure:
        1. Copy plugin_params to self.params
        2. Update self.params with config values
        3. Initialize plugin-specific attributes
        
        Args:
            config: Configuration dictionary containing merged parameters
            generator_plugin_instance: Generator plugin for model access
            feeder_plugin_instance: Feeder plugin for data generation
            preprocessor_plugin_instance: Optional preprocessor plugin
        """
        # Mandatory: Copy plugin_params to self.params and update with config
        self.params = deepcopy(self.plugin_params)
        if config:
            self.params.update(config)
        
        # Store config reference for access to merged parameters
        self.config = config or {}
        
        # Store plugin instances for model access
        self.generator_plugin = generator_plugin_instance
        self.feeder_plugin = feeder_plugin_instance  
        self.preprocessor_plugin = preprocessor_plugin_instance
        
        # Initialize logger
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize core training components using specialized modules
        self._initialize_components()
        
        # Extract generator model from generator plugin
        self._setup_generator_model()
        
        # Initialize training parameters and build models
        self._initialize_training_setup()
        
        self.logger.info("GANTrainerPlugin initialized successfully")
    
    def _initialize_components(self):
        """Initialize specialized component modules."""
        self.training_coordinator = TrainingCoordinator(self.params, self.logger)
        self.model_builder = ModelBuilder(self.params, self.logger)
        self.data_generator = DataGenerator(self.params, self.logger)
        self.model_persistence = ModelPersistence(self.params, self.logger)
        self.training_metrics = TrainingMetrics(self.params, self.logger)
    
    def _setup_generator_model(self):
        """Extract generator model from generator plugin."""
        self.generator = None
        
        if self.generator_plugin:
            # Try different attribute names for generator model
            if hasattr(self.generator_plugin, 'generator_model') and self.generator_plugin.generator_model:
                self.generator = self.generator_plugin.generator_model
                self.logger.info("Generator model retrieved from generator_plugin.generator_model")
            elif hasattr(self.generator_plugin, 'model') and self.generator_plugin.model:
                self.generator = self.generator_plugin.model
                self.logger.info("Generator model retrieved from generator_plugin.model")
            elif hasattr(self.generator_plugin, 'get_model') and callable(self.generator_plugin.get_model):
                self.generator = self.generator_plugin.get_model()
                self.logger.info("Generator model retrieved from generator_plugin.get_model()")
            else:
                self.logger.error("Generator plugin does not have accessible model")
        else:
            self.logger.warning("Generator plugin instance not provided")
    
    def _initialize_training_setup(self):
        """Initialize training setup including model building."""
        # Initialize output directories
        self._setup_output_directories()
        
        # Extract core parameters from generator and config
        self._extract_core_parameters()
        
        # Build discriminator and GAN models if generator is available
        if self.generator:
            self.discriminator = self.model_builder.build_discriminator(
                self.generator, self.seq_len, self.num_discriminator_features
            )
            self.gan_model = self.model_builder.build_gan(
                self.generator, self.discriminator
            )
            self.logger.info("Discriminator and GAN models built successfully")
        else:
            self.discriminator = None
            self.gan_model = None
            self.logger.warning("Generator not available - models not built")
    
    def _setup_output_directories(self):
        """Setup output directories for models, plots, and metrics."""
        self.results_base_dir = self.params.get("results_base_dir", "examples/results/gan_training")
        self.models_dir = os.path.join(self.results_base_dir, self.params.get("save_model_dir", "models"))
        self.plots_dir = os.path.join(self.results_base_dir, self.params.get("save_plot_dir", "plots"))
        self.metrics_dir = os.path.join(self.results_base_dir, self.params.get("save_metrics_dir", "metrics"))
        
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
        
        self.logger.info(f"Output directories setup under {self.results_base_dir}")
    
    def _extract_core_parameters(self):
        """Extract core parameters from generator model and configuration."""
        # Default values
        self.seq_len = self.params.get("seq_len", 144)
        self.latent_dim = self.params.get("latent_dim", 128)
        self.num_discriminator_features = len(self.params.get("discriminator_input_feature_names", ["OPEN", "HIGH", "LOW", "CLOSE"]))
        
        # Try to extract from generator if available
        if self.generator and hasattr(self.generator, 'input_shape'):
            try:
                # Extract sequence length from generator input
                generator_input_shape = self.generator.input_shape
                if isinstance(generator_input_shape, list) and len(generator_input_shape) > 0:
                    # Handle multiple inputs - find latent input
                    for input_shape in generator_input_shape:
                        if len(input_shape) == 3:  # (batch, seq, features)
                            self.seq_len = input_shape[1] or self.seq_len
                            self.latent_dim = input_shape[2] or self.latent_dim
                            break
                elif len(generator_input_shape) == 3:
                    self.seq_len = generator_input_shape[1] or self.seq_len
                    self.latent_dim = generator_input_shape[2] or self.latent_dim
                
                self.logger.info(f"Extracted from generator - seq_len: {self.seq_len}, latent_dim: {self.latent_dim}")
            except Exception as e:
                self.logger.warning(f"Could not extract parameters from generator: {e}")
    
    def train(self, x_real_df, epochs: int, batch_size: int, 
              train_discriminator_n_times: int = 1, train_generator_n_times: int = 1):
        """
        Train the GAN model using the training coordinator.
        
        Args:
            x_real_df: Real training data DataFrame
            epochs: Number of training epochs
            batch_size: Training batch size
            train_discriminator_n_times: Number of discriminator training steps per iteration
            train_generator_n_times: Number of generator training steps per iteration
        """
        try:
            self.logger.info(f"Starting GAN training for {epochs} epochs with batch size {batch_size}")
            
            # Validate that models are available
            if not self.generator or not self.discriminator or not self.gan_model:
                raise ValueError("Generator, discriminator, or GAN model not available for training")
            
            # Delegate training to the training coordinator
            training_history = self.training_coordinator.train(
                x_real_df=x_real_df,
                generator=self.generator,
                discriminator=self.discriminator,
                gan_model=self.gan_model,
                feeder_plugin=self.feeder_plugin,
                epochs=epochs,
                batch_size=batch_size,
                train_discriminator_n_times=train_discriminator_n_times,
                train_generator_n_times=train_generator_n_times,
                save_interval=self.params.get("gan_save_interval", 500),
                models_dir=self.models_dir,
                plots_dir=self.plots_dir,
                metrics_dir=self.metrics_dir
            )
            
            self.logger.info("GAN training completed successfully")
            return training_history
            
        except Exception as e:
            self.logger.error(f"GAN training failed: {e}")
            traceback.print_exc()
            raise
    
    def save_models(self, path_prefix: str = "gan_model_"):
        """
        Save trained models using the model persistence module.
        
        Args:
            path_prefix: Prefix for saved model files
        """
        try:
            self.model_persistence.save_models(
                generator=self.generator,
                discriminator=self.discriminator,
                gan_model=self.gan_model,
                models_dir=self.models_dir,
                path_prefix=path_prefix
            )
            self.logger.info(f"Models saved with prefix {path_prefix}")
        except Exception as e:
            self.logger.error(f"Failed to save models: {e}")
            raise
    
    def load_models(self, path_prefix: str = "gan_model_"):
        """
        Load trained models using the model persistence module.
        
        Args:
            path_prefix: Prefix for model files to load
        """
        try:
            models = self.model_persistence.load_models(
                models_dir=self.models_dir,
                path_prefix=path_prefix
            )
            
            self.generator = models.get('generator')
            self.discriminator = models.get('discriminator') 
            self.gan_model = models.get('gan_model')
            
            self.logger.info(f"Models loaded with prefix {path_prefix}")
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
            raise
    
    # Mandatory plugin methods
    def set_params(self, **kwargs):
        """
        Update plugin parameters with provided configuration.
        
        Args:
            **kwargs: Parameter key-value pairs to update
        """
        self.logger.info(f"Setting parameters: {list(kwargs.keys())}")
        
        # Update self.params with new values
        for key, value in kwargs.items():
            self.params[key] = value
        
        # Update config if available
        if hasattr(self, 'config') and self.config is not None:
            self.config.update(kwargs)
        
        # Reinitialize components with updated parameters
        self._initialize_components()
        
        # Reinitialize training setup if generator is available
        if hasattr(self, 'generator') and self.generator:
            self._initialize_training_setup()
        
        self.logger.info("Parameters updated successfully")

    def process(self, x_real_df, **kwargs):
        """
        Main processing method - trains the GAN model.
        
        Args:
            x_real_df: Real training data DataFrame
            **kwargs: Additional training parameters
        
        Returns:
            Training results dictionary
        """
        try:
            self.logger.info("Starting GAN training process")
            
            # Extract training parameters
            epochs = kwargs.get('epochs', self.params.get('gan_epochs', 1000))
            batch_size = kwargs.get('batch_size', self.params.get('gan_batch_size', 32))
            train_discriminator_n_times = kwargs.get('train_discriminator_n_times', 1)
            train_generator_n_times = kwargs.get('train_generator_n_times', 1)
            
            # Validate that models are available
            if not hasattr(self, 'generator') or not self.generator:
                raise ValueError("Generator model not available for training")
            if not hasattr(self, 'discriminator') or not self.discriminator:
                raise ValueError("Discriminator model not available for training")
            if not hasattr(self, 'gan_model') or not self.gan_model:
                raise ValueError("GAN model not available for training")
            
            # Delegate training to the training coordinator
            training_history = self.training_coordinator.train(
                x_real_df=x_real_df,
                generator=self.generator,
                discriminator=self.discriminator,
                gan_model=self.gan_model,
                feeder_plugin=self.feeder_plugin,
                epochs=epochs,
                batch_size=batch_size,
                train_discriminator_n_times=train_discriminator_n_times,
                train_generator_n_times=train_generator_n_times,
                save_interval=self.params.get("gan_save_interval", 500),
                models_dir=self.models_dir,
                plots_dir=self.plots_dir,
                metrics_dir=self.metrics_dir
            )
            
            # Save final models
            self.save_models("final_")
            
            # Save final metrics and plots
            self.training_metrics.save_metrics(self.metrics_dir)
            self.training_metrics.plot_training_history(self.plots_dir)
            
            # Prepare results
            results = {
                'training_history': training_history,
                'final_metrics': self.training_metrics.get_latest_metrics(),
                'models_dir': self.models_dir,
                'plots_dir': self.plots_dir,
                'metrics_dir': self.metrics_dir
            }
            
            self.logger.info("GAN training process completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"GAN training process failed: {e}")
            traceback.print_exc()
            raise
    
    def get_debug_info(self):
        """
        Return debug information for the plugin.
        
        Returns:
            Dict containing debug information
        """
        debug_info = {var: self.params.get(var) for var in self.plugin_debug_vars}
        
        # Add module debug info
        if hasattr(self, 'training_coordinator') and self.training_coordinator:
            debug_info['training_coordinator'] = self.training_coordinator.get_debug_info()
        if hasattr(self, 'model_builder') and self.model_builder:
            debug_info['model_builder'] = self.model_builder.get_debug_info()
        if hasattr(self, 'training_metrics') and self.training_metrics:
            debug_info['training_metrics'] = self.training_metrics.get_debug_info()
        
        return debug_info
    
    def add_debug_info(self, key, value):
        """
        Add debug information key-value pair.
        
        Args:
            key: Debug info key
            value: Debug info value
        """
        if not hasattr(self, '_debug_info'):
            self._debug_info = {}
        self._debug_info[key] = value
        
        # Reinitialize training setup if generator is available
        if self.generator:
            self._initialize_training_setup()
        
        self.logger.info("Parameters updated successfully")
    
    def get_debug_info(self):
        """
        Return debug information for the plugin.
        
        Returns:
            Dict containing debug information
        """
        debug_info = {}
        
        # Add plugin parameters
        if hasattr(self, 'plugin_debug_vars'):
            debug_info.update({var: self.params.get(var) for var in self.plugin_debug_vars})
        
        # Add custom debug info
        if hasattr(self, '_debug_info'):
            debug_info.update(self._debug_info)
        
        # Add component debug info
        if hasattr(self, 'model_builder') and self.model_builder:
            debug_info['model_builder'] = self.model_builder.get_debug_info()
        if hasattr(self, 'training_metrics') and self.training_metrics:
            debug_info['training_metrics'] = self.training_metrics.get_debug_info()
        
        return debug_info
    
    def add_debug_info(self, key, value):
        """
        Add debug information key-value pair.
        
        Args:
            key: Debug info key
            value: Debug info value
        """
        if not hasattr(self, '_debug_info'):
            self._debug_info = {}
        self._debug_info[key] = value
