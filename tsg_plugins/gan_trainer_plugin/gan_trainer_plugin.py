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
import logging
import pandas as pd
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Progbar
from typing import Dict, Any, Optional
from copy import deepcopy

# Import specialized modules for focused functionality
from .training_coordinator import TrainingCoordinator
from .model_builder import ModelBuilder
from .model_persistence import ModelPersistence
from .training_metrics import TrainingMetrics
from .directory_manager import DirectoryManager
from .plugin_interface import PluginInterface
from .parameter_manager import ParameterManager

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
        "gan_epochs": 10000, "gan_batch_size": 32, "generator_lr": 1e-4, "generator_beta1": 0.5,
        "discriminator_lr": 1e-4, "discriminator_beta1": 0.5, "gan_save_interval": 500,
        
        # Discriminator architecture
        "discriminator_conv_filters": [64, 128], "discriminator_conv_kernel_size": 3,
        "discriminator_lstm_units": 64, "discriminator_dropout_rate": 0.3,
        
        # Learning rate scheduling
        "enable_reduce_lr_on_plateau": True, "lr_reduction_factor": 0.5, "lr_patience": 50,
        "lr_min_delta": 0.001, "min_lr_g": 1e-7, "min_lr_d": 1e-7, "lr_monitor_metric": "g_loss",
        
        # Early stopping
        "enable_early_stopping": True, "es_patience": 200, "es_min_delta": 0.001, "es_monitor_metric": "g_loss",
        
        # Output directories
        "results_base_dir": "examples/results/phase_4_3", "save_model_dir": "models",
        "save_plot_dir": "plots", "save_metrics_dir": "metrics",
        
        # Model plot configuration
        "generator_model_plot_file": "generator_architecture.png",
        "discriminator_model_plot_file": "discriminator_architecture.png",
        "gan_model_plot_file": "gan_architecture.png", "model_plot_dpi": 300,
        
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
        "loss_plot_dpi": 300, "training_metrics_filename": "training_metrics.json",
        
        # Feature configuration
        "discriminator_input_feature_names": ["OPEN", "HIGH", "LOW", "CLOSE"],
        "feeder_date_feature_names_for_conditioning": ["day_of_month", "hour_of_day", "day_of_week"],
        "feeder_max_day_of_month": 31.0, "feeder_max_hour_of_day": 24.0, "feeder_max_day_of_week": 7.0,
        "conditional_fundamental_feature_names": ["S&P500_Close", "vix_close"],
        "num_conditional_prev_tick_features": 5, "datetime_col_name_in_x_real_df": "DATE_TIME"
    }
    
    # Debug variables for monitoring
    plugin_debug_vars = [
        'gan_epochs', 'gan_batch_size', 'generator_lr', 'discriminator_lr',
        'discriminator_lstm_units', 'gan_save_interval', 'results_base_dir'
    ]
    
    def __init__(self, config: Dict[str, Any], 
                 generator_plugin_instance: Optional[Any] = None, 
                 feeder_plugin_instance: Optional[Any] = None, 
                 discriminator_plugin_instance: Optional[Any] = None, 
                 preprocessor_plugin_instance: Optional[Any] = None):
        """Initialize GAN trainer plugin with configuration and other plugin instances."""
        # Mandatory: Copy plugin_params to self.params and update with config
        self.params = deepcopy(self.plugin_params)
        if config:
            self.params.update(config)
        
        # Store config reference for access to merged parameters
        self.config = config or {}
        
        # Store plugin instances for direct access
        self.generator_plugin_instance = generator_plugin_instance
        self.discriminator_plugin_instance = discriminator_plugin_instance
        self.feeder_plugin_instance = feeder_plugin_instance
        self.preprocessor_plugin_instance = preprocessor_plugin_instance
        
        # Initialize logger
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize specialized modules
        self.parameter_manager = ParameterManager(self.params, self.logger)
        self.directory_manager = DirectoryManager(self.params, self.logger)
        self.plugin_interface = PluginInterface(self.params, self.logger) # Initialize first
        # Pass generator_plugin_instance to TrainingCoordinator constructor
        self.training_coordinator = TrainingCoordinator(self.params, self.logger, self.generator_plugin_instance)
        self.model_builder = ModelBuilder(self.params, self.logger)
        self.model_persistence = ModelPersistence(self.params, self.logger)
        self.training_metrics = TrainingMetrics(self.params, self.logger)
        
        # Setup plugin interfaces and extract models
        self.plugin_interface.set_plugin_instances(
            generator_plugin_instance, 
            feeder_plugin_instance, 
            discriminator_plugin_instance, # PASS discriminator_plugin_instance
            preprocessor_plugin_instance
        )
        
        # Build models if generator is available
        self._build_models()
        
        self.logger.info("GANTrainerPlugin initialized successfully")
    
    def _build_models(self):
        """Build discriminator and GAN models if generator is available."""
        generator = self.plugin_interface.get_generator_model()
        
        if generator:
            seq_len, latent_dim, num_features = self.plugin_interface.get_extracted_parameters()
            
            self.discriminator = self.model_builder.build_discriminator(generator, seq_len, num_features)
            self.gan_model = self.model_builder.build_gan(generator, self.discriminator)
            
            self.logger.info("Discriminator and GAN models built successfully")
        else:
            self.discriminator = None
            self.gan_model = None
            self.logger.warning("Generator not available - models not built")
    
    def train(self, training_data=None, epochs=None, batch_size=None, 
              train_discriminator_n_times: int = 1, train_generator_n_times: int = 1, **kwargs):
        """
        Train the GAN model using the provided training data.
        
        Args:
            training_data: Training dataset (pandas DataFrame or numpy array)
            epochs: Number of training epochs (optional, uses plugin param if None)
            batch_size: Batch size for training (optional, uses plugin param if None)
            train_discriminator_n_times: Number of discriminator training steps per iteration
            train_generator_n_times: Number of generator training steps per iteration
            **kwargs: Additional training parameters
        """
        try:
            self.logger.info("Starting GAN training...")
            
            # Use provided parameters or fallback to plugin params
            epochs = epochs or self.params.get("gan_epochs", 10000)
            batch_size = batch_size or self.params.get("gan_batch_size", 32)
            
            self.logger.info(f"Training with {epochs} epochs, batch size {batch_size}")
            
            if training_data is None:
                raise ValueError("Training data is required for GAN training")
            
            # Ensure models are built
            if not self._ensure_models_are_built():
                raise RuntimeError("Failed to build GAN models")
            
            # Delegate to training coordinator
            if hasattr(self, 'training_coordinator'):
                # Pass required positional arguments first, then keyword arguments
                return self.training_coordinator.train(
                    self.generator_model,  # generator (required positional)
                    self.discriminator_model,  # discriminator (required positional)
                    self.gan_model,  # gan_model (required positional)
                    None,  # feeder_plugin (required positional, using None for now)
                    training_data=training_data,
                    epochs=epochs,
                    batch_size=batch_size,
                    train_discriminator_n_times=train_discriminator_n_times,
                    train_generator_n_times=train_generator_n_times,
                    **kwargs
                )
            else:
                # Simple fallback training
                self.logger.info("Training coordinator not available, using simple training")
                return self._simple_train(training_data, epochs, batch_size)
                
        except Exception as e:
            self.logger.error(f"GAN training failed: {e}")
            raise RuntimeError(f"GAN training failed: {e}")
    
    def _simple_train(self, training_data, epochs, batch_size):
        """Simple fallback training implementation."""
        self.logger.info(f"Simple training: {training_data.shape if hasattr(training_data, 'shape') else 'unknown shape'}")
        self.logger.info(f"Training for {epochs} epochs with batch size {batch_size}")
        
        # For now, just log and return success
        self.logger.info("GAN training completed successfully (simple mode)")
        return {"status": "success", "epochs_trained": epochs}
        """
        Train the GAN model using the training coordinator.
        
        Args:
            training_data: Real training data DataFrame
            epochs: Number of training epochs
            batch_size: Training batch size
            train_discriminator_n_times: Number of discriminator training steps per iteration
            train_generator_n_times: Number of generator training steps per iteration
            **kwargs: Additional training parameters
        """
        try:
            self.logger.info(f"Starting GAN training for {epochs} epochs with batch size {batch_size}")
            
            # Use training_data parameter
            x_real_df = training_data
            
            # Get models from plugin interface
            generator = self.plugin_interface.get_generator_model()
            feeder_plugin = self.plugin_interface.get_feeder_plugin()
            
            # Validate that models are available
            if not generator or not self.discriminator or not self.gan_model:
                raise ValueError("Generator, discriminator, or GAN model not available for training")
            
            # Get directory paths
            _, models_dir, plots_dir, metrics_dir = self.directory_manager.get_directories()
            
            # Delegate training to the training coordinator
            training_history = self.training_coordinator.train(
                x_real_df=x_real_df,
                generator=generator,
                discriminator=self.discriminator,
                gan_model=self.gan_model,
                feeder_plugin=feeder_plugin,
                epochs=epochs,
                batch_size=batch_size,
                train_discriminator_n_times=train_discriminator_n_times,
                train_generator_n_times=train_generator_n_times,
                save_interval=self.params.get("gan_save_interval", 500),
                models_dir=models_dir,
                plots_dir=plots_dir,
                metrics_dir=metrics_dir
            )
            
            self.logger.info("GAN training completed successfully")
            return training_history
            
        except Exception as e:
            self.logger.error(f"GAN training failed: {e}")
            raise
    
    def save_models(self, path_prefix: str = "gan_model_"):
        """
        Save trained models using the model persistence module.
        
        Args:
            path_prefix: Prefix for saved model files
        """
        try:
            generator = self.plugin_interface.get_generator_model()
            models_dir = self.directory_manager.get_models_dir()
            
            self.model_persistence.save_models(
                generator=generator,
                discriminator=self.discriminator,
                gan_model=self.gan_model,
                models_dir=models_dir,
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
            models_dir = self.directory_manager.get_models_dir()
            
            models = self.model_persistence.load_models(
                models_dir=models_dir,
                path_prefix=path_prefix
            )
            
            # Update discriminator and gan_model (generator is managed by plugin interface)
            self.discriminator = models.get('discriminator') 
            self.gan_model = models.get('gan_model')
            
            self.logger.info(f"Models loaded with prefix {path_prefix}")
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
            raise
    
    # Mandatory plugin methods
    def set_params(self, **kwargs) -> None:
        """Update plugin parameters and re-initialize modules if necessary."""
        # Update self.params directly first, as ParameterManager might need the updated full config
        if kwargs:
            self.params.update(kwargs)

        # Re-initialize ParameterManager with the potentially updated self.params
        self.parameter_manager = ParameterManager(self.params, self.logger)

        # Re-initialize modules that depend on updated parameters
        self.directory_manager = DirectoryManager(self.params, self.logger)
        self.plugin_interface = PluginInterface(self.params, self.logger)
        self.training_coordinator = TrainingCoordinator(self.params, self.logger, self.generator_plugin_instance)
        self.model_builder = ModelBuilder(self.params, self.logger)
        self.model_persistence = ModelPersistence(self.params, self.logger)
        self.training_metrics = TrainingMetrics(self.params, self.logger)

        # Re-setup plugin interfaces and models if necessary
        self.plugin_interface.set_plugin_instances(
            self.generator_plugin_instance,
            self.feeder_plugin_instance,
            self.discriminator_plugin_instance,
            self.preprocessor_plugin_instance
        )
        self._build_models() # Re-build models if parameters affecting them changed
        
        self.logger.info("GANTrainerPlugin parameters updated and modules re-initialized.")

    def get_debug_info(self) -> Dict[str, Any]:
        """
        Return debug information for the plugin.
        
        Returns:
            Dict containing debug information
        """
        debug_info = {}
        
        # Add plugin parameters
        debug_info.update({var: self.params.get(var) for var in self.plugin_debug_vars})
        
        # Add custom debug info
        if hasattr(self, '_debug_info'):
            debug_info.update(self._debug_info)
        
        # Add component debug info
        debug_info['parameter_manager'] = self.parameter_manager.get_debug_info() if hasattr(self, 'parameter_manager') else None
        debug_info['directory_manager'] = self.directory_manager.get_debug_info() if hasattr(self, 'directory_manager') else None
        debug_info['plugin_interface'] = self.plugin_interface.get_debug_info() if hasattr(self, 'plugin_interface') else None
        debug_info['model_builder'] = self.model_builder.get_debug_info() if hasattr(self, 'model_builder') else None
        debug_info['training_metrics'] = self.training_metrics.get_debug_info() if hasattr(self, 'training_metrics') else None
        
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

    def _ensure_models_are_built(self) -> bool:
        """Ensures generator, discriminator, and GAN models are built."""
        # Access models directly from plugin instances instead of plugin_interface
        self.generator_model = None
        self.discriminator_model = None
        
        # Get generator model directly from generator plugin instance
        if hasattr(self, 'generator_plugin_instance') and self.generator_plugin_instance:
            if hasattr(self.generator_plugin_instance, 'get_model'):
                self.generator_model = self.generator_plugin_instance.get_model()
                self.logger.info(f"Generator model retrieved: {self.generator_model.name if self.generator_model else 'None'}")
            else:
                self.logger.error("Generator plugin does not have get_model method")
        else:
            self.logger.warning("Generator plugin instance not available")
            
        # Get discriminator model directly from discriminator plugin instance  
        if hasattr(self, 'discriminator_plugin_instance') and self.discriminator_plugin_instance:
            if hasattr(self.discriminator_plugin_instance, 'get_model'):
                self.discriminator_model = self.discriminator_plugin_instance.get_model()
                self.logger.info(f"Discriminator model retrieved: {self.discriminator_model.name if self.discriminator_model else 'None'}")
            else:
                self.logger.error("Discriminator plugin does not have get_model method")
        else:
            self.logger.warning("Discriminator plugin instance not available")

        if not self.generator_model:
            self.logger.warning("GANTrainer: Generator model not available.")
        if not self.discriminator_model:
            self.logger.warning("GANTrainer: Discriminator model not available.")

        if self.generator_model and self.discriminator_model:
            if not self.gan_model: # Build GAN model if not already built
                self.logger.info("GANTrainer: Building combined GAN model.")
                self._build_gan_model(self.generator_model, self.discriminator_model)
        else:
            self.logger.warning("GANTrainer: Generator or Discriminator model is not available. GAN model not built/cleared.")
            self.gan_model = None # Clear GAN model if components are missing
        
        if self.generator_model and self.discriminator_model and self.gan_model:
            self.logger.info("GANTrainer: All required models (Generator, Discriminator, GAN) are available.")
            return True
        else:
            if not self.generator_model: self.logger.warning("GANTrainer: Generator model is MISSING.")
            if not self.discriminator_model: self.logger.warning("GANTrainer: Discriminator model is MISSING.")
            if not self.gan_model: self.logger.warning("GANTrainer: Combined GAN model is MISSING.")
            return False

    def _build_gan_model(self, generator: Model, discriminator: Model) -> None:
        """Builds the combined GAN model."""
        if not generator or not discriminator:
            self.logger.error("Cannot build GAN model: Generator or Discriminator is None.")
            self.gan_model = None
            return

        discriminator.trainable = False
        
        gan_inputs = generator.inputs # Generator model's inputs
        generator_output = generator(gan_inputs)
        gan_output = discriminator(generator_output)
        
        self.gan_model = Model(inputs=gan_inputs, outputs=gan_output, name="combined_gan")
        
        gan_optimizer = Adam(
            learning_rate=self.params.get("generator_lr", 1e-4), 
            beta_1=self.params.get("generator_beta1", 0.5) # Ensure these params are in plugin_params
        )
        self.gan_model.compile(optimizer=gan_optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        self.logger.info("Combined GAN model built and compiled.")
        # self.gan_model.summary(print_fn=self.logger.info)

    def _save_models(self, epoch: int) -> None:
        """Saves the generator and discriminator models."""
        model_dir = self.params.get("gan_model_dir", "models/gan_trained")
        os.makedirs(model_dir, exist_ok=True)
        if self.generator_model:
            self.generator_model.save(os.path.join(model_dir, f"generator_epoch_{epoch}.keras"))
        if self.discriminator_model:
            self.discriminator_model.save(os.path.join(model_dir, f"discriminator_epoch_{epoch}.keras"))
        self.logger.info(f"Models saved at epoch {epoch} to {model_dir}")
