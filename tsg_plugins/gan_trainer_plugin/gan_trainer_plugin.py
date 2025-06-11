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
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping # Import EarlyStopping
from tensorflow.keras import backend as K # Import Keras backend
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple

# Assuming TrainingCoordinator is in the same directory or correctly pathed
from .training_coordinator import TrainingCoordinator 
# from ..generator_plugin.generator_plugin import GeneratorPlugin # Example if in different plugin dir
# from ..discriminator_plugin.discriminator_plugin import DiscriminatorPlugin # Example
# from ..feeder_plugin.feeder_plugin import FeederPlugin # Example

from app.utils.logging_utils import get_logger
from tsg_plugins.plugin_base import PluginBase

class GANTrainerPlugin(PluginBase):
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
        "enable_reduce_lr_on_plateau": True, "lr_reduction_factor": 0.5, "lr_patience": 10, # Reduced for faster effect in testing, adjust as needed
        "lr_min_delta": 0.001, "min_lr_g": 1e-7, "min_lr_d": 1e-7, "lr_monitor_metric_g": "g_loss", # Monitor g_loss for generator
        "lr_monitor_metric_d": "d_loss", # Monitor d_loss for discriminator
        
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
        "num_conditional_prev_tick_features": 5, "datetime_col_name_in_x_real_df": "DATE_TIME",
        
        # Additional parameters
        "generator_l2_reg": 0.01,  # Added for generator L2 regularization
    }
    
    # Debug variables for monitoring
    plugin_debug_vars = [
        'gan_epochs', 'gan_batch_size', 'generator_lr', 'discriminator_lr',
        'discriminator_lstm_units', 'gan_save_interval', 'results_base_dir'
    ]
    
    def __init__(self, config: Dict[str, Any], 
                 generator_plugin: Any, 
                 discriminator_plugin: Any,
                 feeder_plugin: Any = None): # Feeder plugin might be needed for data
        """
        Initialize the GANTrainerPlugin.

        Args:
            config: Configuration dictionary.
            generator_plugin: Instance of GeneratorPlugin.
            discriminator_plugin: Instance of DiscriminatorPlugin.
            feeder_plugin: Instance of FeederPlugin.
        """
        self.logger = logging.getLogger(__name__)
        self.main_config = config.copy()
        self.params = {} # Initialize before set_params
        self._initialize_parameters() # Call to populate self.params from main_config and defaults

        # Store plugin instances
        if generator_plugin is None:
            self.logger.error("Critical: GANTrainerPlugin received no GeneratorPlugin instance.")
        self.generator_plugin = generator_plugin

        if discriminator_plugin is None:
            self.logger.error("Critical: GANTrainerPlugin received no DiscriminatorPlugin instance.")
        self.discriminator_plugin = discriminator_plugin

        if feeder_plugin is None:
            self.logger.error("Critical: GANTrainerPlugin received no FeederPlugin instance.")
        self.feeder_plugin = feeder_plugin
        
        self.generator_model: Optional[tf.keras.Model] = None
        self.discriminator_model: Optional[tf.keras.Model] = None
        self.gan_model: Optional[tf.keras.Model] = None
        
        # Initialize TrainingCoordinator
        # Pass self.params which should now be populated
        self.training_coordinator = TrainingCoordinator(
            params=self.params, 
            logger=self.logger, # Pass logger to TrainingCoordinator
            generator_plugin=self.generator_plugin # Pass the stored instance
        )
        self.logger.info("GANTrainerPlugin initialized.")

        self.generator_l2_reg = self.params.get("generator_l2_reg")

        # Initialize ReduceLROnPlateau callbacks
        if self.params.get("enable_reduce_lr_on_plateau"):
            self.lr_scheduler_g = ReduceLROnPlateau(
                monitor=self.params.get("lr_monitor_metric_g", "g_loss"),
                factor=self.params.get("lr_reduction_factor"),
                patience=self.params.get("lr_patience"),
                verbose=1,
                min_delta=self.params.get("lr_min_delta"),
                min_lr=self.params.get("min_lr_g")
            )
            self.lr_scheduler_d = ReduceLROnPlateau(
                monitor=self.params.get("lr_monitor_metric_d", "d_loss"),
                factor=self.params.get("lr_reduction_factor"),
                patience=self.params.get("lr_patience"),
                verbose=1,
                min_delta=self.params.get("lr_min_delta"),
                min_lr=self.params.get("min_lr_d")
            )
            # Associate models with schedulers - this needs to happen after models are compiled
            # We will call them manually in the training loop.
        else:
            self.lr_scheduler_g = None
            self.lr_scheduler_d = None
        
        self.feeder_plugin = feeder_plugin # Store feeder plugin

        # Initialize EarlyStopping callback
        self.early_stopping_callback = None
        if self.params.get("enable_early_stopping"):
            self.logger.info(f"Early stopping enabled. Metric: {self.params.get('es_monitor_metric')}, Patience: {self.params.get('es_patience')}")
            self.early_stopping_callback = EarlyStopping(
                monitor=self.params.get("es_monitor_metric", "g_loss"),
                min_delta=self.params.get("es_min_delta", 0.001),
                patience=self.params.get("es_patience", 50), # Defaulted to 50 from 200 for quicker testing if needed
                verbose=1,
                mode='min', # Assuming lower loss is better
                restore_best_weights=self.params.get("es_restore_best_weights", False) # Default to False
            )
        else:
            self.logger.info("Early stopping is disabled by configuration.")

    def _initialize_parameters(self):
        """Initialize self.params from main_config and plugin_params defaults."""
        # Start with plugin defaults
        self.params = self.plugin_params.copy()
        # Override with main_config values
        if hasattr(self, 'main_config') and self.main_config is not None:
            for key, value in self.main_config.items():
                # Update if key is in plugin_params or if it's a general param
                if key in self.params or not key.startswith(("generator_", "discriminator_", "feeder_")):
                    self.params[key] = value
                # Handle prefixed params specifically for this plugin
                if key.startswith("trainer_"):
                    param_key = key[len("trainer_"):]
                    self.params[param_key] = value
        self.logger.debug(f"GANTrainerPlugin params initialized: {self.params.keys()}")

    def set_params(self, **kwargs) -> None:
        """
        Update plugin parameters.
        """
        self.logger.debug(f"GANTrainerPlugin.set_params called with: {list(kwargs.keys())}")
        # Update main config, which is the primary source of truth from outside
        if hasattr(self, 'main_config') and self.main_config is not None:
            self.main_config.update(kwargs)
        else:
            self.main_config = kwargs.copy()

        # Re-initialize/update self.params based on the potentially changed main_config and new kwargs
        # current_params_copy = self.params.copy() # Keep a copy of current specific params
        self._initialize_parameters() # This will reload from main_config and defaults
        
        # Now, apply kwargs directly to self.params to ensure CLI overrides etc. are effective
        # This prioritizes kwargs passed to set_params directly
        for key, value in kwargs.items():
            if key.startswith("trainer_"):
                param_key = key[len("trainer_"):]
                self.params[param_key] = value
            else:
                # Also allow non-prefixed keys from kwargs to update params if they exist in plugin_params
                if key in self.plugin_params:
                     self.params[key] = value
                # Or if they are general config keys not specific to other plugins
                elif not key.startswith(("generator_", "discriminator_", "feeder_")):
                    self.params[key] = value


        # After updating params, if TrainingCoordinator exists and has a set_params method:
        if hasattr(self, 'training_coordinator') and self.training_coordinator and hasattr(self.training_coordinator, 'set_params'):
            # Pass only the relevant params for TrainingCoordinator if possible, or all if it handles filtering
            self.training_coordinator.set_params(**self.params)
        
        self.logger.debug(f"GANTrainerPlugin params updated: {self.params.keys()}")

    def _build_models(self) -> bool:
        """
        Build or retrieve generator, discriminator, and combined GAN models.
        """
        self.logger.info("GANTrainer: Building/retrieving GAN models...")

        if self.generator_plugin is None:
            self.logger.error("GANTrainer: Generator plugin instance not available.")
            return False
        if not hasattr(self.generator_plugin, 'get_model'):
            self.logger.error("GANTrainer: Generator plugin has no get_model method.")
            return False
        self.generator_model = self.generator_plugin.get_model()
        if self.generator_model is None:
            self.logger.error("GANTrainer: Generator model not available from plugin.")
            return False
        self.logger.info(f"GANTrainer: Generator model retrieved: {self.generator_model.name}")

        if self.discriminator_plugin is None:
            self.logger.error("GANTrainer: Discriminator plugin instance not available.")
            return False
        if not hasattr(self.discriminator_plugin, 'get_model'):
            self.logger.error("GANTrainer: Discriminator plugin has no get_model method.")
            return False
        self.discriminator_model = self.discriminator_plugin.get_model()
        if self.discriminator_model is None:
            self.logger.error("GANTrainer: Discriminator model not available from plugin.")
            return False
        self.logger.info(f"GANTrainer: Discriminator model retrieved: {self.discriminator_model.name}")

        # Log initial trainable status of generator and discriminator
        self.logger.info(f"GANTrainer: Initial generator_model.name: {self.generator_model.name}, trainable: {self.generator_model.trainable}, num_trainable_vars: {len(self.generator_model.trainable_variables)}")
        self.logger.info(f"GANTrainer: Initial discriminator_model.name: {self.discriminator_model.name}, trainable: {self.discriminator_model.trainable}, num_trainable_vars: {len(self.discriminator_model.trainable_variables)}")

        # Build the combined GAN model
        try:
            # Ensure the generator_model itself is marked as trainable for the GAN.
            self.generator_model.trainable = True
            self.logger.info(f"GANTrainer: Set self.generator_model.trainable = True. Current num_trainable_vars: {len(self.generator_model.trainable_variables)}")

            if not isinstance(self.generator_model.inputs, list) or len(self.generator_model.inputs) < 3:
                self.logger.error(f"GANTrainer: Generator model inputs are not as expected (list of 3). Got: {self.generator_model.inputs}")
                return False

            noise_input = self.generator_model.inputs[0]
            conditions_input = self.generator_model.inputs[1]
            context_input = self.generator_model.inputs[2]
            
            generated_sequence = self.generator_model([noise_input, conditions_input, context_input])
            
            # For the combined GAN model used to train the generator, the discriminator's weights should be frozen.
            # The original self.discriminator_model remains trainable for its separate training step.
            discriminator_for_gan = tf.keras.models.clone_model(self.discriminator_model)
            discriminator_for_gan.set_weights(self.discriminator_model.get_weights())
            discriminator_for_gan.trainable = False # Standard GAN practice
            self.logger.info(f"GANTrainer: discriminator_for_gan.trainable set to {discriminator_for_gan.trainable}. Num_trainable_vars: {len(discriminator_for_gan.trainable_variables)}")
            
            gan_output = discriminator_for_gan(generated_sequence)
            
            self.gan_model = tf.keras.Model(
                inputs=[noise_input, conditions_input, context_input],
                outputs=gan_output,
                name="combined_gan_model"
            )
            
            # Log layers of gan_model and their trainable status
            self.logger.info(f"GANTrainer: GAN model layers: {[layer.name for layer in self.gan_model.layers]}")
            if len(self.gan_model.layers) > 0:
                gen_layer_in_gan = self.gan_model.layers[0]
                self.logger.info(f"GANTrainer: GAN model layer 0 ({gen_layer_in_gan.name}) "
                                 f"trainable: {gen_layer_in_gan.trainable}, "
                                 f"num_vars: {len(gen_layer_in_gan.variables)}, "
                                 f"num_trainable_vars: {len(gen_layer_in_gan.trainable_variables)}")
            if len(self.gan_model.layers) > 1:
                disc_layer_in_gan = self.gan_model.layers[1]
                self.logger.info(f"GANTrainer: GAN model layer 1 ({disc_layer_in_gan.name}) "
                                 f"trainable: {disc_layer_in_gan.trainable}, "
                                 f"num_vars: {len(disc_layer_in_gan.variables)}, "
                                 f"num_trainable_vars: {len(disc_layer_in_gan.trainable_variables)}")
            
            generator_optimizer_config = {
                'learning_rate': self.params.get('generator_lr', 1e-4), # Corrected key
                'beta_1': self.params.get('generator_beta1', 0.5)     # Corrected key
            }
            self.logger.info(f"GANTrainer: Generator optimizer config for GAN: {generator_optimizer_config}")
            gan_optimizer = tf.keras.optimizers.Adam(**generator_optimizer_config)

            self.gan_model.compile(optimizer=gan_optimizer, loss='binary_crossentropy', metrics=['accuracy'])
            self.logger.info("GANTrainer: Combined GAN model built and compiled successfully.")
            
            # Log trainable variables of the compiled GAN model and its layers again
            self.logger.info(f"GANTrainer: Compiled GAN model ({self.gan_model.name}) "
                             f"trainable: {self.gan_model.trainable}, "
                             f"num_trainable_vars: {len(self.gan_model.trainable_variables)}")
            if len(self.gan_model.layers) > 0:
                self.logger.info(f"GANTrainer: After GAN compilation, gan_model.layers[0] ({self.gan_model.layers[0].name}) "
                                 f"trainable: {self.gan_model.layers[0].trainable}, "
                                 f"num_trainable_vars: {len(self.gan_model.layers[0].trainable_variables)}")
            # self.gan_model.summary(print_fn=self.logger.info)

        except Exception as e:
            self.logger.error(f"GANTrainer: Failed to build combined GAN model: {e}", exc_info=True)
            self.gan_model = None
            return False
            
        return True

    def train(self, training_data: pd.DataFrame, epochs: Optional[int] = None, batch_size: Optional[int] = None):
        self.logger.info("GANTrainerPlugin: Starting GAN training process...")

        if self.generator_plugin is None or self.discriminator_plugin is None or self.feeder_plugin is None:
            self.logger.error("GANTrainerPlugin: One or more required plugins (Generator, Discriminator, Feeder) are not set.")
            raise RuntimeError("Required plugins not available for training.")

        # Build models if not already built
        if self.generator_model is None or self.discriminator_model is None or self.gan_model is None:
            self.logger.info("GANTrainerPlugin: Models not built or cleared, attempting to build...")
            if not self._build_models():
                self.logger.error("GANTrainerPlugin: Failed to build GAN models. Aborting training.")
                raise RuntimeError("Failed to build GAN models")
        
        # Ensure models are available after attempting to build
        if self.generator_model is None:
            self.logger.error("GANTrainer: Generator model is MISSING after build attempt.")
            raise RuntimeError("Generator model is MISSING.")
        if self.discriminator_model is None:
            self.logger.error("GANTrainer: Discriminator model is MISSING after build attempt.")
            raise RuntimeError("Discriminator model is MISSING.")
        if self.gan_model is None:
            self.logger.error("GANTrainer: Combined GAN model is MISSING after build attempt.")
            raise RuntimeError("Combined GAN model is MISSING.")

        self.logger.info("GANTrainerPlugin: All models are available. Proceeding with TrainingCoordinator.")
        
        current_epochs = epochs if epochs is not None else self.params.get('epochs', 100)
        current_batch_size = batch_size if batch_size is not None else self.params.get('batch_size', 32)

        self.params['epochs'] = current_epochs
        self.params['batch_size'] = current_batch_size
        
        if hasattr(self.training_coordinator, 'set_params'):
             self.training_coordinator.set_params(**self.params)

        try:
            # Ensure models are set for the ReduceLROnPlateau callbacks initialized in __init__
            if self.lr_scheduler_g and self.generator_model:
                self.lr_scheduler_g.set_model(self.generator_model)
                self.logger.info(f"Set generator_model for self.lr_scheduler_g. Monitored metric: {self.lr_scheduler_g.monitor}")
            if self.lr_scheduler_d and self.discriminator_model:
                self.lr_scheduler_d.set_model(self.discriminator_model)
                self.logger.info(f"Set discriminator_model for self.lr_scheduler_d. Monitored metric: {self.lr_scheduler_d.monitor}")

            # Set model for EarlyStopping callback if it exists
            if self.early_stopping_callback and self.generator_model: # Assuming generator_model is primary for ES
                self.early_stopping_callback.set_model(self.generator_model) 
                self.logger.info(f"Set generator_model for self.early_stopping_callback. Monitored metric: {self.early_stopping_callback.monitor}")

            history = self.training_coordinator.train(
                training_data=training_data,
                generator=self.generator_model,
                discriminator=self.discriminator_model,
                gan_model=self.gan_model,
                feeder_plugin=self.feeder_plugin,
                lr_scheduler_g=self.lr_scheduler_g, # Pass LR schedulers
                lr_scheduler_d=self.lr_scheduler_d, # Pass LR schedulers
                early_stopping_callback=self.early_stopping_callback # Pass ES callback
            )
            self.logger.info("GANTrainerPlugin: Training completed by TrainingCoordinator.")
            return history
        except Exception as e:
            self.logger.error(f"GANTrainerPlugin: Error during TrainingCoordinator.train: {e}", exc_info=True)
            raise RuntimeError(f"GAN training failed: {e}")

    def get_debug_info(self) -> Dict[str, Any]:
        """Return debug information for the plugin."""
        debug_info = {
            "plugin_name": self.__class__.__name__,
            "params": self.params,
            "main_config_keys": list(self.main_config.keys()) if hasattr(self, 'main_config') else [],
            "generator_plugin_set": self.generator_plugin is not None,
            "discriminator_plugin_set": self.discriminator_plugin is not None,
            "feeder_plugin_set": self.feeder_plugin is not None,
            "generator_model_built": self.generator_model is not None,
            "discriminator_model_built": self.discriminator_model is not None,
            "gan_model_built": self.gan_model is not None,
        }
        if self.generator_model:
            debug_info["generator_model_summary"] = []
            self.generator_model.summary(print_fn=lambda x: debug_info["generator_model_summary"].append(x))
        if self.discriminator_model:
            debug_info["discriminator_model_summary"] = []
            self.discriminator_model.summary(print_fn=lambda x: debug_info["discriminator_model_summary"].append(x))
        if self.gan_model:
            debug_info["gan_model_summary"] = []
            self.gan_model.summary(print_fn=lambda x: debug_info["gan_model_summary"].append(x))
        return debug_info

    def add_debug_info(self, key: str, value: Any) -> None:
        """Add custom debug information (not standard)."""
        if not hasattr(self, '_custom_debug_info'):
            self._custom_debug_info = {}
        self._custom_debug_info[key] = value
