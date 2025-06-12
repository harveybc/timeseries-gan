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
        "gan_epochs": 10000, "gan_batch_size": 32,
        # Default learning rates, to be overridden by main_config's 'learning_rate',
        # 'generator_lr', or 'discriminator_lr'
        "generator_lr": 1e-4, "discriminator_lr": 1e-4,
        "generator_beta1": 0.5, "discriminator_beta1": 0.5, "gan_save_interval": 500,
        
        # Discriminator architecture
        "discriminator_conv_filters": [64, 128], "discriminator_conv_kernel_size": 3,
        "discriminator_lstm_units": 64, "discriminator_dropout_rate": 0.3,
        
        # Learning rate scheduling - these keys MUST exist in the final config.
        "enable_reduce_lr_on_plateau": True, 
        "lr_reduction_factor": 0.1, # Default from config.py
        "lr_patience": 10,          # Default from config.py
        "lr_min_delta": 0.0001,     # Default from config.py
        "min_lr_g": 1e-7,           # Default from config.py
        "min_lr_d": 1e-7,           # Default from config.py
        "lr_monitor_metric_g": "g_loss", # Default from config.py
        "lr_monitor_metric_d": "d_loss", # Default from config.py
        
        # Early stopping - these keys MUST exist in the final config.
        "enable_early_stopping": True, 
        "es_patience": 50, # More aggressive default for plugin_params, config.py has 10
        "es_min_delta": 0.001, 
        "es_monitor_metric": "g_loss",
        "es_restore_best_weights": False, # Added default
        
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
        self.main_config = config.copy() # Store main_config first
        self.params = {} # Initialize before _initialize_parameters

        # _initialize_parameters will populate self.params from main_config and plugin_params defaults
        self._initialize_parameters() 

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

        # L2 reg is sourced via _initialize_parameters into self.params
        # self.generator_l2_reg = self.params.get("generator_l2_reg") # Already handled by _initialize_parameters

        # Initialize ReduceLROnPlateau callbacks
        # Parameters are now expected to be in self.params after _initialize_parameters
        if self.params["enable_reduce_lr_on_plateau"]: # Use direct access, will raise KeyError if missing
            self.lr_scheduler_g = ReduceLROnPlateau(
                monitor=self.params["lr_monitor_metric_g"],
                factor=self.params["lr_reduction_factor"],
                patience=self.params["lr_patience"],
                verbose=1,
                min_delta=self.params["lr_min_delta"],
                min_lr=self.params["min_lr_g"]
            )
            self.lr_scheduler_d = ReduceLROnPlateau(
                monitor=self.params["lr_monitor_metric_d"],
                factor=self.params["lr_reduction_factor"],
                patience=self.params["lr_patience"],
                verbose=1,
                min_delta=self.params["lr_min_delta"],
                min_lr=self.params["min_lr_d"]
            )
        else:
            self.lr_scheduler_g = None
            self.lr_scheduler_d = None
        
        # self.feeder_plugin = feeder_plugin # Already stored

        # Initialize EarlyStopping callback
        self.early_stopping_callback = None
        if self.params["enable_early_stopping"]: # Use direct access
            self.logger.info(f"Early stopping enabled. Metric: {self.params['es_monitor_metric']}, Patience: {self.params['es_patience']}")
            self.early_stopping_callback = EarlyStopping(
                monitor=self.params["es_monitor_metric"],
                min_delta=self.params["es_min_delta"],
                patience=self.params["es_patience"],
                verbose=1,
                mode='min', 
                restore_best_weights=self.params["es_restore_best_weights"]
            )
        else:
            self.logger.info("Early stopping is disabled by configuration.")

    def _initialize_parameters(self):
        self.logger.debug(f"GANTrainerPlugin: _initialize_parameters called.")
        
        # Start with plugin_params defaults
        self.params = self.plugin_params.copy()
        self.logger.debug(f"self.params initialized with plugin_params defaults. Keys: {list(self.params.keys())}")

        if hasattr(self, 'main_config') and self.main_config is not None:
            self.logger.debug(f"main_config available. Merging main_config into self.params.")
            
            # Override self.params with values from main_config
            # This ensures main_config (from config.py, CLI) takes precedence
            for key, value in self.main_config.items():
                self.params[key] = value # Direct override or addition
                if key in self.plugin_params:
                    self.logger.debug(f"Overrode/updated self.params['{key}'] with value from main_config: {value}")
                else:
                    self.logger.debug(f"Added new key self.params['{key}'] from main_config: {value}")

            # Handle specific learning rate logic:
            # If 'learning_rate' is in main_config, it sets the base for G and D LRs.
            # Individual 'generator_lr' or 'discriminator_lr' in main_config can override this.
            
            general_lr = self.main_config.get("learning_rate")

            if general_lr is not None:
                self.logger.info(f"Found 'learning_rate': {general_lr} in main_config. Setting as base for G/D LRs.")
                # Set 'generator_lr' from 'learning_rate' if 'generator_lr' isn't explicitly in main_config
                if "generator_lr" not in self.main_config:
                    self.params["generator_lr"] = general_lr
                    self.logger.info(f"Set self.params['generator_lr'] to general 'learning_rate': {general_lr}")
                # Set 'discriminator_lr' from 'learning_rate' if 'discriminator_lr' isn't explicitly in main_config
                if "discriminator_lr" not in self.main_config:
                    self.params["discriminator_lr"] = general_lr
                    self.logger.info(f"Set self.params['discriminator_lr'] to general 'learning_rate': {general_lr}")
            
            # Ensure trainer-prefixed parameters from main_config also override:
            # e.g. if main_config has "trainer_generator_lr", it should set self.params["generator_lr"]
            trainer_prefix = "trainer_"
            for key, value in self.main_config.items():
                if key.startswith(trainer_prefix):
                    param_key = key[len(trainer_prefix):]
                    # Check if this unprefixed key is a known parameter (either from plugin_params or already set from non-prefixed main_config)
                    if param_key in self.params: 
                        self.params[param_key] = value
                        self.logger.debug(f"Overrode self.params['{param_key}'] with value from prefixed key '{key}' in main_config: {value}")
                    # If param_key was not in plugin_params, it might be a new param introduced by main_config.
                    # This case is already handled by the general loop above if the key `param_key` itself was in main_config.
                    # This specific block is for `trainer_` prefixed versions of existing params.
        else:
            self.logger.debug("main_config not available. self.params relies solely on plugin_params defaults.")
        
        self.logger.info(f"GANTrainerPlugin params fully initialized. Final self.params keys: {list(self.params.keys())}")
        self.logger.info(f"Final resolved generator_lr: {self.params.get('generator_lr')}, discriminator_lr: {self.params.get('discriminator_lr')}")
        self.logger.info(f"Final resolved lr_patience: {self.params.get('lr_patience')}")

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
                'learning_rate': self.params["generator_lr"], # Direct access, expect key to exist
                'beta_1': self.params["generator_beta1"]      # Direct access
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

    def train(self, training_data: pd.DataFrame, epochs: Optional[int] = None, batch_size: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        """
        Train the GAN model.
        """
        self.logger.info(f"GANTrainerPlugin.train: Entered. Argument epochs: {epochs}, Argument batch_size: {batch_size}")
        self.logger.info(f"GANTrainerPlugin.train: self.params.get('gan_epochs') before resolving current_epochs: {self.params.get('gan_epochs')}")
        self.logger.info(f"GANTrainerPlugin.train: self.plugin_params.get('gan_epochs') for fallback: {self.plugin_params.get('gan_epochs')}")
        self.logger.info(f"GANTrainerPlugin.train: self.params.get('gan_batch_size') before resolving current_batch_size: {self.params.get('gan_batch_size')}")
        self.logger.info(f"GANTrainerPlugin.train: self.plugin_params.get('gan_batch_size') for fallback: {self.plugin_params.get('gan_batch_size')}")

        if not self._build_models():
            self.logger.error("Failed to build models. Aborting training.")
            return {}

        # Determine epochs and batch_size, prioritizing direct arguments, then self.params, then plugin_params defaults
        current_epochs = epochs
        if current_epochs is None:
            self.logger.debug("GANTrainerPlugin.train: Argument 'epochs' is None. Trying self.params.get('gan_epochs').")
            current_epochs = self.params.get('gan_epochs') 
            if current_epochs is None: 
                self.logger.debug("GANTrainerPlugin.train: self.params.get('gan_epochs') is None. Falling back to self.plugin_params['gan_epochs'].")
                current_epochs = self.plugin_params['gan_epochs']
        
        current_batch_size = batch_size
        if current_batch_size is None:
            self.logger.debug("GANTrainerPlugin.train: Argument 'batch_size' is None. Trying self.params.get('gan_batch_size').")
            current_batch_size = self.params.get('gan_batch_size')
            if current_batch_size is None:
                self.logger.debug("GANTrainerPlugin.train: self.params.get('gan_batch_size') is None. Falling back to self.plugin_params['gan_batch_size'].")
                current_batch_size = self.plugin_params['gan_batch_size']

        self.logger.info(f"GANTrainerPlugin.train: Resolved current_epochs to: {current_epochs}, current_batch_size to: {current_batch_size}")

        # Ensure models are compiled and LR schedulers are associated
        # This needs to happen after models are built but before training starts.
        # The optimizers are created in TrainingCoordinator._setup_optimizers()
        # The LR schedulers need their .model attribute set to the respective models.
        if self.lr_scheduler_g and self.gan_model: # Use gan_model for generator's LR scheduler
            self.lr_scheduler_g.set_model(self.gan_model) 
            self.logger.info(f"Linked lr_scheduler_g to gan_model: {self.gan_model.name}")
        else:
            if not self.lr_scheduler_g:
                self.logger.info("lr_scheduler_g is None, not linking.")
            if not self.gan_model:
                self.logger.warning("self.gan_model is None, cannot link lr_scheduler_g.")

        if self.lr_scheduler_d and self.discriminator_model:
            self.lr_scheduler_d.set_model(self.discriminator_model)
            self.logger.info(f"Linked lr_scheduler_d to discriminator_model: {self.discriminator_model.name}")
        else:
            if not self.lr_scheduler_d:
                self.logger.info("lr_scheduler_d is None, not linking.")
            if not self.discriminator_model:
                self.logger.warning("self.discriminator_model is None, cannot link lr_scheduler_d.")

        # Prepare directories
        models_dir = os.path.join(self.params.get("results_base_dir", "results"), self.params.get("save_model_dir", "models"))
        plots_dir = os.path.join(self.params.get("results_base_dir", "results"), self.params.get("save_plot_dir", "plots"))
        metrics_dir = os.path.join(self.params.get("results_base_dir", "results"), self.params.get("save_metrics_dir", "metrics"))
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)
        os.makedirs(metrics_dir, exist_ok=True)
        self.logger.info(f"Models will be saved to: {models_dir}")

        # Pass the actual model instances to the training coordinator
        history = self.training_coordinator.train(
            generator=self.generator_model,
            discriminator=self.discriminator_model,
            gan_model=self.gan_model, # Pass the combined GAN model
            feeder_plugin=self.feeder_plugin, # Pass feeder_plugin
            training_data=training_data,
            epochs=current_epochs,
            batch_size=current_batch_size,
            save_interval=self.params.get("gan_save_interval", 500),
            models_dir=models_dir, # Pass the constructed models_dir
            plots_dir=plots_dir,
            metrics_dir=metrics_dir,
            lr_scheduler_g=self.lr_scheduler_g, # Pass the LR scheduler for G
            lr_scheduler_d=self.lr_scheduler_d, # Pass the LR scheduler for D
            early_stopping_callback=self.early_stopping_callback # Pass the EarlyStopping callback
        )
        return history

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
