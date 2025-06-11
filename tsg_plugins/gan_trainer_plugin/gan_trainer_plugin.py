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

from app.utils.logging_utils import get_logger # Corrected import
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
        self.main_config = config.copy() # Store the global config
        
        # self.params will be populated by _initialize_parameters
        self.params: Dict[str, Any] = {} 
        self._initialize_parameters() # Populate self.params based on main_config and plugin_params

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
            main_config=self.main_config, # Pass main_config
            plugin_params=self.params, 
            generator_plugin=self.generator_plugin, 
            discriminator_plugin=self.discriminator_plugin,
            feeder_plugin=self.feeder_plugin,
            device=None # Or determine device if needed
        )
        self.logger.info("GANTrainerPlugin initialized.")

        self.generator_l2_reg = self.params.get("generator_l2_reg") # This should now be correctly sourced

        # Initialize ReduceLROnPlateau callbacks
        if self.params.get("enable_reduce_lr_on_plateau"):
            self.lr_scheduler_g = ReduceLROnPlateau(
                monitor=self.params.get("lr_monitor_metric_g", "g_loss"),
                factor=self.params.get("lr_reduction_factor"), # Will be from resolved self.params
                patience=self.params.get("lr_patience"),       # Will be from resolved self.params
                verbose=1,
                min_delta=self.params.get("lr_min_delta"),     # Will be from resolved self.params
                min_lr=self.params.get("min_lr_g")             # Will be from resolved self.params
            )
            self.lr_scheduler_d = ReduceLROnPlateau(
                monitor=self.params.get("lr_monitor_metric_d", "d_loss"),
                factor=self.params.get("lr_reduction_factor"), # Will be from resolved self.params
                patience=self.params.get("lr_patience"),       # Will be from resolved self.params
                verbose=1,
                min_delta=self.params.get("lr_min_delta"),     # Will be from resolved self.params
                min_lr=self.params.get("min_lr_d")             # Will be from resolved self.params
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
        """
        Initializes self.params by merging main_config with plugin_params,
        ensuring correct precedence for all relevant training parameters.
        Precedence:
        1. main_config['trainer_<param>']
        2. main_config['<param>']
        3. main_config['<general_equivalent_param>'] (e.g., 'learning_rate' for 'generator_lr')
        4. GANTrainerPlugin.plugin_params['<param>'] (class default)
        5. Keras/hardcoded defaults (applied at point of use if still not found)
        """
        self.logger.debug("GANTrainerPlugin: Starting _initialize_parameters.")
        resolved_params: Dict[str, Any] = {}

        # Define parameter groups and their mappings
        # target_key: (general_config_key, plugin_default_key_in_plugin_params)
        param_mapping = {
            # Optimizer params - Generator
            "generator_lr": ("learning_rate", "generator_lr"),
            "generator_beta1": ("beta1", "generator_beta1"),
            "generator_beta2": ("beta2", None), # No default in plugin_params, general 'beta2' or Keras default
            "generator_epsilon": ("epsilon", None),
            "generator_amsgrad": ("amsgrad", None),
            # Optimizer params - Discriminator
            "discriminator_lr": ("learning_rate", "discriminator_lr"),
            "discriminator_beta1": ("beta1", "discriminator_beta1"),
            "discriminator_beta2": ("beta2", None),
            "discriminator_epsilon": ("epsilon", None),
            "discriminator_amsgrad": ("amsgrad", None),
            # Training loop
            "gan_epochs": ("gan_epochs", "gan_epochs"), # General key is same as specific
            "gan_batch_size": ("gan_batch_size", "gan_batch_size"),
            # Model specific
            "generator_l2_reg": ("generator_l2_reg", "generator_l2_reg"),
            # Callbacks / Other - Add all keys from plugin_params here to ensure they are processed
        }

        # Ensure all keys from plugin_params are in the mapping to be processed
        for p_key in self.plugin_params.keys():
            if p_key not in param_mapping:
                # For params like 'es_patience', general key is itself, plugin default key is itself
                param_mapping[p_key] = (p_key, p_key)
        
        main_cfg = self.main_config if hasattr(self, 'main_config') and self.main_config else {}
        plugin_defaults = self.plugin_params

        for target_key, (general_key, plugin_default_key) in param_mapping.items():
            value = None
            found_source = ""

            # 1. Check 'trainer_<target_key>' in main_config
            trainer_specific_key = f"trainer_{target_key}"
            if trainer_specific_key in main_cfg:
                value = main_cfg[trainer_specific_key]
                found_source = f"main_config['{trainer_specific_key}']"
            
            # 2. Check '<target_key>' in main_config
            if value is None and target_key in main_cfg:
                value = main_cfg[target_key]
                found_source = f"main_config['{target_key}']"
            
            # 3. Check '<general_key>' in main_config (if different from target_key)
            if value is None and general_key and general_key != target_key and general_key in main_cfg:
                value = main_cfg[general_key]
                found_source = f"main_config['{general_key}'] (for {target_key})"

            # 4. Use plugin_params default if plugin_default_key is defined
            if value is None and plugin_default_key and plugin_default_key in plugin_defaults:
                value = plugin_defaults[plugin_default_key]
                found_source = f"plugin_params['{plugin_default_key}']"
            
            if value is not None:
                resolved_params[target_key] = value
                self.logger.debug(f"Resolved param '{target_key}': {value} (from {found_source})")
            else:
                self.logger.debug(f"Param '{target_key}' not resolved by specific rules, will rely on usage-site defaults if any.")

        # Add any other general keys from main_config that weren't specifically mapped/processed
        # and are not prefixed for other plugins.
        for key, m_value in main_cfg.items():
            if key not in resolved_params:
                # Check if it's a general key or a trainer_ prefixed key not yet handled
                is_general = not any(key.startswith(p) for p in ["generator_", "discriminator_", "feeder_"])
                is_unhandled_trainer_key = key.startswith("trainer_") and key[len("trainer_"):] not in resolved_params
                
                if is_general or is_unhandled_trainer_key:
                    actual_key_to_set = key[len("trainer_"):] if is_unhandled_trainer_key else key
                    if actual_key_to_set not in resolved_params: # Avoid double setting
                       resolved_params[actual_key_to_set] = m_value
                       self.logger.debug(f"Added general/unhandled trainer param '{actual_key_to_set}': {m_value} from main_config")
        
        self.params = resolved_params
        self.logger.info(f"GANTrainerPlugin params fully initialized. Example - generator_lr: {self.params.get('generator_lr')}, gan_epochs: {self.params.get('gan_epochs')}")

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
            
            # Keras Adam defaults for fallback if a param is entirely missing
            keras_adam_defaults = {"beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-7, "amsgrad": False}
            default_lr = 1e-4 # Final fallback LR for optimizers if not found in params

            generator_optimizer_config = {
                'learning_rate': self.params.get('generator_lr', default_lr),
                'beta_1': self.params.get('generator_beta1', keras_adam_defaults['beta_1']),
                'beta_2': self.params.get('generator_beta2', keras_adam_defaults['beta_2']),
                'epsilon': self.params.get('generator_epsilon', keras_adam_defaults['epsilon']),
                'amsgrad': self.params.get('generator_amsgrad', keras_adam_defaults['amsgrad'])
            }
            self.logger.info(f"GANTrainer: Compiling GAN model with Generator optimizer config: {generator_optimizer_config}")
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
        
        # Prepare callbacks list
        callbacks_list = []
        if self.lr_scheduler_g:
            callbacks_list.append(self.lr_scheduler_g)
        if self.lr_scheduler_d:
            callbacks_list.append(self.lr_scheduler_d)
        if self.early_stopping_callback:
            callbacks_list.append(self.early_stopping_callback)
        
        self.logger.info(f"Passing {len(callbacks_list)} callbacks to TrainingCoordinator: {[type(cb).__name__ for cb in callbacks_list]}")

        # Prepare directories
        models_dir = os.path.join(self.params.get("results_base_dir", "results"), self.params.get("save_model_dir", "models"))
        plots_dir = os.path.join(self.params.get("results_base_dir", "results"), self.params.get("save_plot_dir", "plots"))
        metrics_dir = os.path.join(self.params.get("results_base_dir", "results"), self.params.get("save_metrics_dir", "metrics"))
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)
        os.makedirs(metrics_dir, exist_ok=True)

        # Delegate to TrainingCoordinator
        self.logger.info("Calling TrainingCoordinator.train()...")
        history = self.training_coordinator.train(
            gan_model=self.gan_model,
            generator=self.generator_model,
            discriminator=self.discriminator_model,
            dataset=training_data, # This is the pd.DataFrame, FeederPlugin will handle conversion
            epochs=current_epochs,
            batch_size=current_batch_size,
            callbacks=callbacks_list # Pass the compiled list of callbacks
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
