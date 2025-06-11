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
        
        # Initialize optimizer instances
        self.generator_optimizer_instance: Optional[tf.keras.optimizers.Optimizer] = None
        self.discriminator_optimizer_instance: Optional[tf.keras.optimizers.Optimizer] = None

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
            # Store the generator optimizer instance
            self.generator_optimizer_instance = tf.keras.optimizers.Adam(**generator_optimizer_config)

            self.gan_model.compile(optimizer=self.generator_optimizer_instance, loss='binary_crossentropy', metrics=['accuracy'])
            self.logger.info("GANTrainer: Combined GAN model built and compiled successfully.")

            # Create and compile discriminator optimizer
            discriminator_optimizer_config = {
                'learning_rate': self.params.get('discriminator_lr', default_lr),
                'beta_1': self.params.get('discriminator_beta1', keras_adam_defaults['beta_1']),
                'beta_2': self.params.get('discriminator_beta2', keras_adam_defaults['beta_2']),
                'epsilon': self.params.get('discriminator_epsilon', keras_adam_defaults['epsilon']),
                'amsgrad': self.params.get('discriminator_amsgrad', keras_adam_defaults['amsgrad'])
            }
            self.logger.info(f"GANTrainer: Compiling Discriminator model with Discriminator optimizer config: {discriminator_optimizer_config}")
            self.discriminator_optimizer_instance = tf.keras.optimizers.Adam(**discriminator_optimizer_config)
            
            if self.discriminator_model:
                self.discriminator_model.compile(optimizer=self.discriminator_optimizer_instance, loss='binary_crossentropy', metrics=['accuracy'])
                self.logger.info("GANTrainer: Discriminator model compiled successfully with its own optimizer.")
            else:
                self.logger.error("GANTrainer: Discriminator model is None, cannot compile.")
                return False

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

    def train(self, training_data: pd.DataFrame, feeder_plugin, generator_plugin, discriminator_plugin, epochs: Optional[int] = None, batch_size: Optional[int] = None):
        self.logger.info(f"GANTrainerPlugin.train: Entered. Argument epochs: {epochs}, Argument batch_size: {batch_size}")

        current_epochs = epochs if epochs is not None else self.params.get('gan_epochs', self.plugin_params.get('gan_epochs'))
        current_batch_size = batch_size if batch_size is not None else self.params.get('gan_batch_size', self.plugin_params.get('gan_batch_size'))
        self.logger.info(f"GANTrainerPlugin.train: self.params.get('gan_epochs') before resolving current_epochs: {self.params.get('gan_epochs')}")
        self.logger.info(f"GANTrainerPlugin.train: self.plugin_params.get('gan_epochs') for fallback: {self.plugin_params.get('gan_epochs')}")
        self.logger.info(f"GANTrainerPlugin.train: self.params.get('gan_batch_size') before resolving current_batch_size: {self.params.get('gan_batch_size')}")
        self.logger.info(f"GANTrainerPlugin.train: self.plugin_params.get('gan_batch_size') for fallback: {self.plugin_params.get('gan_batch_size')}")

        self.generator_plugin = generator_plugin
        self.discriminator_plugin = discriminator_plugin
        self.feeder_plugin = feeder_plugin # Ensure feeder_plugin is stored on self if needed by other methods or for clarity

        self._build_models() # Builds combined_gan_model, generator_model, discriminator_model

        self.logger.info(f"GANTrainerPlugin.train: Original training_data shape: {training_data.shape if training_data is not None else 'None'}")

        if training_data is None:
            self.logger.error("GANTrainerPlugin.train: training_data is None. Cannot proceed.")
            raise ValueError("training_data for GANTrainerPlugin.train cannot be None.")

        # Use FeederPlugin to create sequences
        # Parameter 'seq_len' is expected to be in feeder_plugin.params, sourced from the global config.
        sequence_length = self.feeder_plugin.params.get("seq_len") 
        if not sequence_length:
            self.logger.error("FeederPlugin parameter 'seq_len' (expected from global config) is not set in feeder_plugin.params.")
            # Log available keys in feeder_plugin.params for debugging
            self.logger.debug(f"Available keys in feeder_plugin.params: {list(self.feeder_plugin.params.keys())}")
            raise ValueError("FeederPlugin parameter 'seq_len' is not set in its params. Check config propagation.")

        selected_features = self.feeder_plugin.params.get("selected_features")
        if selected_features:
            self.logger.info(f"Using 'selected_features' from FeederPlugin: {selected_features}")
            missing_cols = [col for col in selected_features if col not in training_data.columns]
            if missing_cols:
                self.logger.error(f"Selected features missing from training_data: {missing_cols}. Available columns: {training_data.columns.tolist()}")
                raise ValueError(f"Selected features {missing_cols} not found in training_data columns.")
            data_for_sequencing = training_data[selected_features]
        else:
            self.logger.warning("'selected_features' not found in FeederPlugin params. Using all numeric columns from training_data for sequencing.")
            data_for_sequencing = training_data.select_dtypes(include=np.number)
            if data_for_sequencing.shape[1] == 0:
                self.logger.error("No numeric columns found in training_data to use for sequencing.")
                raise ValueError("No numeric columns in training_data for sequencing.")
            self.logger.info(f"Using numeric columns for sequencing: {data_for_sequencing.columns.tolist()}")

        # Ensure data_for_sequencing is purely numeric
        non_numeric_cols = [col for col in data_for_sequencing.columns if not pd.api.types.is_numeric_dtype(data_for_sequencing[col])]
        if non_numeric_cols:
            for col_name in non_numeric_cols:
                self.logger.error(f"Column '{col_name}' in data_for_sequencing is non-numeric (dtype: {data_for_sequencing[col_name].dtype}).")
            raise ValueError(f"Non-numeric data found in columns {non_numeric_cols} intended for sequencing. Please ensure all selected features are numeric.")
        
        self.logger.info(f"Data for FeederPlugin.create_sequences has shape: {data_for_sequencing.shape} and columns: {data_for_sequencing.columns.tolist()}")

        try:
            # Ensure the call matches the definition: create_sequences(self, data: np.ndarray, seq_len: int)
            real_sequences_np = self.feeder_plugin.create_sequences(
                data=data_for_sequencing.to_numpy(), # Pass numpy array
                seq_len=sequence_length
            )
            # The method create_sequences is expected to return only one value (the sequences)
            # If it previously returned a tuple like (sequences, None), adjust accordingly.
            # Assuming it now returns just the sequences based on the simplified definition.

        except Exception as e:
            self.logger.error(f"Error during FeederPlugin.create_sequences: {e}", exc_info=True)
            raise

        if real_sequences_np is None or real_sequences_np.ndim != 3:
            self.logger.error(f"FeederPlugin.create_sequences did not return valid 3D sequence data. Shape: {real_sequences_np.shape if real_sequences_np is not None else 'None'}")
            raise ValueError("Failed to create valid sequences from FeederPlugin.")

        self.logger.info(f"GANTrainerPlugin.train: training_data processed by FeederPlugin.create_sequences. Shape of real_sequences_np: {real_sequences_np.shape}")
        
        actual_num_features = real_sequences_np.shape[2]
        self.logger.info(f"Actual number of features in sequences: {actual_num_features}")

        # Log warnings if configured model features differ from actual data features
        # Note: Models are already built. A more robust solution would reconfigure/rebuild models if num_features mismatches.
        if self.generator_plugin and hasattr(self.generator_plugin, 'params'):
            gen_conf_features = self.generator_plugin.params.get('num_features')
            if gen_conf_features != actual_num_features:
                self.logger.warning(
                    f"Generator configured num_features ({gen_conf_features}) "
                    f"differs from actual data features ({actual_num_features}). "
                    f"Model might behave unexpectedly or error out if input shapes mismatch."
                )
        
        if self.discriminator_plugin and hasattr(self.discriminator_plugin, 'params'):
            disc_conf_features = self.discriminator_plugin.params.get('num_features')
            if disc_conf_features != actual_num_features:
                self.logger.warning(
                    f"Discriminator configured num_features ({disc_conf_features}) "
                    f"differs from actual data features ({actual_num_features}). "
                    f"Model might behave unexpectedly or error out if input shapes mismatch."
                )

        # Initialize and assign optimizers to TrainingCoordinator
        self.training_coordinator.g_optimizer = self.generator_optimizer_instance
        self.training_coordinator.d_optimizer = self.discriminator_optimizer_instance
        self.logger.info("GANTrainerPlugin.train: Assigned optimizer instances to TrainingCoordinator.")

        # Callbacks setup
        callbacks_list = []
        if self.lr_scheduler_g:
            self.lr_scheduler_g.set_model(self.gan_model) # Changed from self.combined_gan_model
            callbacks_list.append(self.lr_scheduler_g)
            self.logger.info(f"Linked lr_scheduler_g to gan_model: {self.gan_model.name}") # Changed from self.combined_gan_model
        if self.lr_scheduler_d:
            self.lr_scheduler_d.set_model(self.discriminator_model) # Link scheduler to Discriminator model
            callbacks_list.append(self.lr_scheduler_d)
            self.logger.info(f"Linked lr_scheduler_d to discriminator_model: {self.discriminator_model.name}")
        if self.early_stopping:
            # Early stopping should monitor a metric from the GAN's training, e.g., g_loss or a validation metric if available
            # Ensure its model is set if it needs one, though typically it's set by Keras fit/evaluate
            # For custom loop, it's manually checked. TrainingCoordinator handles this.
            callbacks_list.append(self.early_stopping)

        self.logger.info(f"Passing {len(callbacks_list)} callbacks to TrainingCoordinator: {[cb.__class__.__name__ for cb in callbacks_list]}")
        self.logger.info(f"GANTrainerPlugin.train: Resolved current_epochs to: {current_epochs}, current_batch_size to: {current_batch_size}")
        self.logger.info("Calling TrainingCoordinator.train()...")
        
        history = self.training_coordinator.train(
            gan_model=self.gan_model, # Changed from self.combined_gan_model
            generator=self.generator_model,
            discriminator=self.discriminator_model,
            dataset=real_sequences_np,  # Pass the NumPy array of sequences
            epochs=current_epochs,
            batch_size=current_batch_size,
            callbacks_list=callbacks_list
        )
        self.logger.info("GANTrainerPlugin.train: TrainingCoordinator.train() completed.")
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
