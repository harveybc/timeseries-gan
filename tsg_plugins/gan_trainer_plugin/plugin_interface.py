#!/usr/bin/env python3
"""
plugin_interface.py

Plugin Interface Manager for GAN Trainer Plugin.
Handles interactions with generator, feeder, and preprocessor plugins.

Part of the extreme separation of concerns approach.
Each module is focused and under 200 lines.

Author: TimeSeries-GAN Team
"""

import logging
from typing import Any, Dict, Optional, Tuple

# Assuming Keras Model type
from tensorflow.keras.models import Model


class PluginInterface:
    """Handles interactions with other plugins (Generator, Feeder, Discriminator)."""
    
    def __init__(self, params: Dict[str, Any], logger: logging.Logger):
        """
        Initialize plugin interface manager.
        
        Args:
            params: Plugin parameters dictionary
            logger: Logger instance for this module
        """
        self.params = params
        self.logger = logger
        self.generator_plugin: Optional[Any] = None
        self.feeder_plugin: Optional[Any] = None
        self.discriminator_plugin: Optional[Any] = None # ADDED
        self.preprocessor_plugin: Optional[Any] = None
        self.logger.info("PluginInterface initialized")

    def set_plugin_instances(self, generator_plugin, feeder_plugin, 
                             discriminator_plugin, # ADDED
                             preprocessor_plugin):
        """
        Set plugin instances for interaction.
        
        Args:
            generator_plugin: Generator plugin instance
            feeder_plugin: Feeder plugin instance
            discriminator_plugin: Discriminator plugin instance # ADDED
            preprocessor_plugin: Preprocessor plugin instance
        """
        self.generator_plugin = generator_plugin
        self.feeder_plugin = feeder_plugin
        self.discriminator_plugin = discriminator_plugin # STORE IT
        self.preprocessor_plugin = preprocessor_plugin
        self.logger.info("Plugin instances set in PluginInterface.")
        if not self.generator_plugin:
            self.logger.warning("GeneratorPlugin instance not provided to PluginInterface.")
        if not self.feeder_plugin:
            self.logger.warning("FeederPlugin instance not provided to PluginInterface.")
        if not self.discriminator_plugin:
            self.logger.warning("DiscriminatorPlugin instance not provided to PluginInterface.")


    def get_generator_model(self) -> Optional[Model]:
        """
        Get the extracted generator model.
        """
        if self.generator_plugin and hasattr(self.generator_plugin, 'get_model'):
            model = self.generator_plugin.get_model()
            if model:
                self.logger.info("Retrieved generator model from GeneratorPlugin.")
                return model
            else:
                self.logger.warning("GeneratorPlugin's get_model() returned None.")
                return None
        self.logger.warning("GeneratorPlugin not set or has no get_model method in PluginInterface.")
        return None

    def get_discriminator_model(self) -> Optional[Model]: # ADDED METHOD
        """
        Get the extracted discriminator model.
        """
        if self.discriminator_plugin and hasattr(self.discriminator_plugin, 'get_model'):
            model = self.discriminator_plugin.get_model()
            if model:
                self.logger.info("Retrieved discriminator model from DiscriminatorPlugin.")
                return model
            else:
                self.logger.warning("DiscriminatorPlugin's get_model() returned None.")
                return None
        self.logger.warning("DiscriminatorPlugin not set or has no get_model method in PluginInterface.")
        return None

    def get_feeder_plugin(self) -> Optional[Any]:
        """
        Get the feeder plugin instance.
        """
        if self.feeder_plugin:
            return self.feeder_plugin
        self.logger.warning("FeederPlugin not set in PluginInterface.")
        return None

    def get_extracted_parameters(self) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        """
        Extracts sequence length, latent dim, and num features from generator/config.
        """
        seq_len = self.params.get("seq_len", self.params.get("generator_decoder_input_window_size"))
        
        # Latent dim for Z-generator output, or a general latent_dim if defined
        latent_dim = self.params.get("internal_z_latent_dim", self.params.get("latent_dim")) 
        if isinstance(self.params.get("latent_shape"), list) and len(self.params["latent_shape"]) == 2:
             # If latent_shape is [seq, dim], prefer its dim part
            latent_dim = self.params["latent_shape"][1]

        # Num features should be the output of the generator (e.g., 57)
        # This is what the discriminator will see.
        num_features = None
        generator_model = self.get_generator_model()
        if generator_model and hasattr(generator_model, 'output_shape'):
            if isinstance(generator_model.output_shape, tuple) and len(generator_model.output_shape) > 0 :
                 # For (None, 57) output -> 57
                 # For (None, seq_len, 57) output -> 57
                num_features = generator_model.output_shape[-1]
            else: # Try from config if model output shape is not as expected
                num_features = len(self.params.get("generator_full_feature_names_ordered", [])) \
                               if self.params.get("generator_full_feature_names_ordered") else \
                               len(self.params.get("feature_names_for_discriminator_ordered", []))
        else: # Fallback to config
            num_features = len(self.params.get("generator_full_feature_names_ordered", [])) \
                           if self.params.get("generator_full_feature_names_ordered") else \
                           len(self.params.get("feature_names_for_discriminator_ordered", []))


        if not num_features and self.params.get("num_features"): # Discriminator's own num_features param
            num_features = self.params.get("num_features")


        self.logger.info(f"Extracted parameters for model building: seq_len={seq_len}, latent_dim={latent_dim}, num_features={num_features}")
        return seq_len, latent_dim, num_features

    def get_debug_info(self) -> Dict[str, Any]:
        """
        Get debug information for this module.
        
        Returns:
            Dict containing debug information
        """
        return {
            "generator_plugin_available": self.generator_plugin is not None,
            "feeder_plugin_available": self.feeder_plugin is not None,
            "discriminator_plugin_available": self.discriminator_plugin is not None,
            "preprocessor_plugin_available": self.preprocessor_plugin is not None,
        }

# Ensure you create this file if it doesn't exist, or integrate its logic.
# If `plugin_interface.py` is new, add `from .plugin_interface import PluginInterface`
# to the top of `tsg_plugins/gan_trainer_plugin/gan_trainer_plugin.py`.
