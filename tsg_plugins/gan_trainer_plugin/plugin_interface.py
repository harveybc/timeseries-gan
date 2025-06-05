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
from typing import Dict, Any, Optional, Tuple


class PluginInterface:
    """
    Manages interactions with other plugins for GAN training.
    
    Handles:
    - Generator plugin model extraction and validation
    - Feeder plugin data access and configuration
    - Preprocessor plugin integration
    - Parameter extraction from plugins
    """
    
    def __init__(self, params: Dict[str, Any], logger: logging.Logger):
        """
        Initialize plugin interface manager.
        
        Args:
            params: Plugin parameters dictionary
            logger: Logger instance for this module
        """
        self.params = params
        self.logger = logger
        
        # Plugin instances
        self.generator_plugin = None
        self.feeder_plugin = None
        self.preprocessor_plugin = None
        
        # Extracted models and parameters
        self.generator = None
        self.seq_len = None
        self.latent_dim = None
        self.num_discriminator_features = None
    
    def set_plugin_instances(self, generator_plugin: Optional[Any] = None,
                           feeder_plugin: Optional[Any] = None,
                           preprocessor_plugin: Optional[Any] = None):
        """
        Set plugin instances for interaction.
        
        Args:
            generator_plugin: Generator plugin instance
            feeder_plugin: Feeder plugin instance
            preprocessor_plugin: Preprocessor plugin instance
        """
        self.generator_plugin = generator_plugin
        self.feeder_plugin = feeder_plugin
        self.preprocessor_plugin = preprocessor_plugin
        
        self.logger.info("Plugin instances set for interface")
        
        # Extract generator model if available
        if self.generator_plugin:
            self._extract_generator_model()
    
    def _extract_generator_model(self):
        """Extract generator model from generator plugin."""
        self.generator = None
        
        if not self.generator_plugin:
            self.logger.warning("Generator plugin instance not provided")
            return
        
        # Try different attribute names for generator model
        if hasattr(self.generator_plugin, 'generator_model') and self.generator_plugin.generator_model:
            self.generator = self.generator_plugin.generator_model
            self.logger.info("Generator model retrieved from generator_plugin.generator_model")
        elif hasattr(self.generator_plugin, 'model') and self.generator_plugin.model:
            self.generator = self.generator_plugin.model
            self.logger.info("Generator model retrieved from generator_plugin.model")
        elif hasattr(self.generator_plugin, 'get_model') and callable(self.generator_plugin.get_model):
            try:
                self.generator = self.generator_plugin.get_model()
                self.logger.info("Generator model retrieved from generator_plugin.get_model()")
            except Exception as e:
                self.logger.error(f"Failed to call generator_plugin.get_model(): {e}")
        else:
            self.logger.error("Generator plugin does not have accessible model")
            
        # Extract parameters from generator if successful
        if self.generator:
            self._extract_parameters_from_generator()
    
    def _extract_parameters_from_generator(self):
        """Extract core parameters from generator model."""
        # Set default values
        self.seq_len = self.params.get("seq_len", 144)
        self.latent_dim = self.params.get("latent_dim", 128)
        
        # Use the full feature list for discriminator as per REFERENCE.md
        # The discriminator processes the complete 57-feature output from GeneratorPlugin
        feature_names_for_discriminator = self.params.get("feature_names_for_discriminator_ordered", [])
        if feature_names_for_discriminator:
            self.num_discriminator_features = len(feature_names_for_discriminator)
        else:
            # Fallback to default if not configured
            self.num_discriminator_features = len(
                self.params.get("discriminator_input_feature_names", ["OPEN", "HIGH", "LOW", "CLOSE"])
            )
        
        # Try to extract from generator input shape
        if not hasattr(self.generator, 'input_shape'):
            self.logger.warning("Generator model has no input_shape attribute")
            return
        
        try:
            generator_input_shape = self.generator.input_shape
            
            if isinstance(generator_input_shape, list) and len(generator_input_shape) > 0:
                # Handle multiple inputs - find latent input
                for input_shape in generator_input_shape:
                    if len(input_shape) == 3:  # (batch, seq, features)
                        if input_shape[1]:  # seq_len
                            self.seq_len = input_shape[1]
                        if input_shape[2]:  # features/latent_dim
                            self.latent_dim = input_shape[2]
                        break
            elif len(generator_input_shape) == 3:
                # Single input with shape (batch, seq, features)
                if generator_input_shape[1]:
                    self.seq_len = generator_input_shape[1]
                if generator_input_shape[2]:
                    self.latent_dim = generator_input_shape[2]
            
            self.logger.info(f"Extracted from generator - seq_len: {self.seq_len}, latent_dim: {self.latent_dim}")
            
        except Exception as e:
            self.logger.warning(f"Could not extract parameters from generator: {e}")
    
    def get_generator_model(self):
        """Get the extracted generator model."""
        return self.generator
    
    def get_extracted_parameters(self) -> Tuple[int, int, int]:
        """
        Get extracted parameters from generator.
        
        Returns:
            Tuple of (seq_len, latent_dim, num_discriminator_features)
        """
        return self.seq_len, self.latent_dim, self.num_discriminator_features
    
    def validate_generator_plugin(self) -> bool:
        """
        Validate that generator plugin is properly configured.
        
        Returns:
            True if generator plugin is valid, False otherwise
        """
        if not self.generator_plugin:
            self.logger.error("Generator plugin not provided")
            return False
        
        if not self.generator:
            self.logger.error("Generator model not accessible from plugin")
            return False
        
        # Check if generator has expected methods/attributes
        required_attributes = ['input_shape', 'output_shape']
        missing_attributes = []
        
        for attr in required_attributes:
            if not hasattr(self.generator, attr):
                missing_attributes.append(attr)
        
        if missing_attributes:
            self.logger.warning(f"Generator model missing attributes: {missing_attributes}")
        
        self.logger.info("Generator plugin validation passed")
        return True
    
    def validate_feeder_plugin(self) -> bool:
        """
        Validate that feeder plugin is properly configured.
        
        Returns:
            True if feeder plugin is valid, False otherwise
        """
        if not self.feeder_plugin:
            self.logger.warning("Feeder plugin not provided")
            return False
        
        # Check for expected methods in feeder plugin
        expected_methods = ['generate_batch']
        missing_methods = []
        
        for method in expected_methods:
            if not hasattr(self.feeder_plugin, method) or not callable(getattr(self.feeder_plugin, method)):
                missing_methods.append(method)
        
        if missing_methods:
            self.logger.error(f"Feeder plugin missing methods: {missing_methods}")
            return False
        
        self.logger.info("Feeder plugin validation passed")
        return True
    
    def get_feeder_plugin(self):
        """Get the feeder plugin instance."""
        return self.feeder_plugin
    
    def get_preprocessor_plugin(self):
        """Get the preprocessor plugin instance."""
        return self.preprocessor_plugin
    
    def extract_feeder_parameters(self) -> Dict[str, Any]:
        """
        Extract relevant parameters from feeder plugin configuration.
        
        Returns:
            Dict containing feeder-related parameters
        """
        feeder_params = {}
        
        # Extract feeder-specific parameters from config
        feeder_keys = [
            'feeder_date_feature_names_for_conditioning',
            'feeder_max_day_of_month',
            'feeder_max_hour_of_day', 
            'feeder_max_day_of_week',
            'conditional_fundamental_feature_names',
            'num_conditional_prev_tick_features',
            'datetime_col_name_in_x_real_df'
        ]
        
        for key in feeder_keys:
            if key in self.params:
                feeder_params[key] = self.params[key]
        
        return feeder_params
    
    def get_discriminator_feature_config(self) -> Dict[str, Any]:
        """
        Get discriminator feature configuration.
        
        Returns:
            Dict containing discriminator feature settings
        """
        return {
            'input_feature_names': self.params.get("discriminator_input_feature_names", ["OPEN", "HIGH", "LOW", "CLOSE"]),
            'num_features': self.num_discriminator_features,
            'seq_len': self.seq_len
        }
    
    def get_debug_info(self) -> Dict[str, Any]:
        """
        Get debug information for this module.
        
        Returns:
            Dict containing debug information
        """
        return {
            'generator_plugin_available': self.generator_plugin is not None,
            'feeder_plugin_available': self.feeder_plugin is not None,
            'preprocessor_plugin_available': self.preprocessor_plugin is not None,
            'generator_model_extracted': self.generator is not None,
            'extracted_seq_len': self.seq_len,
            'extracted_latent_dim': self.latent_dim,
            'num_discriminator_features': self.num_discriminator_features,
            'generator_valid': self.validate_generator_plugin() if self.generator_plugin else False,
            'feeder_valid': self.validate_feeder_plugin() if self.feeder_plugin else False
        }
