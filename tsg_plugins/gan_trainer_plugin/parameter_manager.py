#!/usr/bin/env python3
"""
Parameter Manager Module

This module handles parameter extraction, validation and setup,
providing focused functionality for configuration management.
"""

import logging
from typing import Dict, Any, Optional, Tuple


class ParameterManager:
    """Handles parameter extraction, validation and setup."""
    
    def __init__(self, params: Dict[str, Any], logger: logging.Logger):
        """Initialize parameter manager."""
        self.params = params
        self.logger = logger
        
        # Core parameters
        self.seq_len = None
        self.latent_dim = None
        self.num_discriminator_features = None
        self.conditional_dim = None
        self.context_dim = None
        
        self.logger.info("ParameterManager initialized")
    
    def extract_core_parameters(self, generator_model=None) -> Tuple[int, int, int]:
        """
        Extract core parameters from configuration and generator model.
        
        Args:
            generator_model: Generator model to extract parameters from
            
        Returns:
            Tuple of (seq_len, latent_dim, num_discriminator_features)
        """
        # Default values from config
        self.seq_len = self.params.get("seq_len", 144)
        self.latent_dim = self.params.get("latent_dim", 128)
        self.num_discriminator_features = len(
            self.params.get("discriminator_input_feature_names", ["OPEN", "HIGH", "LOW", "CLOSE"])
        )
        
        # Extract from generator if available
        if generator_model and hasattr(generator_model, 'input_shape'):
            self._extract_from_generator(generator_model)
        
        # Extract conditional and context dimensions
        self._extract_conditional_dimensions()
        
        self.logger.info(f"Core parameters: seq_len={self.seq_len}, latent_dim={self.latent_dim}, "
                        f"discriminator_features={self.num_discriminator_features}")
        
        return self.seq_len, self.latent_dim, self.num_discriminator_features
    
    def _extract_from_generator(self, generator_model):
        """Extract parameters from generator model structure."""
        try:
            generator_input_shape = generator_model.input_shape
            
            if isinstance(generator_input_shape, list) and len(generator_input_shape) > 0:
                # Handle multiple inputs - find latent input
                for input_shape in generator_input_shape:
                    if len(input_shape) == 3:  # (batch, seq, features)
                        self.seq_len = input_shape[1] or self.seq_len
                        self.latent_dim = input_shape[2] or self.latent_dim
                        break
                    elif len(input_shape) == 2:  # (batch, features) - likely latent
                        self.latent_dim = input_shape[1] or self.latent_dim
                        
            elif len(generator_input_shape) == 3:  # Single input
                self.seq_len = generator_input_shape[1] or self.seq_len
                self.latent_dim = generator_input_shape[2] or self.latent_dim
            elif len(generator_input_shape) == 2:  # Single latent input
                self.latent_dim = generator_input_shape[1] or self.latent_dim
                
            self.logger.info("Parameters extracted from generator model structure")
            
        except Exception as e:
            self.logger.warning(f"Could not extract parameters from generator: {e}")
    
    def _extract_conditional_dimensions(self):
        """Extract conditional and context dimensions from configuration."""
        # Calculate conditional dimension from feeder configuration
        date_features = self.params.get("feeder_date_feature_names_for_conditioning", [])
        fundamental_features = self.params.get("conditional_fundamental_feature_names", [])
        prev_tick_features = self.params.get("num_conditional_prev_tick_features", 0)
        
        self.conditional_dim = len(date_features) + len(fundamental_features) + prev_tick_features
        
        # Context dimension (e.g., VAE encoder output)
        self.context_dim = self.params.get("context_dim", 0)
        
        self.logger.info(f"Conditional dimensions: conditional={self.conditional_dim}, context={self.context_dim}")
    
    def get_training_parameters(self) -> Dict[str, Any]:
        """Get training-specific parameters."""
        return {
            "gan_epochs": self.params.get("gan_epochs", 10000),
            "gan_batch_size": self.params.get("gan_batch_size", 32),
            "generator_lr": self.params.get("generator_lr", 1e-4),
            "discriminator_lr": self.params.get("discriminator_lr", 1e-4),
            "generator_beta1": self.params.get("generator_beta1", 0.5),
            "discriminator_beta1": self.params.get("discriminator_beta1", 0.5),
            "save_interval": self.params.get("gan_save_interval", 500)
        }
    
    def get_discriminator_parameters(self) -> Dict[str, Any]:
        """Get discriminator architecture parameters."""
        return {
            "conv_filters": self.params.get("discriminator_conv_filters", [64, 128]),
            "conv_kernel_size": self.params.get("discriminator_conv_kernel_size", 3),
            "lstm_units": self.params.get("discriminator_lstm_units", 64),
            "dropout_rate": self.params.get("discriminator_dropout_rate", 0.3),
            "num_features": self.num_discriminator_features,
            "seq_len": self.seq_len
        }
    
    def get_callback_parameters(self) -> Dict[str, Any]:
        """Get training callback parameters."""
        return {
            "enable_reduce_lr": self.params.get("enable_reduce_lr_on_plateau", True),
            "lr_reduction_factor": self.params.get("lr_reduction_factor", 0.5),
            "lr_patience": self.params.get("lr_patience", 50),
            "lr_min_delta": self.params.get("lr_min_delta", 0.001),
            "min_lr_g": self.params.get("min_lr_g", 1e-7),
            "min_lr_d": self.params.get("min_lr_d", 1e-7),
            "lr_monitor_metric": self.params.get("lr_monitor_metric", "g_loss"),
            "enable_early_stopping": self.params.get("enable_early_stopping", True),
            "es_patience": self.params.get("es_patience", 200),
            "es_min_delta": self.params.get("es_min_delta", 0.001),
            "es_monitor_metric": self.params.get("es_monitor_metric", "g_loss")
        }
    
    def validate_parameters(self) -> bool:
        """
        Validate that all required parameters are present and valid.
        
        Returns:
            True if parameters are valid, False otherwise
        """
        try:
            # Check core parameters
            if self.seq_len <= 0:
                self.logger.error(f"Invalid seq_len: {self.seq_len}")
                return False
            
            if self.latent_dim <= 0:
                self.logger.error(f"Invalid latent_dim: {self.latent_dim}")
                return False
            
            if self.num_discriminator_features <= 0:
                self.logger.error(f"Invalid num_discriminator_features: {self.num_discriminator_features}")
                return False
            
            # Check training parameters
            training_params = self.get_training_parameters()
            if training_params["gan_epochs"] <= 0:
                self.logger.error(f"Invalid gan_epochs: {training_params['gan_epochs']}")
                return False
            
            if training_params["gan_batch_size"] <= 0:
                self.logger.error(f"Invalid gan_batch_size: {training_params['gan_batch_size']}")
                return False
            
            self.logger.info("Parameter validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Parameter validation failed: {e}")
            return False
    
    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information."""
        return {
            "seq_len": self.seq_len,
            "latent_dim": self.latent_dim,
            "num_discriminator_features": self.num_discriminator_features,
            "conditional_dim": self.conditional_dim,
            "context_dim": self.context_dim,
            "training_params": self.get_training_parameters(),
            "discriminator_params": self.get_discriminator_parameters(),
            "callback_params": self.get_callback_parameters()
        }
