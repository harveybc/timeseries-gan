#!/usr/bin/env python3
"""
Model Loader Module

This module handles model loading and validation operations,
providing focused functionality for Keras model management.
"""

import os
import zipfile
import logging
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from typing import Optional, Dict, Any


class ModelLoader:
    """Handles model loading and validation operations."""
    
    def __init__(self, params: Dict[str, Any], logger: logging.Logger):
        """Initialize model loader."""
        self.params = params
        self.logger = logger
        self.logger.info("ModelLoader initialized")
    
    def load_model_from_path(self, model_path: str) -> Optional[Model]:
        """
        Load Keras model from specified path with error handling.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            Loaded Keras model or None if loading fails
        """
        if not model_path:
            self.logger.error("Model path is empty or None")
            return None
            
        if not os.path.exists(model_path):
            self.logger.error(f"Model file does not exist: {model_path}")
            return None
        
        try:
            # Try Keras 3 safe mode first
            self.logger.info(f"Attempting to load model from: {model_path}")
            
            # Check if it's a .keras file (zip format)
            if model_path.endswith('.keras'):
                if not zipfile.is_zipfile(model_path):
                    self.logger.error(f"Invalid .keras file format: {model_path}")
                    return None
            
            # Set safe mode for Keras 3 if available
            try:
                tf.keras.config.set_safe_mode(False)
                self.logger.debug("Keras 3 safe mode disabled")
            except AttributeError:
                # Fallback for Keras 2
                try:
                    tf.keras.utils.enable_unsafe_deserialization()
                    self.logger.debug("Keras 2 unsafe deserialization enabled")
                except AttributeError:
                    self.logger.debug("Using default deserialization settings")
            
            # Load the model
            try:
                # Try loading without compiling to support .keras format
                model = load_model(model_path, compile=False)
                self.logger.debug("Model loaded with compile=False")
            except TypeError:
                # Fallback to default loading
                model = load_model(model_path)
            
            # Validate model
            if not self._validate_model(model):
                self.logger.error("Model validation failed")
                return None
            
            self.logger.info(f"Successfully loaded model: {model_path}")
            self.logger.info(f"Model summary: {len(model.layers)} layers")
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error loading model from {model_path}: {e}")
            return None
    
    def _validate_model(self, model: Model) -> bool:
        """
        Validate loaded model structure.
        
        Args:
            model: Loaded Keras model
            
        Returns:
            True if model is valid, False otherwise
        """
        try:
            if model is None:
                return False
            
            # Check if model has inputs and outputs
            if not hasattr(model, 'inputs') or not model.inputs:
                self.logger.error("Model has no inputs")
                return False
                
            if not hasattr(model, 'outputs') or not model.outputs:
                self.logger.error("Model has no outputs")
                return False
            
            # Log model information
            self.logger.info(f"Model inputs: {len(model.inputs)}")
            self.logger.info(f"Model outputs: {len(model.outputs)}")
            
            # Check input names if specified in params
            expected_inputs = [
                self.params.get("decoder_input_name_latent"),
                self.params.get("decoder_input_name_window"),
                self.params.get("decoder_input_name_conditions"),
                self.params.get("decoder_input_name_context")
            ]
            
            input_names = [inp.name for inp in model.inputs if hasattr(inp, 'name')]
            self.logger.info(f"Model input names: {input_names}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating model: {e}")
            return False
    
    def get_model_info(self, model: Model) -> Dict[str, Any]:
        """
        Get detailed information about the loaded model.
        
        Args:
            model: Keras model
            
        Returns:
            Dictionary with model information
        """
        if model is None:
            return {}
        
        try:
            info = {
                "num_layers": len(model.layers),
                "num_parameters": model.count_params(),
                "input_shapes": [inp.shape.as_list() for inp in model.inputs],
                "output_shapes": [out.shape.as_list() for out in model.outputs],
                "input_names": [inp.name for inp in model.inputs],
                "output_names": [out.name for out in model.outputs]
            }
            
            return info
            
        except Exception as e:
            self.logger.error(f"Error getting model info: {e}")
            return {}
    
    def check_model_compatibility(self, model: Model) -> bool:
        """
        Check if model is compatible with plugin requirements.
        
        Args:
            model: Keras model to check
            
        Returns:
            True if compatible, False otherwise
        """
        try:
            # Check expected input names
            expected_inputs = [
                self.params.get("decoder_input_name_latent"),
                self.params.get("decoder_input_name_window"),
                self.params.get("decoder_input_name_conditions"),
                self.params.get("decoder_input_name_context")
            ]
            
            input_names = [inp.name for inp in model.inputs]
            
            # Check if at least the latent input exists
            latent_input = self.params.get("decoder_input_name_latent")
            if latent_input and latent_input not in input_names:
                self.logger.warning(f"Expected latent input '{latent_input}' not found in model")
                return False
            
            self.logger.info("Model compatibility check passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking model compatibility: {e}")
            return False
    
    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information."""
        return {
            "supported_formats": [".keras", ".h5", ".tf"],
            "validation_enabled": True
        }
