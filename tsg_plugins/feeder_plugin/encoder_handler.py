"""
Encoder Handler Module

Handles encoder model loading, management, and encoding operations.
Manages Keras/TensorFlow encoder models for latent vector generation.
"""

import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Dict, Any, Optional, Tuple
import os

logger = logging.getLogger(__name__)


class EncoderHandler:
    """
    Handles encoder model operations for latent vector generation.
    
    Supports Keras/TensorFlow models and provides encoding services 
    for the feeder plugin.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the encoder handler."""
        self.config = config
        
        # Model management
        self.model = None
        self.model_path = None
        
        # Latent space properties
        self.latent_dim = config.get('latent_dim', 128)
        self.expected_input_shape = None
        self.expected_output_shape = None
        
        # Statistics for latent space
        self.latent_stats = {
            'mean': None,
            'std': None,
            'min': None,
            'max': None
        }
        
        # State tracking
        self.is_initialized = False
        self.model_loaded = False
        
        logger.info("EncoderHandler initialized")
    
    def load_model(self, model_path: str) -> bool:
        """
        Load encoder model from file.
        
        Args:
            model_path: Path to the encoder model file
            
        Returns:
            bool: True if model loaded successfully
        """
        try:
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return False
            
            # Load Keras model with error handling for different formats
            try:
                # Try loading as is (works for .keras format)
                self.model = keras.models.load_model(model_path)
            except Exception as e1:
                try:
                    # Try loading with compile=False for problematic custom objects
                    self.model = keras.models.load_model(model_path, compile=False)
                    # Recompile with safe defaults
                    self.model.compile(
                        optimizer='adam',
                        loss='mean_squared_error',
                        metrics=['mean_squared_error']
                    )
                    logger.info("Model recompiled with safe defaults")
                except Exception as e2:
                    logger.error(f"Failed both loading attempts: {str(e1)}, {str(e2)}")
                    return False
            
            self.model_path = model_path
            
            # Get model input/output shapes
            self.expected_input_shape = self.model.input_shape[1:]
            self.expected_output_shape = self.model.output_shape[1:]
            
            # Update latent dimension from model
            if len(self.expected_output_shape) == 1:
                self.latent_dim = self.expected_output_shape[0]
            
            self.model_loaded = True
            logger.info(f"Encoder model loaded successfully from {model_path}")
            logger.info(f"Input shape: {self.expected_input_shape}, Output shape: {self.expected_output_shape}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load encoder model: {str(e)}")
            return False
    
    def validate_model(self) -> bool:
        """
        Validate that the loaded model is working correctly.
        
        Returns:
            bool: True if model is valid
        """
        if not self.model_loaded:
            logger.error("No model loaded for validation")
            return False
        
        try:
            # Create dummy input matching expected shape
            dummy_input = np.random.randn(1, *self.expected_input_shape)
            
            # Test encoding
            encoded = self.model.predict(dummy_input, verbose=0)
            
            if encoded.shape[1:] != self.expected_output_shape:
                logger.error(f"Model output shape mismatch: expected {self.expected_output_shape}, got {encoded.shape[1:]}")
                return False
            
            logger.info("Model validation successful")
            return True
            
        except Exception as e:
            logger.error(f"Model validation failed: {str(e)}")
            return False
    
    def encode_data(self, data: np.ndarray) -> Optional[np.ndarray]:
        """
        Encode input data to latent space.
        
        Args:
            data: Input data array to encode
            
        Returns:
            Optional[np.ndarray]: Encoded latent vectors or None if failed
        """
        if not self.model_loaded:
            logger.error("No encoder model loaded")
            return None
        
        try:
            # Validate input shape
            if data.shape[1:] != self.expected_input_shape:
                logger.error(f"Input shape mismatch: expected {self.expected_input_shape}, got {data.shape[1:]}")
                return None
            
            # Perform encoding
            encoded = self.model.predict(data, verbose=0)
            
            # Update statistics
            self._update_latent_stats(encoded)
            
            logger.debug(f"Encoded {data.shape[0]} samples to latent space")
            return encoded
            
        except Exception as e:
            logger.error(f"Encoding failed: {str(e)}")
            return None
    
    def _update_latent_stats(self, latent_vectors: np.ndarray):
        """Update statistics for latent space vectors."""
        try:
            self.latent_stats['mean'] = np.mean(latent_vectors, axis=0)
            self.latent_stats['std'] = np.std(latent_vectors, axis=0)
            self.latent_stats['min'] = np.min(latent_vectors, axis=0)
            self.latent_stats['max'] = np.max(latent_vectors, axis=0)
        except Exception as e:
            logger.warning(f"Failed to update latent stats: {str(e)}")
    
    def get_latent_stats(self) -> Dict[str, Any]:
        """Get current latent space statistics."""
        return self.latent_stats.copy()
    
    def reset_stats(self):
        """Reset latent space statistics."""
        for key in self.latent_stats:
            self.latent_stats[key] = None
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dict containing model information
        """
        if not self.model_loaded:
            return {'loaded': False}
        
        return {
            'loaded': True,
            'model_path': self.model_path,
            'input_shape': self.expected_input_shape,
            'output_shape': self.expected_output_shape,
            'latent_dim': self.latent_dim,
            'trainable_params': self.model.count_params()
        }
    
    def cleanup(self):
        """Clean up resources."""
        if self.model is not None:
            del self.model
            self.model = None
        
        self.model_loaded = False
        self.is_initialized = False
        self.reset_stats()
        
        logger.info("EncoderHandler cleaned up")
