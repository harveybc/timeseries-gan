#!/usr/bin/env python3
"""
Encoder Handler Module

Handles encoder model loading, validation, and latent space encoding operations
for the feeder plugin.
"""

import os
import numpy as np
import tensorflow as tf
from typing import Any, Dict, Optional
from tensorflow.keras.models import load_model # type: ignore


class EncoderHandler:
    """Handles encoder model operations for latent vector generation."""
    
    def __init__(self):
        """Initialize encoder handler."""
        self.encoder_model = None
        self.latent_mean = None
        self.latent_std = None
        self.latent_samples = None
        self.expected_latent_shape = None
        
    def load_encoder_model(self, model_path: str) -> bool:
        """
        Load encoder model from file.
        
        Args:
            model_path: Path to the encoder model file
            
        Returns:
            True if successful, False otherwise
        """
        if not model_path:
            print("EncoderHandler: No encoder model path provided")
            return False
            
        if not os.path.exists(model_path):
            print(f"EncoderHandler: Encoder model file not found: {model_path}")
            return False
            
        try:
            print(f"EncoderHandler: Loading encoder model from {model_path}")
            self.encoder_model = load_model(model_path)
            print(f"EncoderHandler: Encoder loaded successfully")
            return True
            
        except Exception as e:
            print(f"EncoderHandler: Error loading encoder model: {e}")
            self.encoder_model = None
            return False
    
    def validate_encoder_model(self) -> bool:
        """
        Validate that encoder model is properly loaded.
        
        Returns:
            True if valid, False otherwise
        """
        if self.encoder_model is None:
            return False
            
        try:
            # Check if model has inputs and outputs
            if not hasattr(self.encoder_model, 'inputs') or not self.encoder_model.inputs:
                print("EncoderHandler: Model has no inputs")
                return False
                
            if not hasattr(self.encoder_model, 'outputs') or not self.encoder_model.outputs:
                print("EncoderHandler: Model has no outputs")
                return False
                
            return True
            
        except Exception as e:
            print(f"EncoderHandler: Error validating encoder model: {e}")
            return False
    
    def encode_data(self, data: np.ndarray) -> Optional[np.ndarray]:
        """
        Encode data using the loaded encoder model.
        
        Args:
            data: Input data to encode
            
        Returns:
            Encoded latent representations or None if failed
        """
        if self.encoder_model is None:
            print("EncoderHandler: No encoder model loaded")
            return None
            
        if data is None or len(data) == 0:
            print("EncoderHandler: No data provided for encoding")
            return None
            
        try:
            print(f"EncoderHandler: Encoding {len(data)} samples")
            latent_representations = self.encoder_model.predict(data, verbose=0)
            print(f"EncoderHandler: Encoded to shape {latent_representations.shape}")
            return latent_representations
            
        except Exception as e:
            print(f"EncoderHandler: Error encoding data: {e}")
            return None
    
    def process_latent_representations(self, latent_data: np.ndarray, latent_shape: list) -> bool:
        """
        Process latent representations for sampling preparation.
        
        Args:
            latent_data: Encoded latent representations
            latent_shape: Expected latent shape [sequence_length, features]
            
        Returns:
            True if successful, False otherwise
        """
        if latent_data is None:
            return False
            
        try:
            self.expected_latent_shape = latent_shape
            
            # Handle different dimensionalities of latent data
            if latent_data.ndim == 2:
                # 2D: (samples, latent_dim) -> reshape to match expected shape
                samples, latent_dim = latent_data.shape
                expected_total_dim = latent_shape[0] * latent_shape[1]
                
                if latent_dim != expected_total_dim:
                    print(f"EncoderHandler: Warning - latent dimension mismatch. "
                          f"Got {latent_dim}, expected {expected_total_dim}")
                
                # Reshape to (samples, sequence_length, features)
                self.latent_samples = latent_data.reshape(samples, latent_shape[0], latent_shape[1])
                
            elif latent_data.ndim == 3:
                # 3D: Already in proper shape (samples, sequence_length, features)
                self.latent_samples = latent_data
                
            else:
                print(f"EncoderHandler: Unsupported latent data dimensions: {latent_data.ndim}")
                return False
            
            # Calculate statistics for sampling
            self.latent_mean = np.mean(self.latent_samples, axis=0)
            self.latent_std = np.std(self.latent_samples, axis=0)
            
            print(f"EncoderHandler: Processed {len(self.latent_samples)} latent samples")
            print(f"EncoderHandler: Latent shape: {self.latent_samples.shape}")
            print(f"EncoderHandler: Mean shape: {self.latent_mean.shape}")
            print(f"EncoderHandler: Std shape: {self.latent_std.shape}")
            
            return True
            
        except Exception as e:
            print(f"EncoderHandler: Error processing latent representations: {e}")
            return False
    
    def get_latent_statistics(self) -> Dict[str, Any]:
        """
        Get latent space statistics.
        
        Returns:
            Dictionary with latent statistics
        """
        if self.latent_samples is None:
            return {}
            
        return {
            "samples_count": len(self.latent_samples),
            "latent_shape": self.latent_samples.shape,
            "mean_shape": self.latent_mean.shape if self.latent_mean is not None else None,
            "std_shape": self.latent_std.shape if self.latent_std is not None else None,
            "expected_shape": self.expected_latent_shape
        }
    
    def clear_encoder_state(self) -> None:
        """Clear encoder state and free memory."""
        self.encoder_model = None
        self.latent_mean = None
        self.latent_std = None
        self.latent_samples = None
        self.expected_latent_shape = None
        print("EncoderHandler: Encoder state cleared")
    
    def is_ready_for_sampling(self) -> bool:
        """
        Check if encoder is ready for sampling operations.
        
        Returns:
            True if ready, False otherwise
        """
        return (self.encoder_model is not None and 
                self.latent_samples is not None and
                self.latent_mean is not None and
                self.latent_std is not None)
    
    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information about encoder state."""
        return {
            "encoder_loaded": self.encoder_model is not None,
            "latent_samples_available": self.latent_samples is not None,
            "statistics_calculated": (self.latent_mean is not None and 
                                   self.latent_std is not None),
            "ready_for_sampling": self.is_ready_for_sampling(),
            **self.get_latent_statistics()
        }
