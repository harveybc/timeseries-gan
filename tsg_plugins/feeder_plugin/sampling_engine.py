"""
Sampling Engine Module

Handles latent vector sampling for generation mode.
Provides various sampling strategies and noise generation methods.
"""

import logging
import numpy as np
import torch
from typing import Dict, Any, Optional, Tuple, List

logger = logging.getLogger(__name__)


class SamplingEngine:
    """
    Handles latent vector sampling for data generation.
    
    Provides multiple sampling strategies and noise generation methods
    for creating diverse latent vectors for the GAN generator.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the sampling engine."""
        self.config = config
        self.device = config.get('device', 'cpu')
        
        # Sampling parameters
        self.latent_dim = config.get('latent_dim', 128)
        self.sampling_method = config.get('sampling_method', 'normal')
        self.use_truncation = config.get('use_truncation', False)
        self.truncation_threshold = config.get('truncation_threshold', 2.0)
        self.temperature = config.get('sampling_temperature', 1.0)
        
        # Random seed for reproducibility
        self.random_seed = config.get('random_seed', None)
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            torch.manual_seed(self.random_seed)
        
        # State tracking
        self.is_initialized = False
        self.sampling_stats = {}
        
        logger.info("SamplingEngine initialized")
    
    def initialize(self) -> bool:
        """Initialize the sampling engine."""
        try:
            # Validate sampling parameters
            self._validate_parameters()
            
            # Setup sampling statistics
            self._setup_sampling_stats()
            
            self.is_initialized = True
            logger.info("SamplingEngine initialization completed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize SamplingEngine: {e}")
            return False
    
    def sample_latent_vectors(self, batch_size: int, conditioning: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Sample a batch of latent vectors.
        
        Args:
            batch_size: Number of latent vectors to sample
            conditioning: Optional conditioning information
            
        Returns:
            Array of sampled latent vectors
        """
        try:
            if not self.is_initialized:
                raise ValueError("SamplingEngine not initialized")
            
            # Sample base latent vectors
            if self.sampling_method == 'normal':
                latents = self._sample_normal(batch_size)
            elif self.sampling_method == 'uniform':
                latents = self._sample_uniform(batch_size)
            elif self.sampling_method == 'truncated_normal':
                latents = self._sample_truncated_normal(batch_size)
            elif self.sampling_method == 'spherical':
                latents = self._sample_spherical(batch_size)
            else:
                raise ValueError(f"Unknown sampling method: {self.sampling_method}")
            
            # Apply conditioning if provided
            if conditioning is not None:
                latents = self._apply_conditioning(latents, conditioning)
            
            # Apply temperature scaling
            if self.temperature != 1.0:
                latents = self._apply_temperature_scaling(latents)
            
            # Update sampling statistics
            self._update_sampling_stats(latents)
            
            logger.debug(f"Sampled {batch_size} latent vectors using {self.sampling_method}")
            return latents
            
        except Exception as e:
            logger.error(f"Error sampling latent vectors: {e}")
            return np.zeros((batch_size, self.latent_dim), dtype=np.float32)
    
    def _sample_normal(self, batch_size: int) -> np.ndarray:
        """Sample from standard normal distribution."""
        return np.random.normal(0.0, 1.0, (batch_size, self.latent_dim)).astype(np.float32)
    
    def _sample_uniform(self, batch_size: int) -> np.ndarray:
        """Sample from uniform distribution."""
        return np.random.uniform(-1.0, 1.0, (batch_size, self.latent_dim)).astype(np.float32)
    
    def _sample_truncated_normal(self, batch_size: int) -> np.ndarray:
        """Sample from truncated normal distribution."""
        latents = np.random.normal(0.0, 1.0, (batch_size, self.latent_dim))
        
        if self.use_truncation:
            # Truncate values beyond threshold
            latents = np.clip(latents, -self.truncation_threshold, self.truncation_threshold)
        
        return latents.astype(np.float32)
    
    def _sample_spherical(self, batch_size: int) -> np.ndarray:
        """Sample from spherical (normalized) distribution."""
        latents = np.random.normal(0.0, 1.0, (batch_size, self.latent_dim))
        
        # Normalize to unit sphere
        norms = np.linalg.norm(latents, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)  # Avoid division by zero
        latents = latents / norms
        
        return latents.astype(np.float32)
    
    def _apply_conditioning(self, latents: np.ndarray, conditioning: np.ndarray) -> np.ndarray:
        """Apply conditioning to latent vectors."""
        # Simple concatenation conditioning
        if conditioning.shape[0] != latents.shape[0]:
            logger.warning("Conditioning batch size mismatch, broadcasting")
            conditioning = np.broadcast_to(conditioning, (latents.shape[0], conditioning.shape[-1]))
        
        # Concatenate conditioning to latents
        conditioned_latents = np.concatenate([latents, conditioning], axis=1)
        
        return conditioned_latents
    
    def _apply_temperature_scaling(self, latents: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to latent vectors."""
        return latents * self.temperature
    
    def sample_interpolated_latents(self, latent1: np.ndarray, latent2: np.ndarray, steps: int = 10) -> np.ndarray:
        """
        Sample interpolated latent vectors between two points.
        
        Args:
            latent1: First latent vector
            latent2: Second latent vector
            steps: Number of interpolation steps
            
        Returns:
            Array of interpolated latent vectors
        """
        try:
            # Create interpolation weights
            alphas = np.linspace(0, 1, steps).reshape(-1, 1)
            
            # Linear interpolation
            interpolated = (1 - alphas) * latent1 + alphas * latent2
            
            return interpolated.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error in latent interpolation: {e}")
            return np.zeros((steps, latent1.shape[0]), dtype=np.float32)
    
    def sample_perturbed_latents(self, base_latent: np.ndarray, n_samples: int, noise_scale: float = 0.1) -> np.ndarray:
        """
        Sample perturbed versions of a base latent vector.
        
        Args:
            base_latent: Base latent vector to perturb
            n_samples: Number of perturbed samples
            noise_scale: Scale of perturbation noise
            
        Returns:
            Array of perturbed latent vectors
        """
        try:
            # Generate noise
            noise = np.random.normal(0.0, noise_scale, (n_samples, base_latent.shape[0]))
            
            # Add noise to base latent
            perturbed = base_latent + noise
            
            return perturbed.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error in latent perturbation: {e}")
            return np.zeros((n_samples, base_latent.shape[0]), dtype=np.float32)
    
    def get_default_latent_shape(self) -> Tuple[int]:
        """Get the default latent vector shape."""
        return (self.latent_dim,)
    
    def set_sampling_parameters(self, parameters: Dict[str, Any]):
        """Update sampling parameters."""
        for key, value in parameters.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.info(f"Updated sampling parameter {key} to {value}")
    
    def _validate_parameters(self):
        """Validate sampling parameters."""
        valid_methods = ['normal', 'uniform', 'truncated_normal', 'spherical']
        if self.sampling_method not in valid_methods:
            raise ValueError(f"Invalid sampling method: {self.sampling_method}")
        
        if self.latent_dim <= 0:
            raise ValueError(f"Invalid latent dimension: {self.latent_dim}")
        
        if self.temperature <= 0:
            raise ValueError(f"Invalid temperature: {self.temperature}")
    
    def _setup_sampling_stats(self):
        """Setup sampling statistics tracking."""
        self.sampling_stats = {
            'total_samples': 0,
            'method_counts': {},
            'mean_norm': 0.0,
            'std_norm': 0.0
        }
    
    def _update_sampling_stats(self, latents: np.ndarray):
        """Update sampling statistics."""
        batch_size = latents.shape[0]
        self.sampling_stats['total_samples'] += batch_size
        
        # Track method usage
        method = self.sampling_method
        self.sampling_stats['method_counts'][method] = (
            self.sampling_stats['method_counts'].get(method, 0) + batch_size
        )
        
        # Track latent norms
        norms = np.linalg.norm(latents, axis=1)
        self.sampling_stats['mean_norm'] = np.mean(norms)
        self.sampling_stats['std_norm'] = np.std(norms)
    
    def get_sampling_stats(self) -> Dict[str, Any]:
        """Get current sampling statistics."""
        return self.sampling_stats.copy()
    
    def reset_sampling_stats(self):
        """Reset sampling statistics."""
        self._setup_sampling_stats()
        logger.info("Sampling statistics reset")
    
    def is_ready(self) -> bool:
        """Check if the sampling engine is ready for use."""
        return self.is_initialized
    
    def cleanup(self):
        """Cleanup sampling engine resources."""
        self.sampling_stats.clear()
        self.is_initialized = False
        logger.info("SamplingEngine cleanup completed")
