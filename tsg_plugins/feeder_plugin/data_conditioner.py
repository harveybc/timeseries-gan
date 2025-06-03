"""
Data Conditioner Module

Handles data conditioning operations for different modes (train, generate, evaluate).
Applies necessary transformations and preparations for downstream processing.
"""

import logging
import numpy as np
import torch
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)


class DataConditioner:
    """
    Handles data conditioning operations for GAN training and generation.
    
    Applies mode-specific transformations and preparations to ensure
    data compatibility with the GAN architecture.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the data conditioner."""
        self.config = config
        self.device = config.get('device', 'cpu')
        
        # Conditioning parameters
        self.use_normalization = config.get('use_normalization', True)
        self.normalization_method = config.get('normalization_method', 'standard')
        self.use_feature_scaling = config.get('use_feature_scaling', True)
        self.add_noise = config.get('add_conditioning_noise', False)
        self.noise_level = config.get('conditioning_noise_level', 0.01)
        
        # State tracking
        self.is_initialized = False
        self.normalization_stats = {}
        
        logger.info("DataConditioner initialized")
    
    def initialize(self) -> bool:
        """Initialize the data conditioner."""
        try:
            # Setup normalization parameters
            self._setup_normalization()
            
            self.is_initialized = True
            logger.info("DataConditioner initialization completed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize DataConditioner: {e}")
            return False
    
    def condition_data(self, data: np.ndarray, mode: str) -> np.ndarray:
        """
        Apply conditioning to input data based on mode.
        
        Args:
            data: Input data array
            mode: Operation mode ('train', 'generate', 'evaluate')
            
        Returns:
            Conditioned data array
        """
        try:
            if not self.is_initialized:
                raise ValueError("DataConditioner not initialized")
            
            conditioned_data = data.copy()
            
            # Apply mode-specific conditioning
            if mode == 'train':
                conditioned_data = self._condition_for_training(conditioned_data)
            elif mode == 'generate':
                conditioned_data = self._condition_for_generation(conditioned_data)
            elif mode == 'evaluate':
                conditioned_data = self._condition_for_evaluation(conditioned_data)
            else:
                raise ValueError(f"Unknown mode: {mode}")
            
            # Apply common conditioning steps
            conditioned_data = self._apply_common_conditioning(conditioned_data)
            
            logger.debug(f"Data conditioning completed for mode: {mode}")
            return conditioned_data
            
        except Exception as e:
            logger.error(f"Error in condition_data: {e}")
            return data
    
    def _condition_for_training(self, data: np.ndarray) -> np.ndarray:
        """Apply training-specific conditioning."""
        conditioned_data = data.copy()
        
        # Normalize data for training
        if self.use_normalization:
            conditioned_data = self._normalize_data(conditioned_data, fit=True)
        
        # Add training noise if configured
        if self.add_noise:
            noise = np.random.normal(0, self.noise_level, conditioned_data.shape)
            conditioned_data += noise
        
        # Feature scaling
        if self.use_feature_scaling:
            conditioned_data = self._apply_feature_scaling(conditioned_data)
        
        return conditioned_data
    
    def _condition_for_generation(self, data: np.ndarray) -> np.ndarray:
        """Apply generation-specific conditioning."""
        conditioned_data = data.copy()
        
        # Use pre-computed normalization stats
        if self.use_normalization and self.normalization_stats:
            conditioned_data = self._normalize_data(conditioned_data, fit=False)
        
        # Apply feature scaling with saved parameters
        if self.use_feature_scaling:
            conditioned_data = self._apply_feature_scaling(conditioned_data)
        
        return conditioned_data
    
    def _condition_for_evaluation(self, data: np.ndarray) -> np.ndarray:
        """Apply evaluation-specific conditioning."""
        conditioned_data = data.copy()
        
        # Apply same conditioning as generation but without noise
        if self.use_normalization and self.normalization_stats:
            conditioned_data = self._normalize_data(conditioned_data, fit=False)
        
        if self.use_feature_scaling:
            conditioned_data = self._apply_feature_scaling(conditioned_data)
        
        return conditioned_data
    
    def _apply_common_conditioning(self, data: np.ndarray) -> np.ndarray:
        """Apply conditioning steps common to all modes."""
        conditioned_data = data.copy()
        
        # Ensure data type consistency
        conditioned_data = conditioned_data.astype(np.float32)
        
        # Handle NaN values
        conditioned_data = self._handle_nan_values(conditioned_data)
        
        # Clamp extreme values
        conditioned_data = self._clamp_extreme_values(conditioned_data)
        
        return conditioned_data
    
    def _normalize_data(self, data: np.ndarray, fit: bool = False) -> np.ndarray:
        """Apply normalization to data."""
        if self.normalization_method == 'standard':
            return self._standard_normalization(data, fit)
        elif self.normalization_method == 'minmax':
            return self._minmax_normalization(data, fit)
        elif self.normalization_method == 'robust':
            return self._robust_normalization(data, fit)
        else:
            logger.warning(f"Unknown normalization method: {self.normalization_method}")
            return data
    
    def _standard_normalization(self, data: np.ndarray, fit: bool) -> np.ndarray:
        """Apply standard (z-score) normalization."""
        if fit:
            # Compute and store statistics
            self.normalization_stats['mean'] = np.mean(data, axis=0, keepdims=True)
            self.normalization_stats['std'] = np.std(data, axis=0, keepdims=True)
            # Avoid division by zero
            self.normalization_stats['std'] = np.where(
                self.normalization_stats['std'] == 0, 1.0, self.normalization_stats['std']
            )
        
        # Apply normalization
        mean = self.normalization_stats.get('mean', 0)
        std = self.normalization_stats.get('std', 1)
        return (data - mean) / std
    
    def _minmax_normalization(self, data: np.ndarray, fit: bool) -> np.ndarray:
        """Apply min-max normalization."""
        if fit:
            self.normalization_stats['min'] = np.min(data, axis=0, keepdims=True)
            self.normalization_stats['max'] = np.max(data, axis=0, keepdims=True)
            # Avoid division by zero
            data_range = self.normalization_stats['max'] - self.normalization_stats['min']
            self.normalization_stats['range'] = np.where(data_range == 0, 1.0, data_range)
        
        # Apply normalization
        min_val = self.normalization_stats.get('min', 0)
        range_val = self.normalization_stats.get('range', 1)
        return (data - min_val) / range_val
    
    def _robust_normalization(self, data: np.ndarray, fit: bool) -> np.ndarray:
        """Apply robust normalization using median and IQR."""
        if fit:
            self.normalization_stats['median'] = np.median(data, axis=0, keepdims=True)
            q75 = np.percentile(data, 75, axis=0, keepdims=True)
            q25 = np.percentile(data, 25, axis=0, keepdims=True)
            iqr = q75 - q25
            self.normalization_stats['iqr'] = np.where(iqr == 0, 1.0, iqr)
        
        # Apply normalization
        median = self.normalization_stats.get('median', 0)
        iqr = self.normalization_stats.get('iqr', 1)
        return (data - median) / iqr
    
    def _apply_feature_scaling(self, data: np.ndarray) -> np.ndarray:
        """Apply feature scaling to ensure proper range."""
        # Scale to [-1, 1] range for GAN compatibility
        data_min = np.min(data)
        data_max = np.max(data)
        data_range = data_max - data_min
        
        if data_range > 0:
            scaled_data = 2 * (data - data_min) / data_range - 1
        else:
            scaled_data = data
        
        return scaled_data
    
    def _handle_nan_values(self, data: np.ndarray) -> np.ndarray:
        """Handle NaN values in the data."""
        if np.any(np.isnan(data)):
            logger.warning("NaN values detected in data, replacing with zeros")
            data = np.nan_to_num(data, nan=0.0)
        return data
    
    def _clamp_extreme_values(self, data: np.ndarray) -> np.ndarray:
        """Clamp extreme values to prevent numerical issues."""
        clip_value = self.config.get('clip_value', 10.0)
        return np.clip(data, -clip_value, clip_value)
    
    def _setup_normalization(self):
        """Setup normalization parameters."""
        valid_methods = ['standard', 'minmax', 'robust']
        if self.normalization_method not in valid_methods:
            logger.warning(f"Invalid normalization method: {self.normalization_method}, using 'standard'")
            self.normalization_method = 'standard'
    
    def get_normalization_stats(self) -> Dict[str, Any]:
        """Get current normalization statistics."""
        return self.normalization_stats.copy()
    
    def set_normalization_stats(self, stats: Dict[str, Any]):
        """Set normalization statistics (for loading pre-computed stats)."""
        self.normalization_stats = stats.copy()
        logger.info("Normalization statistics loaded")
    
    def is_ready(self) -> bool:
        """Check if the conditioner is ready for use."""
        return self.is_initialized
    
    def cleanup(self):
        """Cleanup conditioner resources."""
        self.normalization_stats.clear()
        self.is_initialized = False
        logger.info("DataConditioner cleanup completed")
