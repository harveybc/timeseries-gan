"""
Condition Manager Module

Manages conditional information and context for data feeding operations.
Handles condition validation, processing, and integration with encoded data.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Union, Tuple

logger = logging.getLogger(__name__)


class ConditionManager:
    """
    Manages conditional information for the feeder plugin.
    
    Handles condition processing, validation, and integration
    with encoded latent vectors for conditional generation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the condition manager."""
        self.config = config
        
        # Condition parameters
        self.condition_columns = config.get('condition_columns', [])
        self.condition_method = config.get('condition_method', 'concatenate')
        self.condition_dim = config.get('condition_dim', 10)
        self.use_temporal_conditions = config.get('use_temporal_conditions', True)
        
        # Condition processing
        self.condition_encoders = {}
        self.condition_stats = {}
        self.temporal_features = ['hour', 'day', 'month', 'year', 'weekday']
        
        # State tracking
        self.is_initialized = False
        self.has_conditions = False
        self.condition_shape = None
        
        logger.info("ConditionManager initialized")
    
    def initialize(self, sample_data: Optional[pd.DataFrame] = None) -> bool:
        """
        Initialize condition manager with sample data.
        
        Args:
            sample_data: Sample data to analyze for condition setup
            
        Returns:
            bool: True if initialization successful
        """
        try:
            # Determine available conditions
            if sample_data is not None:
                self._analyze_available_conditions(sample_data)
            
            # Setup condition encoders
            self._setup_condition_encoders()
            
            # Calculate condition dimensions
            self._calculate_condition_dimensions()
            
            self.is_initialized = True
            logger.info(f"ConditionManager initialized with condition shape: {self.condition_shape}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize ConditionManager: {str(e)}")
            return False
    
    def extract_conditions(self, data: pd.DataFrame, timestamp_col: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Extract condition vectors from input data.
        
        Args:
            data: Input data containing condition information
            timestamp_col: Name of timestamp column for temporal conditions
            
        Returns:
            Optional[np.ndarray]: Condition vectors or None if failed
        """
        if not self.is_initialized:
            logger.error("ConditionManager not initialized")
            return None
        
        try:
            conditions = []
            
            # Extract explicit condition columns
            if self.condition_columns:
                explicit_conditions = self._extract_explicit_conditions(data)
                if explicit_conditions is not None:
                    conditions.append(explicit_conditions)
            
            # Extract temporal conditions
            if self.use_temporal_conditions and timestamp_col:
                temporal_conditions = self._extract_temporal_conditions(data, timestamp_col)
                if temporal_conditions is not None:
                    conditions.append(temporal_conditions)
            
            # Combine conditions
            if conditions:
                combined_conditions = np.concatenate(conditions, axis=1)
                self.has_conditions = True
                logger.debug(f"Extracted conditions shape: {combined_conditions.shape}")
                return combined_conditions
            else:
                # No conditions available, return zero vector
                zero_conditions = np.zeros((len(data), self.condition_dim))
                logger.debug("No conditions found, returning zero conditions")
                return zero_conditions
                
        except Exception as e:
            logger.error(f"Failed to extract conditions: {str(e)}")
            return None
    
    def process_conditions(self, conditions: np.ndarray) -> Optional[np.ndarray]:
        """
        Process and normalize condition vectors.
        
        Args:
            conditions: Raw condition vectors
            
        Returns:
            Optional[np.ndarray]: Processed conditions or None if failed
        """
        try:
            if conditions is None or len(conditions) == 0:
                return conditions
            
            # Normalize conditions
            processed = self._normalize_conditions(conditions)
            
            # Apply conditioning method
            if self.condition_method == 'concatenate':
                # Direct concatenation (already done)
                result = processed
            elif self.condition_method == 'embed':
                # Apply embedding transformation
                result = self._embed_conditions(processed)
            elif self.condition_method == 'encode':
                # Apply encoding transformation
                result = self._encode_conditions(processed)
            else:
                logger.warning(f"Unknown condition method: {self.condition_method}")
                result = processed
            
            logger.debug(f"Processed conditions shape: {conditions.shape} -> {result.shape}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to process conditions: {str(e)}")
            return None
    
    def combine_with_latents(self, latents: np.ndarray, conditions: np.ndarray) -> Optional[np.ndarray]:
        """
        Combine latent vectors with condition vectors.
        
        Args:
            latents: Encoded latent vectors
            conditions: Processed condition vectors
            
        Returns:
            Optional[np.ndarray]: Combined vectors or None if failed
        """
        try:
            if latents is None:
                logger.error("No latent vectors provided")
                return None
            
            if conditions is None or not self.has_conditions:
                logger.debug("No conditions to combine, returning latents only")
                return latents
            
            # Validate shapes
            if latents.shape[0] != conditions.shape[0]:
                logger.error(f"Shape mismatch: latents {latents.shape[0]} vs conditions {conditions.shape[0]}")
                return None
            
            # Combine vectors
            combined = np.concatenate([latents, conditions], axis=1)
            
            logger.debug(f"Combined vectors shape: {latents.shape} + {conditions.shape} = {combined.shape}")
            return combined
            
        except Exception as e:
            logger.error(f"Failed to combine latents with conditions: {str(e)}")
            return None
    
    def _analyze_available_conditions(self, data: pd.DataFrame):
        """Analyze available condition columns in data."""
        available_columns = data.columns.tolist()
        
        # Filter condition columns that actually exist
        valid_conditions = [col for col in self.condition_columns if col in available_columns]
        
        if len(valid_conditions) != len(self.condition_columns):
            missing = set(self.condition_columns) - set(valid_conditions)
            logger.warning(f"Some condition columns not found: {missing}")
        
        self.condition_columns = valid_conditions
        logger.info(f"Available condition columns: {self.condition_columns}")
    
    def _setup_condition_encoders(self):
        """Setup encoders for different condition types."""
        # For now, use simple statistical encoders
        # In future, could use more sophisticated encoders
        
        for col in self.condition_columns:
            self.condition_encoders[col] = {
                'type': 'numeric',  # Assume numeric for now
                'mean': 0.0,
                'std': 1.0,
                'min': 0.0,
                'max': 1.0
            }
    
    def _extract_explicit_conditions(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        """Extract explicitly specified condition columns."""
        try:
            if not self.condition_columns:
                return None
            
            # Select condition columns
            condition_data = data[self.condition_columns].values
            
            # Update statistics
            for i, col in enumerate(self.condition_columns):
                col_data = condition_data[:, i]
                self.condition_encoders[col].update({
                    'mean': np.mean(col_data),
                    'std': np.std(col_data),
                    'min': np.min(col_data),
                    'max': np.max(col_data)
                })
            
            return condition_data.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Failed to extract explicit conditions: {str(e)}")
            return None
    
    def _extract_temporal_conditions(self, data: pd.DataFrame, timestamp_col: str) -> Optional[np.ndarray]:
        """Extract temporal features as conditions."""
        try:
            if timestamp_col not in data.columns:
                logger.warning(f"Timestamp column '{timestamp_col}' not found")
                return None
            
            # Convert to datetime if needed
            timestamps = pd.to_datetime(data[timestamp_col])
            
            # Extract temporal features
            temporal_data = []
            
            # Hour (0-23) normalized to [0, 1]
            if 'hour' in self.temporal_features:
                hours = timestamps.dt.hour / 23.0
                temporal_data.append(hours.values)
            
            # Day of month (1-31) normalized to [0, 1]
            if 'day' in self.temporal_features:
                days = (timestamps.dt.day - 1) / 30.0
                temporal_data.append(days.values)
            
            # Month (1-12) normalized to [0, 1]
            if 'month' in self.temporal_features:
                months = (timestamps.dt.month - 1) / 11.0
                temporal_data.append(months.values)
            
            # Year (relative to minimum year)
            if 'year' in self.temporal_features:
                min_year = timestamps.dt.year.min()
                max_year = timestamps.dt.year.max()
                if max_year > min_year:
                    years = (timestamps.dt.year - min_year) / (max_year - min_year)
                else:
                    years = np.zeros(len(timestamps))
                temporal_data.append(years.values)
            
            # Weekday (0-6) normalized to [0, 1]
            if 'weekday' in self.temporal_features:
                weekdays = timestamps.dt.weekday / 6.0
                temporal_data.append(weekdays.values)
            
            if temporal_data:
                result = np.column_stack(temporal_data).astype(np.float32)
                logger.debug(f"Extracted temporal conditions shape: {result.shape}")
                return result
            else:
                return None
                
        except Exception as e:
            logger.error(f"Failed to extract temporal conditions: {str(e)}")
            return None
    
    def _normalize_conditions(self, conditions: np.ndarray) -> np.ndarray:
        """Normalize condition vectors."""
        try:
            # Simple min-max normalization
            normalized = np.copy(conditions)
            
            for i in range(conditions.shape[1]):
                col_data = conditions[:, i]
                col_min = np.min(col_data)
                col_max = np.max(col_data)
                
                if col_max > col_min:
                    normalized[:, i] = (col_data - col_min) / (col_max - col_min)
                else:
                    normalized[:, i] = 0.0
            
            return normalized
            
        except Exception as e:
            logger.warning(f"Failed to normalize conditions: {str(e)}")
            return conditions
    
    def _embed_conditions(self, conditions: np.ndarray) -> np.ndarray:
        """Apply embedding transformation to conditions."""
        # Simple linear transformation for now
        # In future, could use learned embeddings
        
        if conditions.shape[1] > self.condition_dim:
            # Reduce dimensions using PCA-like approach
            # For now, just take first few dimensions
            return conditions[:, :self.condition_dim]
        elif conditions.shape[1] < self.condition_dim:
            # Pad with zeros
            padding = np.zeros((conditions.shape[0], self.condition_dim - conditions.shape[1]))
            return np.concatenate([conditions, padding], axis=1)
        else:
            return conditions
    
    def _encode_conditions(self, conditions: np.ndarray) -> np.ndarray:
        """Apply encoding transformation to conditions."""
        # Simple encoding (identity for now)
        return conditions
    
    def _calculate_condition_dimensions(self):
        """Calculate total condition vector dimensions."""
        total_dim = 0
        
        # Explicit conditions
        total_dim += len(self.condition_columns)
        
        # Temporal conditions
        if self.use_temporal_conditions:
            total_dim += len(self.temporal_features)
        
        # Adjust for conditioning method
        if self.condition_method in ['embed', 'encode']:
            total_dim = self.condition_dim
        
        self.condition_shape = (total_dim,) if total_dim > 0 else (0,)
        logger.info(f"Calculated condition dimensions: {self.condition_shape}")
    
    def get_condition_info(self) -> Dict[str, Any]:
        """Get information about current condition setup."""
        return {
            'initialized': self.is_initialized,
            'has_conditions': self.has_conditions,
            'condition_columns': self.condition_columns,
            'condition_method': self.condition_method,
            'condition_shape': self.condition_shape,
            'temporal_features': self.temporal_features,
            'use_temporal_conditions': self.use_temporal_conditions
        }
    
    def reset(self):
        """Reset condition manager state."""
        self.condition_encoders = {}
        self.condition_stats = {}
        self.is_initialized = False
        self.has_conditions = False
        self.condition_shape = None
        
        logger.info("ConditionManager reset")
