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
            # Use the dynamically calculated condition_dim in the log message
            logger.info(f"ConditionManager initialized with condition shape: {self.condition_shape}, dimension: {self.condition_dim}")
            
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
        """Extract temporal features as conditions using cyclical encoding."""
        try:
            if timestamp_col not in data.columns:
                logger.warning(f"Timestamp column '{timestamp_col}' not found")
                return None
            
            timestamp_data = data[timestamp_col]
            if not pd.api.types.is_datetime64_any_dtype(timestamp_data):
                timestamps = pd.to_datetime(timestamp_data)
            else:
                timestamps = timestamp_data
            
            if not isinstance(timestamps, pd.Series):
                timestamps = pd.Series(timestamps)

            temporal_data_parts = []
            
            date_features_to_use = self.config.get("feeder_date_features_for_conditioning", [])
            if not date_features_to_use:
                logger.debug("No date features specified for temporal conditioning.")
                return None

            for feature_name in date_features_to_use:
                values: Optional[pd.Series] = None
                period: Union[float, int, pd.Series] = 0.0

                if feature_name == "hour_of_day":
                    values = timestamps.dt.hour
                    period = 24.0
                elif feature_name == "day_of_week": # Monday=0, Sunday=6
                    values = timestamps.dt.weekday
                    period = 7.0
                elif feature_name == "day_of_month":
                    values = timestamps.dt.day
                    period = timestamps.dt.days_in_month # Series, for accurate period per month
                elif feature_name == "day_of_year":
                    values = timestamps.dt.dayofyear
                    period = self.config.get("feeder_max_day_of_year", 366.0) # Use configured max (e.g., 366 for leap years)
                elif feature_name == "month":
                    values = timestamps.dt.month
                    period = self.config.get("feeder_max_month", 12.0)
                else:
                    logger.warning(f"Unsupported temporal feature specified: {feature_name}")
                    continue
                
                if values is not None:
                    # Ensure 'values' is a Series for consistent operations with 'period' if it's a Series
                    if not isinstance(values, pd.Series):
                         values = pd.Series(values)

                    # Normalize values for cyclical encoding.
                    # For 1-indexed values (like month, day_of_month, day_of_year), 
                    # using value/period directly is common.
                    # For 0-indexed values (like hour, day_of_week), value/period is also common.
                    # The key is that the value should represent a position within the cycle of length 'period'.
                    
                    # sin_transformed = np.sin(2 * np.pi * (values - (1 if feature_name in ["day_of_month", "day_of_year", "month"] else 0)) / period)
                    # cos_transformed = np.cos(2 * np.pi * (values - (1 if feature_name in ["day_of_month", "day_of_year", "month"] else 0)) / period)
                    
                    # Using values directly as they are (0-indexed or 1-indexed) with their respective periods
                    sin_transformed = np.sin(2 * np.pi * values / period)
                    cos_transformed = np.cos(2 * np.pi * values / period)
                    
                    temporal_data_parts.append(sin_transformed.values)
                    temporal_data_parts.append(cos_transformed.values)

            if temporal_data_parts:
                result = np.column_stack(temporal_data_parts).astype(np.float32)
                logger.debug(f"Extracted temporal conditions shape: {result.shape}")
                return result
            else:
                logger.debug("No temporal data parts were generated.")
                return None
                
        except Exception as e:
            logger.error(f"Failed to extract temporal conditions: {str(e)}", exc_info=True)
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
        """Calculate total dimension of condition vectors."""
        total_dim = 0
        
        # Explicit conditions (e.g., fundamental features)
        # self.condition_columns is populated by 'fundamental_features_for_conditioning'
        # from the feeder_config in FeederPlugin.initialize()
        if self.condition_columns:
            total_dim += len(self.condition_columns)
            logger.debug(f"Explicit condition columns ({len(self.condition_columns)}): {self.condition_columns}")
        
        # Temporal conditions
        if self.use_temporal_conditions:
            date_features_to_use = self.config.get("feeder_date_features_for_conditioning", [])
            num_temporal_features = len(date_features_to_use)
            if num_temporal_features > 0:
                total_dim += num_temporal_features * 2 # Each temporal feature yields sin and cos
                logger.debug(f"Temporal features ({num_temporal_features} -> {num_temporal_features * 2} dims): {date_features_to_use}")
            
        self.condition_dim = total_dim
        self.condition_shape = (None, total_dim) if total_dim > 0 else None # Corrected line
        
        # This log message was moved to initialize() to show the final calculated dimension
        # logger.info(f"Calculated condition dimension: {self.condition_dim}")

    def get_condition_dim(self) -> int:
        """Get the current condition dimension."""
        return self.condition_dim
    
    def get_condition_shape(self) -> Tuple[Optional[int], Optional[int]]:
        """Get the current condition shape."""
        return self.condition_shape
    
    def get_condition_columns(self) -> List[str]:
        """Get the list of condition columns."""
        return self.condition_columns
    
    def get_condition_method(self) -> str:
        """Get the current condition method."""
        return self.condition_method
    
    def is_condition_manager_initialized(self) -> bool:
        """Check if the condition manager is initialized."""
        return self.is_initialized
    
    def has_valid_conditions(self) -> bool:
        """Check if there are valid conditions available."""
        return self.has_conditions
