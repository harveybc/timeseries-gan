"""
Data Preprocessor Module

Handles data preprocessing operations for the feeder plugin.
Manages data normalization, validation, and preparation for encoding.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Handles data preprocessing operations for the feeder plugin.
    
    Provides data cleaning, normalization, validation, and preparation
    services for feeding data to the encoder.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the data preprocessor."""
        self.config = config
        
        # Preprocessing parameters
        self.normalization_method = config.get('normalization_method', 'standard')
        self.handle_missing = config.get('handle_missing', 'interpolate')
        self.outlier_method = config.get('outlier_method', 'clip')
        self.outlier_threshold = config.get('outlier_threshold', 3.0)
        
        # State tracking
        self.scaler = None
        self.is_fitted = False
        self.feature_names = None
        self.original_shape = None
        
        # Statistics
        self.data_stats = {
            'mean': None,
            'std': None,
            'min': None,
            'max': None,
            'missing_count': 0,
            'outlier_count': 0
        }
        
        logger.info("DataPreprocessor initialized")
    
    def fit(self, data: Union[np.ndarray, pd.DataFrame]) -> bool:
        """
        Fit the preprocessor to training data.
        
        Args:
            data: Training data to fit on
            
        Returns:
            bool: True if fitting successful
        """
        try:
            # Convert to numpy if needed
            if isinstance(data, pd.DataFrame):
                self.feature_names = data.columns.tolist()
                data = data.values
            
            self.original_shape = data.shape
            
            # Handle missing values for fitting
            clean_data = self._handle_missing_values(data, is_fitting=True)
            
            # Initialize and fit scaler
            if self.normalization_method == 'standard':
                self.scaler = StandardScaler()
            elif self.normalization_method == 'minmax':
                self.scaler = MinMaxScaler()
            elif self.normalization_method == 'none':
                self.scaler = None
            else:
                logger.warning(f"Unknown normalization method: {self.normalization_method}, using standard")
                self.scaler = StandardScaler()
            
            if self.scaler is not None:
                self.scaler.fit(clean_data)
            
            # Calculate and store statistics
            self._calculate_statistics(clean_data)
            
            self.is_fitted = True
            logger.info(f"DataPreprocessor fitted on data shape: {self.original_shape}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to fit preprocessor: {str(e)}")
            return False
    
    def transform(self, data: Union[np.ndarray, pd.DataFrame]) -> Optional[np.ndarray]:
        """
        Transform data using fitted preprocessor.
        
        Args:
            data: Data to transform
            
        Returns:
            Optional[np.ndarray]: Transformed data or None if failed
        """
        if not self.is_fitted:
            logger.error("Preprocessor not fitted. Call fit() first.")
            return None
        
        try:
            # Convert to numpy if needed
            if isinstance(data, pd.DataFrame):
                data = data.values
            
            # Handle missing values
            clean_data = self._handle_missing_values(data, is_fitting=False)
            
            # Handle outliers
            clean_data = self._handle_outliers(clean_data)
            
            # Apply normalization
            if self.scaler is not None:
                normalized_data = self.scaler.transform(clean_data)
            else:
                normalized_data = clean_data
            
            logger.debug(f"Transformed data shape: {data.shape} -> {normalized_data.shape}")
            return normalized_data
            
        except Exception as e:
            logger.error(f"Failed to transform data: {str(e)}")
            return None
    
    def fit_transform(self, data: Union[np.ndarray, pd.DataFrame]) -> Optional[np.ndarray]:
        """
        Fit preprocessor and transform data in one step.
        
        Args:
            data: Data to fit and transform
            
        Returns:
            Optional[np.ndarray]: Transformed data or None if failed
        """
        if not self.fit(data):
            return None
        
        return self.transform(data)
    
    def inverse_transform(self, data: np.ndarray) -> Optional[np.ndarray]:
        """
        Inverse transform normalized data back to original scale.
        
        Args:
            data: Normalized data to inverse transform
            
        Returns:
            Optional[np.ndarray]: Original scale data or None if failed
        """
        if not self.is_fitted:
            logger.error("Preprocessor not fitted. Cannot inverse transform.")
            return None
        
        try:
            if self.scaler is not None:
                return self.scaler.inverse_transform(data)
            else:
                return data
                
        except Exception as e:
            logger.error(f"Failed to inverse transform data: {str(e)}")
            return None
    
    def _handle_missing_values(self, data: np.ndarray, is_fitting: bool = False) -> np.ndarray:
        """Handle missing values in the data."""
        if not np.isnan(data).any():
            return data
        
        missing_count = np.isnan(data).sum()
        if is_fitting:
            self.data_stats['missing_count'] = missing_count
        
        logger.debug(f"Handling {missing_count} missing values using method: {self.handle_missing}")
        
        if self.handle_missing == 'interpolate':
            # Linear interpolation
            for col in range(data.shape[1]):
                mask = np.isnan(data[:, col])
                if mask.any():
                    data[mask, col] = np.interp(
                        np.where(mask)[0],
                        np.where(~mask)[0],
                        data[~mask, col]
                    )
        
        elif self.handle_missing == 'forward_fill':
            # Forward fill
            for col in range(data.shape[1]):
                mask = np.isnan(data[:, col])
                if mask.any():
                    data[:, col] = pd.Series(data[:, col]).fillna(method='ffill').values
        
        elif self.handle_missing == 'backward_fill':
            # Backward fill
            for col in range(data.shape[1]):
                mask = np.isnan(data[:, col])
                if mask.any():
                    data[:, col] = pd.Series(data[:, col]).fillna(method='bfill').values
        
        elif self.handle_missing == 'mean':
            # Fill with column mean
            for col in range(data.shape[1]):
                mask = np.isnan(data[:, col])
                if mask.any():
                    mean_val = np.nanmean(data[:, col])
                    data[mask, col] = mean_val
        
        elif self.handle_missing == 'drop':
            # Remove rows with any missing values
            data = data[~np.isnan(data).any(axis=1)]
        
        return data
    
    def _handle_outliers(self, data: np.ndarray) -> np.ndarray:
        """Handle outliers in the data."""
        if self.outlier_method == 'none':
            return data
        
        outlier_count = 0
        
        if self.outlier_method == 'clip':
            # Clip outliers using z-score
            for col in range(data.shape[1]):
                col_data = data[:, col]
                z_scores = np.abs((col_data - np.mean(col_data)) / np.std(col_data))
                outliers = z_scores > self.outlier_threshold
                
                if outliers.any():
                    outlier_count += outliers.sum()
                    # Clip to threshold
                    lower_bound = np.mean(col_data) - self.outlier_threshold * np.std(col_data)
                    upper_bound = np.mean(col_data) + self.outlier_threshold * np.std(col_data)
                    data[:, col] = np.clip(col_data, lower_bound, upper_bound)
        
        elif self.outlier_method == 'remove':
            # Remove outlier rows
            z_scores = np.abs((data - np.mean(data, axis=0)) / np.std(data, axis=0))
            outlier_mask = (z_scores > self.outlier_threshold).any(axis=1)
            outlier_count = outlier_mask.sum()
            data = data[~outlier_mask]
        
        self.data_stats['outlier_count'] = outlier_count
        
        if outlier_count > 0:
            logger.debug(f"Handled {outlier_count} outliers using method: {self.outlier_method}")
        
        return data
    
    def _calculate_statistics(self, data: np.ndarray):
        """Calculate and store data statistics."""
        try:
            self.data_stats.update({
                'mean': np.mean(data, axis=0),
                'std': np.std(data, axis=0),
                'min': np.min(data, axis=0),
                'max': np.max(data, axis=0)
            })
        except Exception as e:
            logger.warning(f"Failed to calculate statistics: {str(e)}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get data preprocessing statistics."""
        return self.data_stats.copy()
    
    def validate_data(self, data: Union[np.ndarray, pd.DataFrame]) -> bool:
        """
        Validate data for preprocessing.
        
        Args:
            data: Data to validate
            
        Returns:
            bool: True if data is valid
        """
        try:
            if isinstance(data, pd.DataFrame):
                data = data.values
            
            if data is None or len(data) == 0:
                logger.error("Data is empty")
                return False
            
            if not isinstance(data, np.ndarray):
                logger.error("Data must be numpy array or pandas DataFrame")
                return False
            
            if data.ndim != 2:
                logger.error(f"Data must be 2D, got {data.ndim}D")
                return False
            
            # Check for infinite values
            if np.isinf(data).any():
                logger.warning("Data contains infinite values")
            
            # Check data types
            if not np.issubdtype(data.dtype, np.number):
                logger.error("Data must be numeric")
                return False
            
            logger.debug(f"Data validation passed for shape: {data.shape}")
            return True
            
        except Exception as e:
            logger.error(f"Data validation failed: {str(e)}")
            return False
    
    def get_config(self) -> Dict[str, Any]:
        """Get current preprocessor configuration."""
        return {
            'normalization_method': self.normalization_method,
            'handle_missing': self.handle_missing,
            'outlier_method': self.outlier_method,
            'outlier_threshold': self.outlier_threshold,
            'is_fitted': self.is_fitted,
            'feature_names': self.feature_names,
            'original_shape': self.original_shape
        }
    
    def reset(self):
        """Reset the preprocessor to unfitted state."""
        self.scaler = None
        self.is_fitted = False
        self.feature_names = None
        self.original_shape = None
        
        # Reset statistics
        self.data_stats = {
            'mean': None,
            'std': None,
            'min': None,
            'max': None,
            'missing_count': 0,
            'outlier_count': 0
        }
        
        logger.info("DataPreprocessor reset to unfitted state")
