#!/usr/bin/env python3
"""
Normalization Handler Module

This module handles data normalization and denormalization operations,
providing focused functionality for data scaling and transformation.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List


class NormalizationHandler:
    """Handles data normalization and denormalization operations."""
    
    def __init__(self, params: Dict[str, Any], logger: logging.Logger):
        """Initialize normalization handler."""
        self.params = params
        self.logger = logger
        self.normalization_params = None
        self.initial_denormalized_close_anchor = None
        self.previous_normalized_close = None
        
        # Load normalization parameters if file provided
        norm_params_file = params.get("generator_normalization_params_file")
        if norm_params_file:
            self.load_normalization_params(norm_params_file)
        
        self.logger.info("NormalizationHandler initialized")
    
    def load_normalization_params(self, params_file: str) -> Optional[Dict[str, Dict[str, float]]]:
        """
        Load normalization parameters from JSON file.
        
        Args:
            params_file: Path to normalization parameters file
            
        Returns:
            Dictionary with normalization parameters or None if loading fails
        """
        if not params_file or not os.path.exists(params_file):
            self.logger.warning(f"Normalization params file not found: {params_file}")
            return None
        
        try:
            with open(params_file, 'r') as f:
                self.normalization_params = json.load(f)
            
            self.logger.info(f"Loaded normalization parameters from: {params_file}")
            self.logger.info(f"Available features: {list(self.normalization_params.keys())}")
            
            return self.normalization_params
            
        except Exception as e:
            self.logger.error(f"Error loading normalization parameters: {e}")
            return None
    
    def load_initial_close_anchor(self, data_file: str) -> Optional[float]:
        """
        Load initial CLOSE value anchor from data file.
        
        Args:
            data_file: Path to data file containing CLOSE values
            
        Returns:
            Initial CLOSE value or None if loading fails
        """
        if not data_file or not os.path.exists(data_file):
            self.logger.warning(f"Data file for initial CLOSE not found: {data_file}")
            return None
        
        try:
            # Try to read the file
            df = pd.read_csv(data_file)
            
            # Look for CLOSE column
            close_cols = [col for col in df.columns if 'close' in col.lower()]
            if not close_cols:
                self.logger.warning("No CLOSE column found in data file")
                return None
            
            close_col = close_cols[0]  # Use first CLOSE column found
            
            # Get the first non-null CLOSE value
            close_values = df[close_col].dropna()
            if len(close_values) == 0:
                self.logger.warning("No valid CLOSE values found")
                return None
            
            self.initial_denormalized_close_anchor = float(close_values.iloc[0])
            self.logger.info(f"Initial CLOSE anchor loaded: {self.initial_denormalized_close_anchor}")
            
            return self.initial_denormalized_close_anchor
            
        except Exception as e:
            self.logger.error(f"Error loading initial CLOSE anchor: {e}")
            return None
    
    def normalize_feature(self, values: np.ndarray, feature_name: str) -> np.ndarray:
        """
        Normalize feature values using stored parameters.
        
        Args:
            values: Raw feature values
            feature_name: Name of the feature
            
        Returns:
            Normalized values
        """
        if self.normalization_params is None or feature_name not in self.normalization_params:
            self.logger.warning(f"No normalization params for feature: {feature_name}")
            return values
        
        try:
            params = self.normalization_params[feature_name]
            min_val = params.get('min', 0.0)
            max_val = params.get('max', 1.0)
            
            # Min-max normalization
            if max_val != min_val:
                normalized = (values - min_val) / (max_val - min_val)
            else:
                normalized = np.zeros_like(values)
            
            return normalized
            
        except Exception as e:
            self.logger.error(f"Error normalizing feature {feature_name}: {e}")
            return values
    
    def denormalize_feature(self, values: np.ndarray, feature_name: str) -> np.ndarray:
        """
        Denormalize feature values using stored parameters.
        
        Args:
            values: Normalized feature values
            feature_name: Name of the feature
            
        Returns:
            Denormalized values
        """
        if self.normalization_params is None or feature_name not in self.normalization_params:
            self.logger.warning(f"No normalization params for feature: {feature_name}")
            return values
        
        try:
            params = self.normalization_params[feature_name]
            min_val = params.get('min', 0.0)
            max_val = params.get('max', 1.0)
            
            # Reverse min-max normalization
            denormalized = values * (max_val - min_val) + min_val
            
            return denormalized
            
        except Exception as e:
            self.logger.error(f"Error denormalizing feature {feature_name}: {e}")
            return values
    
    def denormalize_ohlc_data(self, normalized_data: np.ndarray, 
                             feature_names: List[str]) -> np.ndarray:
        """
        Denormalize OHLC data with special handling for price relationships.
        
        Args:
            normalized_data: Normalized OHLC data
            feature_names: List of feature names
            
        Returns:
            Denormalized OHLC data
        """
        try:
            denormalized = np.copy(normalized_data)
            
            # Get OHLC feature names
            ohlc_names = self.params.get("ohlc_feature_names", ["OPEN", "HIGH", "LOW", "CLOSE"])
            
            for i, feature_name in enumerate(feature_names):
                if feature_name in ohlc_names and i < denormalized.shape[-1]:
                    denormalized[..., i] = self.denormalize_feature(
                        normalized_data[..., i], feature_name
                    )
            
            return denormalized
            
        except Exception as e:
            self.logger.error(f"Error denormalizing OHLC data: {e}")
            return normalized_data
    
    def handle_log_returns(self, log_returns: np.ndarray, 
                          previous_close: Optional[float] = None) -> np.ndarray:
        """
        Convert log returns to actual prices.
        
        Args:
            log_returns: Log return values
            previous_close: Previous CLOSE price for conversion
            
        Returns:
            Actual price values
        """
        try:
            if previous_close is None:
                previous_close = self.initial_denormalized_close_anchor or 1.0
            
            # Convert log returns to prices
            prices = previous_close * np.exp(np.cumsum(log_returns, axis=-2))
            
            return prices
            
        except Exception as e:
            self.logger.error(f"Error converting log returns: {e}")
            return log_returns
    
    def get_feature_stats(self, feature_name: str) -> Dict[str, float]:
        """
        Get normalization statistics for a feature.
        
        Args:
            feature_name: Name of the feature
            
        Returns:
            Dictionary with min, max, and range values
        """
        if self.normalization_params is None or feature_name not in self.normalization_params:
            return {}
        
        params = self.normalization_params[feature_name]
        return {
            "min": params.get('min', 0.0),
            "max": params.get('max', 1.0),
            "range": params.get('max', 1.0) - params.get('min', 0.0)
        }
    
    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information."""
        debug_info = {
            "normalization_loaded": self.normalization_params is not None,
            "initial_close_anchor": self.initial_denormalized_close_anchor,
            "previous_normalized_close": self.previous_normalized_close
        }
        
        if self.normalization_params:
            debug_info["available_features"] = list(self.normalization_params.keys())
            debug_info["num_features"] = len(self.normalization_params)
        
        return debug_info
