#!/usr/bin/env python3
"""
Feature Processor Module

This module handles feature processing, extraction, and validation,
providing focused functionality for feature engineering and data transformation.
"""

import logging
import numpy as np
from typing import Dict, Any, List


class FeatureProcessor:
    """Handles feature processing, extraction, and validation."""
    
    def __init__(self, params: Dict[str, Any], logger: logging.Logger):
        """Initialize feature processor."""
        self.params = params
        self.logger = logger
        
        # Feature configuration
        self.full_feature_names = params.get("full_feature_names_ordered", [])
        self.decoder_output_features = params.get("decoder_output_feature_names", [])
        self.ohlc_features = params.get("ohlc_feature_names", ["OPEN", "HIGH", "LOW", "CLOSE"])
        self.ti_features = params.get("ti_feature_names", [])
        
        # Create feature mapping
        self.feature_to_idx = {name: i for i, name in enumerate(self.full_feature_names)}
        
        self.logger.info("FeatureProcessor initialized")
        self.logger.info(f"Full features: {len(self.full_feature_names)}")
        self.logger.info(f"Decoder output features: {len(self.decoder_output_features)}")
        self.logger.info(f"Technical indicators: {len(self.ti_features)}")
    
    def validate_feature_consistency(self) -> bool:
        """
        Validate that feature configurations are consistent.
        
        Returns:
            True if consistent, False otherwise
        """
        try:
            # Check that decoder output features are subset of full features
            for feature in self.decoder_output_features:
                if feature not in self.full_feature_names:
                    self.logger.error(f"Decoder output feature '{feature}' not in full feature list")
                    return False
            
            # Check that OHLC features are subset of full features
            for feature in self.ohlc_features:
                if feature not in self.full_feature_names:
                    self.logger.error(f"OHLC feature '{feature}' not in full feature list")
                    return False
            
            # Check that TI features are subset of full features
            for feature in self.ti_features:
                if feature not in self.full_feature_names:
                    self.logger.error(f"TI feature '{feature}' not in full feature list")
                    return False
            
            # Check for duplicates in full feature names
            if len(self.full_feature_names) != len(set(self.full_feature_names)):
                self.logger.error("Duplicate features found in full feature list")
                return False
            
            self.logger.info("Feature consistency validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating feature consistency: {e}")
            return False
    
    def extract_decoder_features(self, full_data: np.ndarray) -> np.ndarray:
        """
        Extract decoder output features from full feature array.
        
        Args:
            full_data: Full feature array
            
        Returns:
            Array with only decoder output features
        """
        try:
            if len(self.decoder_output_features) == 0:
                return full_data
            
            # Get indices for decoder output features
            decoder_indices = []
            for feature_name in self.decoder_output_features:
                if feature_name in self.feature_to_idx:
                    decoder_indices.append(self.feature_to_idx[feature_name])
                else:
                    self.logger.warning(f"Decoder feature '{feature_name}' not found in feature mapping")
            
            if not decoder_indices:
                self.logger.error("No valid decoder output feature indices found")
                return full_data
            
            # Extract features based on array dimensions
            if full_data.ndim == 1:
                decoder_data = full_data[decoder_indices]
            elif full_data.ndim == 2:
                decoder_data = full_data[:, decoder_indices]
            elif full_data.ndim == 3:
                decoder_data = full_data[:, :, decoder_indices]
            else:
                self.logger.error(f"Unsupported array dimensions: {full_data.ndim}")
                return full_data
            
            return decoder_data
            
        except Exception as e:
            self.logger.error(f"Error extracting decoder features: {e}")
            return full_data
    
    def extract_ohlc_features(self, full_data: np.ndarray) -> np.ndarray:
        """
        Extract OHLC features from full feature array.
        
        Args:
            full_data: Full feature array
            
        Returns:
            Array with only OHLC features
        """
        try:
            # Get indices for OHLC features
            ohlc_indices = []
            for feature_name in self.ohlc_features:
                if feature_name in self.feature_to_idx:
                    ohlc_indices.append(self.feature_to_idx[feature_name])
                else:
                    self.logger.warning(f"OHLC feature '{feature_name}' not found in feature mapping")
            
            if not ohlc_indices:
                self.logger.error("No valid OHLC feature indices found")
                return full_data
            
            # Extract features based on array dimensions
            if full_data.ndim == 1:
                ohlc_data = full_data[ohlc_indices]
            elif full_data.ndim == 2:
                ohlc_data = full_data[:, ohlc_indices]
            elif full_data.ndim == 3:
                ohlc_data = full_data[:, :, ohlc_indices]
            else:
                self.logger.error(f"Unsupported array dimensions: {full_data.ndim}")
                return full_data
            
            return ohlc_data
            
        except Exception as e:
            self.logger.error(f"Error extracting OHLC features: {e}")
            return full_data
    
    def get_feature_indices(self, feature_names: List[str]) -> List[int]:
        """
        Get indices for a list of feature names.
        
        Args:
            feature_names: List of feature names
            
        Returns:
            List of corresponding indices
        """
        indices = []
        for name in feature_names:
            if name in self.feature_to_idx:
                indices.append(self.feature_to_idx[name])
            else:
                self.logger.warning(f"Feature '{name}' not found in mapping")
        return indices
    
    def get_feature_info(self) -> Dict[str, Any]:
        """Get information about features."""
        return {
            "total_features": len(self.full_feature_names),
            "decoder_features": len(self.decoder_output_features),
            "ohlc_features": len(self.ohlc_features),
            "ti_features": len(self.ti_features),
            "feature_mapping": self.feature_to_idx,
            "full_feature_names": self.full_feature_names,
            "decoder_output_features": self.decoder_output_features,
            "ohlc_features": self.ohlc_features,
            "ti_features": self.ti_features
        }
    
    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information."""
        return {
            "full_feature_count": len(self.full_feature_names),
            "decoder_feature_count": len(self.decoder_output_features),
            "ohlc_feature_count": len(self.ohlc_features),
            "ti_feature_count": len(self.ti_features),
            "feature_consistency": self.validate_feature_consistency(),
            "feature_mapping_size": len(self.feature_to_idx)
        }
