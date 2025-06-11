#!/usr/bin/env python3
"""
Feature Validator Module

Validates feature name consistency and ensures all configured feature lists
are subsets of the full feature names ordered list.
"""

from typing import Dict, Any, List, Set
import logging


class FeatureValidator:
    """Validates feature name consistency across different feature lists."""
    
    def __init__(self, full_feature_names_ordered: List[str], logger: logging.Logger):
        """
        Initialize validator with the master feature list.
        
        Args:
            full_feature_names_ordered: Complete ordered list of all features
            logger: Logger instance
        """
        self.full_feature_names_ordered = full_feature_names_ordered
        self.full_set = set(full_feature_names_ordered)
        self.logger = logger
        
        if not self.full_set:
            self.logger.error("'full_feature_names_ordered' is empty.")
            raise ValueError("'full_feature_names_ordered' cannot be empty and must be configured.")
        self.logger.debug(f"FeatureValidator initialized with {len(full_feature_names_ordered)} features.")
    
    def validate_feature_name_consistency(self, params: Dict[str, Any]) -> None:
        """
        Validate that all configured feature name lists are subsets of 
        'full_feature_names_ordered' and that critical feature lists are not empty.
        
        Args:
            params: Dictionary containing feature configuration parameters
            
        Raises:
            ValueError: If validation fails
        """
        self.logger.info("FeatureValidator: Validating feature name consistency...")
        
        # Validate critical feature lists
        self._check_subset("decoder_output_feature_names", params, critical=True)
        self._check_subset("ohlc_feature_names", params, critical=True)
        self._check_subset("ti_feature_names", params, critical=False)
        
        # Validate date conditional features (including sin/cos transforms)
        self._validate_date_conditional_features(params)
        
        # Validate feeder conditional features
        self._check_subset("feeder_conditional_feature_names", params, critical=False)
        
        # Validate decoder input names
        self._validate_decoder_input_names(params)
        
        self.logger.info("FeatureValidator: Feature name consistency validation complete.")
    
    def _check_subset(self, list_name: str, params: Dict[str, Any], critical: bool = False) -> None:
        """
        Check if a feature list is a valid subset of full_feature_names_ordered.
        
        Args:
            list_name: Name of the parameter to check
            params: Parameter dictionary
            critical: Whether this list is required to be non-empty
        """
        feature_list = params.get(list_name, [])
        
        if critical and not feature_list:
            raise ValueError(f"'{list_name}' is a critical parameter and cannot be empty.")
        
        current_set = set(feature_list)
        if not current_set.issubset(self.full_set):
            missing = current_set - self.full_set
            raise ValueError(
                f"Features in '{list_name}' are not all present in 'full_feature_names_ordered'. "
                f"Missing: {missing}. Ensure '{list_name}' only contains features from "
                f"'full_feature_names_ordered'."
            )
        
        self.logger.debug(f"FeatureValidator: Feature list '{list_name}' validated successfully against 'full_feature_names_ordered'.")
    
    def _validate_date_conditional_features(self, params: Dict[str, Any]) -> None:
        """
        Validate date conditional features including their sin/cos transformed versions.
        
        Args:
            params: Parameter dictionary
        """
        date_cond_original_names = params.get("date_conditional_feature_names", [])
        
        if not date_cond_original_names:
            return
        
        # Generate expected transformed names
        transformed_date_cond_names = []
        for name in date_cond_original_names:
            transformed_date_cond_names.append(f"{name}_sin")
            transformed_date_cond_names.append(f"{name}_cos")
        
        date_cond_set = set(transformed_date_cond_names)
        if not date_cond_set.issubset(self.full_set):
            missing_transformed = date_cond_set - self.full_set
            raise ValueError(
                f"Transformed date conditional features (from 'date_conditional_feature_names') "
                f"are not all present in 'full_feature_names_ordered'. "
                f"Missing: {missing_transformed}. "
                f"Ensure sin/cos versions of date features are in 'full_feature_names_ordered'."
            )
        
        self.logger.debug("FeatureValidator: Transformed date conditional features validated successfully.")
    
    def _validate_decoder_input_names(self, params: Dict[str, Any]) -> None:
        """
        Validate that decoder input names are properly configured.
        
        Args:
            params: Parameter dictionary
        """
        required_input_names = [
            "decoder_input_name_latent", 
            "decoder_input_name_window", 
            "decoder_input_name_conditions", 
            "decoder_input_name_context"
        ]
        
        for input_name_key in required_input_names:
            if not params.get(input_name_key):
                raise ValueError(
                    f"Decoder input name parameter '{input_name_key}' is not configured "
                    f"in GeneratorPlugin.params."
                )
        
        self.logger.debug("FeatureValidator: All decoder input name parameters are configured.")
    
    def create_feature_index_mapping(self) -> Dict[str, int]:
        """
        Create a mapping from feature names to their indices.
        
        Returns:
            Dictionary mapping feature names to indices
        """
        return {name: i for i, name in enumerate(self.full_feature_names_ordered)}
    
    def get_num_features(self) -> int:
        """
        Get the total number of features.
        
        Returns:
            Number of features in full_feature_names_ordered
        """
        return len(self.full_feature_names_ordered)
