"""
Latent Validator Module

Handles validation and quality control of latent vectors.
Ensures latent vectors meet requirements for GAN processing.
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)


class LatentValidator:
    """
    Handles validation and quality control of latent vectors.
    
    Ensures latent vectors meet dimensional, numerical, and quality
    requirements for proper GAN processing.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the latent validator."""
        self.config = config
        
        # Validation parameters
        self.expected_latent_dim = config.get('latent_dim', 128)
        self.max_norm = config.get('max_latent_norm', 10.0)
        self.min_norm = config.get('min_latent_norm', 0.01)
        self.check_distribution = config.get('check_latent_distribution', True)
        self.distribution_tolerance = config.get('distribution_tolerance', 3.0)
        
        # Quality thresholds
        self.nan_tolerance = config.get('nan_tolerance', 0.0)  # No NaN allowed
        self.inf_tolerance = config.get('inf_tolerance', 0.0)  # No Inf allowed
        self.zero_tolerance = config.get('zero_tolerance', 0.1)  # Max 10% zeros
        
        # Repair options
        self.auto_repair = config.get('auto_repair_latents', True)
        self.repair_method = config.get('repair_method', 'clamp')
        
        # State tracking
        self.is_initialized = False
        self.validation_stats = {}
        
        logger.info("LatentValidator initialized")
    
    def initialize(self) -> bool:
        """Initialize the latent validator."""
        try:
            # Validate parameters
            self._validate_parameters()
            
            # Setup validation statistics
            self._setup_validation_stats()
            
            self.is_initialized = True
            logger.info("LatentValidator initialization completed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize LatentValidator: {e}")
            return False
    
    def validate_latents(self, latents: np.ndarray) -> Optional[np.ndarray]:
        """
        Validate and optionally repair latent vectors.
        
        Args:
            latents: Input latent vectors to validate
            
        Returns:
            Validated (and possibly repaired) latent vectors, or None if validation fails
        """
        try:
            if not self.is_initialized:
                raise ValueError("LatentValidator not initialized")
            
            if latents is None:
                logger.error("Received None latents")
                return None
            
            # Perform validation checks
            validation_results = self._perform_validation_checks(latents)
            
            # Update statistics
            self._update_validation_stats(validation_results)
            
            # Handle validation results
            if validation_results['is_valid']:
                logger.debug("Latent vectors passed validation")
                return latents
            elif self.auto_repair and validation_results['is_repairable']:
                logger.warning("Latent vectors failed validation, attempting repair")
                repaired_latents = self._repair_latents(latents, validation_results)
                
                # Re-validate repaired latents
                repair_validation = self._perform_validation_checks(repaired_latents)
                if repair_validation['is_valid']:
                    logger.info("Latent vectors successfully repaired")
                    return repaired_latents
                else:
                    logger.error("Failed to repair latent vectors")
                    return None
            else:
                logger.error("Latent vectors failed validation and cannot be repaired")
                return None
                
        except Exception as e:
            logger.error(f"Error validating latents: {e}")
            return None
    
    def _perform_validation_checks(self, latents: np.ndarray) -> Dict[str, Any]:
        """Perform comprehensive validation checks on latent vectors."""
        results = {
            'is_valid': True,
            'is_repairable': True,
            'issues': [],
            'shape_valid': True,
            'values_valid': True,
            'distribution_valid': True,
            'stats': {}
        }
        
        # Shape validation
        shape_check = self._check_shape(latents)
        results.update(shape_check)
        if not shape_check['shape_valid']:
            results['is_valid'] = False
            if not shape_check.get('shape_repairable', False):
                results['is_repairable'] = False
        
        # Value validation
        value_check = self._check_values(latents)
        results.update(value_check)
        if not value_check['values_valid']:
            results['is_valid'] = False
        
        # Distribution validation
        if self.check_distribution:
            dist_check = self._check_distribution(latents)
            results.update(dist_check)
            if not dist_check['distribution_valid']:
                results['is_valid'] = False
        
        return results
    
    def _check_shape(self, latents: np.ndarray) -> Dict[str, Any]:
        """Check latent vector shape requirements."""
        results = {
            'shape_valid': True,
            'shape_repairable': True,
            'shape_issues': []
        }
        
        # Check if array is not empty
        if latents.size == 0:
            results['shape_valid'] = False
            results['shape_repairable'] = False
            results['shape_issues'].append('Empty latent array')
            return results
        
        # Check dimensions
        if latents.ndim != 2:
            results['shape_valid'] = False
            results['shape_issues'].append(f'Invalid dimensions: {latents.ndim}, expected 2')
            if latents.ndim == 1:
                results['shape_repairable'] = True
            else:
                results['shape_repairable'] = False
        
        # Check latent dimension
        if latents.ndim >= 2 and latents.shape[1] != self.expected_latent_dim:
            results['shape_valid'] = False
            results['shape_issues'].append(
                f'Invalid latent dimension: {latents.shape[1]}, expected {self.expected_latent_dim}'
            )
            results['shape_repairable'] = False
        
        return results
    
    def _check_values(self, latents: np.ndarray) -> Dict[str, Any]:
        """Check latent vector value requirements."""
        results = {
            'values_valid': True,
            'value_issues': [],
            'value_stats': {}
        }
        
        # Check for NaN values
        nan_count = np.isnan(latents).sum()
        nan_ratio = nan_count / latents.size
        results['value_stats']['nan_ratio'] = nan_ratio
        
        if nan_ratio > self.nan_tolerance:
            results['values_valid'] = False
            results['value_issues'].append(f'Too many NaN values: {nan_ratio:.4f}')
        
        # Check for infinite values
        inf_count = np.isinf(latents).sum()
        inf_ratio = inf_count / latents.size
        results['value_stats']['inf_ratio'] = inf_ratio
        
        if inf_ratio > self.inf_tolerance:
            results['values_valid'] = False
            results['value_issues'].append(f'Too many infinite values: {inf_ratio:.4f}')
        
        # Check for zero values
        zero_count = (latents == 0).sum()
        zero_ratio = zero_count / latents.size
        results['value_stats']['zero_ratio'] = zero_ratio
        
        if zero_ratio > self.zero_tolerance:
            results['values_valid'] = False
            results['value_issues'].append(f'Too many zero values: {zero_ratio:.4f}')
        
        # Check norms
        norms = np.linalg.norm(latents, axis=1)
        results['value_stats']['mean_norm'] = np.mean(norms)
        results['value_stats']['std_norm'] = np.std(norms)
        results['value_stats']['min_norm'] = np.min(norms)
        results['value_stats']['max_norm'] = np.max(norms)
        
        if np.any(norms < self.min_norm):
            results['values_valid'] = False
            results['value_issues'].append(f'Norms too small, min: {np.min(norms):.6f}')
        
        if np.any(norms > self.max_norm):
            results['values_valid'] = False
            results['value_issues'].append(f'Norms too large, max: {np.max(norms):.6f}')
        
        return results
    
    def _check_distribution(self, latents: np.ndarray) -> Dict[str, Any]:
        """Check latent vector distribution properties."""
        results = {
            'distribution_valid': True,
            'distribution_issues': [],
            'distribution_stats': {}
        }
        
        try:
            # Check mean (should be close to 0 for standard normal)
            mean_vals = np.mean(latents, axis=0)
            overall_mean = np.mean(np.abs(mean_vals))
            results['distribution_stats']['mean_deviation'] = overall_mean
            
            if overall_mean > self.distribution_tolerance:
                results['distribution_valid'] = False
                results['distribution_issues'].append(f'Mean deviation too large: {overall_mean:.4f}')
            
            # Check standard deviation (should be close to 1 for standard normal)
            std_vals = np.std(latents, axis=0)
            mean_std = np.mean(std_vals)
            results['distribution_stats']['mean_std'] = mean_std
            
            if abs(mean_std - 1.0) > self.distribution_tolerance:
                results['distribution_valid'] = False
                results['distribution_issues'].append(f'Standard deviation deviation: {mean_std:.4f}')
            
        except Exception as e:
            logger.warning(f"Error checking distribution: {e}")
            results['distribution_valid'] = False
            results['distribution_issues'].append(f'Distribution check failed: {e}')
        
        return results
    
    def _repair_latents(self, latents: np.ndarray, validation_results: Dict[str, Any]) -> np.ndarray:
        """Attempt to repair invalid latent vectors."""
        repaired = latents.copy()
        
        # Repair shape issues
        if not validation_results['shape_valid']:
            repaired = self._repair_shape(repaired, validation_results)
        
        # Repair value issues
        if not validation_results['values_valid']:
            repaired = self._repair_values(repaired, validation_results)
        
        return repaired
    
    def _repair_shape(self, latents: np.ndarray, validation_results: Dict[str, Any]) -> np.ndarray:
        """Repair shape-related issues."""
        # Reshape 1D to 2D if possible
        if latents.ndim == 1 and len(latents) == self.expected_latent_dim:
            return latents.reshape(1, -1)
        
        # For other shape issues, return as-is (not repairable)
        return latents
    
    def _repair_values(self, latents: np.ndarray, validation_results: Dict[str, Any]) -> np.ndarray:
        """Repair value-related issues."""
        repaired = latents.copy()
        
        # Handle NaN and infinite values
        if np.any(np.isnan(repaired)) or np.any(np.isinf(repaired)):
            if self.repair_method == 'zero':
                repaired = np.nan_to_num(repaired, nan=0.0, posinf=0.0, neginf=0.0)
            elif self.repair_method == 'clamp':
                repaired = np.nan_to_num(repaired, nan=0.0, posinf=self.max_norm, neginf=-self.max_norm)
            else:
                repaired = np.nan_to_num(repaired, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Handle norm issues
        norms = np.linalg.norm(repaired, axis=1, keepdims=True)
        
        # Fix very small norms
        small_norm_mask = norms < self.min_norm
        if np.any(small_norm_mask):
            repaired[small_norm_mask.flatten()] = np.random.normal(0, 1, (np.sum(small_norm_mask), repaired.shape[1]))
        
        # Fix very large norms
        large_norm_mask = norms > self.max_norm
        if np.any(large_norm_mask):
            # Normalize to max_norm
            repaired[large_norm_mask.flatten()] = (
                repaired[large_norm_mask.flatten()] / norms[large_norm_mask] * self.max_norm
            )
        
        return repaired
    
    def _validate_parameters(self):
        """Validate validator parameters."""
        if self.expected_latent_dim <= 0:
            raise ValueError(f"Invalid latent dimension: {self.expected_latent_dim}")
        
        if self.max_norm <= self.min_norm:
            raise ValueError(f"Invalid norm range: min={self.min_norm}, max={self.max_norm}")
        
        if not 0 <= self.nan_tolerance <= 1:
            raise ValueError(f"Invalid NaN tolerance: {self.nan_tolerance}")
    
    def _setup_validation_stats(self):
        """Setup validation statistics tracking."""
        self.validation_stats = {
            'total_validations': 0,
            'passed_validations': 0,
            'failed_validations': 0,
            'repaired_validations': 0,
            'common_issues': {}
        }
    
    def _update_validation_stats(self, validation_results: Dict[str, Any]):
        """Update validation statistics."""
        self.validation_stats['total_validations'] += 1
        
        if validation_results['is_valid']:
            self.validation_stats['passed_validations'] += 1
        else:
            self.validation_stats['failed_validations'] += 1
            
            # Track common issues
            all_issues = (
                validation_results.get('shape_issues', []) +
                validation_results.get('value_issues', []) +
                validation_results.get('distribution_issues', [])
            )
            
            for issue in all_issues:
                self.validation_stats['common_issues'][issue] = (
                    self.validation_stats['common_issues'].get(issue, 0) + 1
                )
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get current validation statistics."""
        return self.validation_stats.copy()
    
    def is_ready(self) -> bool:
        """Check if the validator is ready for use."""
        return self.is_initialized
    
    def cleanup(self):
        """Cleanup validator resources."""
        self.validation_stats.clear()
        self.is_initialized = False
        logger.info("LatentValidator cleanup completed")
