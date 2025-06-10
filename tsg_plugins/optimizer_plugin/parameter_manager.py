"""
Parameter Manager Module

Manages hyperparameter bounds, validation, and conversion for optimization.
"""

import logging
from typing import Dict, Any, List, Union, Tuple
import random

logger = logging.getLogger(__name__)


class ParameterManager:
    """Manages hyperparameters for genetic algorithm optimization."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize parameter manager with configuration."""
        self.config = config
        
        # Default parameter bounds
        self.default_bounds = {
            "latent_dim": (16, 128),
            "batch_size": (16, 64),
            "learning_rate": (1e-5, 1e-2)
        }
        
        # Get bounds from config
        self.bounds = config.get('hyperparameter_bounds', self.default_bounds)
        
        # Integer parameters
        self.int_params = {"latent_dim", "batch_size"}
        
        # Parameter keys in consistent order
        self.param_keys = list(self.bounds.keys())
        
        logger.info(f"ParameterManager initialized with {len(self.param_keys)} parameters")
    
    def get_param_keys(self) -> List[str]:
        """Get ordered list of parameter names."""
        return self.param_keys.copy()
    
    def get_bounds(self) -> List[Tuple[float, float]]:
        """Get parameter bounds as list of (min, max) tuples."""
        return [self.bounds[key] for key in self.param_keys]
    
    def individual_to_params(self, individual: List[float]) -> Dict[str, Union[int, float]]:
        """Convert DEAP individual to parameter dictionary."""
        if len(individual) != len(self.param_keys):
            raise ValueError(f"Individual has {len(individual)} values, expected {len(self.param_keys)}")
        
        params = {}
        for i, key in enumerate(self.param_keys):
            value = individual[i]
            
            # Convert to int if needed
            if key in self.int_params:
                value = int(round(value))
            
            params[key] = value
        
        return params
    
    def params_to_individual(self, params: Dict[str, Union[int, float]]) -> List[float]:
        """Convert parameter dictionary to DEAP individual."""
        individual = []
        for key in self.param_keys:
            if key not in params:
                raise ValueError(f"Missing parameter: {key}")
            individual.append(float(params[key]))
        
        return individual
    
    def generate_random_params(self) -> Dict[str, Union[int, float]]:
        """Generate random parameters within bounds."""
        params = {}
        for key in self.param_keys:
            min_val, max_val = self.bounds[key]
            
            if key in self.int_params:
                value = random.randint(int(min_val), int(max_val))
            else:
                value = random.uniform(min_val, max_val)
            
            params[key] = value
        
        return params
    
    def validate_params(self, params: Dict[str, Union[int, float]]) -> bool:
        """Validate parameters are within bounds."""
        for key, value in params.items():
            if key in self.bounds:
                min_val, max_val = self.bounds[key]
                if not (min_val <= value <= max_val):
                    logger.warning(f"Parameter {key}={value} outside bounds [{min_val}, {max_val}]")
                    return False
        return True
    
    def clip_params(self, params: Dict[str, Union[int, float]]) -> Dict[str, Union[int, float]]:
        """Clip parameters to bounds."""
        clipped = {}
        for key, value in params.items():
            if key in self.bounds:
                min_val, max_val = self.bounds[key]
                clipped_value = max(min_val, min(max_val, value))
                
                if key in self.int_params:
                    clipped_value = int(round(clipped_value))
                
                clipped[key] = clipped_value
            else:
                clipped[key] = value
        
        return clipped
