"""
Hyperparameter Handler Module

Manages hyperparameter bounds, validation, and type conversion
for the genetic algorithm optimization process.
"""

import logging
from typing import Dict, Any, List, Union, Tuple

logger = logging.getLogger(__name__)


class HyperparameterHandler:
    """
    Manages hyperparameter configuration and validation for optimization.
    
    Handles parameter bounds, type conversion, and validation logic
    for genetic algorithm optimization.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize hyperparameter handler."""
        self.config = config
        
        # Default hyperparameter bounds
        self.default_bounds = {
            "latent_dim": (4, 64),
            "batch_size": (16, 128)
        }
        
        # Parameters that should be integers
        self.integer_params = {"latent_dim", "batch_size"}
        
        # Get bounds from config or use defaults
        self.bounds = config.get('hyperparameter_bounds', self.default_bounds)
        
        # Validate bounds
        self._validate_bounds()
        
        logger.info(f"HyperparameterHandler initialized with bounds: {self.bounds}")
    
    def _validate_bounds(self):
        """Validate hyperparameter bounds configuration."""
        for param, bounds in self.bounds.items():
            if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
                raise ValueError(f"Invalid bounds for {param}: {bounds}. Must be (min, max)")
            
            min_val, max_val = bounds
            if min_val >= max_val:
                raise ValueError(f"Invalid bounds for {param}: min ({min_val}) >= max ({max_val})")
        
        logger.debug(f"Validated hyperparameter bounds: {self.bounds}")
    
    def get_parameter_keys(self) -> List[str]:
        """Get list of hyperparameter names to optimize."""
        return list(self.bounds.keys())
    
    def get_parameter_bounds(self) -> Dict[str, Tuple[Union[int, float], Union[int, float]]]:
        """Get parameter bounds dictionary."""
        return self.bounds.copy()
    
    def get_integer_params(self) -> set:
        """Get set of parameters that should be integers."""
        return self.integer_params.copy()
    
    def validate_individual(self, individual: List[Union[float, int]]) -> bool:
        """
        Validate that an individual has correct length and values.
        
        Args:
            individual: List of parameter values
            
        Returns:
            bool: True if valid
        """
        param_keys = self.get_parameter_keys()
        
        if len(individual) != len(param_keys):
            logger.error(f"Individual length {len(individual)} != parameter count {len(param_keys)}")
            return False
        
        for i, (param, value) in enumerate(zip(param_keys, individual)):
            min_val, max_val = self.bounds[param]
            
            if not (min_val <= value <= max_val):
                logger.warning(f"Parameter {param} value {value} outside bounds [{min_val}, {max_val}]")
                return False
        
        return True
    
    def individual_to_params(self, individual: List[Union[float, int]]) -> Dict[str, Union[int, float]]:
        """
        Convert individual genes to hyperparameter dictionary.
        
        Args:
            individual: List of parameter values from genetic algorithm
            
        Returns:
            Dict mapping parameter names to values
        """
        param_keys = self.get_parameter_keys()
        
        if len(individual) != len(param_keys):
            raise ValueError(f"Individual length {len(individual)} != parameter count {len(param_keys)}")
        
        params = {}
        for i, key in enumerate(param_keys):
            value = individual[i]
            
            # Convert to integer if required
            if key in self.integer_params:
                value = int(round(value))
            
            params[key] = value
        
        return params
    
    def params_to_individual(self, params: Dict[str, Union[int, float]]) -> List[Union[float, int]]:
        """
        Convert hyperparameter dictionary to individual genes.
        
        Args:
            params: Dictionary of parameter values
            
        Returns:
            List of values for genetic algorithm
        """
        param_keys = self.get_parameter_keys()
        individual = []
        
        for key in param_keys:
            if key not in params:
                raise ValueError(f"Missing parameter {key} in params dictionary")
            individual.append(params[key])
        
        return individual
    
    def get_random_params(self) -> Dict[str, Union[int, float]]:
        """
        Generate random hyperparameters within bounds.
        
        Returns:
            Dict of random parameter values
        """
        import random
        
        params = {}
        for param, (min_val, max_val) in self.bounds.items():
            if param in self.integer_params:
                value = random.randint(int(min_val), int(max_val))
            else:
                value = random.uniform(min_val, max_val)
            params[param] = value
        
        return params
    
    def clip_params(self, params: Dict[str, Union[int, float]]) -> Dict[str, Union[int, float]]:
        """
        Clip parameters to be within bounds.
        
        Args:
            params: Parameter dictionary to clip
            
        Returns:
            Clipped parameter dictionary
        """
        clipped = {}
        
        for param, value in params.items():
            if param in self.bounds:
                min_val, max_val = self.bounds[param]
                clipped_value = max(min_val, min(max_val, value))
                
                if param in self.integer_params:
                    clipped_value = int(round(clipped_value))
                
                clipped[param] = clipped_value
            else:
                clipped[param] = value
        
        return clipped
    
    def get_bounds_for_deap(self) -> Tuple[List[Union[int, float]], List[Union[int, float]]]:
        """
        Get bounds in format suitable for DEAP toolbox registration.
        
        Returns:
            Tuple of (min_bounds, max_bounds) lists
        """
        param_keys = self.get_parameter_keys()
        min_bounds = []
        max_bounds = []
        
        for key in param_keys:
            min_val, max_val = self.bounds[key]
            min_bounds.append(min_val)
            max_bounds.append(max_val)
        
        return min_bounds, max_bounds
    
    def get_parameter_info(self) -> Dict[str, Any]:
        """
        Get detailed information about parameters.
        
        Returns:
            Dict with parameter configuration details
        """
        return {
            'parameter_count': len(self.bounds),
            'parameter_names': self.get_parameter_keys(),
            'bounds': self.bounds,
            'integer_params': list(self.integer_params),
            'float_params': [p for p in self.get_parameter_keys() if p not in self.integer_params]
        }
