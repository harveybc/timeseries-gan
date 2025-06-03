"""
Optimizer Plugin - Main Interface

Clean, modular DEAP-based genetic algorithm optimizer for hyperparameter tuning.
"""

import copy
import logging
from typing import Dict, Any, Union

from .parameter_manager import ParameterManager
from .evaluation_runner import EvaluationRunner
from .genetic_optimizer import GeneticOptimizer

logger = logging.getLogger(__name__)


class OptimizerPlugin:
    """
    Main optimizer plugin interface using genetic algorithms.
    
    Optimizes hyperparameters for synthetic data generation pipelines
    using DEAP genetic algorithms with modular components.
    """
    
    # Default plugin parameters
    plugin_params = {
        "population_size": 20,
        "n_generations": 10,
        "cxpb": 0.7,
        "mutpb": 0.2,
        "tournament_size": 3,
        "hyperparameter_bounds": {
            "latent_dim": (16, 128),
            "batch_size": (16, 64),
            "learning_rate": (1e-5, 1e-2)
        },
        "optimizer_n_samples_per_eval": 100,
        "optimizer_start_datetime": None,
        "random_seed": None
    }
    
    # Debug variables
    plugin_debug_vars = ["population_size", "n_generations", "cxpb", "mutpb"]
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize optimizer plugin."""
        if config is None:
            raise ValueError("Configuration dictionary is required")
        
        # Merge config with defaults
        self.params = copy.deepcopy(self.plugin_params)
        self.set_params(**config)
        
        # Initialize modular components
        self.parameter_manager = ParameterManager(self.params)
        self.evaluation_runner = EvaluationRunner(self.params)
        self.genetic_optimizer = None  # Created when needed
        
        # State tracking
        self.is_initialized = False
        self.optimization_results = None
        
        logger.info("OptimizerPlugin initialized")
    
    def set_params(self, **kwargs: Any) -> None:
        """Update plugin parameters."""
        for key, value in kwargs.items():
            self.params[key] = value
    
    def get_debug_info(self) -> Dict[str, Any]:
        """Get debugging information."""
        return {var: self.params.get(var) for var in self.plugin_debug_vars}
    
    def optimize(
        self,
        feeder_plugin: Any,
        generator_plugin: Any,
        evaluator_plugin: Any,
        config: Dict[str, Any]
    ) -> Dict[str, Union[int, float]]:
        """
        Run genetic algorithm optimization.
        
        Args:
            feeder_plugin: Plugin for data feeding
            generator_plugin: Plugin for data generation
            evaluator_plugin: Plugin for evaluation
            config: Global configuration
            
        Returns:
            Dict: Best hyperparameters found
        """
        logger.info("Starting hyperparameter optimization...")
        
        try:
            # Merge global config
            merged_config = copy.deepcopy(self.params)
            merged_config.update(config)
            
            # Set up evaluation runner with plugins
            self.evaluation_runner.set_plugins(feeder_plugin, generator_plugin, evaluator_plugin)
            
            # Create genetic optimizer
            self.genetic_optimizer = GeneticOptimizer(
                merged_config,
                self.parameter_manager,
                self.evaluation_runner
            )
            
            # Run optimization
            best_params = self.genetic_optimizer.optimize()
            
            # Store results
            self.optimization_results = {
                'best_parameters': best_params,
                'evaluation_stats': self.evaluation_runner.get_evaluation_stats(),
                'optimization_info': self.genetic_optimizer.get_optimization_info()
            }
            
            self.is_initialized = True
            
            logger.info(f"Optimization completed successfully")
            logger.info(f"Best parameters: {best_params}")
            
            return best_params
            
        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}")
            # Return default/random parameters as fallback
            return self.parameter_manager.generate_random_params()
    
    def get_optimization_results(self) -> Dict[str, Any]:
        """Get detailed optimization results."""
        if self.optimization_results is None:
            return {'status': 'not_run'}
        
        return self.optimization_results.copy()
    
    def get_plugin_info(self) -> Dict[str, Any]:
        """Get plugin information."""
        info = {
            'plugin_type': 'optimizer',
            'algorithm': 'genetic_algorithm',
            'initialized': self.is_initialized,
            'parameter_count': len(self.parameter_manager.get_param_keys()),
            'parameters': self.parameter_manager.get_param_keys()
        }
        
        if self.genetic_optimizer:
            info.update(self.genetic_optimizer.get_optimization_info())
        
        return info
    
    def validate_config(self) -> bool:
        """Validate configuration."""
        try:
            # Check required parameters
            required_params = ['population_size', 'n_generations', 'hyperparameter_bounds']
            for param in required_params:
                if param not in self.params:
                    logger.error(f"Missing required parameter: {param}")
                    return False
            
            # Validate parameter bounds
            bounds = self.params['hyperparameter_bounds']
            if not isinstance(bounds, dict) or len(bounds) == 0:
                logger.error("Invalid hyperparameter_bounds")
                return False
            
            # Validate GA parameters
            if self.params['population_size'] < 2:
                logger.error("population_size must be >= 2")
                return False
            
            if self.params['n_generations'] < 1:
                logger.error("n_generations must be >= 1")
                return False
            
            logger.info("Configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {str(e)}")
            return False
    
    def reset(self):
        """Reset plugin state."""
        self.is_initialized = False
        self.optimization_results = None
        self.genetic_optimizer = None
        
        # Reset evaluation runner
        if hasattr(self.evaluation_runner, 'evaluation_count'):
            self.evaluation_runner.evaluation_count = 0
        
        logger.info("OptimizerPlugin reset")
    
    def cleanup(self):
        """Cleanup resources."""
        self.reset()
        logger.info("OptimizerPlugin cleaned up")
