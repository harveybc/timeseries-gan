"""
Optimizer Plugin - Main Interface

Clean, modular DEAP-based genetic algorithm optimizer for hyperparameter tuning.
"""

import copy
import logging
from typing import Dict, Any, Union, Optional

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
    plugin_params: Dict[str, Any] = {
        "population_size": 20,
        "n_generations": 10,
        "cxpb": 0.7,
        "mutpb": 0.2,
        "tournament_size": 3,
        "hyperparameter_bounds": {
            'latent_dim': (16, 128), 
            'batch_size': (16, 64), 
            'learning_rate': (1e-5, 1e-2) # Example, adjust as per actual hyperparams
        },
        "optimizer_n_samples_per_eval": 100,
        "optimizer_start_datetime": None,
        "random_seed": None
    }
    
    # Debug variables
    plugin_debug_vars = ["population_size", "n_generations", "cxpb", "mutpb"]
    
    def __init__(self, 
                 config: Dict[str, Any],
                 generator_plugin_instance: Optional[Any] = None,
                 discriminator_plugin_instance: Optional[Any] = None,
                 trainer_plugin_instance: Optional[Any] = None,
                 feeder_plugin_instance: Optional[Any] = None,
                 evaluator_plugin_instance: Optional[Any] = None, # Added for completeness
                 **kwargs): # Added **kwargs to catch any other unexpected args
        """Initialize optimizer plugin."""
        self.logger = logging.getLogger(__name__)
        self.main_config = config.copy() if config else {}
        self.params = self.plugin_params.copy()

        # Store plugin instances
        self.generator_plugin = generator_plugin_instance
        self.discriminator_plugin = discriminator_plugin_instance
        self.trainer_plugin = trainer_plugin_instance
        self.feeder_plugin = feeder_plugin_instance
        self.evaluator_plugin = evaluator_plugin_instance

        # Initialize params from the global config
        self.set_params(**self.main_config) 
        
        self.logger.info(f"OptimizerPlugin initialized. Received plugins: "
                         f"Gen: {self.generator_plugin is not None}, "
                         f"Disc: {self.discriminator_plugin is not None}, "
                         f"Trainer: {self.trainer_plugin is not None}, "
                         f"Feeder: {self.feeder_plugin is not None}, "
                         f"Eval: {self.evaluator_plugin is not None}")
        if kwargs:
            self.logger.warning(f"OptimizerPlugin received unexpected keyword arguments: {kwargs.keys()}")


    def set_params(self, **kwargs) -> None:
        """Update plugin parameters from global configuration or direct calls."""
        self.logger.debug(f"OptimizerPlugin.set_params called with kwargs: {list(kwargs.keys())}")
        
        # 1. Update main_config with new settings from kwargs
        if hasattr(self, 'main_config') and self.main_config is not None:
            self.main_config.update(kwargs)
        else:
            self.logger.warning("OptimizerPlugin: self.main_config was None, re-initializing from kwargs.")
            self.main_config = kwargs.copy()

        # 2. Re-initialize self.params from plugin_params defaults
        current_params = self.plugin_params.copy()

        # 3. Populate self.params from the updated self.main_config
        # Iterate over the default plugin_params keys to ensure all are considered.
        for param_key_default in self.plugin_params.keys():
            # Check for prefixed version first (e.g., "optimizer_population_size")
            prefixed_key_in_main_config = f"optimizer_{param_key_default}"
            if prefixed_key_in_main_config in self.main_config:
                current_params[param_key_default] = self.main_config[prefixed_key_in_main_config]
            # Then check for non-prefixed version (e.g., "population_size")
            elif param_key_default in self.main_config:
                current_params[param_key_default] = self.main_config[param_key_default]
            # If neither is found, the default from plugin_params.copy() remains.
        
        self.params = current_params
        self.logger.debug(f"OptimizerPlugin.params re-derived: {self.params}")

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
        self.logger.info("Starting hyperparameter optimization...")
        
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
            
            self.logger.info(f"Optimization completed successfully")
            self.logger.info(f"Best parameters: {best_params}")
            
            return best_params
            
        except Exception as e:
            self.logger.error(f"Optimization failed: {str(e)}")
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
                    self.logger.error(f"Missing required parameter: {param}")
                    return False
            
            # Validate parameter bounds
            bounds = self.params['hyperparameter_bounds']
            if not isinstance(bounds, dict) or len(bounds) == 0:
                self.logger.error("Invalid hyperparameter_bounds")
                return False
            
            # Validate GA parameters
            if self.params['population_size'] < 2:
                self.logger.error("population_size must be >= 2")
                return False
            
            if self.params['n_generations'] < 1:
                self.logger.error("n_generations must be >= 1")
                return False
            
            self.logger.info("Configuration validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {str(e)}")
            return False
    
    def reset(self):
        """Reset plugin state."""
        self.is_initialized = False
        self.optimization_results = None
        self.genetic_optimizer = None
        
        # Reset evaluation runner
        if hasattr(self.evaluation_runner, 'evaluation_count'):
            self.evaluation_runner.evaluation_count = 0
        
        self.logger.info("OptimizerPlugin reset")
    
    def cleanup(self):
        """Cleanup resources."""
        self.reset()
        self.logger.info("OptimizerPlugin cleaned up")
