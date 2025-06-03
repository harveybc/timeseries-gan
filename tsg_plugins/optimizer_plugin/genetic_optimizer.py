"""
Genetic Optimizer Module

DEAP-based genetic algorithm implementation for hyperparameter optimization.
"""

import logging
import random
from typing import Dict, Any, List, Union, Tuple, Callable
from deap import algorithms, base, creator, tools
import numpy as np

logger = logging.getLogger(__name__)


class GeneticOptimizer:
    """DEAP-based genetic algorithm optimizer."""
    
    def __init__(self, config: Dict[str, Any], parameter_manager, evaluation_runner):
        """Initialize genetic optimizer."""
        self.config = config
        self.parameter_manager = parameter_manager
        self.evaluation_runner = evaluation_runner
        
        # GA parameters
        self.population_size = config.get('population_size', 20)
        self.n_generations = config.get('n_generations', 10)
        self.crossover_prob = config.get('cxpb', 0.7)
        self.mutation_prob = config.get('mutpb', 0.2)
        self.tournament_size = config.get('tournament_size', 3)
        
        # Set random seed if provided
        seed = config.get('random_seed')
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Initialize DEAP components
        self._setup_deap()
        
        logger.info(f"GeneticOptimizer initialized: pop={self.population_size}, gen={self.n_generations}")
    
    def _setup_deap(self):
        """Setup DEAP toolbox and operators."""
        # Create fitness and individual classes
        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMin)
        
        # Initialize toolbox
        self.toolbox = base.Toolbox()
        
        # Get parameter bounds
        param_keys = self.parameter_manager.get_param_keys()
        bounds = self.parameter_manager.get_bounds()
        
        # Register attribute generators
        for i, (key, (min_val, max_val)) in enumerate(zip(param_keys, bounds)):
            self.toolbox.register(f"attr_{i}", random.uniform, min_val, max_val)
        
        # Register individual and population
        attrs = [getattr(self.toolbox, f"attr_{i}") for i in range(len(param_keys))]
        self.toolbox.register("individual", tools.initCycle, creator.Individual, attrs, n=1)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        # Register genetic operators
        self.toolbox.register("evaluate", self._evaluate_individual)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", self._mutate_individual)
        self.toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)
        
        logger.info("DEAP toolbox setup complete")
    
    def _evaluate_individual(self, individual: List[float]) -> Tuple[float]:
        """Evaluate a single individual."""
        try:
            # Convert individual to parameters
            params = self.parameter_manager.individual_to_params(individual)
            
            # Run evaluation
            fitness = self.evaluation_runner.evaluate_individual(params)
            
            return (fitness,)
            
        except Exception as e:
            logger.error(f"Error evaluating individual: {str(e)}")
            return (float('inf'),)
    
    def _mutate_individual(self, individual: List[float]) -> Tuple[List[float]]:
        """Mutate an individual with bounds checking."""
        bounds = self.parameter_manager.get_bounds()
        
        # Gaussian mutation with bounds
        for i in range(len(individual)):
            if random.random() < 0.1:  # 10% chance per gene
                min_val, max_val = bounds[i]
                
                # Gaussian mutation
                sigma = (max_val - min_val) * 0.1  # 10% of range
                individual[i] += random.gauss(0, sigma)
                
                # Clip to bounds
                individual[i] = max(min_val, min(max_val, individual[i]))
        
        return (individual,)
    
    def optimize(self) -> Dict[str, Union[int, float]]:
        """
        Run genetic algorithm optimization.
        
        Returns:
            Dict: Best hyperparameters found
        """
        logger.info(f"Starting genetic algorithm optimization...")
        
        try:
            # Create initial population
            population = self.toolbox.population(n=self.population_size)
            
            # Statistics tracking
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean)
            stats.register("std", np.std)
            stats.register("min", np.min)
            stats.register("max", np.max)
            
            # Hall of fame to track best individuals
            hall_of_fame = tools.HallOfFame(1)
            
            # Run genetic algorithm
            population, logbook = algorithms.eaSimple(
                population, 
                self.toolbox,
                cxpb=self.crossover_prob,
                mutpb=self.mutation_prob,
                ngen=self.n_generations,
                stats=stats,
                halloffame=hall_of_fame,
                verbose=True
            )
            
            # Get best individual
            best_individual = hall_of_fame[0]
            best_params = self.parameter_manager.individual_to_params(best_individual)
            best_fitness = best_individual.fitness.values[0]
            
            logger.info(f"Optimization complete! Best fitness: {best_fitness}")
            logger.info(f"Best parameters: {best_params}")
            
            # Log statistics
            if logbook:
                logger.info("Optimization statistics:")
                for gen, record in enumerate(logbook):
                    logger.info(f"Gen {gen}: min={record['min']:.4f}, avg={record['avg']:.4f}, std={record['std']:.4f}")
            
            return best_params
            
        except Exception as e:
            logger.error(f"Error in genetic algorithm optimization: {str(e)}")
            # Return random parameters as fallback
            return self.parameter_manager.generate_random_params()
    
    def get_optimization_info(self) -> Dict[str, Any]:
        """Get optimization configuration info."""
        return {
            'population_size': self.population_size,
            'n_generations': self.n_generations,
            'crossover_prob': self.crossover_prob,
            'mutation_prob': self.mutation_prob,
            'tournament_size': self.tournament_size,
            'parameter_count': len(self.parameter_manager.get_param_keys()),
            'parameters': self.parameter_manager.get_param_keys()
        }
