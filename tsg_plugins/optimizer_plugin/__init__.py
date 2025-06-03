"""
Modular Optimizer Plugin Package

Provides genetic algorithm optimization for hyperparameter tuning
of synthetic data generation pipelines.
"""

from .optimizer_plugin import OptimizerPlugin
from .genetic_algorithm_manager import GeneticAlgorithmManager
from .hyperparameter_handler import HyperparameterHandler
from .evaluation_engine import EvaluationEngine
from .plugin_coordinator import PluginCoordinator

__all__ = [
    'OptimizerPlugin',
    'GeneticAlgorithmManager', 
    'HyperparameterHandler',
    'EvaluationEngine',
    'PluginCoordinator'
]
