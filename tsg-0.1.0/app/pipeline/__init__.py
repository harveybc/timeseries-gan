"""
Pipeline module for TimeSeries-GAN.

This package contains operation mode pipelines that handle specific workflows:
- TrainPipeline: GAN model training
- OptimizePipeline: Hyperparameter optimization using genetic algorithms
- GeneratePipeline: Synthetic data generation and evaluation

Each pipeline follows single responsibility principle and encapsulates
all logic specific to its operation mode.
"""

from .train_pipeline import TrainPipeline
from .optimize_pipeline import OptimizePipeline
from .generate_pipeline import GeneratePipeline

__all__ = ['TrainPipeline', 'OptimizePipeline', 'GeneratePipeline']
