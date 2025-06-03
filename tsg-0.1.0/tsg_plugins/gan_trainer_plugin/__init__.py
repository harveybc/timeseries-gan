#!/usr/bin/env python3
"""
GAN Trainer Plugin Package

This package provides a modular GAN training plugin following extreme separation of concerns.
The main plugin delegates to specialized modules for focused functionality.

Modules:
- gan_trainer_plugin: Main plugin interface with mandatory methods
- training_coordinator: Core GAN training orchestration
- model_builder: Discriminator and GAN model construction
- data_generator: Training data generation and batching
- model_persistence: Model saving and loading operations
- training_metrics: Training progress tracking and visualization
- model_builder: Discriminator and GAN model construction
- loss_calculator: GAN loss computation and metrics
- data_generator: Training data generation and batching
- model_persistence: Model saving and loading operations
- training_callbacks: Training callbacks and monitoring
- technical_indicators: TensorFlow technical indicator layer
"""

from .gan_trainer_plugin import GANTrainerPlugin

__all__ = ['GANTrainerPlugin']
