#!/usr/bin/env python3
"""
GAN Trainer Plugin Package

Modular GAN training plugin with extreme separation of concerns.
This package breaks down the monolithic gan_trainer_plugin.py (1320+ lines)
into focused modules following single responsibility principle.

Modules:
- plugin: Main plugin interface (preserves original plugin structure)
- training_coordinator: Main training orchestration logic
- model_builder: Discriminator and GAN model construction
- loss_calculator: GAN loss computation and metrics
- data_generator: Training data generation and batching
- model_persistence: Model saving and loading operations
- training_callbacks: Training callbacks and monitoring
- technical_indicators: TensorFlow technical indicator layer
"""

from .plugin import GANTrainerPlugin

__all__ = ['GANTrainerPlugin']
