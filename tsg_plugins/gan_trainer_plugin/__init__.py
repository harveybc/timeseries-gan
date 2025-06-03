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
- parameter_manager: Parameter extraction and validation
- directory_manager: Output directory setup and management
- plugin_interface: Plugin interactions and model extraction
"""

from .gan_trainer_plugin import GANTrainerPlugin

__all__ = ['GANTrainerPlugin']
