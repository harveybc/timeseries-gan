#!/usr/bin/env python3
"""
Generator Plugin Package

This package provides a modular generator plugin following extreme separation of concerns.
The main plugin delegates to specialized modules for focused functionality.

Modules:
- generator_plugin: Main plugin interface with mandatory methods
- model_loader: Model loading and validation functionality
- data_generator: Core generation logic and data processing
- feature_processor: Feature processing and technical indicators
- normalization_handler: Data normalization and denormalization
- sequence_builder: Sequence building and window management
"""

from .generator_plugin import GeneratorPlugin

__all__ = ['GeneratorPlugin']
