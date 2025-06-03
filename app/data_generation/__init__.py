"""
Data Generation Package

This package contains modules for handling both synthetic and real data processing
in the timeseries-gan pipeline.

Modules:
- synthetic_generator: Handles synthetic data generation using trained models
- real_data_processor: Handles real data loading and preprocessing
"""

from .synthetic_generator import SyntheticDataGenerator
from .real_data_processor import RealDataProcessor

__all__ = ['SyntheticDataGenerator', 'RealDataProcessor']
