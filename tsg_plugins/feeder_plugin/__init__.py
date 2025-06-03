"""
Feeder Plugin Package

Modular feeder plugin for timeseries-gan.
Handles data feeding, encoding, and condition processing.
"""

from .feeder_plugin import FeederPlugin
from .encoder_handler import EncoderHandler
from .data_preprocessor import DataPreprocessor
from .condition_manager import ConditionManager

__all__ = [
    'FeederPlugin',
    'EncoderHandler', 
    'DataPreprocessor',
    'ConditionManager'
]
