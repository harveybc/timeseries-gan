"""
Utils Package

This package contains utility modules for the timeseries-gan pipeline.

Modules:
- latent_shape_inference: Latent shape inference between generator and feeder plugins
- output_manager: Output file management and data combination utilities
"""

from .latent_shape_inference import LatentShapeInference
from .output_manager import OutputManager

__all__ = ['LatentShapeInference', 'OutputManager']
