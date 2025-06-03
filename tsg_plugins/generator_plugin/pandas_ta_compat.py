#!/usr/bin/env python3
"""
Pandas-TA Compatibility Module

Handles numpy compatibility issues with pandas-ta library.
This module ensures pandas-ta can import correctly with newer numpy versions.
"""

import numpy as np

# Fix numpy compatibility for pandas-ta
def fix_numpy_compatibility():
    """Fix numpy compatibility issues for pandas-ta."""
    # Add NaN attribute if it doesn't exist (removed in newer numpy versions)
    if not hasattr(np, 'NaN'):
        np.NaN = np.nan
    
    # Add any other compatibility fixes here as needed
    
# Apply the fix immediately when this module is imported
fix_numpy_compatibility()

# Safe import of pandas-ta
try:
    import pandas_ta as ta
    pandas_ta_available = True
except ImportError as e:
    print(f"Warning: pandas-ta import failed: {e}")
    pandas_ta_available = False
    
    # Create a mock ta module for when pandas_ta is not available
    class MockTA:
        def __getattr__(self, name):
            def mock_function(*args, **kwargs):
                print(f"Warning: pandas_ta function '{name}' called but pandas_ta not available")
                return None
            return mock_function
    
    ta = MockTA()

__all__ = ['ta', 'pandas_ta_available', 'fix_numpy_compatibility']
