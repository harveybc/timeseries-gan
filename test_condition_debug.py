#!/usr/bin/env python3
"""
Debug script to test ConditionManager functionality
"""

import sys
import os
sys.path.append('.')

# Import config
from app.config import DEFAULT_VALUES
import pandas as pd
import numpy as np

print("=== Testing ConditionManager Directly ===")
print(f"Config conditional_features_dim: {DEFAULT_VALUES.get('conditional_features_dim')}")
print(f"Config feeder_date_features_for_conditioning: {DEFAULT_VALUES.get('feeder_date_features_for_conditioning')}")
print(f"Expected features: {len(DEFAULT_VALUES.get('feeder_date_features_for_conditioning', []))} * 2 = {len(DEFAULT_VALUES.get('feeder_date_features_for_conditioning', [])) * 2}")

# Test ConditionManager directly
try:
    from tsg_plugins.feeder_plugin.condition_manager import ConditionManager
    print("✓ ConditionManager imported successfully")
except Exception as e:
    print(f"✗ Failed to import ConditionManager: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Create config for ConditionManager
condition_config = DEFAULT_VALUES.copy()
condition_config['condition_columns'] = DEFAULT_VALUES.get('feeder_fundamental_features_for_conditioning', [])
condition_config['use_temporal_conditions'] = True

print(f"\nCondition config:")
print(f"  condition_columns: {condition_config.get('condition_columns')}")
print(f"  use_temporal_conditions: {condition_config.get('use_temporal_conditions')}")
print(f"  feeder_date_features_for_conditioning: {condition_config.get('feeder_date_features_for_conditioning')}")

# Initialize ConditionManager
cm = ConditionManager(condition_config)
print(f"\nConditionManager created. Initial condition_dim: {cm.condition_dim}")

# Initialize
success = cm.initialize()
print(f"ConditionManager initialized: {success}")
print(f"Final condition_dim: {cm.condition_dim}")
print(f"Final condition_shape: {cm.condition_shape}")

# Test extraction
test_data = pd.DataFrame({
    'DATE_TIME': [pd.Timestamp('2023-01-01 10:00:00')]
})

print(f"\nTesting condition extraction...")
result = cm.extract_conditions(test_data, timestamp_col='DATE_TIME')
print(f"Result shape: {result.shape if result is not None else None}")
print(f"Result:\n{result}")

if result is not None and result.shape[1] == 10:
    print("\n✓ SUCCESS: ConditionManager generated expected 10 features!")
else:
    print(f"\n✗ FAILURE: Expected 10 features, got {result.shape[1] if result is not None else 'None'}")
