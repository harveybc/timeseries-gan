#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, '/home/harveybc/Documents/GitHub/timeseries-gan')

import logging
logging.basicConfig(level=logging.DEBUG)

print("Starting generator test...")

try:
    print("Step 1: Importing GeneratorPlugin...")
    from tsg_plugins.generator_plugin.generator_plugin import GeneratorPlugin
    print("✓ Import successful")

    print("Step 2: Creating config...")
    config = {
        'x_train_file': '/home/harveybc/Documents/GitHub/timeseries-gan/tests/data/sample_data.csv',
        'sequential_model_file': None  # Explicitly set to None
    }
    print("✓ Config created")

    print("Step 3: Initializing GeneratorPlugin...")
    gen = GeneratorPlugin(config)
    print("✓ Initialization successful")

    print("Step 4: Getting model...")
    model = gen.get_model()
    print(f"✓ get_model() returned: {type(model)}")
    
    if model:
        print(f"✓ Model parameters: {model.count_params()}")
        print(f"✓ Input shapes: {[inp.shape for inp in model.inputs]}")
        print(f"✓ Output shape: {model.output.shape}")
    else:
        print("❌ Model is None")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("Test completed.")
