#!/usr/bin/env python3
"""
Test early stopping patience parameter mapping.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, '/home/harveybc/Documents/GitHub/timeseries-gan')

def test_early_stopping_patience_mapping():
    """Test that early_stopping_patience from config.py maps to es_patience in plugin."""
    
    print("=== Testing Early Stopping Patience Parameter Mapping ===\n")
    
    try:
        from app.config import DEFAULT_VALUES
        from tsg_plugins.gan_trainer_plugin.gan_trainer_plugin import GANTrainerPlugin
        from tsg_plugins.generator_plugin.generator_plugin import GeneratorPlugin
        from tsg_plugins.discriminator_plugin import DiscriminatorPlugin
        import logging
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        
        # Check config.py value
        config_patience = DEFAULT_VALUES.get("early_stopping_patience")
        print(f"✓ Config.py early_stopping_patience: {config_patience}")
        
        # Create test config with explicit early_stopping_patience
        test_config = DEFAULT_VALUES.copy()
        test_config["early_stopping_patience"] = 120  # Explicit value
        
        print(f"✓ Test config early_stopping_patience: {test_config['early_stopping_patience']}")
        
        # Create plugin dependencies (minimal setup)
        generator_plugin = GeneratorPlugin(test_config, logger)
        discriminator_plugin = DiscriminatorPlugin(test_config)
        
        # Create GANTrainerPlugin with the test config
        trainer_plugin = GANTrainerPlugin(
            config=test_config,
            generator_plugin=generator_plugin,
            discriminator_plugin=discriminator_plugin
        )
        
        # Check the mapped parameter
        plugin_es_patience = trainer_plugin.params.get("es_patience")
        print(f"✓ Plugin es_patience parameter: {plugin_es_patience}")
        
        # Verify mapping worked correctly
        if plugin_es_patience == test_config["early_stopping_patience"]:
            print("✅ SUCCESS: early_stopping_patience correctly mapped to es_patience!")
            
            # Check early stopping callback
            if trainer_plugin.early_stopping_callback:
                callback_patience = trainer_plugin.early_stopping_callback.patience
                print(f"✓ EarlyStopping callback patience: {callback_patience}")
                
                if callback_patience == test_config["early_stopping_patience"]:
                    print("✅ SUCCESS: EarlyStopping callback uses correct patience value!")
                    return True
                else:
                    print(f"❌ ERROR: EarlyStopping callback patience ({callback_patience}) != config value ({test_config['early_stopping_patience']})")
                    return False
            else:
                print("⚠️  WARNING: EarlyStopping callback not created")
                return False
        else:
            print(f"❌ ERROR: Parameter mapping failed!")
            print(f"  Expected: {test_config['early_stopping_patience']}")
            print(f"  Got: {plugin_es_patience}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_early_stopping_patience_mapping()
    
    if success:
        print("\n🎉 EARLY STOPPING PATIENCE MAPPING TEST PASSED!")
        print("✅ The early_stopping_patience parameter is correctly mapped and used.")
        print("   Training logs should now show the correct patience value (120).")
    else:
        print("\n❌ EARLY STOPPING PATIENCE MAPPING TEST FAILED!")
        sys.exit(1)
