#!/usr/bin/env python3

print("=== Testing GANTrainerPlugin Fix ===")

# Test 1: Import
try:
    from tsg_plugins.gan_trainer_plugin.gan_trainer_plugin import GANTrainerPlugin
    print("✓ GANTrainerPlugin imported successfully")
except Exception as e:
    print(f"✗ Import failed: {e}")
    exit(1)

# Test 2: Instantiation 
try:
    config = {'gan_epochs': 1, 'gan_batch_size': 4}
    plugin = GANTrainerPlugin(config, None, None, None)
    print("✓ GANTrainerPlugin instantiated")
except Exception as e:
    print(f"✗ Instantiation failed: {e}")
    exit(1)

# Test 3: add_debug_info method signature
try:
    debug_dict = {}
    plugin.add_debug_info(debug_dict)
    print("✓ add_debug_info method works correctly")
    print(f"✓ Debug info contains {len(debug_dict)} keys")
except Exception as e:
    print(f"✗ add_debug_info failed: {e}")
    exit(1)

# Test 4: Plugin loading via main function
try:
    from app.config import DEFAULT_VALUES
    from app.main import load_and_initialize_plugins
    
    config = DEFAULT_VALUES.copy()
    config['operation_mode'] = 'train'
    config['trainer'] = 'gan_trainer'
    
    plugins = load_and_initialize_plugins(config)
    if plugins.get('trainer_plugin'):
        print("✓ GANTrainerPlugin loaded via main function")
    else:
        print("✗ GANTrainerPlugin not loaded via main function")
        print(f"Available plugins: {list(plugins.keys())}")
except Exception as e:
    print(f"✗ Main function loading failed: {e}")
    import traceback
    traceback.print_exc()

print("=== Test Complete ===")
