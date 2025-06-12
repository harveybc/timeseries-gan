#!/usr/bin/env python3

print("=== Debugging Trainer Plugin Loading ===")

# Test 1: Direct import
print("\n1. Testing direct import...")
try:
    from tsg_plugins.gan_trainer_plugin.gan_trainer_plugin import GANTrainerPlugin
    print("✓ GANTrainerPlugin imported successfully")
    print(f"Class: {GANTrainerPlugin}")
    print(f"Plugin params keys: {list(GANTrainerPlugin.plugin_params.keys())[:5]}...")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Entry points
print("\n2. Testing entry points...")
try:
    from importlib.metadata import entry_points
    trainer_eps = entry_points().select(group='trainer.plugins')
    for ep in trainer_eps:
        print(f"Entry point: {ep.name} = {ep.value}")
        try:
            loaded_class = ep.load()
            print(f"✓ Loaded: {loaded_class}")
        except Exception as e:
            print(f"✗ Error loading {ep.name}: {e}")
except Exception as e:
    print(f"✗ Error with entry points: {e}")

# Test 3: Plugin loader
print("\n3. Testing plugin loader...")
try:
    from app.plugin_loader import load_plugin
    plugin_class, params = load_plugin('trainer.plugins', 'gan_trainer')
    print(f"✓ Plugin loader successful: {plugin_class}")
    print(f"Params: {params[:5]}...")
except Exception as e:
    print(f"✗ Plugin loader error: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Main function loading
print("\n4. Testing main function plugin loading...")
try:
    from app.config import DEFAULT_VALUES
    from app.main import load_and_initialize_plugins
    
    config = DEFAULT_VALUES.copy()
    print(f"Config trainer value: {config.get('trainer', 'NOT_FOUND')}")
    
    plugins = load_and_initialize_plugins(config)
    print(f"Loaded plugins: {list(plugins.keys())}")
    print(f"Trainer plugin: {plugins.get('trainer_plugin', 'NOT_FOUND')}")
    
except Exception as e:
    print(f"✗ Main loading error: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Debug Complete ===")
