import sys
sys.path.insert(0, '/home/harveybc/Documents/GitHub/timeseries-gan')

try:
    from tsg_plugins.gan_trainer_plugin.gan_trainer_plugin import GANTrainerPlugin
    print("Import successful")
except Exception as e:
    print(f"Import failed: {e}")
