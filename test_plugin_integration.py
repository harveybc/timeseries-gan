#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, '/home/harveybc/Documents/GitHub/timeseries-gan')

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

print("=== Testing GAN Plugin Integration ===")

def test_generator_plugin():
    print("\n1. Testing GeneratorPlugin...")
    try:
        from tsg_plugins.generator_plugin.generator_plugin import GeneratorPlugin
        config = {'x_train_file': '/home/harveybc/Documents/GitHub/timeseries-gan/tests/data/sample_data.csv'}
        gen = GeneratorPlugin(config)
        model = gen.get_model()
        if model:
            print(f"✓ GeneratorPlugin working: {model.count_params()} parameters")
            return gen
        else:
            print("❌ GeneratorPlugin get_model() returned None")
            return None
    except Exception as e:
        print(f"❌ GeneratorPlugin failed: {e}")
        return None

def test_discriminator_plugin():
    print("\n2. Testing DiscriminatorPlugin...")
    try:
        from tsg_plugins.discriminator_plugin import DiscriminatorPlugin
        config = {'sequence_length': 144, 'num_features': 57}
        disc = DiscriminatorPlugin(config)
        model = disc.get_model()
        if model:
            print(f"✓ DiscriminatorPlugin working: {model.count_params()} parameters")
            return disc
        else:
            print("❌ DiscriminatorPlugin get_model() returned None")
            return None
    except Exception as e:
        print(f"❌ DiscriminatorPlugin failed: {e}")
        return None

def test_gan_trainer_plugin():
    print("\n3. Testing GANTrainerPlugin with plugins...")
    try:
        # Create minimal plugins first
        gen_config = {'x_train_file': '/home/harveybc/Documents/GitHub/timeseries-gan/tests/data/sample_data.csv'}
        from tsg_plugins.generator_plugin.generator_plugin import GeneratorPlugin
        gen_plugin = GeneratorPlugin(gen_config)
        
        disc_config = {'sequence_length': 144, 'num_features': 57}
        from tsg_plugins.discriminator_plugin import DiscriminatorPlugin
        disc_plugin = DiscriminatorPlugin(disc_config)
        
        # Test GANTrainer
        from tsg_plugins.gan_trainer_plugin.gan_trainer_plugin import GANTrainerPlugin
        trainer_config = {'gan_epochs': 1, 'gan_batch_size': 4}
        trainer = GANTrainerPlugin(
            config=trainer_config,
            generator_plugin_instance=gen_plugin,
            feeder_plugin_instance=None,
            discriminator_plugin_instance=disc_plugin
        )
        
        # Test model access through interface
        gen_model = trainer.plugin_interface.get_generator_model()
        disc_model = trainer.plugin_interface.get_discriminator_model()
        
        if gen_model and disc_model:
            print(f"✓ GANTrainerPlugin working: Generator={type(gen_model)}, Discriminator={type(disc_model)}")
            return trainer
        else:
            print(f"❌ GANTrainerPlugin models not accessible: gen={type(gen_model)}, disc={type(disc_model)}")
            return None
            
    except Exception as e:
        print(f"❌ GANTrainerPlugin failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    try:
        gen = test_generator_plugin()
        disc = test_discriminator_plugin() 
        trainer = test_gan_trainer_plugin()
        
        if gen and disc and trainer:
            print("\n🎉 All plugin integration tests passed!")
        else:
            print("\n❌ Some plugin tests failed")
            
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
