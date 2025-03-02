# config.py

DEFAULT_VALUES = {
    'train_file': 'tests\\data\\base_d2.csv',
    'save_generator': 'generator.keras',
    'save_discriminator': 'discriminator.keras',
    'optimizer_plugin': 'default',
    'generator_plugin': 'default',
    'discriminator_plugin': 'default',
    'remote_log': None,
    'remote_load_config': None,
    'remote_save_config': None,
    'username': None,
    'password': None,
    'load_config': None,
    'save_config': 'config_out.json',
    'save_log': 'debug_out.json',
    'quiet_mode': False,
    'max_steps': 6300,
    'batch_size': 32,
    'epochs': 150
}

