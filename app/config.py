"""
config.py for SDG (Synthetic Data Generator)

This module defines the default values for every command-line parameter
supported by the sdg application. These defaults are used when no value
is provided via CLI, config file, or remote config.
"""

# --- Basic data and model parameters ---
SEQ_LEN = 18  # Sequence length
LATENT_DIM = 32  # Latent dimension for VAE/GAN
N_FEATURES = 5  # Number of features in the time series # TO BE ADJUSTED BY USER

# --- File paths ---
# Input data paths (assuming they remain in the 'data' folder relative to project root)
X_TRAIN_FILE = "data/x_train.csv"
X_TEST_FILE = "data/x_test.csv"

# Output paths updated to examples/results/phase_4_3/
BASE_OUTPUT_DIR = "examples/results/phase_4_3"
OUTPUT_FILE = f"{BASE_OUTPUT_DIR}/generated_data.csv"

# --- Plugin names (defaults) ---
FEEDER_PLUGIN = "csv_feeder"  # Default feeder plugin
GENERATOR_PLUGIN = "vae_generator"  # Default generator plugin (VAEGeneratorPlugin handles VAE/GAN)
TRAINER_PLUGIN = "default_trainer" # General trainer for pipeline, GAN has its own

# --- VAE specific parameters (if VAE is used or trained separately) ---
VAE_MODEL_PATH = f"{BASE_OUTPUT_DIR}/models/vae/generator_model.h5" # Path to the VAE generator (decoder)
VAE_EPOCHS = 100
VAE_BATCH_SIZE = 32
VAE_SAVE_INTERVAL = 10
VAE_TRAINING_MODE = False # Set to true to run VAE training (if a VAETrainerPlugin is implemented)

# --- Data generation parameters ---
PREPEND_LENGTH = 50  # Number of real data points to prepend to synthetic data
GENERATION_LENGTH = 200  # Number of synthetic data points to generate

# --- GAN specific parameters ---
# Defaults set for GAN training mode
GAN_TRAINING_MODE = True  # Set to true to run GAN training mode
GAN_MODEL_DIR = f"{BASE_OUTPUT_DIR}/models/gan"  # Directory to save/load GAN generator and discriminator
GENERATOR_LSTM_UNITS = 128
GENERATOR_ATTENTION_UNITS = 64
DISCRIMINATOR_LR = 0.0002
DISCRIMINATOR_BETA1 = 0.5
GENERATOR_LR = 0.0002
GENERATOR_BETA1 = 0.5
GAN_EPOCHS = 10000  # Default epochs for GAN training
GAN_BATCH_SIZE = 32    # Default batch size for GAN training
GAN_SAVE_INTERVAL = 500 # How often to save models during GAN training

# --- Activation for the output layer of generators (VAE/GAN) ---
OUTPUT_ACTIVATION = "sigmoid"  # or "tanh", depending on data normalization

# --- Type of generator to use in generation mode ('vae' or 'gan') ---
# This default is for when NOT in GAN_TRAINING_MODE.
# After GAN training, user would change this to 'gan' for generation using the trained GAN.
GENERATOR_TYPE = "vae"

# --- Feature names (USER MUST CONFIGURE THESE TO MATCH THEIR DATA AND MODELS) ---
# Example: if your data has 45 features and your VAE/GAN models expect these.
# N_FEATURES above should match len(FULL_FEATURE_NAMES_ORDERED)
FULL_FEATURE_NAMES_ORDERED = [f"feature_{i}" for i in range(N_FEATURES)] # Placeholder, user must update
DECODER_OUTPUT_FEATURE_NAMES = [f"feature_{i}" for i in range(min(N_FEATURES, 4))] # Placeholder, user must update
OHLC_FEATURE_NAMES = [f"feature_{i}" for i in range(min(N_FEATURES, 4))] # Placeholder, user must update


# --- Configuration dictionary ---
# This dictionary aggregates all parameters.
# It can be overridden by command-line arguments or a JSON config file.
config = {
    "seq_len": SEQ_LEN,
    "latent_dim": LATENT_DIM,
    "n_features": N_FEATURES,
    "x_train_file": X_TRAIN_FILE,
    "x_test_file": X_TEST_FILE,
    "output_file": OUTPUT_FILE,
    "feeder_plugin": FEEDER_PLUGIN,
    "generator_plugin": GENERATOR_PLUGIN, # VAEGeneratorPlugin is default, handles GAN model loading via generator_type
    "trainer_plugin": TRAINER_PLUGIN, # For main pipeline trainer, not GAN specific
    "vae_model_path": VAE_MODEL_PATH,
    "vae_epochs": VAE_EPOCHS,
    "vae_batch_size": VAE_BATCH_SIZE,
    "vae_save_interval": VAE_SAVE_INTERVAL,
    "prepend_length": PREPEND_LENGTH,
    "generation_length": GENERATION_LENGTH,
    "vae_training_mode": VAE_TRAINING_MODE, # For dedicated VAE training mode
    "gan_training_mode": GAN_TRAINING_MODE, # For dedicated GAN training mode
    "gan_model_dir": GAN_MODEL_DIR,
    "generator_lstm_units": GENERATOR_LSTM_UNITS,
    "generator_attention_units": GENERATOR_ATTENTION_UNITS,
    "discriminator_lr": DISCRIMINATOR_LR,
    "discriminator_beta1": DISCRIMINATOR_BETA1,
    "generator_lr": GENERATOR_LR,
    "generator_beta1": GENERATOR_BETA1,
    "gan_epochs": GAN_EPOCHS,
    "gan_batch_size": GAN_BATCH_SIZE,
    "gan_save_interval": GAN_SAVE_INTERVAL,
    "output_activation": OUTPUT_ACTIVATION,
    "generator_type": GENERATOR_TYPE, # Determines if VAEGeneratorPlugin loads a VAE or GAN model
    "full_feature_names_ordered": FULL_FEATURE_NAMES_ORDERED,
    "decoder_output_feature_names": DECODER_OUTPUT_FEATURE_NAMES,
    "ohlc_feature_names": OHLC_FEATURE_NAMES,
}

def get_config():
    """Returns a copy of the configuration dictionary."""
    return config.copy()

# Example of how to update config dynamically (e.g., from CLI args or a file)
def update_config(new_params):
    """Updates the global config with new parameters."""
    config.update(new_params)

# You might also have a function to load config from a JSON file, e.g.:
# import json
# def load_from_json(json_path):
#     with open(json_path, 'r') as f:
#         loaded_config = json.load(f)
#     update_config(loaded_config)
