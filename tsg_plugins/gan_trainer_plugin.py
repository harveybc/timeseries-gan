# optimizer/plugins/gan_plugin.py

import copy  # For deep-copying configuration dicts
import logging  # Standard logging module
import random  # Random number generation
import time  # Timing execution
from typing import Any, Dict, List, Tuple, Union, Optional # Added Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam # Corrected import
from tensorflow.keras.utils import plot_model # ADDED for model architecture plots
import matplotlib.pyplot as plt
import numpy as np
import os # Added os for path operations
import sys # Added sys for exit operations

# Assuming your generator plugin (like VAEGeneratorPlugin) and a new discriminator model definition exist
# from .generator_plugin import VAEGeneratorPlugin # Or your specific generator plugin
# from .discriminator_model import build_discriminator # You'll create this

# Initialize logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set default log level


class GANTrainerPlugin:
    plugin_params: Dict[str, Any] = {
        "gan_epochs": 10000,
        "gan_batch_size": 32,
        "generator_lr": 1e-4,
        "generator_beta1": 0.5,
        "discriminator_lr": 1e-4,
        "discriminator_beta1": 0.5,
        "gan_save_interval": 500,
        "latent_dim": 32, 
        "seq_len": 18,    
        "gan_model_dir": "models/gan_trained",
        "discriminator_lstm_units": 64,
        "discriminator_dense_units": 128,
        "gan_generator_output_actual_seq_len": 1,
        # Callbacks parameters
        "enable_reduce_lr_on_plateau": True,
        "lr_reduction_factor": 0.5, # Factor by which LR is reduced
        "lr_patience": 50,          # Epochs to wait for improvement before reducing LR
        "lr_min_delta": 0.001,      # Minimum change to qualify as improvement for LR reduction
        "min_lr_g": 1e-7,           # Minimum LR for generator
        "min_lr_d": 1e-7,           # Minimum LR for discriminator
        "lr_monitor_metric": "g_loss", # "g_loss" or "d_loss"
        "enable_early_stopping": True,
        "es_patience": 200,         # Epochs to wait for improvement before stopping
        "es_min_delta": 0.001,      # Minimum change to qualify as improvement for early stopping
        "es_monitor_metric": "g_loss", # "g_loss" or "d_loss"
    }

    def __init__(self, config: Dict[str, Any], generator_plugin_instance: Optional[Any] = None, feeder_plugin_instance: Optional[Any] = None, preprocessor_plugin_instance: Optional[Any] = None):
        logger.info("Initializing GANTrainerPlugin.")
        self.config = copy.deepcopy(config)
        self.params = self.plugin_params.copy()
        # Update self.params with config, ensuring plugin_params defaults are taken if not in config
        for key in self.plugin_params:
            if key in self.config:
                self.params[key] = self.config[key]
        # Add any keys from config that were not in plugin_params (e.g. dynamic ones)
        self.params.update(self.config)


        self.generator_plugin = generator_plugin_instance
        self.feeder_plugin = feeder_plugin_instance
        self.preprocessor_plugin = preprocessor_plugin_instance
        
        # Parameters for the GENERATOR'S INPUT
        # self.params["seq_len"] (e.g., 18) is the length of the latent sequence fed to the generator.
        # self.params["latent_dim"] (e.g., 32) is the dim of each latent vector in that sequence.
        self.gen_input_seq_len = self.params.get("seq_len", 18) # From latent_shape[0] via GANTrainerPlugin.plugin_params
        self.gen_input_latent_dim = self.params.get("latent_dim", 32) # From latent_shape[1] via GANTrainerPlugin.plugin_params

        # Parameters for the DISCRIMINATOR'S INPUT
        # This is based on what the generator *outputs* (or what we want the GAN to focus on from it).
        self.discriminator_target_n_features = len(self.params.get("generator_decoder_output_feature_names", []))
        if self.discriminator_target_n_features == 0:
            logger.warning("GANTrainerPlugin: 'generator_decoder_output_feature_names' is empty or not found. Defaulting discriminator_target_n_features to 1. THIS IS LIKELY INCORRECT.")
            self.discriminator_target_n_features = 1
        
        # The VAE decoder outputs a single step, so discriminator sequence length is 1.
        self.discriminator_target_seq_len = self.params.get("gan_generator_output_actual_seq_len", 1)

        logger.info(f"GANTrainerPlugin: Generator INPUT: seq_len={self.gen_input_seq_len}, latent_dim={self.gen_input_latent_dim}")
        logger.info(f"GANTrainerPlugin: Discriminator TARGET INPUT: seq_len={self.discriminator_target_seq_len}, n_features={self.discriminator_target_n_features}")

        self.actual_generator_output_dim = None
        if self.generator_plugin and hasattr(self.generator_plugin, 'model') and self.generator_plugin.model:
            self.generator = self.generator_plugin.model
            if self.generator:
                self.generator.trainable = False # Freeze pre-trained VAE decoder
                logger.info("GANTrainerPlugin: Set VAE decoder (generator) to trainable=False.")
                generator_output_shape = self.generator.output_shape
                if len(generator_output_shape) == 2: # Expected (batch_size, num_features)
                    self.actual_generator_output_dim = generator_output_shape[1]
                    logger.info(f"GANTrainerPlugin: Actual generator output dimension: {self.actual_generator_output_dim}")
                    if self.actual_generator_output_dim != self.discriminator_target_n_features:
                        logger.warning(f"GANTrainerPlugin: Generator output dim ({self.actual_generator_output_dim}) != discriminator target ({self.discriminator_target_n_features}). Output will be sliced.")
                else:
                    logger.error(f"GANTrainerPlugin: Generator output shape {generator_output_shape} is not 2D. Slicing might fail or be incorrect.")
            logger.info("Using generator model from provided generator_plugin instance.")
        else:
            logger.warning("Generator plugin instance or its model not provided. Generator will be None.")
            self.generator = None

        # Get shapes for conditional and context inputs for the VAE DECODER (Generator)
        if self.feeder_plugin and hasattr(self.feeder_plugin, 'params'):
            num_date_feats = len(self.feeder_plugin.params.get("feeder_date_features_for_conditioning", [])) * 2
            num_fund_feats = len(self.feeder_plugin.params.get("feeder_fundamental_features_for_conditioning", []))
            self.conditional_dim_for_generator = num_date_feats + num_fund_feats
            self.context_dim_for_generator = self.feeder_plugin.params.get("feeder_context_vector_dim", 64)
        else:
            # Corrected keys: removed "_names"
            num_date_feats = len(self.params.get("feeder_date_features_for_conditioning", [])) * 2
            num_fund_feats = len(self.params.get("feeder_fundamental_features_for_conditioning", []))
            self.conditional_dim_for_generator = num_date_feats + num_fund_feats
            self.context_dim_for_generator = self.params.get("feeder_context_vector_dim", 64) # Ensure this key is also correct if feeder_plugin is None
        logger.info(f"GANTrainer determined for VAE Decoder (Generator) inputs: conditional_dim={self.conditional_dim_for_generator}, context_dim_gan={self.context_dim_for_generator}")

        self.generator_optimizer = Adam(learning_rate=self.params["generator_lr"], beta_1=self.params["generator_beta1"])
        self.discriminator_optimizer = Adam(learning_rate=self.params["discriminator_lr"], beta_1=self.params["discriminator_beta1"])
        
        self.discriminator = self._build_discriminator()
        self.gan = self._build_gan()
        
        self.gan_model_dir = self.params.get("gan_model_dir", "models/gan_trained")
        os.makedirs(self.gan_model_dir, exist_ok=True)
        logger.info(f"GAN models will be saved in: {os.path.abspath(self.gan_model_dir)}")

    def set_params(self, **params: Any) -> None:
        logger.info(f"GANTrainerPlugin updating parameters: {list(params.keys())}")
        self.config.update(params)
        self.params = self.plugin_params.copy() # Start with defaults
        # Update self.params with config, ensuring plugin_params defaults are taken if not in config
        for key in self.plugin_params: # Iterate over known default keys
            if key in self.config:
                self.params[key] = self.config[key]
        # Add any keys from config that were not in plugin_params (e.g. dynamic ones from main config)
        self.params.update(self.config)


        self.gen_input_seq_len = self.params.get("seq_len", self.gen_input_seq_len)
        self.gen_input_latent_dim = self.params.get("latent_dim", self.gen_input_latent_dim)
        
        self.discriminator_target_n_features = len(self.params.get("generator_decoder_output_feature_names", []))
        if self.discriminator_target_n_features == 0:
            logger.warning("GANTrainerPlugin (set_params): 'generator_decoder_output_feature_names' is empty. Defaulting discriminator_target_n_features to 1.")
            self.discriminator_target_n_features = 1
        self.discriminator_target_seq_len = self.params.get("gan_generator_output_actual_seq_len", 1)

        logger.info(f"GANTrainerPlugin (set_params): Gen INPUT seq_len={self.gen_input_seq_len}, latent_dim={self.gen_input_latent_dim}")
        logger.info(f"GANTrainerPlugin (set_params): Disc TARGET INPUT seq_len={self.discriminator_target_seq_len}, n_features={self.discriminator_target_n_features}")

        if self.generator: # Re-check actual generator output dim if params affecting it could change (unlikely for frozen model)
            generator_output_shape = self.generator.output_shape
            if len(generator_output_shape) == 2:
                self.actual_generator_output_dim = generator_output_shape[1]
                if self.actual_generator_output_dim != self.discriminator_target_n_features:
                    logger.warning(f"GANTrainerPlugin (set_params): Gen output dim ({self.actual_generator_output_dim}) != Disc target ({self.discriminator_target_n_features}). Slicing.")
            else:
                 logger.error(f"GANTrainerPlugin (set_params): Gen output shape {generator_output_shape} not 2D.")
        
        # Re-initialize optimizers and model dir
        self.generator_optimizer = Adam(learning_rate=self.params["generator_lr"], beta_1=self.params["generator_beta1"])
        self.discriminator_optimizer = Adam(learning_rate=self.params["discriminator_lr"], beta_1=self.params["discriminator_beta1"])
        self.gan_model_dir = self.params.get("gan_model_dir", self.gan_model_dir)
        os.makedirs(self.gan_model_dir, exist_ok=True)
        # Potentially rebuild models if critical architectural params changed, though less common for set_params
        # self.discriminator = self._build_discriminator()
        # self.gan = self._build_gan()


    def _build_discriminator(self) -> tf.keras.Model:
        logger.info("Building Discriminator model...")
        # Discriminator input shape is based on the (potentially sliced and reshaped) generator output
        input_shape_disc = (self.discriminator_target_seq_len, self.discriminator_target_n_features)
        logger.info(f"Discriminator input shape: {input_shape_disc}")

        model = tf.keras.Sequential(name="discriminator")
        # If seq_len is 1, LSTM might be overkill or behave like a Dense layer on flattened input.
        model.add(tf.keras.layers.LSTM(self.params.get("discriminator_lstm_units", 64), input_shape=input_shape_disc, return_sequences=True if self.discriminator_target_seq_len > 1 else False))
        if self.discriminator_target_seq_len > 1: # Add Flatten only if LSTM returns sequences
            model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(self.params.get("discriminator_dense_units", 128), activation='relu'))
        model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        
        model.compile(loss='binary_crossentropy', optimizer=self.discriminator_optimizer, metrics=['accuracy'])
        logger.info("Discriminator model built and compiled.")
        model.summary(print_fn=logger.info)
        try:
            plot_model(model, to_file=os.path.join(self.gan_model_dir, 'discriminator_model_plot.png'), show_shapes=True, show_layer_names=True)
            logger.info(f"Discriminator model plot saved to {os.path.join(self.gan_model_dir, 'discriminator_model_plot.png')}")
        except Exception as e:
            logger.warning(f"Could not plot discriminator model: {e}. Ensure pydot and graphviz are installed.")
        return model

    def _build_gan(self) -> tf.keras.Model:
        if not self.generator or not self.discriminator:
            logger.error("Generator or Discriminator not initialized. Cannot build GAN.")
            raise ValueError("Generator or Discriminator not initialized.")
            
        logger.info("Building GAN model (Generator + Discriminator)...")
        self.discriminator.trainable = False
        
        # GAN inputs are those required by the VAE Decoder (Generator)
        gan_latent_input_shape = (self.gen_input_seq_len, self.gen_input_latent_dim)
        gan_latent_input = tf.keras.Input(shape=gan_latent_input_shape, name="gan_input_latent_vector")
        gan_conditional_input = tf.keras.Input(shape=(self.conditional_dim_for_generator,), name="gan_input_conditional_data")
        gan_context_input = tf.keras.Input(shape=(self.context_dim_for_generator,), name="gan_input_context_vector")

        # Order inputs for the VAE Decoder (Generator)
        # (Assuming the input ordering logic from previous patch is still valid for VAE decoder)
        generator_input_names_from_model = [inp.name.split(':')[0] for inp in self.generator.inputs]
        cfg_latent_name = self.params.get("generator_decoder_input_name_latent")
        cfg_context_name = self.params.get("generator_decoder_input_name_context")
        cfg_conditional_name = self.params.get("generator_decoder_input_name_conditions")

        input_map = {
            cfg_latent_name: gan_latent_input,
            cfg_context_name: gan_context_input,
            cfg_conditional_name: gan_conditional_input,
        }
        generator_feed_inputs_ordered = []
        for model_input_name in generator_input_names_from_model:
            found_match = False
            for cfg_name, gan_layer in input_map.items():
                if cfg_name and model_input_name.startswith(cfg_name):
                    generator_feed_inputs_ordered.append(gan_layer)
                    found_match = True
                    break
            if not found_match:
                logger.warning(f"GAN build: Generator model input '{model_input_name}' not mapped.")
        
        if len(generator_feed_inputs_ordered) != len(self.generator.inputs):
             logger.error(f"GAN build: Mismatch in ordered inputs for generator. Expected {len(self.generator.inputs)}, got {len(generator_feed_inputs_ordered)}")
             # This could be a critical error.

        generated_data_raw = self.generator(generator_feed_inputs_ordered) # Shape (None, actual_generator_output_dim) e.g. (None, 23)
        
        # Slice if actual generator output dim differs from what discriminator targets
        generated_data_for_disc = generated_data_raw
        if self.actual_generator_output_dim and self.actual_generator_output_dim != self.discriminator_target_n_features:
            logger.info(f"GAN build: Slicing generator output from {self.actual_generator_output_dim} to {self.discriminator_target_n_features} features.")
            # Slicing KerasTensors is generally fine using Python slicing syntax
            generated_data_for_disc = generated_data_raw[:, :self.discriminator_target_n_features]
        
        # Reshape for discriminator: (None, disc_target_n_features) -> (None, disc_target_seq_len, disc_target_n_features)
        # Use Keras Reshape layer instead of tf.reshape
        target_shape_for_discriminator = (self.discriminator_target_seq_len, self.discriminator_target_n_features)
        generated_data_reshaped = tf.keras.layers.Reshape(target_shape_for_discriminator)(generated_data_for_disc)
        
        gan_output = self.discriminator(generated_data_reshaped)
        
        actual_gan_model_inputs = [gan_latent_input, gan_conditional_input, gan_context_input]
        gan = tf.keras.Model(inputs=actual_gan_model_inputs, outputs=gan_output, name="gan_combined")
        gan.compile(loss='binary_crossentropy', optimizer=self.generator_optimizer)
        logger.info("GAN model built and compiled.")
        gan.summary(print_fn=logger.info)
        try:
            plot_model(gan, to_file=os.path.join(self.gan_model_dir, 'gan_model_plot.png'), show_shapes=True, show_layer_names=True, expand_nested=True)
            logger.info(f"GAN model plot saved to {os.path.join(self.gan_model_dir, 'gan_model_plot.png')}")
            # Optionally plot the generator (VAE decoder) if it's not too complex and hasn't been plotted elsewhere
            if self.generator:
                 plot_model(self.generator, to_file=os.path.join(self.gan_model_dir, 'gan_generator_component_plot.png'), show_shapes=True, show_layer_names=True, expand_nested=True)
                 logger.info(f"GAN's generator component plot saved to {os.path.join(self.gan_model_dir, 'gan_generator_component_plot.png')}")

        except Exception as e:
            logger.warning(f"Could not plot GAN model: {e}. Ensure pydot and graphviz are installed.")
        return gan

    def train(self, x_train_file: Optional[str] = None, data: Optional[np.ndarray] = None) -> None:
        if self.generator is None or self.discriminator is None or self.gan is None:
            logger.error("Models not built. Cannot start training.")
            return

        logger.info("--- Discriminator Model Summary (at start of training) ---")
        self.discriminator.summary(print_fn=logger.info)
        logger.info("--- End Discriminator Model Summary ---")
        
        # Ensure model directory exists for plots even before first save_models call
        os.makedirs(self.gan_model_dir, exist_ok=True)
        # Attempt to plot models here if not done in _build_gan or if dir wasn't ready
        if not os.path.exists(os.path.join(self.gan_model_dir, 'discriminator_model_plot.png')):
            try:
                plot_model(self.discriminator, to_file=os.path.join(self.gan_model_dir, 'discriminator_model_plot.png'), show_shapes=True, show_layer_names=True)
            except Exception: pass # Ignore if fails, already tried in build
        if not os.path.exists(os.path.join(self.gan_model_dir, 'gan_model_plot.png')):
            try:
                plot_model(self.gan, to_file=os.path.join(self.gan_model_dir, 'gan_model_plot.png'), show_shapes=True, show_layer_names=True, expand_nested=True)
            except Exception: pass # Ignore if fails
        if self.generator and not os.path.exists(os.path.join(self.gan_model_dir, 'gan_generator_component_plot.png')):
            try:
                plot_model(self.generator, to_file=os.path.join(self.gan_model_dir, 'gan_generator_component_plot.png'), show_shapes=True, show_layer_names=True, expand_nested=True)
            except Exception: pass


        # Configure preprocessor to output data suitable for discriminator
        # (seq_len=self.discriminator_target_seq_len, features=self.discriminator_target_n_features)
        preprocess_config_for_gan = self.config.copy()
        # The preprocessor should use 'generator_decoder_output_feature_names' for feature selection (now target_n_features)
        # And its 'window_size' or equivalent should be set to self.discriminator_target_seq_len (which is 1)
        # This assumes preprocessor can be configured this way.
        
        preprocess_config_for_gan['seq_len'] = self.discriminator_target_seq_len
        preprocess_config_for_gan['window_size'] = self.discriminator_target_seq_len
        # The line for 'feature_names_to_use' is removed/kept commented as preprocessor cannot be changed.


        real_data_numpy: Optional[np.ndarray] = None # Initialize to ensure it's defined

        if data is None and x_train_file is not None:
            logger.info(f"Loading training data from: {x_train_file} for GAN using preprocessor.")
            if not self.preprocessor_plugin:
                logger.error("Preprocessor plugin not available.")
                raise ValueError("Preprocessor plugin required for x_train_file.")
            
            processed_data_dict = self.preprocessor_plugin.run_preprocessing(config=preprocess_config_for_gan)
            real_data_from_preprocessor = processed_data_dict.get("x_train")
            all_feature_names_from_preprocessor = processed_data_dict.get("feature_names")
            
            if real_data_from_preprocessor is None:
                logger.error(f"Preprocessor did not return 'x_train' data from {x_train_file}.")
                raise ValueError("Failed to load/process training data via preprocessor.")
            if not all_feature_names_from_preprocessor:
                logger.error(f"Preprocessor did not return 'feature_names'. Cannot select features for discriminator.")
                raise ValueError("Preprocessor must return 'feature_names'.")

            if isinstance(real_data_from_preprocessor, pd.DataFrame):
                real_data_from_preprocessor = real_data_from_preprocessor.values
            if not isinstance(real_data_from_preprocessor, np.ndarray):
                real_data_from_preprocessor = np.array(real_data_from_preprocessor)

            # Ensure data from preprocessor is 3D: (samples, seq_len, features_all)
            # The preprocessor output is (N, 1, 54), so it should be 3D.
            if real_data_from_preprocessor.ndim == 2 and self.discriminator_target_seq_len == 1:
                logger.info(f"Reshaping data from preprocessor from {real_data_from_preprocessor.shape} to 3D for GAN.")
                real_data_from_preprocessor = np.reshape(real_data_from_preprocessor, (-1, 1, real_data_from_preprocessor.shape[1]))
            elif real_data_from_preprocessor.ndim != 3:
                logger.error(f"Data from preprocessor has unexpected shape {real_data_from_preprocessor.shape}. Expected 3D or 2D (if seq_len=1).")
                raise ValueError("Preprocessor output shape not suitable for GAN.")

            # --- Select specific features for the discriminator ---
            discriminator_expected_feature_names = self.params.get("generator_decoder_output_feature_names")
            if not discriminator_expected_feature_names:
                logger.error("'generator_decoder_output_feature_names' not configured. Cannot select features for discriminator.")
                raise ValueError("Missing 'generator_decoder_output_feature_names' in configuration.")

            logger.info(f"Preprocessor output {real_data_from_preprocessor.shape[2]} features: {all_feature_names_from_preprocessor}")
            logger.info(f"Discriminator expects {len(discriminator_expected_feature_names)} features: {discriminator_expected_feature_names}")

            try:
                selected_feature_indices = [all_feature_names_from_preprocessor.index(name) for name in discriminator_expected_feature_names]
            except ValueError as e:
                missing_feature = str(e).split("'")[1] # Extract missing feature name
                logger.error(f"Feature selection error: A feature expected by the discriminator ('{missing_feature}') was not found in the preprocessor's output. ")
                logger.error(f"Expected features by discriminator: {discriminator_expected_feature_names}")
                logger.error(f"Available features from preprocessor: {all_feature_names_from_preprocessor}")
                raise ValueError(f"Feature mismatch: '{missing_feature}' not in preprocessor output.") from e
            
            real_data_numpy = real_data_from_preprocessor[:, :, selected_feature_indices]
            logger.info(f"Data sliced from {real_data_from_preprocessor.shape} to {real_data_numpy.shape} for discriminator based on 'generator_decoder_output_feature_names'.")

        elif data is not None:
            logger.info("Using provided 'data' array for training. Assuming it has the correct shape and features for the discriminator.")
            real_data_numpy = data
            if not isinstance(real_data_numpy, np.ndarray):
                real_data_numpy = np.array(real_data_numpy)
        else:
            logger.error("No training data provided (neither x_train_file nor data array).")
            return

        # Validate the final shape of real_data_numpy before proceeding
        logger.info(f"Shape of final training data for GAN (real_data_numpy): {real_data_numpy.shape}")
        if real_data_numpy.ndim != 3 or \
           real_data_numpy.shape[1] != self.discriminator_target_seq_len or \
           real_data_numpy.shape[2] != self.discriminator_target_n_features:
            logger.error(f"Final training data shape {real_data_numpy.shape} MISMATCHES discriminator input expectation (samples, {self.discriminator_target_seq_len}, {self.discriminator_target_n_features}).")
            raise ValueError("Real data final shape mismatch for discriminator after processing.")
        
        # Use real_data_numpy for training from here onwards
        epochs = self.params["gan_epochs"]
        batch_size = self.params["gan_batch_size"]
        save_interval = self.params["gan_save_interval"]

        # Callback: ReduceLROnPlateau settings
        enable_reduce_lr = self.params.get("enable_reduce_lr_on_plateau", False)
        lr_reduction_factor = self.params.get("lr_reduction_factor", 0.5)
        lr_patience = self.params.get("lr_patience", 50)
        lr_min_delta = self.params.get("lr_min_delta", 0.001)
        min_lr_g = self.params.get("min_lr_g", 1e-7)
        min_lr_d = self.params.get("min_lr_d", 1e-7)
        lr_monitor_metric_name = self.params.get("lr_monitor_metric", "g_loss")

        best_lr_metric_val = float('inf')
        lr_patience_counter = 0
        lr_reductions_count = 0

        # Callback: EarlyStopping settings
        enable_early_stop = self.params.get("enable_early_stopping", False)
        es_patience = self.params.get("es_patience", 200)
        es_min_delta = self.params.get("es_min_delta", 0.001)
        es_monitor_metric_name = self.params.get("es_monitor_metric", "g_loss")
        
        best_es_metric_val = float('inf')
        es_patience_counter = 0


        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        d_losses, g_losses = [], []

        for epoch in range(epochs):
            start_time_epoch = time.time()
            
            idx = np.random.randint(0, real_data_numpy.shape[0], batch_size)
            real_batch = real_data_numpy[idx] # Shape (batch_size, disc_target_seq_len, disc_target_n_features)

            if not self.feeder_plugin:
                logger.error("Feeder plugin not available.")
                raise ValueError("Feeder plugin required.")

            # Feeder generates inputs for the VAE Decoder (Generator)
            feeder_output_for_gen = self.feeder_plugin.generate(
                n_ticks_to_generate=batch_size,
                # Feeder needs to know the generator's input sequence length (gen_input_seq_len)
                # This should be handled by FeederPlugin's own config (e.g. latent_shape)
            )
            
            latent_input_for_generator = feeder_output_for_gen.get('latent_vector_batch') # Shape (batch, gen_input_seq_len, gen_input_latent_dim)
            conditional_input_for_generator = feeder_output_for_gen.get('conditional_data_batch') # Shape (batch, conditional_dim_for_generator)
            context_input_for_generator = feeder_output_for_gen.get('context_h_batch') # Shape (batch, context_dim_for_generator)

            # (Error handling for missing feeder outputs as before)
            if latent_input_for_generator is None or conditional_input_for_generator is None or context_input_for_generator is None:
                logger.error("FeederPlugin did not return all required inputs for the generator.")
                # Simplified fallback for brevity, ensure shapes are correct
                latent_input_for_generator = np.random.normal(0, 1, (batch_size, self.gen_input_seq_len, self.gen_input_latent_dim))
                conditional_input_for_generator = np.zeros((batch_size, self.conditional_dim_for_generator))
                context_input_for_generator = np.zeros((batch_size, self.context_dim_for_generator))

            # Order inputs for self.generator.predict
            generator_input_names_from_model = [inp.name.split(':')[0] for inp in self.generator.inputs]
            cfg_latent_name = self.params.get("generator_decoder_input_name_latent")
            cfg_context_name = self.params.get("generator_decoder_input_name_context")
            cfg_conditional_name = self.params.get("generator_decoder_input_name_conditions")
            input_data_map_predict = {
                cfg_latent_name: latent_input_for_generator,
                cfg_context_name: context_input_for_generator,
                cfg_conditional_name: conditional_input_for_generator,
            }
            generator_predict_inputs_ordered = []
            for model_input_name in generator_input_names_from_model:
                found_match = False
                for cfg_name, data_array in input_data_map_predict.items():
                    if cfg_name and model_input_name.startswith(cfg_name):
                        generator_predict_inputs_ordered.append(data_array)
                        found_match = True
                        break
                if not found_match: logger.warning(f"Train loop: Predict input '{model_input_name}' not mapped.")
            
            # Change this line:
            # generated_batch_raw = self.generator.predict(generator_predict_inputs_ordered)
            # To this:
            generated_batch_raw = self.generator.predict(generator_predict_inputs_ordered, verbose=0)


            # Slice if needed
            generated_batch_for_disc = generated_batch_raw
            if self.actual_generator_output_dim and self.actual_generator_output_dim != self.discriminator_target_n_features:
                generated_batch_for_disc = generated_batch_raw[:, :self.discriminator_target_n_features] # Slice to (bs, 21)
            
            # Reshape for discriminator
            generated_batch_reshaped = np.reshape(
                generated_batch_for_disc, 
                (-1, self.discriminator_target_seq_len, self.discriminator_target_n_features) # e.g. (bs, 1, 21)
            )

            d_loss_real = self.discriminator.train_on_batch(real_batch, valid)
            d_loss_fake = self.discriminator.train_on_batch(generated_batch_reshaped, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train Generator (GAN)
            # Feeder generates inputs for VAE Decoder again for GAN's forward pass
            feeder_output_for_gan_train = self.feeder_plugin.generate(n_ticks_to_generate=batch_size)
            latent_input_for_gan_train = feeder_output_for_gan_train.get('latent_vector_batch')
            conditional_input_for_gan_train = feeder_output_for_gan_train.get('conditional_data_batch')
            context_input_for_gan_train = feeder_output_for_gan_train.get('context_h_batch')
            # (Error handling for missing feeder outputs as before)
            if latent_input_for_gan_train is None or conditional_input_for_gan_train is None or context_input_for_gan_train is None:
                logger.error("FeederPlugin did not return all required inputs for GAN training step.")
                latent_input_for_gan_train = np.random.normal(0, 1, (batch_size, self.gen_input_seq_len, self.gen_input_latent_dim))
                conditional_input_for_gan_train = np.zeros((batch_size, self.conditional_dim_for_generator))
                context_input_for_gan_train = np.zeros((batch_size, self.context_dim_for_generator))

            gan_train_inputs_ordered = [latent_input_for_gan_train, conditional_input_for_gan_train, context_input_for_gan_train]
            g_loss = self.gan.train_on_batch(gan_train_inputs_ordered, valid)
            
            epoch_time = time.time() - start_time_epoch
            d_losses.append(d_loss[0])
            g_losses.append(g_loss)

            # --- Callback Logic ---
            current_lr_metric_val = g_loss if lr_monitor_metric_name == "g_loss" else d_loss[0]
            current_es_metric_val = g_loss if es_monitor_metric_name == "g_loss" else d_loss[0]

            # ReduceLROnPlateau
            if enable_reduce_lr:
                if current_lr_metric_val < best_lr_metric_val - lr_min_delta:
                    best_lr_metric_val = current_lr_metric_val
                    lr_patience_counter = 0
                else:
                    lr_patience_counter += 1
                
                if lr_patience_counter >= lr_patience:
                    lr_patience_counter = 0
                    lr_reductions_count += 1
                    
                    old_lr_g = float(tf.keras.backend.get_value(self.generator_optimizer.learning_rate))
                    # Cast new_lr_g to the optimizer's learning rate dtype before setting
                    lr_dtype_g = self.generator_optimizer.learning_rate.dtype # Get the DType object
                    new_lr_g = tf.cast(max(old_lr_g * lr_reduction_factor, min_lr_g), dtype=lr_dtype_g)
                    tf.keras.backend.set_value(self.generator_optimizer.learning_rate, new_lr_g)
                    
                    old_lr_d = float(tf.keras.backend.get_value(self.discriminator_optimizer.learning_rate))
                    # Cast new_lr_d to the optimizer's learning rate dtype before setting
                    lr_dtype_d = self.discriminator_optimizer.learning_rate.dtype # Get the DType object
                    new_lr_d = tf.cast(max(old_lr_d * lr_reduction_factor, min_lr_d), dtype=lr_dtype_d)
                    tf.keras.backend.set_value(self.discriminator_optimizer.learning_rate, new_lr_d)
                    
                    # For logging, convert back to float if needed, or use .numpy()
                    log_new_lr_g = float(new_lr_g.numpy()) if hasattr(new_lr_g, 'numpy') else float(new_lr_g)
                    log_new_lr_d = float(new_lr_d.numpy()) if hasattr(new_lr_d, 'numpy') else float(new_lr_d)
                    logger.info(f"ReduceLROnPlateau: Epoch {epoch}. Reducing LRs. G: {old_lr_g:.2e}->{log_new_lr_g:.2e}, D: {old_lr_d:.2e}->{log_new_lr_d:.2e}. Reductions: {lr_reductions_count}.")
            
            # EarlyStopping
            if enable_early_stop:
                if current_es_metric_val < best_es_metric_val - es_min_delta:
                    best_es_metric_val = current_es_metric_val
                    es_patience_counter = 0
                else:
                    es_patience_counter += 1

                if es_patience_counter >= es_patience:
                    logger.info(f"EarlyStopping: Epoch {epoch}. Monitored metric ({es_monitor_metric_name}) did not improve for {es_patience} epochs. Stopping training.")
                    break # Exit epoch loop

            log_message_parts = [
                f"{epoch}/{epochs}",
                f"[D loss: {d_loss[0]:.4f}, acc.: {100*d_loss[1]:.2f}%]",
                f"[G loss: {g_loss:.4f}]",
                f"time: {epoch_time:.2f}s"
            ]
            if enable_reduce_lr:
                log_message_parts.append(f"[LR Patience: {lr_patience_counter}/{lr_patience}, Reductions: {lr_reductions_count}]")
            if enable_early_stop:
                log_message_parts.append(f"[ES Patience: {es_patience_counter}/{es_patience}]")

            if epoch % 100 == 0 or epoch == epochs -1 :
                logger.info(" ".join(log_message_parts))


            if epoch % save_interval == 0 and epoch > 0:
                self.save_models(epoch)
        
        self.save_models(epochs)
        logger.info("GAN Training completed.")
        self._plot_losses(d_losses, g_losses)


    def _plot_losses(self, d_losses: List[float], g_losses: List[float]):
        """Plots generator and discriminator loss."""
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 5))
            plt.plot(d_losses, label='Discriminator Loss')
            plt.plot(g_losses, label='Generator Loss')
            plt.title('GAN Training Losses')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            loss_plot_file = self.params.get("loss_plot_file", os.path.join(self.gan_model_dir, "gan_loss_plot.png"))
            plt.savefig(loss_plot_file)
            plt.close()
            logger.info(f"GAN loss plot saved to {loss_plot_file}")
        except ImportError:
            logger.warning("Matplotlib not installed. Skipping loss plotting.")
        except Exception as e:
            logger.error(f"Error plotting losses: {e}")


    def save_models(self, epoch: int) -> None:
        """Saves the generator and discriminator models."""
        if self.generator:
            g_path = os.path.join(self.gan_model_dir, f"generator_epoch_{epoch}.h5")
            self.generator.save(g_path)
            logger.info(f"Saved generator model to {g_path}")
        if self.discriminator:
            d_path = os.path.join(self.gan_model_dir, f"discriminator_epoch_{epoch}.h5")
            self.discriminator.save(d_path)
            logger.info(f"Saved discriminator model to {d_path}")

    def get_generator(self) -> Optional[tf.keras.Model]:
        """Returns the trained generator model."""
        return self.generator

    def get_discriminator(self) -> Optional[tf.keras.Model]:
        """Returns the trained discriminator model."""
        return self.discriminator
