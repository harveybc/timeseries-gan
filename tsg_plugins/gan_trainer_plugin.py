# optimizer/plugins/gan_plugin.py

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
import pandas as pd
import pandas_ta as ta
import numpy as np
import logging
import os
import time
from copy import deepcopy
from typing import Any, Dict, List, Tuple, Union, Optional

# Assuming your generator plugin (like VAEGeneratorPlugin) and a new discriminator model definition exist
# from .generator_plugin import VAEGeneratorPlugin # Or your specific generator plugin
# from .discriminator_model import build_discriminator # You'll create this

# Initialize logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set default log level
# ADD INFO MESSAGE ABOUT GENERATOR WARNING
logger.info("GANTrainerPlugin: The VAE generator is intended to be frozen during GAN training. A 'UserWarning: The model does not have any trainable weights.' may appear when generator.predict() is called; this is expected for the frozen generator and does not affect discriminator or GAN training.")


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
        # Discriminator architecture params
        "discriminator_conv_filters": [64, 128], # List of filters for Conv1D layers
        "discriminator_conv_kernel_size": 3,      # Kernel size for Conv1D layers
        "discriminator_lstm_units": 64,
        "discriminator_dropout_rate": 0.3,
        # "discriminator_dense_units": 128, # REMOVED - Replaced by Conv1D/LSTM focus
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
        self.config = deepcopy(config)
        self.params = self.plugin_params.copy()
        self.set_params(**self.config)

        self.generator_plugin_instance = generator_plugin_instance
        self.feeder_plugin_instance = feeder_plugin_instance
        self.preprocessor_plugin_instance = preprocessor_plugin_instance

        if not self.generator_plugin_instance:
            raise ValueError("Generator plugin instance is required for GANTrainerPlugin.")
        if not self.feeder_plugin_instance:
            raise ValueError("Feeder plugin instance is required for GANTrainerPlugin.")
            
        self.generator: Optional[Model] = self.generator_plugin_instance.get_model()
        if self.generator is None:
            raise ValueError("Could not retrieve model from generator_plugin_instance.")
        self.generator.trainable = False # Freeze the generator

        # Feature configuration for TI calculation
        self.base_feature_names = self.config.get("base_feature_names_ordered", [])
        self.discriminator_feature_names = self.config.get("feature_names_for_discriminator_ordered", [])

        if not self.base_feature_names:
            raise ValueError("Config 'base_feature_names_ordered' is required for TI calculation.")
        if not self.discriminator_feature_names:
            raise ValueError("Config 'feature_names_for_discriminator_ordered' is required.")

        self.num_base_features = len(self.base_feature_names)
        self.num_features_for_discriminator = len(self.discriminator_feature_names)
        
        # Verify generator output matches num_base_features
        # Assuming generator model is already built and its output shape is known
        # For example, via generator_plugin_instance.params.get("output_feature_dim")
        # or self.generator.output_shape[-1]
        gen_output_dim_config = self.generator_plugin_instance.params.get("output_feature_dim")
        if gen_output_dim_config != self.num_base_features:
            logger.warning(
                f"Generator's configured output_feature_dim ({gen_output_dim_config}) "
                f"differs from num_base_features ({self.num_base_features}) derived from "
                f"'base_feature_names_ordered'. Ensure generator produces {self.num_base_features} base features."
            )
            # This could be an issue if the pre-trained generator doesn't output exactly num_base_features.
            # For now, we proceed assuming it does, or _calculate_technical_indicators will fail.

        if not all(self.discriminator_feature_names[i] == self.base_feature_names[i] for i in range(self.num_base_features)):
            raise ValueError("'base_feature_names_ordered' must be the prefix of 'feature_names_for_discriminator_ordered'.")
        
        self.ti_names_to_calculate = self.discriminator_feature_names[self.num_base_features:]
        logger.info(f"Base features to be generated: {self.base_feature_names}")
        logger.info(f"TIs to be calculated on generated base features: {self.ti_names_to_calculate}")
        logger.info(f"Total features for discriminator: {self.num_features_for_discriminator} ({self.discriminator_feature_names})")


        self.seq_len = self.params["seq_len"]
        self.latent_dim = self.params["latent_dim"]
        
        # Corrected conditional_dim for generator based on feeder plugin's output
        self.conditional_dim_for_generator = self.feeder_plugin_instance.get_conditional_dim()
        self.context_vector_dim_for_generator = self.feeder_plugin_instance.get_context_vector_dim()


        self.discriminator: Optional[Model] = self._build_discriminator()
        self.gan: Optional[Model] = self._build_gan()

        self.d_optimizer = Adam(learning_rate=self.params["discriminator_lr"], beta_1=self.params["discriminator_beta1"])
        self.g_optimizer = Adam(learning_rate=self.params["generator_lr"], beta_1=self.params["generator_beta1"])
        
        # For manual callbacks
        self.best_lr_metric = float('inf')
        self.lr_patience_counter = 0
        self.best_es_metric = float('inf')
        self.es_patience_counter = 0

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


    def _build_discriminator(self) -> Model:
        logger.info("Building Discriminator...")
        
        data_input = layers.Input(shape=(self.seq_len, self.num_features_for_discriminator), name="discriminator_input")
        
        x = data_input
        
        # Conv1D layers
        conv_filters = self.params.get("discriminator_conv_filters", [64, 128])
        kernel_size = self.params.get("discriminator_conv_kernel_size", 3)
        for filters in conv_filters:
            x = layers.Conv1D(filters=filters, kernel_size=kernel_size, padding='causal', activation='relu')(x)
            x = layers.BatchNormalization()(x) # Added BatchNormalization
            x = layers.SpatialDropout1D(self.params.get("discriminator_dropout_rate", 0.3) / 2)(x) # Added SpatialDropout1D

        # Bidirectional LSTM layer
        lstm_units = self.params.get("discriminator_lstm_units", 64)
        # x = layers.Bidirectional(layers.LSTM(units=lstm_units, return_sequences=True))(x) # If another LSTM or TimeDistributed Dense follows
        # x = layers.BatchNormalization()(x) # Added BatchNormalization
        x = layers.Bidirectional(layers.LSTM(units=lstm_units, return_sequences=False))(x) # return_sequences=False for final feature vector
        x = layers.BatchNormalization()(x) # Added BatchNormalization
        
        x = layers.Dropout(self.params.get("discriminator_dropout_rate", 0.3))(x)
        
        # Output layer
        output = layers.Dense(1, activation='sigmoid', name="discriminator_output")(x)
        
        model = Model(data_input, output, name="Discriminator")
        logger.info("Discriminator built successfully.")
        model.summary(print_fn=logger.info)
        return model

    def _build_gan(self) -> Model:
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
        if data is None and x_train_file is None:
            raise ValueError("Either data or x_train_file must be provided for training.")

        if data is None:
            logger.info(f"Loading training data from {x_train_file}...")
            # Assuming x_train_file is a numpy file or similar loadable format
            # This part needs to align with how data is actually stored/loaded.
            # For now, let's assume it's a .npy file with shape (num_samples, seq_len, num_features_from_preprocessor)
            # The preprocessor outputs 54 features.
            try:
                real_data_full = np.load(x_train_file)
                logger.info(f"Loaded data shape: {real_data_full.shape}")
            except Exception as e:
                logger.error(f"Failed to load data from {x_train_file}: {e}")
                raise
        else:
            real_data_full = data
            logger.info(f"Using provided data with shape: {real_data_full.shape}")

        if real_data_full.shape[-1] < self.num_features_for_discriminator:
            raise ValueError(
                f"Loaded data has {real_data_full.shape[-1]} features, but discriminator expects "
                f"{self.num_features_for_discriminator} features after slicing/TI calculation. "
                f"Ensure input data (from preprocessor) has at least {self.num_features_for_discriminator} features."
            )

        gan_epochs = self.params["gan_epochs"]
        batch_size = self.params["gan_batch_size"]
        gan_save_interval = self.params["gan_save_interval"]

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        # For storing losses
        d_losses_history = []
        g_losses_history = []
        d_accs_history = []

        # Initialize learning rates for manual callback tracking
        current_lr_g = float(self.g_optimizer.learning_rate.numpy())
        current_lr_d = float(self.d_optimizer.learning_rate.numpy())

        logger.info(f"Starting GAN training for {gan_epochs} epochs with batch size {batch_size}...")
        logger.info(f"Initial Generator LR: {current_lr_g}, Discriminator LR: {current_lr_d}")
        logger.info(f"Generator is FROZEN. Discriminator is TRAINABLE.")
        logger.info(f"Discriminator input shape: (batch_size, {self.seq_len}, {self.num_features_for_discriminator})")
        logger.info(f"Generator output (base features) shape: (batch_size, {self.seq_len}, {self.num_base_features})")


        for epoch in range(gan_epochs):
            start_time_epoch = time.time()
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            
            # Get a random batch of real samples
            idx = np.random.randint(0, real_data_full.shape[0], batch_size)
            real_data_batch_raw = real_data_full[idx] # (batch_size, seq_len, features_from_preprocessor)
            
            # Slice to the features the discriminator will see (first N features from preprocessor output)
            # These are assumed to be correctly ordered (base + TIs) by the unmodifiable preprocessor
            real_data_for_discriminator = real_data_batch_raw[:, :, :self.num_features_for_discriminator]

            # Generate a batch of new series using the FeederPlugin and Generator
            # FeederPlugin provides latent_input, conditional_input, context_input
            feeder_outputs = self.feeder_plugin_instance.generate(n_samples=batch_size)
            latent_input_batch = feeder_outputs["latent_vectors"]
            conditional_input_batch = feeder_outputs["conditional_vectors"]
            context_input_batch = feeder_outputs["context_vectors"]

            generator_inputs = [latent_input_batch, conditional_input_batch, context_input_batch]
            
            # Generator produces base features
            generated_base_features_raw = self.generator.predict(generator_inputs, verbose=0) # (batch, seq_len, num_base_features)

            # Calculate TIs for generated data
            generated_features_with_tis = self._calculate_technical_indicators(generated_base_features_raw) # (batch, seq_len, num_features_for_discriminator)
            generated_data_for_discriminator = generated_features_with_tis

            # Train the discriminator
            # Note: tf.GradientTape is used for custom training loops.
            # If using model.compile() and model.train_on_batch(), this is handled internally.
            # The current structure seems to use train_on_batch.
            
            d_loss_real = self.discriminator.train_on_batch(real_data_for_discriminator, valid)
            d_loss_fake = self.discriminator.train_on_batch(generated_data_for_discriminator, fake)
            d_loss = 0.5 * np.add(d_loss_real[0], d_loss_fake[0]) # Loss is typically the first element
            d_acc = 0.5 * np.add(d_loss_real[1], d_loss_fake[1]) # Accuracy is typically the second

            # ---------------------
            #  Train Generator
            # ---------------------
            # The GAN model (combined) trains the generator to fool the discriminator
            
            # Generate new inputs for the generator for this training step
            feeder_outputs_for_g = self.feeder_plugin_instance.generate(n_samples=batch_size)
            latent_input_batch_g = feeder_outputs_for_g["latent_vectors"]
            conditional_input_batch_g = feeder_outputs_for_g["conditional_vectors"]
            context_input_batch_g = feeder_outputs_for_g["context_vectors"]
            
            gan_inputs = [latent_input_batch_g, conditional_input_batch_g, context_input_batch_g]
            g_loss = self.gan.train_on_batch(gan_inputs, valid) # Generator tries to make discriminator output "valid"

            # Store losses
            d_losses_history.append(d_loss)
            g_losses_history.append(g_loss) # g_loss from gan.train_on_batch is a single value (or list if multiple losses)
            d_accs_history.append(d_acc)

            epoch_duration = time.time() - start_time_epoch

            # Print the progress
            print(f"Epoch {epoch+1}/{gan_epochs} [{epoch_duration:.2f}s] - D_loss: {d_loss:.4f}, D_acc: {d_acc:.4f}, G_loss: {g_loss:.4f} (LR G: {current_lr_g:.1e}, LR D: {current_lr_d:.1e})")

            # Manual ReduceLROnPlateau
            if self.params["enable_reduce_lr_on_plateau"]:
                metric_to_monitor_lr = g_loss if self.params["lr_monitor_metric"] == "g_loss" else d_loss # Adjust as needed
                if metric_to_monitor_lr < (self.best_lr_metric - self.params["lr_min_delta"]):
                    self.best_lr_metric = metric_to_monitor_lr
                    self.lr_patience_counter = 0
                else:
                    self.lr_patience_counter += 1

                if self.lr_patience_counter >= self.params["lr_patience"]:
                    new_lr_g = max(current_lr_g * self.params["lr_reduction_factor"], self.params["min_lr_g"])
                    new_lr_d = max(current_lr_d * self.params["lr_reduction_factor"], self.params["min_lr_d"])
                    if new_lr_g < current_lr_g :
                        self.g_optimizer.learning_rate.assign(new_lr_g)
                        current_lr_g = new_lr_g
                        logger.info(f"Reduced generator LR to {current_lr_g:.1e}")
                    if new_lr_d < current_lr_d:
                        self.d_optimizer.learning_rate.assign(new_lr_d)
                        current_lr_d = new_lr_d
                        logger.info(f"Reduced discriminator LR to {current_lr_d:.1e}")
                    self.lr_patience_counter = 0 # Reset counter after reduction
                print(f"  ReduceLROnPlateau: Counter {self.lr_patience_counter}/{self.params['lr_patience']}, Best Metric: {self.best_lr_metric:.4f}")


            # Manual EarlyStopping
            if self.params["enable_early_stopping"]:
                metric_to_monitor_es = g_loss if self.params["es_monitor_metric"] == "g_loss" else d_loss # Adjust
                if metric_to_monitor_es < (self.best_es_metric - self.params["es_min_delta"]):
                    self.best_es_metric = metric_to_monitor_es
                    self.es_patience_counter = 0
                else:
                    self.es_patience_counter += 1

                if self.es_patience_counter >= self.params["es_patience"]:
                    logger.info(f"Early stopping triggered at epoch {epoch+1}.")
                    print(f"  EarlyStopping: Counter {self.es_patience_counter}/{self.params['es_patience']}, Best Metric: {self.best_es_metric:.4f}. Stopping training.")
                    break 
                print(f"  EarlyStopping: Counter {self.es_patience_counter}/{self.params['es_patience']}, Best Metric: {self.best_es_metric:.4f}")


            if epoch % gan_save_interval == 0 and epoch > 0:
                self.save_models(epoch)
                self._plot_losses(d_losses_history, g_losses_history) # Plot intermediate losses

        logger.info("GAN Training finished.")
        self.save_models(gan_epochs) # Save final models
        self._plot_losses(d_losses_history, g_losses_history) # Plot final losses

    def _calculate_technical_indicators(self, base_features_batch_np: np.ndarray) -> np.ndarray:
        """
        Calculates technical indicators on the generated base features.
        Input: base_features_batch_np (batch_size, seq_len, num_base_features)
        Output: combined_features_batch_np (batch_size, seq_len, num_features_for_discriminator)
        """
        batch_size, seq_len, num_base_feat_input = base_features_batch_np.shape

        if num_base_feat_input != self.num_base_features:
            error_msg = (
                f"Input for TI calculation has {num_base_feat_input} features, "
                f"expected {self.num_base_features} base features based on 'base_feature_names_ordered': {self.base_feature_names}."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        if not self.ti_names_to_calculate:
            if self.num_base_features == self.num_features_for_discriminator:
                logger.info("No TIs to calculate, and base features count matches discriminator's expected features count. Skipping TI calculation.")
                return base_features_batch_np
            else:
                error_msg = (
                    f"TI Calculation: Configuration Error. No TIs are specified in 'ti_names_to_calculate', "
                    f"but the number of base features ({self.num_base_features} from 'base_feature_names_ordered') "
                    f"does not match the total number of features expected by the discriminator "
                    f"({self.num_features_for_discriminator} from 'feature_names_for_discriminator_ordered'). "
                    "This implies 'feature_names_for_discriminator_ordered' is not simply 'base_feature_names_ordered' + TIs, "
                    "or there's a mismatch in counts."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

        all_combined_features_list = []

        for i in range(batch_size):
            sample_base_features_df = pd.DataFrame(base_features_batch_np[i, :, :], columns=self.base_feature_names)
            df_with_tas = sample_base_features_df.copy()

            # --- Define OHLCV column mapping ---
            # Use direct mapping if names exist in base_feature_names, otherwise fallback or log warning.
            base_cols_set = set(self.base_feature_names)
            col_map = {}

            # Define preferred exact names
            ohlc_preferred_names = {
                'open': 'OPEN', 'high': 'HIGH', 'low': 'LOW', 'close': 'CLOSE', 'volume': 'VOLUME' # Case-sensitive
            }

            # Map preferred names if they exist in base_feature_names
            for key, preferred_name in ohlc_preferred_names.items():
                if preferred_name in base_cols_set:
                    col_map[key] = preferred_name
                else:
                    col_map[key] = None # Explicitly set to None if not found

            # Fallback for CLOSE if not found by preferred name (critical for most TIs)
            if not col_map['close']:
                if 'CLOSE' in base_cols_set: col_map['close'] = 'CLOSE'
                elif 'Close' in base_cols_set: col_map['close'] = 'Close'
                elif 'close' in base_cols_set: col_map['close'] = 'close'
                elif self.base_feature_names: col_map['close'] = self.base_feature_names[0] # Fallback to first column
                else: raise ValueError("TI Calc: 'close' column cannot be determined and base_feature_names is empty.")
                logger.info(f"TI Calc: 'close' mapped to '{col_map['close']}' (fallback or specific case).")

            # Fallbacks for other OHLC if not found by preferred name (less critical, used by some TIs)
            # These will try to find any case variation or use close as a last resort.
            for ohlc_key in ['open', 'high', 'low']:
                if not col_map[ohlc_key]: # If preferred name wasn't found
                    found_alternative = False
                    for base_col_name in self.base_feature_names:
                        if base_col_name.lower() == ohlc_key:
                            col_map[ohlc_key] = base_col_name
                            found_alternative = True
                            break
                    if not found_alternative:
                        col_map[ohlc_key] = col_map['close'] # Default to 'close' if not found
                        logger.warning(f"TI Calc: '{ohlc_key}' not found. Defaulting to '{col_map['close']}'. Some TIs might be inaccurate.")
            
            if not col_map['volume']: # If preferred 'VOLUME' wasn't found
                found_vol_alt = False
                for base_col_name in self.base_feature_names:
                    if base_col_name.lower() == 'volume':
                        col_map['volume'] = base_col_name
                        found_vol_alt = True
                        break
                if not found_vol_alt:
                     # Volume is optional for many TIs, so can be None if not found.
                     # Some TIs (like OBV) will fail if it's None and they are requested.
                    col_map['volume'] = None
                    logger.info("TI Calc: 'volume' column not found. TIs requiring volume may not be calculated or may error.")
            
            logger.debug(f"TI Calc OHLCV Mapping for sample {i}: {col_map}")


            processed_indicator_calls = set() # Stores (indicator_type_normalised, params_tuple_str)

            # --- Technical Indicator Calculation ---

            # RSI (e.g., RSI_14)
            rsi_configs = set()
            for ti_name in self.ti_names_to_calculate:
                if ti_name.upper().startswith("RSI_"):
                    parts = ti_name.split('_')
                    if len(parts) == 2:
                        try: rsi_configs.add(int(parts[1]))
                        except ValueError: logger.warning(f"Could not parse RSI param from {ti_name}")
            for length in rsi_configs:
                call_key = ('rsi', str((length,)))
                if call_key not in processed_indicator_calls:
                    try:
                        df_with_tas.ta.rsi(close=df_with_tas[col_map['close']], length=length, append=True)
                        processed_indicator_calls.add(call_key)
                        logger.debug(f"Calculated RSI_{length}")
                    except Exception as e: logger.warning(f"Error calculating RSI_{length}: {e}")

            # EMA (e.g., EMA_14)
            ema_configs = set()
            for ti_name in self.ti_names_to_calculate:
                if ti_name.upper().startswith("EMA_"):
                    parts = ti_name.split('_')
                    if len(parts) == 2:
                        try: ema_configs.add(int(parts[1]))
                        except ValueError: logger.warning(f"Could not parse EMA param from {ti_name}")
            for length in ema_configs:
                call_key = ('ema', str((length,)))
                if call_key not in processed_indicator_calls:
                    try:
                        df_with_tas.ta.ema(close=df_with_tas[col_map['close']], length=length, append=True)
                        processed_indicator_calls.add(call_key)
                        logger.debug(f"Calculated EMA_{length}")
                    except Exception as e: logger.warning(f"Error calculating EMA_{length}: {e}")
                
            # MACD (e.g., MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9)
            macd_configs = set() # Stores tuples of (fast, slow, signal)
            for ti_name in self.ti_names_to_calculate:
                if ti_name.upper().startswith("MACD"): # Catches MACD_, MACDh_, MACDs_
                    parts = ti_name.split('_')
                    # Expect MACD_F_S_SIG, MACDh_F_S_SIG, MACDs_F_S_SIG
                    if len(parts) == 4:
                        try: macd_configs.add((int(parts[1]), int(parts[2]), int(parts[3])))
                        except ValueError: logger.warning(f"Could not parse MACD params from {ti_name}")
            for f,s,sig in macd_configs:
                call_key = ('macd', str((f,s,sig)))
                if call_key not in processed_indicator_calls:
                    try:
                        df_with_tas.ta.macd(close=df_with_tas[col_map['close']], fast=f, slow=s, signal=sig, append=True)
                        processed_indicator_calls.add(call_key) # Marks this combination as processed
                        logger.debug(f"Calculated MACD family for {f}_{s}_{sig}")
                    except Exception as e: logger.warning(f"Error calculating MACD({f},{s},{sig}): {e}")

            # Stochastic Oscillator (e.g., STOCHk_14_3_3, STOCHd_14_3_3)
            stoch_configs = set() # Stores (k, d, smooth_k)
            for ti_name in self.ti_names_to_calculate:
                if ti_name.upper().startswith("STOCH"):
                    parts = ti_name.split('_') # STOCHk_K_D_SmoothK or STOCHd_K_D_SmoothK
                    if len(parts) == 4:
                        try: stoch_configs.add((int(parts[1]), int(parts[2]), int(parts[3])))
                        except ValueError: logger.warning(f"Could not parse STOCH params from {ti_name}")
            for k,d,smooth_k in stoch_configs: # k,d,smooth_k from pandas-ta are k,d,smooth_k
                call_key = ('stoch', str((k,d,smooth_k)))
                if call_key not in processed_indicator_calls:
                    if col_map['high'] and col_map['low'] and col_map['close']:
                        try:
                            df_with_tas.ta.stoch(high=df_with_tas[col_map['high']], low=df_with_tas[col_map['low']], close=df_with_tas[col_map['close']], k=k, d=d, smooth_k=smooth_k, append=True)
                            processed_indicator_calls.add(call_key)
                            logger.debug(f"Calculated STOCH family for {k}_{d}_{smooth_k}")
                        except Exception as e: logger.warning(f"Error calculating STOCH({k},{d},{smooth_k}): {e}")
                    else: logger.warning(f"Skipping STOCH({k},{d},{smooth_k}) due to missing HLC columns.")

            # ADX (e.g., ADX_14, DMP_14, DMN_14)
            adx_configs = set() # Stores (length)
            for ti_name in self.ti_names_to_calculate:
                if ti_name.upper().startswith("ADX_") or ti_name.upper().startswith("DMP_") or ti_name.upper().startswith("DMN_"):
                    parts = ti_name.split('_')
                    if len(parts) == 2: # ADX_L, DMP_L, DMN_L
                        try: adx_configs.add(int(parts[1]))
                        except ValueError: logger.warning(f"Could not parse ADX/DMP/DMN param from {ti_name}")
            for length in adx_configs:
                call_key = ('adx', str((length,)))
                if call_key not in processed_indicator_calls:
                    if col_map['high'] and col_map['low'] and col_map['close']:
                        try:
                            df_with_tas.ta.adx(high=df_with_tas[col_map['high']], low=df_with_tas[col_map['low']], close=df_with_tas[col_map['close']], length=length, append=True)
                            processed_indicator_calls.add(call_key)
                            logger.debug(f"Calculated ADX family for {length}")
                        except Exception as e: logger.warning(f"Error calculating ADX({length}): {e}")
                    else: logger.warning(f"Skipping ADX({length}) due to missing HLC columns.")
            
            # ATR (e.g., ATRr_14) - pandas-ta typically generates ATRr_length or ATR_length
            atr_configs = set() # Stores (length)
            for ti_name in self.ti_names_to_calculate:
                # pandas-ta might produce ATRr_L or ATR_L. Config uses ATRr_L.
                if ti_name.upper().startswith("ATRR_") or ti_name.upper().startswith("ATR_"):
                    parts = ti_name.split('_')
                    if len(parts) == 2:
                        try: atr_configs.add(int(parts[1]))
                        except ValueError: logger.warning(f"Could not parse ATR param from {ti_name}")
            for length in atr_configs:
                call_key = ('atr', str((length,)))
                if call_key not in processed_indicator_calls:
                    if col_map['high'] and col_map['low'] and col_map['close']:
                        try:
                            # mamode="rma" is often default for "true range" ATR.
                            # pandas-ta default is mamode="sma". ATRr uses rma.
                            # If config implies "true range" (ATRr), use mamode="rma"
                            # For simplicity, we'll assume if "ATRr" is in config, user wants rma.
                            # The column name from pandas-ta will be ATRr_LENGTH if mamode='rma'
                            df_with_tas.ta.atr(high=df_with_tas[col_map['high']], low=df_with_tas[col_map['low']], close=df_with_tas[col_map['close']], length=length, mamode="rma", append=True)
                            processed_indicator_calls.add(call_key)
                            logger.debug(f"Calculated ATRr_{length}")
                        except Exception as e: logger.warning(f"Error calculating ATRr_{length}: {e}")
                    else: logger.warning(f"Skipping ATRr_{length} due to missing HLC columns.")

            # CCI (e.g., CCI_14_0.015)
            cci_configs = set() # Stores (length, constant)
            for ti_name in self.ti_names_to_calculate:
                if ti_name.upper().startswith("CCI_"):
                    parts = ti_name.split('_') # CCI_L_C
                    if len(parts) == 3:
                        try: cci_configs.add((int(parts[1]), float(parts[2])))
                        except ValueError: logger.warning(f"Could not parse CCI params from {ti_name}")
            for length, constant in cci_configs:
                call_key = ('cci', str((length, constant)))
                if call_key not in processed_indicator_calls:
                    if col_map['high'] and col_map['low'] and col_map['close']:
                        try:
                            df_with_tas.ta.cci(high=df_with_tas[col_map['high']], low=df_with_tas[col_map['low']], close=df_with_tas[col_map['close']], length=length, c=constant, append=True)
                            processed_indicator_calls.add(call_key)
                            logger.debug(f"Calculated CCI_{length}_{constant}")
                        except Exception as e: logger.warning(f"Error calculating CCI({length},{constant}): {e}")
                    else: logger.warning(f"Skipping CCI({length},{constant}) due to missing HLC columns.")

            # Williams %R (e.g., WILLR_14)
            willr_configs = set() # Stores (length)
            for ti_name in self.ti_names_to_calculate:
                if ti_name.upper().startswith("WILLR_"):
                    parts = ti_name.split('_')
                    if len(parts) == 2:
                        try: willr_configs.add(int(parts[1]))
                        except ValueError: logger.warning(f"Could not parse WILLR param from {ti_name}")
            for length in willr_configs:
                call_key = ('willr', str((length,)))
                if call_key not in processed_indicator_calls:
                    if col_map['high'] and col_map['low'] and col_map['close']:
                        try:
                            df_with_tas.ta.willr(high=df_with_tas[col_map['high']], low=df_with_tas[col_map['low']], close=df_with_tas[col_map['close']], length=length, append=True)
                            processed_indicator_calls.add(call_key)
                            logger.debug(f"Calculated WILLR_{length}")
                        except Exception as e: logger.warning(f"Error calculating WILLR_{length}: {e}")
                    else: logger.warning(f"Skipping WILLR_{length} due to missing HLC columns.")

            # Momentum (e.g., MOM_14)
            mom_configs = set() # Stores (length)
            for ti_name in self.ti_names_to_calculate:
                if ti_name.upper().startswith("MOM_"):
                    parts = ti_name.split('_')
                    if len(parts) == 2:
                        try: mom_configs.add(int(parts[1]))
                        except ValueError: logger.warning(f"Could not parse MOM param from {ti_name}")
            for length in mom_configs:
                call_key = ('mom', str((length,)))
                if call_key not in processed_indicator_calls:
                    try:
                        df_with_tas.ta.mom(close=df_with_tas[col_map['close']], length=length, append=True)
                        processed_indicator_calls.add(call_key)
                        logger.debug(f"Calculated MOM_{length}")
                    except Exception as e: logger.warning(f"Error calculating MOM_{length}: {e}")

            # ROC (Rate of Change) (e.g., ROC_14)
            roc_configs = set() # Stores (length)
            for ti_name in self.ti_names_to_calculate:
                if ti_name.upper().startswith("ROC_"): # ROC, not ROC_
                    parts = ti_name.split('_')
                    if len(parts) == 2: # ROC_L
                        try: roc_configs.add(int(parts[1]))
                        except ValueError: logger.warning(f"Could not parse ROC param from {ti_name}")
            for length in roc_configs:
                call_key = ('roc', str((length,)))
                if call_key not in processed_indicator_calls:
                    try:
                        df_with_tas.ta.roc(close=df_with_tas[col_map['close']], length=length, append=True)
                        processed_indicator_calls.add(call_key)
                        logger.debug(f"Calculated ROC_{length}")
                    except Exception as e: logger.warning(f"Error calculating ROC_{length}: {e}")

            # Bollinger Bands (e.g., BBL_20_2.0, BBM_20_2.0, BBU_20_2.0, BBB_20_2.0, BBP_20_2.0)
            bbands_configs = set() # Stores (length, std_dev)
            for ti_name in self.ti_names_to_calculate:
                if ti_name.upper().startswith("BBL_") or ti_name.upper().startswith("BBM_") or \
                   ti_name.upper().startswith("BBU_") or ti_name.upper().startswith("BBB_") or \
                   ti_name.upper().startswith("BBP_"):
                    parts = ti_name.split('_') # e.g., BBL_L_STD
                    if len(parts) == 3:
                        try: bbands_configs.add((int(parts[1]), float(parts[2])))
                        except ValueError: logger.warning(f"Could not parse BBANDS params from {ti_name}")
            for length, std in bbands_configs:
                call_key = ('bbands', str((length, std)))
                if call_key not in processed_indicator_calls:
                    try:
                        df_with_tas.ta.bbands(close=df_with_tas[col_map['close']], length=length, std=std, append=True)
                        processed_indicator_calls.add(call_key)
                        logger.debug(f"Calculated BBANDS family for {length}_{std}")
                    except Exception as e: logger.warning(f"Error calculating BBANDS({length},{std}): {e}")
            
            # --- End Technical Indicator Calculation ---

            # Ensure all columns required by discriminator are present, fill NaNs, and order correctly.
            # Verify that pandas_ta generated the columns with the exact names expected in self.discriminator_feature_names
            missing_cols = [col for col in self.discriminator_feature_names if col not in df_with_tas.columns]
            if missing_cols:
                logger.warning(f"Sample {i}: After TA calculation, the following expected columns are MISSING: {missing_cols}. They will be added with NaNs (then 0). This might indicate a mismatch between 'feature_names_for_discriminator_ordered' and pandas_ta output names.")
            
            # Reindex to ensure correct order and presence of all expected columns (base + TIs)
            # Columns not generated by pandas_ta (or base features if they were somehow dropped) will be added as NaN.
            df_final_sample = df_with_tas.reindex(columns=self.discriminator_feature_names)
            
            # Fill NaNs - common at the start of series due to TA lookback periods.
            # Using 0 for now. Consider ffill() then bfill() then 0, or other strategies.
            df_final_sample = df_final_sample.fillna(0) 

            all_combined_features_list.append(df_final_sample.to_numpy())

        combined_batch_np = np.stack(all_combined_features_list, axis=0)
        
        if combined_batch_np.shape[-1] != self.num_features_for_discriminator:
            error_msg = (
                f"Output of TI calculation has {combined_batch_np.shape[-1]} features, "
                f"but discriminator expects {self.num_features_for_discriminator}. "
                f"Expected feature names: {self.discriminator_feature_names}. "
                f"Resulting columns: {list(df_final_sample.columns) if 'df_final_sample' in locals() else 'Error before final df construction'}."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        return combined_batch_np

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
            # Change .h5 to .keras
            g_path = os.path.join(self.gan_model_dir, f"generator_epoch_{epoch}.keras")
            self.generator.save(g_path)
            logger.info(f"Saved generator model to {g_path}")
        if self.discriminator:
            # Change .h5 to .keras
            d_path = os.path.join(self.gan_model_dir, f"discriminator_epoch_{epoch}.keras")
            self.discriminator.save(d_path)
            logger.info(f"Saved discriminator model to {d_path}")
        # Also, if you save the combined GAN model, update its extension too.
        # Example if you were to save self.gan:
        # if self.gan:
        #     gan_path = os.path.join(self.gan_model_dir, f"gan_epoch_{epoch}.keras")
        #     self.gan.save(gan_path)
        #     logger.info(f"Saved GAN model to {gan_path}")

    def get_generator(self) -> Optional[Model]: # Corrected Keras Model type
        """Returns the trained generator model."""
        return self.generator

    def get_discriminator(self) -> Optional[Model]: # Corrected Keras Model type
        """Returns the trained discriminator model."""
        return self.discriminator
