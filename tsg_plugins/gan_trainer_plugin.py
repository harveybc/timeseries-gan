# optimizer/plugins/gan_plugin.py

import copy  # For deep-copying configuration dicts
import logging  # Standard logging module
import random  # Random number generation
import time  # Timing execution
from typing import Any, Dict, List, Tuple, Union, Optional # Added Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam # type: ignore
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
        "latent_dim": 32, # Example, should align with feeder/generator
        "seq_len": 18,    # Example, should align with feeder/generator
        "n_features": 5,  # Example, should align with data
        "gan_model_dir": "models/gan_trained", # Directory to save GAN models
        # Add other GAN specific parameters here (e.g., discriminator architecture details)
        "discriminator_lstm_units": 64,
        "discriminator_dense_units": 128,
    }

    def __init__(self, config: Dict[str, Any], generator_plugin_instance: Optional[Any] = None, feeder_plugin_instance: Optional[Any] = None, preprocessor_plugin_instance: Optional[Any] = None):
        logger.info("Initializing GANTrainerPlugin.")
        self.config = copy.deepcopy(config)
        self.params = {**self.plugin_params, **self.config} # Merge config into params

        self.generator_plugin = generator_plugin_instance
        self.feeder_plugin = feeder_plugin_instance
        self.preprocessor_plugin = preprocessor_plugin_instance
        
        self.latent_dim = self.params.get("latent_dim", 32)
        # If latent_shape is [seq_len, features], take seq_len from there for discriminator input if needed
        # For now, assuming seq_len and n_features are directly available or inferred for discriminator build
        self.seq_len = self.params.get("seq_len", self.params.get("latent_shape", [18, 32])[0]) 
        self.n_features = self.params.get("n_features", self.params.get("generator_decoder_output_feature_names", []))
        if isinstance(self.n_features, list):
            self.n_features = len(self.n_features) # If it's a list of names, get the count

        if self.generator_plugin and hasattr(self.generator_plugin, 'model'):
            self.generator = self.generator_plugin.model
            logger.info("Using generator model from provided generator_plugin instance.")
        else:
            logger.warning("Generator plugin instance or its model not provided to GANTrainerPlugin. Generator will be None.")
            self.generator = None # Or load one based on config if GANTrainer is independent

        self.discriminator = self._build_discriminator()
        self.gan = self._build_gan()

        self.generator_optimizer = Adam(learning_rate=self.params["generator_lr"], beta_1=self.params["generator_beta1"])
        self.discriminator_optimizer = Adam(learning_rate=self.params["discriminator_lr"], beta_1=self.params["discriminator_beta1"])
        
        # Ensure model save directory exists
        self.gan_model_dir = self.params.get("gan_model_dir", "models/gan_trained")
        os.makedirs(self.gan_model_dir, exist_ok=True)
        logger.info(f"GAN models will be saved in: {os.path.abspath(self.gan_model_dir)}")


    def set_params(self, **params: Any) -> None:
        """Allows updating configuration parameters after initialization."""
        logger.info(f"GANTrainerPlugin updating parameters: {params}")
        self.config.update(params)
        self.params = {**self.plugin_params, **self.config} # Re-merge with new params
        # Potentially re-initialize parts of the plugin if critical params change
        self.latent_dim = self.params.get("latent_dim", self.latent_dim)
        self.seq_len = self.params.get("seq_len", self.params.get("latent_shape", [self.seq_len, self.latent_dim])[0])
        self.n_features = self.params.get("n_features", self.params.get("generator_decoder_output_feature_names", []))
        if isinstance(self.n_features, list):
            self.n_features = len(self.n_features)

        # Re-initialize optimizers if learning rates change
        self.generator_optimizer = Adam(learning_rate=self.params["generator_lr"], beta_1=self.params["generator_beta1"])
        self.discriminator_optimizer = Adam(learning_rate=self.params["discriminator_lr"], beta_1=self.params["discriminator_beta1"])
        self.gan_model_dir = self.params.get("gan_model_dir", self.gan_model_dir)
        os.makedirs(self.gan_model_dir, exist_ok=True)


    def _build_discriminator(self) -> tf.keras.Model:
        """Builds and compiles the Discriminator model."""
        logger.info("Building Discriminator model...")
        # This should ideally call a function from discriminator_model.py
        # For now, a placeholder:
        # from .discriminator_model import build_discriminator # Assuming this exists
        # model = build_discriminator(input_shape=(self.seq_len, self.n_features), config=self.params)
        
        # Placeholder simple LSTM discriminator
        input_shape = (self.seq_len, self.n_features) # Example: (timesteps, features_per_timestep)
        logger.info(f"Discriminator input shape: {input_shape}")

        model = tf.keras.Sequential(name="discriminator")
        model.add(tf.keras.layers.LSTM(self.params.get("discriminator_lstm_units", 64), input_shape=input_shape, return_sequences=False))
        model.add(tf.keras.layers.Dense(self.params.get("discriminator_dense_units", 128), activation='relu'))
        model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        
        model.compile(loss='binary_crossentropy', optimizer=self.discriminator_optimizer, metrics=['accuracy'])
        logger.info("Discriminator model built and compiled.")
        model.summary(print_fn=logger.info)
        return model

    def _build_gan(self) -> tf.keras.Model:
        """Builds the combined GAN model (Generator + Discriminator)."""
        if not self.generator or not self.discriminator:
            logger.error("Generator or Discriminator not initialized. Cannot build GAN.")
            raise ValueError("Generator or Discriminator not initialized.")
            
        logger.info("Building GAN model (Generator + Discriminator)...")
        self.discriminator.trainable = False  # Discriminator is not trained during GAN training phase
        
        # Assuming generator takes latent vector as input.
        # The input to the GAN model will be the input to the generator.
        # This needs to match how the FeederPlugin provides latent vectors.
        # Example: if latent_shape is [seq_len, latent_dim]
        generator_input_shape = tuple(self.params.get("latent_shape", (self.seq_len, self.latent_dim)))
        
        gan_input = tf.keras.Input(shape=generator_input_shape, name="gan_input_latent_vector")
        generated_data = self.generator(gan_input) # Output of generator
        gan_output = self.discriminator(generated_data)  # Pass generated data to discriminator
        
        gan = tf.keras.Model(gan_input, gan_output, name="gan_combined")
        gan.compile(loss='binary_crossentropy', optimizer=self.generator_optimizer)
        logger.info("GAN model built and compiled.")
        gan.summary(print_fn=logger.info)
        return gan

    def train(self, x_train_file: Optional[str] = None, data: Optional[np.ndarray] = None) -> None:
        """
        Main training loop for the GAN.
        Either x_train_file or data (as numpy array) should be provided.
        """
        if self.generator is None or self.discriminator is None or self.gan is None:
            logger.error("Models not built. Cannot start training.")
            return

        if data is None and x_train_file is not None:
            logger.info(f"Loading training data from: {x_train_file}")
            if not self.preprocessor_plugin:
                logger.error("Preprocessor plugin not available to process the training data file.")
                raise ValueError("Preprocessor plugin required when x_train_file is provided.")
            
            # Use preprocessor to load and process data
            # The preprocessor should return data in the shape expected by the GAN (e.g., (n_samples, seq_len, n_features))
            # This is a simplified call; preprocessor might need more specific config.
            # We need to pass a config to run_preprocessing that specifies it's for GAN training data.
            preprocess_config = self.config.copy()
            # Potentially override some preprocessor settings if needed for GAN data
            # preprocess_config['target_column'] = None # Example if GAN doesn't need a specific target like supervised models
            
            processed_data_dict = self.preprocessor_plugin.run_preprocessing(config=preprocess_config)
            real_data = processed_data_dict.get("x_train") # Assuming preprocessor returns processed data under 'x_train'
            
            if real_data is None:
                logger.error(f"Preprocessor did not return 'x_train' data from {x_train_file}.")
                raise ValueError("Failed to load or process training data.")
            
            if isinstance(real_data, pd.DataFrame):
                real_data = real_data.values
            
            # Ensure real_data is 3D (samples, seq_len, features) if seq_len > 1 for discriminator
            # Or 2D (samples, features) if seq_len = 1
            # This depends on how your preprocessor and discriminator are set up.
            # For this example, assuming preprocessor gives (samples, seq_len, features)
            # matching discriminator's input_shape.
            if real_data.ndim == 2 and self.seq_len > 1: # If preprocessor gives 2D but discriminator expects 3D
                 # This might indicate a mismatch or that preprocessor needs to window it.
                 # For now, let's assume preprocessor output matches discriminator input needs.
                 logger.warning(f"Real data is 2D but seq_len is {self.seq_len}. Ensure preprocessor handles windowing if discriminator expects 3D sequences.")
            elif real_data.ndim == 3:
                 logger.info(f"Using 3D real data of shape {real_data.shape} for training.")
            
            # If your generator (SCVAE decoder) outputs data that is then windowed by the discriminator,
            # the `real_data` here should also be windowed similarly.
            # For simplicity, let's assume `real_data` from preprocessor is already in the correct shape
            # (e.g. (num_samples, self.seq_len, self.n_features))

        elif data is not None:
            logger.info("Using provided data for training.")
            real_data = data
        else:
            logger.error("No training data provided (neither x_train_file nor data).")
            return

        if not isinstance(real_data, np.ndarray):
            real_data = np.array(real_data)

        logger.info(f"Shape of training data for GAN: {real_data.shape}")
        if real_data.shape[1] != self.seq_len or real_data.shape[2] != self.n_features:
             logger.warning(f"Training data shape {real_data.shape} might not match discriminator input ({self.seq_len}, {self.n_features}). Ensure consistency.")


        epochs = self.params["gan_epochs"]
        batch_size = self.params["gan_batch_size"]
        save_interval = self.params["gan_save_interval"]

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        # For logging losses
        d_losses, g_losses = [], []

        for epoch in range(epochs):
            start_time_epoch = time.time()
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            
            # Select a random batch of real data
            idx = np.random.randint(0, real_data.shape[0], batch_size)
            real_batch = real_data[idx]

            # Generate a batch of new data using FeederPlugin and Generator
            # The FeederPlugin should provide latent vectors compatible with the generator
            if not self.feeder_plugin:
                logger.error("Feeder plugin not available to generate latent noise for generator.")
                raise ValueError("Feeder plugin required for GAN training.")

            # Feeder generates latent noise + any conditions
            # Assuming feeder_plugin.generate() returns a dictionary or object
            # from which we can extract the latent vector for the generator.
            # The shape of this latent vector must match self.generator.input_shape.
            # For VAEGeneratorPlugin, this is typically `feeder_outputs_sequence['latent_vector']`
            # and might be 3D: (batch_size, latent_seq_len, latent_dim_features)
            
            # Simplification: Feeder generates latent vectors of shape (batch_size, latent_dim)
            # or (batch_size, latent_seq_len, latent_dim_features)
            # This needs to align with your FeederPlugin's output and Generator's input.
            # Let's assume feeder_plugin.generate gives a dict with 'latent_vector'
            feeder_output_for_gen = self.feeder_plugin.generate(n_ticks_to_generate=batch_size) 
            
            # The VAEGeneratorPlugin's generate method is complex and expects initial windows etc.
            # For GAN training, we typically feed noise directly to the generator part of the VAE.
            # So, self.generator here should be the decoder part.
            # The input to this decoder needs to be latent vectors.
            
            # Assuming feeder_output_for_gen['latent_vector'] is the correct input for self.generator
            # and has shape (batch_size, seq_len_for_latent, latent_dim_features)
            # or (batch_size, latent_dim) if generator is simpler.
            # Let's assume latent_input_for_generator is correctly shaped.
            latent_input_for_generator = feeder_output_for_gen.get('latent_vector') 
            if latent_input_for_generator is None:
                # Fallback: generate random noise if feeder doesn't provide 'latent_vector'
                # This shape must match the generator's input.
                # If generator input is (None, seq_len, latent_dim)
                gen_input_shape_from_config = tuple(self.params.get("latent_shape", (self.seq_len, self.latent_dim)))
                if len(gen_input_shape_from_config) == 2: # (seq_len, features)
                    latent_input_for_generator = np.random.normal(0, 1, (batch_size, gen_input_shape_from_config[0], gen_input_shape_from_config[1]))
                elif len(gen_input_shape_from_config) == 1: # (features_flat)
                    latent_input_for_generator = np.random.normal(0, 1, (batch_size, gen_input_shape_from_config[0]))
                else:
                    raise ValueError(f"Unsupported latent_shape for noise generation: {gen_input_shape_from_config}")

            generated_batch = self.generator.predict(latent_input_for_generator)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(real_batch, valid)
            d_loss_fake = self.discriminator.train_on_batch(generated_batch, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------
            # Generate noise for generator training (same way as above)
            feeder_output_for_gan = self.feeder_plugin.generate(n_ticks_to_generate=batch_size)
            latent_input_for_gan_train = feeder_output_for_gan.get('latent_vector')
            if latent_input_for_gan_train is None:
                gen_input_shape_from_config = tuple(self.params.get("latent_shape", (self.seq_len, self.latent_dim)))
                if len(gen_input_shape_from_config) == 2:
                    latent_input_for_gan_train = np.random.normal(0, 1, (batch_size, gen_input_shape_from_config[0], gen_input_shape_from_config[1]))
                elif len(gen_input_shape_from_config) == 1:
                    latent_input_for_gan_train = np.random.normal(0, 1, (batch_size, gen_input_shape_from_config[0]))

            # Train the generator (to fool the discriminator)
            g_loss = self.gan.train_on_batch(latent_input_for_gan_train, valid)
            
            epoch_time = time.time() - start_time_epoch
            d_losses.append(d_loss[0])
            g_losses.append(g_loss)

            if epoch % 100 == 0 or epoch == epochs -1 : # Log every 100 epochs
                logger.info(f"{epoch}/{epochs} [D loss: {d_loss[0]:.4f}, acc.: {100*d_loss[1]:.2f}%] [G loss: {g_loss:.4f}] time: {epoch_time:.2f}s")

            if epoch % save_interval == 0 and epoch > 0:
                self.save_models(epoch)
        
        self.save_models(epochs) # Save final models
        logger.info("GAN Training completed.")
        # Plot losses
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
