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
        
        # Get shapes for conditional and context inputs from feeder or generator params
        # These are needed for building the GAN model input layers
        # Assuming feeder_plugin is available during __init__ or these params are in main config
        if self.feeder_plugin and hasattr(self.feeder_plugin, 'params'):
            num_date_feats = len(self.feeder_plugin.params.get("date_feature_names_for_conditioning", [])) * 2 # sin/cos
            num_fund_feats = len(self.feeder_plugin.params.get("fundamental_feature_names_for_conditioning", []))
            self.conditional_dim = num_date_feats + num_fund_feats
            self.context_dim_gan = self.feeder_plugin.params.get("context_vector_dim", 64)
        else: # Fallback if feeder_plugin not yet fully available, try from self.params (main config)
            num_date_feats = len(self.params.get("feeder_date_feature_names_for_conditioning", [])) * 2
            num_fund_feats = len(self.params.get("feeder_fundamental_feature_names_for_conditioning", []))
            self.conditional_dim = num_date_feats + num_fund_feats
            self.context_dim_gan = self.params.get("feeder_context_vector_dim", 64)
        logger.info(f"GANTrainer determined conditional_dim: {self.conditional_dim}, context_dim_gan: {self.context_dim_gan}")


        if self.generator_plugin and hasattr(self.generator_plugin, 'model'):
            self.generator = self.generator_plugin.model # This is the VAE decoder
            # --- ADDED: Freeze the pre-trained VAE decoder (generator) ---
            if self.generator:
                self.generator.trainable = False
                logger.info("GANTrainerPlugin: Set VAE decoder (generator) to trainable=False for GAN training.")
            # --- END ADDED ---
            logger.info("Using generator model from provided generator_plugin instance.")
        else:
            logger.warning("Generator plugin instance or its model not provided to GANTrainerPlugin. Generator will be None.")
            self.generator = None # Or load one based on config if GANTrainer is independent

        # --- MOVED OPTIMIZER INITIALIZATION UP ---
        self.generator_optimizer = Adam(learning_rate=self.params["generator_lr"], beta_1=self.params["generator_beta1"])
        self.discriminator_optimizer = Adam(learning_rate=self.params["discriminator_lr"], beta_1=self.params["discriminator_beta1"])
        # --- END MOVED ---

        self.discriminator = self._build_discriminator()
        self.gan = self._build_gan()

        # self.generator_optimizer = Adam(learning_rate=self.params["generator_lr"], beta_1=self.params["generator_beta1"]) # REMOVED FROM HERE
        # self.discriminator_optimizer = Adam(learning_rate=self.params["discriminator_lr"], beta_1=self.params["discriminator_beta1"]) # REMOVED FROM HERE
        
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
        
        # Define input layers for the GAN based on what the VAE Decoder (self.generator) expects
        # These shapes should align with what FeederPlugin provides.
        
        # 1. Latent vector input
        latent_input_shape = tuple(self.params.get("latent_shape", (self.seq_len, self.latent_dim)))
        gan_latent_input = tf.keras.Input(shape=latent_input_shape, name="gan_input_latent_vector")
        
        # 2. Conditional data input
        # self.conditional_dim was calculated in __init__
        gan_conditional_input = tf.keras.Input(shape=(self.conditional_dim,), name="gan_input_conditional_data")
        
        # 3. Context vector input
        # self.context_dim_gan was calculated in __init__
        gan_context_input = tf.keras.Input(shape=(self.context_dim_gan,), name="gan_input_context_vector")

        # The order of inputs to self.generator matters and must match its internal definition.
        # We need to get the expected input names from the generator model itself.
        # These names are configured in the GeneratorPlugin (e.g., "generator_decoder_input_name_latent")
        
        if not hasattr(self.generator, 'inputs') or not self.generator.inputs:
            logger.error("Cannot determine generator's input signature. self.generator.inputs is not available.")
            raise ValueError("Generator model input signature unknown.")

        generator_input_names_from_model = [inp.name.split(':')[0] for inp in self.generator.inputs]
        logger.info(f"Generator model expects inputs named: {generator_input_names_from_model}")

        # Get the configured names for each type of input from self.params (which includes generator_plugin.params)
        cfg_latent_name = self.params.get("generator_decoder_input_name_latent")
        cfg_context_name = self.params.get("generator_decoder_input_name_context")
        cfg_conditional_name = self.params.get("generator_decoder_input_name_conditions")
        # cfg_window_name = self.params.get("generator_decoder_input_name_window") # Not used for GAN direct generation from Z

        if not cfg_latent_name or not cfg_context_name or not cfg_conditional_name:
            logger.error("One or more generator input names (latent, context, conditions) are not configured in params.")
            raise ValueError("Missing generator input name configuration.")

        # Map our GAN input layers to the generator's expected input names
        input_map = {
            cfg_latent_name: gan_latent_input,
            cfg_context_name: gan_context_input,
            cfg_conditional_name: gan_conditional_input,
        }

        # Order the GAN input layers according to the generator model's input order
        gan_input_layers_ordered = []
        for model_input_name in generator_input_names_from_model:
            found_match = False
            for cfg_name, gan_layer in input_map.items():
                # Check if the model_input_name (e.g., "decoder_input_z_seq_2") starts with the configured base name (e.g., "decoder_input_z_seq")
                # This handles cases where Keras might append suffixes like "_1", "_2" to layer names if models are rebuilt.
                if model_input_name.startswith(cfg_name):
                    gan_input_layers_ordered.append(gan_layer)
                    found_match = True
                    break
            if not found_match:
                # This check is important. If the generator expects an input (e.g. window) that we are not providing for GAN mode,
                # it will fail. The current VAE-GAN setup assumes direct generation from Z, C, H.
                # If the generator *always* requires the window input, this GAN model structure is incompatible.
                # However, typically for GANs, we bypass window inputs if the generator can operate from Z,C,H alone.
                logger.warning(f"Generator model input '{model_input_name}' does not match any configured GAN inputs (latent, context, conditions). This input will not be provided to the generator in the GAN model. This might be okay if it's an optional input or not used in this generation path.")
                # If it's a critical missing input, this will error out when self.generator is called.

        if len(gan_input_layers_ordered) != len(self.generator.inputs):
             logger.error(f"Mismatch between ordered GAN inputs ({len(gan_input_layers_ordered)}) and generator's expected inputs ({len(self.generator.inputs)}). This indicates a problem mapping configured input names to the model's actual input signature.")
             logger.error(f"Ordered GAN inputs based on mapping: {[layer.name for layer in gan_input_layers_ordered]}")
             logger.error(f"Generator's expected inputs by name: {generator_input_names_from_model}")
             logger.error(f"Configured input names: Latent='{cfg_latent_name}', Context='{cfg_context_name}', Conditions='{cfg_conditional_name}'")
             # This often happens if the generator model expects an input (e.g., 'input_x_window') that we are not providing in this GAN setup.
             # For a VAE-GAN, we typically only feed Z, C, H to the decoder.
             # If the decoder *requires* the window, then this GAN architecture is problematic for that specific decoder.
             # For now, we proceed, and Keras will raise an error if the call to self.generator is invalid.
             # A more robust solution would be to ensure the generator can operate with Z,C,H or to provide dummy/appropriate window data.
             # Given the error "expects 3 input(s)", it seems it wants Z, C, H.

        # Define the list of inputs for the tf.keras.Model
        actual_gan_model_inputs = [gan_latent_input, gan_conditional_input, gan_context_input]

        generated_data = self.generator(gan_input_layers_ordered) # Pass the ordered list of KerasTensors
        gan_output = self.discriminator(generated_data)
        
        gan = tf.keras.Model(inputs=actual_gan_model_inputs, outputs=gan_output, name="gan_combined")
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

            if not self.feeder_plugin:
                logger.error("Feeder plugin not available to generate latent noise for generator.")
                raise ValueError("Feeder plugin required for GAN training.")

            feeder_output_for_gen = self.feeder_plugin.generate(n_ticks_to_generate=batch_size) 
            
            latent_input_for_generator = feeder_output_for_gen.get('latent_vector_batch')
            conditional_input_for_generator = feeder_output_for_gen.get('conditional_data_batch')
            context_input_for_generator = feeder_output_for_gen.get('context_h_batch')

            if latent_input_for_generator is None or conditional_input_for_generator is None or context_input_for_generator is None:
                logger.error("FeederPlugin did not return all required inputs (latent, conditional, context) for the generator.")
                # Fallback to random noise for latent if others are missing too, this will likely fail if C and H are also None.
                gen_input_shape_from_config = tuple(self.params.get("latent_shape", (self.seq_len, self.latent_dim)))
                if len(gen_input_shape_from_config) == 2: # (seq_len, features)
                    latent_input_for_generator = np.random.normal(0, 1, (batch_size, gen_input_shape_from_config[0], gen_input_shape_from_config[1]))
                elif len(gen_input_shape_from_config) == 1: # (features_flat)
                    latent_input_for_generator = np.random.normal(0, 1, (batch_size, gen_input_shape_from_config[0]))
                # Create dummy conditional and context if missing, with correct shapes
                if conditional_input_for_generator is None:
                    conditional_input_for_generator = np.zeros((batch_size, self.conditional_dim))
                if context_input_for_generator is None:
                    context_input_for_generator = np.zeros((batch_size, self.context_dim_gan))


            # Order the inputs for self.generator.predict according to its expected input signature
            # This reuses the logic from _build_gan to determine the order
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
                    if model_input_name.startswith(cfg_name):
                        generator_predict_inputs_ordered.append(data_array)
                        found_match = True
                        break
                if not found_match:
                    # This should ideally not happen if _build_gan logic is correct and generator only needs Z,C,H
                    logger.error(f"Could not find data for generator input '{model_input_name}' during predict step.")
                    # Handle error or provide a dummy input if absolutely necessary and safe
                    # For now, this will likely lead to an error in predict if an input is truly missing.

            if len(generator_predict_inputs_ordered) != len(self.generator.inputs):
                 logger.error(f"Predict step: Mismatch between ordered inputs for generator ({len(generator_predict_inputs_ordered)}) and generator's expected inputs ({len(self.generator.inputs)}).")
                 # This is a critical error if it occurs.

            generated_batch = self.generator.predict(generator_predict_inputs_ordered)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(real_batch, valid)
            d_loss_fake = self.discriminator.train_on_batch(generated_batch, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------
            feeder_output_for_gan_train = self.feeder_plugin.generate(n_ticks_to_generate=batch_size)
            latent_input_for_gan_train = feeder_output_for_gan_train.get('latent_vector_batch')
            conditional_input_for_gan_train = feeder_output_for_gan_train.get('conditional_data_batch')
            context_input_for_gan_train = feeder_output_for_gan_train.get('context_h_batch')
            
            if latent_input_for_gan_train is None or conditional_input_for_gan_train is None or context_input_for_gan_train is None:
                logger.error("FeederPlugin did not return all required inputs (latent, conditional, context) for GAN training step.")
                # Fallback similar to above
                gen_input_shape_from_config = tuple(self.params.get("latent_shape", (self.seq_len, self.latent_dim)))
                if len(gen_input_shape_from_config) == 2:
                    latent_input_for_gan_train = np.random.normal(0, 1, (batch_size, gen_input_shape_from_config[0], gen_input_shape_from_config[1]))
                elif len(gen_input_shape_from_config) == 1:
                     latent_input_for_gan_train = np.random.normal(0, 1, (batch_size, gen_input_shape_from_config[0]))
                if conditional_input_for_gan_train is None:
                    conditional_input_for_gan_train = np.zeros((batch_size, self.conditional_dim))
                if context_input_for_gan_train is None:
                    context_input_for_gan_train = np.zeros((batch_size, self.context_dim_gan))

            # Inputs for self.gan.train_on_batch must match the order of self.gan.inputs
            # which are [gan_latent_input, gan_conditional_input, gan_context_input]
            gan_train_inputs_ordered = [latent_input_for_gan_train, conditional_input_for_gan_train, context_input_for_gan_train]
            
            g_loss = self.gan.train_on_batch(gan_train_inputs_ordered, valid)
            
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
