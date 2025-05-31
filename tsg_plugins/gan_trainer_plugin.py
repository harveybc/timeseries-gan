# optimizer/plugins/gan_plugin.py

"""
Optimizer Plugin using DEAP for Synthetic Data Generator.

This plugin employs a genetic algorithm to tune key hyperparameters
for the synthetic-data generation pipeline, optimizing downstream
predictor performance.

Plugin Parameters
-----------------
- population_size (int): Number of individuals in each generation.
- n_generations (int): Number of evolutionary generations to run.
- cxpb (float): Crossover probability.
- mutpb (float): Mutation probability.
- hyperparameter_bounds (dict): Bounds for each hyperparameter to optimize.

Methods
-------
- set_params(**kwargs)
- optimize(feeder_plugin, generator_plugin, evaluator_plugin, config)
"""

import copy  # For deep-copying configuration dicts
import logging  # Standard logging module
import random  # Random number generation
import time  # Timing execution
from typing import Any, Dict, List, Tuple, Union

from deap import algorithms, base, creator, tools  # DEAP components
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, RepeatVector, TimeDistributed, Bidirectional, Conv1DTranspose, AdditiveAttention, LayerNormalization, LeakyReLU, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
from tsg_plugins.plugin_api import FeederPlugin, GeneratorPlugin, TrainerPlugin # Corrected import path

# Initialize logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set default log level


class GANTrainerPlugin(TrainerPlugin):
    """
    DEAP-based optimizer plugin for synthetic data generation.

    This plugin tunes:
      - latent_dim (int)
      - mmd_lambda (float)
      - kl_beta (float)
      - batch_size (int)

    Attributes
    ----------
    params : Dict[str, Any]
        Copy of plugin_params merged with user configuration.
    """

    #: Default optimizer configuration
    plugin_params = {
        "gan_epochs": 100,             # Number of epochs for GAN training
        "gan_batch_size": 32,          # Batch size for GAN training
        "generator_lr": 0.0002,        # Learning rate for the generator
        "discriminator_lr": 0.0002,    # Learning rate for the discriminator
        "adam_beta1": 0.5,             # Adam optimizer beta1
        # "input_sequence_length": 64, # Example: User should configure this based on their data
        # "num_features": 54,          # Example: User should configure this
        "discriminator_lstm_units": 64, # Units for the LSTM layer in discriminator
        "discriminator_dense_units": 128,# Units for the Dense layer in discriminator
         "random_seed": None,          # Optional seed for reproducibility
     }
     #: Keys included in debug output
    plugin_debug_vars = ["population_size", "n_generations", "cxpb", "mutpb"]

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize optimizer plugin with default parameters.
        """
        if config is None:
            raise ValueError("Se requiere el diccionario de configuración ('config').")
        # Copia parámetros por defecto y aplica la configuración
        self.params = self.plugin_params.copy()
        self.set_params(**config)
        # Deep copy to avoid mutating the class attribute
        self.params: Dict[str, Any] = copy.deepcopy(self.plugin_params)

    def set_params(self, **kwargs: Any) -> None:
        """
        Update plugin parameters from global configuration.

        Parameters
        ----------
        **kwargs : Any
            Arbitrary keyword arguments to update internal params.
        """
        for key, value in kwargs.items():
            self.params[key] = value

    def get_debug_info(self) -> Dict[str, Any]:
        """
        Retrieve debugging information.

        Returns
        -------
        Dict[str, Any]
            Subset of params useful for debugging.
        """
        return {var: self.params.get(var) for var in self.plugin_debug_vars}

    def optimize(
        self,
        feeder_plugin: Any,
        generator_plugin: Any,
        evaluator_plugin: Any,
        preprocessor_plugin: Any,
        config: Dict[str, Any],
    ) -> Dict:
        """
        Train GAN by alternating discriminator and generator updates.
        """
        # 1) Acquire generator model
        gen_model = getattr(generator_plugin, "model", None)
        if gen_model is None:
            raise RuntimeError("GANTrainerPlugin: GeneratorPlugin must expose attribute 'model'.")
        
        # Ensure the generator's output shape is defined for the discriminator
        # Expected: (batch_size, sequence_length, num_features)
        generator_output_shape = gen_model.output_shape[1:] # Exclude batch_size
        print(f"GANTrainerPlugin: Using generator output shape for discriminator input: {generator_output_shape}")

        # 2) Build discriminator
        self.discriminator = self._build_discriminator(config, input_shape=generator_output_shape)

        # 3) Compile discriminator and adversarial model
        from tensorflow.keras.optimizers import Adam
        d_optimizer = Adam(learning_rate=self.params.get("discriminator_lr"), beta_1=self.params.get("adam_beta1"))
        g_optimizer = Adam(learning_rate=self.params.get("generator_lr"), beta_1=self.params.get("adam_beta1"))

        self.discriminator.compile(optimizer=d_optimizer, loss="binary_crossentropy", metrics=['accuracy'])
        real_input = gen_model.input
        fake_output = self.discriminator(gen_model.output)
        self.adversarial = Model(inputs=real_input, outputs=fake_output)
        # For the adversarial model, we only want to train the generator
        self.discriminator.trainable = False 
        self.adversarial.compile(optimizer=g_optimizer, loss="binary_crossentropy", metrics=['accuracy'])

        # 4) Training loop
        epochs = self.params.get("gan_epochs", 1)
        batch_size = self.params.get("gan_batch_size", 32)
        
        print(f"GANTrainerPlugin: Starting GAN training for {epochs} epochs with batch size {batch_size}.")

        for epoch in range(epochs):
            epoch_d_loss_real = []
            epoch_d_loss_fake = []
            epoch_g_loss = []
            
            for real_batch in feeder_plugin.fetch_batch():
                if real_batch.shape[0] != batch_size:
                    # If the last batch from feeder is smaller, adjust noise and fake batch size
                    current_batch_size = real_batch.shape[0]
                    if current_batch_size == 0: continue
                else:
                    current_batch_size = batch_size

                 # Generate noise and fake batch
                noise = generator_plugin.sample_noise(current_batch_size) # Use current_batch_size
                fake_batch = gen_model.predict(noise)

                # Create labels
                real_labels = np.ones((current_batch_size, 1))
                fake_labels = np.zeros((current_batch_size, 1))

                # Train discriminator
                self.discriminator.trainable = True # Ensure discriminator is trainable
                d_loss_real = self.discriminator.train_on_batch(real_batch, real_labels)
                d_loss_fake = self.discriminator.train_on_batch(fake_batch, fake_labels)
                epoch_d_loss_real.append(d_loss_real[0]) # loss value
                epoch_d_loss_fake.append(d_loss_fake[0]) # loss value

                 # Train generator via adversarial model
                # For adversarial training, discriminator weights are frozen (done during compile)
                self.discriminator.trainable = False 
                noise_for_g = generator_plugin.sample_noise(current_batch_size) # Use current_batch_size
                g_loss = self.adversarial.train_on_batch(noise_for_g, real_labels) # Generator tries to make discriminator output 1 (real)
                epoch_g_loss.append(g_loss[0]) # loss value
            
            avg_d_loss_real = np.mean(epoch_d_loss_real) if epoch_d_loss_real else 0
            avg_d_loss_fake = np.mean(epoch_d_loss_fake) if epoch_d_loss_fake else 0
            avg_g_loss = np.mean(epoch_g_loss) if epoch_g_loss else 0
            print(f"Epoch {epoch+1}/{epochs} -> D_Loss_Real: {avg_d_loss_real:.4f}, D_Loss_Fake: {avg_d_loss_fake:.4f}, G_Loss: {avg_g_loss:.4f}")

        # 5) Store trained generator
        self.trained_generator = gen_model
        print("GANTrainerPlugin: GAN training completed.")
        return {}

    def get_trained_generator(self) -> Any:
        """
        Return the GAN-trained generator model.
        """
        return getattr(self, "trained_generator", None)

    def _build_discriminator(self, config: Dict[str, Any], input_shape: Tuple) -> Any:
        """
        Build a simple discriminator model. Override for custom architectures.
        Now uses an LSTM layer suitable for sequence data.
        """
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import Dense, LSTM, Input

        # input_shape should be (sequence_length, num_features), e.g., (64, 54)
        # This is now taken from generator's output shape.
        lstm_units = self.params.get("discriminator_lstm_units", 64)
        dense_units = self.params.get("discriminator_dense_units", 128)
        
        print(f"GANTrainerPlugin: Building discriminator with input shape {input_shape}, LSTM units: {lstm_units}, Dense units: {dense_units}")

        model = Sequential([
            Input(shape=input_shape), # Use Input layer for explicit shape
            LSTM(lstm_units), # LSTM layer to process sequences
            Dense(dense_units, activation="relu"),
            Dense(1, activation="sigmoid"),
        ])
        return model

    def _build_generator(self):
        seq_len = self.config.get("seq_len")
        latent_dim = self.config.get("latent_dim")
        n_features = self.config.get("n_features")
        
        # Based on VAE Decoder architecture
        inputs = Input(shape=(latent_dim,))
        # Potentially repeat or dense layer to match BiLSTM input requirements if latent_dim is small
        x = RepeatVector(seq_len)(inputs) # Assuming latent_dim is fed to each time step of BiLSTM
                                        # This might need adjustment based on how latent_dim is used.
                                        # If latent_dim is features for a single step, then input shape might be (1, latent_dim)
                                        # and then Dense to expand to seq_len * internal_lstm_units

        # BiLSTM layer
        # The number of units in LSTM should be determined based on complexity and n_features
        # For now, let's use a value, e.g., 128 or relate it to n_features or seq_len
        lstm_units = self.config.get("generator_lstm_units", 128) 
        x = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)

        # 3 Conv1DTranspose layers
        # Kernel sizes and filters should be chosen to upscale to the desired seq_len and n_features
        # Example filter counts and kernel sizes, these need careful tuning
        filters = [64, 32, n_features] 
        kernel_sizes = [5, 5, 7] # These are examples
        strides = [1, 1, 1] # Strides might need to be > 1 if upsampling in length too, depends on BiLSTM output

        # Reshape BiLSTM output if necessary to be compatible with Conv1DTranspose
        # Conv1DTranspose expects (batch, steps, channels)
        # If BiLSTM output is (batch, seq_len, lstm_units*2), it's likely compatible.

        for i in range(3):
            x = Conv1DTranspose(filters=filters[i], 
                                kernel_size=kernel_sizes[i], 
                                strides=strides[i], # Adjust if upsampling sequence length
                                padding='same', # Or 'causal' if appropriate
                                activation='relu' if i < 2 else 'linear')(x) # Linear for final output layer
            x = LayerNormalization()(x) # Optional: add normalization

        # Additive Attention layer
        # Attention mechanism might need adjustment based on exact input/output shapes
        # Assuming x is (batch, seq_len, n_features)
        # Query for attention could be a learned variable or derived from input/state
        # For simplicity, let's assume self-attention on the output sequence
        attention_units = self.config.get("generator_attention_units", 64)
        query = Dense(attention_units, name='attention_query')(x) 
        value = Dense(attention_units, name='attention_value')(x)
        # Using x as content for self-attention style
        attention_output = AdditiveAttention(name='additive_attention')([query, value, x]) 

        # Final output layer to ensure correct shape (seq_len, n_features)
        # This might be redundant if the last Conv1DTranspose and Attention already produce this.
        # If attention_output is (batch, seq_len, attention_units), a final Dense layer is needed.
        if attention_output.shape[-1] != n_features:
             outputs = TimeDistributed(Dense(n_features, activation=self.config.get("output_activation", "sigmoid")))(attention_output)
        else:
            # Apply activation if the number of features already matches
            outputs = tf.keras.layers.Activation(self.config.get("output_activation", "sigmoid"))(attention_output)


        # Ensure output shape is (seq_len, n_features)
        # This might require a Reshape layer if the above layers don't naturally produce it.
        # For example, if output is (batch_size, seq_len, n_features)
        # and the model is used to generate one sequence at a time, this is fine.
        # If used within a TimeDistributed wrapper later, it might need adjustment.

        self.generator = Model(inputs, outputs, name="generator")
        # print("--- Generator Summary (GANTrainer) ---")
        # self.generator.summary()
        return self.generator

    def _build_models_and_compile(self):
        # Ensure generator and discriminator are built
        if self.generator is None:
            self._build_generator()
        if self.discriminator is None:
            self._build_discriminator()

        # Compile Discriminator
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=Adam(self.config.get("discriminator_lr", 0.0002), 
                                                  beta_1=self.config.get("discriminator_beta1", 0.5)),
                                   metrics=['accuracy'])

        # GAN (Combined Model)
        self.discriminator.trainable = False
        
        latent_dim = self.config.get("latent_dim")
        gan_input = Input(shape=(latent_dim,))
        generated_sequence = self.generator(gan_input)
        gan_output = self.discriminator(generated_sequence)
        
        self.gan = Model(gan_input, gan_output)
        self.gan.compile(loss='binary_crossentropy', 
                         optimizer=Adam(self.config.get("generator_lr", 0.0002), 
                                        beta_1=self.config.get("generator_beta1", 0.5)))
        
        print("--- Generator Summary (GANTrainer) ---")
        self.generator.summary()
        print("--- Discriminator Summary (GANTrainer) ---")
        self.discriminator.summary()
        print("--- GAN Summary (GANTrainer) ---")
        self.gan.summary()


    def train(self, x_train_file):
        self._build_models_and_compile() # Ensure models are built and compiled before training

        feeder_name = self.config.get("feeder_plugin", "default_feeder")
        # This part needs a proper plugin loading mechanism from main.py
        # For now, assuming FeederPlugin can be directly instantiated if it's the default
        if feeder_name == "default_feeder": # Or however your default feeder is identified
            from tsg_plugins.feeder_plugin import CSVFeederPlugin # Assuming this is the default
            feeder = CSVFeederPlugin(self.config)
        else:
            # Placeholder for dynamic plugin loading if you have multiple feeders
            raise NotImplementedError(f"Feeder plugin {feeder_name} not implemented in GANTrainerPlugin direct instantiation.")

        data = feeder.feed(x_train_file) 
        
        if data is None or data.shape[0] == 0:
            print("Error: No data loaded for GAN training. Check x_train_file and FeederPlugin.")
            return
        
        # Reshape data if it's 2D (samples, features*seq_len) to 3D (samples, seq_len, features)
        if len(data.shape) == 2:
            seq_len = self.config.get("seq_len")
            n_features = self.config.get("n_features")
            if data.shape[1] == seq_len * n_features:
                data = data.reshape((data.shape[0], seq_len, n_features))
            else:
                print(f"Error: Cannot reshape data of shape {data.shape} to ({data.shape[0]}, {seq_len}, {n_features}). Check data format.")
                return
        elif len(data.shape) != 3:
            print(f"Error: Data has unexpected shape {data.shape}. Expected 3D (samples, seq_len, features) or 2D that can be reshaped.")
            return

        epochs = self.config.get("gan_epochs", 100)
        batch_size = self.config.get("gan_batch_size", 32)
        latent_dim = self.config.get("latent_dim")
        
        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            idx = np.random.randint(0, data.shape[0], batch_size)
            real_sequences = data[idx]
            
            noise = np.random.normal(0, 1, (batch_size, latent_dim))
            fake_sequences = self.generator.predict(noise)
            
            d_loss_real = self.discriminator.train_on_batch(real_sequences, real_labels)
            d_loss_fake = self.discriminator.train_on_batch(fake_sequences, fake_labels)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            noise = np.random.normal(0, 1, (batch_size, latent_dim))
            g_loss = self.gan.train_on_batch(noise, real_labels)
            
            if epoch % self.config.get("gan_save_interval", 100) == 0 or epoch == epochs -1:
                print(f"{epoch} [D loss: {d_loss[0]:.4f}, acc.: {100*d_loss[1]:.2f}%] [G loss: {g_loss:.4f}]")
                self.save_model()

        print("GAN Training Finished.")
        self.save_model()


    def save_model(self):
        if not self.generator or not self.discriminator:
            print("Models not built. Cannot save.")
            return
            
        model_dir = self.config.get("gan_model_dir", "models/gan")
        os.makedirs(model_dir, exist_ok=True)
        self.generator.save(os.path.join(model_dir, "generator_model.h5"))
        self.discriminator.save(os.path.join(model_dir, "discriminator_model.h5"))
        print(f"GAN models saved to {model_dir}")

    def load_model(self):
        model_dir = self.config.get("gan_model_dir", "models/gan")
        gen_path = os.path.join(model_dir, "generator_model.h5")
        # disc_path = os.path.join(model_dir, "discriminator_model.h5") # Discriminator not always needed for generation

        loaded_successfully = False
        if os.path.exists(gen_path):
            try:
                self.generator = tf.keras.models.load_model(gen_path, compile=False)
                print(f"Generator loaded from {gen_path}")
                loaded_successfully = True
            except Exception as e:
                print(f"Error loading generator model from {gen_path}: {e}")
        else:
            print(f"Warning: Generator model not found at {gen_path}.")

        if not loaded_successfully:
            print("Building new generator model as loading failed or no model found.")
            self._build_generator() # Build a new one if loading fails
        
        # Discriminator and GAN model are primarily for training, 
        # so their loading can be deferred or handled by a separate compile/setup step if resuming training.
        # For now, GANTrainerPlugin.load_model focuses on getting the generator ready.
        # If discriminator is needed by other parts, it should be loaded too.
        # self._build_models_and_compile() # Re-compile if needed for further training or full GAN use


    def get_generator(self):
        if self.generator is None:
            self.load_model() 
        return self.generator
