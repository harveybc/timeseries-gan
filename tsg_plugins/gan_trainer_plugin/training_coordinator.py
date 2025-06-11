#!/usr/bin/env python3
"""
Training Coordinator Module

This module handles the core GAN training orchestration, managing the training loop,
loss calculations, and coordination between generator and discriminator training.
"""

import tensorflow as tf
import numpy as np
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K # Add this import

class TrainingCoordinator:
    """Coordinates GAN training process with proper orchestration."""
    
    def __init__(self, params: Dict[str, Any], logger: logging.Logger, generator_plugin: Any): # Added generator_plugin
        """Initialize training coordinator."""
        self.params = params
        self.logger = logger
        self.generator_plugin = generator_plugin # Store generator_plugin instance
        
        # Training state
        self.current_epoch = 0
        self.training_history = {
            'generator_losses': [],
            'discriminator_losses': [],
            'epochs': [],
            'timestamps': []
        }
        
        # Optimizers
        self.generator_optimizer = None
        self.discriminator_optimizer = None
        
        self.logger.info("TrainingCoordinator initialized")
    
    def _setup_optimizers(self):
        """Setup optimizers for generator and discriminator."""
        self.generator_optimizer = Adam(
            learning_rate=self.params.get("generator_lr", 1e-4),
            beta_1=self.params.get("generator_beta1", 0.5)
        )
        
        self.discriminator_optimizer = Adam(
            learning_rate=self.params.get("discriminator_lr", 1e-4),
            beta_1=self.params.get("discriminator_beta1", 0.5)
        )
        
        self.logger.info("Optimizers setup complete")
    
    def train(self, generator: tf.keras.Model, discriminator: tf.keras.Model, 
              gan_model: tf.keras.Model, feeder_plugin: Any, 
              training_data: pd.DataFrame = None, epochs: int = None, 
              batch_size: int = None, train_discriminator_n_times: int = 1, 
              train_generator_n_times: int = 1, save_interval: int = 500, 
              models_dir: str = "", plots_dir: str = "", metrics_dir: str = "",
              lr_scheduler_g: Optional[tf.keras.callbacks.Callback] = None, # Add LR schedulers
              lr_scheduler_d: Optional[tf.keras.callbacks.Callback] = None, # Add LR schedulers
              **kwargs) -> Dict[str, Any]:
        """
        Main training loop for GAN.
        
        Args:
            generator: Generator model
            discriminator: Discriminator model
            gan_model: Combined GAN model
            feeder_plugin: Plugin for data feeding
            training_data: Real training data
            epochs: Number of training epochs
            batch_size: Training batch size
            train_discriminator_n_times: Discriminator training steps per iteration
            train_generator_n_times: Generator training steps per iteration
            save_interval: Interval for saving models
            models_dir: Directory to save models
            plots_dir: Directory to save plots
            metrics_dir: Directory to save metrics
            lr_scheduler_g: Learning rate scheduler for the generator.
            lr_scheduler_d: Learning rate scheduler for the discriminator.
        
        Returns:
            Training history dictionary
        """
        self.logger.info(f"Starting GAN training for {epochs} epochs with batch_size {batch_size}")
        
        # Use defaults if not provided
        if epochs is None:
            epochs = self.params.get("epochs", 1000)
        if batch_size is None:
            batch_size = self.params.get("batch_size", 32)
        if training_data is None:
            raise ValueError("training_data parameter is required")
        
        # Setup optimizers
        self._setup_optimizers()
        
        # Prepare real data for training
        self.logger.info("Calling _prepare_real_data...")
        real_data = self._prepare_real_data(training_data) # Removed batch_size argument
        self.logger.info(f"Real data preparation complete. Shape: {real_data.shape}")
        
        self.logger.info("Starting training loop...")
        # Training loop
        for epoch in range(epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            self.logger.debug(f"Epoch {epoch+1}/{epochs}: Starting discriminator training step...")
            # Train discriminator
            d_loss = self._train_discriminator_step(
                real_data, generator, discriminator, batch_size, train_discriminator_n_times
            )
            self.logger.debug(f"Epoch {epoch+1}/{epochs}: Discriminator training step completed. D_loss: {d_loss:.4f}")
            
            self.logger.debug(f"Epoch {epoch+1}/{epochs}: Starting generator training step...")
            # Train generator
            g_loss = self._train_generator_step(
                gan_model, batch_size, train_generator_n_times
            )
            self.logger.debug(f"Epoch {epoch+1}/{epochs}: Generator training step completed. G_loss: {g_loss:.4f}")
            
            # Record training metrics
            epoch_time = time.time() - epoch_start_time
            self._record_epoch_metrics(epoch, g_loss, d_loss, epoch_time)

            # Call ReduceLROnPlateau callbacks
            if lr_scheduler_g:
                lr_scheduler_g.on_epoch_end(epoch, logs={'g_loss': g_loss, 'lr': K.get_value(self.generator_optimizer.learning_rate)})
            if lr_scheduler_d:
                lr_scheduler_d.on_epoch_end(epoch, logs={'d_loss': d_loss, 'lr': K.get_value(self.discriminator_optimizer.learning_rate)})
            
            # Log progress every epoch
            log_interval_epochs = self.params.get("log_interval_epochs", 1)
            if (epoch + 1) % log_interval_epochs == 0:
                self.logger.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"G_loss: {g_loss:.4f}, D_loss: {d_loss:.4f}, "
                    f"G_LR: {K.get_value(self.generator_optimizer.learning_rate):.1e}, "
                    f"D_LR: {K.get_value(self.discriminator_optimizer.learning_rate):.1e}, "
                    f"Time: {epoch_time:.2f}s"
                )
            
            # Save models at intervals
            if epoch % save_interval == 0 and epoch > 0:
                self._save_checkpoint(epoch, generator, discriminator, gan_model, models_dir)
        
        self.logger.info("GAN training completed")
        return self.training_history
    
    def _prepare_real_data(self, training_data: pd.DataFrame) -> np.ndarray:
        self.logger.info("Preparing real data for discriminator...")
        if not self.generator_plugin:
            self.logger.error("GeneratorPlugin not initialized in TrainingCoordinator.")
            raise RuntimeError("GeneratorPlugin not initialized in TrainingCoordinator.")

        # Ensure training_data is a DataFrame or can be converted
        training_data_df: pd.DataFrame
        if not isinstance(training_data, pd.DataFrame):
            if hasattr(training_data, 'df') and isinstance(getattr(training_data, 'df'), pd.DataFrame):
                training_data_df = getattr(training_data, 'df').copy()
                self.logger.info("Extracted DataFrame from training_data object's 'df' attribute.")
            else:
                self.logger.error(f"training_data is not a DataFrame and has no 'df' attribute of type DataFrame. Actual type: {type(training_data)}")
                raise TypeError("training_data must be a pandas DataFrame or an object with a 'df' attribute that is a DataFrame.")
        else:
            training_data_df = training_data.copy()

        self.logger.debug(f"Input training_data_df shape: {training_data_df.shape}, columns: {training_data_df.columns.tolist() if isinstance(training_data_df, pd.DataFrame) else 'N/A'}")

        data_array_2d: Optional[np.ndarray] = None 

        try:
            # Step 1: Get processed data from the plugin
            # This call is expected to perform feature engineering and return either a DataFrame or a NumPy array.
            processed_data = self.generator_plugin.prepare_features_for_discriminator(training_data_df)
            self.logger.info(f"Plugin `prepare_features_for_discriminator` returned type: {type(processed_data)}. Log from plugin: Successfully prepared features...")


            # Step 2: Ensure it's a 2D NumPy array
            if isinstance(processed_data, pd.DataFrame):
                self.logger.info("Converting processed DataFrame to NumPy array in chunks...")
                num_rows = len(processed_data)
                if num_rows == 0:
                    self.logger.info("Processed DataFrame is empty. Assigning empty NumPy array.")
                    data_array_2d = np.array([])
                else:
                    chunk_size = self.params.get("data_conversion_chunk_size", 1000)
                    num_chunks = (num_rows + chunk_size - 1) // chunk_size
                    np_arrays = []
                    self.logger.info(f"Will convert {num_rows} rows from DataFrame to NumPy in {num_chunks} chunks of approx {chunk_size} rows each.")
                    for i in range(num_chunks):
                        start_idx = i * chunk_size
                        end_idx = min((i + 1) * chunk_size, num_rows)
                        df_chunk = processed_data.iloc[start_idx:end_idx] # Safe: processed_data is DataFrame
                        np_arrays.append(df_chunk.to_numpy())
                        if (i + 1) % max(1, num_chunks // 10) == 0 or (i + 1) == num_chunks:
                            self.logger.info(f"Converted DataFrame chunk {i+1}/{num_chunks} to NumPy array.")
                    data_array_2d = np.concatenate(np_arrays, axis=0)
                    self.logger.info(f"Conversion from DataFrame to NumPy array complete. Final shape: {data_array_2d.shape}")
            
            elif isinstance(processed_data, np.ndarray):
                self.logger.info(f"Data from plugin is already a NumPy array. Shape: {processed_data.shape}")
                if processed_data.ndim == 0 or (processed_data.ndim > 0 and processed_data.shape[0] == 0): 
                    self.logger.info("Processed NumPy array is empty or scalar. Assigning empty NumPy array.")
                    data_array_2d = np.array([])
                else:
                    data_array_2d = processed_data
            
            else:
                err_msg = f"Unexpected data type from generator_plugin.prepare_features_for_discriminator: {type(processed_data)}. Expected pd.DataFrame or np.ndarray."
                self.logger.error(err_msg)
                raise TypeError(err_msg)

        except Exception as e:
            self.logger.error(f"Error during data preparation by plugin or conversion to 2D NumPy array: {e}", exc_info=True)
            raise

        if data_array_2d is None: # Should not happen if logic above is correct, but as a safeguard
            self.logger.error("data_array_2d is None after processing and conversion attempts.")
            raise ValueError("Failed to produce a valid 2D NumPy array from the input data.")

        if data_array_2d.size == 0:
             self.logger.warning("Resulting 2D data array is empty. Returning empty array. This might cause issues downstream.")
             return np.array([])

        # Step 3: Validate the number of features of the 2D NumPy array
        expected_features = self.params.get("expected_feature_count_for_discriminator", 51) 
        # Consider making "expected_feature_count_for_discriminator" a formal parameter in config.py
        # Or derive from self.params.get("generator_full_feature_names_ordered", [])
        # For now, using a direct param or a default.

        if data_array_2d.ndim != 2:
            self.logger.error(f"Processed data is not 2D after all conversions. Shape: {data_array_2d.shape}")
            raise ValueError(f"Processed data must be 2D for feature validation, but got shape {data_array_2d.shape}")

        if data_array_2d.shape[1] != expected_features:
            self.logger.error(f"Feature mismatch in prepared real data: Processed data has {data_array_2d.shape[1]} features, expected {expected_features}.")
            raise ValueError(f"Processed data has {data_array_2d.shape[1]} features, expected {expected_features}.")
        
        self.logger.info(f"Real data (2D NumPy array) successfully prepared with {data_array_2d.shape[1]} features. Shape: {data_array_2d.shape}")
        
        # Step 4: Convert 2D data to 3D sequences for discriminator
        seq_len = self.params.get("seq_len", 144)
        num_samples, num_features_in_array = data_array_2d.shape
        
        if num_samples < seq_len : 
            self.logger.error(f"Cannot create sequences of length {seq_len} from data with only {num_samples} samples.")
            raise ValueError(f"Not enough data ({num_samples} samples) to create sequences of length {seq_len}.")

        num_sequences = num_samples - seq_len + 1
        
        self.logger.info(f"Allocating memory for {num_sequences} sequences of shape ({seq_len}, {num_features_in_array}).")
        sequences_array = np.empty((num_sequences, seq_len, num_features_in_array), dtype=data_array_2d.dtype)
        for i in range(num_sequences):
            sequences_array[i] = data_array_2d[i:i + seq_len]
        
        self.logger.info(f"Created {len(sequences_array)} sequences. Output shape {sequences_array.shape}")
        
        return sequences_array
    
    def _train_discriminator_step(self, real_data: tf.Tensor, generator: tf.keras.Model,
                                  discriminator: tf.keras.Model, batch_size: int,
                                  n_times: int) -> float:
        """
        Train discriminator for n_times steps.
        
        Args:
            real_data: Real training data
            generator: Generator model
            discriminator: Discriminator model
            batch_size: Batch size
            n_times: Number of training steps
        
        Returns:
            Average discriminator loss
        """
        total_loss = 0.0
        
        for _ in range(n_times):
            # Sample real data batch
            batch_indices = tf.random.uniform([batch_size], 0, tf.shape(real_data)[0], dtype=tf.int32)
            real_batch = tf.gather(real_data, batch_indices)
            
            # Generate fake data with proper input shapes for composite generator
            # Per REFERENCE.md, generator expects: [noise_input, conditions_input, context_input] - 3 inputs
            noise_dim = self.params.get("noise_dim", 100)
            conditional_features_dim = self.params.get("conditional_features_dim", 10)
            context_vector_dim = self.params.get("context_vector_dim", 64)
            
            # Generate inputs for composite generator (3 inputs as per REFERENCE.md)
            noise = tf.random.normal([batch_size, noise_dim])
            conditions = tf.random.normal([batch_size, conditional_features_dim])
            context = tf.random.normal([batch_size, context_vector_dim])
            
            # Generate fake batch with correct 3 inputs
            fake_batch = generator([noise, conditions, context], training=False)
            
            # Debug: Log shapes to verify compatibility
            self.logger.debug(f"Real batch shape: {real_batch.shape}, Fake batch shape: {fake_batch.shape}")
            
            # Ensure both real and fake batches have the same shape for discriminator
            if real_batch.shape != fake_batch.shape:
                self.logger.error(f"Shape mismatch between real and fake batches. Real: {real_batch.shape}, Fake: {fake_batch.shape}")
                raise ValueError(f"Real and fake batch shapes must match. Real: {real_batch.shape}, Fake: {fake_batch.shape}")
            
            # Train discriminator
            with tf.GradientTape() as tape:
                # Predictions
                real_pred = discriminator(real_batch, training=True)
                fake_pred = discriminator(fake_batch, training=True)
                
                # Calculate loss
                d_loss = self._discriminator_loss(real_pred, fake_pred)
            
            # Apply gradients
            gradients = tape.gradient(d_loss, discriminator.trainable_variables)
            
            # Debug: Check discriminator trainable variables
            if len(discriminator.trainable_variables) == 0:
                self.logger.error("Discriminator has no trainable variables!")
                raise RuntimeError("Discriminator model has no trainable variables. Model may not be compiled correctly.")
            
            self.logger.debug(f"Discriminator has {len(discriminator.trainable_variables)} trainable variables")
            
            # Filter out None gradients and ensure we have valid gradient-variable pairs
            valid_grads_and_vars = []
            for grad, var in zip(gradients, discriminator.trainable_variables):
                if grad is not None:
                    valid_grads_and_vars.append((grad, var))
            
            if len(valid_grads_and_vars) == 0:
                self.logger.warning("No valid gradients found for discriminator. Skipping gradient update.")
            else:
                self.discriminator_optimizer.apply_gradients(valid_grads_and_vars)
            
            total_loss += d_loss.numpy()
        
        return total_loss / n_times
    
    def _train_generator_step(self, gan_model: tf.keras.Model, batch_size: int,
                             n_times: int) -> float:
        """
        Train generator for n_times steps.
        
        Args:
            gan_model: Combined GAN model (with discriminator frozen)
            batch_size: Batch size for generating inputs
            n_times: Number of training steps for the generator in this call
        
        Returns:
            Average generator loss
        """
        total_loss = 0.0
        
        # Retrieve necessary dimensions from self.params, consistent with _train_discriminator_step
        # These are defined in app/config.py and documented in REFERENCE_Config_FileTree.md
        noise_dim = self.params.get("noise_dim", 100) 
        conditional_features_dim = self.params.get("conditional_features_dim", 10)
        context_vector_dim = self.params.get("context_vector_dim", 64)

        self.logger.debug(f"Generator step: noise_dim={noise_dim}, cond_dim={conditional_features_dim}, ctx_dim={context_vector_dim}, batch_size={batch_size}")

        for _ in range(n_times):
            # Generate inputs for the composite generator model, which expects 3 inputs:
            # 1. Noise vector
            # 2. Conditional features vector (e.g., cyclical date/time)
            # 3. Context vector (e.g., for sequential state, can be random/zero for non-iterative step)
            # This matches the input structure of the composite generator defined in GeneratorPlugin
            # and aligns with how inputs are prepared in _train_discriminator_step for the generator call.
            
            noise = tf.random.normal([batch_size, noise_dim], name="gen_step_noise_input")
            conditions = tf.random.normal([batch_size, conditional_features_dim], name="gen_step_conditions_input")
            context = tf.random.normal([batch_size, context_vector_dim], name="gen_step_context_input")
            
            self.logger.debug(f"Generated inputs for GAN model (generator training): noise_shape={noise.shape}, conditions_shape={conditions.shape}, context_shape={context.shape}")

            with tf.GradientTape() as tape:
                # The gan_model here is expected to have the discriminator's layers frozen.
                # It takes the generator's inputs and passes them through G, then D.
                fake_pred = gan_model([noise, conditions, context], training=True) # training=True for G's layers
                
                g_loss = self._generator_loss(fake_pred)

            generator_trainable_vars = gan_model.trainable_variables 

            if not generator_trainable_vars:
                self.logger.critical(f"CRITICAL ERROR: The 'gan_model' (used for training the generator) has NO trainable variables that belong to the generator.")
                self.logger.critical(f"This means either the generator sub-model itself has no trainable weights, or the gan_model is not set up correctly with discriminator.trainable=False.")
                self.logger.critical(f"GAN Model Name (for G training): {gan_model.name}")
                self.logger.critical(f"GAN Model Trainable Variables (should be G's): {[(v.name, v.shape) for v in generator_trainable_vars]}")
                if hasattr(self, 'generator_model') and self.generator_model:
                    self.logger.critical(f"Generator Model Name (original): {self.generator_model.name}")
                    self.logger.critical(f"Generator Model (original) Trainable Weights Count: {len(self.generator_model.trainable_weights)}")
                else:
                    self.logger.critical("Original generator_model not found on self for detailed logging.")
                raise RuntimeError("Generator component of gan_model has no trainable variables. Robust check failed.")
            
            self.logger.debug(f"Found {len(generator_trainable_vars)} trainable variables for the generator in gan_model.")

            gradients = tape.gradient(g_loss, generator_trainable_vars)
            
            # Filter out None gradients and ensure we have valid gradient-variable pairs
            valid_grads_and_vars = []
            for grad, var in zip(gradients, generator_trainable_vars):
                if grad is not None:
                    valid_grads_and_vars.append((grad, var))
            
            if not valid_grads_and_vars: # Check if the list is empty
                self.logger.warning("No valid gradients found for generator. Skipping gradient update.")
            else:
                self.generator_optimizer.apply_gradients(valid_grads_and_vars)
                
            total_loss += g_loss.numpy()
        
        return total_loss / n_times
    
    def _discriminator_loss(self, real_pred: tf.Tensor, fake_pred: tf.Tensor) -> tf.Tensor:
        """
        Calculate discriminator loss.
        
        Args:
            real_pred: Discriminator predictions on real data
            fake_pred: Discriminator predictions on fake data
        
        Returns:
            Discriminator loss
        """
        real_loss = tf.keras.losses.binary_crossentropy(tf.ones_like(real_pred), real_pred)
        fake_loss = tf.keras.losses.binary_crossentropy(tf.zeros_like(fake_pred), fake_pred)
        return tf.reduce_mean(real_loss + fake_loss)
    
    def _generator_loss(self, fake_pred: tf.Tensor) -> tf.Tensor:
        """
        Calculate generator loss.
        
        Args:
            fake_pred: Discriminator predictions on generated data
        
        Returns:
            Generator loss
        """
        return tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones_like(fake_pred), fake_pred))
    
    def _record_epoch_metrics(self, epoch: int, g_loss: float, d_loss: float, epoch_time: float):
        """Record metrics for current epoch."""
        self.training_history['epochs'].append(epoch)
        self.training_history['generator_losses'].append(g_loss)
        self.training_history['discriminator_losses'].append(d_loss)
        self.training_history['timestamps'].append(time.time())
    
    def _save_checkpoint(self, epoch: int, generator: tf.keras.Model,
                        discriminator: tf.keras.Model, gan_model: tf.keras.Model,
                        models_dir: str):
        """Save model checkpoint."""
        try:
            import os
            
            generator_path = os.path.join(models_dir, f"generator_epoch_{epoch}.keras")
            discriminator_path = os.path.join(models_dir, f"discriminator_epoch_{epoch}.keras")
            gan_path = os.path.join(models_dir, f"gan_epoch_{epoch}.keras")
            
            generator.save(generator_path)
            discriminator.save(discriminator_path)
            gan_model.save(gan_path)
            
            self.logger.info(f"Checkpoint saved at epoch {epoch}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save checkpoint: {e}")
    
    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information."""
        return {
            "current_epoch": self.current_epoch,
            "training_history_length": len(self.training_history['epochs']),
            "last_generator_loss": self.training_history['generator_losses'][-1] if self.training_history['generator_losses'] else None,
            "last_discriminator_loss": self.training_history['discriminator_losses'][-1] if self.training_history['discriminator_losses'] else None
        }
