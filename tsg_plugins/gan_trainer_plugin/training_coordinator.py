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
        
        Returns:
            Training history dictionary
        """
        self.logger.info(f"Starting GAN training for {epochs} epochs")
        
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
        real_data = self._prepare_real_data(training_data) # Removed batch_size argument
        
        # Training loop
        for epoch in range(epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Train discriminator
            d_loss = self._train_discriminator_step(
                real_data, generator, discriminator, batch_size, train_discriminator_n_times
            )
            
            # Train generator
            g_loss = self._train_generator_step(
                gan_model, batch_size, train_generator_n_times
            )
            
            # Record training metrics
            epoch_time = time.time() - epoch_start_time
            self._record_epoch_metrics(epoch, g_loss, d_loss, epoch_time)
            
            # Log progress
            if epoch % 100 == 0:
                self.logger.info(f"Epoch {epoch}/{epochs} - G_loss: {g_loss:.4f}, D_loss: {d_loss:.4f}, Time: {epoch_time:.2f}s")
            
            # Save models at intervals
            if epoch % save_interval == 0 and epoch > 0:
                self._save_checkpoint(epoch, generator, discriminator, gan_model, models_dir)
        
        self.logger.info("GAN training completed")
        return self.training_history
    
    def _prepare_real_data(self, training_data: pd.DataFrame) -> np.ndarray:
        self.logger.info("Preparing real data for discriminator...")
        if not self.generator_plugin:
            raise RuntimeError("GeneratorPlugin not initialized in TrainingCoordinator.")

        # Ensure training_data is a DataFrame
        if not isinstance(training_data, pd.DataFrame):
            if hasattr(training_data, 'df') and isinstance(training_data.df, pd.DataFrame):
                training_data_df = training_data.df.copy()
                self.logger.info("Extracted DataFrame from training_data object.")
            else:
                self.logger.error(f"training_data is not a DataFrame and has no 'df' attribute. Type: {type(training_data)}")
                raise TypeError("training_data must be a pandas DataFrame or an object with a 'df' attribute that is a DataFrame.")
        else:
            training_data_df = training_data.copy()

        self.logger.debug(f"Input training_data_df shape: {training_data_df.shape}, columns: {training_data_df.columns.tolist()}")

        try:
            processed_df = self.generator_plugin.prepare_features_for_discriminator(training_data_df)
            if isinstance(processed_df, np.ndarray):
                self.logger.info(f"Real data processed by GeneratorPlugin. Shape: {processed_df.shape}")
            else:
                self.logger.info(f"Real data processed by GeneratorPlugin. Shape: {processed_df.shape}, Columns: {processed_df.columns.tolist()}")
        except Exception as e:
            self.logger.error(f"Error during generator_plugin.prepare_features_for_discriminator: {e}", exc_info=True)
            raise

        # Validate the number of features after processing
        # Align with the 51-feature architecture (23 base + 8 cyclical + 20 TIs)
        expected_features = 51 # As per updated REFERENCE.md and config.py
        
        # More robust way to get expected_features if params are reliably populated
        # full_feature_list = self.params.get("generator_full_feature_names_ordered", [])
        # datetime_col_name = self.params.get("datetime_col_name", "DATE_TIME")
        # if full_feature_list:
        #     calculated_expected_features = len([f for f in full_feature_list if f != datetime_col_name])
        #     if calculated_expected_features > 0:
        #         expected_features = calculated_expected_features
        #     else:
        #         self.logger.warning(f"'generator_full_feature_names_ordered' resulted in 0 numeric features. Defaulting to {expected_features}.")
        # else:
        #     self.logger.warning(f"'generator_full_feature_names_ordered' not found in params. Defaulting to {expected_features} features.")


        if processed_df.shape[1] != expected_features:
            if isinstance(processed_df, np.ndarray):
                self.logger.error(f"Feature mismatch in prepared real data: Processed data has {processed_df.shape[1]} features, but expected {expected_features}.")
            else:
                self.logger.error(f"Feature mismatch in prepared real data: Processed data has {processed_df.shape[1]} features, but expected {expected_features}. Columns: {processed_df.columns.tolist()}")
            raise ValueError(f"Processed data has {processed_df.shape[1]} features, expected {expected_features}.")
        
        self.logger.info(f"Real data successfully prepared with {processed_df.shape[1]} features.")
        
        # Convert to numpy array
        if isinstance(processed_df, np.ndarray):
            data_array = processed_df
        else:
            data_array = processed_df.values
        
        # Convert 2D data to 3D sequences for discriminator
        # Discriminator expects (batch_size, sequence_length, features)
        seq_len = self.params.get("seq_len", 144)
        
        # Create sequences from the data
        sequences = []
        num_samples, num_features = data_array.shape
        
        # Generate sequences of length seq_len
        for i in range(num_samples - seq_len + 1):
            sequence = data_array[i:i + seq_len]
            sequences.append(sequence)
        
        if len(sequences) == 0:
            self.logger.error(f"Cannot create sequences of length {seq_len} from data with {num_samples} samples")
            raise ValueError(f"Not enough data to create sequences of length {seq_len}")
        
        sequences_array = np.array(sequences)
        self.logger.info(f"Created {len(sequences)} sequences of shape {sequences_array.shape}")
        
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
            gan_model: Combined GAN model
            batch_size: Batch size
            n_times: Number of training steps
        
        Returns:
            Average generator loss
        """
        total_loss = 0.0
        
        for _ in range(n_times):
            # Generate inputs for composite generator (3 inputs as per REFERENCE.md)
            noise_dim = self.params.get("noise_dim", 100)
            conditional_features_dim = self.params.get("conditional_features_dim", 10)
            context_vector_dim = self.params.get("context_vector_dim", 64)
            
            # Generate inputs for composite generator
            noise = tf.random.normal([batch_size, noise_dim])
            conditions = tf.random.normal([batch_size, conditional_features_dim])
            context = tf.random.normal([batch_size, context_vector_dim])
            
            # Train generator via GAN model
            with tf.GradientTape() as tape:
                # Generate fake data and get discriminator prediction
                fake_pred = gan_model([noise, conditions, context], training=True)
                
                # Calculate generator loss
                g_loss = self._generator_loss(fake_pred)
            
            # Apply gradients to generator only
            generator_vars = gan_model.layers[0].trainable_variables  # Assuming generator is first layer
            gradients = tape.gradient(g_loss, generator_vars)
            
            # Debug: Check generator trainable variables
            if len(generator_vars) == 0:
                self.logger.error("Generator has no trainable variables!")
                raise RuntimeError("Generator model has no trainable variables. Model may not be compiled correctly.")
            
            self.logger.debug(f"Generator has {len(generator_vars)} trainable variables")
            
            # Filter out None gradients and ensure we have valid gradient-variable pairs
            valid_grads_and_vars = []
            for grad, var in zip(gradients, generator_vars):
                if grad is not None:
                    valid_grads_and_vars.append((grad, var))
            
            if len(valid_grads_and_vars) == 0:
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
