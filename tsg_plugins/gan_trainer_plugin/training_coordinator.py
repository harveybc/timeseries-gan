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
from tqdm import tqdm # Import tqdm
import os # Import os for path joining

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
        # Parameters are expected to be in self.params, sourced from main_config
        # Use direct access (self.params[key]) to ensure error if missing critical param
        try:
            gen_lr = self.params["generator_lr"]
            gen_beta1 = self.params["generator_beta1"]
            disc_lr = self.params["discriminator_lr"]
            disc_beta1 = self.params["discriminator_beta1"]
        except KeyError as e:
            self.logger.error(f"Missing critical optimizer parameter in self.params: {e}. "
                              f"Ensure 'generator_lr', 'generator_beta1', 'discriminator_lr', 'discriminator_beta1' "
                              f"are defined in config.py or plugin defaults and correctly passed.")
            raise

        self.generator_optimizer = Adam(
            learning_rate=gen_lr,
            beta_1=gen_beta1
        )
        
        self.discriminator_optimizer = Adam(
            learning_rate=disc_lr,
            beta_1=disc_beta1
        )
        
        self.logger.info(f"Optimizers setup complete. G_LR: {gen_lr}, D_LR: {disc_lr}")
    
    def train(self, generator: tf.keras.Model, discriminator: tf.keras.Model, 
              gan_model: tf.keras.Model, feeder_plugin: Any, 
              training_data: pd.DataFrame = None, epochs: int = None, 
              batch_size: int = None, train_discriminator_n_times: int = 1, 
              train_generator_n_times: int = 1, save_interval: int = 500, 
              models_dir: str = "", plots_dir: str = "", metrics_dir: str = "",
              lr_scheduler_g: Optional[tf.keras.callbacks.Callback] = None,
              lr_scheduler_d: Optional[tf.keras.callbacks.Callback] = None,
              early_stopping_callback: Optional[tf.keras.callbacks.Callback] = None, # Add early_stopping_callback
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
            early_stopping_callback: Early stopping callback.
        
        Returns:
            Training history dictionary
        """
        self.logger.info(f"TrainingCoordinator.train: Received epochs argument: {epochs}, batch_size argument: {batch_size}")
        
        # The 'epochs' and 'batch_size' arguments passed here should be the definitive values from GANTrainerPlugin.
        final_epochs = epochs
        if final_epochs is None:
            self.logger.warning("TrainingCoordinator.train: 'epochs' argument from caller was None. This is unexpected.")
            # Fallback to TrainingCoordinator's own params if caller failed to provide epochs.
            # TC.params should have 'gan_epochs' if GANTrainerPlugin.set_params(TC) worked.
            final_epochs = self.params.get("gan_epochs") 
            if final_epochs is None:
                self.logger.error("TrainingCoordinator.train: 'gan_epochs' not found in self.params either. Defaulting to a hardcoded 1000 for epochs.")
                final_epochs = 1000 # Last resort default for epochs
            else:
                self.logger.info(f"TrainingCoordinator.train: Using 'gan_epochs' from self.params for epochs: {final_epochs}")

        final_batch_size = batch_size
        if final_batch_size is None:
            self.logger.warning("TrainingCoordinator.train: 'batch_size' argument from caller was None. This is unexpected.")
            final_batch_size = self.params.get("gan_batch_size")
            if final_batch_size is None:
                self.logger.error("TrainingCoordinator.train: 'gan_batch_size' not found in self.params either. Defaulting to a hardcoded 32 for batch_size.")
                final_batch_size = 32 # Last resort default for batch_size
            else:
                self.logger.info(f"TrainingCoordinator.train: Using 'gan_batch_size' from self.params for batch_size: {final_batch_size}")
        
        if training_data is None:
            self.logger.error("TrainingCoordinator.train: training_data parameter is required and was None.")
            raise ValueError("training_data parameter is required")
        
        self.logger.info(f"TrainingCoordinator: Starting GAN training for {final_epochs} epochs with batch_size {final_batch_size}")

        # Setup optimizers
        self._setup_optimizers()

        # The following block for linking optimizers to LR schedulers' models is removed.
        # This is now handled in GANTrainerPlugin by calling set_model() on the schedulers
        # with the already compiled Keras models (gan_model and discriminator_model),
        # which have their optimizers set during their compilation.
        # self.logger.info("Linking TrainingCoordinator's optimizers to models held by LR schedulers.")
        # if lr_scheduler_g:
        #     if hasattr(lr_scheduler_g, 'model') and lr_scheduler_g.model:
        #         if hasattr(lr_scheduler_g.model, 'optimizer') and lr_scheduler_g.model.optimizer is not self.generator_optimizer:
        #             self.logger.info(f"Updating optimizer for lr_scheduler_g.model ({lr_scheduler_g.model.name}) to TrainingCoordinator's generator_optimizer.")
        #             lr_scheduler_g.model.optimizer = self.generator_optimizer
        #         elif not hasattr(lr_scheduler_g.model, 'optimizer'):
        #              self.logger.info(f"Assigning TrainingCoordinator's generator_optimizer to lr_scheduler_g.model ({lr_scheduler_g.model.name}).")
        #              lr_scheduler_g.model.optimizer = self.generator_optimizer
        #     else:
        #         self.logger.warning("lr_scheduler_g was passed but has no .model attribute set or it's None. Cannot link optimizer.")
        # 
        # if lr_scheduler_d:
        #     if hasattr(lr_scheduler_d, 'model') and lr_scheduler_d.model:
        #         if hasattr(lr_scheduler_d.model, 'optimizer') and lr_scheduler_d.model.optimizer is not self.discriminator_optimizer:
        #             self.logger.info(f"Updating optimizer for lr_scheduler_d.model ({lr_scheduler_d.model.name}) to TrainingCoordinator's discriminator_optimizer.")
        #             lr_scheduler_d.model.optimizer = self.discriminator_optimizer
        #         elif not hasattr(lr_scheduler_d.model, 'optimizer'):
        #             self.logger.info(f"Assigning TrainingCoordinator's discriminator_optimizer to lr_scheduler_d.model ({lr_scheduler_d.model.name}).")
        #             lr_scheduler_d.model.optimizer = self.discriminator_optimizer
        #     else:
        #         self.logger.warning("lr_scheduler_d was passed but has no .model attribute set or it's None. Cannot link optimizer.")

        # Call on_train_begin for EarlyStopping callback if it exists
        if early_stopping_callback:
            self.logger.info("Calling on_train_begin for EarlyStopping callback.")
            early_stopping_callback.on_train_begin(logs=None) # Initialize EarlyStopping callback

        # Print model summaries directly to stdout for diagnosis
        self.logger.info("Attempting to print Generator Model Summary (if logger is working)...")
        print("\nGenerator Model Summary (direct print):")
        generator.summary() # Prints to stdout by default
        self.logger.info("Attempting to print Discriminator Model Summary (if logger is working)...")
        print("\nDiscriminator Model Summary (direct print):")
        discriminator.summary() # Prints to stdout by default
        
        # Prepare real data for training
        self.logger.info("Calling _prepare_real_data...")
        real_data = self._prepare_real_data(training_data) # Removed batch_size argument
        self.logger.info(f"_prepare_real_data returned. Type of real_data: {type(real_data)}")
        
        if isinstance(real_data, np.ndarray):
            self.logger.info(f"real_data is a NumPy array. Attempting to access shape. Dtype: {real_data.dtype}, Size: {real_data.size}")
            if real_data.size == 0 and final_epochs > 0: # Check if data is empty but epochs are expected
                self.logger.error("real_data is empty but training epochs are scheduled. This will likely lead to errors.")
                # Potentially raise an error or handle as appropriate
                # For now, just log and let it proceed to see the error or if it handles it
            
            shape_info = real_data.shape # Access shape here
            self.logger.info(f"Real data preparation complete. Shape: {shape_info}")
        else:
            self.logger.warning(f"Real data preparation complete, but real_data is not a NumPy array. Type: {type(real_data)}. This might cause issues.")

        self.logger.info("Checking if final_epochs > 0 before starting training loop...")
        if not isinstance(final_epochs, int) or final_epochs <= 0:
            self.logger.warning(f"Number of epochs is '{final_epochs}' (type: {type(final_epochs)}). Training loop will not run or will cause an error.")
            if isinstance(final_epochs, int) and final_epochs <=0:
                 self.logger.info("GAN training considered complete as final_epochs <= 0.")
                 return self.training_history # Exit if no epochs to run
            # If not an int, it might error out in range(final_epochs) anyway, but good to log.

        self.logger.info(f"Starting training loop for {final_epochs} epochs...")
        # Training loop
        for epoch in range(final_epochs): # Use final_epochs
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            self.logger.debug(f"Epoch {epoch+1}/{final_epochs}: Starting discriminator training step...")
            # Train discriminator
            d_loss_avg, d_loss_real_avg, d_loss_fake_avg = self._train_discriminator_step(
                real_data, generator, discriminator, final_batch_size, train_discriminator_n_times # Use final_batch_size
            )
            self.logger.debug(f"Epoch {epoch+1}/{final_epochs}: Discriminator training step completed. D_loss_avg: {d_loss_avg:.4f}, D_loss_real: {d_loss_real_avg:.4f}, D_loss_fake: {d_loss_fake_avg:.4f}")
            
            self.logger.debug(f"Epoch {epoch+1}/{final_epochs}: Starting generator training step...")
            # Train generator
            g_loss = self._train_generator_step(
                gan_model, final_batch_size, train_generator_n_times # Use final_batch_size
            )
            self.logger.debug(f"Epoch {epoch+1}/{final_epochs}: Generator training step completed. G_loss: {g_loss:.4f}")
            
            # Record training metrics (using the average total discriminator loss)
            epoch_time = time.time() - epoch_start_time
            self._record_epoch_metrics(epoch, g_loss, d_loss_avg, epoch_time)

            # Call ReduceLROnPlateau callbacks
            if lr_scheduler_g:
                # Ensure the optimizer in the model that ReduceLROnPlateau is monitoring is the one we just created.
                # The model (gan_model) should have been compiled with an optimizer instance.
                # We need to ensure K.get_value(model.optimizer.learning_rate) works.
                # If ReduceLROnPlateau was given a model that was compiled with a *different* optimizer instance,
                # or if the learning rate is managed externally to that optimizer instance, this might not reflect correctly.
                # However, standard Keras behavior is that ReduceLROnPlateau modifies the LR of the optimizer
                # attached to the model it was .set_model() with.
                current_g_lr = K.get_value(self.generator_optimizer.learning_rate) # Get LR from TC's optimizer instance
                lr_scheduler_g.on_epoch_end(epoch, logs={'g_loss': g_loss, 'lr': current_g_lr})
                self.logger.debug(f"Called lr_scheduler_g.on_epoch_end. Monitored g_loss: {g_loss:.4f}, Reported G_LR: {current_g_lr:.1e}")
            if lr_scheduler_d:
                current_d_lr = K.get_value(self.discriminator_optimizer.learning_rate) # Get LR from TC's optimizer instance
                lr_scheduler_d.on_epoch_end(epoch, logs={'d_loss': d_loss_avg, 'lr': current_d_lr})
                self.logger.debug(f"Called lr_scheduler_d.on_epoch_end. Monitored d_loss: {d_loss_avg:.4f}, Reported D_LR: {current_d_lr:.1e}")
            
            # Handle Early Stopping
            if early_stopping_callback:
                monitor_metric_name = self.params.get("es_monitor_metric", "g_loss")
                current_metric_value = g_loss # Default to g_loss
                if monitor_metric_name == "d_loss":
                    current_metric_value = d_loss_avg
                elif monitor_metric_name == "combined_loss": # Example if we had a combined loss
                    current_metric_value = (g_loss + d_loss_avg) / 2
                
                logs_for_es = {monitor_metric_name: current_metric_value}
                early_stopping_callback.on_epoch_end(epoch, logs=logs_for_es)
                if early_stopping_callback.model and getattr(early_stopping_callback.model, 'stop_training', False):
                    self.logger.info(f"Early stopping triggered at epoch {epoch+1} "
                                     f"monitoring '{monitor_metric_name}' with value {current_metric_value:.4f}.")
                    # Save models before breaking, as this is the last state due to early stopping
                    self._save_checkpoint(epoch + 1, generator, discriminator, gan_model, models_dir, is_final_save=True)
                    break # Exit training loop
            
            # Log progress every epoch with comprehensive PhD-level metrics
            log_interval_epochs = self.params.get("log_interval_epochs", 1)
            if (epoch + 1) % log_interval_epochs == 0:
                # Format losses with scientific notation for small values
                g_loss_str = f"{g_loss:.4e}" if g_loss < 1e-3 else f"{g_loss:.4f}"
                d_loss_str = f"{d_loss_avg:.4e}" if d_loss_avg < 1e-3 else f"{d_loss_avg:.4f}"
                d_real_str = f"{d_loss_real_avg:.4e}" if d_loss_real_avg < 1e-3 else f"{d_loss_real_avg:.4f}"
                d_fake_str = f"{d_loss_fake_avg:.4e}" if d_loss_fake_avg < 1e-3 else f"{d_loss_fake_avg:.4f}"
                
                # Get current learning rates
                current_g_lr = K.get_value(self.generator_optimizer.learning_rate)
                current_d_lr = K.get_value(self.discriminator_optimizer.learning_rate)
                
                # Primary metrics
                log_msg = (
                    f"Epoch {epoch+1}/{final_epochs} │ "
                    f"G_loss: {g_loss_str} │ D_loss: {d_loss_str} │ "
                    f"D_real: {d_real_str} │ D_fake: {d_fake_str} │ "
                    f"G_LR: {current_g_lr:.2e} │ D_LR: {current_d_lr:.2e}"
                )
                
                # Learning rate scheduler patience (ReduceLROnPlateau)
                lr_patience_info = ""
                if lr_scheduler_g and hasattr(lr_scheduler_g, 'wait'):
                    g_wait = getattr(lr_scheduler_g, 'wait', 0)
                    g_patience = getattr(lr_scheduler_g, 'patience', 0)
                    g_cooldown = getattr(lr_scheduler_g, 'cooldown_counter', 0)
                    lr_patience_info += f" │ LR_G: {g_wait}/{g_patience} (cd:{g_cooldown})"
                
                if lr_scheduler_d and hasattr(lr_scheduler_d, 'wait'):
                    d_wait = getattr(lr_scheduler_d, 'wait', 0)
                    d_patience = getattr(lr_scheduler_d, 'patience', 0)
                    d_cooldown = getattr(lr_scheduler_d, 'cooldown_counter', 0)
                    lr_patience_info += f" │ LR_D: {d_wait}/{d_patience} (cd:{d_cooldown})"
                
                # Early stopping patience
                es_patience_info = ""
                if early_stopping_callback and hasattr(early_stopping_callback, 'wait'):
                    es_wait = getattr(early_stopping_callback, 'wait', 0)
                    es_patience = getattr(early_stopping_callback, 'patience', self.params.get('es_patience', 0))
                    monitor_metric = getattr(early_stopping_callback, 'monitor', 'loss')
                    es_patience_info = f" │ ES: {es_wait}/{es_patience} ({monitor_metric})"
                
                # Complete log message
                log_msg += lr_patience_info + es_patience_info + f" │ Time: {epoch_time:.2f}s"
                self.logger.info(log_msg)
            
            # Save models at intervals - THIS SECTION IS REMOVED
            # if epoch % save_interval == 0 and epoch > 0:
            #     self._save_checkpoint(epoch, generator, discriminator, gan_model, models_dir)
        
        self.logger.info("GAN training completed")
        
        # Save final models after the training loop finishes,
        # unless early stopping already saved and exited.
        if not (early_stopping_callback and early_stopping_callback.model and getattr(early_stopping_callback.model, 'stop_training', False)):
            self.logger.info(f"Saving final models after {final_epochs} epochs...") # Use final_epochs
            self._save_checkpoint(final_epochs, generator, discriminator, gan_model, models_dir, is_final_save=True) # Use final_epochs

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
            self.logger.info("Calling generator_plugin.prepare_features_for_discriminator...")
            processed_data = self.generator_plugin.prepare_features_for_discriminator(training_data_df)
            # The log "Successfully prepared features..." is printed from WITHIN the call above.
            self.logger.info(f"Plugin `prepare_features_for_discriminator` returned. Output type: {type(processed_data)}")


            # Step 2: Ensure it's a 2D NumPy array
            if isinstance(processed_data, pd.DataFrame):
                self.logger.info("Output from plugin is DataFrame. Converting to NumPy array in chunks...")
                num_rows = len(processed_data)
                if num_rows == 0:
                    self.logger.info("Processed DataFrame is empty. Assigning empty NumPy array.")
                    data_array_2d = np.array([])
                else:
                    chunk_size = self.params.get("data_conversion_chunk_size", 1000)
                    num_chunks = (num_rows + chunk_size - 1) // chunk_size
                    np_arrays = []
                    self.logger.info(f"Will convert {num_rows} rows from DataFrame to NumPy in {num_chunks} chunks of approx {chunk_size} rows each.")
                    # Using tqdm for the DataFrame to NumPy conversion loop
                    for i in tqdm(range(num_chunks), desc="Converting DataFrame to NumPy", unit="chunk"):
                        start_idx = i * chunk_size
                        end_idx = min((i + 1) * chunk_size, num_rows)
                        df_chunk = processed_data.iloc[start_idx:end_idx] 
                        np_arrays.append(df_chunk.to_numpy())
                        # Reduced logging frequency inside the loop as tqdm provides progress
                    data_array_2d = np.concatenate(np_arrays, axis=0)
                    self.logger.info(f"Conversion from DataFrame to NumPy array complete. Final shape: {data_array_2d.shape if data_array_2d is not None else 'None'}")
            
            elif isinstance(processed_data, np.ndarray):
                self.logger.info(f"Output from plugin is already a NumPy array. Shape: {processed_data.shape}")
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

        if data_array_2d is None: 
            self.logger.error("data_array_2d is None after processing and conversion attempts.")
            raise ValueError("Failed to produce a valid 2D NumPy array from the input data.")

        if data_array_2d.size == 0:
             self.logger.warning("Resulting 2D data array is empty. Returning empty array. This might cause issues downstream.")
             return np.array([])

        # Step 3: Validate the number of features of the 2D NumPy array
        self.logger.info("Validating features of the 2D NumPy array...")
        expected_features = self.params.get("expected_feature_count_for_discriminator", 51) 

        if data_array_2d.ndim != 2:
            self.logger.error(f"Processed data is not 2D after all conversions. Shape: {data_array_2d.shape}")
            raise ValueError(f"Processed data must be 2D for feature validation, but got shape {data_array_2d.shape}")

        if data_array_2d.shape[1] != expected_features:
            self.logger.error(f"Feature mismatch in prepared real data: Processed data has {data_array_2d.shape[1]} features, expected {expected_features}.")
            raise ValueError(f"Processed data has {data_array_2d.shape[1]} features, expected {expected_features}.")
        
        self.logger.info(f"2D NumPy array successfully prepared and validated. Shape: {data_array_2d.shape}")
        
        # Step 4: Convert 2D data to 3D sequences for discriminator
        self.logger.info("Starting conversion of 2D data to 3D sequences...")
        seq_len = self.params.get("seq_len", 144)
        num_samples, num_features_in_array = data_array_2d.shape
        
        if num_samples < seq_len : 
            self.logger.error(f"Cannot create sequences of length {seq_len} from data with only {num_samples} samples.")
            raise ValueError(f"Not enough data ({num_samples} samples) to create sequences of length {seq_len}.")

        num_sequences = num_samples - seq_len + 1
        
        self.logger.info(f"Allocating memory for {num_sequences} sequences of shape ({seq_len}, {num_features_in_array}).")
        sequences_array = np.empty((num_sequences, seq_len, num_features_in_array), dtype=data_array_2d.dtype)
        
        self.logger.info(f"Creating {num_sequences} sequences. This may take some time...")
        # Add tqdm progress bar here for the sequence creation loop
        for i in tqdm(range(num_sequences), desc="Creating 3D sequences", unit="sequence"):
            sequences_array[i] = data_array_2d[i:i + seq_len]
        
        self.logger.info(f"Successfully created {len(sequences_array)} sequences. Output shape {sequences_array.shape}")
        self.logger.info(f"Preparing to return sequences_array from _prepare_real_data. Type: {type(sequences_array)}, Dtype: {sequences_array.dtype}, Size: {sequences_array.size}")
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
            A tuple containing:
                - Average total discriminator loss (float)
                - Average discriminator loss on real samples (float)
                - Average discriminator loss on fake samples (float)
        """
        total_d_loss_val = 0.0
        total_d_real_loss_val = 0.0
        total_d_fake_loss_val = 0.0
        
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
                
                # Calculate loss components
                d_loss, d_real_loss, d_fake_loss = self._discriminator_loss(real_pred, fake_pred)
            
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
            
            total_d_loss_val += d_loss.numpy()
            total_d_real_loss_val += d_real_loss.numpy()
            total_d_fake_loss_val += d_fake_loss.numpy()
        
        return total_d_loss_val / n_times, total_d_real_loss_val / n_times, total_d_fake_loss_val / n_times
    
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
        Calculate discriminator loss and its components.
        
        Args:
            real_pred: Discriminator predictions on real data
            fake_pred: Discriminator predictions on fake data
        
        Returns:
            A tuple containing:
                - Total discriminator loss (tf.Tensor)
                - Mean loss on real samples (tf.Tensor)
                - Mean loss on fake samples (tf.Tensor)
        """
        real_loss = tf.keras.losses.binary_crossentropy(tf.ones_like(real_pred), real_pred)
        fake_loss = tf.keras.losses.binary_crossentropy(tf.zeros_like(fake_pred), fake_pred)
        
        d_real_loss_mean = tf.reduce_mean(real_loss)
        d_fake_loss_mean = tf.reduce_mean(fake_loss)
        
        total_d_loss = d_real_loss_mean + d_fake_loss_mean # As per original formulation, sum of means
        return total_d_loss, d_real_loss_mean, d_fake_loss_mean
    
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
                         models_dir: str, is_final_save: bool = False): # Add is_final_save flag
        """
        Save model checkpoints. If is_final_save is True, uses specific filenames from config.
        
        Args:
            epoch: Current epoch number
            generator: Generator model
            discriminator: Discriminator model
            gan_model: Combined GAN model
            models_dir: Directory to save models (primarily for intermediate checkpoints or fallbacks)
            is_final_save: Boolean, if True, save with final configured names.
        """
        if is_final_save:
            gen_path_key = "save_generator_sequential_model_file"
            disc_path_key = "save_discriminator_sequential_model_file"

            # --- Generator Saving ---
            generator_save_path_config = self.params.get(gen_path_key)
            if generator_save_path_config:
                final_gen_path = generator_save_path_config
                gen_parent_dir = os.path.dirname(final_gen_path)
                if gen_parent_dir: # Ensure parent_dir is not empty (e.g. if path is just a filename)
                    os.makedirs(gen_parent_dir, exist_ok=True)
                    self.logger.info(f"Ensured directory for final generator model: {gen_parent_dir}")
            else:
                self.logger.warning(f"'{gen_path_key}' not in config. Using fallback for final generator save.")
                os.makedirs(models_dir, exist_ok=True) # Ensure models_dir exists for fallback
                self.logger.info(f"Ensured fallback directory for generator: {models_dir}")
                final_gen_path = os.path.join(models_dir, f"generator_final_epoch_{epoch}.keras")
            
            self.logger.info(f"Saving final generator model to: {final_gen_path}")
            generator.save(final_gen_path)

            # --- Discriminator Saving ---
            discriminator_save_path_config = self.params.get(disc_path_key)
            if discriminator_save_path_config:
                final_disc_path = discriminator_save_path_config
                disc_parent_dir = os.path.dirname(final_disc_path)
                if disc_parent_dir: # Ensure parent_dir is not empty
                    os.makedirs(disc_parent_dir, exist_ok=True)
                    self.logger.info(f"Ensured directory for final discriminator model: {disc_parent_dir}")
            else:
                self.logger.warning(f"'{disc_path_key}' not in config. Using fallback for final discriminator save.")
                os.makedirs(models_dir, exist_ok=True) # Ensure models_dir exists for fallback
                self.logger.info(f"Ensured fallback directory for discriminator: {models_dir}")
                final_disc_path = os.path.join(models_dir, f"discriminator_final_epoch_{epoch}.keras")

            self.logger.info(f"Saving final discriminator model to: {final_disc_path}")
            discriminator.save(final_disc_path)
            
            # GAN model saving for final state is not explicitly requested by config keys,
            # but could be added here if needed, similar to generator/discriminator.
            self.logger.info(f"Final models saved for epoch {epoch}.")

        else: # Intermediate checkpoint saving (currently not called by train loop but logic kept)
            # For intermediate saves, always use the models_dir
            os.makedirs(models_dir, exist_ok=True) # Ensure models_dir exists
            self.logger.info(f"Ensured directory for intermediate checkpoint: {models_dir}")

            gen_template = self.params.get("save_generator_epoch_template", "generator_epoch_{epoch}.keras")
            disc_template = self.params.get("save_discriminator_epoch_template", "discriminator_epoch_{epoch}.keras")
            gan_template = self.params.get("save_gan_epoch_template", "gan_epoch_{epoch}.keras")

            generator_save_path = os.path.join(models_dir, gen_template.format(epoch=epoch))
            discriminator_save_path = os.path.join(models_dir, disc_template.format(epoch=epoch))
            gan_save_path = os.path.join(models_dir, gan_template.format(epoch=epoch))
            
            self.logger.info(f"Saving checkpoint models for epoch {epoch} to {models_dir}...")
            generator.save(generator_save_path)
            discriminator.save(discriminator_save_path)
            gan_model.save(gan_save_path) # Save the combined GAN model as well
            self.logger.info(f"Checkpoint models saved for epoch {epoch} to {models_dir}")

    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information."""
        return {
            "current_epoch": self.current_epoch,
            "training_history_length": len(self.training_history['epochs']),
            "last_generator_loss": self.training_history['generator_losses'][-1] if self.training_history['generator_losses'] else None,
            "last_discriminator_loss": self.training_history['discriminator_losses'][-1] if self.training_history['discriminator_losses'] else None
        }
