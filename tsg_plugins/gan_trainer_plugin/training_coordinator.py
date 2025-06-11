#!/usr/bin/env python3
"""
Training Coordinator Module

This module handles the core GAN training orchestration, managing the training loop,
loss calculations, and coordination between generator and discriminator training.
"""

import logging
import time
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping # Ensure ReduceLROnPlateau is imported

from app.utils.logging_utils import get_logger # Corrected import
from app.utils.output_manager import OutputManager # Changed import

logger = get_logger(__name__)

class TrainingCoordinator:
    """Coordinates GAN training process with proper orchestration."""
    
    def __init__(self, main_config, plugin_params, generator_plugin, discriminator_plugin, feeder_plugin, device):
        self.main_config = main_config
        self.params = plugin_params  # These are resolved parameters from GANTrainerPlugin
        self.generator_plugin = generator_plugin
        self.discriminator_plugin = discriminator_plugin
        self.feeder_plugin = feeder_plugin
        self.device = device
        self.current_epoch = 0
        self.stop_training = False # For EarlyStopping

        # Instantiate OutputManager
        self.output_manager = OutputManager(self.main_config)


        # Optimizers will be set by GANTrainerPlugin instance directly
        self.g_optimizer = None
        self.d_optimizer = None
        self.gan_optimizer_g = None # For the combined GAN model's generator part
        self.gan_optimizer_d = None # For the combined GAN model's discriminator part

        # Models will be passed in the train method
        self.gan_model = None
        self.generator_model = None
        self.discriminator_model = None
        
        # Callbacks will be passed in the train method
        self.callbacks = []

        # For ReduceLROnPlateau and EarlyStopping, GANTrainerPlugin will create and manage them.
        # TrainingCoordinator just needs to call their hooks.
        # self.reduce_lr_g = None # Managed by GANTrainerPlugin
        # self.reduce_lr_d = None # Managed by GANTrainerPlugin
        # self.early_stopping = None # Managed by GANTrainerPlugin

        # Ensure output directories exist
        self.output_dir = self.params.get('output_dir', 'results/')
        self.output_manager._ensure_directory_exists(self.output_dir) # Changed call
        self.checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        self.output_manager._ensure_directory_exists(self.checkpoint_dir) # Changed call
        
        logger.info("TrainingCoordinator initialized.")
        logger.debug(f"TrainingCoordinator params: {self.params}")

    def _setup_optimizers(self):
        """
        Sets up the optimizers for the generator and discriminator.
        This method is called by GANTrainerPlugin after it resolves parameters.
        Or, GANTrainerPlugin might directly assign optimizer instances.
        This method ensures optimizers are created if not already assigned.
        """
        # Default Adam parameters from Keras if not specified
        default_beta1 = 0.9 
        default_beta2 = 0.999
        default_epsilon = 1e-7
        default_amsgrad = False

        # Generator Optimizer
        gen_lr = self.params.get("generator_lr", self.params.get("learning_rate", 1e-4)) # Fallback to general LR
        gen_beta1 = self.params.get("generator_beta1", self.params.get("beta1", default_beta1))
        gen_beta2 = self.params.get("generator_beta2", self.params.get("beta2", default_beta2))
        gen_epsilon = self.params.get("generator_epsilon", self.params.get("epsilon", default_epsilon))
        gen_amsgrad = self.params.get("generator_amsgrad", self.params.get("amsgrad", default_amsgrad))

        logger.info(f"Setting up Generator Optimizer with: LR={gen_lr}, Beta1={gen_beta1}, Beta2={gen_beta2}, Epsilon={gen_epsilon}, AMSGrad={gen_amsgrad}")
        self.g_optimizer = Adam(
            learning_rate=gen_lr,
            beta_1=gen_beta1,
            beta_2=gen_beta2,
            epsilon=gen_epsilon,
            amsgrad=gen_amsgrad
        )
        
        # Discriminator Optimizer
        disc_lr = self.params.get("discriminator_lr", self.params.get("learning_rate", 1e-4)) # Fallback to general LR
        disc_beta1 = self.params.get("discriminator_beta1", self.params.get("beta1", default_beta1))
        disc_beta2 = self.params.get("discriminator_beta2", self.params.get("beta2", default_beta2))
        disc_epsilon = self.params.get("discriminator_epsilon", self.params.get("epsilon", default_epsilon))
        disc_amsgrad = self.params.get("discriminator_amsgrad", self.params.get("amsgrad", default_amsgrad))

        logger.info(f"Setting up Discriminator Optimizer with: LR={disc_lr}, Beta1={disc_beta1}, Beta2={disc_beta2}, Epsilon={disc_epsilon}, AMSGrad={disc_amsgrad}")
        self.d_optimizer = Adam(
            learning_rate=disc_lr,
            beta_1=disc_beta1,
            beta_2=disc_beta2,
            epsilon=disc_epsilon,
            amsgrad=disc_amsgrad
        )
        
        # Ensure GAN optimizers are also aligned if the GAN model uses separate optimizer instances
        # In the current setup, GANTrainerPlugin compiles GAN with the same g_optimizer_instance and d_optimizer_instance
        if hasattr(self.gan_model, 'optimizer') and isinstance(self.gan_model.optimizer, dict):
            if 'generator_optimizer' in self.gan_model.optimizer:
                self.gan_optimizer_g = self.gan_model.optimizer['generator_optimizer']
            if 'discriminator_optimizer' in self.gan_model.optimizer:
                self.gan_optimizer_d = self.gan_model.optimizer['discriminator_optimizer']
        
        logger.info(f"Optimizers configured. Initial G LR: {K.get_value(self.g_optimizer.learning_rate if self.g_optimizer else 'N/A')}, Initial D LR: {K.get_value(self.d_optimizer.learning_rate if self.d_optimizer else 'N/A')}")

    def train(self, gan_model, generator, discriminator, dataset, epochs, batch_size, callbacks=None):
        logger.info("Starting GAN training process...")
        self.gan_model = gan_model
        self.generator_model = generator # This is the generator Keras model
        self.discriminator_model = discriminator # This is the discriminator Keras model
        self.batch_size = batch_size # Store batch_size

        # GANTrainerPlugin is responsible for creating/configuring optimizers and assigning them
        # to self.g_optimizer, self.d_optimizer before calling this train method.
        # It also compiles the models with these optimizers.
        # Call _setup_optimizers as a fallback or for GAN-specific optimizers if needed.
        # self._setup_optimizers() # GANTrainerPlugin should have already set these.

        if self.g_optimizer is None or self.d_optimizer is None:
            logger.error("Optimizers not set in TrainingCoordinator. Aborting training.")
            return

        logger.info(f"Using G LR: {K.get_value(self.g_optimizer.learning_rate)}, D LR: {K.get_value(self.d_optimizer.learning_rate)}")

        self.callbacks = callbacks if callbacks else []
        
        # GANTrainerPlugin should have already called set_model on callbacks.
        # Example: self.reduce_lr_g.set_model(self.generator_model)
        # self.early_stopping.set_model(self.gan_model) or specific model

        # Call on_train_begin for all callbacks
        for callback in self.callbacks:
            if hasattr(callback, 'on_train_begin'):
                logger.debug(f"Calling on_train_begin for {type(callback).__name__}")
                callback.on_train_begin()

        start_epoch = self.params.get('start_epoch', 0) # For resuming training

        # Determine steps_per_epoch
        # This assumes dataset is a tf.data.Dataset. If it's a Python generator, len() might not work.
        # The FeederPlugin should ideally provide a way to get the number of steps.
        if hasattr(dataset, '__len__'): # Check if dataset has a defined length (e.g. a list or numpy array)
             # this is not reliable for tf.data.Dataset unless it's been batched and cardinality is known
            num_samples = len(dataset) * self.batch_size # Assuming dataset yields batches, this is wrong.
                                                      # Or if dataset yields individual samples, then it's num_samples = len(dataset)
            # Let's assume feeder plugin provides num_samples or steps_per_epoch in params for now
            # For tf.data.Dataset, it's better to iterate until it's exhausted or use cardinality.
            # Using tf.data.experimental.cardinality if available and dataset is a tf.data.Dataset
            if isinstance(dataset, tf.data.Dataset):
                cardinality = tf.data.experimental.cardinality(dataset)
                if cardinality == tf.data.experimental.INFINITE_CARDINALITY:
                    logger.warning("Dataset has infinite cardinality. 'steps_per_epoch' must be provided in parameters.")
                    self.steps_per_epoch = self.params.get('steps_per_epoch')
                    if self.steps_per_epoch is None:
                        logger.error("'steps_per_epoch' is required for infinite datasets. Aborting.")
                        return
                elif cardinality == tf.data.experimental.UNKNOWN_CARDINALITY:
                    logger.warning("Dataset has unknown cardinality. 'steps_per_epoch' should be provided in parameters for reliable epoch tracking.")
                    self.steps_per_epoch = self.params.get('steps_per_epoch')
                    if self.steps_per_epoch is None:
                        logger.warning("'steps_per_epoch' not provided for unknown cardinality dataset. Epochs will run until dataset is exhausted.")
                else: # Finite cardinality
                    self.steps_per_epoch = int(cardinality.numpy()) # Cardinality of batched dataset
                    logger.info(f"Dataset has finite cardinality: {self.steps_per_epoch} steps per epoch.")
            else: # Not a tf.data.Dataset, try to get from params or len(): # Corrected line
                self.steps_per_epoch = self.params.get('steps_per_epoch')
                if self.steps_per_epoch is None:
                    try:
                        num_samples = self.feeder_plugin.get_num_samples('train') # Assuming feeder has this
                        self.steps_per_epoch = (num_samples + self.batch_size - 1) // self.batch_size
                        logger.info(f"Calculated steps_per_epoch: {self.steps_per_epoch} from feeder num_samples: {num_samples}")
                    except AttributeError:
                        logger.warning("'steps_per_epoch' not found in params and feeder_plugin does not have get_num_samples. Training might be unpredictable.")
                        # If still None, the train_epoch loop will just run until dataset is exhausted for one pass.
        else: # dataset does not have __len__ (e.g. pure iterator not from tf.data with cardinality)
            self.steps_per_epoch = self.params.get('steps_per_epoch')
            if self.steps_per_epoch is None:
                logger.warning("Dataset has no __len__ and 'steps_per_epoch' not provided. Epochs will run until dataset is exhausted or for a default number of steps if configured elsewhere.")


        for epoch in range(start_epoch, epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            logger.info(f"Starting Epoch {epoch+1}/{epochs}")

            # Call on_epoch_begin for all callbacks
            for callback in self.callbacks:
                if hasattr(callback, 'on_epoch_begin'):
                    logger.debug(f"Calling on_epoch_begin for {type(callback).__name__} (Epoch: {epoch+1})")
                    callback.on_epoch_begin(epoch)
            
            epoch_logs = self.train_epoch(dataset, epoch, epochs, self.steps_per_epoch, self.batch_size)

            epoch_duration = time.time() - epoch_start_time
            
            # Update logs with learning rates for clarity
            if self.g_optimizer:
                epoch_logs['g_lr'] = K.get_value(self.g_optimizer.learning_rate)
            if self.d_optimizer:
                epoch_logs['d_lr'] = K.get_value(self.d_optimizer.learning_rate)
            epoch_logs['epoch_duration'] = epoch_duration

            log_message_parts = [f"Epoch {epoch+1}/{epochs} completed in {epoch_duration:.2f}s"]
            for key, value in epoch_logs.items():
                if isinstance(value, float):
                    log_message_parts.append(f"{key}: {value:.4f}")
                else:
                    log_message_parts.append(f"{key}: {value}")
            logger.info(" | ".join(log_message_parts))
            logger.debug(f"Raw epoch_logs for callbacks: {epoch_logs}")

            # Call on_epoch_end for all callbacks
            for callback in self.callbacks:
                if hasattr(callback, 'on_epoch_end'):
                    logger.debug(f"Calling on_epoch_end for {type(callback).__name__} (Epoch: {epoch+1})")
                    if isinstance(callback, ReduceLROnPlateau):
                        monitored_value = epoch_logs.get(callback.monitor)
                        optimizer_to_log = None
                        if hasattr(callback, 'model') and hasattr(callback.model, 'optimizer'):
                            optimizer_to_log = callback.model.optimizer
                        
                        logger.info(f"  ReduceLROnPlateau ({callback.monitor}):")
                        logger.info(f"    Patience: {callback.patience}, Wait: {callback.wait}, Cooldown: {callback.cooldown_counter}/{callback.cooldown}")
                        logger.info(f"    Best: {callback.best:.4f}, Current: {f'{monitored_value:.4f}' if monitored_value is not None else 'N/A'}, Mode: {callback.mode}, MinDelta: {callback.min_delta}")
                        if optimizer_to_log:
                            logger.info(f"    LR (before on_epoch_end): {K.get_value(optimizer_to_log.learning_rate):.7f}")
                        else:
                            logger.warning(f"    Could not get LR for {type(callback).__name__} as model or optimizer not found on callback.")

                    callback.on_epoch_end(epoch, logs=epoch_logs) # This is where LR might change

                    if isinstance(callback, ReduceLROnPlateau):
                        optimizer_to_log = None
                        if hasattr(callback, 'model') and hasattr(callback.model, 'optimizer'):
                            optimizer_to_log = callback.model.optimizer
                        if optimizer_to_log:
                            logger.info(f"    LR (after on_epoch_end):  {K.get_value(optimizer_to_log.learning_rate):.7f}")
                        # Check if early stopping was triggered by this callback if it's also an EarlyStopping instance (unlikely for ReduceLROnPlateau)
                        if hasattr(callback, 'stopped_epoch') and callback.stopped_epoch > 0:
                             logger.info(f"    {type(callback).__name__} indicated stopping at epoch {callback.stopped_epoch}.")
                             # self.stop_training = True # EarlyStopping callback itself handles this by setting self.model.stop_training

            # Check for early stopping (Keras EarlyStopping sets model.stop_training)
            # The GAN model might be the one EarlyStopping is attached to, or a specific sub-model.
            # GANTrainerPlugin sets the model for EarlyStopping.
            # We need a reliable way to check this. Let's assume EarlyStopping callback sets a flag on itself or the model.
            # A common pattern is `self.model.stop_training = True`.
            # The `gan_model` is the primary model for the overall training loop.
            if hasattr(self.gan_model, 'stop_training') and self.gan_model.stop_training:
                logger.info("Early stopping signal received from GAN model. Terminating training.")
                self.stop_training = True # Ensure our loop condition catches this

            if self.stop_training: # Check if any callback (like EarlyStopping) set this
                logger.info(f"Training stopped at epoch {epoch+1} due to a callback (e.g., EarlyStopping).")
                break
            
            # Checkpoint saving (moved to _save_checkpoint, called from GANTrainerPlugin or here)
            if (epoch + 1) % self.params.get("save_model_interval_epochs", 100) == 0 or (epoch + 1) == epochs : # Added save_model_interval_epochs
                logger.info(f"Saving checkpoint at epoch {epoch+1}")
                self._save_checkpoint(epoch + 1)


        # Call on_train_end for all callbacks
        for callback in self.callbacks:
            if hasattr(callback, 'on_train_end'):
                logger.debug(f"Calling on_train_end for {type(callback).__name__}")
                callback.on_train_end()

        logger.info("GAN training process finished.")

        # Save final models if not done by checkpointing logic for the last epoch
        if not self.stop_training: # Only save final if not early stopped, or handle separately
            logger.info("Saving final trained models...")
            self._save_final_models()


    def train_epoch(self, dataset, current_epoch, total_epochs, steps_per_epoch, batch_size):
        epoch_g_loss = 0
        epoch_d_loss = 0
        epoch_d_acc_real = 0
        epoch_d_acc_fake = 0
        steps_this_epoch = 0

        # dataset is expected to be an iterator or iterable yielding batches
        # e.g. a tf.data.Dataset that has been batched by the feeder plugin
        for batch_data in dataset: # Iterate over the dataset
            # If dataset yields (noise_batch, real_batch, conditional_batch), adapt _train_gan_batch
            # For now, assume batch_data is what _train_gan_batch expects (e.g., real samples)
            # and _train_gan_batch handles noise generation internally based on batch_size.
            
            # Determine actual batch size from data, if possible, or use configured batch_size
            # This is important if the last batch is smaller.
            if isinstance(batch_data, tuple) and hasattr(batch_data[0], 'shape'): # e.g. (real_samples, conditions)
                current_batch_size = tf.shape(batch_data[0])[0].numpy()
            elif hasattr(batch_data, 'shape'): # e.g. just real_samples
                current_batch_size = tf.shape(batch_data)[0].numpy()
            else: # Fallback if shape is not easily accessible (e.g. complex nested structure)
                current_batch_size = batch_size 

            if current_batch_size == 0: # Skip empty batches if any
                continue

            batch_metrics = self._train_gan_batch(batch_data, current_batch_size) # Pass current_batch_size
            
            batch_d_loss = batch_metrics.get('d_loss', 0.0)
            batch_g_loss = batch_metrics.get('g_loss', 0.0)
            batch_d_real_acc = batch_metrics.get('d_real_acc', 0.0)
            batch_d_fake_acc = batch_metrics.get('d_fake_acc', 0.0)
            
            epoch_d_loss += batch_d_loss
            epoch_g_loss += batch_g_loss
            epoch_d_acc_real += batch_d_real_acc
            epoch_d_acc_fake += batch_d_fake_acc
            steps_this_epoch += 1

            if steps_this_epoch % self.params.get("log_interval_steps", 50) == 0:
                logger.info(f"  Epoch {current_epoch+1}/{total_epochs}, Step {steps_this_epoch}/{steps_per_epoch or 'Unknown'}: "
                            f"d_loss: {batch_d_loss:.4f}, g_loss: {batch_g_loss:.4f}, "
                            f"d_acc_real: {batch_d_real_acc:.4f}, d_acc_fake: {batch_d_fake_acc:.4f}")

            if steps_per_epoch is not None and steps_this_epoch >= steps_per_epoch:
                logger.info(f"Reached specified steps_per_epoch ({steps_per_epoch}). Ending epoch.")
                break # Exit loop after completing defined steps for the epoch
        
        if steps_this_epoch == 0:
            logger.warning(f"Epoch {current_epoch+1} completed with 0 steps. Check dataset and steps_per_epoch configuration.")
            return {
                'g_loss': 0, 'd_loss': 0,
                'd_real_accuracy': 0, 'd_fake_accuracy': 0,
            }

        avg_g_loss = epoch_g_loss / steps_this_epoch
        avg_d_loss = epoch_d_loss / steps_this_epoch
        avg_d_real_acc = epoch_d_acc_real / steps_this_epoch
        avg_d_fake_acc = epoch_d_acc_fake / steps_this_epoch
        
        logs = {
            'g_loss': avg_g_loss,
            'd_loss': avg_d_loss,
            'd_real_accuracy': avg_d_real_acc,
            'd_fake_accuracy': avg_d_fake_acc,
        }
        return logs

    def _train_gan_batch(self, batch_data, current_batch_size): # Renamed batch_size to current_batch_size
        """
        Handles the training for a single batch for both Discriminator and Generator.
        This is a conceptual placeholder. The actual implementation depends on the GAN structure.
        It should use self.g_optimizer and self.d_optimizer.
        """
        # This method needs to be implemented based on the specific GAN architecture.
        # It typically involves:
        # 1. Training Discriminator:
        #    - Get real samples from batch_data.
        #    - Generate fake samples using the generator.
        #    - Calculate discriminator loss on real and fake samples.
        #    - Apply gradients to discriminator.
        # 2. Training Generator:
        #    - Generate fake samples.
        #    - Calculate generator loss based on discriminator's output for fake samples.
        #    - Apply gradients to generator.

        # Example using tf.GradientTape (if not using model.train_on_batch)
        # This assumes self.generator_model and self.discriminator_model are Keras models.
        # And batch_data is structured appropriately, e.g., (real_samples, conditional_input)
        
        # Assuming batch_data directly contains real samples or a tuple (real_samples, conditions)
        if isinstance(batch_data, tuple):
            real_samples = batch_data[0]
            # conditional_input = batch_data[1] # If conditional
        else:
            real_samples = batch_data # If only real samples are passed

        # Ensure real_samples is a Tensor
        if not isinstance(real_samples, tf.Tensor):
            try:
                real_samples = tf.convert_to_tensor(real_samples, dtype=tf.float32)
            except Exception as e:
                logger.error(f"Failed to convert real_samples to tensor: {e}")
                # Return zero losses or raise error
                return {'d_loss': 0.0, 'g_loss': 0.0, 'd_real_acc': 0.0, 'd_fake_acc': 0.0}


        # For conditional GANs, noise and conditions are fed to generator
        noise_dim = self.params.get('noise_dim', 100)
        # Use current_batch_size for noise generation
        noise = tf.random.normal([current_batch_size, noise_dim]) 
        
        # If conditional inputs are part of batch_data:
        # conditional_inputs = batch_data[1] # Example
        # generator_inputs = [noise, conditional_inputs]

        # For a simple GAN, just noise:
        generator_inputs = noise

        # --- Train Discriminator ---
        # Ensure discriminator is trainable for its update step
        if hasattr(self.discriminator_model, 'trainable'):
            self.discriminator_model.trainable = True

        d_loss = 0.0
        d_real_output_avg = 0.0
        d_fake_output_avg = 0.0

        with tf.GradientTape() as tape:
            generated_samples = self.generator_model(generator_inputs, training=True)
            
            real_output = self.discriminator_model(real_samples, training=True)
            fake_output = self.discriminator_model(generated_samples, training=True) # Use generated_samples
            
            # Example loss calculation (binary crossentropy)
            # Ensure labels are of the same dtype as outputs (usually float32)
            real_loss = tf.keras.losses.binary_crossentropy(tf.ones_like(real_output, dtype=tf.float32), real_output)
            fake_loss = tf.keras.losses.binary_crossentropy(tf.zeros_like(fake_output, dtype=tf.float32), fake_output)
            d_loss = (tf.reduce_mean(real_loss) + tf.reduce_mean(fake_loss)) / 2

        if self.discriminator_model.trainable_variables: # Check if there are trainable variables
            d_grads = tape.gradient(d_loss, self.discriminator_model.trainable_variables)
            self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator_model.trainable_variables))
        else:
            logger.warning("Discriminator has no trainable variables. Skipping D update.")


        # Calculate accuracies (conceptual)
        d_real_acc = tf.reduce_mean(tf.cast(tf.math.greater_equal(real_output, 0.5), tf.float32)).numpy()
        d_fake_acc = tf.reduce_mean(tf.cast(tf.math.less(fake_output, 0.5), tf.float32)).numpy()


        # --- Train Generator ---
        # Ensure discriminator is NOT trainable during generator update step (common practice)
        if hasattr(self.discriminator_model, 'trainable'):
            self.discriminator_model.trainable = False
        
        g_loss = 0.0

        with tf.GradientTape() as tape:
            # Noise for generator training step
            noise_g = tf.random.normal([current_batch_size, noise_dim])
            # generator_inputs_g = [noise_g, conditional_inputs] # if conditional
            generator_inputs_g = noise_g # if simple
            
            generated_samples_g = self.generator_model(generator_inputs_g, training=True)
            # We want the discriminator to think these are real
            fake_output_g = self.discriminator_model(generated_samples_g, training=False) # training=False for D here
            
            # Generator loss: tries to make discriminator output 1 for fake samples
            g_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones_like(fake_output_g, dtype=tf.float32), fake_output_g))

        if self.generator_model.trainable_variables: # Check if there are trainable variables
            g_grads = tape.gradient(g_loss, self.generator_model.trainable_variables)
            self.g_optimizer.apply_gradients(zip(g_grads, self.generator_model.trainable_variables))
        else:
            logger.warning("Generator has no trainable variables. Skipping G update.")
        
        # Restore discriminator trainability if it was changed
        if hasattr(self.discriminator_model, 'trainable'):
            self.discriminator_model.trainable = True

        return {
            'd_loss': d_loss.numpy() if isinstance(d_loss, tf.Tensor) else d_loss,
            'g_loss': g_loss.numpy() if isinstance(g_loss, tf.Tensor) else g_loss,
            'd_real_acc': d_real_acc, # Already numpy float
            'd_fake_acc': d_fake_acc  # Already numpy float
        }

    def _save_checkpoint(self, epoch):
        # Ensure checkpoint directory exists
        self.output_manager._ensure_directory_exists(self.checkpoint_dir) # Changed call

        # Save generator model
        if self.generator_model:
            g_model_path = os.path.join(self.checkpoint_dir, f"generator_epoch_{epoch}.keras")
            try:
                self.generator_model.save(g_model_path)
                logger.info(f"Generator model checkpoint saved to {g_model_path}")
            except Exception as e:
                logger.error(f"Error saving generator model checkpoint: {e}")

        # Save discriminator model
        if self.discriminator_model:
            d_model_path = os.path.join(self.checkpoint_dir, f"discriminator_epoch_{epoch}.keras")
            try:
                self.discriminator_model.save(d_model_path)
                logger.info(f"Discriminator model checkpoint saved to {d_model_path}")
            except Exception as e:
                logger.error(f"Error saving discriminator model checkpoint: {e}")
        
        # Optionally, save GAN model state if it's a custom model with its own state
        if self.gan_model and hasattr(self.gan_model, 'save_weights'): # More robust check
            gan_model_path = os.path.join(self.checkpoint_dir, f"gan_epoch_{epoch}.weights.h5") # Keras convention for weights
            try:
                self.gan_model.save_weights(gan_model_path)
                logger.info(f"GAN model weights checkpoint saved to {gan_model_path}")
            except Exception as e:
                logger.error(f"Error saving GAN model weights checkpoint: {e}")
        
        # Save optimizer states (important for resuming training)
        # This requires a bit more care, often saved alongside the model or as separate files.
        # Keras models saved with model.save() usually include optimizer state if model was compiled.
        # If using custom loops and optimizers directly, you might need to manually save/load optimizer states.
        # For now, relying on model.save() to handle this for compiled models.
        logger.debug(f"Checkpoint for epoch {epoch} completed.")


    def _save_final_models(self):
        """Saves the final generator and discriminator models."""
        logger.info("Saving final models...")

        # Generator
        save_path_g_key = "save_generator_sequential_model_file"
        default_g_path = os.path.join(self.output_dir, "final_generator_model.keras")
        save_path_g = self.main_config.get(save_path_g_key, default_g_path)
        
        # Ensure the path is absolute, using output_dir as base if it's relative
        if not os.path.isabs(save_path_g):
            save_path_g = os.path.join(self.output_dir, save_path_g)
            logger.info(f"Relative path for {save_path_g_key} provided. Saving to: {save_path_g}")

        if self.generator_model:
            try:
                self.output_manager._ensure_directory_exists(os.path.dirname(save_path_g)) # Changed call
                self.generator_model.save(save_path_g)
                logger.info(f"Final generator model saved to {save_path_g}")
            except Exception as e:
                logger.error(f"Error saving final generator model to {save_path_g}: {e}")
        else:
            logger.warning("Generator model not available to save.")

        # Discriminator
        save_path_d_key = "save_discriminator_sequential_model_file"
        default_d_path = os.path.join(self.output_dir, "final_discriminator_model.keras")
        save_path_d = self.main_config.get(save_path_d_key, default_d_path)

        if not os.path.isabs(save_path_d):
            save_path_d = os.path.join(self.output_dir, save_path_d)
            logger.info(f"Relative path for {save_path_d_key} provided. Saving to: {save_path_d}")
            
        if self.discriminator_model:
            try:
                self.output_manager._ensure_directory_exists(os.path.dirname(save_path_d)) # Changed call
                self.discriminator_model.save(save_path_d)
                logger.info(f"Final discriminator model saved to {save_path_d}")
            except Exception as e:
                logger.error(f"Error saving final discriminator model to {save_path_d}: {e}")
        else:
            logger.warning("Discriminator model not available to save.")

    def load_checkpoint(self, checkpoint_path_g, checkpoint_path_d, checkpoint_path_gan=None):
        pass # Placeholder for actual checkpoint loading logic
