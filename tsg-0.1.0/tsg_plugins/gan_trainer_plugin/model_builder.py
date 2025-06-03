#!/usr/bin/env python3
"""
Model Builder Module

This module handles the construction of discriminator and GAN models,
providing focused functionality for model architecture building.
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Conv1D, Dropout, LeakyReLU, 
    Flatten, BatchNormalization, GlobalAveragePooling1D
)
from tensorflow.keras.optimizers import Adam
import logging
from typing import Any, Dict, Optional


class ModelBuilder:
    """Builds discriminator and GAN models with configurable architectures."""
    
    def __init__(self, params: Dict[str, Any], logger: logging.Logger):
        """Initialize model builder."""
        self.params = params
        self.logger = logger
        
        # Model references
        self.discriminator = None
        self.gan_model = None
        
        self.logger.info("ModelBuilder initialized")
    
    def build_discriminator(self, generator: tf.keras.Model, seq_len: int, 
                          num_features: int) -> tf.keras.Model:
        """
        Build discriminator model.
        
        Args:
            generator: Generator model for reference
            seq_len: Sequence length
            num_features: Number of input features
        
        Returns:
            Compiled discriminator model
        """
        self.logger.info(f"Building discriminator with seq_len={seq_len}, num_features={num_features}")
        
        try:
            # Input layer
            input_layer = Input(shape=(seq_len, num_features), name="discriminator_input")
            
            # Convolutional layers
            x = input_layer
            conv_filters = self.params.get("discriminator_conv_filters", [64, 128])
            kernel_size = self.params.get("discriminator_conv_kernel_size", 3)
            dropout_rate = self.params.get("discriminator_dropout_rate", 0.3)
            
            for i, filters in enumerate(conv_filters):
                x = Conv1D(
                    filters=filters,
                    kernel_size=kernel_size,
                    padding='same',
                    name=f"conv1d_{i+1}"
                )(x)
                x = LeakyReLU(alpha=0.2, name=f"leaky_relu_{i+1}")(x)
                x = BatchNormalization(name=f"batch_norm_{i+1}")(x)
                x = Dropout(dropout_rate, name=f"dropout_{i+1}")(x)
            
            # LSTM layer
            lstm_units = self.params.get("discriminator_lstm_units", 64)
            x = LSTM(
                units=lstm_units,
                return_sequences=False,
                name="lstm_layer"
            )(x)
            x = Dropout(dropout_rate, name="lstm_dropout")(x)
            
            # Dense layers
            x = Dense(64, activation='relu', name="dense_1")(x)
            x = Dropout(dropout_rate, name="dense_dropout")(x)
            
            # Output layer - binary classification (real/fake)
            output = Dense(1, activation='sigmoid', name="discriminator_output")(x)
            
            # Create model
            discriminator = Model(inputs=input_layer, outputs=output, name="discriminator")
            
            # Compile model
            discriminator.compile(
                optimizer=Adam(
                    learning_rate=self.params.get("discriminator_lr", 1e-4),
                    beta_1=self.params.get("discriminator_beta1", 0.5)
                ),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            self.discriminator = discriminator
            self.logger.info(f"Discriminator built successfully with {discriminator.count_params()} parameters")
            
            return discriminator
            
        except Exception as e:
            self.logger.error(f"Error building discriminator: {e}")
            raise
    
    def build_gan(self, generator: tf.keras.Model, discriminator: tf.keras.Model) -> tf.keras.Model:
        """
        Build combined GAN model.
        
        Args:
            generator: Pre-trained generator model
            discriminator: Discriminator model
        
        Returns:
            Combined GAN model
        """
        self.logger.info("Building combined GAN model")
        
        try:
            # Freeze discriminator weights for GAN training
            discriminator.trainable = False
            
            # Get generator input shape
            if hasattr(generator, 'input_shape'):
                if isinstance(generator.input_shape, list):
                    # Multiple inputs - use the first one (latent input)
                    input_shape = generator.input_shape[0][1:]  # Remove batch dimension
                else:
                    input_shape = generator.input_shape[1:]  # Remove batch dimension
            else:
                # Default input shape
                seq_len = self.params.get("seq_len", 18)
                latent_dim = self.params.get("latent_dim", 32)
                input_shape = (seq_len, latent_dim)
            
            # Create GAN input
            gan_input = Input(shape=input_shape, name="gan_input")
            
            # Generate fake data
            generated_data = generator(gan_input)
            
            # Get discriminator prediction on generated data
            gan_output = discriminator(generated_data)
            
            # Create combined model
            gan_model = Model(inputs=gan_input, outputs=gan_output, name="gan")
            
            # Compile GAN model
            gan_model.compile(
                optimizer=Adam(
                    learning_rate=self.params.get("generator_lr", 1e-4),
                    beta_1=self.params.get("generator_beta1", 0.5)
                ),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            self.gan_model = gan_model
            self.logger.info(f"GAN model built successfully with {gan_model.count_params()} parameters")
            
            return gan_model
            
        except Exception as e:
            self.logger.error(f"Error building GAN model: {e}")
            raise
    
    def build_simple_discriminator(self, seq_len: int, num_features: int) -> tf.keras.Model:
        """
        Build a simple discriminator for testing purposes.
        
        Args:
            seq_len: Sequence length
            num_features: Number of input features
        
        Returns:
            Simple discriminator model
        """
        self.logger.info("Building simple discriminator for testing")
        
        try:
            # Input layer
            input_layer = Input(shape=(seq_len, num_features), name="simple_discriminator_input")
            
            # Simple architecture
            x = GlobalAveragePooling1D()(input_layer)
            x = Dense(64, activation='relu')(x)
            x = Dropout(0.3)(x)
            x = Dense(32, activation='relu')(x)
            x = Dropout(0.3)(x)
            output = Dense(1, activation='sigmoid')(x)
            
            # Create and compile model
            model = Model(inputs=input_layer, outputs=output, name="simple_discriminator")
            model.compile(
                optimizer=Adam(learning_rate=1e-4),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            self.logger.info("Simple discriminator built successfully")
            return model
            
        except Exception as e:
            self.logger.error(f"Error building simple discriminator: {e}")
            raise
    
    def get_model_summary(self, model: tf.keras.Model) -> str:
        """
        Get model summary as string.
        
        Args:
            model: Keras model
        
        Returns:
            Model summary string
        """
        try:
            import io
            import sys
            
            # Capture model summary
            old_stdout = sys.stdout
            sys.stdout = summary_buffer = io.StringIO()
            model.summary()
            sys.stdout = old_stdout
            
            return summary_buffer.getvalue()
            
        except Exception as e:
            self.logger.error(f"Error getting model summary: {e}")
            return f"Error getting summary: {e}"
    
    def save_model_plots(self, models_dict: Dict[str, tf.keras.Model], plots_dir: str):
        """
        Save model architecture plots.
        
        Args:
            models_dict: Dictionary of model name to model object
            plots_dir: Directory to save plots
        """
        try:
            from tensorflow.keras.utils import plot_model
            import os
            
            for model_name, model in models_dict.items():
                if model is not None:
                    plot_path = os.path.join(plots_dir, f"{model_name}_architecture.png")
                    plot_model(
                        model,
                        to_file=plot_path,
                        show_shapes=True,
                        show_layer_names=True,
                        dpi=self.params.get("model_plot_dpi", 300)
                    )
                    self.logger.info(f"Saved {model_name} architecture plot to {plot_path}")
                    
        except Exception as e:
            self.logger.warning(f"Failed to save model plots: {e}")
    
    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information."""
        return {
            "discriminator_built": self.discriminator is not None,
            "gan_model_built": self.gan_model is not None,
            "discriminator_params": self.discriminator.count_params() if self.discriminator else 0,
            "gan_model_params": self.gan_model.count_params() if self.gan_model else 0
        }
