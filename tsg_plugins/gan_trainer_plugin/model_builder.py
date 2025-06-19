#!/usr/bin/env python3
"""
Model Builder Module

This module handles the construction of discriminator and GAN models,
providing focused functionality for model architecture building.
"""

import tensorflow as tf
import logging

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
            seq_len: Sequence length (not used if generator outputs point-wise features)
            num_features: Number of input features
        
        Returns:
            Compiled discriminator model
        """
        # Check generator output shape to determine discriminator input shape
        if hasattr(generator, 'output_shape'):
            gen_output_shape = generator.output_shape
            self.logger.info(f"Generator output shape: {gen_output_shape}")
            
            # If generator outputs point-wise features (e.g., (None, 23)), 
            # discriminator should handle point-wise input
            if len(gen_output_shape) == 2:  # (batch_size, features)
                actual_num_features = gen_output_shape[1]
                self.logger.info(f"Building discriminator for point-wise input: ({actual_num_features} features)")
                return self._build_pointwise_discriminator(actual_num_features)
            # If generator outputs sequences (e.g., (None, seq_len, features)),
            # discriminator should handle sequence input  
            elif len(gen_output_shape) == 3:  # (batch_size, seq_len, features)
                actual_seq_len = gen_output_shape[1]
                actual_num_features = gen_output_shape[2]
                self.logger.info(f"Building discriminator for sequence input: ({actual_seq_len}, {actual_num_features})")
                return self._build_sequence_discriminator(actual_seq_len, actual_num_features)
        
        # Fallback to original sequence-based discriminator
        self.logger.info(f"Building discriminator with seq_len={seq_len}, num_features={num_features}")
        return self._build_sequence_discriminator(seq_len, num_features)
    
    def _build_pointwise_discriminator(self, num_features: int) -> tf.keras.Model:
        """
        Build discriminator for point-wise input (e.g., (None, 23)).
        
        Args:
            num_features: Number of input features
            
        Returns:
            Compiled discriminator model
        """
        self.logger.info(f"Building point-wise discriminator with {num_features} features")
        
        try:
            # Input layer for point-wise features
            input_layer = Input(shape=(num_features,), name="discriminator_input")
            
            # Dense layers for point-wise classification
            dropout_rate = self.params.get("discriminator_dropout_rate", 0.3)
            
            x = Dense(128, activation='relu', name="dense_1")(input_layer)
            x = Dropout(dropout_rate, name="dropout_1")(x)
            x = BatchNormalization(name="batch_norm_1")(x)
            
            x = Dense(64, activation='relu', name="dense_2")(x)
            x = Dropout(dropout_rate, name="dropout_2")(x)
            x = BatchNormalization(name="batch_norm_2")(x)
            
            x = Dense(32, activation='relu', name="dense_3")(x)
            x = Dropout(dropout_rate, name="dropout_3")(x)
            
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
            self.logger.info(f"Point-wise discriminator built successfully with {discriminator.count_params()} parameters")
            
            return discriminator
            
        except Exception as e:
            self.logger.error(f"Error building point-wise discriminator: {e}")
            raise
    
    def _build_sequence_discriminator(self, seq_len: int, num_features: int) -> tf.keras.Model:
        """
        Build discriminator for sequence input (e.g., (None, seq_len, features)).
        
        Args:
            seq_len: Sequence length
            num_features: Number of input features
            
        Returns:
            Compiled discriminator model
        """
        self.logger.info(f"Building sequence discriminator with seq_len={seq_len}, num_features={num_features}")
        
        try:
            # Input layer
            input_layer = Input(shape=(seq_len, num_features), name="discriminator_input")
            
            # Convolutional layers with proper dimensionality reduction
            x = input_layer
            conv_filters = self.params.get("discriminator_conv_filters", [32, 16, 8])  # Decreasing filters
            conv_strides = self.params.get("discriminator_conv_strides", [2, 2, 2])    # Stride=2 for downsampling
            kernel_size = self.params.get("discriminator_conv_kernel_size", 5)
            dropout_rate = self.params.get("discriminator_dropout_rate", 0.3)
            
            self.logger.info(f"Building discriminator with decreasing conv filters: {conv_filters}")
            self.logger.info(f"Using conv strides for downsampling: {conv_strides}")
            
            for i, (filters, stride) in enumerate(zip(conv_filters, conv_strides)):
                x = Conv1D(
                    filters=filters,
                    kernel_size=kernel_size,
                    strides=stride,  # Add stride for downsampling
                    padding='same',
                    name=f"conv1d_{i+1}"
                )(x)
                x = LeakyReLU(alpha=0.2, name=f"leaky_relu_conv_{i+1}")(x)
                x = BatchNormalization(name=f"batch_norm_conv_{i+1}")(x)
                x = Dropout(dropout_rate, name=f"dropout_conv_{i+1}")(x)
                
                # Log the shape after each conv layer
                self.logger.debug(f"After conv1d_{i+1}: expected output shape with {filters} filters, stride {stride}")
            
            # Bidirectional LSTM layer for sequence processing
            lstm_units = self.params.get("discriminator_lstm_units", 32)
            x = tf.keras.layers.Bidirectional(
                LSTM(
                    units=lstm_units,
                    return_sequences=False,  # Return only the last output
                    name="discriminator_lstm"
                ),
                name="bidirectional_lstm"
            )(x)
            x = Dropout(dropout_rate, name="lstm_dropout")(x)
            
            # Dense layers with decreasing units
            dense_units = self.params.get("discriminator_dense_units", [16, 8])
            for i, units in enumerate(dense_units):
                x = Dense(units, activation='relu', name=f"dense_{i+1}")(x)
                x = LeakyReLU(alpha=0.2, name=f"leaky_relu_dense_{i+1}")(x)
                x = Dropout(dropout_rate, name=f"dropout_dense_{i+1}")(x)
            
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
            generator: Pre-trained generator model (GeneratorPlugin output)
            discriminator: Discriminator model
        
        Returns:
            Combined GAN model
        """
        self.logger.info("Building combined GAN model")
        self.logger.info(f"Generator input spec: {[input.shape for input in generator.inputs] if generator.inputs else 'No inputs'}")
        self.logger.info(f"Generator output spec: {generator.output.shape if hasattr(generator, 'output') else 'No output'}")
        self.logger.info(f"Discriminator input spec: {discriminator.input.shape if hasattr(discriminator, 'input') else 'No input'}")
        
        try:
            # Freeze discriminator weights for GAN training
            discriminator.trainable = False
            
            # The generator (GeneratorPlugin) should output sequences that match discriminator input
            # According to REFERENCE.md, the final output should be 57-feature sequences
            # The generator already has the correct inputs/outputs structure
            
            # Use generator's actual input structure
            gan_inputs = generator.inputs
            
            # Generate fake data using generator
            generated_data = generator(gan_inputs)
            
            # Get discriminator prediction on generated data
            gan_output = discriminator(generated_data)
            
            # Create combined model with generator's inputs
            gan_model = Model(
                inputs=gan_inputs, 
                outputs=gan_output, 
                name="gan"
            )
            
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
