#!/usr/bin/env python3
"""
Discriminator Plugin for VAE-GAN System

Implements a discriminator that evaluates the quality of synthetic vs real time series data.
The discriminator takes full 57-feature sequences and outputs binary classification (real/fake).

Author: TimeSeries-GAN Team
"""

import sys
import traceback
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import (Input, Dense, LSTM, Conv1D, Bidirectional, 
                                     Concatenate, Flatten, LeakyReLU, ReLU) # Removed BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from typing import Dict, Any, List, Optional, Tuple
import logging


class DiscriminatorPlugin:
    """
    Discriminator plugin for distinguishing real vs synthetic time series data.
    
    Architecture:
    - Input: (batch_size, sequence_length, num_features) where num_features = 51 (aligned)
    - Conv1D layers for feature extraction
    - Bidirectional LSTM for temporal pattern recognition
    - Dense layers for classification
    - Output: (batch_size, 1) probability of being real data
    """
    
    plugin_params = {
        # Input configuration
        "sequence_length": 144,  # Standard sequence length from REFERENCE.md
        "num_features": 51,      # Aligned to 51 features
        "feature_names": [],     # Will be populated from config
        
        # Architecture parameters
        "conv_filters": [64, 128, 256],  # Progressive feature extraction
        "conv_kernel_sizes": [7, 5, 3],  # Multi-scale temporal patterns
        "conv_strides": [1, 1, 1],       # Stride for each conv layer
        "conv_activation": "leaky_relu",
        
        # LSTM configuration
        "lstm_units": 128,
        "use_bidirectional_lstm": True,
        
        # Dense layers
        "dense_units": [64, 32],
        "final_activation": "sigmoid",
        
        # Training parameters
        "learning_rate": 1e-4,
        "beta_1": 0.5,           # Adam optimizer beta_1
        "beta_2": 0.999,         # Adam optimizer beta_2
        "loss_function": "binary_crossentropy",
        "metrics": ["accuracy"],
        "leaky_relu_alpha": 0.2,
        "label_smoothing": 0.1,
        "model_save_path": None,
        "load_pretrained_model": False,
        "pretrained_model_path": None,
    }
    
    plugin_debug_vars = [
        "sequence_length", "num_features", "conv_filters", "lstm_units", 
        "learning_rate", "use_bidirectional_lstm", "dense_units"
    ]
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the DiscriminatorPlugin.
        
        Args:
            config: Configuration dictionary
        """
        if config is None:
            raise ValueError("Configuration dictionary ('config') is required.")
        
        # Initialize parameters
        self.params = self.plugin_params.copy()
        self.main_config = config.copy()
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize core attributes
        self.model: Optional[Model] = None
        self.compiled: bool = False
        
        # Update parameters from config
        for key, value in config.items():
            if key in self.plugin_params:
                self.params[key] = value
            elif key.startswith('discriminator_'):
                # Handle prefixed parameters
                param_key = key.replace('discriminator_', '')
                if param_key in self.plugin_params:
                    self.params[param_key] = value
        
        # Build model if we have sufficient configuration
        if self.params.get("num_features") and self.params.get("sequence_length"):
            self._build_model()
    
    def set_params(self, **kwargs) -> None:
        """
        Update plugin parameters and rebuild model if needed.
        
        Args:
            **kwargs: Parameter updates
        """
        print(f"DiscriminatorPlugin.set_params called with kwargs: {list(kwargs.keys())}")
        
        # Store old architecture values for change detection
        old_num_features = self.params.get("num_features")
        old_sequence_length = self.params.get("sequence_length")
        old_lstm_units = self.params.get("lstm_units")
        
        # Update main config
        if hasattr(self, 'main_config') and self.main_config is not None:
            self.main_config.update(kwargs)
        else:
            self.main_config = kwargs.copy()
        
        # Update plugin parameters (handle both prefixed and non-prefixed)
        for param_key in self.plugin_params.keys():
            prefixed_key = f"discriminator_{param_key}"
            
            if prefixed_key in kwargs:
                self.params[param_key] = kwargs[prefixed_key]
            elif param_key in kwargs:
                self.params[param_key] = kwargs[param_key]
        
        # Check if architecture changed and rebuild if needed
        if (self.params.get("num_features") != old_num_features or
            self.params.get("sequence_length") != old_sequence_length or
            self.params.get("lstm_units") != old_lstm_units):
            self.logger.info("DiscriminatorPlugin: Architecture parameters changed. Rebuilding model...")
            self._build_model()
        elif self.model is None:
            self.logger.info("DiscriminatorPlugin: No model exists. Building model...")
            self._build_model()
    
    def _build_model(self) -> Model:
        """Builds the discriminator model."""
        seq_len = self.params.get("sequence_length")
        num_features = self.params.get("num_features")
        conv_filters = self.params.get("conv_filters", [64, 128])
        conv_kernel_sizes = self.params.get("conv_kernel_sizes", [7,5,3]) # Allow multiple kernel sizes
        conv_strides = self.params.get("conv_strides", [1,1,1]) # Allow multiple strides
        conv_activation = self.params.get("conv_activation", "leaky_relu")
        
        lstm_units = self.params.get("lstm_units", 128)
        use_bidirectional_lstm = self.params.get("use_bidirectional_lstm", True)
        
        dense_units = self.params.get("dense_units", [64, 32])
        final_activation = self.params.get("final_activation", "sigmoid")
        # leaky_relu_alpha = self.params.get("leaky_relu_alpha", 0.2) # Used by LeakyReLU layer directly

        input_layer = Input(shape=(seq_len, num_features))
        x = input_layer

        # Convolutional layers
        for i, filters in enumerate(conv_filters):
            kernel_size = conv_kernel_sizes[i] if i < len(conv_kernel_sizes) else conv_kernel_sizes[-1]
            strides = conv_strides[i] if i < len(conv_strides) else conv_strides[-1]
            x = Conv1D(
                filters=filters, 
                kernel_size=kernel_size, 
                strides=strides, 
                padding="same"
                # No kernel_regularizer or bias_regularizer
            )(x)
            if conv_activation == "leaky_relu":
                x = LeakyReLU(negative_slope=self.params.get("leaky_relu_alpha", 0.2))(x)
            else:
                x = tf.keras.layers.Activation(conv_activation)(x)
            # No BatchNormalization
            # No Dropout

        # LSTM layer
        if use_bidirectional_lstm:
            x = Bidirectional(LSTM(lstm_units))(x) # No dropout, recurrent_dropout, regularizers
        else:
            x = LSTM(lstm_units)(x) # No dropout, recurrent_dropout, regularizers
        # No Dropout after LSTM

        # Dense layers
        for units in dense_units:
            x = Dense(units)(x) # No regularizers
            if conv_activation == "leaky_relu": # Assuming same activation for dense for consistency
                x = LeakyReLU(negative_slope=self.params.get("leaky_relu_alpha", 0.2))(x)
            else:
                x = tf.keras.layers.Activation(conv_activation)(x) # Or a different dense_activation param
            # No Dropout

        output_layer = Dense(1, activation=final_activation)(x)
        
        model = Model(input_layer, output_layer, name="discriminator")
        self.logger.info("Discriminator model built.") # Changed from self.add_debug_info
        return model

    def _compile_model(self) -> None:
        if self.model is None:
            self.model = self._build_model()
        
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.params.get("learning_rate"),
            beta_1=self.params.get("beta_1") 
            # beta_2 is also a common Adam param, ensure it\'s in defaults if used
        )
        self.model.compile(
            optimizer=optimizer,
            loss=self.params.get("loss_function"), # e.g., \'binary_crossentropy\'
            metrics=self.params.get("metrics", ["accuracy"])
        )
        self.logger.info("Discriminator model compiled.") # Changed from self.add_debug_info

    def get_model(self) -> Optional[Model]:
        """
        Get the discriminator model.
        
        Returns:
            Optional[Model]: The discriminator model if available, None otherwise
        """
        if self.model is None:
            self.logger.warning("Discriminator model not built. Building model now...")
            try:
                self._build_model()
            except Exception as e:
                self.logger.error(f"Failed to build discriminator model: {e}")
                return None
        return self.model
    
    def build_model(self) -> None:
        """Public interface for building the discriminator model."""
        self._build_model()
    
    def predict(self, sequences: np.ndarray) -> np.ndarray:
        """
        Predict real/fake probabilities for input sequences.
        
        Args:
            sequences: Input sequences with shape (batch_size, sequence_length, num_features)
            
        Returns:
            np.ndarray: Predictions with shape (batch_size, 1)
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        return self.model.predict(sequences)
    
    def train_on_batch(self, real_sequences: np.ndarray, fake_sequences: np.ndarray, 
                       label_smoothing: Optional[float] = None) -> Dict[str, float]:
        """
        Train discriminator on a batch of real and fake sequences.
        
        Args:
            real_sequences: Real sequences with shape (batch_size, sequence_length, num_features)
            fake_sequences: Fake sequences with same shape
            label_smoothing: Optional label smoothing factor
            
        Returns:
            Dict containing training metrics
        """
        if self.model is None or not self.compiled:
            raise ValueError("Model not built or compiled. Call build_model() first.")
        
        batch_size = real_sequences.shape[0]
        
        # Prepare labels
        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))
        
        # Apply label smoothing if specified
        if label_smoothing is not None:
            real_labels = real_labels - label_smoothing * np.random.random(real_labels.shape)
            fake_labels = fake_labels + label_smoothing * np.random.random(fake_labels.shape)
        
        # Train on real data
        real_loss = self.model.train_on_batch(real_sequences, real_labels)
        
        # Train on fake data
        fake_loss = self.model.train_on_batch(fake_sequences, fake_labels)
        
        # Calculate average metrics
        if isinstance(real_loss, list):
            metrics = {
                "discriminator_loss": (real_loss[0] + fake_loss[0]) / 2,
                "discriminator_accuracy": (real_loss[1] + fake_loss[1]) / 2
            }
        else:
            metrics = {"discriminator_loss": (real_loss + fake_loss) / 2}
        
        return metrics
    
    def evaluate_sequences(self, real_sequences: np.ndarray, fake_sequences: np.ndarray) -> Dict[str, float]:
        """
        Evaluate discriminator performance on real and fake sequences.
        
        Args:
            real_sequences: Real sequences 
            fake_sequences: Fake sequences
            
        Returns:
            Dict containing evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        batch_size = real_sequences.shape[0]
        
        # Prepare labels
        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))
        
        # Evaluate on real data
        real_metrics = self.model.evaluate(real_sequences, real_labels, verbose=0)
        
        # Evaluate on fake data  
        fake_metrics = self.model.evaluate(fake_sequences, fake_labels, verbose=0)
        
        # Calculate average metrics
        if isinstance(real_metrics, list):
            return {
                "real_loss": real_metrics[0],
                "real_accuracy": real_metrics[1] if len(real_metrics) > 1 else None,
                "fake_loss": fake_metrics[0],
                "fake_accuracy": fake_metrics[1] if len(fake_metrics) > 1 else None,
                "avg_loss": (real_metrics[0] + fake_metrics[0]) / 2,
                "avg_accuracy": (real_metrics[1] + fake_metrics[1]) / 2 if len(real_metrics) > 1 else None
            }
        else:
            return {
                "real_loss": real_metrics,
                "fake_loss": fake_metrics,
                "avg_loss": (real_metrics + fake_metrics) / 2
            }
    
    def save_model(self, filepath: str) -> None:
        """
        Save the discriminator model to file.
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Call build_model() first.")
        
        self.model.save(filepath)
        self.logger.info(f"Discriminator model saved to {filepath}")
    
    def load_model(self) -> None:
        """
        Load a pre-trained discriminator model.
        """
        pretrained_path = self.params.get("pretrained_model_path")
        if not pretrained_path:
            self.logger.warning("No pretrained model path specified")
            return
        
        try:
            from tensorflow.keras.models import load_model
            self.model = load_model(pretrained_path)
            self.compiled = True
            self.logger.info(f"Discriminator model loaded from {pretrained_path}")
        except Exception as e:
            self.logger.error(f"Failed to load discriminator model: {e}")
            raise
    
    def get_debug_info(self) -> Dict[str, Any]:
        """
        Get debug information about the discriminator.
        
        Returns:
            Dict containing debug information
        """
        debug_info = {}
        
        for var in self.plugin_debug_vars:
            if var in self.params:
                debug_info[f"discriminator_{var}"] = self.params[var]
        
        if self.model is not None:
            debug_info["discriminator_model_built"] = True
            debug_info["discriminator_model_compiled"] = self.compiled
            debug_info["discriminator_total_params"] = self.model.count_params()
        else:
            debug_info["discriminator_model_built"] = False
        
        return debug_info
    
    def add_debug_info(self, debug_dict: Dict[str, Any]) -> None:
        """
        Add discriminator debug information to provided dictionary.
        
        Args:
            debug_dict: Dictionary to add debug info to
        """
        debug_dict.update(self.get_debug_info())
