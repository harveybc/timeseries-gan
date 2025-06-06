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
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Dropout, LeakyReLU, BatchNormalization, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from typing import Dict, Any, List, Optional, Tuple
import logging


class DiscriminatorPlugin:
    """
    Discriminator plugin for distinguishing real vs synthetic time series data.
    
    Architecture:
    - Input: (batch_size, sequence_length, num_features) where num_features = 57
    - Conv1D layers for feature extraction
    - Bidirectional LSTM for temporal pattern recognition
    - Dense layers for classification
    - Output: (batch_size, 1) probability of being real data
    """
    
    plugin_params = {
        # Input configuration
        "sequence_length": 144,  # Standard sequence length
        "num_features": 57,      # Full feature set size
        "feature_names": [],     # Will be populated from config
        
        # Architecture parameters
        "conv_filters": [64, 128, 256],  # Progressive feature extraction
        "conv_kernel_sizes": [7, 5, 3],  # Multi-scale temporal patterns
        "conv_strides": [1, 1, 1],       # Stride for each conv layer
        "conv_activation": "leaky_relu",
        "conv_dropout_rate": 0.3,
        
        # LSTM configuration
        "lstm_units": 128,
        "lstm_dropout": 0.3,
        "lstm_recurrent_dropout": 0.3,
        "use_bidirectional_lstm": True,
        
        # Dense layers
        "dense_units": [64, 32],
        "dense_dropout_rate": 0.5,
        "final_activation": "sigmoid",
        
        # Training parameters
        "learning_rate": 1e-4,
        "beta_1": 0.5,           # Adam optimizer beta_1
        "beta_2": 0.999,         # Adam optimizer beta_2
        "loss_function": "binary_crossentropy",
        "metrics": ["accuracy"],
        "use_batch_normalization": True,
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
    
    def _build_model(self) -> None:
        """
        Build the discriminator model architecture.
        """
        try:
            self.logger.info("Building discriminator model...")
            
            sequence_length = self.params.get("sequence_length", 144)
            num_features = self.params.get("num_features", 57)
            
            # Input layer
            input_layer = Input(shape=(sequence_length, num_features), name="discriminator_input")
            x = input_layer
            
            # Conv1D layers for feature extraction
            conv_filters = self.params.get("conv_filters", [64, 128, 256])
            conv_kernel_sizes = self.params.get("conv_kernel_sizes", [7, 5, 3])
            conv_dropout_rate = self.params.get("conv_dropout_rate", 0.3)
            leaky_relu_alpha = self.params.get("leaky_relu_alpha", 0.2)
            use_batch_norm = self.params.get("use_batch_normalization", True)
            
            for i, (filters, kernel_size) in enumerate(zip(conv_filters, conv_kernel_sizes)):
                x = Conv1D(filters, kernel_size, padding='same', name=f"conv1d_{i+1}")(x)
                if use_batch_norm:
                    x = BatchNormalization(name=f"bn_conv_{i+1}")(x)
                x = LeakyReLU(alpha=leaky_relu_alpha, name=f"leaky_relu_conv_{i+1}")(x)
                x = Dropout(conv_dropout_rate, name=f"dropout_conv_{i+1}")(x)
            
            # LSTM layer for temporal patterns
            lstm_units = self.params.get("lstm_units", 128)
            lstm_dropout = self.params.get("lstm_dropout", 0.3)
            lstm_recurrent_dropout = self.params.get("lstm_recurrent_dropout", 0.3)
            use_bidirectional = self.params.get("use_bidirectional_lstm", True)
            
            if use_bidirectional:
                x = Bidirectional(
                    LSTM(lstm_units, 
                         dropout=lstm_dropout,
                         recurrent_dropout=lstm_recurrent_dropout,
                         return_sequences=False),
                    name="bidirectional_lstm"
                )(x)
            else:
                x = LSTM(lstm_units,
                        dropout=lstm_dropout,
                        recurrent_dropout=lstm_recurrent_dropout,
                        return_sequences=False,
                        name="lstm")(x)
            
            # Dense layers for classification
            dense_units = self.params.get("dense_units", [64, 32])
            dense_dropout_rate = self.params.get("dense_dropout_rate", 0.5)
            
            for i, units in enumerate(dense_units):
                x = Dense(units, name=f"dense_{i+1}")(x)
                if use_batch_norm:
                    x = BatchNormalization(name=f"bn_dense_{i+1}")(x)
                x = LeakyReLU(alpha=leaky_relu_alpha, name=f"leaky_relu_dense_{i+1}")(x)
                x = Dropout(dense_dropout_rate, name=f"dropout_dense_{i+1}")(x)
            
            # Output layer
            final_activation = self.params.get("final_activation", "sigmoid")
            output = Dense(1, activation=final_activation, name="discriminator_output")(x)
            
            # Create model
            self.model = Model(inputs=input_layer, outputs=output, name="discriminator")
            
            self.logger.info(f"Discriminator model built successfully with {self.model.count_params()} parameters")
            self.logger.info(f"Input shape: {self.model.input.shape}")
            self.logger.info(f"Output shape: {self.model.output.shape}")
            
            # Compile the model
            self._compile_model()
            
        except Exception as e:
            self.logger.error(f"Error building discriminator model: {e}")
            self.logger.error(traceback.format_exc())
            self.model = None
            raise
    
    def _compile_model(self) -> None:
        """
        Compile the discriminator model with optimizer and loss function.
        """
        if self.model is None:
            self.logger.warning("No model to compile")
            return
        
        try:
            learning_rate = self.params.get("learning_rate", 1e-4)
            beta_1 = self.params.get("beta_1", 0.5)
            beta_2 = self.params.get("beta_2", 0.999)
            loss_function = self.params.get("loss_function", "binary_crossentropy")
            metrics = self.params.get("metrics", ["accuracy"])
            
            optimizer = Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)
            
            self.model.compile(
                optimizer=optimizer,
                loss=loss_function,
                metrics=metrics
            )
            
            self.compiled = True
            self.logger.info("Discriminator model compiled successfully")
            
        except Exception as e:
            self.logger.error(f"Error compiling discriminator model: {e}")
            self.compiled = False
            raise
    
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
