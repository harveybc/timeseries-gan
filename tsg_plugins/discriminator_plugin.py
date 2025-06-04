#!/usr/bin/env python3
"""
Discriminator Plugin for VAE-GAN System

Implements a discriminator that evaluates the quality of synthetic vs real time series data.
The discriminator takes full 57-feature sequences and outputs binary classification (real/fake).

Author: TimeSeries-GAN Team
"""

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
        
        # Regularization
        "use_batch_normalization": True,
        "leaky_relu_alpha": 0.2,
        "label_smoothing": 0.1,  # For training stability
        
        # Model persistence
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
        Initialize discriminator plugin.
        
        Args:
            config: Configuration dictionary
        """
        if config is None:
            raise ValueError("Configuration dictionary ('config') is required.")
        
        # Initialize parameters
        self.params = self.plugin_params.copy()
        self.config = config.copy()
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Update parameters from config
        self.set_params(**config)
        
        # Initialize model
        self.model: Optional[Model] = None
        self.compiled = False
        
        # Build model if parameters are sufficient
        if self.params.get("sequence_length") and self.params.get("num_features"):
            self._build_model()
    
    def set_params(self, **kwargs) -> None:
        """
        Update discriminator parameters.
        
        Args:
            **kwargs: Parameter updates
        """
        # Update main config
        self.config.update(kwargs)
        
        # Update plugin parameters (handle both prefixed and non-prefixed)
        for param_key in self.plugin_params.keys():
            prefixed_key = f"discriminator_{param_key}"
            
            if prefixed_key in kwargs:
                self.params[param_key] = kwargs[prefixed_key]
            elif param_key in kwargs:
                self.params[param_key] = kwargs[param_key]
        
        # Handle special mappings
        if "generator_full_feature_names_ordered" in kwargs:
            self.params["feature_names"] = kwargs["generator_full_feature_names_ordered"]
            self.params["num_features"] = len(kwargs["generator_full_feature_names_ordered"])
        
        # Rebuild model if key parameters changed
        if any(key in kwargs for key in ["sequence_length", "num_features", "conv_filters", "lstm_units"]):
            if self.params.get("sequence_length") and self.params.get("num_features"):
                self._build_model()
    
    def _build_model(self) -> None:
        """Build the discriminator model architecture."""
        try:
            self.logger.info(f"Building discriminator model with seq_len={self.params['sequence_length']}, "
                           f"num_features={self.params['num_features']}")
            
            # Input layer
            input_layer = Input(
                shape=(self.params["sequence_length"], self.params["num_features"]),
                name="discriminator_input"
            )
            
            x = input_layer
            
            # Convolutional layers for feature extraction
            conv_filters = self.params["conv_filters"]
            conv_kernel_sizes = self.params["conv_kernel_sizes"]
            conv_strides = self.params["conv_strides"]
            
            # Ensure lists are same length
            if len(conv_kernel_sizes) != len(conv_filters):
                conv_kernel_sizes = [conv_kernel_sizes[0]] * len(conv_filters)
            if len(conv_strides) != len(conv_filters):
                conv_strides = [conv_strides[0]] * len(conv_filters)
            
            for i, (filters, kernel_size, stride) in enumerate(zip(conv_filters, conv_kernel_sizes, conv_strides)):
                x = Conv1D(
                    filters=filters,
                    kernel_size=kernel_size,
                    strides=stride,
                    padding="same",
                    name=f"conv1d_{i+1}"
                )(x)
                
                if self.params["use_batch_normalization"]:
                    x = BatchNormalization(name=f"batch_norm_{i+1}")(x)
                
                x = LeakyReLU(alpha=self.params["leaky_relu_alpha"], name=f"leaky_relu_{i+1}")(x)
                x = Dropout(self.params["conv_dropout_rate"], name=f"conv_dropout_{i+1}")(x)
            
            # LSTM layer for temporal pattern recognition
            if self.params["use_bidirectional_lstm"]:
                x = Bidirectional(
                    LSTM(
                        units=self.params["lstm_units"],
                        return_sequences=False,  # We want final output for classification
                        dropout=self.params["lstm_dropout"],
                        recurrent_dropout=self.params["lstm_recurrent_dropout"],
                        name="lstm_layer"
                    ),
                    name="bidirectional_lstm"
                )(x)
            else:
                x = LSTM(
                    units=self.params["lstm_units"],
                    return_sequences=False,
                    dropout=self.params["lstm_dropout"],
                    recurrent_dropout=self.params["lstm_recurrent_dropout"],
                    name="lstm_layer"
                )(x)
            
            # Dense layers
            dense_units = self.params["dense_units"]
            for i, units in enumerate(dense_units):
                x = Dense(units, activation="relu", name=f"dense_{i+1}")(x)
                x = Dropout(self.params["dense_dropout_rate"], name=f"dense_dropout_{i+1}")(x)
            
            # Output layer - binary classification
            output = Dense(
                1, 
                activation=self.params["final_activation"], 
                name="discriminator_output"
            )(x)
            
            # Create model
            self.model = Model(inputs=input_layer, outputs=output, name="discriminator")
            
            # Compile model
            self._compile_model()
            
            self.logger.info(f"Discriminator model built successfully with {self.model.count_params():,} parameters")
            
            # Print model summary
            if self.logger.isEnabledFor(logging.INFO):
                self.model.summary(print_fn=self.logger.info)
                
        except Exception as e:
            self.logger.error(f"Error building discriminator model: {e}")
            self.model = None
            raise
    
    def _compile_model(self) -> None:
        """Compile the discriminator model."""
        if self.model is None:
            raise RuntimeError("Model must be built before compilation")
        
        optimizer = Adam(
            learning_rate=self.params["learning_rate"],
            beta_1=self.params["beta_1"],
            beta_2=self.params["beta_2"]
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss=self.params["loss_function"],
            metrics=self.params["metrics"]
        )
        
        self.compiled = True
        self.logger.info("Discriminator model compiled successfully")
    
    def predict(self, sequences: np.ndarray) -> np.ndarray:
        """
        Predict real/fake probabilities for input sequences.
        
        Args:
            sequences: Input sequences of shape (batch_size, sequence_length, num_features)
        
        Returns:
            Probabilities of being real data, shape (batch_size, 1)
        """
        if self.model is None:
            raise RuntimeError("Discriminator model not built")
        
        if sequences.ndim != 3:
            raise ValueError(f"Expected 3D input (batch_size, seq_len, features), got shape {sequences.shape}")
        
        if sequences.shape[1] != self.params["sequence_length"]:
            raise ValueError(f"Expected sequence length {self.params['sequence_length']}, got {sequences.shape[1]}")
        
        if sequences.shape[2] != self.params["num_features"]:
            raise ValueError(f"Expected {self.params['num_features']} features, got {sequences.shape[2]}")
        
        return self.model.predict(sequences)
    
    def train_on_batch(self, real_sequences: np.ndarray, fake_sequences: np.ndarray, 
                       label_smoothing: Optional[float] = None) -> Dict[str, float]:
        """
        Train discriminator on a batch of real and fake sequences.
        
        Args:
            real_sequences: Real data sequences (batch_size, seq_len, features)
            fake_sequences: Synthetic data sequences (batch_size, seq_len, features)
            label_smoothing: Override default label smoothing
        
        Returns:
            Dictionary containing loss and accuracy metrics
        """
        if self.model is None or not self.compiled:
            raise RuntimeError("Discriminator model not built or compiled")
        
        batch_size = real_sequences.shape[0]
        if fake_sequences.shape[0] != batch_size:
            raise ValueError("Real and fake sequences must have same batch size")
        
        # Prepare training data
        x_batch = np.concatenate([real_sequences, fake_sequences], axis=0)
        
        # Prepare labels with optional smoothing
        smoothing = label_smoothing if label_smoothing is not None else self.params["label_smoothing"]
        real_labels = np.ones((batch_size, 1)) - smoothing  # Slightly less than 1
        fake_labels = np.zeros((batch_size, 1)) + smoothing  # Slightly more than 0
        y_batch = np.concatenate([real_labels, fake_labels], axis=0)
        
        # Train on batch
        loss, accuracy = self.model.train_on_batch(x_batch, y_batch)
        
        return {"loss": float(loss), "accuracy": float(accuracy)}
    
    def evaluate_sequences(self, real_sequences: np.ndarray, fake_sequences: np.ndarray) -> Dict[str, float]:
        """
        Evaluate discriminator performance on real vs fake sequences.
        
        Args:
            real_sequences: Real data sequences
            fake_sequences: Fake data sequences
        
        Returns:
            Dictionary with evaluation metrics
        """
        real_predictions = self.predict(real_sequences)
        fake_predictions = self.predict(fake_sequences)
        
        # Calculate metrics
        real_accuracy = np.mean(real_predictions > 0.5)
        fake_accuracy = np.mean(fake_predictions <= 0.5)
        overall_accuracy = (real_accuracy + fake_accuracy) / 2
        
        # Discriminative score (how well it distinguishes real from fake)
        discriminative_score = np.abs(np.mean(real_predictions) - np.mean(fake_predictions))
        
        return {
            "real_accuracy": float(real_accuracy),
            "fake_accuracy": float(fake_accuracy), 
            "overall_accuracy": float(overall_accuracy),
            "discriminative_score": float(discriminative_score),
            "mean_real_prediction": float(np.mean(real_predictions)),
            "mean_fake_prediction": float(np.mean(fake_predictions))
        }
    
    def get_model(self) -> Optional[Model]:
        """Get the discriminator model."""
        return self.model
    
    def save_model(self, filepath: str) -> None:
        """Save the discriminator model."""
        if self.model is None:
            raise RuntimeError("No model to save")
        
        self.model.save(filepath)
        self.logger.info(f"Discriminator model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a pre-trained discriminator model."""
        try:
            self.model = tf.keras.models.load_model(filepath)
            self.compiled = True
            self.logger.info(f"Discriminator model loaded from {filepath}")
        except Exception as e:
            self.logger.error(f"Error loading model from {filepath}: {e}")
            raise
    
    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information about the discriminator."""
        debug_info = {}
        
        for var_name in self.plugin_debug_vars:
            if var_name in self.params:
                debug_info[f"discriminator_{var_name}"] = self.params[var_name]
        
        if self.model is not None:
            debug_info["discriminator_model_params"] = self.model.count_params()
            debug_info["discriminator_compiled"] = self.compiled
        
        return debug_info
    
    def add_debug_info(self, debug_dict: Dict[str, Any]) -> None:
        """Add discriminator debug information to existing dictionary."""
        debug_dict.update(self.get_debug_info())
