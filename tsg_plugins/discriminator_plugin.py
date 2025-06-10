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
        # Set up logging first
        self.logger = logging.getLogger(__name__) 
        self.logger.debug(f"Initializing DiscriminatorPlugin with config: {config}")

        if config is None:
            self.logger.error("Configuration dictionary ('config') is required.")
            raise ValueError("Configuration dictionary ('config') is required.")
        
        # Initialize parameters
        self.params = self.plugin_params.copy()
        self.main_config = config.copy()
        
        # Initialize core attributes
        self.model: Optional[Model] = None
        self.compiled: bool = False
        
        # Update parameters from config
        self.logger.debug("Updating parameters from config...")
        for key, value in config.items():
            if key in self.plugin_params:
                self.params[key] = value
                self.logger.debug(f"Set self.params['{key}'] = {value}")
            elif key.startswith('discriminator_'):
                param_key = key.replace('discriminator_', '')
                if param_key in self.plugin_params:
                    self.params[param_key] = value
                    self.logger.debug(f"Set self.params['{param_key}'] = {value} (from {key})")
        
        self.logger.info(f"DiscriminatorPlugin params after init: {self.params}")

        # Build model if we have sufficient configuration
        if self.params.get("num_features") and self.params.get("sequence_length"):
            self.logger.info("Sufficient configuration found, building model during __init__.")
            self._build_model() # Call internal build method
        else:
            self.logger.warning("Insufficient configuration for initial model build (num_features or sequence_length missing).")
        self.logger.debug("DiscriminatorPlugin initialized.")
    
    def set_params(self, **kwargs) -> None:
        """
        Update plugin parameters and rebuild model if needed.
        
        Args:
            **kwargs: Parameter updates
        """
        self.logger.debug(f"DiscriminatorPlugin.set_params called with kwargs: {kwargs}")
        
        # Store old architecture values for change detection
        old_num_features = self.params.get("num_features")
        old_sequence_length = self.params.get("sequence_length")
        old_lstm_units = self.params.get("lstm_units")
        
        # Update main config
        if hasattr(self, 'main_config') and self.main_config is not None:
            self.main_config.update(kwargs)
        else:
            self.main_config = kwargs.copy()
        
        self.logger.debug(f"Discriminator params before update: {self.params}")
        # Update plugin parameters (handle both prefixed and non-prefixed)
        for param_key in self.plugin_params.keys():
            prefixed_key = f"discriminator_{param_key}"
            
            if prefixed_key in kwargs:
                self.params[param_key] = kwargs[prefixed_key]
                self.logger.debug(f"Updated self.params['{param_key}'] = {kwargs[prefixed_key]} from prefixed key {prefixed_key}")
            elif param_key in kwargs:
                self.params[param_key] = kwargs[param_key]
                self.logger.debug(f"Updated self.params['{param_key}'] = {kwargs[param_key]}")
        
        self.logger.info(f"DiscriminatorPlugin params after update in set_params: {self.params}")
        # Check if architecture changed and rebuild if needed
        architecture_changed = (
            self.params.get("num_features") != old_num_features or
            self.params.get("sequence_length") != old_sequence_length or
            self.params.get("lstm_units") != old_lstm_units
        )

        if architecture_changed:
            self.logger.info("DiscriminatorPlugin: Architecture parameters changed. Rebuilding model...")
            self._build_model()
        elif self.model is None: # If model is None but architecture didn't change, still try to build
            self.logger.info("DiscriminatorPlugin: No model exists. Building model...")
            self._build_model()
    
    def _build_model(self) -> Model: # Changed: return type hint to Model
        """Builds the discriminator model."""
        self.logger.info("Building discriminator model...")
        seq_len = self.params.get("sequence_length")
        num_features = self.params.get("num_features")
        conv_filters = self.params.get("conv_filters", [64, 128])
        conv_kernel_sizes = self.params.get("conv_kernel_sizes", [7,5,3]) 
        conv_strides = self.params.get("conv_strides", [1,1,1]) 
        conv_activation = self.params.get("conv_activation", "leaky_relu")
        
        lstm_units = self.params.get("lstm_units", 128)
        use_bidirectional_lstm = self.params.get("use_bidirectional_lstm", True)
        
        dense_units = self.params.get("dense_units", [64, 32])
        final_activation = self.params.get("final_activation", "sigmoid")
        
        self.logger.debug(f"Discriminator architecture params: seq_len={seq_len}, num_features={num_features}, conv_filters={conv_filters}, conv_kernel_sizes={conv_kernel_sizes}, conv_strides={conv_strides}, lstm_units={lstm_units}, dense_units={dense_units}")

        input_layer = Input(shape=(seq_len, num_features), name="discriminator_input")
        x = input_layer

        # Convolutional layers
        for i, filters in enumerate(conv_filters):
            kernel_size = conv_kernel_sizes[i] if i < len(conv_kernel_sizes) else conv_kernel_sizes[-1]
            strides = conv_strides[i] if i < len(conv_strides) else conv_strides[-1]
            x = Conv1D(
                filters=filters, 
                kernel_size=kernel_size, 
                strides=strides, 
                padding="same",
                name=f"conv1d_{i+1}"
            )(x)
            if conv_activation == "leaky_relu":
                x = LeakyReLU(negative_slope=self.params.get("leaky_relu_alpha", 0.2), name=f"leaky_relu_conv_{i+1}")(x)
            else:
                x = tf.keras.layers.Activation(conv_activation, name=f"activation_conv_{i+1}")(x)

        # LSTM layer
        if use_bidirectional_lstm:
            x = Bidirectional(LSTM(lstm_units, name="lstm_core"), name="bidirectional_lstm")(x) 
        else:
            x = LSTM(lstm_units, name="lstm_core")(x) 

        # Dense layers
        for i, units in enumerate(dense_units):
            x = Dense(units, name=f"dense_{i+1}")(x) 
            # Assuming same activation for dense for consistency or use a specific 'dense_activation' param
            dense_activation_type = self.params.get("dense_activation", conv_activation) # Allow separate dense activation
            if dense_activation_type == "leaky_relu":
                x = LeakyReLU(negative_slope=self.params.get("leaky_relu_alpha", 0.2), name=f"leaky_relu_dense_{i+1}")(x)
            else:
                x = tf.keras.layers.Activation(dense_activation_type, name=f"activation_dense_{i+1}")(x)

        output_layer = Dense(1, activation=final_activation, name="discriminator_output")(x)
        
        model = Model(input_layer, output_layer, name="discriminator")
        self.model = model # Assign to self.model
        self.logger.info("Discriminator model built successfully.")
        if self.params.get("print_model_summary", False): 
            self.logger.info("Discriminator Model Summary:")
            self.model.summary(print_fn=self.logger.info)
        return self.model # Return the built model

    def _compile_model(self) -> None:
        self.logger.info("Compiling discriminator model...")
        if self.model is None:
            self.logger.warning("Model is None, attempting to build before compiling.")
            self._build_model() # Ensure model is built
        
        if self.model is None: # Check again after build attempt
            self.logger.error("Failed to build model, cannot compile.")
            raise RuntimeError("Discriminator model could not be built, compilation failed.")

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.params.get("learning_rate"),
            beta_1=self.params.get("beta_1"),
            beta_2=self.params.get("beta_2", 0.999) # Ensure beta_2 is included
        )
        self.model.compile(
            optimizer=optimizer,
            loss=self.params.get("loss_function"), 
            metrics=self.params.get("metrics", ["accuracy"])
        )
        self.compiled = True # Set compiled flag
        self.logger.info(f"Discriminator model compiled with optimizer: {optimizer.get_config()}, loss: {self.params.get('loss_function')}, metrics: {self.params.get('metrics', ['accuracy'])}.")

    def get_model(self) -> Optional[Model]:
        """
        Get the discriminator model. Builds and compiles if not already done.
        
        Returns:
            Optional[Model]: The discriminator model if available, None otherwise
        """
        if self.model is None:
            self.logger.warning("Discriminator model not built. Building model now...")
            try:
                self._build_model() # This assigns to self.model
            except Exception as e:
                self.logger.error(f"Failed to build discriminator model in get_model: {e}", exc_info=True)
                return None
        
        if not self.compiled:
            self.logger.warning("Discriminator model not compiled. Compiling model now...")
            try:
                self._compile_model()
            except Exception as e:
                self.logger.error(f"Failed to compile discriminator model in get_model: {e}", exc_info=True)
                return None # Or raise, depending on desired behavior
                
        return self.model
    
    def build_model(self) -> None: # Public build, ensures compilation too
        """Public interface for building and compiling the discriminator model."""
        self.logger.info("Public build_model called.")
        self._build_model()
        if not self.compiled: # Ensure compilation after build
             self._compile_model()
    
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
    
    def load_model(self) -> None: # Keep import local if it's problematic globally
        """
        Load a pre-trained discriminator model.
        """
        pretrained_path = self.params.get("pretrained_model_path")
        self.logger.info(f"Attempting to load discriminator model from: {pretrained_path}")
        if not pretrained_path:
            self.logger.warning("No pretrained model path specified in params. Cannot load model.")
            return
        
        try:
            from tensorflow.keras.models import load_model as keras_load_model # Alias to avoid conflict if any
            self.model = keras_load_model(pretrained_path)
            self.compiled = True # Assume loaded model was compiled
            self.logger.info(f"Discriminator model loaded successfully from {pretrained_path}")
            if self.params.get("print_model_summary", False) and self.model:
                self.logger.info("Loaded Discriminator Model Summary:")
                self.model.summary(print_fn=self.logger.info)
        except Exception as e:
            self.logger.error(f"Failed to load discriminator model from {pretrained_path}: {e}", exc_info=True)
            # Optionally, re-raise or handle (e.g., try to build a new one)
            # For now, we'll let it fail and self.model will remain None or the old model
            raise # Re-raise to make the failure clear
    
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
