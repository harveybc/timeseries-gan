#!/usr/bin/env python3
"""
Discriminator Plugin for VAE-GAN System

Implements a discriminator that evaluates the quality of synthetic vs real time series data.
The discriminator takes full 57-feature sequences and outputs binary classification (real/fake).

Author: TimeSeries-GAN Team
"""

import logging
import numpy as np
import os
import traceback
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Dropout, LeakyReLU, BatchNormalization, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from typing import Dict, Any, List, Optional, Tuple


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
        self.logger.info("DEBUG: DiscriminatorPlugin.__init__ called")
        
        # Update parameters from config
        self.set_params(**config)
        
        # Initialize model
        self.model: Optional[Model] = None
        self.compiled = False
        
        # Initialize model if configured
        if not self.params.get("load_pretrained_model", False):
            self.logger.info("DEBUG: DiscriminatorPlugin building model on init")
            self._build_model()
        else:
            self.logger.info("DEBUG: DiscriminatorPlugin loading pretrained model")
            self.load_model()
    
    def set_params(self, **kwargs) -> None:
        """
        Update discriminator parameters.
        
        Args:
            **kwargs: Parameter updates
        """
        self.logger.info(f"DEBUG: DiscriminatorPlugin.set_params called with {len(kwargs)} parameters")
        
        # Update main config
        self.config.update(kwargs)
        
        # Update plugin parameters (handle both prefixed and non-prefixed)
        for param_key in self.plugin_params.keys():
            prefixed_key = f"discriminator_{param_key}"
            
            if prefixed_key in kwargs:
                self.params[param_key] = kwargs[prefixed_key]
                self.logger.debug(f"DEBUG: Set {param_key} = {kwargs[prefixed_key]} (from {prefixed_key})")
            elif param_key in kwargs:
                self.params[param_key] = kwargs[param_key]
                self.logger.debug(f"DEBUG: Set {param_key} = {kwargs[param_key]}")
        
        # Handle special mappings
        if "generator_full_feature_names_ordered" in kwargs:
            self.params["feature_names"] = kwargs["generator_full_feature_names_ordered"]
            self.params["num_features"] = len(kwargs["generator_full_feature_names_ordered"])
            self.logger.info(f"DEBUG: Updated feature_names count to {self.params['num_features']}")
        
        # Rebuild model if key parameters changed
        if any(key in kwargs for key in ["sequence_length", "num_features", "conv_filters", "lstm_units"]):
            if self.params.get("sequence_length") and self.params.get("num_features"):
                self.logger.info("DEBUG: Key parameters changed, rebuilding discriminator model")
                self._build_model()
    
    def _build_model(self) -> None:
        """Build the discriminator model architecture."""
        try:
            self.logger.info(f"DEBUG: Building discriminator model with seq_len={self.params['sequence_length']}, "
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
                
                if self.params["conv_dropout_rate"] > 0:
                    x = Dropout(self.params["conv_dropout_rate"], name=f"conv_dropout_{i+1}")(x)
            
            # LSTM layer for temporal pattern recognition
            if self.params["use_bidirectional_lstm"]:
                x = Bidirectional(
                    LSTM(
                        self.params["lstm_units"],
                        dropout=self.params["lstm_dropout"],
                        recurrent_dropout=self.params["lstm_recurrent_dropout"],
                        name="lstm_layer"
                    ),
                    name="bidirectional_lstm"
                )(x)
            else:
                x = LSTM(
                    self.params["lstm_units"],
                    dropout=self.params["lstm_dropout"],
                    recurrent_dropout=self.params["lstm_recurrent_dropout"],
                    name="lstm_layer"
                )(x)
            
            # Dense layers for classification
            for i, units in enumerate(self.params["dense_units"]):
                x = Dense(units, activation='relu', name=f"dense_{i+1}")(x)
                if self.params["dense_dropout_rate"] > 0:
                    x = Dropout(self.params["dense_dropout_rate"], name=f"dense_dropout_{i+1}")(x)
            
            # Final output layer
            output = Dense(1, activation=self.params["final_activation"], name="discriminator_output")(x)
            
            # Create model
            self.model = Model(inputs=input_layer, outputs=output, name="discriminator")
            
            # Compile model
            self._compile_model()
            
            self.logger.info(f"DEBUG: Discriminator model built successfully with {self.model.count_params()} parameters")
            
        except Exception as e:
            self.logger.error(f"DEBUG: Error building discriminator model: {e}")
            self.logger.error(traceback.format_exc())
            self.model = None
    
    def _compile_model(self) -> None:
        """Compile the discriminator model."""
        if self.model is None:
            self.logger.error("DEBUG: Cannot compile model - model is None")
            return
        
        try:
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
            self.logger.info("DEBUG: Discriminator model compiled successfully")
            
        except Exception as e:
            self.logger.error(f"DEBUG: Error compiling discriminator model: {e}")
            self.compiled = False
    
    def get_model(self) -> Optional[Model]:
        """
        Get the discriminator model.
        
        Returns:
            Optional[Model]: The discriminator model if available, None otherwise
        """
        self.logger.info(f"DEBUG: DiscriminatorPlugin.get_model() called, model exists: {self.model is not None}")
        
        if self.model is None:
            self.logger.warning("DEBUG: Discriminator model not built. Building model now...")
            self._build_model()
        
        if self.model is not None:
            self.logger.info(f"DEBUG: Returning discriminator model with {self.model.count_params()} parameters")
        else:
            self.logger.error("DEBUG: Failed to build discriminator model, returning None")
        
        return self.model
    
    def build_model(self) -> None:
        """Public interface for building the discriminator model."""
        self.logger.info("DEBUG: DiscriminatorPlugin.build_model() called")
        self._build_model()
    
    def predict(self, sequences: np.ndarray) -> np.ndarray:
        """Predict real/fake probability for input sequences."""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        return self.model.predict(sequences)
    
    def train_on_batch(self, real_sequences: np.ndarray, fake_sequences: np.ndarray, 
                       label_smoothing: Optional[float] = None) -> Dict[str, float]:
        """Train discriminator on a batch of real and fake sequences."""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Prepare data
        batch_size = real_sequences.shape[0]
        
        # Combine real and fake data
        x_batch = np.concatenate([real_sequences, fake_sequences], axis=0)
        
        # Create labels (1 for real, 0 for fake)
        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))
        
        # Apply label smoothing if specified
        if label_smoothing is not None:
            real_labels -= label_smoothing
        
        y_batch = np.concatenate([real_labels, fake_labels], axis=0)
        
        # Train on batch
        loss, accuracy = self.model.train_on_batch(x_batch, y_batch)
        
        return {"loss": float(loss), "accuracy": float(accuracy)}
    
    def evaluate_sequences(self, real_sequences: np.ndarray, fake_sequences: np.ndarray) -> Dict[str, float]:
        """Evaluate discriminator performance on real and fake sequences."""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Prepare data
        batch_size = real_sequences.shape[0]
        
        x_batch = np.concatenate([real_sequences, fake_sequences], axis=0)
        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))
        y_batch = np.concatenate([real_labels, fake_labels], axis=0)
        
        # Evaluate
        loss, accuracy = self.model.evaluate(x_batch, y_batch, verbose=0)
        
        return {"loss": float(loss), "accuracy": float(accuracy)}
    
    def save_model(self, filepath: str) -> None:
        """Save the discriminator model."""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        self.model.save(filepath)
        self.logger.info(f"DEBUG: Discriminator model saved to {filepath}")
    
    def load_model(self) -> None:
        """Load a pre-trained discriminator model."""
        model_path = self.params.get("pretrained_model_path")
        if not model_path or not os.path.exists(model_path):
            raise ValueError(f"Pretrained model path not found: {model_path}")
        
        from tensorflow.keras.models import load_model
        self.model = load_model(model_path)
        self.compiled = True
        self.logger.info(f"DEBUG: Discriminator model loaded from {model_path}")
    
    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information about the discriminator plugin."""
        debug_info = {}
        for var in self.plugin_debug_vars:
            debug_info[f"discriminator_{var}"] = self.params.get(var)
        
        debug_info["discriminator_model_built"] = self.model is not None
        debug_info["discriminator_model_compiled"] = self.compiled
        
        if self.model is not None:
            debug_info["discriminator_model_params"] = self.model.count_params()
        
        return debug_info
    
    def add_debug_info(self, debug_dict: Dict[str, Any]) -> None:
        """Add discriminator debug information to existing debug dictionary."""
        debug_dict.update(self.get_debug_info())
