#!/usr/bin/env python3
"""
Data Generator Module

This module handles training data generation and batching for GAN training,
providing focused functionality for data preparation and management.
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import logging
from typing import Any, Dict, List, Optional, Tuple, Generator


class DataGenerator:
    """Handles data generation and batching for GAN training."""
    
    def __init__(self, params: Dict[str, Any], logger: logging.Logger):
        """Initialize data generator."""
        self.params = params
        self.logger = logger
        
        # Data configuration
        self.seq_len = params.get("seq_len", 18)
        self.batch_size = params.get("gan_batch_size", 32)
        self.latent_dim = params.get("latent_dim", 32)
        
        # Cached data
        self.real_data_cache = None
        self.data_stats = {}
        
        self.logger.info("DataGenerator initialized")
    
    def prepare_real_data(self, x_real_df: pd.DataFrame) -> tf.Tensor:
        """
        Prepare real data for training.
        
        Args:
            x_real_df: Real data DataFrame
        
        Returns:
            Prepared real data tensor
        """
        self.logger.info("Preparing real data for training")
        
        try:
            # Convert DataFrame to numpy array
            if isinstance(x_real_df, pd.DataFrame):
                real_data_np = x_real_df.values
                feature_names = list(x_real_df.columns)
            else:
                real_data_np = x_real_df
                feature_names = [f"feature_{i}" for i in range(real_data_np.shape[1])]
            
            # Store original stats
            self.data_stats = {
                "original_shape": real_data_np.shape,
                "feature_names": feature_names,
                "mean": np.mean(real_data_np, axis=0),
                "std": np.std(real_data_np, axis=0),
                "min": np.min(real_data_np, axis=0),
                "max": np.max(real_data_np, axis=0)
            }
            
            # Reshape data for sequence processing
            processed_data = self._reshape_for_sequences(real_data_np)
            
            # Convert to TensorFlow tensor
            real_data_tensor = tf.convert_to_tensor(processed_data, dtype=tf.float32)
            
            # Cache for reuse
            self.real_data_cache = real_data_tensor
            
            self.logger.info(f"Real data prepared: {real_data_tensor.shape}")
            return real_data_tensor
            
        except Exception as e:
            self.logger.error(f"Error preparing real data: {e}")
            raise
    
    def _reshape_for_sequences(self, data: np.ndarray) -> np.ndarray:
        """
        Reshape data into sequences.
        
        Args:
            data: Input data array
        
        Returns:
            Reshaped data for sequence processing
        """
        if len(data.shape) == 2:
            # Data is (timesteps, features) - create sequences
            timesteps, num_features = data.shape
            
            # Calculate number of complete sequences
            num_sequences = timesteps // self.seq_len
            
            if num_sequences == 0:
                # Not enough data for even one sequence
                self.logger.warning(f"Not enough data for sequences: {timesteps} < {self.seq_len}")
                # Pad or repeat data
                if timesteps < self.seq_len:
                    # Repeat data to reach seq_len
                    repeat_factor = (self.seq_len // timesteps) + 1
                    data = np.tile(data, (repeat_factor, 1))[:self.seq_len]
                    num_sequences = 1
            
            # Reshape to (num_sequences, seq_len, num_features)
            sequences = data[:num_sequences * self.seq_len].reshape(
                num_sequences, self.seq_len, num_features
            )
            
            self.logger.info(f"Reshaped {data.shape} to sequences: {sequences.shape}")
            return sequences
            
        elif len(data.shape) == 3:
            # Data is already in sequence format
            self.logger.info(f"Data already in sequence format: {data.shape}")
            return data
        
        else:
            raise ValueError(f"Unsupported data shape: {data.shape}")
    
    def generate_noise(self, batch_size: Optional[int] = None) -> tf.Tensor:
        """
        Generate random noise for generator input.
        
        Args:
            batch_size: Batch size (uses default if None)
        
        Returns:
            Random noise tensor
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        # Generate Gaussian noise
        noise = tf.random.normal([batch_size, self.seq_len, self.latent_dim])
        
        return noise
    
    def get_real_batch(self, batch_size: Optional[int] = None) -> tf.Tensor:
        """
        Get a random batch of real data.
        
        Args:
            batch_size: Batch size (uses default if None)
        
        Returns:
            Batch of real data
        """
        if self.real_data_cache is None:
            raise ValueError("Real data not prepared. Call prepare_real_data() first.")
        
        if batch_size is None:
            batch_size = self.batch_size
        
        # Get dataset size
        dataset_size = tf.shape(self.real_data_cache)[0]
        
        # Sample random indices
        indices = tf.random.uniform([batch_size], 0, dataset_size, dtype=tf.int32)
        
        # Get batch
        batch = tf.gather(self.real_data_cache, indices)
        
        return batch
    
    def create_data_generator(self, real_data: tf.Tensor, batch_size: Optional[int] = None) -> Generator:
        """
        Create a data generator for training.
        
        Args:
            real_data: Real training data
            batch_size: Batch size
        
        Yields:
            Batches of real data and corresponding noise
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        dataset_size = real_data.shape[0]
        
        while True:
            # Generate random indices for this batch
            indices = np.random.choice(dataset_size, size=batch_size, replace=True)
            
            # Get real data batch
            real_batch = tf.gather(real_data, indices)
            
            # Generate noise batch
            noise_batch = self.generate_noise(batch_size)
            
            yield real_batch, noise_batch
    
    def create_tf_dataset(self, real_data: tf.Tensor, batch_size: Optional[int] = None) -> tf.data.Dataset:
        """
        Create TensorFlow dataset for training.
        
        Args:
            real_data: Real training data
            batch_size: Batch size
        
        Returns:
            TensorFlow dataset
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        # Create dataset from real data
        dataset = tf.data.Dataset.from_tensor_slices(real_data)
        
        # Shuffle and batch
        dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def validate_data_shapes(self, real_data: tf.Tensor, generated_data: tf.Tensor) -> bool:
        """
        Validate that real and generated data have compatible shapes.
        
        Args:
            real_data: Real data tensor
            generated_data: Generated data tensor
        
        Returns:
            True if shapes are compatible
        """
        real_shape = real_data.shape
        gen_shape = generated_data.shape
        
        # Check if sequence length and features match
        if len(real_shape) >= 2 and len(gen_shape) >= 2:
            seq_len_match = real_shape[-2] == gen_shape[-2]
            features_match = real_shape[-1] == gen_shape[-1]
            
            if seq_len_match and features_match:
                self.logger.info("Data shapes are compatible")
                return True
            else:
                self.logger.warning(f"Shape mismatch - Real: {real_shape}, Generated: {gen_shape}")
                return False
        
        self.logger.warning(f"Invalid data shapes - Real: {real_shape}, Generated: {gen_shape}")
        return False
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the prepared data.
        
        Returns:
            Dictionary with data statistics
        """
        stats = self.data_stats.copy()
        
        if self.real_data_cache is not None:
            stats["processed_shape"] = self.real_data_cache.shape.as_list()
            stats["num_sequences"] = self.real_data_cache.shape[0]
        
        return stats
    
    def normalize_data(self, data: tf.Tensor, method: str = "minmax") -> tf.Tensor:
        """
        Normalize data using specified method.
        
        Args:
            data: Input data tensor
            method: Normalization method ("minmax", "zscore", "none")
        
        Returns:
            Normalized data tensor
        """
        if method == "minmax":
            # Min-max normalization to [0, 1]
            data_min = tf.reduce_min(data, axis=[0, 1], keepdims=True)
            data_max = tf.reduce_max(data, axis=[0, 1], keepdims=True)
            normalized = (data - data_min) / (data_max - data_min + 1e-8)
            
        elif method == "zscore":
            # Z-score normalization
            data_mean = tf.reduce_mean(data, axis=[0, 1], keepdims=True)
            data_std = tf.math.reduce_std(data, axis=[0, 1], keepdims=True)
            normalized = (data - data_mean) / (data_std + 1e-8)
            
        elif method == "none":
            # No normalization
            normalized = data
            
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        self.logger.info(f"Data normalized using {method} method")
        return normalized
    
    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information."""
        return {
            "seq_len": self.seq_len,
            "batch_size": self.batch_size,
            "latent_dim": self.latent_dim,
            "real_data_cached": self.real_data_cache is not None,
            "data_stats": self.data_stats
        }
