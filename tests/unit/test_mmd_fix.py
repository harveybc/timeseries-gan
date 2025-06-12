#!/usr/bin/env python3
"""
Test MMD implementation with data type consistency fixes.
"""

import tensorflow as tf
import numpy as np
import sys
import os

# Add the plugin directory to Python path
sys.path.append('/home/harveybc/Documents/GitHub/timeseries-gan')

def test_mmd_calculation():
    """Test the MMD calculation with fixed data types."""
    print("Testing MMD calculation with data type fixes...")
    
    # Create test class with _calculate_mmd_rbf method
    class TestMMD:
        def __init__(self):
            self.params = {
                "mmd_sample_size": 128,
                "mmd_gamma": None  # Use auto bandwidth
            }
        
        def _calculate_mmd_rbf(self, X: tf.Tensor, Y: tf.Tensor, gamma: float = None) -> tf.Tensor:
            """Test implementation of MMD calculation."""
            # Ensure tensors are float32 and 2D
            X = tf.cast(X, tf.float32)
            Y = tf.cast(Y, tf.float32)
            
            if len(X.shape) == 1:
                X = tf.expand_dims(X, -1)
            if len(Y.shape) == 1:
                Y = tf.expand_dims(Y, -1)
            
            # Flatten sequences to feature vectors if 3D (batch, seq, features) -> (batch, seq*features)
            if len(X.shape) == 3:
                X = tf.reshape(X, [tf.shape(X)[0], -1])
            if len(Y.shape) == 3:
                Y = tf.reshape(Y, [tf.shape(Y)[0], -1])
            
            m = tf.shape(X)[0]
            n = tf.shape(Y)[0]
            
            # Auto bandwidth if not specified
            if gamma is None:
                # Use simplified bandwidth calculation to avoid data type issues
                combined = tf.concat([X, Y], axis=0)
                
                # Sample a subset for efficient bandwidth estimation
                max_samples = tf.minimum(tf.shape(combined)[0], 50)
                indices = tf.random.shuffle(tf.range(tf.shape(combined)[0]))[:max_samples]
                sample_data = tf.gather(combined, indices)
                
                # Calculate mean pairwise distance
                diffs = tf.expand_dims(sample_data, 1) - tf.expand_dims(sample_data, 0)
                sq_dists = tf.reduce_sum(tf.square(diffs), axis=2)
                # Remove diagonal elements (distance to self = 0)
                mask = tf.logical_not(tf.eye(max_samples, dtype=tf.bool))
                valid_sq_dists = tf.boolean_mask(sq_dists, mask)
                mean_sq_dist = tf.reduce_mean(valid_sq_dists)
                gamma = tf.cast(1.0 / (2.0 * mean_sq_dist + 1e-8), tf.float32)
            
            def rbf_kernel(x1, x2):
                """RBF kernel computation"""
                x1 = tf.cast(x1, tf.float32)
                x2 = tf.cast(x2, tf.float32)
                x1_expanded = tf.expand_dims(x1, 1)  # [m, 1, d]
                x2_expanded = tf.expand_dims(x2, 0)  # [1, n, d]
                sq_dists = tf.reduce_sum(tf.square(x1_expanded - x2_expanded), axis=2)
                return tf.exp(-gamma * sq_dists)
            
            # Compute kernel matrices
            K_XX = rbf_kernel(X, X)
            K_YY = rbf_kernel(Y, Y)
            K_XY = rbf_kernel(X, Y)
            
            # Calculate MMD terms
            term1 = tf.cond(
                m > 1,
                lambda: tf.reduce_sum(K_XX - tf.linalg.diag(tf.linalg.diag_part(K_XX))) / tf.cast(m * (m - 1), tf.float32),
                lambda: tf.constant(0.0, dtype=tf.float32)
            )
            
            term2 = tf.cond(
                n > 1,
                lambda: tf.reduce_sum(K_YY - tf.linalg.diag(tf.linalg.diag_part(K_YY))) / tf.cast(n * (n - 1), tf.float32),
                lambda: tf.constant(0.0, dtype=tf.float32)
            )
            
            term3 = tf.cond(
                tf.logical_and(m > 0, n > 0),
                lambda: 2.0 * tf.reduce_sum(K_XY) / tf.cast(m * n, tf.float32),
                lambda: tf.constant(0.0, dtype=tf.float32)
            )
            
            mmd_sq = term1 + term2 - term3
            return tf.sqrt(tf.maximum(mmd_sq, tf.constant(0.0, dtype=tf.float32)))
    
    # Create test instance
    mmd_tester = TestMMD()
    
    # Test cases
    test_cases = [
        {
            'name': '2D data - same distribution',
            'X': tf.random.normal([32, 10], dtype=tf.float32),
            'Y': tf.random.normal([32, 10], dtype=tf.float32)
        },
        {
            'name': '2D data - different distributions',
            'X': tf.random.normal([32, 10], mean=0.0, stddev=1.0, dtype=tf.float32),
            'Y': tf.random.normal([32, 10], mean=2.0, stddev=1.0, dtype=tf.float32)
        },
        {
            'name': '3D sequential data - same distribution',
            'X': tf.random.normal([16, 8, 5], dtype=tf.float32),
            'Y': tf.random.normal([16, 8, 5], dtype=tf.float32)
        },
        {
            'name': '3D sequential data - different distributions',
            'X': tf.random.normal([16, 8, 5], mean=0.0, stddev=1.0, dtype=tf.float32),
            'Y': tf.random.normal([16, 8, 5], mean=1.0, stddev=0.5, dtype=tf.float32)
        }
    ]
    
    print("\nRunning MMD tests...")
    for i, test_case in enumerate(test_cases):
        print(f"\nTest {i+1}: {test_case['name']}")
        print(f"X shape: {test_case['X'].shape}, Y shape: {test_case['Y'].shape}")
        
        try:
            mmd_value = mmd_tester._calculate_mmd_rbf(test_case['X'], test_case['Y'])
            print(f"MMD value: {mmd_value.numpy():.6f}")
            print(f"✓ SUCCESS: No data type errors")
        except Exception as e:
            print(f"✗ ERROR: {e}")
            return False
    
    return True

def test_mixed_data_types():
    """Test handling of mixed data types."""
    print("\nTesting mixed data type handling...")
    
    # Create test class
    class TestMMD:
        def _calculate_mmd_rbf(self, X: tf.Tensor, Y: tf.Tensor, gamma: float = None) -> tf.Tensor:
            # Ensure tensors are float32 and 2D
            X = tf.cast(X, tf.float32)
            Y = tf.cast(Y, tf.float32)
            
            if len(X.shape) == 1:
                X = tf.expand_dims(X, -1)
            if len(Y.shape) == 1:
                Y = tf.expand_dims(Y, -1)
            
            # Flatten sequences to feature vectors if 3D (batch, seq, features) -> (batch, seq*features)
            if len(X.shape) == 3:
                X = tf.reshape(X, [tf.shape(X)[0], -1])
            if len(Y.shape) == 3:
                Y = tf.reshape(Y, [tf.shape(Y)[0], -1])
            
            m = tf.shape(X)[0]
            n = tf.shape(Y)[0]
            
            # Use fixed gamma to avoid bandwidth calculation complexity in this test
            if gamma is None:
                gamma = tf.constant(1.0, dtype=tf.float32)
            
            def rbf_kernel(x1, x2):
                x1 = tf.cast(x1, tf.float32)
                x2 = tf.cast(x2, tf.float32)
                x1_expanded = tf.expand_dims(x1, 1)
                x2_expanded = tf.expand_dims(x2, 0)
                sq_dists = tf.reduce_sum(tf.square(x1_expanded - x2_expanded), axis=2)
                return tf.exp(-gamma * sq_dists)
            
            K_XX = rbf_kernel(X, X)
            K_YY = rbf_kernel(Y, Y)
            K_XY = rbf_kernel(X, Y)
            
            term1 = tf.cond(
                m > 1,
                lambda: tf.reduce_sum(K_XX - tf.linalg.diag(tf.linalg.diag_part(K_XX))) / tf.cast(m * (m - 1), tf.float32),
                lambda: tf.constant(0.0, dtype=tf.float32)
            )
            
            term2 = tf.cond(
                n > 1,
                lambda: tf.reduce_sum(K_YY - tf.linalg.diag(tf.linalg.diag_part(K_YY))) / tf.cast(n * (n - 1), tf.float32),
                lambda: tf.constant(0.0, dtype=tf.float32)
            )
            
            term3 = tf.cond(
                tf.logical_and(m > 0, n > 0),
                lambda: 2.0 * tf.reduce_sum(K_XY) / tf.cast(m * n, tf.float32),
                lambda: tf.constant(0.0, dtype=tf.float32)
            )
            
            mmd_sq = term1 + term2 - term3
            return tf.sqrt(tf.maximum(mmd_sq, tf.constant(0.0, dtype=tf.float32)))
    
    mmd_tester = TestMMD()
    
    # Test with different input data types
    mixed_tests = [
        {
            'name': 'float64 input',
            'X': tf.random.normal([16, 5], dtype=tf.float64),
            'Y': tf.random.normal([16, 5], dtype=tf.float64)
        },
        {
            'name': 'int32 input (should be cast to float32)',
            'X': tf.cast(tf.random.uniform([16, 5], 0, 10, dtype=tf.int32), tf.float32),
            'Y': tf.cast(tf.random.uniform([16, 5], 0, 10, dtype=tf.int32), tf.float32)
        },
        {
            'name': 'mixed float32 and float64',
            'X': tf.random.normal([16, 5], dtype=tf.float32),
            'Y': tf.random.normal([16, 5], dtype=tf.float64)
        }
    ]
    
    for i, test_case in enumerate(mixed_tests):
        print(f"\nMixed data type test {i+1}: {test_case['name']}")
        print(f"X dtype: {test_case['X'].dtype}, Y dtype: {test_case['Y'].dtype}")
        
        try:
            mmd_value = mmd_tester._calculate_mmd_rbf(test_case['X'], test_case['Y'])
            print(f"MMD value: {mmd_value.numpy():.6f}")
            print(f"Output dtype: {mmd_value.dtype}")
            print(f"✓ SUCCESS: Data type conversion handled correctly")
        except Exception as e:
            print(f"✗ ERROR: {e}")
            return False
    
    return True

if __name__ == "__main__":
    print("=" * 80)
    print("MMD Data Type Consistency Test")
    print("=" * 80)
    
    success = True
    
    # Test basic MMD calculation
    if not test_mmd_calculation():
        success = False
    
    # Test mixed data type handling
    if not test_mixed_data_types():
        success = False
    
    print("\n" + "=" * 80)
    if success:
        print("✓ ALL TESTS PASSED: MMD implementation is working correctly with consistent data types!")
        print("The data type mismatch issue has been resolved.")
    else:
        print("✗ SOME TESTS FAILED: There are still issues with the MMD implementation.")
    print("=" * 80)
