"""
Integration tests for the modular feeder plugin.

Tests the interaction between FeederPlugin, EncoderHandler, DataPreprocessor,
and ConditionManager modules with realistic data scenarios.
"""

import pytest
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import tempfile
import os
import logging
from unittest.mock import Mock, patch

# Import the modular feeder plugin components
from tsg_plugins.feeder_plugin import (
    FeederPlugin, 
    EncoderHandler, 
    DataPreprocessor, 
    ConditionManager
)

# Configure logging for tests
logging.basicConfig(level=logging.INFO)


class TestFeederPluginIntegration:
    """Integration tests for the complete feeder plugin system."""
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
        return {
            'latent_dim': 64,
            'normalization_method': 'standard',
            'handle_missing': 'interpolate',
            'outlier_method': 'clip',
            'outlier_threshold': 3.0,
            'condition_columns': ['volume', 'spread'],
            'condition_method': 'concatenate',
            'condition_dim': 10,
            'use_temporal_conditions': True
        }
    
    @pytest.fixture
    def sample_data(self):
        """Create sample time series data for testing."""
        np.random.seed(42)
        n_samples = 100
        
        # Create time series with OHLC + technical indicators
        dates = pd.date_range('2024-01-01', periods=n_samples, freq='H')
        
        data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.randn(n_samples) * 10 + 100,
            'high': np.random.randn(n_samples) * 10 + 105,
            'low': np.random.randn(n_samples) * 10 + 95,
            'close': np.random.randn(n_samples) * 10 + 100,
            'volume': np.random.randint(1000, 10000, n_samples),
            'rsi': np.random.rand(n_samples) * 100,
            'macd': np.random.randn(n_samples) * 2,
            'spread': np.random.rand(n_samples) * 0.01,
            'volatility': np.random.rand(n_samples) * 0.5
        })
        
        return data
    
    @pytest.fixture
    def mock_encoder_model(self, tmp_path):
        """Create a mock Keras encoder model for testing."""
        model_path = tmp_path / "test_encoder.keras"
        
        # Create a simple encoder model
        model = keras.Sequential([
            keras.layers.Input(shape=(9,)),  # 9 features (excluding timestamp)
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(32)  # Latent dimension
        ])
        
        # Use string names to avoid serialization issues
        model.compile(
            optimizer='adam', 
            loss='mean_squared_error',
            metrics=['mean_squared_error']
        )
        model.save(str(model_path))
        
        return str(model_path)
    
    def test_encoder_handler_integration(self, sample_config, sample_data, mock_encoder_model):
        """Test EncoderHandler with realistic data and model."""
        encoder_handler = EncoderHandler(sample_config)
        
        # Test model loading
        assert encoder_handler.load_model(mock_encoder_model)
        assert encoder_handler.model_loaded
        
        # Test model validation
        assert encoder_handler.validate_model()
        
        # Test encoding with real data
        numeric_data = sample_data.select_dtypes(include=[np.number]).values
        encoded = encoder_handler.encode_data(numeric_data)
        
        assert encoded is not None
        assert encoded.shape[0] == len(sample_data)
        assert encoded.shape[1] == 32  # Output dimension from mock model
        
        # Test statistics collection
        stats = encoder_handler.get_latent_stats()
        assert stats['mean'] is not None
        assert stats['std'] is not None
        
        # Test model info
        info = encoder_handler.get_model_info()
        assert info['loaded'] is True
        assert info['latent_dim'] is not None
    
    def test_data_preprocessor_integration(self, sample_config, sample_data):
        """Test DataPreprocessor with realistic data scenarios."""
        preprocessor = DataPreprocessor(sample_config)
        
        # Test fitting and transformation
        numeric_data = sample_data.select_dtypes(include=[np.number])
        
        # Test fit_transform
        processed = preprocessor.fit_transform(numeric_data)
        assert processed is not None
        assert processed.shape[0] == len(sample_data)
        assert preprocessor.is_fitted
        
        # Test separate transform
        new_data = numeric_data.iloc[:50]  # Subset of data
        transformed = preprocessor.transform(new_data)
        assert transformed is not None
        assert transformed.shape[0] == 50
        
        # Test inverse transform
        original_scale = preprocessor.inverse_transform(transformed)
        assert original_scale is not None
        assert original_scale.shape == transformed.shape
        
        # Test statistics
        stats = preprocessor.get_statistics()
        assert stats['mean'] is not None
        assert stats['std'] is not None
        
        # Test with missing values
        data_with_missing = numeric_data.copy()
        data_with_missing.iloc[10:15, 1:3] = np.nan
        
        processed_missing = preprocessor.transform(data_with_missing)
        assert processed_missing is not None
        assert not np.isnan(processed_missing).any()
    
    def test_condition_manager_integration(self, sample_config, sample_data):
        """Test ConditionManager with realistic condition extraction."""
        condition_manager = ConditionManager(sample_config)
        
        # Test initialization
        assert condition_manager.initialize(sample_data)
        assert condition_manager.is_initialized
        
        # Test condition extraction
        conditions = condition_manager.extract_conditions(sample_data, 'timestamp')
        assert conditions is not None
        assert conditions.shape[0] == len(sample_data)
        
        # Test condition processing
        processed_conditions = condition_manager.process_conditions(conditions)
        assert processed_conditions is not None
        assert processed_conditions.shape[0] == len(sample_data)
        
        # Test combining with latents
        mock_latents = np.random.randn(len(sample_data), 32)
        combined = condition_manager.combine_with_latents(mock_latents, processed_conditions)
        assert combined is not None
        assert combined.shape[0] == len(sample_data)
        assert combined.shape[1] == mock_latents.shape[1] + processed_conditions.shape[1]
        
        # Test temporal conditions
        info = condition_manager.get_condition_info()
        assert info['use_temporal_conditions'] is True
        assert info['has_conditions'] is True
    
    def test_feeder_plugin_complete_integration(self, sample_config, sample_data, mock_encoder_model):
        """Test complete FeederPlugin integration with all modules."""
        feeder_plugin = FeederPlugin(sample_config)
        
        # Test initialization
        assert feeder_plugin.initialize(mock_encoder_model, sample_data)
        assert feeder_plugin.is_initialized
        assert feeder_plugin.encoder_loaded
        assert feeder_plugin.preprocessor_fitted
        
        # Test complete data processing pipeline
        result = feeder_plugin.process_data(sample_data, timestamp_col='timestamp')
        
        assert result is not None
        assert 'latents' in result
        assert 'conditions' in result
        assert 'combined' in result
        assert 'metadata' in result
        
        # Validate result structure
        latents = result['latents']
        conditions = result['conditions']
        combined = result['combined']
        metadata = result['metadata']
        
        assert latents.shape[0] == len(sample_data)
        assert conditions is not None
        assert combined is not None
        assert combined.shape[1] > latents.shape[1]  # Should include conditions
        
        # Test metadata
        assert metadata['input_shape'] == sample_data.shape
        assert metadata['has_conditions'] is True
        assert metadata['latent_dim'] > 0
        assert metadata['condition_dim'] > 0
        
        # Test encode_only functionality
        latents_only = feeder_plugin.encode_only(sample_data)
        assert latents_only is not None
        assert np.array_equal(latents_only, latents)
        
        # Test plugin info
        info = feeder_plugin.get_plugin_info()
        assert info['plugin_type'] == 'feeder'
        assert info['initialized'] is True
        assert info['encoder_loaded'] is True
        
        # Test statistics
        stats = feeder_plugin.get_statistics()
        assert 'encoder_stats' in stats
        assert 'preprocessor_stats' in stats
        assert 'condition_info' in stats
    
    def test_error_handling_and_edge_cases(self, sample_config):
        """Test error handling and edge cases across modules."""
        feeder_plugin = FeederPlugin(sample_config)
        
        # Test uninitialized state
        result = feeder_plugin.process_data(pd.DataFrame({'a': [1, 2, 3]}))
        assert result is None
        
        # Test with invalid encoder path
        assert not feeder_plugin.initialize('/invalid/path/model.h5')
        
        # Test with empty data
        encoder_handler = EncoderHandler(sample_config)
        empty_data = np.array([]).reshape(0, 5)
        result = encoder_handler.encode_data(empty_data)
        assert result is None
        
        # Test preprocessor with non-numeric data
        preprocessor = DataPreprocessor(sample_config)
        text_data = pd.DataFrame({'text': ['a', 'b', 'c']})
        assert not preprocessor.fit(text_data)
        
        # Test condition manager with missing timestamp
        condition_manager = ConditionManager(sample_config)
        condition_manager.initialize()
        no_timestamp_data = pd.DataFrame({'value': [1, 2, 3]})
        conditions = condition_manager.extract_conditions(no_timestamp_data, 'missing_col')
        assert conditions is not None  # Should return zero conditions
    
    def test_module_reset_and_cleanup(self, sample_config, sample_data, mock_encoder_model):
        """Test reset and cleanup functionality across modules."""
        feeder_plugin = FeederPlugin(sample_config)
        
        # Initialize and process data
        feeder_plugin.initialize(mock_encoder_model, sample_data)
        feeder_plugin.process_data(sample_data, timestamp_col='timestamp')
        
        # Verify initial state
        assert feeder_plugin.processed_samples > 0
        assert feeder_plugin.is_initialized
        
        # Test reset
        feeder_plugin.reset()
        assert not feeder_plugin.is_initialized
        assert not feeder_plugin.encoder_loaded
        assert not feeder_plugin.preprocessor_fitted
        assert feeder_plugin.processed_samples == 0
        
        # Test cleanup
        feeder_plugin.cleanup()
        # Should not raise any exceptions
    
    def test_configuration_variations(self, sample_data, mock_encoder_model):
        """Test different configuration scenarios."""
        # Test with minimal configuration
        minimal_config = {
            'latent_dim': 32,
            'normalization_method': 'none',
            'condition_columns': [],
            'use_temporal_conditions': False
        }
        
        feeder_minimal = FeederPlugin(minimal_config)
        assert feeder_minimal.initialize(mock_encoder_model, sample_data)
        
        result = feeder_minimal.process_data(sample_data)
        assert result is not None
        assert result['conditions'] is not None  # Should be zero vector
        
        # Test with maximal configuration
        maximal_config = {
            'latent_dim': 128,
            'normalization_method': 'minmax',
            'handle_missing': 'mean',
            'outlier_method': 'remove',
            'outlier_threshold': 2.0,
            'condition_columns': ['volume', 'spread', 'volatility'],
            'condition_method': 'embed',
            'condition_dim': 20,
            'use_temporal_conditions': True
        }
        
        feeder_maximal = FeederPlugin(maximal_config)
        assert feeder_maximal.initialize(mock_encoder_model, sample_data)
        
        result = feeder_maximal.process_data(sample_data, timestamp_col='timestamp')
        assert result is not None
        assert result['metadata']['condition_dim'] > 0


class TestModuleInteroperability:
    """Test interoperability between individual modules."""
    
    @pytest.fixture
    def mock_encoder_model(self, tmp_path):
        """Create a mock Keras encoder model for testing."""
        model_path = tmp_path / "test_encoder.keras"
        
        # Create a simple encoder model
        model = keras.Sequential([
            keras.layers.Input(shape=(9,)),  # 9 features (excluding timestamp)
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(32)  # Latent dimension
        ])
        
        # Use string names to avoid serialization issues
        model.compile(
            optimizer='adam', 
            loss='mean_squared_error',
            metrics=['mean_squared_error']
        )
        model.save(str(model_path))
        
        return str(model_path)
    
    def test_encoder_preprocessor_integration(self, mock_encoder_model):
        """Test integration between EncoderHandler and DataPreprocessor."""
        config = {'latent_dim': 64, 'normalization_method': 'standard'}
        
        encoder = EncoderHandler(config)
        preprocessor = DataPreprocessor(config)
        
        # Create test data
        raw_data = pd.DataFrame(np.random.randn(50, 9))
        
        # Fit preprocessor and load encoder
        preprocessor.fit(raw_data)
        encoder.load_model(mock_encoder_model)
        
        # Process through pipeline
        processed_data = preprocessor.transform(raw_data)
        encoded_data = encoder.encode_data(processed_data)
        
        assert encoded_data is not None
        assert encoded_data.shape[0] == 50
    
    def test_condition_encoder_integration(self, mock_encoder_model):
        """Test integration between ConditionManager and EncoderHandler."""
        config = {
            'latent_dim': 64,
            'condition_columns': ['feature1', 'feature2'],
            'use_temporal_conditions': True
        }
        
        condition_manager = ConditionManager(config)
        encoder = EncoderHandler(config)
        
        # Create test data with conditions
        test_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=30, freq='H'),
            'feature1': np.random.randn(30),
            'feature2': np.random.randn(30),
            'value1': np.random.randn(30),
            'value2': np.random.randn(30)
        })
        
        # Initialize and process
        condition_manager.initialize(test_data)
        encoder.load_model(mock_encoder_model)
        
        # Extract conditions
        conditions = condition_manager.extract_conditions(test_data, 'timestamp')
        
        # Encode values
        value_data = test_data[['feature1', 'feature2', 'value1', 'value2']].values
        # Pad to match encoder input shape (9 features)
        padded_data = np.pad(value_data, ((0, 0), (0, 5)), mode='constant')
        encoded = encoder.encode_data(padded_data)
        
        # Combine
        combined = condition_manager.combine_with_latents(encoded, conditions)
        
        assert combined is not None
        assert combined.shape[1] > encoded.shape[1]


# Fixture for pytest configuration
@pytest.fixture(scope="session")
def tf_config():
    """Configure TensorFlow for testing."""
    # Suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.get_logger().setLevel('ERROR')


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "--tb=short"])
