#!/usr/bin/env python3
"""
Integration tests for Generator Plugin modules.

This module tests the integration between all generator plugin components
to ensure they work together correctly and maintain the plugin interface contract.
"""

import unittest
import tempfile
import json
import os
import logging
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from typing import Dict, Any

# Import generator plugin modules
from tsg_plugins.generator_plugin import GeneratorPlugin
from tsg_plugins.generator_plugin.normalization_handler import NormalizationHandler
from tsg_plugins.generator_plugin.model_loader import ModelLoader
from tsg_plugins.generator_plugin.feature_processor import FeatureProcessor
from tsg_plugins.generator_plugin.technical_indicator_calculator import TechnicalIndicatorCalculator
from tsg_plugins.generator_plugin.data_generator import DataGenerator
from tsg_plugins.generator_plugin.sequence_builder import SequenceBuilder
from tsg_plugins.generator_plugin.feature_validator import FeatureValidator
from tsg_plugins.generator_plugin.initial_data_handler import InitialDataHandler


class TestGeneratorPluginIntegration(unittest.TestCase):
    """Integration tests for Generator Plugin components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test configuration
        self.test_config = {
            "generator_sequential_model_file": None,
            "generator_decoder_input_window_size": 144,
            "generator_full_feature_names_ordered": [
                "OPEN", "HIGH", "LOW", "CLOSE", "RSI", "MACD", 
                "day_of_month_sin", "day_of_month_cos"
            ],
            "generator_decoder_output_feature_names": ["OPEN", "HIGH", "LOW", "CLOSE"],
            "generator_ohlc_feature_names": ["OPEN", "HIGH", "LOW", "CLOSE"],
            "generator_ti_feature_names": ["RSI", "MACD"],
            "generator_date_conditional_feature_names": ["day_of_month"],
            "generator_feeder_conditional_feature_names": [],
            "generator_ti_calculation_min_lookback": 50,
            "generator_ti_params": {
                "rsi_length": 14,
                "macd_fast": 12,
                "macd_slow": 26,
                "macd_signal": 9
            },
            "generator_decoder_input_name_latent": "decoder_input_z_seq",
            "generator_decoder_input_name_window": "input_x_window",
            "generator_decoder_input_name_conditions": "decoder_input_conditions",
            "generator_decoder_input_name_context": "decoder_input_h_context",
            "generator_normalization_params_file": None,
            "x_train_file": None
        }
        
        # Create test normalization parameters
        self.norm_params = {
            "OPEN": {"min": 1.0, "max": 2.0},
            "HIGH": {"min": 1.1, "max": 2.1},
            "LOW": {"min": 0.9, "max": 1.9},
            "CLOSE": {"min": 1.0, "max": 2.0},
            "RSI": {"min": 0.0, "max": 100.0},
            "MACD": {"min": -0.1, "max": 0.1}
        }
        
        # Create test data file
        self.test_data = pd.DataFrame({
            "OPEN": [1.5, 1.6, 1.7],
            "HIGH": [1.6, 1.7, 1.8],
            "LOW": [1.4, 1.5, 1.6],
            "CLOSE": [1.55, 1.65, 1.75]
        })
        
        # Setup logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_files(self):
        """Create test files for integration testing."""
        # Create normalization params file
        norm_file = os.path.join(self.temp_dir, "norm_params.json")
        with open(norm_file, 'w') as f:
            json.dump(self.norm_params, f)
        
        # Create test data file
        data_file = os.path.join(self.temp_dir, "test_data.csv")
        self.test_data.to_csv(data_file, index=False)
        
        return norm_file, data_file
    
    def test_normalization_handler_integration(self):
        """Test NormalizationHandler integration."""
        norm_file, data_file = self.create_test_files()
        
        # Test initialization
        params = self.test_config.copy()
        params["generator_normalization_params_file"] = norm_file
        
        handler = NormalizationHandler(params, self.logger)
        
        # Test normalization parameters loaded
        self.assertIsNotNone(handler.normalization_params)
        self.assertEqual(len(handler.normalization_params), 6)
        
        # Test normalization/denormalization
        test_values = np.array([1.5, 1.6, 1.7])
        normalized = handler.normalize_feature(test_values, "OPEN")
        denormalized = handler.denormalize_feature(normalized, "OPEN")
        
        np.testing.assert_array_almost_equal(test_values, denormalized, decimal=6)
        
        # Test initial close anchor loading
        anchor = handler.load_initial_close_anchor(data_file)
        self.assertIsNotNone(anchor)
        self.assertEqual(anchor, 1.55)  # First CLOSE value
        
    def test_feature_validator_integration(self):
        """Test FeatureValidator integration."""
        full_features = self.test_config["generator_full_feature_names_ordered"]
        validator = FeatureValidator(full_features)
        
        # Test validation passes with valid configuration
        # Update config to include all required parameters
        test_config = self.test_config.copy()
        test_config["decoder_output_feature_names"] = test_config["generator_decoder_output_feature_names"]
        test_config["ohlc_feature_names"] = test_config["generator_ohlc_feature_names"]
        test_config["ti_feature_names"] = test_config["generator_ti_feature_names"]
        test_config["date_conditional_feature_names"] = test_config["generator_date_conditional_feature_names"]
        test_config["feeder_conditional_feature_names"] = test_config["generator_feeder_conditional_feature_names"]
        
        try:
            validator.validate_feature_name_consistency(test_config)
            # Should not raise exception
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Feature validation failed unexpectedly: {e}")
        
        # Test validation fails with invalid configuration
        invalid_config = self.test_config.copy()
        invalid_config["generator_decoder_output_feature_names"] = ["INVALID_FEATURE"]
        
        validator_invalid = FeatureValidator(full_features)
        
        with self.assertRaises(ValueError):
            validator_invalid.validate_feature_name_consistency(invalid_config)
    
    def test_technical_indicator_calculator_integration(self):
        """Test TechnicalIndicatorCalculator integration."""
        ti_names = self.test_config["generator_ti_feature_names"]
        ti_params = self.test_config["generator_ti_params"]
        calculator = TechnicalIndicatorCalculator(ti_names, ti_params)
        
        # Create test OHLC data
        ohlc_data = pd.DataFrame({
            "OPEN": np.random.uniform(1.0, 2.0, 100),
            "HIGH": np.random.uniform(1.5, 2.5, 100),
            "LOW": np.random.uniform(0.5, 1.5, 100),
            "CLOSE": np.random.uniform(1.0, 2.0, 100)
        })
        
        # Ensure HIGH >= OPEN, CLOSE and LOW <= OPEN, CLOSE
        for i in range(len(ohlc_data)):
            ohlc_data.loc[i, "HIGH"] = max(ohlc_data.loc[i, "OPEN"], 
                                          ohlc_data.loc[i, "CLOSE"], 
                                          ohlc_data.loc[i, "HIGH"])
            ohlc_data.loc[i, "LOW"] = min(ohlc_data.loc[i, "OPEN"], 
                                         ohlc_data.loc[i, "CLOSE"], 
                                         ohlc_data.loc[i, "LOW"])
        
        # Test TI calculation
        ohlc_names = self.test_config["generator_ohlc_feature_names"]
        ti_results = calculator.calculate_technical_indicators(ohlc_data, ohlc_names)
        
        self.assertIsInstance(ti_results, pd.DataFrame)
        self.assertEqual(len(ti_results), 1)  # Should return last row
        
        # Check that requested TIs are present
        expected_columns = self.test_config["generator_ti_feature_names"]
        for col in expected_columns:
            self.assertIn(col, ti_results.columns)
    
    def test_sequence_builder_integration(self):
        """Test SequenceBuilder integration."""
        # Create required dependencies
        norm_handler = NormalizationHandler(self.test_config, self.logger)
        ti_calculator = TechnicalIndicatorCalculator(self.test_config, self.logger)
        
        # Create feature mapping
        features = self.test_config["generator_full_feature_names_ordered"]
        feature_to_idx = {name: idx for idx, name in enumerate(features)}
        num_all_features = len(features)
        
        builder = SequenceBuilder(
            self.test_config, feature_to_idx, num_all_features, 
            norm_handler, ti_calculator
        )
        
        # Test window creation
        test_window = np.random.random((144, 8))  # 144 timesteps, 8 features
        
        processed_window = builder.process_window(test_window)
        
        self.assertEqual(processed_window.shape, test_window.shape)
        self.assertEqual(processed_window.dtype, np.float32)
        
        # Test sequence validation
        valid_sequence = np.random.random((10, 8))
        self.assertTrue(builder.validate_sequence(valid_sequence))
        
        # Test invalid sequence
        invalid_sequence = np.array([])
        self.assertFalse(builder.validate_sequence(invalid_sequence))
    
    def test_data_generator_integration(self):
        """Test DataGenerator integration."""
        # Create mock model for testing
        mock_model = MagicMock()
        mock_model.predict.return_value = np.random.random((1, 1, 4))  # Batch, time, features
        
        # Create required dependencies
        norm_handler = NormalizationHandler(self.test_config, self.logger)
        ti_calculator = TechnicalIndicatorCalculator(self.test_config, self.logger)
        
        # Create feature mapping
        features = self.test_config["generator_full_feature_names_ordered"]
        feature_to_idx = {name: idx for idx, name in enumerate(features)}
        
        generator = DataGenerator(
            self.test_config, feature_to_idx, norm_handler, ti_calculator
        )
        generator.model = mock_model
        
        # Test data generation
        latent_input = np.random.random((1, 144, 10))  # Batch, time, latent_dim
        window_input = np.random.random((1, 144, 8))   # Batch, time, features
        conditions = np.random.random((1, 4))          # Batch, conditions
        context = np.random.random((1, 64))            # Batch, context
        
        result = generator.generate_batch(latent_input, window_input, conditions, context)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.shape[0], 1)  # Batch size
        
    def test_model_loader_integration(self):
        """Test ModelLoader integration with mock model."""
        loader = ModelLoader(self.test_config, self.logger)
        
        # Test loading with non-existent file
        result = loader.load_model_from_path("non_existent_file.keras")
        self.assertIsNone(result)
        
        # Test model validation
        mock_model = MagicMock()
        mock_model.input_names = ["decoder_input_z_seq", "input_x_window"]
        
        is_valid = loader._validate_model(mock_model)
        self.assertTrue(is_valid)
        
    def test_initial_data_handler_integration(self):
        """Test InitialDataHandler integration."""
        norm_file, data_file = self.create_test_files()
        
        # Create required dependency
        norm_handler = NormalizationHandler(self.test_config, self.logger)
        norm_handler.load_normalization_params(norm_file)
        
        handler = InitialDataHandler(norm_handler)
        
        # Test loading initial close anchor
        anchor = handler.load_initial_close_anchor(data_file)
        self.assertIsNotNone(anchor)
        self.assertIsInstance(anchor, float)
        
        # Test processing initial window
        initial_window = np.random.random((144, 8))
        processed = handler.process_initial_window(initial_window)
        
        self.assertEqual(processed.shape, initial_window.shape)
        self.assertEqual(processed.dtype, np.float32)
    
    @patch('tsg_plugins.generator_plugin.model_loader.load_model')
    def test_generator_plugin_full_integration(self, mock_load_model):
        """Test full GeneratorPlugin integration without actual model file."""
        norm_file, data_file = self.create_test_files()
        
        # Mock the model loading
        mock_model = MagicMock()
        mock_model.input_names = [
            "decoder_input_z_seq",
            "input_x_window", 
            "decoder_input_conditions",
            "decoder_input_h_context"
        ]
        mock_model.predict.return_value = np.random.random((1, 1, 4))
        mock_load_model.return_value = mock_model
        
        # Update config with test files and mock model handling
        config = self.test_config.copy()
        config["generator_normalization_params_file"] = norm_file
        config["x_train_file"] = data_file
        config["generator_sequential_model_file"] = None  # Don't load model for this test
        
        # Test plugin initialization
        plugin = GeneratorPlugin(config)
        
        # Test mandatory plugin interface methods
        self.assertTrue(hasattr(plugin, 'plugin_params'))
        self.assertTrue(hasattr(plugin, 'plugin_debug_vars'))
        self.assertTrue(callable(getattr(plugin, '__init__')))
        self.assertTrue(callable(getattr(plugin, 'set_params')))
        self.assertTrue(callable(getattr(plugin, 'get_debug_info')))
        self.assertTrue(callable(getattr(plugin, 'add_debug_info')))
        
        # Test debug info
        debug_info = plugin.get_debug_info()
        self.assertIsInstance(debug_info, dict)
        self.assertIn('sequential_model_file', debug_info)
        
        # Test set_params
        new_params = {"generator_decoder_input_window_size": 200}
        plugin.set_params(**new_params)
        self.assertEqual(plugin.params["decoder_input_window_size"], 200)
        
        # Test add_debug_info
        test_debug = {}
        plugin.add_debug_info(test_debug)
        self.assertIsInstance(test_debug, dict)
        self.assertTrue(len(test_debug) > 0)
    
    def test_module_interdependency(self):
        """Test that modules work together correctly."""
        norm_file, data_file = self.create_test_files()
        
        # Initialize all components
        norm_handler = NormalizationHandler(self.test_config, self.logger)
        norm_handler.load_normalization_params(norm_file)
        
        validator = FeatureValidator(self.test_config["generator_full_feature_names_ordered"])
        calculator = TechnicalIndicatorCalculator(self.test_config, self.logger)
        builder = SequenceBuilder(self.test_config, self.logger)
        
        # Test workflow: validate -> normalize -> calculate TIs -> build sequence
        
        # 1. Validate configuration
        validator.validate_feature_consistency()
        
        # 2. Normalize test data
        test_values = np.array([1.5])
        normalized = norm_handler.normalize_feature(test_values, "CLOSE")
        
        # 3. Calculate technical indicators
        ohlc_data = pd.DataFrame({
            "OPEN": [1.5, 1.6, 1.7] * 20,   # 60 rows
            "HIGH": [1.6, 1.7, 1.8] * 20,
            "LOW": [1.4, 1.5, 1.6] * 20,
            "CLOSE": [1.55, 1.65, 1.75] * 20
        })
        
        ti_results = calculator.calculate_technical_indicators(ohlc_data)
        self.assertIsNotNone(ti_results)
        
        # 4. Build sequence
        window = np.random.random((144, 8))
        processed_window = builder.process_window(window)
        self.assertIsNotNone(processed_window)
        
        # All steps completed successfully
        self.assertTrue(True)
    
    def test_error_handling_integration(self):
        """Test error handling across modules."""
        # Test with invalid configuration
        invalid_config = {
            "generator_full_feature_names_ordered": [],  # Empty list
            "generator_decoder_output_feature_names": ["CLOSE"],
            "generator_ohlc_feature_names": ["OPEN", "HIGH", "LOW", "CLOSE"]
        }
        
        # Feature validator should catch this
        with self.assertRaises(ValueError):
            validator = FeatureValidator([])  # Empty feature list
        with self.assertRaises(ValueError):
            validator.validate_feature_consistency()
        
        # Normalization handler should handle missing files gracefully
        norm_handler = NormalizationHandler(invalid_config, self.logger)
        result = norm_handler.load_normalization_params("non_existent_file.json")
        self.assertIsNone(result)
        
        # TI calculator should handle empty data gracefully
        calculator = TechnicalIndicatorCalculator(self.test_config, self.logger)
        empty_df = pd.DataFrame()
        ti_results = calculator.calculate_technical_indicators(empty_df)
        self.assertIsNotNone(ti_results)
        self.assertEqual(len(ti_results), 1)  # Should return NaN placeholder
    
    def test_performance_characteristics(self):
        """Test performance characteristics of integrated modules."""
        import time
        
        norm_file, data_file = self.create_test_files()
        
        # Test normalization performance
        norm_handler = NormalizationHandler(self.test_config, self.logger)
        norm_handler.load_normalization_params(norm_file)
        
        large_array = np.random.random(10000)
        
        start_time = time.time()
        normalized = norm_handler.normalize_feature(large_array, "CLOSE")
        norm_time = time.time() - start_time
        
        # Should complete in reasonable time (< 1 second)
        self.assertLess(norm_time, 1.0)
        self.assertEqual(len(normalized), len(large_array))
        
        # Test TI calculation performance
        calculator = TechnicalIndicatorCalculator(self.test_config, self.logger)
        
        large_ohlc = pd.DataFrame({
            "OPEN": np.random.uniform(1.0, 2.0, 1000),
            "HIGH": np.random.uniform(1.5, 2.5, 1000),
            "LOW": np.random.uniform(0.5, 1.5, 1000),
            "CLOSE": np.random.uniform(1.0, 2.0, 1000)
        })
        
        start_time = time.time()
        ti_results = calculator.calculate_technical_indicators(
            large_ohlc, self.test_config["generator_ohlc_feature_names"]
        )
        ti_time = time.time() - start_time
        
        # Should complete in reasonable time (< 5 seconds)
        self.assertLess(ti_time, 5.0)
        self.assertIsNotNone(ti_results)


class TestGeneratorPluginModuleStructure(unittest.TestCase):
    """Test module structure and organization."""
    
    def test_module_imports(self):
        """Test that all modules can be imported successfully."""
        try:
            from tsg_plugins.generator_plugin import GeneratorPlugin
            from tsg_plugins.generator_plugin.normalization_handler import NormalizationHandler
            from tsg_plugins.generator_plugin.model_loader import ModelLoader
            from tsg_plugins.generator_plugin.feature_processor import FeatureProcessor
            from tsg_plugins.generator_plugin.technical_indicator_calculator import TechnicalIndicatorCalculator
            from tsg_plugins.generator_plugin.data_generator import DataGenerator
            from tsg_plugins.generator_plugin.sequence_builder import SequenceBuilder
            from tsg_plugins.generator_plugin.feature_validator import FeatureValidator
            from tsg_plugins.generator_plugin.initial_data_handler import InitialDataHandler
            
            # If we get here, all imports succeeded
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Module import failed: {e}")
    
    def test_module_line_counts(self):
        """Test that modules are within target line counts."""
        import inspect
        
        # Define maximum line counts for each module
        max_lines = {
            'NormalizationHandler': 250,
            'ModelLoader': 200,
            'FeatureProcessor': 200,
            'TechnicalIndicatorCalculator': 250,
            'DataGenerator': 200,
            'SequenceBuilder': 200,
            'FeatureValidator': 200,
            'InitialDataHandler': 200
        }
        
        # Get source files and count lines
        from tsg_plugins.generator_plugin import normalization_handler, model_loader, feature_processor
        from tsg_plugins.generator_plugin import technical_indicator_calculator, data_generator, sequence_builder
        from tsg_plugins.generator_plugin import feature_validator, initial_data_handler
        
        modules_to_check = [
            (normalization_handler.NormalizationHandler, 'NormalizationHandler'),
            (model_loader.ModelLoader, 'ModelLoader'),
            (feature_processor.FeatureProcessor, 'FeatureProcessor'),
            (technical_indicator_calculator.TechnicalIndicatorCalculator, 'TechnicalIndicatorCalculator'),
            (data_generator.DataGenerator, 'DataGenerator'),
            (sequence_builder.SequenceBuilder, 'SequenceBuilder'),
            (feature_validator.FeatureValidator, 'FeatureValidator'),
            (initial_data_handler.InitialDataHandler, 'InitialDataHandler')
        ]
        
        for module_class, module_name in modules_to_check:
            try:
                source_lines = inspect.getsourcelines(module_class)[0]
                line_count = len(source_lines)
                max_allowed = max_lines.get(module_name, 200)
                
                self.assertLessEqual(
                    line_count, 
                    max_allowed,
                    f"{module_name} has {line_count} lines, should be <= {max_allowed}"
                )
            except OSError:
                # Skip if source is not available (compiled modules)
                pass


if __name__ == '__main__':
    unittest.main()
