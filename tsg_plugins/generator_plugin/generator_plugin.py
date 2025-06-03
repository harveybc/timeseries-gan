#!/usr/bin/env python3
"""
Generator Plugin - Main Interface

Main plugin interface that orchestrates specialized modules for synthetic data generation.
Maintains mandatory plugin structure while delegating to focused modules.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from typing import Dict, Any, List, Optional
import logging

from .model_loader import ModelLoader
from .normalization_handler import NormalizationHandler
from .feature_validator import FeatureValidator
from .initial_data_handler import InitialDataHandler
from .data_generator import DataGenerator
from .technical_indicator_calculator import TechnicalIndicatorCalculator
from .sequence_builder import SequenceBuilder


class GeneratorPlugin:
    """
    Main generator plugin interface following extreme separation of concerns.
    Orchestrates specialized modules for focused functionality.
    """
    
    # Mandatory plugin parameters
    plugin_params = {
        "sequential_model_file": None,
        "decoder_input_window_size": 144,
        "full_feature_names_ordered": [],
        "decoder_output_feature_names": [],
        "ohlc_feature_names": ["OPEN", "HIGH", "LOW", "CLOSE"],
        "ti_feature_names": [],
        "date_conditional_feature_names": [],
        "feeder_conditional_feature_names": [],
        "ti_calculation_min_lookback": 200,
        "ti_params": {},
        "decoder_input_name_latent": "decoder_input_z_seq",
        "decoder_input_name_window": "input_x_window",
        "decoder_input_name_conditions": "decoder_input_conditions",
        "decoder_input_name_context": "decoder_input_h_context",
        "generator_normalization_params_file": None
    }
    
    plugin_debug_vars = [
        "sequential_model_file", "decoder_input_window_size", "batch_size_inference",
        "full_feature_names_ordered", "decoder_output_feature_names",
        "ti_calculation_min_lookback"
    ]
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the GeneratorPlugin with modular architecture.
        
        Args:
            config: Configuration dictionary
        """
        if config is None:
            raise ValueError("Configuration dictionary ('config') is required.")
        
        # Initialize parameters and main config
        self.params = self.plugin_params.copy()
        self.main_config = config.copy()
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize core attributes
        self.sequential_model: Optional[Model] = None
        self.model: Optional[Model] = None  # Alias for sequential_model
        self.feature_to_idx: Dict[str, int] = {}
        self.num_all_features: int = 0
        
        # Determine initial close file path from config
        initial_close_file_path = config.get("x_train_file", config.get("real_data_file"))
        
        # Initialize specialized modules
        self._initialize_modules()
        
        # Set parameters and validate
        self.set_params(**config)
        
        # Validate configuration
        self._validate_plugin_configuration()
        
        # Load initial close anchor
        if self.initial_data_handler.get_initial_close_anchor() is None:
            self.initial_data_handler.load_initial_close_anchor(initial_close_file_path)
    
    def _initialize_modules(self) -> None:
        """Initialize all specialized modules."""
        # Normalization handler (needs to be first)
        self.normalization_handler = NormalizationHandler(self.params, self.logger)
        
        # Model loader
        self.model_loader = ModelLoader(self.params, self.logger)
        
        # Initial data handler
        self.initial_data_handler = InitialDataHandler(self.normalization_handler)
        
        # Other modules will be initialized after feature validation
        self.feature_validator = None
        self.data_generator = None
        self.ti_calculator = None
        self.sequence_builder = None
    
    def _initialize_feature_dependent_modules(self) -> None:
        """Initialize modules that depend on feature configuration."""
        if not self.params.get("full_feature_names_ordered"):
            return
        
        # Feature validator
        self.feature_validator = FeatureValidator(self.params["full_feature_names_ordered"])
        
        # Create feature mapping
        self.feature_to_idx = self.feature_validator.create_feature_index_mapping()
        self.num_all_features = self.feature_validator.get_num_features()
        
        # Technical indicator calculator
        self.ti_calculator = TechnicalIndicatorCalculator(
            self.params["ti_feature_names"], 
            self.params["ti_params"]
        )
        
        # Data generator
        self.data_generator = DataGenerator(
            self.params, self.feature_to_idx, 
            self.normalization_handler, self.ti_calculator
        )
        self.data_generator.set_main_config(self.main_config)
        
        # Sequence builder
        self.sequence_builder = SequenceBuilder(
            self.params, self.feature_to_idx, self.num_all_features,
            self.normalization_handler, self.ti_calculator
        )
    
    def set_params(self, **kwargs) -> None:
        """
        Update plugin parameters and reload components as needed.
        
        Args:
            **kwargs: Parameter updates
        """
        print(f"GeneratorPlugin.set_params called with kwargs: {list(kwargs.keys())}")
        
        # Store old values for change detection
        old_model_file = self.params.get("sequential_model_file")
        old_norm_file = self.params.get("generator_normalization_params_file")
        old_full_feature_names = self.params.get("full_feature_names_ordered")
        
        # Update main config
        if hasattr(self, 'main_config') and self.main_config is not None:
            self.main_config.update(kwargs)
        else:
            self.main_config = kwargs.copy()
        
        # Update plugin parameters (handle both prefixed and non-prefixed)
        for param_key in self.plugin_params.keys():
            prefixed_key = f"generator_{param_key}"
            
            if prefixed_key in kwargs:
                self.params[param_key] = kwargs[prefixed_key]
            elif param_key in kwargs:
                self.params[param_key] = kwargs[param_key]
        
        # Handle special normalization parameter
        if "generator_normalization_params_file" in kwargs:
            self.params["generator_normalization_params_file"] = kwargs["generator_normalization_params_file"]
        
        # Check for model changes
        new_model_file = self.params.get("sequential_model_file")
        if new_model_file != old_model_file or (new_model_file and self.sequential_model is None):
            self._load_model(new_model_file)
        elif not new_model_file and old_model_file:
            print("GeneratorPlugin: Model path cleared. Cleaning loaded model.")
            self.sequential_model = None
            self.model = None
        
        # Check for normalization parameter changes
        new_norm_file = self.params.get("generator_normalization_params_file")
        if new_norm_file != old_norm_file:
            if new_norm_file:
                self.normalization_handler.load_normalization_params(new_norm_file)
            else:
                self.normalization_handler.normalization_params = None
        
        # Check for feature configuration changes
        if (self.params.get("full_feature_names_ordered") != old_full_feature_names or
            self._features_config_changed(kwargs)):
            if self.params.get("full_feature_names_ordered"):
                self._initialize_feature_dependent_modules()
                if self.feature_validator:
                    self.feature_validator.validate_feature_name_consistency(self.params)
        
        # Handle initial close anchor reload
        self._handle_initial_close_anchor_reload()
    
    def _load_model(self, model_path: str) -> None:
        """Load model using model loader."""
        if not model_path:
            self.sequential_model = None
            self.model = None
            print("GeneratorPlugin: Warning - Attempted to load model with empty path.")
            return
        
        loaded_model = self.model_loader.load_model_from_path(model_path)
        if loaded_model is not None:
            self.sequential_model = loaded_model
            self.model = loaded_model  # Maintain alias
            print(f"GeneratorPlugin: Model successfully loaded from {model_path}")
        else:
            self.sequential_model = None
            self.model = None
            raise IOError(f"Failed to load model from {model_path}")
    
    def _features_config_changed(self, kwargs: Dict[str, Any]) -> bool:
        """Check if any feature configuration parameters changed."""
        feature_config_keys = [
            "decoder_output_feature_names", "ohlc_feature_names", 
            "ti_feature_names", "date_conditional_feature_names", 
            "feeder_conditional_feature_names",
            "generator_decoder_output_feature_names", 
            "generator_ohlc_feature_names",
            "generator_ti_feature_names",
            "generator_date_conditional_feature_names",
            "generator_feeder_conditional_feature_names"
        ]
        return any(key in kwargs for key in feature_config_keys)
    
    def _handle_initial_close_anchor_reload(self) -> None:
        """Handle reloading of initial close anchor if needed."""
        initial_close_file_path = self.main_config.get(
            "x_train_file", self.main_config.get("real_data_file")
        )
        
        if self.initial_data_handler.should_reload_close_anchor(initial_close_file_path):
            print(f"GeneratorPlugin: Reloading initial close anchor from: {initial_close_file_path}")
            self.initial_data_handler.load_initial_close_anchor(initial_close_file_path)
            self.initial_data_handler.update_last_loaded_path(initial_close_file_path)
        elif not initial_close_file_path and self.initial_data_handler.get_initial_close_anchor() is not None:
            print("GeneratorPlugin: Warning - Initial close file path became None, keeping existing anchor.")
            self.initial_data_handler.update_last_loaded_path(None)
    
    def _validate_plugin_configuration(self) -> None:
        """Validate critical plugin configuration."""
        model_path = self.params.get("sequential_model_file")
        if not model_path:
            raise ValueError("'sequential_model_file' parameter is required and cannot be empty.")
        
        if self.sequential_model is None:
            raise IOError(f"Sequential model could not be loaded from {model_path}")
        
        if not self.params.get("full_feature_names_ordered"):
            raise ValueError("'full_feature_names_ordered' parameter is required.")
        
        if not self.params.get("decoder_output_feature_names"):
            raise ValueError("'decoder_output_feature_names' parameter is required.")
    
    def generate(self, feeder_outputs_sequence: List[Dict[str, np.ndarray]],
                sequence_length_T: int,
                initial_full_feature_window: Optional[np.ndarray] = None,
                initial_datetimes_for_window: Optional[pd.Series] = None,
                true_prev_close_for_initial_window_log_return: Optional[float] = None) -> np.ndarray:
        """
        Generate synthetic sequence using the loaded model and feeder outputs.
        
        Args:
            feeder_outputs_sequence: List of feeder outputs for each time step
            sequence_length_T: Length of sequence to generate
            initial_full_feature_window: Optional initial feature window
            initial_datetimes_for_window: Optional datetime series for initial window
            true_prev_close_for_initial_window_log_return: Previous close for log return
            
        Returns:
            Generated sequence array with shape (1, sequence_length, num_features)
        """
        if not self.sequence_builder:
            raise RuntimeError("SequenceBuilder not initialized. Ensure feature configuration is complete.")
        
        # Setup initial window
        current_input_feature_window = self.initial_data_handler.setup_initial_window_data(
            initial_full_feature_window,
            self.params["decoder_input_window_size"],
            self.num_all_features
        )
        
        # Extract OHLC history from initial window
        ohlc_history_for_ti_list = self.initial_data_handler.extract_ohlc_history_from_window(
            current_input_feature_window,
            self.params["ohlc_feature_names"],
            self.feature_to_idx,
            self.params["ti_calculation_min_lookback"]
        )
        
        # Pre-fill derived features if datetime info is available
        if (initial_datetimes_for_window is not None and 
            len(initial_datetimes_for_window) == self.params["decoder_input_window_size"]):
            
            self.data_generator.pre_fill_derived_features_in_window(
                current_input_feature_window,
                initial_datetimes_for_window,
                ohlc_history_for_ti_list,
                true_prev_close_for_initial_window_log_return
            )
        
        # Generate the sequence
        return self.sequence_builder.build_sequence(
            self.sequential_model,
            feeder_outputs_sequence,
            sequence_length_T,
            current_input_feature_window,
            ohlc_history_for_ti_list
        )
    
    def get_debug_info(self) -> Dict[str, Any]:
        """
        Return debug information dictionary.
        
        Returns:
            Dictionary with debug variable values
        """
        return {var: self.params.get(var) for var in self.plugin_debug_vars}
    
    def add_debug_info(self, debug_info: Dict[str, Any]) -> None:
        """
        Add debug information to provided dictionary.
        
        Args:
            debug_info: Dictionary to update with debug info
        """
        debug_info.update(self.get_debug_info())
    
    def get_model(self) -> Optional[Model]:
        """
        Get the loaded Keras generator model.
        
        Returns:
            The loaded Keras model or None if not loaded
        """
        return self.model
    
    def update_model(self, new_model: Model) -> None:
        """
        Update the generator model (used by GANTrainerPlugin).
        
        Args:
            new_model: New Keras model to use
        """
        if not isinstance(new_model, Model):
            raise TypeError(f"new_model must be a Keras Model, received {type(new_model)}")
        
        print("GeneratorPlugin: Updating sequential_model with new model instance.")
        self.sequential_model = new_model
        self.model = new_model  # Maintain alias consistency
