"""
Feeder Plugin

Main interface for the feeder plugin that coordinates data feeding operations.
Integrates encoder handling, data preprocessing, and condition management.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, Union
import os

from .encoder_handler import EncoderHandler
from .data_preprocessor import DataPreprocessor
from .condition_manager import ConditionManager

logger = logging.getLogger(__name__)


class FeederPlugin:
    """
    Main feeder plugin interface.
    
    Coordinates data feeding operations by integrating:
    - Encoder model management and latent encoding
    - Data preprocessing and normalization
    - Condition extraction and processing
    """
    
    # Default plugin configuration
    plugin_params: Dict[str, Any] = {
        'latent_dim': 64,
        'normalization_method': 'standard',
        'handle_missing': 'interpolate',
        'outlier_method': 'clip',
        'outlier_threshold': 3.0,
        'condition_columns': [],
        'condition_method': 'concatenate',
        'condition_dim': 0,
        'use_temporal_conditions': False
    }
    plugin_debug_vars = ['latent_dim', 'normalization_method', 'handle_missing', 'outlier_method']
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the feeder plugin."""
        # Mandatory parameter initialization
        self.params = self.plugin_params.copy()
        if config:
            self.params.update(config)
        self.config = config
        
        # Prepare condition manager configuration
        # Map feeder-specific config keys to ConditionManager keys
        condition_config = config.copy()
        condition_config['condition_columns'] = config.get('feeder_fundamental_features_for_conditioning', [])
        condition_config['use_temporal_conditions'] = True  # Always use temporal conditions for feeder
        
        # Initialize sub-modules
        self.encoder_handler = EncoderHandler(config)
        self.data_preprocessor = DataPreprocessor(config)
        self.condition_manager = ConditionManager(condition_config)
        
        # Plugin state
        self.is_initialized = False
        self.encoder_loaded = False
        self.preprocessor_fitted = False
        
        # Data tracking
        self.last_input_shape = None
        self.last_output_shape = None
        self.processed_samples = 0
        
        logger.info("FeederPlugin initialized")
    
    def generate(self, n_ticks_to_generate: int, target_datetimes: pd.Series) -> Optional[np.ndarray]:
        """
        Generate conditional features for a sequence of target datetimes.

        Args:
            n_ticks_to_generate: The number of ticks (samples) for which to generate conditions.
            target_datetimes: A pandas Series of datetimes for which to generate conditions.

        Returns:
            Optional[np.ndarray]: An array of processed conditional features, or None if generation fails.
        """
        logger.info(f"FeederPlugin: Generating conditional features for {n_ticks_to_generate} ticks.")

        if not hasattr(self, 'condition_manager') or self.condition_manager is None:
            logger.error("ConditionManager not initialized in FeederPlugin.")
            return None

        if target_datetimes is None or len(target_datetimes) != n_ticks_to_generate:
            logger.error(
                f"Target datetimes are invalid or length mismatch. Expected {n_ticks_to_generate}, "
                f"got {len(target_datetimes) if target_datetimes is not None else 'None'}."
            )
            return None

        datetime_col_name = self.config.get("datetime_col_name", "DATE_TIME")
        data_df = pd.DataFrame({datetime_col_name: target_datetimes})

        # Extract raw conditions (e.g., cyclical features from datetimes)
        # This relies on ConditionManager being configured to use temporal conditions
        # and knowing which datetime column to use (timestamp_col).
        raw_conditions = self.condition_manager.extract_conditions(data_df, timestamp_col=datetime_col_name)
        if raw_conditions is None:
            logger.error("FeederPlugin: Failed to extract raw conditions via ConditionManager.")
            return None

        # Process conditions (e.g., normalization if applicable and configured)
        # This step assumes that if scaling is involved, the ConditionManager's
        # scalers have been appropriately fitted (e.g., during FeederPlugin.initialize with sample_data)
        # or are not needed for the type of conditions being generated (e.g., pre-normalized cyclical features).
        processed_conditions = self.condition_manager.process_conditions(raw_conditions)
        if processed_conditions is None:
            logger.error("FeederPlugin: Failed to process conditions via ConditionManager.")
            return None
        
        if processed_conditions.shape[0] != n_ticks_to_generate:
            logger.error(
                f"FeederPlugin: Shape mismatch in processed conditions. Expected {n_ticks_to_generate} samples, "
                f"got {processed_conditions.shape[0]}. Raw conditions shape: {raw_conditions.shape}"
            )
            return None

        logger.info(f"FeederPlugin: Successfully generated processed conditions with shape: {processed_conditions.shape}")
        return processed_conditions

    def initialize(self, encoder_path: str, sample_data: Optional[pd.DataFrame] = None) -> bool:
        """
        Initialize the feeder plugin with encoder model and sample data.
        
        Args:
            encoder_path: Path to the encoder model file
            sample_data: Sample data for preprocessing fitting and condition setup
            
        Returns:
            bool: True if initialization successful
        """
        try:
            # Load encoder model
            if not self.encoder_handler.load_model(encoder_path):
                logger.error("Failed to load encoder model")
                return False
            
            if not self.encoder_handler.validate_model():
                logger.error("Encoder model validation failed")
                return False
            
            self.encoder_loaded = True
            
            # Initialize condition manager
            if not self.condition_manager.initialize(sample_data):
                logger.error("Failed to initialize condition manager")
                return False
            
            # Fit preprocessor if sample data provided
            if sample_data is not None:
                if not self._fit_preprocessor(sample_data):
                    logger.error("Failed to fit data preprocessor")
                    return False
            
            self.is_initialized = True
            logger.info("FeederPlugin initialization completed successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize FeederPlugin: {str(e)}")
            return False
    
    def process_data(self, 
                    data: Union[np.ndarray, pd.DataFrame],
                    timestamp_col: Optional[str] = None,
                    fit_preprocessor: bool = False) -> Optional[Dict[str, np.ndarray]]:
        """
        Process input data through the complete feeder pipeline.
        
        Args:
            data: Input data to process
            timestamp_col: Name of timestamp column for condition extraction
            fit_preprocessor: Whether to fit preprocessor on this data
            
        Returns:
            Optional[Dict]: Processed data containing latents, conditions, and metadata
        """
        if not self.is_initialized:
            logger.error("FeederPlugin not initialized. Call initialize() first.")
            return None
        
        try:
            # Convert to DataFrame if needed for condition extraction
            if isinstance(data, np.ndarray):
                data_df = pd.DataFrame(data)
            else:
                data_df = data.copy()
            
            self.last_input_shape = data_df.shape
            
            # Fit preprocessor if requested
            if fit_preprocessor:
                if not self._fit_preprocessor(data_df):
                    logger.error("Failed to fit preprocessor")
                    return None
            
            # Step 1: Preprocess data
            processed_data = self._preprocess_data(data_df)
            if processed_data is None:
                logger.error("Data preprocessing failed")
                return None
            
            # Step 2: Encode to latent space
            latent_vectors = self._encode_data(processed_data)
            if latent_vectors is None:
                logger.error("Data encoding failed")
                return None
            
            # Step 3: Extract conditions
            conditions = self._extract_conditions(data_df, timestamp_col)
            
            # Step 4: Process conditions
            processed_conditions = self._process_conditions(conditions)
            
            # Step 5: Combine latents with conditions
            combined_vectors = self._combine_vectors(latent_vectors, processed_conditions)
            
            # Update tracking
            self.last_output_shape = combined_vectors.shape if combined_vectors is not None else latent_vectors.shape
            self.processed_samples += len(data_df)
            
            # Prepare result
            result = {
                'latents': latent_vectors,
                'conditions': processed_conditions,
                'combined': combined_vectors,
                'metadata': {
                    'input_shape': self.last_input_shape,
                    'output_shape': self.last_output_shape,
                    'has_conditions': processed_conditions is not None,
                    'latent_dim': latent_vectors.shape[1] if latent_vectors is not None else 0,
                    'condition_dim': processed_conditions.shape[1] if processed_conditions is not None else 0
                }
            }
            
            logger.debug(f"Processed {len(data_df)} samples successfully")
            return result
            
        except Exception as e:
            logger.error(f"Failed to process data: {str(e)}")
            return None
    
    def encode_only(self, data: Union[np.ndarray, pd.DataFrame]) -> Optional[np.ndarray]:
        """
        Encode data to latent space without condition processing.
        
        Args:
            data: Input data to encode
            
        Returns:
            Optional[np.ndarray]: Latent vectors or None if failed
        """
        if not self.encoder_loaded:
            logger.error("No encoder model loaded")
            return None
        
        try:
            # Convert to DataFrame for preprocessing
            if isinstance(data, np.ndarray):
                data_df = pd.DataFrame(data)
            else:
                data_df = data.copy()
            
            # Preprocess and encode
            processed_data = self._preprocess_data(data_df)
            if processed_data is None:
                return None
            
            latent_vectors = self._encode_data(processed_data)
            return latent_vectors
            
        except Exception as e:
            logger.error(f"Failed to encode data: {str(e)}")
            return None
    
    def _fit_preprocessor(self, data: pd.DataFrame) -> bool:
        """Fit the data preprocessor on training data."""
        try:
            # Select numeric columns only for preprocessing
            numeric_data = data.select_dtypes(include=[np.number])
            
            if len(numeric_data.columns) == 0:
                logger.error("No numeric columns found for preprocessing")
                return False
            
            success = self.data_preprocessor.fit(numeric_data)
            if success:
                self.preprocessor_fitted = True
                logger.info("Data preprocessor fitted successfully")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to fit preprocessor: {str(e)}")
            return False
    
    def _preprocess_data(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        """Preprocess data using the fitted preprocessor."""
        try:
            if not self.preprocessor_fitted:
                logger.warning("Preprocessor not fitted, attempting to fit on current data")
                if not self._fit_preprocessor(data):
                    return None
            
            # Select numeric columns
            numeric_data = data.select_dtypes(include=[np.number])
            
            if len(numeric_data.columns) == 0:
                logger.error("No numeric columns found for preprocessing")
                return None
            
            # Validate data
            if not self.data_preprocessor.validate_data(numeric_data):
                logger.error("Data validation failed")
                return None
            
            # Transform data
            processed = self.data_preprocessor.transform(numeric_data)
            return processed
            
        except Exception as e:
            logger.error(f"Failed to preprocess data: {str(e)}")
            return None
    
    def _encode_data(self, data: np.ndarray) -> Optional[np.ndarray]:
        """Encode preprocessed data to latent space."""
        try:
            if not self.encoder_loaded:
                logger.error("No encoder model loaded")
                return None
            
            latent_vectors = self.encoder_handler.encode_data(data)
            return latent_vectors
            
        except Exception as e:
            logger.error(f"Failed to encode data: {str(e)}")
            return None
    
    def _extract_conditions(self, data: pd.DataFrame, timestamp_col: Optional[str]) -> Optional[np.ndarray]:
        """Extract condition vectors from data."""
        try:
            conditions = self.condition_manager.extract_conditions(data, timestamp_col)
            return conditions
            
        except Exception as e:
            logger.error(f"Failed to extract conditions: {str(e)}")
            return None
    
    def _process_conditions(self, conditions: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Process and normalize condition vectors."""
        try:
            if conditions is None:
                return None
            
            processed = self.condition_manager.process_conditions(conditions)
            return processed
            
        except Exception as e:
            logger.error(f"Failed to process conditions: {str(e)}")
            return None
    
    def _combine_vectors(self, latents: np.ndarray, conditions: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Combine latent vectors with condition vectors."""
        try:
            combined = self.condition_manager.combine_with_latents(latents, conditions)
            return combined
            
        except Exception as e:
            logger.error(f"Failed to combine vectors: {str(e)}")
            return None
    
    def get_plugin_info(self) -> Dict[str, Any]:
        """Get comprehensive plugin information."""
        return {
            'plugin_type': 'feeder',
            'initialized': self.is_initialized,
            'encoder_loaded': self.encoder_loaded,
            'preprocessor_fitted': self.preprocessor_fitted,
            'processed_samples': self.processed_samples,
            'last_input_shape': self.last_input_shape,
            'last_output_shape': self.last_output_shape,
            'encoder_info': self.encoder_handler.get_model_info(),
            'preprocessor_config': self.data_preprocessor.get_config(),
            'condition_info': self.condition_manager.get_condition_info()
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics from all modules."""
        return {
            'encoder_stats': self.encoder_handler.get_latent_stats(),
            'preprocessor_stats': self.data_preprocessor.get_statistics(),
            'condition_info': self.condition_manager.get_condition_info(),
            'processing_stats': {
                'processed_samples': self.processed_samples,
                'last_input_shape': self.last_input_shape,
                'last_output_shape': self.last_output_shape
            }
        }
    
    def reset(self):
        """Reset all plugin components to initial state."""
        try:
            self.encoder_handler.cleanup()
            self.data_preprocessor.reset()
            self.condition_manager.reset()
            
            # Reset plugin state
            self.is_initialized = False
            self.encoder_loaded = False
            self.preprocessor_fitted = False
            self.last_input_shape = None
            self.last_output_shape = None
            self.processed_samples = 0
            
            logger.info("FeederPlugin reset completed")
            
        except Exception as e:
            logger.error(f"Failed to reset FeederPlugin: {str(e)}")
    
    def cleanup(self):
        """Clean up plugin resources."""
        self.reset()
        logger.info("FeederPlugin cleanup completed")
    
    def set_params(self, **kwargs):
        """Update plugin parameters with provided configuration."""
        # Store old values for change detection
        old_encoder_path = self.params.get('encoder_model_path')
        old_x_train_file = self.params.get('x_train_file')
        
        # Update parameters
        for key, value in kwargs.items():
            self.params[key] = value
        
        # Check for encoder path and attempt initialization
        encoder_path = self.params.get('encoder_model_path') or kwargs.get('encoder_model_path')
        x_train_file = self.params.get('x_train_file') or kwargs.get('x_train_file')
        
        # Initialize if we have an encoder path and it's different from before, or if we're not initialized yet
        if encoder_path and (encoder_path != old_encoder_path or not self.is_initialized):
            logger.info(f"FeederPlugin: Initializing with encoder path: {encoder_path}")
            
            # Load sample data if available for condition manager initialization
            sample_data = None
            if x_train_file and os.path.exists(x_train_file):
                try:
                    logger.info(f"FeederPlugin: Loading sample data from: {x_train_file}")
                    sample_data = pd.read_csv(x_train_file)
                    logger.info(f"FeederPlugin: Sample data loaded with shape: {sample_data.shape}")
                except Exception as e:
                    logger.warning(f"FeederPlugin: Failed to load sample data from {x_train_file}: {e}")
                    sample_data = None
            
            # Initialize the feeder plugin
            if not self.initialize(encoder_path, sample_data):
                logger.error("FeederPlugin: Initialization failed")
            else:
                logger.info("FeederPlugin: Initialization completed successfully")
        elif not encoder_path:
            logger.warning("FeederPlugin: No encoder_model_path provided, cannot initialize")

    def get_debug_info(self):
        """Return debug information for the plugin."""
        return {var: self.params.get(var) for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        """Add plugin debug information to the given dictionary."""
        debug_info.update(self.get_debug_info())
