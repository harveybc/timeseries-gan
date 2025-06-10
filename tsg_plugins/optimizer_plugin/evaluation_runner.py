"""
Evaluation Runner Module

Runs evaluation of individuals using plugins and computes fitness scores.
"""

import logging
import copy
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Union, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class EvaluationRunner:
    """Runs evaluation of genetic algorithm individuals."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize evaluation runner."""
        self.config = config
        self.n_samples = config.get('optimizer_n_samples_per_eval', 100)
        self.evaluation_count = 0
        
        # Store plugin references
        self.feeder_plugin = None
        self.generator_plugin = None
        self.evaluator_plugin = None
        
        logger.info(f"EvaluationRunner initialized with {self.n_samples} samples per evaluation")
    
    def set_plugins(self, feeder_plugin: Any, generator_plugin: Any, evaluator_plugin: Any):
        """Set plugin references for evaluation."""
        self.feeder_plugin = feeder_plugin
        self.generator_plugin = generator_plugin
        self.evaluator_plugin = evaluator_plugin
        
        logger.info("Plugins set for evaluation")
    
    def evaluate_individual(self, params: Dict[str, Union[int, float]]) -> float:
        """
        Evaluate a single individual (parameter set).
        
        Args:
            params: Dictionary of hyperparameters
            
        Returns:
            float: Fitness score (lower is better)
        """
        self.evaluation_count += 1
        
        try:
            logger.debug(f"Evaluating individual {self.evaluation_count}: {params}")
            
            # Create temporary config with these parameters
            eval_config = copy.deepcopy(self.config)
            eval_config.update(params)
            
            # Generate synthetic data using the parameters
            synthetic_data = self._generate_synthetic_data(eval_config)
            if synthetic_data is None:
                logger.warning(f"Failed to generate synthetic data for params: {params}")
                return float('inf')
            
            # Evaluate the synthetic data quality
            fitness = self._evaluate_synthetic_data(synthetic_data, eval_config)
            
            logger.debug(f"Individual {self.evaluation_count} fitness: {fitness}")
            return fitness
            
        except Exception as e:
            logger.error(f"Error evaluating individual {self.evaluation_count}: {str(e)}")
            return float('inf')
    
    def _generate_synthetic_data(self, config: Dict[str, Any]) -> np.ndarray:
        """Generate synthetic data using current parameters."""
        try:
            # Create temporary plugins with new config
            temp_feeder = copy.deepcopy(self.feeder_plugin)
            temp_generator = copy.deepcopy(self.generator_plugin)
            
            # Update their configurations
            temp_feeder.set_params(**config)
            temp_generator.set_params(**config)
            
            # Generate latent vectors
            latent_dim = config.get('latent_dim', 64)
            feeder_output = temp_feeder.generate(n_samples=self.n_samples, latent_dim=latent_dim)
            
            if feeder_output is None:
                logger.warning("Feeder plugin returned None")
                return None
            
            # Generate synthetic data
            # Get initial window size
            window_size = temp_generator.params.get("decoder_input_window_size", 60)
            feature_count = len(temp_generator.params.get("full_feature_names_ordered", []))
            
            if feature_count == 0:
                logger.warning("No features defined in generator")
                return None
            
            # Create initial window
            initial_window = np.zeros((window_size, feature_count), dtype=np.float32)
            
            # Generate sequence
            synthetic_batch = temp_generator.generate(
                feeder_outputs_sequence=feeder_output,
                sequence_length_T=self.n_samples,
                initial_full_feature_window=initial_window
            )
            
            if synthetic_batch is None or len(synthetic_batch) == 0:
                logger.warning("Generator returned empty result")
                return None
            
            return synthetic_batch[0]  # Return first batch
            
        except Exception as e:
            logger.error(f"Error in synthetic data generation: {str(e)}")
            return None
    
    def _evaluate_synthetic_data(self, synthetic_data: np.ndarray, config: Dict[str, Any]) -> float:
        """Evaluate quality of synthetic data."""
        try:
            # Create temporary evaluator
            temp_evaluator = copy.deepcopy(self.evaluator_plugin)
            temp_evaluator.set_params(**config)
            
            # Run evaluation
            metrics = temp_evaluator.evaluate(synthetic_data=synthetic_data)
            
            if metrics is None or not isinstance(metrics, dict):
                logger.warning("Evaluator returned invalid metrics")
                return float('inf')
            
            # Extract fitness score (assuming lower is better)
            # This could be MSE, MAE, or a composite score
            fitness = metrics.get('mse', float('inf'))
            
            # Handle invalid fitness values
            if not isinstance(fitness, (int, float)) or np.isnan(fitness) or np.isinf(fitness):
                logger.warning(f"Invalid fitness value: {fitness}")
                return float('inf')
            
            return float(fitness)
            
        except Exception as e:
            logger.error(f"Error in synthetic data evaluation: {str(e)}")
            return float('inf')
    
    def _generate_evaluation_timestamps(self, n_samples: int) -> pd.Series:
        """Generate timestamps for evaluation data."""
        try:
            # Get periodicity from config
            periodicity = self.config.get('dataset_periodicity', '1h')
            start_time = self.config.get('optimizer_start_datetime', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            
            # Parse start time
            start_dt = pd.to_datetime(start_time)
            
            # Determine time delta
            if 'h' in periodicity.lower():
                hours = int(periodicity.lower().replace('h', '').replace('our', ''))
                delta = timedelta(hours=hours)
            elif 'min' in periodicity.lower() or 'm' in periodicity.lower():
                minutes = int(periodicity.lower().replace('min', '').replace('m', ''))
                delta = timedelta(minutes=minutes)
            elif 'd' in periodicity.lower():
                days = int(periodicity.lower().replace('d', '').replace('ay', ''))
                delta = timedelta(days=days)
            else:
                delta = timedelta(hours=1)  # Default
            
            # Generate timestamps
            timestamps = [start_dt + i * delta for i in range(n_samples)]
            return pd.Series(timestamps)
            
        except Exception as e:
            logger.warning(f"Error generating timestamps: {str(e)}, using default")
            return pd.Series(pd.date_range(start='2024-01-01', periods=n_samples, freq='h'))
    
    def get_evaluation_stats(self) -> Dict[str, Any]:
        """Get evaluation statistics."""
        return {
            'total_evaluations': self.evaluation_count,
            'samples_per_evaluation': self.n_samples
        }
