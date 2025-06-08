"""
Metrics Evaluator Module

This module handles evaluation metrics computation for comparing synthetic and real data
in the timeseries-gan pipeline. It provides comprehensive statistical and ML-based
evaluation metrics.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.spatial.distance import jensenshannon
import warnings
import os
from datetime import datetime

logger = logging.getLogger(__name__)


class MetricsEvaluator:
    """
    Handles evaluation metrics computation for synthetic vs real data comparison.
    
    This class provides various statistical and machine learning based metrics
    to evaluate the quality of synthetic time series data.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the MetricsEvaluator.
        
        Args:
            config: Configuration dictionary containing evaluation parameters
        """
        self.config = config
        self.metrics_config = config.get('evaluation', {})
        self.statistical_tests = self.metrics_config.get('statistical_tests', True)
        self.ml_evaluation = self.metrics_config.get('ml_evaluation', True)
        self.distribution_tests = self.metrics_config.get('distribution_tests', True)
        
    def evaluate_synthetic_data(self, real_data: np.ndarray, 
                              synthetic_data: np.ndarray) -> Dict[str, Any]:
        """
        Comprehensive evaluation of synthetic data quality.
        
        Args:
            real_data: Original real data
            synthetic_data: Generated synthetic data
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        logger.info("Starting comprehensive evaluation of synthetic data")
        
        results = {
            'data_info': self._get_data_info(real_data, synthetic_data),
            'basic_statistics': {},
            'distribution_metrics': {},
            'correlation_metrics': {},
            'ml_metrics': {}
        }
        
        try:
            # Basic statistical comparison
            results['basic_statistics'] = self._compute_basic_statistics(
                real_data, synthetic_data
            )
            
            # Distribution comparison
            if self.distribution_tests:
                results['distribution_metrics'] = self._compute_distribution_metrics(
                    real_data, synthetic_data
                )
            
            # Correlation analysis
            results['correlation_metrics'] = self._compute_correlation_metrics(
                real_data, synthetic_data
            )
            
            # ML-based evaluation
            if self.ml_evaluation:
                results['ml_metrics'] = self._compute_ml_metrics(
                    real_data, synthetic_data
                )
            
            # Overall quality score
            results['overall_score'] = self._compute_overall_score(results)
            
            logger.info(f"Evaluation completed. Overall score: {results['overall_score']:.3f}")
            
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            results['error'] = str(e)
            
        return results
    
    def _get_data_info(self, real_data: np.ndarray, 
                      synthetic_data: np.ndarray) -> Dict[str, Any]:
        """Get basic information about the datasets."""
        return {
            'real_shape': real_data.shape,
            'synthetic_shape': synthetic_data.shape,
            'real_features': real_data.shape[1] if len(real_data.shape) > 1 else 1,
            'synthetic_features': synthetic_data.shape[1] if len(synthetic_data.shape) > 1 else 1,
            'shape_match': real_data.shape == synthetic_data.shape
        }
    
    def _compute_basic_statistics(self, real_data: np.ndarray, 
                                synthetic_data: np.ndarray) -> Dict[str, Any]:
        """Compute basic statistical comparisons."""
        logger.debug("Computing basic statistics")
        
        stats_real = {
            'mean': np.mean(real_data, axis=0),
            'std': np.std(real_data, axis=0),
            'min': np.min(real_data, axis=0),
            'max': np.max(real_data, axis=0),
            'median': np.median(real_data, axis=0)
        }
        
        stats_synthetic = {
            'mean': np.mean(synthetic_data, axis=0),
            'std': np.std(synthetic_data, axis=0),
            'min': np.min(synthetic_data, axis=0),
            'max': np.max(synthetic_data, axis=0),
            'median': np.median(synthetic_data, axis=0)
        }
        
        # Compute differences
        differences = {}
        for key in stats_real:
            diff = np.abs(stats_real[key] - stats_synthetic[key])
            differences[f'{key}_diff'] = np.mean(diff) if hasattr(diff, '__len__') else diff
        
        return {
            'real_stats': {k: v.tolist() if hasattr(v, 'tolist') else v 
                          for k, v in stats_real.items()},
            'synthetic_stats': {k: v.tolist() if hasattr(v, 'tolist') else v 
                               for k, v in stats_synthetic.items()},
            'differences': differences
        }
    
    def _compute_distribution_metrics(self, real_data: np.ndarray, 
                                    synthetic_data: np.ndarray) -> Dict[str, Any]:
        """Compute distribution comparison metrics."""
        logger.debug("Computing distribution metrics")
        
        metrics = {}
        
        try:
            # Flatten data for distribution comparison
            real_flat = real_data.flatten()
            synthetic_flat = synthetic_data.flatten()
            
            # Jensen-Shannon divergence
            # Create histograms with same bins
            bins = np.linspace(
                min(real_flat.min(), synthetic_flat.min()),
                max(real_flat.max(), synthetic_flat.max()),
                50
            )
            
            real_hist, _ = np.histogram(real_flat, bins=bins, density=True)
            synthetic_hist, _ = np.histogram(synthetic_flat, bins=bins, density=True)
            
            # Normalize to probabilities
            real_hist = real_hist / np.sum(real_hist)
            synthetic_hist = synthetic_hist / np.sum(synthetic_hist)
            
            # Add small epsilon to avoid zero probabilities
            epsilon = 1e-10
            real_hist = real_hist + epsilon
            synthetic_hist = synthetic_hist + epsilon
            
            # Renormalize
            real_hist = real_hist / np.sum(real_hist)
            synthetic_hist = synthetic_hist / np.sum(synthetic_hist)
            
            js_divergence = jensenshannon(real_hist, synthetic_hist)
            metrics['jensen_shannon_divergence'] = js_divergence
            
            # Kolmogorov-Smirnov test
            if self.statistical_tests:
                ks_statistic, ks_pvalue = stats.ks_2samp(real_flat, synthetic_flat)
                metrics['ks_statistic'] = ks_statistic
                metrics['ks_pvalue'] = ks_pvalue
                
                # Anderson-Darling test (if sample sizes are reasonable)
                if len(real_flat) <= 10000 and len(synthetic_flat) <= 10000:
                    try:
                        ad_statistic, ad_critical, ad_significance = stats.anderson_ksamp(
                            [real_flat, synthetic_flat]
                        )
                        metrics['anderson_darling_statistic'] = ad_statistic
                        metrics['anderson_darling_critical'] = ad_critical.tolist()
                    except Exception as e:
                        logger.warning(f"Anderson-Darling test failed: {str(e)}")
            
        except Exception as e:
            logger.warning(f"Distribution metrics computation failed: {str(e)}")
            metrics['error'] = str(e)
        
        return metrics
    
    def _compute_correlation_metrics(self, real_data: np.ndarray, 
                                   synthetic_data: np.ndarray) -> Dict[str, Any]:
        """Compute correlation-based metrics."""
        logger.debug("Computing correlation metrics")
        
        metrics = {}
        
        try:
            # Only compute if data has multiple features
            if len(real_data.shape) > 1 and real_data.shape[1] > 1:
                real_corr = np.corrcoef(real_data.T)
                synthetic_corr = np.corrcoef(synthetic_data.T)
                
                # Correlation matrix difference (Frobenius norm)
                corr_diff = np.linalg.norm(real_corr - synthetic_corr, 'fro')
                metrics['correlation_matrix_diff'] = corr_diff
                
                # Average correlation difference
                corr_abs_diff = np.abs(real_corr - synthetic_corr)
                metrics['avg_correlation_diff'] = np.mean(corr_abs_diff)
                
            else:
                metrics['note'] = 'Single feature data - correlation metrics not applicable'
                
        except Exception as e:
            logger.warning(f"Correlation metrics computation failed: {str(e)}")
            metrics['error'] = str(e)
        
        return metrics
    
    def _compute_ml_metrics(self, real_data: np.ndarray, 
                          synthetic_data: np.ndarray) -> Dict[str, Any]:
        """Compute ML-based evaluation metrics."""
        logger.debug("Computing ML-based metrics")
        
        metrics = {}
        
        try:
            # Prepare data for ML evaluation
            # Use real data to train predictor, test on synthetic
            if len(real_data.shape) > 1 and real_data.shape[1] > 1:
                X_real = real_data[:, :-1]
                y_real = real_data[:, -1]
                
                X_synthetic = synthetic_data[:, :-1]
                y_synthetic = synthetic_data[:, -1]
                
                # Train-test split on real data
                X_train, X_test, y_train, y_test = train_test_split(
                    X_real, y_real, test_size=0.2, random_state=42
                )
                
                # Scale data
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                X_synthetic_scaled = scaler.transform(X_synthetic)
                
                # Train Random Forest on real data
                rf = RandomForestRegressor(n_estimators=100, random_state=42)
                rf.fit(X_train_scaled, y_train)
                
                # Predictions
                y_pred_real = rf.predict(X_test_scaled)
                y_pred_synthetic = rf.predict(X_synthetic_scaled)
                
                # Metrics
                mse_real = mean_squared_error(y_test, y_pred_real)
                mae_real = mean_absolute_error(y_test, y_pred_real)
                
                mse_synthetic = mean_squared_error(y_synthetic, y_pred_synthetic)
                mae_synthetic = mean_absolute_error(y_synthetic, y_pred_synthetic)
                
                metrics['predictor_performance'] = {
                    'real_mse': mse_real,
                    'real_mae': mae_real,
                    'synthetic_mse': mse_synthetic,
                    'synthetic_mae': mae_synthetic,
                    'mse_ratio': mse_synthetic / mse_real if mse_real > 0 else float('inf'),
                    'mae_ratio': mae_synthetic / mae_real if mae_real > 0 else float('inf')
                }
                
            else:
                metrics['note'] = 'Insufficient features for ML evaluation'
                
        except Exception as e:
            logger.warning(f"ML metrics computation failed: {str(e)}")
            metrics['error'] = str(e)
        
        return metrics
    
    def _compute_overall_score(self, results: Dict[str, Any]) -> float:
        """Compute an overall quality score (0-1, higher is better)."""
        logger.debug("Computing overall quality score")
        
        try:
            score_components = []
            
            # Basic statistics score (lower differences are better)
            if 'basic_statistics' in results and 'differences' in results['basic_statistics']:
                diffs = results['basic_statistics']['differences']
                # Average relative differences (normalized)
                avg_diff = np.mean(list(diffs.values()))
                stat_score = max(0, 1 - avg_diff)  # Assume differences are normalized
                score_components.append(stat_score)
            
            # Distribution score
            if 'distribution_metrics' in results:
                dist_metrics = results['distribution_metrics']
                if 'jensen_shannon_divergence' in dist_metrics:
                    js_score = max(0, 1 - dist_metrics['jensen_shannon_divergence'])
                    score_components.append(js_score)
            
            # Correlation score
            if 'correlation_metrics' in results:
                corr_metrics = results['correlation_metrics']
                if 'avg_correlation_diff' in corr_metrics:
                    corr_score = max(0, 1 - corr_metrics['avg_correlation_diff'])
                    score_components.append(corr_score)
            
            # ML score
            if 'ml_metrics' in results and 'predictor_performance' in results['ml_metrics']:
                ml_perf = results['ml_metrics']['predictor_performance']
                if 'mse_ratio' in ml_perf and ml_perf['mse_ratio'] != float('inf'):
                    # Good if ratio is close to 1
                    ml_score = max(0, 1 - abs(1 - ml_perf['mse_ratio']))
                    score_components.append(ml_score)
            
            # Return average of available scores, or 0.5 if no scores computed
            if score_components:
                return np.mean(score_components)
            else:
                return 0.5
                
        except Exception as e:
            logger.warning(f"Overall score computation failed: {str(e)}")
            return 0.5
    
    def save_evaluation_report(self, results: Dict[str, Any], output_path: str) -> None:
        """Save evaluation results to a file."""
        try:
            import json
            
            # Convert numpy arrays to lists for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                else:
                    return obj
            
            results_serializable = convert_numpy(results)
            
            with open(output_path, 'w') as f:
                json.dump(results_serializable, f, indent=2)
                
            logger.info(f"Evaluation report saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save evaluation report: {str(e)}")
    
    def evaluate(self, synthetic_data: pd.DataFrame, real_data_path: Optional[str] = None, 
                stage_name: str = "default") -> Dict[str, Any]:
        """
        Evaluate synthetic data against real data.
        
        Args:
            synthetic_data: Generated synthetic data as DataFrame
            real_data_path: Path to real data file for comparison (optional)
            stage_name: Name of the evaluation stage
            
        Returns:
            Dictionary containing evaluation results
        """
        logger.info(f"Starting evaluation for stage: {stage_name}")
        
        # Convert synthetic data to numpy for evaluation
        if isinstance(synthetic_data, pd.DataFrame):
            # Filter out datetime columns to avoid Timestamp issues
            numeric_columns = synthetic_data.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                synthetic_array = synthetic_data[numeric_columns].values
                logger.info(f"Using {len(numeric_columns)} numeric columns for evaluation")
            else:
                logger.warning("No numeric columns found in synthetic data")
                synthetic_array = np.array([])
        else:
            synthetic_array = synthetic_data
        
        # Load real data if path provided
        real_array = None
        if real_data_path and os.path.exists(real_data_path):
            try:
                if real_data_path.endswith('.csv'):
                    real_df = pd.read_csv(real_data_path)
                    # Remove datetime columns if present
                    datetime_cols = ['DATE_TIME', 'date_time', 'datetime']
                    for col in datetime_cols:
                        if col in real_df.columns:
                            real_df = real_df.drop(columns=[col])
                    real_array = real_df.values
                elif real_data_path.endswith('.npy'):
                    real_array = np.load(real_data_path)
                
                logger.info(f"Loaded real data with shape: {real_array.shape}")
            except Exception as e:
                logger.warning(f"Could not load real data from {real_data_path}: {e}")
        
        # Perform evaluation
        if real_array is not None:
            # Full evaluation with real data comparison
            results = self.evaluate_synthetic_data(real_array, synthetic_array)
        else:
            # Limited evaluation without real data
            results = self._evaluate_synthetic_only(synthetic_array)
        
        # Add metadata
        results['stage_name'] = stage_name
        results['evaluation_timestamp'] = datetime.now().isoformat()
        results['synthetic_data_shape'] = synthetic_array.shape
        
        logger.info(f"Evaluation completed for stage: {stage_name}")
        return results
    
    def _evaluate_synthetic_only(self, synthetic_data: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate synthetic data without real data comparison.
        
        Args:
            synthetic_data: Generated synthetic data
            
        Returns:
            Dictionary containing basic evaluation metrics
        """
        logger.info("Performing synthetic-only evaluation (no real data comparison)")
        
        return {
            'data_info': {
                'synthetic_shape': synthetic_data.shape,
                'synthetic_features': synthetic_data.shape[1] if len(synthetic_data.shape) > 1 else 1,
            },
            'basic_statistics': {
                'mean': np.mean(synthetic_data, axis=0).tolist(),
                'std': np.std(synthetic_data, axis=0).tolist(),
                'min': np.min(synthetic_data, axis=0).tolist(),
                'max': np.max(synthetic_data, axis=0).tolist(),
                'median': np.median(synthetic_data, axis=0).tolist()
            },
            'data_quality': {
                'has_nan': bool(np.isnan(synthetic_data).any()),
                'has_inf': bool(np.isinf(synthetic_data).any()),
                'finite_ratio': float(np.isfinite(synthetic_data).mean())
            },
            'overall_score': 0.5,  # Neutral score without comparison
            'evaluation_type': 'synthetic_only'
        }
