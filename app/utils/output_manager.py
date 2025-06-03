"""
Output Manager Module

This module handles output file management and data combination operations
for the timeseries-gan pipeline. It provides utilities for saving results,
managing output directories, and combining datasets.
"""

import logging
import os
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import json
import pickle
from datetime import datetime

logger = logging.getLogger(__name__)


class OutputManager:
    """
    Handles output file management and data combination operations.
    
    This class provides utilities for managing output files, saving results,
    and combining datasets in various formats.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the OutputManager.
        
        Args:
            config: Configuration dictionary containing output parameters
        """
        self.config = config
        self.base_output_dir = config.get('output_dir', './output')
        self.create_timestamped_dirs = config.get('create_timestamped_dirs', True)
        self.supported_formats = ['csv', 'npy', 'pkl', 'json']
        
        # Create base output directory
        self._ensure_directory_exists(self.base_output_dir)
        
    def create_output_directory(self, subdir: Optional[str] = None) -> str:
        """
        Create and return output directory path.
        
        Args:
            subdir: Optional subdirectory name
            
        Returns:
            Full path to the created output directory
        """
        if self.create_timestamped_dirs:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if subdir:
                output_dir = os.path.join(self.base_output_dir, f"{subdir}_{timestamp}")
            else:
                output_dir = os.path.join(self.base_output_dir, timestamp)
        else:
            output_dir = os.path.join(self.base_output_dir, subdir) if subdir else self.base_output_dir
        
        self._ensure_directory_exists(output_dir)
        logger.info(f"Created output directory: {output_dir}")
        return output_dir
    
    def save_data(self, data: Union[np.ndarray, pd.DataFrame, Dict], 
                  filename: str, output_dir: Optional[str] = None,
                  format: str = 'csv') -> str:
        """
        Save data to file in specified format.
        
        Args:
            data: Data to save
            filename: Base filename (without extension)
            output_dir: Output directory (uses default if None)
            format: Output format ('csv', 'npy', 'pkl', 'json')
            
        Returns:
            Full path to saved file
        """
        if format not in self.supported_formats:
            raise ValueError(f"Unsupported format: {format}. Supported: {self.supported_formats}")
        
        if output_dir is None:
            output_dir = self.base_output_dir
        
        self._ensure_directory_exists(output_dir)
        
        # Add appropriate extension
        if not filename.endswith(f'.{format}'):
            filename = f"{filename}.{format}"
        
        filepath = os.path.join(output_dir, filename)
        
        try:
            if format == 'csv':
                self._save_csv(data, filepath)
            elif format == 'npy':
                self._save_npy(data, filepath)
            elif format == 'pkl':
                self._save_pickle(data, filepath)
            elif format == 'json':
                self._save_json(data, filepath)
            
            logger.info(f"Data saved to: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to save data to {filepath}: {str(e)}")
            raise
    
    def save_model_results(self, results: Dict[str, Any], 
                          output_dir: str, prefix: str = '') -> Dict[str, str]:
        """
        Save comprehensive model results.
        
        Args:
            results: Dictionary containing various result data
            output_dir: Output directory
            prefix: Optional prefix for filenames
            
        Returns:
            Dictionary mapping result types to saved file paths
        """
        saved_files = {}
        
        try:
            for key, value in results.items():
                if value is None:
                    continue
                    
                filename = f"{prefix}{key}" if prefix else key
                
                # Determine appropriate format based on data type
                if isinstance(value, (np.ndarray, pd.DataFrame)):
                    if isinstance(value, pd.DataFrame) or (
                        isinstance(value, np.ndarray) and len(value.shape) == 2
                    ):
                        filepath = self.save_data(value, filename, output_dir, 'csv')
                    else:
                        filepath = self.save_data(value, filename, output_dir, 'npy')
                elif isinstance(value, dict):
                    filepath = self.save_data(value, filename, output_dir, 'json')
                else:
                    # Save as pickle for other types
                    filepath = self.save_data(value, filename, output_dir, 'pkl')
                
                saved_files[key] = filepath
                
        except Exception as e:
            logger.error(f"Failed to save model results: {str(e)}")
            raise
        
        return saved_files
    
    def combine_datasets(self, datasets: List[Union[np.ndarray, pd.DataFrame]], 
                        axis: int = 0) -> Union[np.ndarray, pd.DataFrame]:
        """
        Combine multiple datasets along specified axis.
        
        Args:
            datasets: List of datasets to combine
            axis: Axis along which to combine (0 for rows, 1 for columns)
            
        Returns:
            Combined dataset
        """
        if not datasets:
            raise ValueError("No datasets provided for combination")
        
        logger.info(f"Combining {len(datasets)} datasets along axis {axis}")
        
        # Check if all datasets are of the same type
        first_type = type(datasets[0])
        if not all(isinstance(ds, first_type) for ds in datasets):
            # Convert all to numpy arrays if types are mixed
            datasets = [self._to_numpy(ds) for ds in datasets]
            return np.concatenate(datasets, axis=axis)
        
        if isinstance(datasets[0], pd.DataFrame):
            if axis == 0:
                return pd.concat(datasets, axis=0, ignore_index=True)
            else:
                return pd.concat(datasets, axis=1)
        else:
            return np.concatenate(datasets, axis=axis)
    
    def save_training_history(self, history: Dict[str, List], 
                            output_dir: str) -> str:
        """
        Save training history (losses, metrics over epochs).
        
        Args:
            history: Dictionary containing training history
            output_dir: Output directory
            
        Returns:
            Path to saved history file
        """
        history_df = pd.DataFrame(history)
        filepath = self.save_data(history_df, 'training_history', output_dir, 'csv')
        
        # Also save as JSON for easy reading
        json_path = self.save_data(history, 'training_history', output_dir, 'json')
        
        logger.info(f"Training history saved to {filepath} and {json_path}")
        return filepath
    
    def save_configuration(self, config: Dict[str, Any], 
                          output_dir: str) -> str:
        """
        Save configuration used for the run.
        
        Args:
            config: Configuration dictionary
            output_dir: Output directory
            
        Returns:
            Path to saved configuration file
        """
        # Create a copy and remove non-serializable items
        config_copy = self._make_serializable(config)
        filepath = self.save_data(config_copy, 'config', output_dir, 'json')
        logger.info(f"Configuration saved to {filepath}")
        return filepath
    
    def create_summary_report(self, results: Dict[str, Any], 
                            output_dir: str) -> str:
        """
        Create a human-readable summary report.
        
        Args:
            results: Results dictionary
            output_dir: Output directory
            
        Returns:
            Path to summary report file
        """
        report_path = os.path.join(output_dir, 'summary_report.txt')
        
        try:
            with open(report_path, 'w') as f:
                f.write("Timeseries-GAN Run Summary\n")
                f.write("=" * 40 + "\n\n")
                f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Write key results
                for key, value in results.items():
                    if isinstance(value, dict):
                        f.write(f"{key.upper()}:\n")
                        for subkey, subvalue in value.items():
                            f.write(f"  {subkey}: {subvalue}\n")
                        f.write("\n")
                    elif isinstance(value, (int, float, str)):
                        f.write(f"{key}: {value}\n")
                    elif hasattr(value, 'shape'):
                        f.write(f"{key} shape: {value.shape}\n")
                
            logger.info(f"Summary report saved to {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"Failed to create summary report: {str(e)}")
            raise
    
    def _ensure_directory_exists(self, directory: str) -> None:
        """Ensure directory exists, create if it doesn't."""
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _save_csv(self, data: Union[np.ndarray, pd.DataFrame], filepath: str) -> None:
        """Save data as CSV."""
        if isinstance(data, np.ndarray):
            pd.DataFrame(data).to_csv(filepath, index=False)
        elif isinstance(data, pd.DataFrame):
            data.to_csv(filepath, index=False)
        else:
            raise ValueError(f"Cannot save {type(data)} as CSV")
    
    def _save_npy(self, data: Union[np.ndarray, pd.DataFrame], filepath: str) -> None:
        """Save data as numpy array."""
        if isinstance(data, pd.DataFrame):
            data = data.values
        np.save(filepath, data)
    
    def _save_pickle(self, data: Any, filepath: str) -> None:
        """Save data as pickle."""
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def _save_json(self, data: Union[Dict, List], filepath: str) -> None:
        """Save data as JSON."""
        serializable_data = self._make_serializable(data)
        with open(filepath, 'w') as f:
            json.dump(serializable_data, f, indent=2)
    
    def _to_numpy(self, data: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Convert data to numpy array."""
        if isinstance(data, pd.DataFrame):
            return data.values
        return data
    
    def _make_serializable(self, obj: Any) -> Any:
        """Make object JSON serializable."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif hasattr(obj, '__dict__'):
            # For objects with __dict__, try to serialize their attributes
            try:
                return {k: self._make_serializable(v) for k, v in obj.__dict__.items()
                       if not k.startswith('_')}
            except:
                return str(obj)
        else:
            return obj
