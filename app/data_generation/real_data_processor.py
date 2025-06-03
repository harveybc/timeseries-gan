#!/usr/bin/env python3
"""
real_data_processor.py

Real data processing module for TimeSeries-GAN.
Handles loading and processing of real data segments for integration
with synthetic data generation pipeline.

This module encapsulates real data processing logic following
single responsibility principle and extreme separation of concerns.

Author: TimeSeries-GAN Team
"""

import os
import pandas as pd
from typing import Dict, Any


class RealDataProcessor:
    """
    Processor for real time series data segments.
    
    This class handles loading and processing of real data segments
    that can be integrated with synthetic data for combined outputs.
    Supports various data loading configurations and preprocessing options.
    
    Attributes:
        config: Configuration dictionary containing data processing parameters
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize real data processor with configuration.
        
        Args:
            config: Configuration dictionary containing processing parameters
        """
        self.config = config
    
    def process(self, max_steps: int) -> pd.DataFrame:
        """
        Process real data segment for integration with synthetic data.
        
        Args:
            max_steps: Maximum number of data points to load from real data
            
        Returns:
            pd.DataFrame: Processed real data segment
            
        Raises:
            RuntimeError: If real data processing fails
        """
        try:
            print(f"Processing real data segment (max_steps: {max_steps})...")
            
            if max_steps <= 0:
                print("No real data requested (max_steps <= 0)")
                return pd.DataFrame()
            
            # Load real data from file
            real_data = self._load_real_data()
            
            # Extract specified number of steps
            if not real_data.empty:
                real_data_segment = self._extract_data_segment(real_data, max_steps)
                print(f"✓ Real data segment processed. Shape: {real_data_segment.shape}")
                return real_data_segment
            else:
                print("⚠ Warning: No real data available")
                return pd.DataFrame()
                
        except Exception as e:
            raise RuntimeError(f"Real data processing failed: {e}")
    
    def _load_real_data(self) -> pd.DataFrame:
        """
        Load real data from configured file path.
        
        Returns:
            pd.DataFrame: Loaded real data
            
        Raises:
            ValueError: If data file is not found or cannot be loaded
        """
        x_train_file_path = self.config.get("x_train_file")
        datetime_col_name = self.config.get("datetime_col_name", "DATE_TIME")
        
        if not x_train_file_path:
            raise ValueError("x_train_file path not configured")
        
        if not os.path.exists(x_train_file_path):
            raise ValueError(f"Real data file not found: {x_train_file_path}")
        
        try:
            print(f"Loading real data from: {x_train_file_path}")
            
            # Load data with datetime parsing if datetime column exists
            if self._has_datetime_column(x_train_file_path, datetime_col_name):
                real_data = pd.read_csv(x_train_file_path, parse_dates=[datetime_col_name])
            else:
                real_data = pd.read_csv(x_train_file_path)
                print(f"⚠ Warning: Datetime column '{datetime_col_name}' not found in real data")
            
            if real_data.empty:
                print("⚠ Warning: Real data file is empty")
            else:
                print(f"✓ Real data loaded. Shape: {real_data.shape}")
            
            return real_data
            
        except Exception as e:
            raise ValueError(f"Failed to load real data: {e}")
    
    def _has_datetime_column(self, file_path: str, datetime_col_name: str) -> bool:
        """
        Check if the CSV file has the specified datetime column.
        
        Args:
            file_path: Path to the CSV file
            datetime_col_name: Name of the datetime column to check
            
        Returns:
            bool: True if datetime column exists, False otherwise
        """
        try:
            # Read just the header to check column names
            header_df = pd.read_csv(file_path, nrows=0)
            return datetime_col_name in header_df.columns
        except Exception:
            return False
    
    def _extract_data_segment(self, real_data: pd.DataFrame, max_steps: int) -> pd.DataFrame:
        """
        Extract specified number of data points from the end of real data.
        
        Args:
            real_data: Full real data DataFrame
            max_steps: Maximum number of data points to extract
            
        Returns:
            pd.DataFrame: Extracted data segment
        """
        try:
            # Take the last max_steps rows
            if len(real_data) <= max_steps:
                data_segment = real_data.copy()
                print(f"✓ Extracted all available data ({len(real_data)} rows)")
            else:
                data_segment = real_data.tail(max_steps).copy()
                print(f"✓ Extracted last {max_steps} rows from real data")
            
            # Reset index for clean output
            data_segment = data_segment.reset_index(drop=True)
            
            return data_segment
            
        except Exception as e:
            print(f"⚠ Warning: Failed to extract data segment: {e}")
            return pd.DataFrame()
    
    def get_data_info(self) -> Dict[str, Any]:
        """
        Get information about the real data file.
        
        Returns:
            Dict containing data file information
        """
        try:
            x_train_file_path = self.config.get("x_train_file")
            
            if not x_train_file_path or not os.path.exists(x_train_file_path):
                return {"available": False, "reason": "File not found"}
            
            # Get basic file info
            file_size = os.path.getsize(x_train_file_path)
            
            # Get data shape info
            try:
                sample_data = pd.read_csv(x_train_file_path, nrows=5)
                total_rows = len(pd.read_csv(x_train_file_path))
                
                return {
                    "available": True,
                    "file_path": x_train_file_path,
                    "file_size_bytes": file_size,
                    "total_rows": total_rows,
                    "columns": list(sample_data.columns),
                    "num_columns": len(sample_data.columns)
                }
            except Exception:
                return {
                    "available": True,
                    "file_path": x_train_file_path,
                    "file_size_bytes": file_size,
                    "total_rows": "unknown",
                    "columns": "unknown",
                    "num_columns": "unknown"
                }
                
        except Exception as e:
            return {"available": False, "reason": f"Error accessing file: {e}"}
