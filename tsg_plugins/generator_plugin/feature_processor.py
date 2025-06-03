#!/usr/bin/env python3
"""
Feature Processor Module

This module handles feature processing and technical indicator calculations,
providing focused functionality for feature engineering and data transformation.
"""

import logging
import numpy as np
import pandas as pd
from .pandas_ta_compat import ta, pandas_ta_available
from typing import Dict, Any, List, Optional


class FeatureProcessor:
    """Handles feature processing and technical indicator calculations."""
    
    def __init__(self, params: Dict[str, Any], logger: logging.Logger):
        """Initialize feature processor."""
        self.params = params
        self.logger = logger
        
        # Feature configuration
        self.full_feature_names = params.get("full_feature_names_ordered", [])
        self.decoder_output_features = params.get("decoder_output_feature_names", [])
        self.ohlc_features = params.get("ohlc_feature_names", ["OPEN", "HIGH", "LOW", "CLOSE"])
        self.ti_features = params.get("ti_feature_names", [])
        self.ti_params = params.get("ti_params", {})
        self.ti_min_lookback = params.get("ti_calculation_min_lookback", 200)
        
        # Create feature mapping
        self.feature_to_idx = {name: i for i, name in enumerate(self.full_feature_names)}
        
        self.logger.info("FeatureProcessor initialized")
        self.logger.info(f"Full features: {len(self.full_feature_names)}")
        self.logger.info(f"Decoder output features: {len(self.decoder_output_features)}")
        self.logger.info(f"Technical indicators: {len(self.ti_features)}")
    
    def validate_feature_consistency(self) -> bool:
        """
        Validate that feature configurations are consistent.
        
        Returns:
            True if consistent, False otherwise
        """
        try:
            # Check that decoder output features are subset of full features
            for feature in self.decoder_output_features:
                if feature not in self.full_feature_names:
                    self.logger.error(f"Decoder output feature '{feature}' not in full feature list")
                    return False
            
            # Check that OHLC features are in full features
            for feature in self.ohlc_features:
                if feature not in self.full_feature_names:
                    self.logger.warning(f"OHLC feature '{feature}' not in full feature list")
            
            # Check that TI features are in full features
            for feature in self.ti_features:
                if feature not in self.full_feature_names:
                    self.logger.warning(f"TI feature '{feature}' not in full feature list")
            
            self.logger.info("Feature consistency validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating feature consistency: {e}")
            return False
    
    def calculate_technical_indicators(self, ohlc_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for OHLC data.
        
        Args:
            ohlc_data: DataFrame with OHLC columns
            
        Returns:
            DataFrame with calculated technical indicators
        """
        try:
            if len(ohlc_data) < self.ti_min_lookback:
                self.logger.warning(f"Insufficient data for TI calculation: {len(ohlc_data)} < {self.ti_min_lookback}")
                return self._create_empty_ti_dataframe(len(ohlc_data))
            
            # Ensure required columns exist
            required_cols = ['open', 'high', 'low', 'close']
            for col in required_cols:
                if col not in ohlc_data.columns:
                    self.logger.error(f"Required column '{col}' not found in OHLC data")
                    return self._create_empty_ti_dataframe(len(ohlc_data))
            
            ti_data = pd.DataFrame(index=ohlc_data.index)
            
            # Calculate each technical indicator
            for ti_name in self.ti_features:
                try:
                    ti_values = self._calculate_single_indicator(ohlc_data, ti_name)
                    if ti_values is not None:
                        ti_data[ti_name] = ti_values
                    else:
                        ti_data[ti_name] = 0.0  # Fill with zeros if calculation fails
                        self.logger.warning(f"TI calculation failed for {ti_name}, using zeros")
                
                except Exception as e:
                    self.logger.error(f"Error calculating TI {ti_name}: {e}")
                    ti_data[ti_name] = 0.0
            
            return ti_data
            
        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {e}")
            return self._create_empty_ti_dataframe(len(ohlc_data))
    
    def _calculate_single_indicator(self, ohlc_data: pd.DataFrame, ti_name: str) -> Optional[pd.Series]:
        """Calculate a single technical indicator."""
        try:
            # Parse TI name and parameters
            parts = ti_name.split('_')
            if len(parts) < 2:
                self.logger.warning(f"Invalid TI name format: {ti_name}")
                return None
            
            base_name = parts[0].upper()
            
            # Get parameters from TI name or config
            params = self._parse_ti_parameters(ti_name, parts[1:])
            
            # Calculate based on indicator type
            if base_name == 'EMA':
                length = params.get('length', 14)
                return ta.ema(ohlc_data['close'], length=length)
            
            elif base_name == 'SMA':
                length = params.get('length', 14)
                return ta.sma(ohlc_data['close'], length=length)
            
            elif base_name == 'RSI':
                length = params.get('length', 14)
                return ta.rsi(ohlc_data['close'], length=length)
            
            elif base_name == 'MACD':
                fast = params.get('fast', 12)
                slow = params.get('slow', 26)
                signal = params.get('signal', 9)
                macd_result = ta.macd(ohlc_data['close'], fast=fast, slow=slow, signal=signal)
                return macd_result.iloc[:, 0] if macd_result is not None else None
            
            elif base_name == 'BBANDS':
                length = params.get('length', 20)
                std = params.get('std', 2)
                bb_result = ta.bbands(ohlc_data['close'], length=length, std=std)
                return bb_result.iloc[:, 1] if bb_result is not None else None  # Middle band
            
            elif base_name == 'STOCH':
                k = params.get('k', 14)
                d = params.get('d', 3)
                stoch_result = ta.stoch(ohlc_data['high'], ohlc_data['low'], ohlc_data['close'], k=k, d=d)
                return stoch_result.iloc[:, 0] if stoch_result is not None else None  # %K
            
            elif base_name == 'ATR':
                length = params.get('length', 14)
                return ta.atr(ohlc_data['high'], ohlc_data['low'], ohlc_data['close'], length=length)
            
            else:
                self.logger.warning(f"Unsupported TI type: {base_name}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error calculating {ti_name}: {e}")
            return None
    
    def _parse_ti_parameters(self, ti_name: str, param_parts: List[str]) -> Dict[str, Any]:
        """Parse technical indicator parameters from name or config."""
        params = {}
        
        # Get from global TI params config
        if ti_name in self.ti_params:
            params.update(self.ti_params[ti_name])
        
        # Parse from name parts
        base_name = ti_name.split('_')[0].upper()
        
        if base_name in ['EMA', 'SMA', 'RSI', 'ATR'] and param_parts:
            try:
                params['length'] = int(param_parts[0])
            except (ValueError, IndexError):
                pass
        
        elif base_name == 'MACD' and len(param_parts) >= 2:
            try:
                params['fast'] = int(param_parts[0])
                params['slow'] = int(param_parts[1])
                if len(param_parts) >= 3:
                    params['signal'] = int(param_parts[2])
            except ValueError:
                pass
        
        elif base_name == 'BBANDS' and len(param_parts) >= 2:
            try:
                params['length'] = int(param_parts[0])
                params['std'] = float(param_parts[1])
            except ValueError:
                pass
        
        elif base_name == 'STOCH' and len(param_parts) >= 2:
            try:
                params['k'] = int(param_parts[0])
                params['d'] = int(param_parts[1])
            except ValueError:
                pass
        
        return params
    
    def _create_empty_ti_dataframe(self, length: int) -> pd.DataFrame:
        """Create empty DataFrame for TI data."""
        ti_data = pd.DataFrame(index=range(length))
        for ti_name in self.ti_features:
            ti_data[ti_name] = 0.0
        return ti_data
    
    def extract_decoder_features(self, full_data: np.ndarray) -> np.ndarray:
        """
        Extract decoder output features from full feature array.
        
        Args:
            full_data: Full feature array
            
        Returns:
            Array with only decoder output features
        """
        try:
            if len(self.decoder_output_features) == 0:
                return full_data
            
            # Get indices for decoder output features
            decoder_indices = []
            for feature_name in self.decoder_output_features:
                if feature_name in self.feature_to_idx:
                    decoder_indices.append(self.feature_to_idx[feature_name])
                else:
                    self.logger.warning(f"Decoder feature '{feature_name}' not found in feature mapping")
            
            if not decoder_indices:
                self.logger.error("No valid decoder output feature indices found")
                return full_data
            
            # Extract features
            decoder_data = full_data[..., decoder_indices]
            
            return decoder_data
            
        except Exception as e:
            self.logger.error(f"Error extracting decoder features: {e}")
            return full_data
    
    def get_feature_info(self) -> Dict[str, Any]:
        """Get information about features."""
        return {
            "total_features": len(self.full_feature_names),
            "decoder_features": len(self.decoder_output_features),
            "ohlc_features": len(self.ohlc_features),
            "ti_features": len(self.ti_features),
            "feature_mapping": self.feature_to_idx
        }
    
    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information."""
        return {
            "full_feature_count": len(self.full_feature_names),
            "decoder_feature_count": len(self.decoder_output_features),
            "ti_feature_count": len(self.ti_features),
            "ti_min_lookback": self.ti_min_lookback,
            "feature_consistency": self.validate_feature_consistency()
        }
