#!/usr/bin/env python3
"""
Technical Indicator Calculator Module

Handles all technical indicator calculations using pandas-ta library.
Manages minimum lookback requirements and error handling for insufficient data.
"""

import numpy as np
import pandas as pd
from .pandas_ta_compat import ta, pandas_ta_available
from typing import Dict, Any, List, Optional


class TechnicalIndicatorCalculator:
    """Calculates technical indicators from OHLC data using pandas-ta."""
    
    def __init__(self, ti_feature_names: List[str], ti_params: Dict[str, Any]):
        """
        Initialize the calculator with TI configuration.
        
        Args:
            ti_feature_names: List of TI names to calculate
            ti_params: Dictionary of TI parameters (lengths, etc.)
        """
        self.ti_feature_names = ti_feature_names
        self.ti_params = ti_params
    
    def calculate_technical_indicators(self, ohlc_history_df: pd.DataFrame, 
                                     ohlc_feature_names: List[str],
                                     return_last_row_only: bool = True) -> pd.DataFrame: # Added return_last_row_only
        """
        Calculate technical indicators from OHLC history DataFrame.
        
        Args:
            ohlc_history_df: DataFrame with OHLC columns
            ohlc_feature_names: List of OHLC feature names [OPEN, HIGH, LOW, CLOSE]
            return_last_row_only: If True, returns only the TIs for the last row. 
                                  If False, returns TIs for all rows.
            
        Returns:
            DataFrame with TI values.
        """
        if ohlc_history_df.empty:
            # If returning full history and input is empty, placeholder should reflect that
            # However, current _create_nan_placeholder_df is single row.
            # For now, this path will yield a single row of NaNs.
            # If ohlc_history_df is empty, the calling merge will likely result in empty or all NaN TI columns.
            return self._create_nan_placeholder_df()

        # Map feature names to standard pandas-ta names
        ohlc_map = {
            ohlc_feature_names[0]: 'open',
            ohlc_feature_names[1]: 'high', 
            ohlc_feature_names[2]: 'low',
            ohlc_feature_names[3]: 'close'
        }
        
        # Check for missing OHLC columns
        missing_ohlc_cols = [col for col in ohlc_feature_names if col not in ohlc_history_df.columns]
        if missing_ohlc_cols:
            print(f"TechnicalIndicatorCalculator: Warning - Missing OHLC columns: {missing_ohlc_cols}")
            return self._create_nan_placeholder_df()
            
        # Rename columns for pandas-ta
        df = ohlc_history_df.rename(columns=ohlc_map)
        ti_df_results = pd.DataFrame(index=df.index)
        
        # Calculate each indicator
        self._calculate_rsi(df, ti_df_results)
        self._calculate_macd(df, ti_df_results)
        self._calculate_stochastic(df, ti_df_results)
        self._calculate_adx(df, ti_df_results)
        self._calculate_atr(df, ti_df_results)
        self._calculate_cci(df, ti_df_results)
        self._calculate_willr(df, ti_df_results)
        self._calculate_momentum(df, ti_df_results)
        self._calculate_roc(df, ti_df_results)
        self._calculate_ema(df, ti_df_results)
        
        # Ensure all requested TI columns exist
        for ti_name in self.ti_feature_names:
            if ti_name not in ti_df_results.columns:
                ti_df_results[ti_name] = np.nan
        
        # Return results based on the flag
        return self._extract_last_row_results(ti_df_results, return_last_row_only=return_last_row_only)
    
    def _create_nan_placeholder_df(self) -> pd.DataFrame:
        """Create DataFrame with NaN values for all TI features."""
        nan_placeholder_df = pd.DataFrame(columns=self.ti_feature_names, index=[0]).astype(np.float32)
        return nan_placeholder_df.fillna(np.nan)
    
    def _calculate_rsi(self, df: pd.DataFrame, ti_df_results: pd.DataFrame) -> None:
        """Calculate RSI indicator."""
        rsi_len = self.ti_params.get("rsi_length", 14)
        if 'RSI' in self.ti_feature_names:
            if len(df['close']) >= rsi_len:
                rsi_series = ta.rsi(df['close'], length=rsi_len)
                ti_df_results['RSI'] = rsi_series if isinstance(rsi_series, pd.Series) else np.nan
            else:
                ti_df_results['RSI'] = np.nan
    
    def _calculate_macd(self, df: pd.DataFrame, ti_df_results: pd.DataFrame) -> None:
        """Calculate MACD indicators."""
        macd_fast = self.ti_params.get("macd_fast", 12)
        macd_slow = self.ti_params.get("macd_slow", 26)
        macd_signal = self.ti_params.get("macd_signal", 9)
        min_len_macd = macd_slow + macd_signal - 1
        
        macd_names = ['MACD', 'MACD_Histogram', 'MACD_Signal']
        if any(x in self.ti_feature_names for x in macd_names):
            if len(df['close']) >= min_len_macd:
                macd_output_df = ta.macd(df['close'], fast=macd_fast, slow=macd_slow, signal=macd_signal)
                if isinstance(macd_output_df, pd.DataFrame) and not macd_output_df.empty:
                    macd_col_name = f"MACD_{macd_fast}_{macd_slow}_{macd_signal}"
                    hist_col_name = f"MACDh_{macd_fast}_{macd_slow}_{macd_signal}"
                    signal_col_name = f"MACDs_{macd_fast}_{macd_slow}_{macd_signal}"

                    if 'MACD' in self.ti_feature_names:
                        ti_df_results['MACD'] = macd_output_df.get(macd_col_name, np.nan)
                    if 'MACD_Histogram' in self.ti_feature_names:
                        ti_df_results['MACD_Histogram'] = macd_output_df.get(hist_col_name, np.nan)
                    if 'MACD_Signal' in self.ti_feature_names:
                        ti_df_results['MACD_Signal'] = macd_output_df.get(signal_col_name, np.nan)
                else:
                    self._set_macd_nan(ti_df_results)
            else:
                self._set_macd_nan(ti_df_results)
    
    def _set_macd_nan(self, ti_df_results: pd.DataFrame) -> None:
        """Set MACD indicators to NaN."""
        for macd_name in ['MACD', 'MACD_Histogram', 'MACD_Signal']:
            if macd_name in self.ti_feature_names:
                ti_df_results[macd_name] = np.nan
    
    def _calculate_stochastic(self, df: pd.DataFrame, ti_df_results: pd.DataFrame) -> None:
        """Calculate Stochastic indicators."""
        stoch_k_period = self.ti_params.get("stoch_k", 14)
        stoch_d_period = self.ti_params.get("stoch_d", 3)
        stoch_smooth_k_period = self.ti_params.get("stoch_smooth_k", 3)

        min_len_for_k_smooth = stoch_k_period + stoch_smooth_k_period - 1
        min_len_for_d_final = min_len_for_k_smooth + stoch_d_period - 1

        stoch_k_wanted = 'Stochastic_%K' in self.ti_feature_names
        stoch_d_wanted = 'Stochastic_%D' in self.ti_feature_names

        # Initialize to NaN
        if stoch_k_wanted: 
            ti_df_results['Stochastic_%K'] = np.nan
        if stoch_d_wanted: 
            ti_df_results['Stochastic_%D'] = np.nan

        if stoch_d_period > 0:
            if len(df['high']) >= min_len_for_d_final and (stoch_k_wanted or stoch_d_wanted):
                try:
                    stoch_output_df = ta.stoch(df['high'], df['low'], df['close'], 
                                             k=stoch_k_period, 
                                             d=stoch_d_period,
                                             smooth_k=stoch_smooth_k_period)
                    
                    if isinstance(stoch_output_df, pd.DataFrame) and not stoch_output_df.empty:
                        k_col_name = f"STOCHk_{stoch_k_period}_{stoch_d_period}_{stoch_smooth_k_period}"
                        d_col_name = f"STOCHd_{stoch_k_period}_{stoch_d_period}_{stoch_smooth_k_period}"

                        if stoch_k_wanted and k_col_name in stoch_output_df.columns:
                            ti_df_results['Stochastic_%K'] = stoch_output_df[k_col_name]
                        
                        if stoch_d_wanted and d_col_name in stoch_output_df.columns:
                            ti_df_results['Stochastic_%D'] = stoch_output_df[d_col_name]
                except Exception as e:
                    print(f"TechnicalIndicatorCalculator: Error calculating Stochastic: {e}")
    
    def _calculate_adx(self, df: pd.DataFrame, ti_df_results: pd.DataFrame) -> None:
        """Calculate ADX and DI indicators."""
        adx_len_param = self.ti_params.get("adx_length", 14)
        min_len_dmi_lines = adx_len_param
        min_len_adx_value = adx_len_param * 2 - 1
        
        adx_names = ['ADX', 'DI+', 'DI-']
        if any(x in self.ti_feature_names for x in adx_names):
            if len(df['high']) >= min_len_dmi_lines:
                adx_output_df = ta.adx(df['high'], df['low'], df['close'], length=adx_len_param)
                if isinstance(adx_output_df, pd.DataFrame) and not adx_output_df.empty:
                    adx_col_name = f"ADX_{adx_len_param}"
                    dip_col_name = f"DMP_{adx_len_param}"
                    dim_col_name = f"DMN_{adx_len_param}"

                    if 'DI+' in self.ti_feature_names:
                        ti_df_results['DI+'] = adx_output_df.get(dip_col_name, np.nan)
                    if 'DI-' in self.ti_feature_names:
                        ti_df_results['DI-'] = adx_output_df.get(dim_col_name, np.nan)
                    
                    if 'ADX' in self.ti_feature_names:
                        if len(df['high']) >= min_len_adx_value:
                            ti_df_results['ADX'] = adx_output_df.get(adx_col_name, np.nan)
                        else:
                            ti_df_results['ADX'] = np.nan
                else:
                    self._set_adx_nan(ti_df_results)
            else:
                self._set_adx_nan(ti_df_results)
    
    def _set_adx_nan(self, ti_df_results: pd.DataFrame) -> None:
        """Set ADX indicators to NaN."""
        for adx_name in ['ADX', 'DI+', 'DI-']:
            if adx_name in self.ti_feature_names:
                ti_df_results[adx_name] = np.nan
    
    def _calculate_atr(self, df: pd.DataFrame, ti_df_results: pd.DataFrame) -> None:
        """Calculate ATR indicator."""
        atr_len = self.ti_params.get("atr_length", 14)
        if 'ATR' in self.ti_feature_names:
            if len(df['high']) >= atr_len:
                atr_series = ta.atr(df['high'], df['low'], df['close'], length=atr_len)
                ti_df_results['ATR'] = atr_series if isinstance(atr_series, pd.Series) else np.nan
            else:
                ti_df_results['ATR'] = np.nan
    
    def _calculate_cci(self, df: pd.DataFrame, ti_df_results: pd.DataFrame) -> None:
        """Calculate CCI indicator."""
        cci_len = self.ti_params.get("cci_length", 14)
        if 'CCI' in self.ti_feature_names:
            if len(df['high']) >= cci_len:
                cci_series = ta.cci(df['high'], df['low'], df['close'], length=cci_len)
                ti_df_results['CCI'] = cci_series if isinstance(cci_series, pd.Series) else np.nan
            else:
                ti_df_results['CCI'] = np.nan
    
    def _calculate_willr(self, df: pd.DataFrame, ti_df_results: pd.DataFrame) -> None:
        """Calculate Williams %R indicator."""
        willr_len = self.ti_params.get("willr_length", 14)
        if 'WilliamsR' in self.ti_feature_names:
            if len(df['high']) >= willr_len:
                willr_series = ta.willr(df['high'], df['low'], df['close'], length=willr_len)
                ti_df_results['WilliamsR'] = willr_series if isinstance(willr_series, pd.Series) else np.nan
            else:
                ti_df_results['WilliamsR'] = np.nan
    
    def _calculate_momentum(self, df: pd.DataFrame, ti_df_results: pd.DataFrame) -> None:
        """Calculate Momentum indicator."""
        mom_len = self.ti_params.get("mom_length", 14)
        if 'Momentum' in self.ti_feature_names:
            if len(df['close']) >= mom_len + 1:
                mom_series = ta.mom(df['close'], length=mom_len)
                ti_df_results['Momentum'] = mom_series if isinstance(mom_series, pd.Series) else np.nan
            else:
                ti_df_results['Momentum'] = np.nan
    
    def _calculate_roc(self, df: pd.DataFrame, ti_df_results: pd.DataFrame) -> None:
        """Calculate ROC (Rate of Change) indicator."""
        roc_len = self.ti_params.get("roc_length", 14)
        if 'ROC' in self.ti_feature_names:
            if len(df['close']) >= roc_len + 1:
                roc_series = ta.roc(df['close'], length=roc_len)
                ti_df_results['ROC'] = roc_series if isinstance(roc_series, pd.Series) else np.nan
            else:
                ti_df_results['ROC'] = np.nan
    
    def _calculate_ema(self, df: pd.DataFrame, ti_df_results: pd.DataFrame) -> None:
        """Calculate EMA indicator."""
        ema_len = self.ti_params.get("ema_length", 14)
        if 'EMA' in self.ti_feature_names:
            if len(df['close']) >= ema_len:
                ema_series = ta.ema(df['close'], length=ema_len)
                ti_df_results['EMA'] = ema_series if isinstance(ema_series, pd.Series) else np.nan
            else:
                ti_df_results['EMA'] = np.nan
    
    def _extract_last_row_results(self, ti_df_results: pd.DataFrame, return_last_row_only: bool = True) -> pd.DataFrame: # Added return_last_row_only
        """Extract and format TI results."""
        if ti_df_results.empty:
            return self._create_nan_placeholder_df()

        # Get columns that were actually calculated (or ensured to exist)
        # This list should effectively be self.ti_feature_names due to the loop in calculate_technical_indicators
        final_ti_columns_present = [col for col in self.ti_feature_names 
                                   if col in ti_df_results.columns]
        
        if not final_ti_columns_present:
            return self._create_nan_placeholder_df()

        # Select data based on the flag
        if return_last_row_only:
            # Return the last row of calculated TIs
            processed_tis = ti_df_results[final_ti_columns_present].tail(1).reset_index(drop=True)
        else:
            # Return all rows of calculated TIs, preserving index
            processed_tis = ti_df_results[final_ti_columns_present].copy()
        
        # Reindex to ensure all originally requested TIs are present and in order
        # This also handles cases where some TIs might not have been calculable and remained all NaN
        # For multi-row data, this reindex preserves the original index of processed_tis.
        processed_tis_reindexed = processed_tis.reindex(columns=self.ti_feature_names, fill_value=np.nan)
        
        return processed_tis_reindexed.astype(np.float32)
