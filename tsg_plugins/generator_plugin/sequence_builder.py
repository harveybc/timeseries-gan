#!/usr/bin/env python3
"""
Sequence Builder Module

Handles sequence building, window management, and derived feature calculations
during the generation process.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from tqdm.auto import tqdm
import logging


class SequenceBuilder:
    """Builds synthetic sequences with proper feature derivation and window management."""
    
    def __init__(self, params: Dict[str, Any], feature_to_idx: Dict[str, int],
                 num_all_features: int, normalization_handler, ti_calculator, logger: logging.Logger): # Add logger argument
        """
        Initialize sequence builder.
        
        Args:
            params: Plugin parameters
            feature_to_idx: Mapping from feature names to indices
            num_all_features: Total number of features
            normalization_handler: Instance for normalization/denormalization
            ti_calculator: Technical indicator calculator
            logger: Logger instance
        """
        self.params = params
        self.feature_to_idx = feature_to_idx
        self.num_all_features = num_all_features
        self.normalization_handler = normalization_handler
        self.ti_calculator = ti_calculator
        self.logger = logger # Store logger
        self.previous_normalized_close = None
        self.logger.debug("SequenceBuilder initialized.") # Use logger
    
    def build_sequence(self, model, feeder_outputs_sequence: List[Dict[str, np.ndarray]],
                      sequence_length_T: int, current_input_feature_window: np.ndarray,
                      ohlc_history_for_ti_list: List[Dict[str, float]]) -> np.ndarray:
        """
        Build the synthetic sequence using the model and feeder outputs.
        
        Args:
            model: The loaded Keras model for generation
            feeder_outputs_sequence: List of feeder outputs for each time step
            sequence_length_T: Length of sequence to generate
            current_input_feature_window: Initial feature window
            ohlc_history_for_ti_list: OHLC history for TI calculations
            
        Returns:
            Generated sequence array with shape (1, sequence_length, num_features)
        """
        generated_sequence_all_features_list = []
        ohlc_names = self.params["ohlc_feature_names"]
        min_ohlc_hist_len = self.params["ti_calculation_min_lookback"]
        
        # Initialize previous normalized close from window
        self._initialize_previous_close(current_input_feature_window)
        
        # Main generation loop
        for t in tqdm(range(sequence_length_T), desc="Generating synthetic sequence", 
                     unit="step", dynamic_ncols=True):
            
            current_tick_features = self._generate_single_tick(
                model, feeder_outputs_sequence[t], t, ohlc_history_for_ti_list
            )
            
            generated_sequence_all_features_list.append(current_tick_features)
            
            # Update rolling window
            current_input_feature_window = self._update_rolling_window(
                current_input_feature_window, current_tick_features
            )
            
            # Manage OHLC history
            self._manage_ohlc_history(
                ohlc_history_for_ti_list, current_tick_features, min_ohlc_hist_len
            )
        
        # Convert to final array format
        final_sequence = np.array(generated_sequence_all_features_list, dtype=np.float32)
        return self._finalize_sequence(final_sequence)
    
    def _initialize_previous_close(self, current_input_feature_window: np.ndarray) -> None:
        """Initialize previous normalized close from the input window."""
        if ('CLOSE' in self.feature_to_idx and 
            current_input_feature_window.shape[0] > 0):
            
            last_norm_close = current_input_feature_window[-1, self.feature_to_idx['CLOSE']]
            if pd.notnull(last_norm_close):
                self.previous_normalized_close = float(last_norm_close)
                print(f"SequenceBuilder: Initialized previous_normalized_close: "
                      f"{self.previous_normalized_close}")
    
    def _generate_single_tick(self, model, feeder_step_output: Dict[str, np.ndarray],
                             t: int, ohlc_history_for_ti_list: List[Dict[str, float]]) -> np.ndarray:
        """
        Generate features for a single time step.
        
        Args:
            model: Keras model for generation
            feeder_step_output: Output from feeder for this step
            t: Current time step
            ohlc_history_for_ti_list: OHLC history for TI calculations
            
        Returns:
            Feature array for this time step
        """
        current_tick_features = np.full(self.num_all_features, np.nan, dtype=np.float32)
        
        # 1. Get model prediction
        model_inputs = self._prepare_model_inputs(feeder_step_output)
        decoder_output = model.predict(model_inputs, verbose=0)
        decoded_features = self._extract_decoded_features(decoder_output)
        
        # 2. Fill features from decoder output
        self._fill_decoder_features(current_tick_features, decoded_features)
        
        # 3. Calculate CLOSE using OPEN + BC-BO if available
        norm_close = self._calculate_close_value(current_tick_features, t)
        
        # 4. Fill conditional features (date features, fundamentals)
        self._fill_conditional_features(current_tick_features, feeder_step_output)
        
        # 5. Fill raw date features
        self._fill_raw_date_features(current_tick_features, feeder_step_output)
        
        # 6. Calculate and fill technical indicators
        self._fill_technical_indicators(current_tick_features, ohlc_history_for_ti_list, norm_close)
        
        # 7. Calculate derived OHLC features
        self._fill_derived_ohlc_features(current_tick_features)
        
        # 8. Calculate log return
        self._fill_log_return(current_tick_features, norm_close)
        
        # 9. Fill any remaining NaN features
        self._fill_remaining_features(current_tick_features, t)
        
        return current_tick_features
    
    def _prepare_model_inputs(self, feeder_step_output: Dict[str, np.ndarray]) -> List[np.ndarray]:
        """Prepare inputs for the Keras model."""
        zt_original = feeder_step_output["Z"]
        
        # Ensure proper dimensionality
        if zt_original.ndim == 2:
            zt = np.expand_dims(zt_original, axis=0)
        elif zt_original.ndim == 3 and zt_original.shape[0] == 1:
            zt = zt_original
        else:
            raise ValueError(f"Unexpected shape for Z from Feeder: {zt_original.shape}")
        
        conditional_data_t = feeder_step_output["conditional_data"]
        if conditional_data_t.ndim == 1:
            conditional_data_t = np.expand_dims(conditional_data_t, axis=0)
        
        context_h_t = feeder_step_output.get("context_h", np.zeros((1, 1)))
        if context_h_t.ndim == 1:
            context_h_t = np.expand_dims(context_h_t, axis=0)
        
        # Prepare model inputs in correct order
        return self._assemble_model_inputs(zt, context_h_t, conditional_data_t)
    
    def _assemble_model_inputs(self, zt: np.ndarray, context_h_t: np.ndarray,
                              conditional_data_t: np.ndarray) -> List[np.ndarray]:
        """Assemble model inputs in the correct order expected by the model."""
        # Get the actual model input names
        model_input_names = [inp.name.split(':')[0] for inp in self.model.inputs] if hasattr(self, 'model') else []
        
        # Map parameter names to data
        available_data = {
            self.params["decoder_input_name_latent"]: zt,
            self.params["decoder_input_name_context"]: context_h_t,
            self.params["decoder_input_name_conditions"]: conditional_data_t,
        }
        
        to_feed = []
        window_input_name = self.params.get("decoder_input_name_window")
        
        # Assemble inputs in model's expected order
        for model_input_name in model_input_names:
            if model_input_name == window_input_name:
                continue  # Skip window input
            
            if model_input_name in available_data:
                to_feed.append(available_data[model_input_name])
            else:
                raise ValueError(f"Model expects input '{model_input_name}' but it's not available")
        
        return to_feed
    
    def _extract_decoded_features(self, decoder_output: np.ndarray) -> np.ndarray:
        """Extract features from decoder output."""
        if decoder_output.ndim == 3 and decoder_output.shape[1] == 1:
            return decoder_output[0, 0, :]
        elif decoder_output.ndim == 2 and decoder_output.shape[0] == 1:
            return decoder_output[0, :]
        else:
            raise ValueError(f"Unexpected decoder output shape: {decoder_output.shape}")
    
    def _fill_decoder_features(self, current_tick_features: np.ndarray,
                              decoded_features: np.ndarray) -> None:
        """Fill features from decoder output."""
        for i, name in enumerate(self.params["decoder_output_feature_names"]):
            if name in self.feature_to_idx:
                current_tick_features[self.feature_to_idx[name]] = decoded_features[i]
    
    def _calculate_close_value(self, current_tick_features: np.ndarray, t: int) -> float:
        """Calculate the CLOSE value for this tick."""
        ohlc_names = self.params["ohlc_feature_names"]
        norm_open = current_tick_features[self.feature_to_idx[ohlc_names[0]]] if ohlc_names[0] in self.feature_to_idx else np.nan
        
        # Try to calculate CLOSE from OPEN + BC-BO
        if ("OPEN" in self.feature_to_idx and "BC-BO" in self.feature_to_idx and 
            "CLOSE" in self.feature_to_idx):
            
            norm_bc_bo = current_tick_features[self.feature_to_idx["BC-BO"]]
            
            if pd.notnull(norm_open) and pd.notnull(norm_bc_bo):
                denorm_open = self.normalization_handler.denormalize_value(norm_open, "OPEN")
                denorm_bc_bo = self.normalization_handler.denormalize_value(norm_bc_bo, "BC-BO")
                
                if pd.notnull(denorm_open) and pd.notnull(denorm_bc_bo):
                    denormalized_close = denorm_open + denorm_bc_bo
                    norm_close = self.normalization_handler.normalize_value(denormalized_close, "CLOSE")
                    current_tick_features[self.feature_to_idx["CLOSE"]] = norm_close
                    return norm_close
        
        # Fallback strategies for CLOSE
        if "CLOSE" in self.feature_to_idx:
            if t == 0:  # First synthetic tick
                if pd.notnull(norm_open):
                    norm_close = norm_open
                elif self.previous_normalized_close is not None:
                    norm_close = self.previous_normalized_close
                else:
                    norm_close = 0.01  # Last resort
            else:  # t > 0
                if pd.notnull(norm_open):
                    norm_close = norm_open
                else:
                    norm_close = self.previous_normalized_close or 0.01
            
            current_tick_features[self.feature_to_idx["CLOSE"]] = norm_close
            return norm_close
        
        return np.nan
    
    def _fill_conditional_features(self, current_tick_features: np.ndarray,
                                  feeder_step_output: Dict[str, np.ndarray]) -> None:
        """Fill conditional features from feeder output."""
        conditional_data_t = feeder_step_output["conditional_data"]
        if conditional_data_t.ndim == 1:
            conditional_data_t = np.expand_dims(conditional_data_t, axis=0)
        
        cond_input_idx = 0
        
        # Date conditional features (sin/cos)
        for original_date_feat_name in self.params["date_conditional_feature_names"]:
            for suffix in ["_sin", "_cos"]:
                feat_name = f"{original_date_feat_name}{suffix}"
                if feat_name in self.feature_to_idx:
                    current_tick_features[self.feature_to_idx[feat_name]] = conditional_data_t[0, cond_input_idx]
                cond_input_idx += 1
        
        # Feeder conditional features
        for name in self.params["feeder_conditional_feature_names"]:
            if name in self.feature_to_idx:
                current_tick_features[self.feature_to_idx[name]] = conditional_data_t[0, cond_input_idx]
            cond_input_idx += 1
    
    def _fill_raw_date_features(self, current_tick_features: np.ndarray,
                               feeder_step_output: Dict[str, np.ndarray]) -> None:
        """Fill raw date features."""
        dt_obj = feeder_step_output["datetimes"]
        
        raw_date_map = {
            "day_of_month": dt_obj.day,
            "hour_of_day": dt_obj.hour,
            "day_of_week": dt_obj.dayofweek
        }
        
        for raw_feat_name, raw_val in raw_date_map.items():
            if raw_feat_name in self.feature_to_idx:
                normalized_val = self.normalization_handler.normalize_value(float(raw_val), raw_feat_name)
                current_tick_features[self.feature_to_idx[raw_feat_name]] = normalized_val
    
    def _fill_technical_indicators(self, current_tick_features: np.ndarray,
                                  ohlc_history_for_ti_list: List[Dict[str, float]],
                                  norm_close: float) -> None:
        """Calculate and fill technical indicators."""
        ohlc_names = self.params["ohlc_feature_names"]
        
        # Get current OHLC values
        current_ohlc_norm = {
            ohlc_names[0]: current_tick_features[self.feature_to_idx[ohlc_names[0]]],  # OPEN
            ohlc_names[1]: current_tick_features[self.feature_to_idx[ohlc_names[1]]],  # HIGH
            ohlc_names[2]: current_tick_features[self.feature_to_idx[ohlc_names[2]]],  # LOW
            ohlc_names[3]: norm_close  # CLOSE
        }
        
        # Denormalize for TI calculation
        current_ohlc_denorm = {
            name: self.normalization_handler.denormalize_value(
                current_ohlc_norm.get(name, np.nan), name
            )
            for name in ohlc_names
        }
        
        # Add to history if all values are valid
        if all(pd.notnull(v) for v in current_ohlc_denorm.values()):
            ohlc_history_for_ti_list.append(current_ohlc_denorm)
            
            if len(ohlc_history_for_ti_list) >= 1:
                ohlc_df = pd.DataFrame(ohlc_history_for_ti_list)
                calculated_tis = self.ti_calculator.calculate_technical_indicators(
                    ohlc_df, ohlc_names
                ).iloc[0]
                
                # Fill TI features
                for ti_name in self.params["ti_feature_names"]:
                    if ti_name in self.feature_to_idx:
                        denorm_ti_val = calculated_tis.get(ti_name, np.nan)
                        if pd.notnull(denorm_ti_val):
                            norm_ti_val = self.normalization_handler.normalize_value(denorm_ti_val, ti_name)
                            current_tick_features[self.feature_to_idx[ti_name]] = norm_ti_val
    
    def _fill_derived_ohlc_features(self, current_tick_features: np.ndarray) -> None:
        """Fill derived OHLC features that are not direct decoder outputs."""
        decoder_outputs_set = set(self.params.get("decoder_output_feature_names", []))
        
        # Get denormalized OHLC values
        ohlc_indices = [self.feature_to_idx[name] for name in self.params["ohlc_feature_names"] 
                       if name in self.feature_to_idx]
        
        if len(ohlc_indices) == 4:
            denorm_ohlc = [
                self.normalization_handler.denormalize_value(
                    current_tick_features[idx], self.params["ohlc_feature_names"][i]
                )
                for i, idx in enumerate(ohlc_indices)
            ]
            
            dn_o, dn_h, dn_l, dn_c = denorm_ohlc
            
            # Calculate derived features
            derived_ohlc_map = {
                "BC-BO": dn_c - dn_o if pd.notnull(dn_c) and pd.notnull(dn_o) else np.nan,
                "BH-BL": dn_h - dn_l if pd.notnull(dn_h) and pd.notnull(dn_l) else np.nan,
                "BH-BO": dn_h - dn_o if pd.notnull(dn_h) and pd.notnull(dn_o) else np.nan,
                "BO-BL": dn_o - dn_l if pd.notnull(dn_o) and pd.notnull(dn_l) else np.nan,
            }
            
            for feat_name, denorm_val in derived_ohlc_map.items():
                if (feat_name in self.feature_to_idx and 
                    feat_name not in decoder_outputs_set and 
                    pd.notnull(denorm_val)):
                    norm_val = self.normalization_handler.normalize_value(denorm_val, feat_name)
                    current_tick_features[self.feature_to_idx[feat_name]] = norm_val
    
    def _fill_log_return(self, current_tick_features: np.ndarray, norm_close: float) -> None:
        """Calculate and fill log return."""
        if "log_return" in self.feature_to_idx:
            log_return_val = 0.0
            
            if (self.previous_normalized_close is not None and
                self.previous_normalized_close > 1e-9 and
                pd.notnull(norm_close) and
                norm_close > 1e-9):
                
                ratio = norm_close / self.previous_normalized_close
                if ratio > 1e-9:
                    log_return_val = np.log(ratio)
            
            norm_log_return = self.normalization_handler.normalize_value(log_return_val, "log_return")
            current_tick_features[self.feature_to_idx["log_return"]] = norm_log_return
        
        # Update previous close
        if pd.notnull(norm_close):
            self.previous_normalized_close = norm_close
    
    def _fill_remaining_features(self, current_tick_features: np.ndarray, t: int) -> None:
        """Fill any remaining NaN features with fallback values."""
        for i, feat_name in enumerate(self.params["full_feature_names_ordered"]):
            if np.isnan(current_tick_features[i]):
                # Use random small value as fallback
                current_tick_features[i] = np.random.uniform(0.01, 0.1)
                
        # Fill DATE_TIME with time index
        if 'DATE_TIME' in self.feature_to_idx:
            current_tick_features[self.feature_to_idx['DATE_TIME']] = np.float32(t)
    
    def _update_rolling_window(self, window: np.ndarray, new_features: np.ndarray) -> np.ndarray:
        """Update the rolling window with new features."""
        # Replace NaNs before adding to window (for model stability)
        clean_features = np.nan_to_num(new_features.copy(), nan=0.01)
        window = np.roll(window, -1, axis=0)
        window[-1, :] = clean_features
        return window
    
    def _manage_ohlc_history(self, ohlc_history_list: List[Dict[str, float]],
                           current_features: np.ndarray, min_hist_len: int) -> None:
        """Manage OHLC history list size."""
        if len(ohlc_history_list) > min_hist_len + 50:
            ohlc_history_list.pop(0)
    
    def _finalize_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """Finalize the generated sequence."""
        if np.isnan(sequence).any():
            nan_counts = np.sum(np.isnan(sequence), axis=0)
            nan_features = [
                self.params["full_feature_names_ordered"][i] 
                for i, count in enumerate(nan_counts) if count > 0
            ]
            print(f"SequenceBuilder: Warning - NaNs found in final sequence. "
                  f"Features with NaNs: {nan_features}. Replacing with 0.01.")
            sequence = np.nan_to_num(sequence, nan=0.01)
        
        return np.expand_dims(sequence, axis=0)
