#!/usr/bin/env python3
"""
Generator Plugin - Main Interface

Main plugin interface that orchestrates specialized modules for synthetic data generation.
Maintains mandatory plugin structure while delegating to focused modules.
"""

import sys
import traceback
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Conv1D, Dense, Reshape, Concatenate, ZeroPadding1D, TimeDistributed
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
from tensorflow import keras
keras.config.enable_unsafe_deserialization()


class GeneratorPlugin:
    """
    Main generator plugin interface following extreme separation of concerns.
    Orchestrates specialized modules for focused functionality.
    """
    
    plugin_params = {
        "sequential_model_file": None, # This will be the VAE DECODER path
        "decoder_input_window_size": 144, # Output sequence length from VAE Decoder
        "full_feature_names_ordered": [],
        "decoder_output_feature_names": [],
        "ohlc_feature_names": ["OPEN", "HIGH", "LOW", "CLOSE"],
        "ti_feature_names": [],
        "date_conditional_feature_names": [],
        "feeder_conditional_feature_names": [],
        "ti_calculation_min_lookback": 200,
        "ti_params": {},
        # VAE Decoder Input Names (matching the pre-trained model)
        "decoder_input_name_latent": "decoder_input_z_seq", 
        "decoder_input_name_window": "input_x_window", # This VAE decoder input might not be used by our new composite model directly if we generate z_seq
        "decoder_input_name_conditions": "decoder_input_conditions",
        "decoder_input_name_context": "decoder_input_h_context",
        "generator_normalization_params_file": None,
        # New params for our internal Z-generator
        "internal_z_sequence_length": 18, # As per your spec (batch_size, 18, 32)
        "internal_z_latent_dim": 32,    # As per your spec
        "feeder_noise_dim": 100, # Example: dimension of noise from FeederPlugin
        "context_vector_dim": 64, # For decoder_input_h_context
        "conditional_features_dim": 10 # For decoder_input_conditions
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
    
    def _load_model(self, vae_decoder_model_path: str) -> None:
        """
        Build the composite generator model.
        This model includes:
        1. An internal sub-model to generate decoder_input_z_seq from feeder noise.
        2. The pre-trained VAE Decoder (loaded from vae_decoder_model_path).
        """
        if not vae_decoder_model_path:
            self.sequential_model = None
            self.model = None
            self.logger.warning("GeneratorPlugin: Attempted to load VAE decoder with empty path.")
            return

        self.logger.info(f"Building composite generator with VAE decoder from: {vae_decoder_model_path}")

        try:
            # --- 1. Load the pre-trained VAE Decoder ---
            # Ensure Keras unsafe deserialization is enabled if needed for custom layers/optimizers
            # keras.config.enable_unsafe_deserialization() # Already at the top
            
            loaded_vae_decoder = self.model_loader.load_model_from_path(vae_decoder_model_path)
            if loaded_vae_decoder is None:
                raise IOError(f"Failed to load VAE decoder from {vae_decoder_model_path}")
            
            # SET THE LOADED VAE DECODER TO BE TRAINABLE
            loaded_vae_decoder.trainable = True
            self.logger.info(f"Loaded VAE decoder '{loaded_vae_decoder.name}'. Set trainable=True.")

            # --- 2. Define Inputs for the new Composite Generator ---
            # This is the noise that the FeederPlugin will provide
            feeder_noise_input = Input(
                shape=(self.params["feeder_noise_dim"],), 
                name="composite_generator_feeder_noise_input"
            )
            # This is the previous timestep's generated output (excluding TIs)
            # It will be used to populate decoder_input_h_context
            # Its shape should match the number of features in decoder_output_feature_names
            # For simplicity, let's assume its dimensionality is fixed or derived from decoder_output_feature_names
            # For now, let's use context_vector_dim directly, assuming it's prepared correctly.
            previous_step_output_input = Input(
                shape=(self.params["context_vector_dim"],), # This will feed h_context
                name="composite_generator_previous_step_output_input"
            )
            # These are the date/time conditions for the current step
            current_step_conditions_input = Input(
                shape=(self.params["conditional_features_dim"],), # This will feed decoder_input_conditions
                name="composite_generator_current_step_conditions_input"
            )

            # --- 3. Build the internal Z-sequence generator sub-model ---
            # This sub-model takes feeder_noise_input and generates decoder_input_z_seq
            
            x = Dense(self.params["internal_z_sequence_length"] * self.params["internal_z_latent_dim"], activation='relu')(feeder_noise_input) # Increased units for better representation
            x = Reshape((self.params["internal_z_sequence_length"], self.params["internal_z_latent_dim"]))(x) # Reshape to (batch, 18, 32) directly if Dense output matches
            
            # Bidirectional LSTM layer
            x = Bidirectional(LSTM(self.params["internal_z_latent_dim"] * 2, return_sequences=True))(x) # More units in LSTM
            
            # Conv1D layer to refine features per timestep and ensure correct output dimension
            internal_z_seq_output = Conv1D(
                filters=self.params["internal_z_latent_dim"], 
                kernel_size=1, # Pointwise convolution to adjust feature dimension
                padding="same", 
                activation='tanh', # tanh is common for latent vectors
                name="internal_z_seq_output"
            )(x)
            
            # --- 4. Prepare inputs for the loaded VAE Decoder ---
            vae_decoder_input_z_seq = internal_z_seq_output 
            vae_decoder_input_h_context = previous_step_output_input
            vae_decoder_input_conditions = current_step_conditions_input

            # Log the names of the input layers of the loaded VAE decoder
            vae_decoder_actual_input_names = [inp.name for inp in loaded_vae_decoder.inputs]
            self.logger.info(f"Loaded VAE Decoder actual input layer names: {vae_decoder_actual_input_names}")

            # Prepare the dictionary of inputs for the VAE decoder
            # The keys in this dictionary MUST match the names of the input layers of the loaded_vae_decoder
            # or the names provided when the VAE decoder model was originally defined.
            
            # Expected names from params:
            # self.params[\"decoder_input_name_latent\"] -> 'decoder_input_z_seq'
            # self.params[\"decoder_input_name_conditions\"] -> 'decoder_input_conditions'
            # self.params[\"decoder_input_name_context\"] -> 'decoder_input_h_context'
            # self.params[\"decoder_input_name_window\"] -> 'input_x_window' (potentially)

            vae_inputs_for_loaded_decoder = {}
            
            # Map our generated/provided tensors to the VAE decoder's *actual* input layers
            # This requires knowing the exact names the VAE decoder expects.
            # We assume the names in plugin_params are the target names for the VAE decoder's inputs.

            # Attempt to map by the names defined in params
            # This is crucial: the names must match what the loaded_vae_decoder expects.
            # For example, if loaded_vae_decoder.inputs[0] is named 'z_input_for_vae', then
            # self.params[\"decoder_input_name_latent\"] should be 'z_input_for_vae'.
            
            # Let's iterate through the VAE decoder's actual inputs and try to map them
            # from our prepared tensors using the names stored in self.params.
            
            param_to_tensor_map = {
                self.params["decoder_input_name_latent"]: vae_decoder_input_z_seq,
                self.params["decoder_input_name_conditions"]: vae_decoder_input_conditions,
                self.params["decoder_input_name_context"]: vae_decoder_input_h_context,
                # How to handle self.params[\"decoder_input_name_window\"] ('input_x_window')?
                # If the VAE decoder requires 'input_x_window', we need to provide it.
                # For a GAN generating from noise, this input might not be directly available
                # from the composite generator's external inputs.
                # It might need to be a learned constant, zeros, or derived differently.
            }

            # Check if 'input_x_window' is one of the VAE decoder's inputs
            input_x_window_name = self.params["decoder_input_name_window"] # Default: "input_x_window"
            
            # Placeholder for input_x_window if needed.
            # This input is (batch_size, sequence_length, num_features_for_x_window)
            # The VAE decoder was trained with 144 timesteps, 57 features.
            # If this input is strictly required and not internally handled by the VAE decoder
            # when z_seq is provided, we might need to create a dummy input or a learnable one.
            # For now, we will only pass the three main inputs (z, h_context, conditions)
            # and rely on the VAE decoder to handle the absence of input_x_window if it can,
            # or error out if it's mandatory. The error handling below will catch this.

            # Construct the input list/dict for the VAE decoder based on its actual input layers
            # This is the most robust way: use the VAE decoder's .inputs attribute
            final_vae_decoder_inputs = []
            if len(vae_decoder_actual_input_names) == 3: # Assuming order: z, h_context, conditions
                 self.logger.info("Attempting to connect VAE decoder with 3 inputs in assumed order: z_seq, h_context, conditions.")
                 # This order needs to match how the VAE decoder was defined.
                 # A safer bet is to map by name if possible, or ensure the order is known.
                 # For now, using the map created earlier, and then ordering them.
                 # This part is highly dependent on the VAE model's construction.
                 # If the VAE model was created as `Model(inputs=[z_input, context_input, cond_input], ...)`
                 # then the list order must match.
                 
                 # Let's try to build the list based on the names in self.params, assuming they are correct
                 # and in the order the VAE expects if it takes a list.
                 # This is still a bit risky if the internal names of VAE inputs differ from these param names.
                 
                 # A more robust approach if the VAE model expects a dictionary for its inputs:
                 # vae_inputs_for_loaded_decoder_dict = {
                 #    self.params[\"decoder_input_name_latent\"]: vae_decoder_input_z_seq,
                 #    self.params[\"decoder_input_name_context\"]: vae_decoder_input_h_context,
                 #    self.params[\"decoder_input_name_conditions\"]: vae_decoder_input_conditions
                 # }
                 # If input_x_window is required, it must be added here.
                 # For example, if vae_decoder_actual_input_names includes input_x_window_name:
                 #    # Create a dummy input for input_x_window for the composite model
                 #    dummy_x_window_input_shape = (self.params[\"decoder_input_window_size\"], self.params.get(\"num_features_for_x_window\", 57)) # Assuming 57 features
                 #    composite_input_x_window = Input(shape=dummy_x_window_input_shape, name=\"composite_input_x_window_for_vae\")
                 #    # Add this to the composite model's main inputs
                 #    # And map it:
                 #    vae_inputs_for_loaded_decoder_dict[input_x_window_name] = composite_input_x_window
                 #    # The main composite model inputs list would then include composite_input_x_window.

                 # For now, sticking to the 3 inputs as per the primary task description.
                 # The order in the list matters if the model expects a list.
                 # The VAE decoder's `loaded_vae_decoder.inputs` gives the Keras Input *layers*.
                 # Their names are in `vae_decoder_actual_input_names`.
                 
                 # Let's try to pass inputs as a dictionary, which is safer if the model supports it.
                 # The keys of the dictionary should be the *names of the input layers* of the VAE decoder.
                 
                 input_mapping_for_vae = {}
                 # Try to map based on the names in params, assuming they match VAE's input layer names
                 if self.params["decoder_input_name_latent"] in vae_decoder_actual_input_names:
                     input_mapping_for_vae[self.params["decoder_input_name_latent"]] = vae_decoder_input_z_seq
                 if self.params["decoder_input_name_context"] in vae_decoder_actual_input_names:
                     input_mapping_for_vae[self.params["decoder_input_name_context"]] = vae_decoder_input_h_context
                 if self.params["decoder_input_name_conditions"] in vae_decoder_actual_input_names:
                     input_mapping_for_vae[self.params["decoder_input_name_conditions"]] = vae_decoder_input_conditions
                 
                 # Check if all VAE inputs are covered (excluding input_x_window for now unless explicitly handled)
                 # This is a simplified check. A more robust check would ensure all `vae_decoder_actual_input_names`
                 # (perhaps excluding 'input_x_window' if we are intentionally omitting it) are keys in `input_mapping_for_vae`.

                 self.logger.info(f"Constructed VAE input map: {input_mapping_for_vae.keys()}")

                 if not input_mapping_for_vae: # If the map is empty, names probably didn't match
                     raise ValueError("Could not map any prepared inputs to the VAE decoder's input layers. Check names in params.")

                 # Call the VAE decoder. If it was created with named inputs, a dict should work.
                 # If it was created with a list of inputs, Keras might still map a dict if names are unique.
                 vae_decoder_output = loaded_vae_decoder(input_mapping_for_vae)

            elif input_x_window_name in vae_decoder_actual_input_names and len(vae_decoder_actual_input_names) > 3 :
                 # This case suggests input_x_window is also expected.
                 # For the composite GAN generator, we are *not* taking input_x_window as a direct external input.
                 # If the VAE decoder *must* have it, we need to provide a placeholder or a learned tensor.
                 # This is a common challenge when repurposing a VAE decoder.
                 self.logger.warning(f"VAE Decoder expects input '{input_x_window_name}' which is not directly provided by the composite generator's current inputs. This might lead to errors if not handled by the VAE decoder internally or if a placeholder is not supplied.")
                 # For now, we'll proceed without it and see if Keras/TF can handle it (e.g., if it's optional or has a default).
                 # If an error occurs, this is the likely cause.
                 # To properly handle this, we would need to:
                 # 1. Add an Input layer for 'input_x_window' to the *composite model's* inputs.
                 # 2. Pass this new Input layer to the `loaded_vae_decoder`.
                 # However, the task is to generate z_seq internally.
                 # A possible solution is to use a ZeroPadding or a similar layer if the VAE expects it but we don't provide meaningful data.
                 # This is an advanced topic. For now, we assume the VAE can operate with z, h_ctx, cond.
                 
                 # Fallback to previous attempt if above logic is too complex for now
                 input_mapping_for_vae = {}
                 if self.params["decoder_input_name_latent"] in vae_decoder_actual_input_names:
                     input_mapping_for_vae[self.params["decoder_input_name_latent"]] = vae_decoder_input_z_seq
                 if self.params["decoder_input_name_context"] in vae_decoder_actual_input_names:
                     input_mapping_for_vae[self.params["decoder_input_name_context"]] = vae_decoder_input_h_context
                 if self.params["decoder_input_name_conditions"] in vae_decoder_actual_input_names:
                     input_mapping_for_vae[self.params["decoder_input_name_conditions"]] = vae_decoder_input_conditions
                 
                 self.logger.info(f"Constructed VAE input map (with potential missing '{input_x_window_name}'): {input_mapping_for_vae.keys()}")
                 if not input_mapping_for_vae:
                     raise ValueError("Could not map any prepared inputs to the VAE decoder's input layers (case with input_x_window). Check names in params.")
                 vae_decoder_output = loaded_vae_decoder(input_mapping_for_vae)
            else:
                # If the number of inputs is different, or names don't match, this will likely fail.
                # Defaulting to the original list-based attempt, hoping the order in params is correct.
                self.logger.warning(f"VAE Decoder has an unexpected number of inputs ({len(vae_decoder_actual_input_names)}) or names do not match expected pattern. Attempting connection with primary inputs.")
                # This is the risky part from before.
                # The VAE decoder must be called with inputs that match its definition.
                # If it expects a list: loaded_vae_decoder([input1, input2, ...])
                # If it expects a dict: loaded_vae_decoder({'name1': input1, 'name2': input2, ...})
                
                # Try to build the input list in the order of VAE's own input layers,
                # fetching tensors from our param_to_tensor_map using the VAE's layer names.
                ordered_inputs_for_vae = []
                successfully_mapped_inputs = True
                for name in vae_decoder_actual_input_names:
                    if name == self.params["decoder_input_name_latent"]:
                        ordered_inputs_for_vae.append(vae_decoder_input_z_seq)
                    elif name == self.params["decoder_input_name_context"]:
                        ordered_inputs_for_vae.append(vae_decoder_input_h_context)
                    elif name == self.params["decoder_input_name_conditions"]:
                        ordered_inputs_for_vae.append(vae_decoder_input_conditions)
                    elif name == input_x_window_name:
                        # This is where we'd need to provide input_x_window if it's mandatory.
                        # For now, if it's encountered and we haven't prepared it, this will be an issue.
                        self.logger.error(f"VAE Decoder expects '{input_x_window_name}' but it's not explicitly prepared by the composite generator in this path.")
                        # To make this runnable, one might add a dummy input here, but it's better to define it for the composite model.
                        # For example:
                        # dummy_x_shape = (self.params["decoder_input_window_size"], 1) # Minimal dummy
                        # dummy_x_tensor = tf.zeros((1, *dummy_x_shape)) # This is not a Keras symbolic tensor
                        # This part needs a Keras Input tensor if it's to be part of the model graph.
                        # For now, let this path likely lead to an error if input_x_window is required.
                        successfully_mapped_inputs = False 
                        break 
                    else:
                        self.logger.error(f"Unknown VAE input layer name encountered: {name}. Cannot map.")
                        successfully_mapped_inputs = False
                        break
                
                if not successfully_mapped_inputs or len(ordered_inputs_for_vae) != len(vae_decoder_actual_input_names):
                    raise ValueError(f"Mismatch when trying to order inputs for VAE decoder. Expected {len(vae_decoder_actual_input_names)} inputs, got {len(ordered_inputs_for_vae)}. Check VAE input names and param configurations.")

                self.logger.info(f"Attempting to call VAE decoder with an ordered list of {len(ordered_inputs_for_vae)} inputs.")
                vae_decoder_output = loaded_vae_decoder(ordered_inputs_for_vae)


            # --- 5. Create the new Composite Generator Model ---
            # The inputs to the composite model are those defined in step 2.
            composite_model_inputs = [feeder_noise_input, previous_step_output_input, current_step_conditions_input]
            
            # If input_x_window was added as an input to the composite model (e.g., composite_input_x_window),
            # it should be included in composite_model_inputs here.
            # For example:
            # if 'composite_input_x_window' in locals():
            #    composite_model_inputs.append(composite_input_x_window)


            self.sequential_model = Model(
                inputs=composite_model_inputs,
                outputs=vae_decoder_output, # This should be the output from the VAE decoder's 'reconstruction_out' layer
                name="composite_sc_vae_gan_generator"
            )
            self.model = self.sequential_model # Maintain alias

            self.logger.info("Composite SC-VAE-GAN generator model built successfully.")
            self.model.summary(print_fn=self.logger.info)

        except Exception as e:
            self.logger.error(f"Error building composite generator model: {e}")
            self.logger.error(traceback.format_exc()) # Log the full traceback
            self.sequential_model = None
            self.model = None
            raise IOError(f"Failed to build composite generator model: {e}")

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
        # Model path is for the VAE DECODER now
        vae_decoder_model_path = self.params.get("sequential_model_file")
        if not vae_decoder_model_path:
            raise ValueError("'sequential_model_file' (for VAE Decoder) parameter is required.")
        
        if self.sequential_model is None: # This is now the composite model
            raise RuntimeError(f"Composite generator model (using VAE decoder from {vae_decoder_model_path}) could not be built.")
        
        if not self.params.get("full_feature_names_ordered"):
            raise ValueError("'full_feature_names_ordered' parameter is required.")
        
        if not self.params.get("decoder_output_feature_names"): # These are outputs of the VAE DECODER
            raise ValueError("'decoder_output_feature_names' parameter is required.")

    def generate(self, 
                 feeder_outputs_batch: List[Dict[str, np.ndarray]], # Batch of feeder outputs
                 sequence_length_T: int, # This is the GAN output sequence length
                 # initial_full_feature_window and related args might change or be handled differently
                 # For iterative generation, we mainly need the very first seed.
                 initial_context_vector: Optional[np.ndarray] = None, # (batch_size, context_vector_dim)
                 initial_conditions_vector: Optional[np.ndarray] = None # (batch_size, conditional_features_dim)
                 ) -> List[np.ndarray]: # Return a list of generated sequences (one per item in batch)
        """
        Generate synthetic sequences using the composite generator model iteratively.
        
        Args:
            feeder_outputs_batch: List of dictionaries from FeederPlugin. Each dict contains
                                  'noise' (for internal_z_seq_output) and 'datetime_features' (for conditions).
                                  Length of list is batch_size.
            sequence_length_T: Desired length of the output sequences (GAN's T).
            initial_context_vector: The very first h_context (e.g., zeros or from real data).
                                    Shape (batch_size, context_vector_dim).
            initial_conditions_vector: The very first conditions_t (e.g., from real data or start datetime).
                                       Shape (batch_size, conditional_features_dim).

        Returns:
            List of generated sequences. Each element is np.ndarray of shape 
            (sequence_length_T, num_decoder_output_features).
        """
        if self.model is None:
            raise RuntimeError("Composite generator model is not built/loaded.")
        if not self.sequence_builder: # SequenceBuilder might be less relevant now or needs adaptation
            self.logger.warning("SequenceBuilder not initialized. Its role might change with iterative generation.")
        
        batch_size = len(feeder_outputs_batch)
        num_decoder_output_features = len(self.params["decoder_output_feature_names"])

        # Initialize lists to store the full generated sequences for each item in the batch
        batch_generated_sequences = [np.zeros((sequence_length_T, num_decoder_output_features)) for _ in range(batch_size)]

        # Prepare initial context and conditions
        current_h_context_batch = initial_context_vector
        if current_h_context_batch is None:
            current_h_context_batch = np.zeros((batch_size, self.params["context_vector_dim"]), dtype=np.float32)
        
        # The conditions will change per step based on feeder_outputs_batch's datetime info
        # For the very first step, we might use initial_conditions_vector if provided,
        # otherwise derive from the first datetime in feeder_outputs_batch.

        self.logger.info(f"Starting iterative generation for {batch_size} samples, sequence length {sequence_length_T}.")

        for t in range(sequence_length_T):
            self.logger.debug(f"Generating step {t+1}/{sequence_length_T}")
            
            # Prepare inputs for the composite model for the current step `t` for the whole batch
            batch_feeder_noise = []
            batch_current_step_conditions = []

            for i in range(batch_size):
                feeder_output_for_sample = feeder_outputs_batch[i] # This should ideally provide noise for each step or a base noise
                
                # For simplicity, let's assume feeder_outputs_batch[i]['noise'] is the base noise for the z-generator
                # If feeder_output_for_sample['noise'] is per step, adjust accordingly.
                # Here, assuming feeder_output_for_sample['noise'] is a single vector for the z-generator input
                noise_for_z_gen = feeder_output_for_sample.get('noise') # Expected shape (feeder_noise_dim,)
                if noise_for_z_gen is None:
                    # Fallback if FeederPlugin doesn't provide 'noise' per sample directly
                    noise_for_z_gen = np.random.randn(self.params["feeder_noise_dim"]).astype(np.float32)
                batch_feeder_noise.append(noise_for_z_gen)

                # Derive current_step_conditions from feeder_outputs_batch[i]['datetime_features'] for step `t`
                # This requires FeederPlugin to provide datetime features for each step `t` of `sequence_length_T`
                # Or, we need a way to advance datetime and calculate these features here.
                # For now, let's assume feeder_output_for_sample['datetime_features_sequence'][t] exists
                # and is a vector of shape (conditional_features_dim,)
                # This part needs careful design with FeederPlugin.
                # Simplified: use a placeholder or initial_conditions_vector for all steps if not dynamic
                if initial_conditions_vector is not None and t == 0 :
                     conditions_for_step_t = initial_conditions_vector[i]
                elif 'datetime_features_sequence' in feeder_output_for_sample and t < len(feeder_output_for_sample['datetime_features_sequence']):
                     conditions_for_step_t = feeder_output_for_sample['datetime_features_sequence'][t] # Expected (conditional_features_dim,)
                else: # Fallback: re-use initial or zeros
                     conditions_for_step_t = np.zeros(self.params["conditional_features_dim"], dtype=np.float32)
                batch_current_step_conditions.append(conditions_for_step_t)

            # Convert lists to numpy arrays for batch processing
            batch_feeder_noise_np = np.array(batch_feeder_noise)
            batch_current_step_conditions_np = np.array(batch_current_step_conditions)
            # current_h_context_batch is already a numpy array (batch_size, context_vector_dim)

            # Predict one step using the composite model
            # Inputs: [feeder_noise, previous_step_output (for h_context), current_step_conditions]
            predicted_step_batch = self.model.predict_on_batch([
                batch_feeder_noise_np,
                current_h_context_batch, 
                batch_current_step_conditions_np
            ]) # Expected output shape: (batch_size, num_decoder_output_features)

            # Store the predicted step and update h_context for the next iteration
            for i in range(batch_size):
                batch_generated_sequences[i][t, :] = predicted_step_batch[i]
            
            # Update current_h_context_batch for the next step using the *current* predictions
            # The VAE decoder output (predicted_step_batch) needs to be transformed into the
            # shape/content expected by decoder_input_h_context (batch_size, 64).
            # This might involve selecting specific features or applying a transformation.
            # For now, a direct use or a simple transformation is assumed.
            # If predicted_step_batch features are different from context_vector_dim, this needs a mapping.
            if predicted_step_batch.shape[1] == self.params["context_vector_dim"]:
                current_h_context_batch = predicted_step_batch.astype(np.float32)
            else:
                # Placeholder: if dimensions don't match, we need a strategy.
                # e.g., take first context_vector_dim features, or apply a Dense layer.
                # This is a simplification.
                self.logger.warning(f"Dimension mismatch for h_context. Decoder output features: {predicted_step_batch.shape[1]}, expected context_dim: {self.params['context_vector_dim']}. Using zeros or truncated.")
                if predicted_step_batch.shape[1] > self.params["context_vector_dim"]:
                     current_h_context_batch = predicted_step_batch[:, :self.params["context_vector_dim"]].astype(np.float32)
                else: # Pad with zeros if too short
                    padding_needed = self.params["context_vector_dim"] - predicted_step_batch.shape[1]
                    current_h_context_batch = np.pad(predicted_step_batch, ((0,0), (0, padding_needed)), 'constant').astype(np.float32)


        # --- Post-processing after generating all T steps ---
        # The batch_generated_sequences now contains the raw outputs from VAE decoder.
        # We need to:
        # 1. Potentially denormalize (if normalization was applied before VAE decoder training)
        # 2. Calculate technical indicators on these generated sequences.
        # 3. Assemble the full feature set as defined by full_feature_names_ordered.

        final_output_sequences = []
        for i in range(batch_size):
            # This is the raw output from the VAE decoder for one sample
            raw_generated_df = pd.DataFrame(batch_generated_sequences[i], columns=self.params["decoder_output_feature_names"])
            
            # Placeholder for datetimes - FeederPlugin should provide these for the full sequence_length_T
            # For now, creating a dummy datetime index for TI calculation.
            # This needs to be coordinated with FeederPlugin.
            if 'datetime_sequence_for_output' in feeder_outputs_batch[i] and \
               len(feeder_outputs_batch[i]['datetime_sequence_for_output']) == sequence_length_T:
                datetimes_for_output = pd.to_datetime(feeder_outputs_batch[i]['datetime_sequence_for_output'])
            else: # Fallback dummy datetimes
                datetimes_for_output = pd.date_range(start="2000-01-01", periods=sequence_length_T, freq="H") 
            
            raw_generated_df.index = datetimes_for_output

            # --- Denormalization (if applicable) ---
            # If the VAE decoder was trained on normalized data, denormalize here.
            # This uses the NormalizationHandler.
            # denormalized_df = self.normalization_handler.denormalize_data(raw_generated_df, self.params["decoder_output_feature_names"])
            # For now, assume raw_generated_df is what we work with for TIs.
            # The `DataGenerator` module might be adapted for this post-processing.
            
            # --- Calculate Technical Indicators ---
            # We need OHLC from the raw_generated_df to calculate TIs.
            # Ensure OHLC columns are present in decoder_output_feature_names or can be derived.
            # This part is similar to what DataGenerator and SequenceBuilder did.
            
            # Create a temporary DataFrame with enough history for TI calculation if needed.
            # For simplicity, assume raw_generated_df has the necessary OHLC.
            # The ti_calculator expects a DataFrame with 'OPEN', 'HIGH', 'LOW', 'CLOSE' columns.
            
            # This is a simplified call. The ti_calculator might need more context or history.
            # The `DataGenerator`'s `_calculate_technical_indicators` can be reused/adapted.
            
            # Let's assume we need to construct a DataFrame that ti_calculator can use.
            # It needs at least OHLC.
            ohlc_present = all(col in raw_generated_df.columns for col in self.params["ohlc_feature_names"])
            if not ohlc_present and "CLOSE" in raw_generated_df.columns: # If only CLOSE is there, make dummy OHLC
                for col in ["OPEN", "HIGH", "LOW"]:
                    if col not in raw_generated_df.columns:
                        raw_generated_df[col] = raw_generated_df["CLOSE"]
            
            if self.ti_calculator:
                ti_df = self.ti_calculator.calculate_technical_indicators(raw_generated_df.copy()) # Pass a copy
                # Merge TIs with generated data
                generated_with_ti_df = pd.concat([raw_generated_df, ti_df], axis=1)
            else:
                generated_with_ti_df = raw_generated_df

            # --- Assemble final features based on full_feature_names_ordered ---
            # This involves selecting columns, potentially adding date features (already in conditions), etc.
            # The `DataGenerator`'s `_assemble_final_feature_vector` logic is relevant here.
            
            # For now, just ensure the columns are in the right order and all are present.
            # This is a placeholder for the full feature assembly.
            final_df_for_sample = pd.DataFrame(index=generated_with_ti_df.index)
            for col_name in self.params["full_feature_names_ordered"]:
                if col_name in generated_with_ti_df.columns:
                    final_df_for_sample[col_name] = generated_with_ti_df[col_name]
                elif col_name == "DATE_TIME":
                     final_df_for_sample[col_name] = generated_with_ti_df.index
                # Add logic for date features (sin/cos) if not already handled by conditions
                # Add logic for fundamental features if they are part of full_feature_names_ordered
                # and need to be sourced/repeated.
                else:
                    final_df_for_sample[col_name] = 0 # Placeholder for missing features

            final_output_sequences.append(final_df_for_sample[self.params["full_feature_names_ordered"]].values)
            
        # The output for GAN trainer is typically a list of numpy arrays (one per batch item)
        # Each array is (sequence_length_T, num_gan_output_features)
        # num_gan_output_features is len(self.params["base_feature_names_ordered"]) for the GAN trainer
        # The GAN trainer will then add TIs on top of this for the discriminator.
        
        # So, the output of this `generate` method should be the "base features" that the GAN trainer expects.
        # Let's assume `base_feature_names_ordered` from config defines these.
        
        gan_trainer_output_batch = []
        for final_sequence_np in final_output_sequences:
            # Convert back to DataFrame to easily select columns
            temp_df = pd.DataFrame(final_sequence_np, columns=self.params["full_feature_names_ordered"])
            base_features_df = temp_df[self.params["base_feature_names_ordered"]]
            gan_trainer_output_batch.append(base_features_df.values)

        return gan_trainer_output_batch # List of (T, num_base_features)
