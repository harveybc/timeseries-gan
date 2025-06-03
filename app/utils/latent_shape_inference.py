#!/usr/bin/env python3
"""
latent_shape_inference.py

Utility module for inferring latent shape dimensions from generator models.
Handles compatibility between generator and feeder plugins by automatically
detecting and configuring latent space dimensions.

This module provides focused functionality for latent shape inference
following single responsibility principle.

Author: TimeSeries-GAN Team
"""

from typing import Dict, Any, Optional, Tuple, List


def infer_and_set_latent_shape(config: Dict[str, Any], generator_plugin, 
                               feeder_plugin, trainer_plugin=None) -> None:
    """
    Infer latent shape from generator model and update plugin configurations.
    
    Analyzes the generator model's input layer to determine the expected
    latent shape and updates feeder and trainer plugins accordingly.
    This ensures compatibility between plugins that need to work with
    the same latent space dimensions.
    
    Args:
        config: Configuration dictionary to update with inferred shape
        generator_plugin: Generator plugin containing the decoder model
        feeder_plugin: Feeder plugin to update with latent shape
        trainer_plugin: Optional trainer plugin to update with latent dimensions
        
    Raises:
        ValueError: If latent shape cannot be inferred from generator model
    """
    print("Inferring latent shape from generator model...")
    
    try:
        # Extract decoder model from generator plugin
        decoder_model = _extract_decoder_model(generator_plugin)
        
        if not decoder_model:
            print("WARNING: Could not extract decoder model from generator plugin")
            return
        
        # Infer latent shape from model inputs
        latent_shape = _infer_latent_shape_from_model(decoder_model, generator_plugin)
        
        if latent_shape:
            # Update plugin configurations with inferred shape
            _update_plugins_with_latent_shape(
                config, feeder_plugin, trainer_plugin, latent_shape
            )
            print(f"✓ Latent shape inference completed: {latent_shape}")
        else:
            print("WARNING: Could not infer latent shape from generator model")
            
    except Exception as e:
        print(f"WARNING: Latent shape inference failed: {e}")
        # Don't fail the pipeline for inference issues


def _extract_decoder_model(generator_plugin):
    """
    Extract decoder model from generator plugin.
    
    Args:
        generator_plugin: Generator plugin instance
        
    Returns:
        Model instance or None if not found
    """
    # Try common attribute names for decoder model
    decoder_model = getattr(generator_plugin, "sequential_model", None)
    
    if decoder_model is None:
        decoder_model = getattr(generator_plugin, "model", None)
        
    if decoder_model is None:
        decoder_model = getattr(generator_plugin, "decoder", None)
        
    return decoder_model


def _infer_latent_shape_from_model(decoder_model, generator_plugin) -> Optional[Tuple[int, ...]]:
    """
    Infer latent shape from decoder model input layer.
    
    Args:
        decoder_model: Decoder model instance
        generator_plugin: Generator plugin for accessing configuration
        
    Returns:
        Tuple representing inferred latent shape or None if inference fails
    """
    try:
        # Check if model has inputs
        if not hasattr(decoder_model, 'inputs') or not decoder_model.inputs:
            return None
        
        # Get latent input name from generator configuration
        latent_input_name = generator_plugin.params.get("decoder_input_name_latent")
        
        if not latent_input_name:
            print("WARNING: Generator plugin missing 'decoder_input_name_latent' parameter")
            return None
        
        # Find latent input tensor
        latent_input_tensor = _find_latent_input_tensor(
            decoder_model.inputs, latent_input_name
        )
        
        if latent_input_tensor is None:
            return None
        
        # Extract shape from input tensor
        return _extract_shape_from_tensor(latent_input_tensor)
        
    except Exception as e:
        print(f"Error during shape inference: {e}")
        return None


def _find_latent_input_tensor(model_inputs, latent_input_name):
    """
    Find the latent input tensor by name from model inputs.
    
    Args:
        model_inputs: List of model input tensors
        latent_input_name: Name of the latent input tensor to find
        
    Returns:
        Input tensor or None if not found
    """
    for input_tensor in model_inputs:
        input_layer_name = input_tensor.name.split(':')[0]
        if input_layer_name == latent_input_name:
            return input_tensor
    
    print(f"WARNING: Latent input '{latent_input_name}' not found in model inputs")
    return None


def _extract_shape_from_tensor(input_tensor) -> Optional[Tuple[int, ...]]:
    """
    Extract shape tuple from input tensor.
    
    Args:
        input_tensor: Input tensor to extract shape from
        
    Returns:
        Shape tuple or None if extraction fails
    """
    try:
        current_shape = input_tensor.shape
        shape_list = []
        
        # Handle different shape types
        if hasattr(current_shape, 'as_list'):
            shape_list = current_shape.as_list()
        elif isinstance(current_shape, tuple):
            shape_list = list(current_shape)
        else:
            try:
                shape_list = list(current_shape)
            except TypeError:
                print(f"ERROR: Could not convert shape {current_shape} to list")
                return None
        
        # Validate and process shape
        if shape_list and len(shape_list) >= 2:
            # Remove batch dimension (first element is None)
            if shape_list[0] is None and len(shape_list) >= 3:
                return tuple(shape_list[1:])
            elif len(shape_list) == 2:
                return tuple(shape_list)
        
        print(f"WARNING: Unexpected shape format: {shape_list}")
        return None
        
    except Exception as e:
        print(f"Error extracting shape from tensor: {e}")
        return None


def _update_plugins_with_latent_shape(config: Dict[str, Any], feeder_plugin, 
                                     trainer_plugin, latent_shape: Tuple[int, ...]) -> None:
    """
    Update plugin configurations with inferred latent shape.
    
    Args:
        config: Configuration dictionary to update
        feeder_plugin: Feeder plugin to update
        trainer_plugin: Optional trainer plugin to update
        latent_shape: Inferred latent shape tuple
    """
    try:
        # Update configuration
        config['latent_shape'] = list(latent_shape)
        
        # Update feeder plugin
        if feeder_plugin:
            feeder_plugin.set_params(latent_shape=list(latent_shape))
            print(f"✓ Feeder plugin updated with latent shape: {latent_shape}")
        
        # Update trainer plugin if available
        if trainer_plugin and len(latent_shape) >= 2:
            trainer_plugin.set_params(
                seq_len=latent_shape[0],
                latent_dim=latent_shape[1]
            )
            print(f"✓ Trainer plugin updated with seq_len: {latent_shape[0]}, latent_dim: {latent_shape[1]}")
            
    except Exception as e:
        print(f"WARNING: Failed to update plugins with latent shape: {e}")
