#!/usr/bin/env python3
"""
Script to inspect the actual VAE decoder model inputs and outputs.
"""
import tensorflow as tf
from tensorflow import keras
keras.config.enable_unsafe_deserialization()

def inspect_vae_decoder():
    """Inspect the pre-trained VAE decoder model."""
    model_path = "examples/results/phase_4_3/phase_4_3_cnn_small_decoder_model.keras"
    
    try:
        print(f"Loading VAE decoder from: {model_path}")
        vae_decoder = tf.keras.models.load_model(model_path)
        
        print(f"\nVAE Decoder Model: {vae_decoder.name}")
        print(f"Number of inputs: {len(vae_decoder.inputs)}")
        print(f"Number of outputs: {len(vae_decoder.outputs)}")
        
        print("\nInput details:")
        for i, inp in enumerate(vae_decoder.inputs):
            print(f"  Input {i}: name='{inp.name}', shape={inp.shape}")
        
        print("\nOutput details:")
        for i, out in enumerate(vae_decoder.outputs):
            print(f"  Output {i}: name='{out.name}', shape={out.shape}")
            
        print("\nModel summary:")
        vae_decoder.summary()
        
        return vae_decoder
        
    except Exception as e:
        print(f"Error loading VAE decoder: {e}")
        return None

if __name__ == "__main__":
    inspect_vae_decoder()
