#!/usr/bin/env python3
"""
Simple test to verify the conditional dimension fix.
"""

import os
import sys
import tensorflow as tf

# Enable unsafe deserialization
tf.keras.config.enable_unsafe_deserialization()

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_conditional_fix():
    """Test if the conditional dimension fix resolves the compatibility issue."""
    print("🧪 Testing Conditional Dimension Fix")
    print("=" * 50)
    
    try:
        from app.config import DEFAULT_VALUES
        from tsg_plugins.generator_plugin.generator_plugin import GeneratorPlugin
        
        # Check current config
        conditional_dim = DEFAULT_VALUES.get('conditional_features_dim')
        context_dim = DEFAULT_VALUES.get('context_vector_dim')
        date_features = DEFAULT_VALUES.get('feeder_date_features_for_conditioning', [])
        
        print(f"📋 Current Configuration:")
        print(f"   conditional_features_dim: {conditional_dim}")
        print(f"   context_vector_dim: {context_dim}")
        print(f"   date_features_for_conditioning: {date_features}")
        print(f"   Expected cyclical features: {len(date_features) * 2}")
        
        # Check VAE decoder expectations
        vae_path = "examples/results/phase_4_3/phase_4_3_cnn_small_decoder_model.keras"
        if os.path.exists(vae_path):
            print(f"\n📋 VAE Decoder Analysis:")
            try:
                vae_model = tf.keras.models.load_model(vae_path, compile=False)
                for i, inp in enumerate(vae_model.inputs):
                    print(f"   Input {i}: '{inp.name}' -> shape: {inp.shape}")
                
                # Check if dimensions match
                expected_cond_dim = vae_model.inputs[2].shape[1]  # Third input is conditions
                print(f"\n📊 Compatibility Check:")
                print(f"   VAE expects: {expected_cond_dim} conditional dimensions")
                print(f"   Config provides: {conditional_dim} conditional dimensions")
                
                if expected_cond_dim == conditional_dim:
                    print(f"   ✅ MATCH! Dimensions are compatible")
                else:
                    print(f"   ❌ MISMATCH! Need to fix config")
                    
            except Exception as e:
                print(f"   ❌ Error loading VAE: {e}")
        else:
            print(f"   ❌ VAE decoder not found at: {vae_path}")
        
        # Test generator plugin instantiation
        print(f"\n🧪 Testing GeneratorPlugin instantiation:")
        try:
            generator = GeneratorPlugin(DEFAULT_VALUES)
            generator.set_params(**DEFAULT_VALUES)
            print("   ✅ GeneratorPlugin created successfully")
            
            # Try to get the model (this is where the error was occurring)
            model = generator.get_model()
            if model is not None:
                print("   ✅ Generator model built successfully!")
                print(f"   Model input shapes: {[inp.shape for inp in model.inputs]}")
                print(f"   Model output shape: {model.output.shape}")
                return True
            else:
                print("   ❌ Generator model is None")
                return False
                
        except Exception as e:
            print(f"   ❌ Error creating generator: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_conditional_fix()
    print("\n" + "=" * 50)
    if success:
        print("🎉 SUCCESS: Conditional dimension fix works!")
    else:
        print("❌ FAILURE: Fix needs more work")
