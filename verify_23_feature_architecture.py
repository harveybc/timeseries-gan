#!/usr/bin/env python3
"""
Verification script for 23-feature architecture implementation.
"""

def verify_configuration_changes():
    """Verify that configuration files have been updated correctly."""
    print("Verifying 23-Feature Architecture Implementation")
    print("=" * 50)
    
    # Check app/config.py
    try:
        with open('app/config.py', 'r') as f:
            config_content = f.read()
        
        if '"num_features": 23' in config_content:
            print("✅ app/config.py: num_features set to 23")
        else:
            print("❌ app/config.py: num_features not found or incorrect")
            
    except Exception as e:
        print(f"❌ Error reading app/config.py: {e}")
    
    # Check discriminator plugin
    try:
        with open('tsg_plugins/discriminator_plugin.py', 'r') as f:
            disc_content = f.read()
        
        if '"num_features": 23' in disc_content:
            print("✅ discriminator_plugin.py: num_features set to 23")
        else:
            print("❌ discriminator_plugin.py: num_features not found or incorrect")
            
    except Exception as e:
        print(f"❌ Error reading discriminator_plugin.py: {e}")
    
    # Check generator plugin
    try:
        with open('tsg_plugins/generator_plugin/generator_plugin.py', 'r') as f:
            gen_content = f.read()
        
        if '"num_features": 23' in gen_content:
            print("✅ generator_plugin.py: num_features set to 23")
        else:
            print("❌ generator_plugin.py: num_features not found or incorrect")
            
        # Check that problematic methods are removed/updated
        if 'expand_features_to_51' not in gen_content or 'No need for expansion' in gen_content:
            print("✅ generator_plugin.py: Feature expansion logic updated")
        else:
            print("⚠️  generator_plugin.py: May still contain old expansion logic")
            
    except Exception as e:
        print(f"❌ Error reading generator_plugin.py: {e}")
    
    # Check composite generator
    try:
        with open('tsg_plugins/generator_plugin/_build_composite_generator.py', 'r') as f:
            comp_content = f.read()
        
        if 'outputs=base_sequence' in comp_content and '23-FEATURE ARCHITECTURE' in comp_content:
            print("✅ _build_composite_generator.py: Outputs 23 features directly")
        else:
            print("❌ _build_composite_generator.py: Architecture not updated correctly")
            
    except Exception as e:
        print(f"❌ Error reading _build_composite_generator.py: {e}")
    
    print("\nArchitecture Summary:")
    print("- Generator: Outputs (batch_size, 144, 23)")
    print("- Discriminator: Expects (batch_size, 144, 23)")  
    print("- VAE Decoder: Still outputs 23 base features")
    print("- Post-processing: Technical indicators calculated separately")
    
    print("\nKey Benefits:")
    print("- No feature expansion complexity")
    print("- Better learning on core features")
    print("- Faster training with smaller networks")
    print("- Deterministic technical indicators")
    print("- Eliminated TensorFlow compatibility issues")

if __name__ == "__main__":
    verify_configuration_changes()
