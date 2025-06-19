#!/usr/bin/env python3
"""
Quick verification test for the GeneratorPlugin method signature fix.
This tests the specific issue: "_build_composite_generator() takes 1 positional argument but 2 were given"
"""

import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(__file__))

def test_method_signature_fix():
    """Test that the _build_composite_generator method signature has been fixed."""
    
    print("=== Testing GeneratorPlugin Method Signature Fix ===")
    
    try:
        # Import the GeneratorPlugin
        from tsg_plugins.generator_plugin.generator_plugin import GeneratorPlugin
        print("✓ Successfully imported GeneratorPlugin")
        
        # Check the method signature using inspection
        import inspect
        method = getattr(GeneratorPlugin, '_build_composite_generator')
        signature = inspect.signature(method)
        
        print(f"Method signature: {signature}")
        
        # Verify the signature accepts a vae_decoder_model parameter
        params = list(signature.parameters.keys())
        expected_params = ['self', 'vae_decoder_model']
        
        if 'vae_decoder_model' in params:
            print("✓ Method accepts vae_decoder_model parameter")
        else:
            print("❌ Method does not accept vae_decoder_model parameter")
            return False
            
        # Test instantiation with minimal config
        config = {'x_train_file': 'tests/data/sample_data.csv'}
        plugin = GeneratorPlugin(config)
        print("✓ GeneratorPlugin instantiated successfully")
        
        # Test calling the method with no arguments (should work)
        try:
            model1 = plugin._build_composite_generator()
            print("✓ _build_composite_generator() called successfully without VAE decoder")
        except Exception as e:
            print(f"❌ Error calling method without VAE decoder: {e}")
            return False
        
        # Test calling the method with None argument (should work)
        try:
            model2 = plugin._build_composite_generator(None)
            print("✓ _build_composite_generator(None) called successfully")
        except Exception as e:
            print(f"❌ Error calling method with None: {e}")
            return False
        
        print("✅ All signature tests passed! The fix is working correctly.")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_load_model_workflow():
    """Test the _load_model workflow that was causing the original error."""
    
    print("\n=== Testing _load_model Workflow ===")
    
    try:
        from tsg_plugins.generator_plugin.generator_plugin import GeneratorPlugin
        
        # Create plugin instance
        config = {'x_train_file': 'tests/data/sample_data.csv'}
        plugin = GeneratorPlugin(config)
        
        # Test the _load_model method with a non-existent file (should handle gracefully)
        try:
            plugin._load_model("non_existent_file.keras")
            print("✓ _load_model handled non-existent file gracefully")
        except Exception as e:
            # This is expected to fail, but should not be due to method signature
            if "takes 1 positional argument but 2 were given" in str(e):
                print("❌ Method signature error still present!")
                return False
            else:
                print(f"✓ _load_model failed as expected (not due to signature): {type(e).__name__}")
        
        print("✅ _load_model workflow test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error in _load_model workflow test: {e}")
        return False

if __name__ == "__main__":
    print("Running GeneratorPlugin fix verification tests...\n")
    
    success1 = test_method_signature_fix()
    success2 = test_load_model_workflow()
    
    if success1 and success2:
        print("\n🎉 ALL TESTS PASSED! The GeneratorPlugin fix is working correctly.")
        print("The error '_build_composite_generator() takes 1 positional argument but 2 were given' has been resolved.")
    else:
        print("\n❌ Some tests failed. Please check the issues above.")
        sys.exit(1)
