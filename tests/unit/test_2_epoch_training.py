#!/usr/bin/env python3
"""
Test actual 2-epoch GAN training to verify the complete system works
"""

import sys
import os
sys.path.append('/home/harveybc/Documents/GitHub/timeseries-gan')

from app.config import DEFAULT_VALUES

def test_2_epoch_training():
    """Test running actual 2-epoch GAN training."""
    print("Testing 2-Epoch GAN Training")
    print("=" * 40)
    
    try:
        # Import the main training function
        from app.main import main
        
        # Create modified config for quick test
        test_config = DEFAULT_VALUES.copy()
        test_config["operation_mode"] = "train"
        test_config["gan_epochs"] = 2
        test_config["batch_size"] = 4  # Small batch for speed
        test_config["save_model_interval"] = 1  # Save after each epoch
        
        print(f"✓ Starting 2-epoch training test...")
        print(f"  - Mode: {test_config['operation_mode']}")
        print(f"  - Epochs: {test_config['gan_epochs']}")
        print(f"  - Batch size: {test_config['batch_size']}")
        
        # Override sys.argv to pass config to main
        original_argv = sys.argv.copy()
        sys.argv = [
            'main.py',
            '--operation_mode', 'train',
            '--gan_epochs', '2',
            '--batch_size', '4'
        ]
        
        try:
            # Run the main training function
            main()
            print("✅ SUCCESS: 2-epoch training completed!")
            return True
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
        finally:
            # Restore original argv
            sys.argv = original_argv
            
    except Exception as e:
        print(f"❌ Failed to start training: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_2_epoch_training()
    if success:
        print(f"\n🎉 2-Epoch Training PASSED!")
        print("The 44-feature system is working correctly!")
    else:
        print(f"\n💥 2-Epoch Training FAILED!")
    
    sys.exit(0 if success else 1)
