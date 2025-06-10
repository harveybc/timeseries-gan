#!/usr/bin/env python3
"""Debug script to check feature list consistency."""

try:
    from app.config import DEFAULT_VALUES
    
    # Get the lists from config
    full_features = set(DEFAULT_VALUES['generator_full_feature_names_ordered'])
    decoder_features = set(DEFAULT_VALUES['generator_decoder_output_feature_names'])

    print("Full feature names count:", len(full_features))
    print("Decoder output feature names count:", len(decoder_features))

    # Check which decoder features are missing from full features
    missing_features = decoder_features - full_features
    if missing_features:
        print(f"\nMissing features in full_feature_names_ordered: {sorted(missing_features)}")
    else:
        print("\n✅ All decoder features are present in full_feature_names_ordered!")

    # Print the CLOSE tick features specifically
    print("\nCLOSE tick features in full list:")
    close_features = [f for f in full_features if 'CLOSE_' in f and 'tick' in f]
    for feature in sorted(close_features):
        print(f"  - {feature}")
        
    print("\nCLOSE tick features in decoder list:") 
    close_decoder = [f for f in decoder_features if 'CLOSE_' in f and 'tick' in f]
    for feature in sorted(close_decoder):
        print(f"  - {feature}")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
