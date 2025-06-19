#!/usr/bin/env python3
"""
Analyze Conditioning Incompatibility

This script analyzes the incompatibility between autoencoder training and current GAN system
for conditional inputs, and proposes solutions.
"""

import os
import sys
import numpy as np
import pandas as pd

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def analyze_current_conditioning_setup():
    """Analyze the current GAN system's conditional input generation."""
    print("🔍 ANALYZING CURRENT GAN CONDITIONING SETUP")
    print("=" * 60)
    
    try:
        from app.config import DEFAULT_VALUES
        from tsg_plugins.feeder_plugin.condition_manager import ConditionManager
        
        # Get config values
        date_features = DEFAULT_VALUES.get('feeder_date_features_for_conditioning', [])
        conditional_dim = DEFAULT_VALUES.get('conditional_features_dim', 10)
        context_dim = DEFAULT_VALUES.get('context_vector_dim', 64)
        
        print(f"📋 Current Configuration:")
        print(f"   Date features for conditioning: {date_features}")
        print(f"   Expected cyclical features: {len(date_features) * 2} (sin/cos pairs)")
        print(f"   Configured conditional_features_dim: {conditional_dim}")
        print(f"   Configured context_vector_dim: {context_dim}")
        
        # Test condition manager
        condition_manager = ConditionManager(DEFAULT_VALUES)
        
        # Create sample data
        sample_data = pd.DataFrame({
            'DATE_TIME': pd.date_range('2024-01-01', periods=5, freq='1H'),
            'feature1': np.random.randn(5),
            'feature2': np.random.randn(5)
        })
        
        if condition_manager.initialize(sample_data):
            print(f"✅ ConditionManager initialized successfully")
            print(f"   Calculated condition dimension: {condition_manager.get_condition_dim()}")
            
            # Extract conditions
            conditions = condition_manager.extract_conditions(
                sample_data, 
                timestamp_col='DATE_TIME'
            )
            
            if conditions is not None:
                print(f"✅ Conditions extracted successfully")
                print(f"   Conditions shape: {conditions.shape}")
                print(f"   Conditions sample (first row): {conditions[0]}")
                print(f"   Conditions range: [{conditions.min():.3f}, {conditions.max():.3f}]")
                return {
                    'success': True,
                    'conditions': conditions,
                    'dimension': condition_manager.get_condition_dim(),
                    'range': [conditions.min(), conditions.max()],
                    'meaningful_values': True
                }
            else:
                print(f"❌ Failed to extract conditions")
                return {'success': False}
        else:
            print(f"❌ Failed to initialize ConditionManager")
            return {'success': False}
            
    except Exception as e:
        print(f"❌ Error testing current system: {e}")
        return {'success': False}

def analyze_autoencoder_training_format():
    """Analyze the format used during autoencoder training."""
    print("\n🔍 ANALYZING AUTOENCODER TRAINING FORMAT")
    print("=" * 60)
    
    # Based on the analysis from feature-extractor repo
    print("📋 Autoencoder Training Format (from feature-extractor analysis):")
    print("   - Conditional inputs (conditions_t_train) were generated as ZERO arrays")
    print("   - Shape: (num_samples, conditioning_dim=6)")
    print("   - Context inputs (h_context_train) were generated as ZERO arrays") 
    print("   - Shape: (num_samples, rnn_hidden_dim=64)")
    print("   - All values: 0.0 (no meaningful conditioning during training)")
    
    # Simulate the training format
    batch_size = 5
    conditioning_dim = 6  # From autoencoder training
    context_dim = 64     # From autoencoder training
    
    conditions_zeros = np.zeros((batch_size, conditioning_dim), dtype=np.float32)
    context_zeros = np.zeros((batch_size, context_dim), dtype=np.float32)
    
    print(f"✅ Simulated autoencoder training inputs:")
    print(f"   Conditions shape: {conditions_zeros.shape}")
    print(f"   Context shape: {context_zeros.shape}")
    print(f"   Conditions sample: {conditions_zeros[0]}")
    print(f"   Context sample: {context_zeros[0]}")
    
    return {
        'conditions': conditions_zeros,
        'context': context_zeros,
        'conditioning_dim': conditioning_dim,
        'context_dim': context_dim,
        'range': [0.0, 0.0],
        'meaningful_values': False
    }

def compare_formats(current_analysis, autoencoder_analysis):
    """Compare the two formats and identify incompatibilities."""
    print("\n🔍 COMPATIBILITY ANALYSIS")
    print("=" * 60)
    
    if not current_analysis.get('success', False):
        print("❌ Cannot compare - current system analysis failed")
        return {'compatible': False, 'issues': ['Current system failed']}
    
    issues = []
    
    # Check dimensional compatibility
    current_dim = current_analysis['dimension']
    autoencoder_dim = autoencoder_analysis['conditioning_dim']
    
    print(f"📊 Dimensional Analysis:")
    print(f"   Current GAN system: {current_dim} dimensions")
    print(f"   Autoencoder training: {autoencoder_dim} dimensions")
    
    if current_dim != autoencoder_dim:
        print(f"   ❌ DIMENSIONAL MISMATCH: {current_dim} vs {autoencoder_dim}")
        issues.append(f"Dimensional mismatch: {current_dim} vs {autoencoder_dim}")
    else:
        print(f"   ✅ Dimensions match")
    
    # Check value distribution compatibility
    current_range = current_analysis['range']
    autoencoder_range = autoencoder_analysis['range']
    current_meaningful = current_analysis['meaningful_values']
    autoencoder_meaningful = autoencoder_analysis['meaningful_values']
    
    print(f"\n📊 Value Distribution Analysis:")
    print(f"   Current GAN range: [{current_range[0]:.3f}, {current_range[1]:.3f}]")
    print(f"   Autoencoder training range: [{autoencoder_range[0]:.3f}, {autoencoder_range[1]:.3f}]")
    print(f"   Current uses meaningful values: {current_meaningful}")
    print(f"   Autoencoder used meaningful values: {autoencoder_meaningful}")
    
    if current_meaningful != autoencoder_meaningful:
        print(f"   ❌ VALUE DISTRIBUTION MISMATCH: meaningful vs zeros")
        issues.append("Value distribution mismatch: meaningful cyclical features vs zero arrays")
    else:
        print(f"   ✅ Value distributions compatible")
    
    # Context dimension check
    current_context_dim = 64  # From config
    autoencoder_context_dim = autoencoder_analysis['context_dim']
    
    print(f"\n📊 Context Vector Analysis:")
    print(f"   Current GAN context dim: {current_context_dim}")
    print(f"   Autoencoder training context dim: {autoencoder_context_dim}")
    
    if current_context_dim != autoencoder_context_dim:
        print(f"   ❌ CONTEXT DIMENSION MISMATCH: {current_context_dim} vs {autoencoder_context_dim}")
        issues.append(f"Context dimension mismatch: {current_context_dim} vs {autoencoder_context_dim}")
    else:
        print(f"   ✅ Context dimensions match")
    
    return {
        'compatible': len(issues) == 0,
        'issues': issues,
        'current_dim': current_dim,
        'autoencoder_dim': autoencoder_dim,
        'current_meaningful': current_meaningful,
        'autoencoder_meaningful': autoencoder_meaningful
    }

def propose_solutions(compatibility_analysis):
    """Propose solutions for the identified incompatibilities."""
    print("\n🔧 PROPOSED SOLUTIONS")
    print("=" * 60)
    
    if compatibility_analysis['compatible']:
        print("✅ No incompatibilities found - system should work correctly")
        return []
    
    issues = compatibility_analysis['issues']
    solutions = []
    
    print("❌ CRITICAL INCOMPATIBILITIES DETECTED:")
    for i, issue in enumerate(issues, 1):
        print(f"   {i}. {issue}")
    
    print(f"\n🛠️  SOLUTION OPTIONS:")
    
    # Solution 1: Modify GAN system to use zeros (quickest fix)
    print(f"\n1️⃣  SOLUTION 1: Modify GAN System to Match Autoencoder Training")
    print(f"   📋 Description: Change the current GAN system to use zero conditional vectors")
    print(f"   ✅ Pros:")
    print(f"      - Quickest to implement")
    print(f"      - Guaranteed compatibility with trained VAE decoder")
    print(f"      - No retraining required")
    print(f"   ❌ Cons:")
    print(f"      - Loses temporal conditioning capability")
    print(f"      - Generated data won't reflect temporal patterns")
    print(f"      - Underutilizes GAN's conditional generation potential")
    
    solutions.append({
        'name': 'Zero Conditional Vectors',
        'description': 'Modify GAN to use zero conditional vectors',
        'difficulty': 'Easy',
        'files_to_modify': [
            'tsg_plugins/feeder_plugin/condition_manager.py',
            'app/config.py'
        ]
    })
    
    # Solution 2: Retrain autoencoder with meaningful features
    print(f"\n2️⃣  SOLUTION 2: Retrain Autoencoder with Meaningful Conditional Features")
    print(f"   📋 Description: Retrain the VAE decoder with meaningful temporal features")
    print(f"   ✅ Pros:")
    print(f"      - Enables full temporal conditioning")
    print(f"      - Better quality synthetic data")
    print(f"      - Utilizes complete GAN capability")
    print(f"   ❌ Cons:")
    print(f"      - Requires significant time and compute resources")
    print(f"      - Need to modify feature-extractor training pipeline")
    print(f"      - Risk of degraded autoencoder performance")
    
    solutions.append({
        'name': 'Retrain Autoencoder',
        'description': 'Retrain VAE with meaningful conditional features',
        'difficulty': 'Hard',
        'files_to_modify': [
            '../feature-extractor/app/data_processor.py',
            '../feature-extractor/training pipeline'
        ]
    })
    
    # Solution 3: Create conditioning adapter
    print(f"\n3️⃣  SOLUTION 3: Create Conditioning Adapter Layer")
    print(f"   📋 Description: Add adapter layer to transform meaningful features to expected format")
    print(f"   ✅ Pros:")
    print(f"      - Preserves both systems as-is")
    print(f"      - Could potentially bridge the gap")
    print(f"      - Moderate implementation effort")
    print(f"   ❌ Cons:")
    print(f"      - Complex to implement correctly")
    print(f"      - May introduce artifacts")
    print(f"      - Uncertain effectiveness")
    
    solutions.append({
        'name': 'Conditioning Adapter',
        'description': 'Create adapter layer for format transformation',
        'difficulty': 'Medium',
        'files_to_modify': [
            'tsg_plugins/generator_plugin/generator_plugin.py',
            'tsg_plugins/feeder_plugin/condition_manager.py'
        ]
    })
    
    return solutions

def recommend_best_solution(solutions):
    """Recommend the best solution based on the analysis."""
    print(f"\n🎯 RECOMMENDATION")
    print("=" * 60)
    
    print(f"Based on the analysis, I recommend starting with SOLUTION 1:")
    print(f"")
    print(f"🥇 RECOMMENDED: Zero Conditional Vectors Approach")
    print(f"   Reasoning:")
    print(f"   - Provides immediate compatibility")
    print(f"   - Low risk of introducing new issues")
    print(f"   - Can be implemented and tested quickly")
    print(f"   - Allows system to function while planning longer-term improvements")
    print(f"")
    print(f"📅 Future Enhancement Path:")
    print(f"   1. Implement Solution 1 for immediate compatibility")
    print(f"   2. Test and validate the system works correctly")
    print(f"   3. Plan Solution 2 (autoencoder retraining) for enhanced capability")
    print(f"   4. Implement temporal conditioning with properly trained models")

def main():
    """Main analysis function."""
    print("🚀 CONDITIONING INCOMPATIBILITY ANALYSIS")
    print("=" * 80)
    
    # Analyze current system
    current_analysis = analyze_current_conditioning_setup()
    
    # Analyze autoencoder training format
    autoencoder_analysis = analyze_autoencoder_training_format()
    
    # Compare formats
    compatibility_analysis = compare_formats(current_analysis, autoencoder_analysis)
    
    # Propose solutions
    solutions = propose_solutions(compatibility_analysis)
    
    # Make recommendation
    recommend_best_solution(solutions)
    
    print("\n" + "=" * 80)
    print("📝 ANALYSIS COMPLETE")
    print("   Next step: Choose and implement a solution")
    print("=" * 80)

if __name__ == "__main__":
    main()
