"""
Model-Agnostic QSAR Pipeline Examples
======================================

This file demonstrates how to use the model-agnostic pipeline with:
- Different models (Random Forest, Ridge, XGBoost, Neural Networks, etc.)
- Different featurizers (Morgan, MACCS, RDKit descriptors, etc.)
- Custom configurations

The pipeline handles ALL data leakage prevention and validation automatically!
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MACCSkeys
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor


# ============================================================================
# EXAMPLE 1: Random Forest + Morgan Fingerprints
# ============================================================================

def example_1_random_forest_morgan():
    """
    Most common QSAR setup:
    - Model: Random Forest
    - Features: Morgan fingerprints (radius=2, 1024 bits)
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Random Forest + Morgan Fingerprints")
    print("="*70)
    
    # Load your data
    df = pd.read_csv('your_data.csv')
    
    # Define featurizer
    def morgan_featurizer(smiles_list):
        """Convert SMILES to Morgan fingerprints"""
        fingerprints = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
                fingerprints.append(np.array(fp))
            else:
                fingerprints.append(np.zeros(1024))
        return np.array(fingerprints)
    
    # Define model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42
    )
    
    # Create pipeline (handles ALL leakage prevention automatically)
    from qsar_validation import ModelAgnosticQSARPipeline
    
    pipeline = ModelAgnosticQSARPipeline(
        featurizer=morgan_featurizer,
        model=model,
        smiles_col='SMILES',
        target_col='Activity'
    )
    
    # Run complete validation (one function call!)
    results = pipeline.fit_predict_validate(df, verbose=True)
    
    # Get summary
    summary = pipeline.get_results_summary()
    print("\nResults Summary:")
    print(summary)
    
    # Make predictions on new compounds
    new_smiles = ['CCO', 'c1ccccc1', 'CC(=O)O']
    predictions = pipeline.predict(new_smiles)
    print(f"\nPredictions for new compounds: {predictions}")
    
    return results


# ============================================================================
# EXAMPLE 2: Ridge Regression + RDKit Descriptors
# ============================================================================

def example_2_ridge_rdkit_descriptors():
    """
    Linear model with interpretable features:
    - Model: Ridge regression
    - Features: RDKit 2D descriptors
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Ridge Regression + RDKit Descriptors")
    print("="*70)
    
    df = pd.read_csv('your_data.csv')
    
    # Define featurizer for RDKit descriptors
    def rdkit_descriptor_featurizer(smiles_list):
        """Convert SMILES to RDKit 2D descriptors"""
        descriptor_names = [name[0] for name in Descriptors.descList]
        features = []
        
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                desc_values = [desc_func(mol) for name, desc_func in Descriptors.descList]
                features.append(desc_values)
            else:
                features.append([0] * len(descriptor_names))
        
        return np.array(features)
    
    # Define model
    model = Ridge(alpha=1.0)
    
    # Create pipeline
    from qsar_validation import ModelAgnosticQSARPipeline
    
    pipeline = ModelAgnosticQSARPipeline(
        featurizer=rdkit_descriptor_featurizer,
        model=model,
        smiles_col='SMILES',
        target_col='Activity'
    )
    
    results = pipeline.fit_predict_validate(df, verbose=True)
    
    return results


# ============================================================================
# EXAMPLE 3: XGBoost + MACCS Keys
# ============================================================================

def example_3_xgboost_maccs():
    """
    Gradient boosting with structural keys:
    - Model: XGBoost
    - Features: MACCS keys (166 bits)
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: XGBoost + MACCS Keys")
    print("="*70)
    
    try:
        from xgboost import XGBRegressor
    except ImportError:
        print("XGBoost not installed. Install with: pip install xgboost")
        return None
    
    df = pd.read_csv('your_data.csv')
    
    # Define featurizer
    def maccs_featurizer(smiles_list):
        """Convert SMILES to MACCS keys"""
        fingerprints = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                fp = MACCSkeys.GenMACCSKeys(mol)
                fingerprints.append(np.array(fp))
            else:
                fingerprints.append(np.zeros(167))  # MACCS has 167 bits
        return np.array(fingerprints)
    
    # Define model
    model = XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    
    # Create pipeline
    from qsar_validation import ModelAgnosticQSARPipeline
    
    pipeline = ModelAgnosticQSARPipeline(
        featurizer=maccs_featurizer,
        model=model,
        smiles_col='SMILES',
        target_col='Activity'
    )
    
    results = pipeline.fit_predict_validate(df, verbose=True)
    
    return results


# ============================================================================
# EXAMPLE 4: Neural Network + Custom Features
# ============================================================================

def example_4_neural_network_custom():
    """
    Neural network with custom feature combination:
    - Model: Multi-layer Perceptron
    - Features: Morgan + RDKit descriptors (concatenated)
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Neural Network + Custom Features")
    print("="*70)
    
    df = pd.read_csv('your_data.csv')
    
    # Define custom featurizer (combines multiple types)
    def custom_featurizer(smiles_list):
        """Combine Morgan fingerprints + RDKit descriptors"""
        features = []
        
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                # Morgan fingerprint
                morgan = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=512)
                morgan_array = np.array(morgan)
                
                # RDKit descriptors (first 50)
                descriptors = [desc_func(mol) for name, desc_func in Descriptors.descList[:50]]
                desc_array = np.array(descriptors)
                
                # Concatenate
                combined = np.concatenate([morgan_array, desc_array])
                features.append(combined)
            else:
                features.append(np.zeros(562))  # 512 + 50
        
        return np.array(features)
    
    # Define model
    model = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42
    )
    
    # Create pipeline
    from qsar_validation import ModelAgnosticQSARPipeline
    
    pipeline = ModelAgnosticQSARPipeline(
        featurizer=custom_featurizer,
        model=model,
        smiles_col='SMILES',
        target_col='Activity'
    )
    
    results = pipeline.fit_predict_validate(df, verbose=True)
    
    return results


# ============================================================================
# EXAMPLE 5: Custom Configuration
# ============================================================================

def example_5_custom_config():
    """
    Customize the validation pipeline configuration
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Custom Configuration")
    print("="*70)
    
    df = pd.read_csv('your_data.csv')
    
    # Simple featurizer
    def morgan_featurizer(smiles_list):
        fingerprints = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
                fingerprints.append(np.array(fp))
            else:
                fingerprints.append(np.zeros(1024))
        return np.array(fingerprints)
    
    # Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Custom configuration
    custom_config = {
        'use_scaffold_split': True,          # Use scaffold-based splitting
        'remove_duplicates': True,            # Remove duplicates before split
        'scale_features': True,               # Scale features
        'detect_activity_cliffs': True,       # Detect activity cliffs
        'run_y_randomization': True,          # Run Y-randomization test
        'n_randomization_runs': 20,           # 20 randomization runs (default: 10)
        'cv_folds': 10,                       # 10-fold CV (default: 5)
        'test_size': 0.15,                    # 15% test set (default: 0.2)
        'val_size': 0.05,                     # 5% validation set (default: 0.1)
        'random_state': 42
    }
    
    # Create pipeline with custom config
    from qsar_validation import ModelAgnosticQSARPipeline
    
    pipeline = ModelAgnosticQSARPipeline(
        featurizer=morgan_featurizer,
        model=model,
        smiles_col='SMILES',
        target_col='Activity',
        validation_config=custom_config
    )
    
    results = pipeline.fit_predict_validate(df, verbose=True)
    
    return results


# ============================================================================
# EXAMPLE 6: Multiple Models Comparison
# ============================================================================

def example_6_model_comparison():
    """
    Compare multiple models with the same featurizer
    """
    print("\n" + "="*70)
    print("EXAMPLE 6: Multiple Models Comparison")
    print("="*70)
    
    df = pd.read_csv('your_data.csv')
    
    # Single featurizer
    def morgan_featurizer(smiles_list):
        fingerprints = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
                fingerprints.append(np.array(fp))
            else:
                fingerprints.append(np.zeros(1024))
        return np.array(fingerprints)
    
    # Define multiple models
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1),
        'SVR': SVR(kernel='rbf'),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    from qsar_validation import ModelAgnosticQSARPipeline
    
    results_comparison = {}
    
    for model_name, model in models.items():
        print(f"\n\n{'='*70}")
        print(f"Testing: {model_name}")
        print('='*70)
        
        pipeline = ModelAgnosticQSARPipeline(
            featurizer=morgan_featurizer,
            model=model,
            smiles_col='SMILES',
            target_col='Activity'
        )
        
        results = pipeline.fit_predict_validate(df, verbose=False)
        
        results_comparison[model_name] = {
            'test_r2': results['performance']['test']['r2'],
            'test_rmse': results['performance']['test']['rmse'],
            'cv_r2': results['cross_validation']['cv_r2_mean']
        }
        
        print(f"Test R²: {results_comparison[model_name]['test_r2']:.3f}")
        print(f"Test RMSE: {results_comparison[model_name]['test_rmse']:.3f}")
    
    # Print comparison
    print("\n\n" + "="*70)
    print("MODEL COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Model':<25} {'Test R²':<12} {'Test RMSE':<12} {'CV R²':<12}")
    print("-"*70)
    
    for model_name, metrics in results_comparison.items():
        print(f"{model_name:<25} {metrics['test_r2']:<12.3f} {metrics['test_rmse']:<12.3f} {metrics['cv_r2']:<12.3f}")
    
    return results_comparison


# ============================================================================
# EXAMPLE 7: Minimal Example (Just 10 Lines!)
# ============================================================================

def example_7_minimal():
    """
    Absolute minimal example - just the essentials!
    """
    import pandas as pd
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from sklearn.ensemble import RandomForestRegressor
    from qsar_validation import ModelAgnosticQSARPipeline
    
    # 1. Load data
    df = pd.read_csv('your_data.csv')
    
    # 2. Define featurizer (any function: SMILES -> features)
    def my_featurizer(smiles_list):
        fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, 1024) for s in smiles_list]
        return np.array([np.array(fp) for fp in fps])
    
    # 3. Choose your model (any sklearn-compatible model)
    my_model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # 4. Create pipeline (handles ALL validation + leakage prevention)
    pipeline = ModelAgnosticQSARPipeline(my_featurizer, my_model, 'SMILES', 'Activity')
    
    # 5. Run everything!
    results = pipeline.fit_predict_validate(df)
    
    # Done! You now have:
    # - Data leakage-free split (scaffold-based)
    # - Trained model
    # - Complete validation metrics
    # - Activity cliff detection
    # - Y-randomization test
    # - All checks passed


# ============================================================================
# RUN EXAMPLES
# ============================================================================

if __name__ == '__main__':
    print("""
    Model-Agnostic QSAR Pipeline Examples
    ======================================
    
    Choose an example to run:
    
    1. Random Forest + Morgan Fingerprints (most common)
    2. Ridge Regression + RDKit Descriptors (interpretable)
    3. XGBoost + MACCS Keys (gradient boosting)
    4. Neural Network + Custom Features (deep learning)
    5. Custom Configuration (customize validation)
    6. Multiple Models Comparison (compare several models)
    7. Minimal Example (just 10 lines!)
    
    """)
    
    choice = input("Enter example number (1-7): ").strip()
    
    examples = {
        '1': example_1_random_forest_morgan,
        '2': example_2_ridge_rdkit_descriptors,
        '3': example_3_xgboost_maccs,
        '4': example_4_neural_network_custom,
        '5': example_5_custom_config,
        '6': example_6_model_comparison,
        '7': example_7_minimal
    }
    
    if choice in examples:
        examples[choice]()
    else:
        print("Invalid choice. Running Example 1 by default...")
        example_1_random_forest_morgan()
