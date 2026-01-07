"""
Feature Engineering Examples (Proper Data Leakage Prevention)
============================================================

Demonstrates CORRECT usage of:
1. Feature Scaling (fit on train only)
2. Feature Selection (nested CV)
3. PCA (train fold only)

All examples show proper nested CV workflows to prevent data leakage.

Author: QSAR Validation Framework
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error

# Import our modules
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from qsar_validation.feature_scaling import FeatureScaler
from qsar_validation.feature_selection import FeatureSelector, NestedCVFeatureSelector
from qsar_validation.pca_module import PCATransformer


def example_1_feature_scaling_in_cv():
    """
    Example 1: Proper Feature Scaling in Cross-Validation
    
    Demonstrates:
    - Fit scaler on train fold only
    - Transform train and val separately
    - No leakage
    """
    print("\n" + "="*80)
    print("Example 1: Feature Scaling in CV (CORRECT)")
    print("="*80)
    
    # Generate synthetic data
    np.random.seed(42)
    X = np.random.randn(200, 50) * 100  # 200 samples, 50 features
    y = np.random.randn(200)
    
    print(f"\nData: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Feature range: [{X.min():.2f}, {X.max():.2f}]")
    
    # Split train/test
    test_size = 40
    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]
    
    print(f"\nTrain: {X_train.shape[0]} samples")
    print(f"Test: {X_test.shape[0]} samples")
    
    # Cross-validation on training set
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    model = Ridge(alpha=1.0)
    
    fold_scores = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"\n--- Fold {fold_idx + 1}/5 ---")
        
        X_train_fold = X_train[train_idx]
        y_train_fold = y_train[train_idx]
        X_val_fold = X_train[val_idx]
        y_val_fold = y_train[val_idx]
        
        # CRITICAL: Fit scaler on THIS fold's training data only
        scaler = FeatureScaler(method='standard')
        scaler.fit(X_train_fold)
        
        # Transform train and val
        X_train_scaled = scaler.transform(X_train_fold)
        X_val_scaled = scaler.transform(X_val_fold)
        
        print(f"Scaled range: [{X_train_scaled.min():.2f}, {X_train_scaled.max():.2f}]")
        
        # Train model
        model.fit(X_train_scaled, y_train_fold)
        y_pred = model.predict(X_val_scaled)
        
        score = r2_score(y_val_fold, y_pred)
        fold_scores.append(score)
        print(f"R² score: {score:.4f}")
    
    print(f"\n✅ CV complete (no leakage)")
    print(f"Mean R²: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
    
    # Final model: fit scaler on ALL training data, then train
    print(f"\n--- Final Model ---")
    final_scaler = FeatureScaler(method='standard')
    final_scaler.fit(X_train)
    
    X_train_scaled = final_scaler.transform(X_train)
    X_test_scaled = final_scaler.transform(X_test)
    
    model.fit(X_train_scaled, y_train)
    y_test_pred = model.predict(X_test_scaled)
    test_score = r2_score(y_test, y_test_pred)
    
    print(f"Test R²: {test_score:.4f}")


def example_2_feature_selection_nested_cv():
    """
    Example 2: Feature Selection in Nested CV
    
    Demonstrates:
    - Select features within each CV fold
    - Nested CV prevents leakage
    - Different features selected per fold
    """
    print("\n" + "="*80)
    print("Example 2: Feature Selection in Nested CV (CORRECT)")
    print("="*80)
    
    # Generate synthetic data with some informative features
    np.random.seed(42)
    n_samples = 200
    n_features = 100
    
    # First 10 features are informative
    X_informative = np.random.randn(n_samples, 10)
    y = np.sum(X_informative, axis=1) + np.random.randn(n_samples) * 0.1
    
    # Add 90 noise features
    X_noise = np.random.randn(n_samples, n_features - 10)
    X = np.hstack([X_informative, X_noise])
    
    print(f"\nData: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"(First 10 features are informative, rest are noise)")
    
    # Split train/test
    test_size = 40
    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]
    
    print(f"\nTrain: {X_train.shape[0]} samples")
    print(f"Test: {X_test.shape[0]} samples")
    
    # Nested CV with feature selection
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    model = Ridge(alpha=1.0)
    
    fold_scores = []
    selected_features_per_fold = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"\n--- Fold {fold_idx + 1}/5 ---")
        
        X_train_fold = X_train[train_idx]
        y_train_fold = y_train[train_idx]
        X_val_fold = X_train[val_idx]
        y_val_fold = y_train[val_idx]
        
        # CRITICAL: Select features on THIS fold's training data only
        selector = FeatureSelector(method='univariate', n_features=20)
        selector.fit(X_train_fold, y_train_fold)
        
        # Get selected features
        selected = selector.get_selected_feature_indices()
        selected_features_per_fold.append(selected)
        
        # Check how many informative features were selected
        n_informative_selected = np.sum(selected < 10)
        print(f"Selected: {len(selected)} features ({n_informative_selected}/10 informative)")
        
        # Transform train and val
        X_train_selected = selector.transform(X_train_fold)
        X_val_selected = selector.transform(X_val_fold)
        
        # Train model
        model.fit(X_train_selected, y_train_fold)
        y_pred = model.predict(X_val_selected)
        
        score = r2_score(y_val_fold, y_pred)
        fold_scores.append(score)
        print(f"R² score: {score:.4f}")
    
    print(f"\n✅ Nested CV complete (no leakage)")
    print(f"Mean R²: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
    
    # Analyze feature stability across folds
    print(f"\n--- Feature Selection Stability ---")
    all_selected = set()
    for features in selected_features_per_fold:
        all_selected.update(features)
    
    print(f"Total unique features selected: {len(all_selected)}")
    print(f"Features selected in all folds: {len(set.intersection(*[set(f) for f in selected_features_per_fold]))}")


def example_3_pca_in_cv():
    """
    Example 3: PCA in Cross-Validation
    
    Demonstrates:
    - Fit PCA on train fold only
    - Transform train and val separately
    - Different PCA per fold
    """
    print("\n" + "="*80)
    print("Example 3: PCA in CV (CORRECT)")
    print("="*80)
    
    # Generate high-dimensional data
    np.random.seed(42)
    X = np.random.randn(200, 100)  # 200 samples, 100 features
    y = np.random.randn(200)
    
    print(f"\nData: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Split train/test
    test_size = 40
    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]
    
    # Cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    model = Ridge(alpha=1.0)
    
    fold_scores = []
    components_per_fold = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"\n--- Fold {fold_idx + 1}/5 ---")
        
        X_train_fold = X_train[train_idx]
        y_train_fold = y_train[train_idx]
        X_val_fold = X_train[val_idx]
        y_val_fold = y_train[val_idx]
        
        # CRITICAL: Fit PCA on THIS fold's training data only
        pca = PCATransformer(n_components=0.95)  # Keep 95% variance
        pca.fit(X_train_fold)
        
        components_per_fold.append(pca.n_components_out)
        
        print(f"PCA components: {pca.n_components_out}")
        print(f"Variance explained: {pca.get_cumulative_variance()[-1]*100:.1f}%")
        
        # Transform train and val
        X_train_pca = pca.transform(X_train_fold)
        X_val_pca = pca.transform(X_val_fold)
        
        # Train model
        model.fit(X_train_pca, y_train_fold)
        y_pred = model.predict(X_val_pca)
        
        score = r2_score(y_val_fold, y_pred)
        fold_scores.append(score)
        print(f"R² score: {score:.4f}")
    
    print(f"\n✅ CV complete (no leakage)")
    print(f"Mean R²: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
    print(f"Components per fold: {components_per_fold}")
    print(f"Mean components: {np.mean(components_per_fold):.1f} ± {np.std(components_per_fold):.1f}")


def example_4_complete_pipeline():
    """
    Example 4: Complete Feature Engineering Pipeline
    
    Combines:
    1. Feature scaling
    2. Feature selection
    3. PCA
    
    All within nested CV.
    """
    print("\n" + "="*80)
    print("Example 4: Complete Pipeline (Scaling + Selection + PCA)")
    print("="*80)
    
    # Generate data
    np.random.seed(42)
    X = np.random.randn(200, 100) * 100
    y = np.random.randn(200)
    
    print(f"\nData: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Split train/test
    test_size = 40
    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]
    
    # Cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
    
    fold_scores = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"\n--- Fold {fold_idx + 1}/5 ---")
        
        X_train_fold = X_train[train_idx]
        y_train_fold = y_train[train_idx]
        X_val_fold = X_train[val_idx]
        y_val_fold = y_train[val_idx]
        
        print(f"Starting features: {X_train_fold.shape[1]}")
        
        # Step 1: Scale features
        scaler = FeatureScaler(method='standard')
        scaler.fit(X_train_fold)
        X_train_scaled = scaler.transform(X_train_fold)
        X_val_scaled = scaler.transform(X_val_fold)
        
        # Step 2: Select features
        selector = FeatureSelector(method='variance', n_features=50)
        selector.fit(X_train_scaled)
        X_train_selected = selector.transform(X_train_scaled)
        X_val_selected = selector.transform(X_val_scaled)
        
        print(f"After selection: {X_train_selected.shape[1]} features")
        
        # Step 3: Apply PCA
        pca = PCATransformer(n_components=0.95)
        pca.fit(X_train_selected)
        X_train_pca = pca.transform(X_train_selected)
        X_val_pca = pca.transform(X_val_selected)
        
        print(f"After PCA: {X_train_pca.shape[1]} components")
        print(f"Final reduction: {X_train_fold.shape[1]} → {X_train_pca.shape[1]} ({X_train_pca.shape[1]/X_train_fold.shape[1]*100:.1f}%)")
        
        # Train model
        model.fit(X_train_pca, y_train_fold)
        y_pred = model.predict(X_val_pca)
        
        score = r2_score(y_val_fold, y_pred)
        fold_scores.append(score)
        print(f"R² score: {score:.4f}")
    
    print(f"\n✅ Complete pipeline (no leakage)")
    print(f"Mean R²: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")


def example_5_compare_methods():
    """
    Example 5: Compare Different Feature Engineering Approaches
    
    Compares:
    - No feature engineering
    - Scaling only
    - Scaling + selection
    - Scaling + PCA
    - Scaling + selection + PCA
    """
    print("\n" + "="*80)
    print("Example 5: Compare Feature Engineering Methods")
    print("="*80)
    
    # Generate data
    np.random.seed(42)
    n_samples = 200
    n_features = 100
    
    # Create data with some structure
    X_informative = np.random.randn(n_samples, 20)
    y = np.sum(X_informative[:, :10], axis=1) + np.random.randn(n_samples) * 0.5
    X_noise = np.random.randn(n_samples, n_features - 20) * 0.1
    X = np.hstack([X_informative, X_noise])
    
    print(f"\nData: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Split train/test
    test_size = 40
    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]
    
    # Compare methods
    methods = {
        "No engineering": [],
        "Scaling only": ['scale'],
        "Scale + Select": ['scale', 'select'],
        "Scale + PCA": ['scale', 'pca'],
        "Scale + Select + PCA": ['scale', 'select', 'pca']
    }
    
    model = Ridge(alpha=1.0)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    results = {}
    
    for method_name, steps in methods.items():
        print(f"\n--- {method_name} ---")
        
        fold_scores = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            X_train_fold = X_train[train_idx]
            y_train_fold = y_train[train_idx]
            X_val_fold = X_train[val_idx]
            y_val_fold = y_train[val_idx]
            
            X_train_proc = X_train_fold.copy()
            X_val_proc = X_val_fold.copy()
            
            # Apply processing steps
            if 'scale' in steps:
                scaler = FeatureScaler(method='standard')
                scaler.fit(X_train_proc)
                X_train_proc = scaler.transform(X_train_proc)
                X_val_proc = scaler.transform(X_val_proc)
            
            if 'select' in steps:
                selector = FeatureSelector(method='univariate', n_features=30)
                selector.fit(X_train_proc, y_train_fold)
                X_train_proc = selector.transform(X_train_proc)
                X_val_proc = selector.transform(X_val_proc)
            
            if 'pca' in steps:
                pca = PCATransformer(n_components=0.95)
                pca.fit(X_train_proc)
                X_train_proc = pca.transform(X_train_proc)
                X_val_proc = pca.transform(X_val_proc)
            
            # Train and evaluate
            model.fit(X_train_proc, y_train_fold)
            y_pred = model.predict(X_val_proc)
            score = r2_score(y_val_fold, y_pred)
            fold_scores.append(score)
        
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        results[method_name] = (mean_score, std_score)
        
        print(f"  R²: {mean_score:.4f} ± {std_score:.4f}")
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Method':<30} {'Mean R²':<15} {'Std R²'}")
    print("-" * 80)
    for method_name, (mean, std) in sorted(results.items(), key=lambda x: x[1][0], reverse=True):
        print(f"{method_name:<30} {mean:>6.4f}         ± {std:.4f}")


if __name__ == '__main__':
    print("\n" + "="*80)
    print("Feature Engineering Examples")
    print("All examples demonstrate PROPER data leakage prevention")
    print("="*80)
    
    example_1_feature_scaling_in_cv()
    print("\n")
    
    example_2_feature_selection_nested_cv()
    print("\n")
    
    example_3_pca_in_cv()
    print("\n")
    
    example_4_complete_pipeline()
    print("\n")
    
    example_5_compare_methods()
    print("\n")
    
    print("="*80)
    print("✅ All examples complete!")
    print("="*80)
