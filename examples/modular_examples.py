"""
Modular Usage Examples
======================

This file demonstrates how to use individual modules independently,
rather than the full ModelAgnosticQSARPipeline.

Each example shows:
1. How to import and use a specific module
2. What inputs it needs
3. What outputs it provides
4. How to combine with other modules or your own code

Examples:
---------
1. Standalone Duplicate Removal
2. Standalone Scaffold Splitting
3. Standalone Feature Scaling
4. Standalone Cross-Validation
5. Standalone Performance Metrics
6. Standalone Bias Analysis
7. Standalone Complexity Analysis
8. Custom Pipeline (Mix & Match)
9. Minimal Leakage Prevention (3 modules only)
10. Advanced Custom Workflow
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

# ============================================================================
# Example 1: Standalone Duplicate Removal
# ============================================================================

def example_1_duplicate_removal():
    """Use DuplicateRemoval module independently."""
    print("\n" + "="*80)
    print("Example 1: Standalone Duplicate Removal")
    print("="*80)
    
    from qsar_validation.duplicate_removal import DuplicateRemoval
    
    # Create sample data with duplicates
    df = pd.DataFrame({
        'SMILES': ['CCO', 'CCO', 'CC(C)O', 'CCO', 'CCCC'],
        'Activity': [5.0, 5.1, 6.0, 4.9, 3.0],
        'Name': ['ethanol_1', 'ethanol_2', 'isopropanol', 'ethanol_3', 'butane']
    })
    
    print(f"\nOriginal data: {len(df)} compounds")
    print(df)
    
    # Initialize module
    remover = DuplicateRemoval(smiles_col='SMILES')
    
    # Check for duplicates
    has_duplicates = remover.check_duplicates(df)
    print(f"\nHas duplicates: {has_duplicates}")
    
    # Strategy 1: Keep first occurrence
    clean_df_first = remover.remove_duplicates(df, strategy='first')
    print(f"\nStrategy 'first': {len(clean_df_first)} compounds")
    print(clean_df_first)
    
    # Strategy 2: Average duplicate activities
    clean_df_avg = remover.remove_duplicates(df, strategy='average')
    print(f"\nStrategy 'average': {len(clean_df_avg)} compounds")
    print(clean_df_avg)
    
    print("\n✅ Module used: DuplicateRemoval")
    print("✅ Purpose: Remove duplicates BEFORE any splitting (prevents leakage)")


# ============================================================================
# Example 2: Standalone Scaffold Splitting
# ============================================================================

def example_2_scaffold_splitting():
    """Use ScaffoldSplitter module independently."""
    print("\n" + "="*80)
    print("Example 2: Standalone Scaffold Splitting")
    print("="*80)
    
    from qsar_validation.scaffold_splitting import ScaffoldSplitter
    
    # Create sample data with different scaffolds
    df = pd.DataFrame({
        'SMILES': [
            'c1ccccc1CC',    # benzene scaffold
            'c1ccccc1CCC',   # benzene scaffold
            'c1ccccc1CCCC',  # benzene scaffold
            'CCCCCC',        # aliphatic
            'CCCCCCC',       # aliphatic
            'CCCCCCCC',      # aliphatic
            'c1ccncc1CC',    # pyridine scaffold
            'c1ccncc1CCC',   # pyridine scaffold
        ],
        'Activity': [5.0, 5.5, 6.0, 3.0, 3.2, 3.5, 7.0, 7.2]
    })
    
    print(f"\nData: {len(df)} compounds")
    
    # Initialize module
    splitter = ScaffoldSplitter(smiles_col='SMILES')
    
    # Get scaffold for each compound
    print("\nScaffolds:")
    for idx, smiles in enumerate(df['SMILES']):
        scaffold = splitter.get_scaffold(smiles)
        print(f"  {smiles:20s} -> {scaffold}")
    
    # Split data
    train_idx, val_idx, test_idx = splitter.split(
        df, 
        test_size=0.25,
        val_size=0.25,
        random_state=42
    )
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_idx)} compounds")
    print(f"  Val:   {len(val_idx)} compounds")
    print(f"  Test:  {len(test_idx)} compounds")
    
    # Verify no scaffold overlap
    overlap = splitter.check_scaffold_overlap(train_idx, test_idx, df)
    print(f"\nScaffold overlap between train and test: {overlap}")
    
    if overlap == 0:
        print("✅ NO DATA LEAKAGE: Zero scaffold overlap!")
    else:
        print(f"⚠️  WARNING: {overlap} compounds have overlapping scaffolds!")
    
    # Get the split datasets
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    test_df = df.iloc[test_idx]
    
    print("\n✅ Module used: ScaffoldSplitter")
    print("✅ Purpose: Split by scaffolds (prevents leakage)")


# ============================================================================
# Example 3: Standalone Feature Scaling
# ============================================================================

def example_3_feature_scaling():
    """Use FeatureScaler module independently."""
    print("\n" + "="*80)
    print("Example 3: Standalone Feature Scaling")
    print("="*80)
    
    from qsar_validation.feature_scaling import FeatureScaler
    
    # Create sample features (e.g., from fingerprints)
    np.random.seed(42)
    train_features = np.random.rand(100, 50)  # 100 samples, 50 features
    test_features = np.random.rand(30, 50)    # 30 samples, 50 features
    
    print(f"\nOriginal features:")
    print(f"  Train shape: {train_features.shape}")
    print(f"  Train mean: {train_features.mean():.3f}, std: {train_features.std():.3f}")
    print(f"  Test shape: {test_features.shape}")
    print(f"  Test mean: {test_features.mean():.3f}, std: {test_features.std():.3f}")
    
    # Method 1: Standard scaling (mean=0, std=1)
    print("\n--- Standard Scaling ---")
    scaler_std = FeatureScaler(method='standard')
    scaler_std.fit(train_features)  # Fit on train ONLY!
    
    train_scaled = scaler_std.transform(train_features)
    test_scaled = scaler_std.transform(test_features)
    
    print(f"  Train scaled: mean={train_scaled.mean():.3f}, std={train_scaled.std():.3f}")
    print(f"  Test scaled: mean={test_scaled.mean():.3f}, std={test_scaled.std():.3f}")
    
    # Method 2: MinMax scaling (range [0, 1])
    print("\n--- MinMax Scaling ---")
    scaler_mm = FeatureScaler(method='minmax')
    scaler_mm.fit(train_features)
    
    train_scaled = scaler_mm.transform(train_features)
    test_scaled = scaler_mm.transform(test_features)
    
    print(f"  Train scaled: min={train_scaled.min():.3f}, max={train_scaled.max():.3f}")
    print(f"  Test scaled: min={test_scaled.min():.3f}, max={test_scaled.max():.3f}")
    
    # Method 3: Robust scaling (uses median and IQR)
    print("\n--- Robust Scaling ---")
    scaler_rob = FeatureScaler(method='robust')
    scaler_rob.fit(train_features)
    
    train_scaled = scaler_rob.transform(train_features)
    test_scaled = scaler_rob.transform(test_features)
    
    print(f"  Train scaled: median={np.median(train_scaled):.3f}")
    print(f"  Test scaled: median={np.median(test_scaled):.3f}")
    
    print("\n✅ Module used: FeatureScaler")
    print("✅ Purpose: Scale features using TRAIN statistics only (prevents leakage)")


# ============================================================================
# Example 4: Standalone Cross-Validation
# ============================================================================

def example_4_cross_validation():
    """Use CrossValidator module independently."""
    print("\n" + "="*80)
    print("Example 4: Standalone Cross-Validation")
    print("="*80)
    
    from qsar_validation.cross_validation import CrossValidator
    
    # Create sample data
    np.random.seed(42)
    X = np.random.rand(100, 50)
    y = np.random.rand(100) * 10
    
    print(f"\nData: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Initialize module
    cv = CrossValidator(n_folds=5, random_state=42)
    
    # Test with different models
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42),
        'Ridge': Ridge(alpha=1.0)
    }
    
    for model_name, model in models.items():
        print(f"\n--- {model_name} ---")
        
        # Perform cross-validation
        cv_scores = cv.cross_validate(model, X, y, scoring='r2')
        
        print(f"  Fold scores: {[f'{s:.3f}' for s in cv_scores]}")
        print(f"  Mean: {np.mean(cv_scores):.3f}")
        print(f"  Std:  {np.std(cv_scores):.3f}")
        print(f"  CV Result: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
    
    # Show fold structure
    print("\n--- Fold Structure ---")
    for fold_idx, (train_idx, val_idx) in enumerate(cv.get_folds(X)):
        print(f"  Fold {fold_idx}: train={len(train_idx)}, val={len(val_idx)}")
    
    print("\n✅ Module used: CrossValidator")
    print("✅ Purpose: Perform k-fold CV for model evaluation")


# ============================================================================
# Example 5: Standalone Performance Metrics
# ============================================================================

def example_5_performance_metrics():
    """Use PerformanceMetrics module independently."""
    print("\n" + "="*80)
    print("Example 5: Standalone Performance Metrics")
    print("="*80)
    
    from qsar_validation.performance_metrics import PerformanceMetrics
    
    # Create sample predictions
    np.random.seed(42)
    y_true = np.random.rand(50) * 10
    y_pred = y_true + np.random.randn(50) * 0.5  # Add noise
    
    print(f"\nData: {len(y_true)} predictions")
    print(f"  True values range: [{y_true.min():.2f}, {y_true.max():.2f}]")
    print(f"  Predicted values range: [{y_pred.min():.2f}, {y_pred.max():.2f}]")
    
    # Initialize module
    metrics_calc = PerformanceMetrics()
    
    # Calculate all metrics at once
    print("\n--- All Metrics ---")
    metrics = metrics_calc.calculate_all_metrics(y_true, y_pred, set_name='Test')
    
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")
    
    # Calculate individual metrics
    print("\n--- Individual Metrics ---")
    r2 = metrics_calc.calculate_r2(y_true, y_pred)
    rmse = metrics_calc.calculate_rmse(y_true, y_pred)
    mae = metrics_calc.calculate_mae(y_true, y_pred)
    pearson_r = metrics_calc.calculate_pearson_r(y_true, y_pred)
    spearman_rho = metrics_calc.calculate_spearman_rho(y_true, y_pred)
    
    print(f"  R² Score: {r2:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  Pearson r: {pearson_r:.4f}")
    print(f"  Spearman ρ: {spearman_rho:.4f}")
    
    print("\n✅ Module used: PerformanceMetrics")
    print("✅ Purpose: Calculate comprehensive metrics for model evaluation")


# ============================================================================
# Example 6: Standalone Dataset Bias Analysis
# ============================================================================

def example_6_bias_analysis():
    """Use DatasetBiasAnalysis module independently."""
    print("\n" + "="*80)
    print("Example 6: Standalone Dataset Bias Analysis")
    print("="*80)
    
    from qsar_validation.dataset_bias_analysis import DatasetBiasAnalysis
    
    # Create sample data
    np.random.seed(42)
    X_train = np.random.rand(100, 50)
    X_test = np.random.rand(30, 50)
    y_train = np.random.rand(100) * 8 + 2  # Range [2, 10]
    y_test = np.random.rand(30) * 6 + 3     # Range [3, 9]
    
    print(f"\nTrain: {X_train.shape[0]} samples")
    print(f"Test:  {X_test.shape[0]} samples")
    print(f"Train activity range: [{y_train.min():.2f}, {y_train.max():.2f}]")
    print(f"Test activity range:  [{y_test.min():.2f}, {y_test.max():.2f}]")
    
    # Initialize module
    bias_analyzer = DatasetBiasAnalysis()
    
    # Analyze bias
    print("\n--- Bias Analysis ---")
    bias_report = bias_analyzer.analyze_bias(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test
    )
    
    print(f"\nBias Report:")
    for key, value in bias_report.items():
        if key != 'warnings':
            print(f"  {key}: {value}")
    
    if bias_report['warnings']:
        print(f"\n⚠️  Warnings:")
        for warning in bias_report['warnings']:
            print(f"  - {warning}")
    else:
        print(f"\n✅ No bias warnings detected")
    
    # Check specific aspects
    print("\n--- Specific Checks ---")
    coverage = bias_analyzer.check_activity_coverage(y_train, y_test)
    print(f"  Test set activity coverage: {coverage:.1%}")
    
    print("\n✅ Module used: DatasetBiasAnalysis")
    print("✅ Purpose: Detect dataset bias issues that affect generalization")


# ============================================================================
# Example 7: Standalone Model Complexity Analysis
# ============================================================================

def example_7_complexity_analysis():
    """Use ModelComplexityAnalysis module independently."""
    print("\n" + "="*80)
    print("Example 7: Standalone Model Complexity Analysis")
    print("="*80)
    
    from qsar_validation.model_complexity_analysis import ModelComplexityAnalysis
    
    # Create sample data with LOW samples-to-features ratio (overfitting risk!)
    np.random.seed(42)
    X = np.random.rand(30, 100)  # 30 samples, 100 features (ratio = 0.3!)
    y = np.random.rand(30) * 10
    
    print(f"\nData: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Samples-to-features ratio: {X.shape[0] / X.shape[1]:.2f}")
    
    # Train a complex model
    model = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42)
    model.fit(X, y)
    
    print(f"\nModel: {model.__class__.__name__}")
    print(f"  n_estimators: {model.n_estimators}")
    print(f"  max_depth: {model.max_depth}")
    
    # Initialize module
    complexity_analyzer = ModelComplexityAnalysis()
    
    # Analyze complexity
    print("\n--- Complexity Analysis ---")
    complexity_report = complexity_analyzer.analyze_complexity(
        model=model,
        X=X,
        y=y
    )
    
    print(f"\nComplexity Report:")
    for key, value in complexity_report.items():
        if key != 'warnings':
            print(f"  {key}: {value}")
    
    if complexity_report['warnings']:
        print(f"\n⚠️  Complexity Warnings:")
        for warning in complexity_report['warnings']:
            print(f"  - {warning}")
    else:
        print(f"\n✅ No complexity warnings")
    
    print("\n✅ Module used: ModelComplexityAnalysis")
    print("✅ Purpose: Detect overfitting risk from model/data complexity")


# ============================================================================
# Example 8: Custom Pipeline (Mix & Match Modules)
# ============================================================================

def example_8_custom_pipeline():
    """Build a custom pipeline using multiple modules."""
    print("\n" + "="*80)
    print("Example 8: Custom Pipeline (Mix & Match)")
    print("="*80)
    
    from qsar_validation.duplicate_removal import DuplicateRemoval
    from qsar_validation.scaffold_splitting import ScaffoldSplitter
    from qsar_validation.feature_scaling import FeatureScaler
    from qsar_validation.performance_metrics import PerformanceMetrics
    
    # Create sample data
    df = pd.DataFrame({
        'SMILES': [
            'CCO', 'CCO',  # Duplicates
            'c1ccccc1CC', 'c1ccccc1CCC', 'c1ccccc1CCCC',  # Benzene
            'CCCCCC', 'CCCCCCC',  # Aliphatic
            'c1ccncc1CC', 'c1ccncc1CCC',  # Pyridine
        ],
        'Activity': [5.0, 5.1, 6.0, 6.5, 7.0, 3.0, 3.5, 8.0, 8.5]
    })
    
    print(f"\n📊 Initial data: {len(df)} compounds")
    
    # Step 1: Remove duplicates
    print("\n🔧 Step 1: Remove duplicates")
    remover = DuplicateRemoval(smiles_col='SMILES')
    df = remover.remove_duplicates(df, strategy='average')
    print(f"  After duplicate removal: {len(df)} compounds")
    
    # Step 2: Scaffold split
    print("\n🔧 Step 2: Scaffold-based splitting")
    splitter = ScaffoldSplitter(smiles_col='SMILES')
    train_idx, val_idx, test_idx = splitter.split(df, test_size=0.25, val_size=0.25)
    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    # Verify no leakage
    overlap = splitter.check_scaffold_overlap(train_idx, test_idx, df)
    print(f"  Scaffold overlap: {overlap} ✅")
    
    # Step 3: Featurization (simple example)
    print("\n🔧 Step 3: Feature generation")
    
    def simple_featurizer(smiles_list):
        """Simple featurizer for demo."""
        features = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=256)
                features.append(np.array(fp))
            else:
                features.append(np.zeros(256))
        return np.array(features)
    
    X_train = simple_featurizer(df.iloc[train_idx]['SMILES'].tolist())
    X_test = simple_featurizer(df.iloc[test_idx]['SMILES'].tolist())
    y_train = df.iloc[train_idx]['Activity'].values
    y_test = df.iloc[test_idx]['Activity'].values
    
    print(f"  Train features: {X_train.shape}")
    print(f"  Test features: {X_test.shape}")
    
    # Step 4: Feature scaling
    print("\n🔧 Step 4: Feature scaling")
    scaler = FeatureScaler(method='standard')
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"  Features scaled (train mean: {X_train_scaled.mean():.3f})")
    
    # Step 5: Model training (your own model)
    print("\n🔧 Step 5: Model training")
    model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
    model.fit(X_train_scaled, y_train)
    print(f"  Model trained: {model.__class__.__name__}")
    
    # Step 6: Predictions
    print("\n🔧 Step 6: Predictions")
    y_pred = model.predict(X_test_scaled)
    print(f"  Predictions generated: {len(y_pred)} samples")
    
    # Step 7: Performance metrics
    print("\n🔧 Step 7: Performance evaluation")
    metrics_calc = PerformanceMetrics()
    metrics = metrics_calc.calculate_all_metrics(y_test, y_pred, set_name='Test')
    
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")
    
    print("\n✅ Custom pipeline complete!")
    print("✅ Modules used: DuplicateRemoval, ScaffoldSplitter, FeatureScaler, PerformanceMetrics")


# ============================================================================
# Example 9: Minimal Leakage Prevention (3 Modules Only)
# ============================================================================

def example_9_minimal_leakage_prevention():
    """Use only the essential leakage prevention modules."""
    print("\n" + "="*80)
    print("Example 9: Minimal Leakage Prevention (Essential Modules Only)")
    print("="*80)
    
    from qsar_validation.duplicate_removal import DuplicateRemoval
    from qsar_validation.scaffold_splitting import ScaffoldSplitter
    from qsar_validation.feature_scaling import FeatureScaler
    
    # Sample data
    df = pd.DataFrame({
        'SMILES': ['CCO', 'CCO', 'c1ccccc1', 'c1ccccc1C', 'CCCC', 'CCCCC'],
        'Activity': [5.0, 5.0, 7.0, 7.5, 3.0, 3.5]
    })
    
    print(f"\n📊 Data: {len(df)} compounds")
    
    # Three essential steps for leakage prevention:
    
    # 1. Remove duplicates BEFORE splitting
    print("\n🛡️  Step 1: Remove duplicates (Leakage Prevention)")
    remover = DuplicateRemoval(smiles_col='SMILES')
    df_clean = remover.remove_duplicates(df, strategy='average')
    print(f"  {len(df)} → {len(df_clean)} compounds")
    
    # 2. Scaffold-based split
    print("\n🛡️  Step 2: Scaffold split (Leakage Prevention)")
    splitter = ScaffoldSplitter(smiles_col='SMILES')
    train_idx, _, test_idx = splitter.split(df_clean, test_size=0.3)
    overlap = splitter.check_scaffold_overlap(train_idx, test_idx, df_clean)
    print(f"  Train: {len(train_idx)}, Test: {len(test_idx)}")
    print(f"  Scaffold overlap: {overlap} {'✅' if overlap == 0 else '⚠️'}")
    
    # 3. Scale using train statistics only
    print("\n🛡️  Step 3: Proper feature scaling (Leakage Prevention)")
    
    # Dummy features for demo
    X_train = np.random.rand(len(train_idx), 100)
    X_test = np.random.rand(len(test_idx), 100)
    
    scaler = FeatureScaler(method='standard')
    scaler.fit(X_train)  # Fit on train only!
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"  Fitted on train, transformed both sets")
    
    print("\n✅ Leakage prevention complete with just 3 modules!")
    print("✅ Now you can do your own modeling...")


# ============================================================================
# Example 10: Advanced Custom Workflow
# ============================================================================

def example_10_advanced_workflow():
    """Advanced example with custom requirements."""
    print("\n" + "="*80)
    print("Example 10: Advanced Custom Workflow")
    print("="*80)
    
    from qsar_validation.duplicate_removal import DuplicateRemoval
    from qsar_validation.scaffold_splitting import ScaffoldSplitter
    from qsar_validation.cross_validation import CrossValidator
    from qsar_validation.performance_metrics import PerformanceMetrics
    from qsar_validation.dataset_bias_analysis import DatasetBiasAnalysis
    
    # Simulate a real workflow
    print("\n📋 Scenario: Custom workflow with specific requirements")
    print("  - Remove duplicates (average activities)")
    print("  - Scaffold split (80/20)")
    print("  - NO feature scaling (binary fingerprints)")
    print("  - 3-fold CV on training set")
    print("  - Comprehensive metrics")
    print("  - Bias analysis")
    
    # Generate sample data
    np.random.seed(42)
    smiles_list = ['CCO', 'CCO', 'CCC', 'CCCC', 'c1ccccc1', 'c1ccccc1C', 
                   'c1ccccc1CC', 'CCCCC', 'CCCCCC', 'c1ccncc1']
    activities = np.random.rand(len(smiles_list)) * 10
    
    df = pd.DataFrame({'SMILES': smiles_list, 'Activity': activities})
    
    print(f"\n📊 Initial data: {len(df)} compounds")
    
    # Workflow execution
    print("\n" + "-"*80)
    
    # 1. Duplicates
    print("\n[1/6] Removing duplicates...")
    remover = DuplicateRemoval(smiles_col='SMILES')
    df = remover.remove_duplicates(df, strategy='average')
    print(f"      Result: {len(df)} unique compounds")
    
    # 2. Split
    print("\n[2/6] Scaffold-based splitting (80/20)...")
    splitter = ScaffoldSplitter(smiles_col='SMILES')
    train_idx, _, test_idx = splitter.split(df, test_size=0.2, val_size=0.0)
    print(f"      Result: {len(train_idx)} train, {len(test_idx)} test")
    
    # 3. Features (binary fingerprints - no scaling needed)
    print("\n[3/6] Generating binary fingerprints (no scaling needed)...")
    
    def get_fingerprints(smiles_list):
        fps = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=512)
                fps.append(np.array(fp))
            else:
                fps.append(np.zeros(512))
        return np.array(fps)
    
    X_train = get_fingerprints(df.iloc[train_idx]['SMILES'].tolist())
    X_test = get_fingerprints(df.iloc[test_idx]['SMILES'].tolist())
    y_train = df.iloc[train_idx]['Activity'].values
    y_test = df.iloc[test_idx]['Activity'].values
    print(f"      Result: {X_train.shape[1]} features per compound")
    
    # 4. Cross-validation
    print("\n[4/6] Performing 3-fold cross-validation on training set...")
    cv = CrossValidator(n_folds=3, random_state=42)
    model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
    cv_scores = cv.cross_validate(model, X_train, y_train, scoring='r2')
    print(f"      Result: CV R² = {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
    
    # 5. Train and evaluate
    print("\n[5/6] Training model and evaluating on test set...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    metrics_calc = PerformanceMetrics()
    metrics = metrics_calc.calculate_all_metrics(y_test, y_pred, set_name='Test')
    print(f"      Result: Test R² = {metrics['Test_R2']:.3f}, RMSE = {metrics['Test_RMSE']:.3f}")
    
    # 6. Bias analysis
    print("\n[6/6] Analyzing dataset bias...")
    bias_analyzer = DatasetBiasAnalysis()
    bias_report = bias_analyzer.analyze_bias(X_train, X_test, y_train, y_test)
    
    if bias_report['warnings']:
        print(f"      Warnings: {len(bias_report['warnings'])}")
        for warning in bias_report['warnings'][:2]:  # Show first 2
            print(f"        - {warning}")
    else:
        print(f"      Result: No bias detected ✅")
    
    print("\n" + "-"*80)
    print("\n✅ Advanced workflow complete!")
    print("✅ Modules used: 5 different modules combined as needed")


# ============================================================================
# MAIN: Run All Examples
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("MODULAR USAGE EXAMPLES")
    print("Demonstrating individual module usage")
    print("="*80)
    
    examples = [
        ("1", "Standalone Duplicate Removal", example_1_duplicate_removal),
        ("2", "Standalone Scaffold Splitting", example_2_scaffold_splitting),
        ("3", "Standalone Feature Scaling", example_3_feature_scaling),
        ("4", "Standalone Cross-Validation", example_4_cross_validation),
        ("5", "Standalone Performance Metrics", example_5_performance_metrics),
        ("6", "Standalone Bias Analysis", example_6_bias_analysis),
        ("7", "Standalone Complexity Analysis", example_7_complexity_analysis),
        ("8", "Custom Pipeline (Mix & Match)", example_8_custom_pipeline),
        ("9", "Minimal Leakage Prevention", example_9_minimal_leakage_prevention),
        ("10", "Advanced Custom Workflow", example_10_advanced_workflow),
    ]
    
    print("\nAvailable examples:")
    for num, title, _ in examples:
        print(f"  {num}. {title}")
    
    print("\n" + "="*80)
    choice = input("\nEnter example number to run (or 'all' for all examples): ").strip()
    
    if choice.lower() == 'all':
        for num, title, func in examples:
            func()
    elif choice in [ex[0] for ex in examples]:
        for num, title, func in examples:
            if num == choice:
                func()
                break
    else:
        print("Invalid choice. Running all examples...")
        for num, title, func in examples:
            func()
    
    print("\n" + "="*80)
    print("EXAMPLES COMPLETE")
    print("="*80)
    print("\n💡 Key Takeaway: Each module is independent!")
    print("   Pick the ones you need, ignore the rest.")
    print("   Full control over your validation workflow! 🚀")
    print("="*80 + "\n")
