# Using the QSAR Validation Framework

## A Notebook-Free Guide for Any Low-Data QSAR Project

This guide shows how to use the QSAR validation framework as a standalone Python package on any dataset, without requiring Jupyter notebooks.

## Table of Contents
1. [Installation](#installation)
2. [Basic Usage](#basic-usage)
3. [Complete Workflow Example](#complete-workflow-example)
4. [Advanced Usage](#advanced-usage)
5. [Integration with Your Pipeline](#integration-with-your-pipeline)
6. [Low-Data Regime Best Practices](#low-data-regime-best-practices)

---

## Installation

### Option 1: Install from GitHub (Recommended)
```bash
pip install git+https://github.com/bhatnira/Roy-QSAR-Generative-dev.git
```

### Option 2: Local Installation
```bash
# Clone the repository
git clone https://github.com/bhatnira/Roy-QSAR-Generative-dev.git
cd Roy-QSAR-Generative-dev

# Install in development mode
pip install -e .
```

### Option 3: Install Dependencies Only
```bash
pip install -r requirements.txt
```

---

## Basic Usage

### 1. Quick Validation (Minimal Code)

```python
import pandas as pd
from qsar_validation import run_comprehensive_validation

# Load your dataset
df = pd.read_csv('your_data.csv')

# Run all validation checks with one function
results = run_comprehensive_validation(
    df,
    smiles_col='SMILES',      # Your SMILES column name
    target_col='Activity'      # Your target column name
)

# Results contain all validation metrics
print(f"Number of unique scaffolds: {results['scaffold_diversity']['n_scaffolds']}")
print(f"Activity cliffs detected: {len(results['activity_cliffs'])}")
print(f"Experimental error estimate: {results['experimental_error']['experimental_error']}")
```

### 2. Step-by-Step Validation

```python
from qsar_validation import (
    DatasetBiasAnalyzer,
    ActivityCliffDetector,
    ModelComplexityAnalyzer,
    AssayNoiseEstimator
)

# Step 1: Analyze dataset bias
analyzer = DatasetBiasAnalyzer(smiles_col='SMILES', target_col='Activity')
diversity = analyzer.analyze_scaffold_diversity(df)
distribution = analyzer.analyze_activity_distribution(df)

# Step 2: Detect activity cliffs
cliff_detector = ActivityCliffDetector(smiles_col='SMILES', target_col='Activity')
cliffs = cliff_detector.detect_activity_cliffs(df, 
                                                similarity_threshold=0.85,
                                                activity_threshold=2.0)

# Step 3: Check model complexity appropriateness
ModelComplexityAnalyzer.analyze_complexity(
    n_samples=len(df),
    n_features=1024,  # Your feature count
    model_type='random_forest'
)

# Step 4: Estimate experimental noise
noise_estimator = AssayNoiseEstimator()
error = noise_estimator.estimate_experimental_error(df, target_col='Activity')
```

---

## Complete Workflow Example

### Scenario: Building a QSAR Model with < 200 Compounds

```python
#!/usr/bin/env python3
"""
complete_qsar_workflow.py

Complete QSAR modeling workflow with proper validation
for low-data regime (< 200 compounds)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

# Import validation framework
from qsar_validation import (
    DatasetBiasAnalyzer,
    ActivityCliffDetector,
    ModelComplexityAnalyzer,
    PerformanceMetricsCalculator,
    YRandomizationTester,
    AssayNoiseEstimator,
    print_comprehensive_validation_checklist
)

# Import utilities (if installed)
try:
    from qsar_validation.utils import scaffold_split, generate_fingerprints
except ImportError:
    print("Note: Utility functions not available. Using basic splits.")


def load_and_validate_data(csv_path, smiles_col='SMILES', target_col='Activity'):
    """Load data and run comprehensive validation."""
    
    print("=" * 70)
    print("STEP 1: LOAD AND VALIDATE DATASET")
    print("=" * 70)
    
    # Load data
    df = pd.read_csv(csv_path)
    print(f"\nLoaded {len(df)} compounds")
    
    # Run validation
    print("\n--- Dataset Bias Analysis ---")
    analyzer = DatasetBiasAnalyzer(smiles_col, target_col)
    diversity = analyzer.analyze_scaffold_diversity(df)
    distribution = analyzer.analyze_activity_distribution(df)
    
    print("\n--- Activity Cliff Detection ---")
    cliff_detector = ActivityCliffDetector(smiles_col, target_col)
    cliffs = cliff_detector.detect_activity_cliffs(df)
    
    print("\n--- Experimental Error Estimation ---")
    noise_estimator = AssayNoiseEstimator()
    exp_error = noise_estimator.estimate_experimental_error(df, target_col)
    
    return df, {
        'diversity': diversity,
        'cliffs': cliffs,
        'exp_error': exp_error
    }


def prepare_features(df, smiles_col='SMILES', feature_type='morgan'):
    """
    Prepare molecular features.
    Replace this with your own feature generation.
    """
    print("\n" + "=" * 70)
    print("STEP 2: FEATURE GENERATION")
    print("=" * 70)
    
    from rdkit import Chem
    from rdkit.Chem import AllChem
    
    # Generate Morgan fingerprints as example
    fps = []
    valid_indices = []
    
    for idx, smi in enumerate(df[smiles_col]):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
            fps.append(fp)
            valid_indices.append(idx)
    
    # Convert to numpy array
    X = np.array([[int(b) for b in fp] for fp in fps])
    
    print(f"Generated features: {X.shape}")
    return X, valid_indices


def train_and_evaluate(X, y, df_subset, validation_info):
    """Train model with proper validation."""
    
    print("\n" + "=" * 70)
    print("STEP 3: MODEL TRAINING WITH VALIDATION")
    print("=" * 70)
    
    # Check model complexity appropriateness
    print("\n--- Model Complexity Analysis ---")
    ModelComplexityAnalyzer.analyze_complexity(
        n_samples=len(X),
        n_features=X.shape[1],
        model_type='random_forest'
    )
    
    # Split data (scaffold-based recommended)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Train baseline model (Ridge regression)
    print("\n--- Baseline Model (Ridge Regression) ---")
    baseline = Ridge(alpha=1.0)
    baseline.fit(X_train, y_train)
    y_pred_baseline = baseline.predict(X_test)
    
    baseline_metrics = PerformanceMetricsCalculator.calculate_all_metrics(
        y_test, y_pred_baseline, set_name="Baseline Test"
    )
    
    # Train main model (Random Forest with regularization)
    print("\n--- Main Model (Regularized Random Forest) ---")
    model = RandomForestRegressor(
        n_estimators=50,          # Limited for low data
        max_depth=5,              # Prevent overfitting
        min_samples_leaf=5,       # Regularization
        max_features='sqrt',      # Feature subsampling
        random_state=42
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    metrics = PerformanceMetricsCalculator.calculate_all_metrics(
        y_test, y_pred, set_name="Test"
    )
    
    # Y-Randomization test
    print("\n--- Y-Randomization Test ---")
    rand_results = YRandomizationTester.perform_y_randomization(
        X=X_train,
        y=y_train,
        model=model,
        n_iterations=10
    )
    
    # Compare to experimental error
    exp_error = validation_info['exp_error']['experimental_error']
    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"Test RMSE:              {metrics['rmse']:.3f}")
    print(f"Baseline RMSE:          {baseline_metrics['rmse']:.3f}")
    print(f"Experimental error:     {exp_error:.3f}")
    print(f"Y-random R²:            {rand_results['r2_mean']:.3f}")
    
    if metrics['rmse'] < exp_error:
        print("\n[WARNING] RMSE below experimental error - possible overfitting!")
    
    return model, metrics


def main():
    """Complete workflow."""
    
    # Print validation checklist
    print_comprehensive_validation_checklist()
    
    # Workflow
    csv_path = 'your_data.csv'  # Replace with your data
    
    # 1. Load and validate
    df, validation_info = load_and_validate_data(
        csv_path,
        smiles_col='SMILES',
        target_col='Activity'
    )
    
    # 2. Prepare features
    X, valid_indices = prepare_features(df, smiles_col='SMILES')
    df_subset = df.iloc[valid_indices].reset_index(drop=True)
    y = df_subset['Activity'].values
    
    # 3. Train and evaluate
    model, metrics = train_and_evaluate(X, y, df_subset, validation_info)
    
    print("\n" + "=" * 70)
    print("WORKFLOW COMPLETE")
    print("=" * 70)
    
    return model, metrics


if __name__ == "__main__":
    # Uncomment when you have data
    # model, metrics = main()
    print("Example workflow - modify with your data paths and parameters")
```

---

## Advanced Usage

### Custom Validation Pipeline

```python
from qsar_validation import (
    DatasetBiasAnalyzer,
    ActivityCliffDetector,
    ModelComplexityAnalyzer,
    PerformanceMetricsCalculator
)

class CustomQSARValidator:
    """Custom validation pipeline for your specific needs."""
    
    def __init__(self, smiles_col='SMILES', target_col='Activity'):
        self.smiles_col = smiles_col
        self.target_col = target_col
        self.results = {}
    
    def validate_dataset(self, df):
        """Run dataset-level validation."""
        print("[STEP 1] Dataset Validation")
        
        # Scaffold diversity
        analyzer = DatasetBiasAnalyzer(self.smiles_col, self.target_col)
        self.results['diversity'] = analyzer.analyze_scaffold_diversity(df)
        
        # Activity cliffs
        detector = ActivityCliffDetector(self.smiles_col, self.target_col)
        self.results['cliffs'] = detector.detect_activity_cliffs(df)
        
        return self.results
    
    def validate_model(self, X, y, model, model_type='random_forest'):
        """Run model-level validation."""
        print("[STEP 2] Model Validation")
        
        # Complexity check
        ModelComplexityAnalyzer.analyze_complexity(
            n_samples=len(X),
            n_features=X.shape[1],
            model_type=model_type
        )
        
        return self.results
    
    def validate_predictions(self, y_true, y_pred):
        """Validate predictions."""
        print("[STEP 3] Prediction Validation")
        
        metrics = PerformanceMetricsCalculator.calculate_all_metrics(
            y_true, y_pred, set_name="Validation"
        )
        
        self.results['metrics'] = metrics
        return metrics
    
    def get_report(self):
        """Generate validation report."""
        report = []
        report.append("=" * 70)
        report.append("VALIDATION REPORT")
        report.append("=" * 70)
        
        if 'diversity' in self.results:
            report.append(f"\nScaffold Diversity: {self.results['diversity']['diversity_ratio']:.3f}")
        
        if 'cliffs' in self.results:
            report.append(f"Activity Cliffs: {len(self.results['cliffs'])}")
        
        if 'metrics' in self.results:
            report.append(f"Test RMSE: {self.results['metrics']['rmse']:.3f}")
            report.append(f"Test R²: {self.results['metrics']['r2']:.3f}")
        
        return "\n".join(report)


# Usage
validator = CustomQSARValidator(smiles_col='SMILES', target_col='pIC50')
validator.validate_dataset(df)
validator.validate_model(X_train, y_train, model, 'random_forest')
validator.validate_predictions(y_test, y_pred)
print(validator.get_report())
```

---

## Integration with Your Pipeline

### As a Preprocessing Step

```python
from qsar_validation import run_comprehensive_validation

def preprocess_pipeline(data_path):
    """Integrate validation into preprocessing."""
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Run validation BEFORE modeling
    validation = run_comprehensive_validation(df, 'SMILES', 'Activity')
    
    # Check for issues
    if validation['scaffold_diversity']['diversity_ratio'] < 0.3:
        print("[WARNING] Low scaffold diversity - limited applicability domain")
    
    if len(validation['activity_cliffs']) > len(df) * 0.1:
        print("[WARNING] Many activity cliffs - consider local models")
    
    # Proceed with modeling only if validation passes
    return df, validation
```

### As a Model Evaluation Step

```python
from qsar_validation import PerformanceMetricsCalculator, YRandomizationTester

def evaluate_model(model, X_test, y_test, X_train, y_train):
    """Comprehensive model evaluation."""
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = PerformanceMetricsCalculator.calculate_all_metrics(
        y_test, y_pred, set_name="Test"
    )
    
    # Y-randomization test
    rand_results = YRandomizationTester.perform_y_randomization(
        X_train, y_train, model, n_iterations=10
    )
    
    # Decision logic
    if rand_results['r2_mean'] > 0.2:
        print("[FAIL] Model is overfitting (y-random R² > 0.2)")
        return None
    
    if metrics['rmse'] < 0.3:
        print("[WARNING] RMSE suspiciously low - check for leakage")
    
    return metrics
```

---

## Low-Data Regime Best Practices

### When n < 200 compounds

```python
# Use these settings in your models:

# 1. Random Forest
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=50,        # Not 100+ 
    max_depth=5,            # Limit depth
    min_samples_leaf=5,     # Require minimum samples
    max_features='sqrt',    # Feature subsampling
    random_state=42
)

# 2. Gradient Boosting
from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor(
    n_estimators=50,
    learning_rate=0.05,     # Slow learning
    max_depth=3,            # Shallow trees
    subsample=0.8,          # Stochastic boosting
    random_state=42
)

# 3. Linear Models (Best for very low data)
from sklearn.linear_model import Ridge, ElasticNet

model = Ridge(alpha=1.0)  # Always use regularization

# Or
model = ElasticNet(alpha=0.5, l1_ratio=0.5)
```

### Validation Strategy

```python
from qsar_validation import ModelComplexityAnalyzer

# Check before training
ModelComplexityAnalyzer.analyze_complexity(
    n_samples=150,          # Your sample count
    n_features=2048,        # Your feature count
    model_type='random_forest'
)

# Ratio guidelines:
# < 5:  Use simple linear models only
# < 10: Use regularized models
# < 20: Be very careful with complex models
# > 20: Can use moderate complexity
```

---

## Command-Line Interface (Optional)

Create a CLI script:

```python
#!/usr/bin/env python3
"""
qsar_validate.py - Command-line validation tool
"""

import argparse
import pandas as pd
from qsar_validation import run_comprehensive_validation, print_comprehensive_validation_checklist

def main():
    parser = argparse.ArgumentParser(
        description='QSAR Dataset Validation Tool'
    )
    parser.add_argument('data', help='Path to CSV file')
    parser.add_argument('--smiles', default='SMILES', help='SMILES column name')
    parser.add_argument('--target', default='Activity', help='Target column name')
    parser.add_argument('--checklist', action='store_true', help='Show validation checklist')
    
    args = parser.parse_args()
    
    if args.checklist:
        print_comprehensive_validation_checklist()
        return
    
    # Load and validate
    df = pd.read_csv(args.data)
    results = run_comprehensive_validation(df, args.smiles, args.target)
    
    print("\n[DONE] Validation complete!")

if __name__ == "__main__":
    main()
```

Usage:
```bash
# Show checklist
python qsar_validate.py --checklist

# Validate dataset
python qsar_validate.py mydata.csv --smiles SMILES --target pIC50
```

---

## Minimal Example for Quick Testing

```python
"""Minimal working example - 10 lines of code"""

import pandas as pd
from qsar_validation import run_comprehensive_validation

# Your data (CSV with SMILES and activity)
df = pd.read_csv('data.csv')

# One function does everything
results = run_comprehensive_validation(df, 'SMILES', 'Activity')

# Done! Check the printed output for warnings and recommendations
```

---

## Troubleshooting

### Import Errors
```python
# If imports fail, use absolute imports:
import sys
sys.path.append('/path/to/QSAR_Models')
from src.qsar_validation import run_comprehensive_validation
```

### Missing Dependencies
```bash
pip install rdkit pandas numpy scikit-learn scipy
```

### RDKit Issues
```bash
# Use conda for RDKit
conda install -c conda-forge rdkit
```

---

## Summary

This framework is designed to work **without notebooks** and can be:

1. **Imported** as a Python package
2. **Integrated** into existing pipelines
3. **Used** from command line
4. **Extended** with custom validators
5. **Applied** to any QSAR dataset

The key is that it's **modular**, **reproducible**, and specifically designed for **low-data regimes** (< 200 compounds).

For more examples, see the `examples/` directory in the repository.
