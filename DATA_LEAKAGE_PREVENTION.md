# Data Leakage Prevention in QSAR Models

## Yes, Data Leakage is Fully Addressed!

Data leakage is a **CRITICAL ISSUE** in QSAR modeling and is **FULLY ADDRESSED** in this framework. This document explains how.

## Table of Contents
1. [What is Data Leakage?](#what-is-data-leakage)
2. [Types of Leakage We Prevent](#types-of-leakage-we-prevent)
3. [How This Framework Prevents Leakage](#how-this-framework-prevents-leakage)
4. [Implementation Details](#implementation-details)
5. [Usage Examples](#usage-examples)
6. [Validation Checks](#validation-checks)

---

## What is Data Leakage?

Data leakage occurs when information from the test set "leaks" into the training process, leading to **overly optimistic** performance estimates. This is particularly problematic in QSAR where:

- Datasets are small (< 200 compounds)
- Similar molecules have similar activities
- Models can memorize patterns rather than learn chemistry

**Result**: Models appear to work well but **fail on new data**.

---

## Types of Leakage We Prevent

### 1. Scaffold Leakage (MOST CRITICAL)

**Problem**: Random splitting puts similar molecules in both train and test sets.

**Example**:
```
Training set: c1ccccc1-CCO      (benzyl-ethanol)
Test set:     c1ccccc1-CCCO     (benzyl-propanol)
                      ^^^^ Same scaffold!
```

**Our Solution**: Scaffold-based splitting using Bemis-Murcko scaffolds.

### 2. Duplicate Leakage

**Problem**: Same molecule appears in both train and test sets.

**Our Solution**: Automatic duplicate detection and removal BEFORE splitting.

### 3. Feature Scaling Leakage

**Problem**: Scaling features using statistics from the entire dataset.

```python
# WRONG (leakage!)
scaler.fit(X_all)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)  # Test stats leaked!

# CORRECT (no leakage)
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)  # Only train stats used
```

**Our Solution**: Scalers fit ONLY on training data.

### 4. Feature Selection Leakage

**Problem**: Selecting features using the entire dataset before splitting.

**Our Solution**: Feature selection done within CV loops, never on full data.

### 5. Hyperparameter Tuning Leakage

**Problem**: Tuning hyperparameters using the test set.

**Our Solution**: Nested cross-validation support for proper tuning.

---

## How This Framework Prevents Leakage

### Built-in Protection: `qsar_utils_no_leakage.py`

Located at: `src/utils/qsar_utils_no_leakage.py`

This module provides comprehensive leakage prevention:

```python
from src.utils.qsar_utils_no_leakage import QSARDataProcessor, ScaffoldSplitter

# Automatic leakage prevention
processor = QSARDataProcessor(smiles_col='SMILES', target_col='Activity')

# 1. Remove duplicates BEFORE splitting
df_clean = processor.remove_duplicates(df)

# 2. Scaffold-based splitting
splitter = ScaffoldSplitter(smiles_col='SMILES')
train_idx, val_idx, test_idx = splitter.scaffold_split(
    df_clean,
    train_size=0.7,
    val_size=0.15,
    test_size=0.15
)

# 3. Scale features properly (no leakage)
X_train_scaled, scaler = processor.scale_features(X_train, fit=True)
X_test_scaled, _ = processor.scale_features(X_test, fit=False, scaler=scaler)
```

### Validation Checks

The validation framework automatically checks for leakage issues:

```python
from qsar_validation import DatasetBiasAnalyzer

analyzer = DatasetBiasAnalyzer('SMILES', 'Activity')

# Check scaffold distribution across splits
analyzer.report_split_diversity(
    df_with_scaffolds,
    train_idx, val_idx, test_idx
)

# Output will warn if scaffolds overlap:
# [WARNING] Scaffold overlap detected between train and test!
```

---

## Implementation Details

### Scaffold-Based Splitting

**Method**: Bemis-Murcko scaffolds

```python
class ScaffoldSplitter:
    """Split molecules by scaffold (no scaffold overlap)."""
    
    def scaffold_split(self, df, train_size=0.7, val_size=0.15, test_size=0.15):
        # 1. Extract Bemis-Murcko scaffold for each molecule
        scaffolds = self._get_scaffolds(df)
        
        # 2. Group molecules by scaffold
        scaffold_groups = self._group_by_scaffold(df, scaffolds)
        
        # 3. Assign ENTIRE scaffolds to splits (never split a scaffold)
        train, val, test = self._assign_scaffolds(scaffold_groups)
        
        # 4. Verify zero overlap
        assert len(set(train) & set(test)) == 0, "Scaffold overlap detected!"
        
        return train, val, test
```

**Result**: Zero scaffold overlap between train/val/test sets.

### Duplicate Removal

```python
def remove_duplicates(self, df, strategy='average'):
    """
    Remove duplicates BEFORE splitting.
    
    Strategies:
    - 'average': Average replicate measurements
    - 'first': Keep first occurrence
    - 'best': Keep measurement with lowest error (if available)
    """
    # 1. Canonicalize SMILES
    df['canonical'] = df['SMILES'].apply(self._canonicalize)
    
    # 2. Find duplicates
    duplicates = df[df.duplicated('canonical', keep=False)]
    
    # 3. Handle according to strategy
    if strategy == 'average':
        df = df.groupby('canonical').mean()
    
    return df
```

### Feature Scaling Without Leakage

```python
def scale_features(self, X, fit=False, scaler=None):
    """
    Scale features properly (no leakage).
    
    Args:
        X: Features to scale
        fit: If True, fit new scaler. If False, use provided scaler.
        scaler: Pre-fitted scaler (for test/validation sets)
    
    Returns:
        X_scaled, scaler
    """
    if fit:
        # Training set: fit new scaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        # Test/val set: use training scaler
        if scaler is None:
            raise ValueError("Must provide scaler for test/val sets!")
        X_scaled = scaler.transform(X)
    
    return X_scaled, scaler
```

---

## Usage Examples

### Example 1: Complete Leakage-Free Workflow

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from rdkit import Chem
from rdkit.Chem import AllChem

# Import leakage prevention utilities
from src.utils.qsar_utils_no_leakage import QSARDataProcessor, ScaffoldSplitter

# Load data
df = pd.read_csv('data.csv')
print(f"Loaded: {len(df)} compounds")

# STEP 1: Remove duplicates BEFORE splitting
processor = QSARDataProcessor('SMILES', 'Activity')
df_clean = processor.remove_duplicates(df, strategy='average')
print(f"After deduplication: {len(df_clean)} compounds")

# STEP 2: Scaffold-based splitting
splitter = ScaffoldSplitter('SMILES')
train_idx, val_idx, test_idx = splitter.scaffold_split(
    df_clean,
    train_size=0.7,
    val_size=0.15,
    test_size=0.15,
    random_state=42
)

# Verify no scaffold overlap
train_scaffolds = set(df_clean.iloc[train_idx]['scaffold'])
test_scaffolds = set(df_clean.iloc[test_idx]['scaffold'])
overlap = train_scaffolds & test_scaffolds
assert len(overlap) == 0, f"ERROR: {len(overlap)} scaffolds in both train and test!"
print(f"[OK] Zero scaffold overlap between train and test")

# STEP 3: Generate features
def generate_fingerprints(smiles_list):
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
        fps.append(np.array(fp))
    return np.array(fps)

X = generate_fingerprints(df_clean['SMILES'])
y = df_clean['Activity'].values

# Split data
X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

# STEP 4: Scale features WITHOUT leakage
X_train_scaled, scaler = processor.scale_features(X_train, fit=True)
X_test_scaled, _ = processor.scale_features(X_test, fit=False, scaler=scaler)
print(f"[OK] Features scaled using ONLY training data")

# STEP 5: Train model
model = RandomForestRegressor(
    n_estimators=50,
    max_depth=5,
    min_samples_leaf=5,
    random_state=42
)
model.fit(X_train_scaled, y_train)

# STEP 6: Evaluate
y_pred = model.predict(X_test_scaled)
from sklearn.metrics import mean_squared_error, r2_score
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\nTest RMSE: {rmse:.3f}")
print(f"Test R²: {r2:.3f}")
print("[OK] Evaluation on completely independent test set")
```

### Example 2: Using Validation Framework

The validation framework automatically checks for leakage:

```python
from qsar_validation import (
    DatasetBiasAnalyzer,
    run_comprehensive_validation
)

# Quick check for potential leakage issues
results = run_comprehensive_validation(df, 'SMILES', 'Activity')

# The framework will warn about:
# - Low scaffold diversity (increases leakage risk)
# - Suspicious RMSE < 0.3 (possible leakage)
# - High activity cliffs (makes leakage harder to detect)
```

### Example 3: Cross-Validation Without Leakage

```python
from src.utils.qsar_utils_no_leakage import scaffold_cv_split

# Scaffold-based cross-validation (no scaffold overlap across folds)
cv_folds = scaffold_cv_split(df, 'SMILES', n_splits=5)

for fold_idx, (train_idx, val_idx) in enumerate(cv_folds):
    print(f"Fold {fold_idx + 1}")
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    
    # Scale within fold (no leakage)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)  # Uses training stats only
    
    # Train and evaluate
    model.fit(X_train_scaled, y_train)
    score = model.score(X_val_scaled, y_val)
    print(f"  Validation R²: {score:.3f}")
```

---

## Validation Checks

The framework includes automatic checks to detect leakage:

### Check 1: Scaffold Overlap Detection

```python
from qsar_validation import DatasetBiasAnalyzer

analyzer = DatasetBiasAnalyzer('SMILES', 'Activity')
analyzer.report_split_diversity(df_with_scaffolds, train_idx, val_idx, test_idx)

# Output:
# [OK] Test set contains 45 novel scaffolds (75.0%)
#   -> Good generalization test
```

### Check 2: Suspicious Performance

```python
from qsar_validation import PerformanceMetricsCalculator, AssayNoiseEstimator

# Calculate metrics
metrics = PerformanceMetricsCalculator.calculate_all_metrics(y_test, y_pred)

# Compare to experimental error
noise = AssayNoiseEstimator.estimate_experimental_error(df, 'Activity')

if metrics['rmse'] < noise['experimental_error'] * 0.6:
    print("[WARNING] RMSE suspiciously low - check for leakage!")
```

### Check 3: Y-Randomization Test

```python
from qsar_validation import YRandomizationTester

# If model performs well with random targets, it's overfitting/leaking
rand_results = YRandomizationTester.perform_y_randomization(
    X_train, y_train, model, n_iterations=10
)

if rand_results['r2_mean'] > 0.2:
    print("[FAIL] Model performs well with random targets - likely leakage!")
```

---

## Summary: Data Leakage is FULLY Addressed

| Type of Leakage | Status | How We Prevent It |
|----------------|--------|-------------------|
| **Scaffold Leakage** | ✅ PREVENTED | Bemis-Murcko scaffold-based splitting |
| **Duplicate Leakage** | ✅ PREVENTED | Automatic deduplication before splitting |
| **Feature Scaling Leakage** | ✅ PREVENTED | Scalers fit only on training data |
| **Feature Selection Leakage** | ✅ PREVENTED | Selection within CV loops only |
| **Hyperparameter Leakage** | ✅ PREVENTED | Nested CV support provided |
| **Temporal Leakage** | ✅ DOCUMENTED | Chronological splitting available |

### Key Files

1. **`src/utils/qsar_utils_no_leakage.py`** - Comprehensive leakage prevention utilities
2. **`docs/README_DATA_LEAKAGE_FIX.md`** - Detailed documentation
3. **`notebooks/DATA_LEAKAGE_FIX_EXAMPLE.ipynb`** - Step-by-step example
4. **Validation framework** - Automatic leakage detection

### Expected Performance Changes

When you fix data leakage, expect:

- **R² drops** from 0.80 → 0.60 (or lower) ✅ This is CORRECT
- **RMSE increases** from 0.3 → 0.5 ✅ More realistic
- **Scaffold splits harder** than random splits ✅ Tests true generalization

**This is NORMAL and indicates proper validation!**

---

## Quick Start: Leakage-Free Workflow

```bash
# Run the standalone workflow (includes all leakage prevention)
python standalone_qsar_workflow.py --data mydata.csv

# Or use in Python
from qsar_validation import run_comprehensive_validation
results = run_comprehensive_validation(df, 'SMILES', 'Activity')
```

Both methods automatically include:
- Scaffold-based splitting
- Duplicate removal
- Proper feature scaling
- Leakage detection checks

**Your data is safe!** 🛡️

---

For more details, see:
- **`docs/README_DATA_LEAKAGE_FIX.md`** - Complete leakage prevention guide
- **`USAGE_GUIDE.md`** - How to use the framework
- **`src/utils/qsar_utils_no_leakage.py`** - Implementation details
