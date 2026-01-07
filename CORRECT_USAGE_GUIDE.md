# ✅ CORRECT USAGE GUIDE - QSAR Framework v4.1.0

## 🔴 Common Import Errors - FIXED!

The README examples had incorrect module names. Here are the **CORRECT** imports and usage patterns.

---

## ✅ CORRECT IMPORTS

### For Duplicate Removal (NOT a separate module)

**❌ WRONG (from README):**
```python
from qsar_validation.duplicate_removal import DuplicateRemoval
```

**✅ CORRECT:**
```python
from utils.qsar_utils_no_leakage import QSARDataProcessor

# Use it like this:
processor = QSARDataProcessor(smiles_col='SMILES', target_col='Activity')
df = processor.remove_duplicates(df, strategy='average')
```

---

## 📋 Complete Correct Import Reference

### Core Data Processing

```python
# Data processor with duplicate removal
from utils.qsar_utils_no_leakage import QSARDataProcessor

# Usage:
processor = QSARDataProcessor(smiles_col='SMILES', target_col='Activity')
df = processor.canonicalize_smiles(df)
df = processor.remove_duplicates(df, strategy='average')
```

### Data Splitting

```python
# Advanced splitter with all strategies
from qsar_validation.splitting_strategies import AdvancedSplitter

# Usage:
splitter = AdvancedSplitter()
splits = splitter.scaffold_split(
    df, 
    smiles_col='SMILES', 
    target_col='Activity',
    test_size=0.2
)
train_idx = splits['train_idx']
test_idx = splits['test_idx']
```

### Feature Engineering

```python
# Feature scaling
from qsar_validation.feature_scaling import FeatureScaler

# Usage:
scaler = FeatureScaler(method='standard')
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

```python
# Feature selection
from qsar_validation.feature_selection import FeatureSelector

# Usage:
selector = FeatureSelector(method='variance', threshold=0.01)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)
```

```python
# PCA
from qsar_validation.pca_module import PCATransformer

# Usage:
pca = PCATransformer(n_components=0.95)  # 95% variance
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
```

### Dataset Quality Analysis

```python
from qsar_validation.dataset_quality_analysis import DatasetQualityAnalyzer

# Usage:
analyzer = DatasetQualityAnalyzer(smiles_col='SMILES', activity_col='Activity')
quality_report = analyzer.analyze(df)
```

### Model Complexity Control

```python
from qsar_validation.model_complexity_control import ModelComplexityController

# Usage:
controller = ModelComplexityController(
    n_samples=len(X_train),
    n_features=X_train.shape[1]
)
recommendations = controller.recommend_models()
```

### Performance Validation

```python
from qsar_validation.performance_validation import PerformanceValidator

# Usage:
validator = PerformanceValidator()
cv_results = validator.cross_validate(model, X_train, y_train, cv=5)
```

### Activity Analysis

```python
from qsar_validation.activity_cliffs_detection import ActivityCliffsDetector

# Usage:
detector = ActivityCliffsDetector(smiles_col='SMILES', activity_col='Activity')
cliffs = detector.detect_cliffs(df)
```

### Uncertainty Estimation

```python
from qsar_validation.uncertainty_estimation import UncertaintyEstimator

# Usage:
estimator = UncertaintyEstimator()
predictions, uncertainty = estimator.predict_with_uncertainty(model, X_test)
```

### Metrics

```python
from qsar_validation.metrics import PerformanceMetricsCalculator

# Usage:
metrics_calc = PerformanceMetricsCalculator()
metrics = metrics_calc.calculate_all_metrics(y_true, y_pred)
```

---

## 🚀 COMPLETE WORKING EXAMPLE (COPY-PASTE READY)

### For Google Colab:

```python
# ===== CELL 1: SETUP =====
import os
try:
    import google.colab
    !git clone https://github.com/bhatnira/Roy-QSAR-Generative-dev.git
    %cd Roy-QSAR-Generative-dev/notebooks
    !pip install -q rdkit-pypi pandas numpy scikit-learn matplotlib seaborn
    print("✅ Setup complete!")
except:
    print("✅ Running locally")

# ===== CELL 2: CORRECT IMPORTS =====
import sys
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# Add framework to path
sys.path.insert(0, '../src')

# CORRECT imports
from utils.qsar_utils_no_leakage import QSARDataProcessor
from qsar_validation.splitting_strategies import AdvancedSplitter
from qsar_validation.feature_scaling import FeatureScaler
from qsar_validation.feature_selection import FeatureSelector
from qsar_validation.dataset_quality_analysis import DatasetQualityAnalyzer
from qsar_validation.performance_validation import PerformanceValidator
from qsar_validation.activity_cliffs_detection import ActivityCliffsDetector

print("✅ All modules imported successfully!")

# ===== CELL 3: LOAD DATA =====
# Example with dummy data
df = pd.DataFrame({
    'SMILES': ['CCO', 'CCOCC', 'c1ccccc1', 'CCO', 'CCCO'],  # Note: duplicate CCO
    'Activity': [5.2, 6.1, 7.3, 5.3, 6.8]
})
print(f"Original dataset: {len(df)} molecules")

# ===== CELL 4: CLEAN DATA =====
processor = QSARDataProcessor(smiles_col='SMILES', target_col='Activity')

# Canonicalize SMILES
df = processor.canonicalize_smiles(df)

# Remove duplicates (averages replicates)
df = processor.remove_duplicates(df, strategy='average')
print(f"Clean dataset: {len(df)} molecules")

# ===== CELL 5: SPLIT DATA =====
splitter = AdvancedSplitter()
splits = splitter.scaffold_split(
    df,
    smiles_col='SMILES',
    target_col='Activity',
    test_size=0.3
)

train_idx = splits['train_idx']
test_idx = splits['test_idx']

print(f"Train: {len(train_idx)} | Test: {len(test_idx)}")

# ===== CELL 6: GENERATE FEATURES =====
# (Your feature generation code here - fingerprints, descriptors, etc.)
from rdkit import Chem
from rdkit.Chem import AllChem

def get_morgan_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=512)
        return np.array(fp)
    return np.zeros(512)

X = np.array([get_morgan_fingerprint(s) for s in df['SMILES']])
y = df['Activity'].values

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

print(f"Features: {X.shape}")

# ===== CELL 7: SCALE FEATURES =====
scaler = FeatureScaler(method='standard')
X_train_scaled = scaler.fit_transform(X_train)  # Fit on train only!
X_test_scaled = scaler.transform(X_test)        # Transform test

print("✅ Features scaled")

# ===== CELL 8: TRAIN MODEL =====
model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train_scaled, y_train)

print("✅ Model trained")

# ===== CELL 9: VALIDATE =====
validator = PerformanceValidator()

# Cross-validation on training set
cv_results = validator.cross_validate(
    model, X_train_scaled, y_train, cv=3
)
print(f"CV R²: {cv_results['test_r2'].mean():.3f}")

# Test set evaluation
test_predictions = model.predict(X_test_scaled)
test_r2 = validator.calculate_r2(y_test, test_predictions)
print(f"Test R²: {test_r2:.3f}")

print("\n✅ Analysis complete!")
```

---

## 📚 Module Availability Reference

### What EXISTS in src/qsar_validation/:

✅ `activity_cliffs.py` / `activity_cliffs_detection.py`
✅ `assay_noise.py`
✅ `dataset_analysis.py` / `dataset_quality_analysis.py`
✅ `feature_scaling.py`
✅ `feature_selection.py`
✅ `metrics.py`
✅ `model_agnostic_pipeline.py`
✅ `model_complexity.py` / `model_complexity_control.py`
✅ `pca_module.py`
✅ `performance_validation.py`
✅ `randomization.py`
✅ `splitting_strategies.py`
✅ `uncertainty_estimation.py`
✅ `validation_runner.py`

### What EXISTS in src/utils/:

✅ `qsar_utils_no_leakage.py` (contains QSARDataProcessor)
✅ `qsar_validation_utils.py`

### What DOES NOT EXIST:

❌ `duplicate_removal.py` - Use `QSARDataProcessor` instead
❌ `scaffold_splitter.py` - Use `AdvancedSplitter` instead
❌ Separate splitter files - All in `splitting_strategies.py`

---

## 🎯 Quick Fix for Common Errors

### Error: "No module named 'qsar_validation.duplicate_removal'"

**Fix:**
```python
# Don't import from qsar_validation.duplicate_removal
# Instead:
from utils.qsar_utils_no_leakage import QSARDataProcessor
processor = QSARDataProcessor(smiles_col='SMILES')
df = processor.remove_duplicates(df)
```

### Error: "No module named 'qsar_validation.scaffold_splitter'"

**Fix:**
```python
# Don't import scaffold_splitter separately
# Instead:
from qsar_validation.splitting_strategies import AdvancedSplitter
splitter = AdvancedSplitter()
splits = splitter.scaffold_split(df, smiles_col='SMILES', target_col='Activity')
```

### Error: "No module named 'utils'"

**Fix:**
```python
# Make sure framework is in path
import sys
import os
sys.path.insert(0, '/path/to/Roy-QSAR-Generative-dev/src')

# Or if in notebooks folder:
sys.path.insert(0, '../src')
```

---

## 📖 Correct Minimal Example

```python
import sys
sys.path.insert(0, '../src')  # Adjust path as needed

# Step 1: Import (CORRECT way)
from utils.qsar_utils_no_leakage import QSARDataProcessor
from qsar_validation.splitting_strategies import AdvancedSplitter
from qsar_validation.feature_scaling import FeatureScaler

# Step 2: Use
processor = QSARDataProcessor(smiles_col='SMILES', target_col='Activity')
df = processor.remove_duplicates(df)

splitter = AdvancedSplitter()
splits = splitter.scaffold_split(df, smiles_col='SMILES', target_col='Activity')

scaler = FeatureScaler(method='standard')
X_scaled = scaler.fit_transform(X)
```

---

## 🔍 How to Check What's Available

```python
# List all available modules
import os
src_path = '../src/qsar_validation'
modules = [f for f in os.listdir(src_path) if f.endswith('.py') and not f.startswith('__')]
print("Available modules:")
for m in modules:
    print(f"  • {m}")
```

---

## ✅ Updated Quick Reference

| Functionality | Correct Import | Correct Usage |
|---------------|----------------|---------------|
| Duplicate removal | `from utils.qsar_utils_no_leakage import QSARDataProcessor` | `processor.remove_duplicates(df)` |
| Scaffold splitting | `from qsar_validation.splitting_strategies import AdvancedSplitter` | `splitter.scaffold_split(df, ...)` |
| Feature scaling | `from qsar_validation.feature_scaling import FeatureScaler` | `scaler.fit_transform(X)` |
| Feature selection | `from qsar_validation.feature_selection import FeatureSelector` | `selector.fit_transform(X, y)` |
| PCA | `from qsar_validation.pca_module import PCATransformer` | `pca.fit_transform(X)` |
| Quality analysis | `from qsar_validation.dataset_quality_analysis import DatasetQualityAnalyzer` | `analyzer.analyze(df)` |
| Complexity control | `from qsar_validation.model_complexity_control import ModelComplexityController` | `controller.recommend_models()` |
| Validation | `from qsar_validation.performance_validation import PerformanceValidator` | `validator.cross_validate(...)` |
| Activity cliffs | `from qsar_validation.activity_cliffs_detection import ActivityCliffsDetector` | `detector.detect_cliffs(df)` |
| Uncertainty | `from qsar_validation.uncertainty_estimation import UncertaintyEstimator` | `estimator.predict_with_uncertainty(...)` |

---

**Use this guide instead of the README examples to avoid import errors!** ✅
