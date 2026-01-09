# QSAR Validation Framework v4.1.0

Modular QSAR validation with data leakage prevention. Works with sklearn, XGBoost, LightGBM, PyTorch, TensorFlow.

[![GitHub](https://img.shields.io/badge/GitHub-Roy--QSAR--Generative--dev-blue?logo=github)](https://github.com/bhatnira/Roy-QSAR-Generative-dev)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Quick Start

### Google Colab (Recommended)

```python
# Install from GitHub
!pip install git+https://github.com/bhatnira/Roy-QSAR-Generative-dev.git

# Import and use
from utils.qsar_utils_no_leakage import quick_clean
from qsar_validation.splitting_strategies import AdvancedSplitter
from qsar_validation.feature_scaling import FeatureScaler
```

### Local Installation

```bash
# Editable install (changes reflect immediately)
git clone https://github.com/bhatnira/Roy-QSAR-Generative-dev.git
cd Roy-QSAR-Generative-dev
pip install -e .
```

---

## Available Modules

| Module | Import | Purpose |
|--------|--------|---------|
| **Data Cleaning** | `from utils.qsar_utils_no_leakage import quick_clean` | Remove duplicates, canonicalize SMILES |
| **Splitting** | `from qsar_validation.splitting_strategies import AdvancedSplitter` | Scaffold/temporal/cluster splits |
| **Scaling** | `from qsar_validation.feature_scaling import FeatureScaler` | StandardScaler (fit on train only) |
| **Selection** | `from qsar_validation.feature_selection import FeatureSelector` | Variance, correlation filtering |
| **Validation** | `from qsar_validation.performance_validation import PerformanceValidator` | Cross-validation, metrics |
| **Dataset Quality** | `from qsar_validation.dataset_quality_analysis import DatasetQualityAnalyzer` | Scaffold diversity, chemical space |
| **Activity Cliffs** | `from qsar_validation.activity_cliffs_detection import ActivityCliffsDetector` | Detect activity cliffs |
| **Uncertainty** | `from qsar_validation.uncertainty_estimation import UncertaintyEstimator` | Prediction uncertainty |
| **Model Complexity** | `from qsar_validation.model_complexity_control import ModelComplexityController` | Model recommendations |
| **PCA** | `from qsar_validation.pca_module import PCATransformer` | Dimensionality reduction |

---

## Module Usage Guide

### 1. Data Cleaning

```python
from utils.qsar_utils_no_leakage import quick_clean, QSARDataProcessor

# Quick cleaning (simple)
df_clean = quick_clean(df, smiles_col='SMILES', target_col='pIC50')

# Advanced cleaning with full control
processor = QSARDataProcessor(smiles_col='SMILES', target_col='pIC50')
df = processor.canonicalize_smiles(df)
df_clean = processor.remove_duplicates(df, strategy='average')
near_dups = processor.detect_near_duplicates(df_clean['SMILES'], threshold=0.95)
```

### 2. Data Splitting

```python
from qsar_validation.splitting_strategies import AdvancedSplitter

splitter = AdvancedSplitter(smiles_col='SMILES', strategy='scaffold')

# Option A: Get indices
train_idx, val_idx, test_idx = splitter.split(df, test_size=0.2, val_size=0.1)

# Option B: Direct split
train_idx, test_idx = splitter.scaffold_split(df['SMILES'], test_size=0.2)

# Temporal split (requires date column)
splitter_temporal = AdvancedSplitter(strategy='temporal', date_col='Date')
train_idx, test_idx = splitter_temporal.split(df, test_size=0.2)

# Cluster split
splitter_cluster = AdvancedSplitter(strategy='cluster', n_clusters=5)
train_idx, test_idx = splitter_cluster.split(df, test_size=0.2)
```

### 3. Feature Scaling

```python
from qsar_validation.feature_scaling import FeatureScaler

# Standard scaling (most common)
scaler = FeatureScaler(method='standard')
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# MinMax scaling (0-1 range)
scaler = FeatureScaler(method='minmax')
X_train_scaled = scaler.fit_transform(X_train)

# Robust scaling (better for outliers)
scaler = FeatureScaler(method='robust')
X_train_scaled = scaler.fit_transform(X_train)
```

### 4. Feature Selection

```python
from qsar_validation.feature_selection import FeatureSelector

selector = FeatureSelector()

# Variance threshold
X_selected = selector.variance_threshold(X_train, threshold=0.01)
X_test_selected = selector.transform(X_test)

# Correlation filter
X_selected = selector.correlation_filter(X_train, threshold=0.95)

# Model-based selection (Random Forest)
X_selected = selector.model_based_selection(
    X_train, y_train, 
    model='rf', 
    n_features=100
)
X_test_selected = selector.transform(X_test)

# Univariate selection (F-test)
X_selected = selector.univariate_selection(
    X_train, y_train, 
    method='f_regression', 
    k=100
)
```

### 5. Performance Validation

```python
from qsar_validation.performance_validation import PerformanceValidator

validator = PerformanceValidator()

# Calculate all metrics
metrics = validator.calculate_comprehensive_metrics(y_test, y_pred, set_name='Test')
print(f"R²: {metrics['r2']:.3f}")
print(f"RMSE: {metrics['rmse']:.3f}")
print(f"Spearman: {metrics['spearman']:.3f}")

# Y-randomization test (negative control)
random_results = validator.y_randomization_test(
    X_train, y_train, 
    model=model, 
    n_iterations=100
)
print(f"Random R²: {random_results['mean_r2']:.3f} (should be ~0)")

# Cross-validation
cv_results = validator.cross_validate(
    model, X_train, y_train, 
    cv=5, 
    scoring='r2'
)
print(f"CV R²: {cv_results['mean']:.3f} ± {cv_results['std']:.3f}")
```

### 6. Dataset Quality Analysis

```python
from qsar_validation.dataset_quality_analysis import DatasetQualityAnalyzer

analyzer = DatasetQualityAnalyzer(
    smiles_col='SMILES',
    activity_col='pIC50'
)

report = analyzer.analyze(df)

# Check quality metrics
print(f"Dataset size: {report['n_molecules']}")
print(f"Unique scaffolds: {report['n_scaffolds']}")
print(f"Diversity ratio: {report['diversity_ratio']:.2f}")
print(f"Activity range: {report['activity_range']:.2f}")

# Check warnings
if report['warnings']:
    print("Warnings:", report['warnings'])
```

### 7. Activity Cliffs Detection

```python
from qsar_validation.activity_cliffs_detection import ActivityCliffsDetector

detector = ActivityCliffsDetector()

# Detect activity cliffs
cliffs = detector.detect_cliffs(
    df['SMILES'],
    df['pIC50'],
    similarity_threshold=0.85,
    activity_threshold=1.0  # 1 log unit difference
)

print(f"Activity cliffs found: {len(cliffs)}")

# Calculate SALI (Structure-Activity Landscape Index)
sali_results = detector.calculate_sali(df['SMILES'], df['pIC50'])
print(f"Mean SALI: {sali_results['mean_sali']:.3f}")
```

### 8. Uncertainty Estimation

```python
from qsar_validation.uncertainty_estimation import UncertaintyEstimator

estimator = UncertaintyEstimator()

# Bootstrap uncertainty
predictions, uncertainties = estimator.bootstrap_uncertainty(
    model, X_test, 
    n_bootstrap=100
)

print(f"Predictions: {predictions[:5]}")
print(f"Uncertainties: {uncertainties[:5]}")

# Ensemble uncertainty (requires ensemble model)
from sklearn.ensemble import RandomForestRegressor
ensemble_model = RandomForestRegressor(n_estimators=100)
ensemble_model.fit(X_train, y_train)

pred_mean, pred_std = estimator.ensemble_uncertainty(ensemble_model, X_test)
```

### 9. Model Complexity Control

```python
from qsar_validation.model_complexity_control import ModelComplexityController

controller = ModelComplexityController(
    n_samples=len(X_train),
    n_features=X_train.shape[1]
)

# Get model recommendations
recommendations = controller.recommend_models()
print("Recommended models:", recommendations)

# Get safe hyperparameter grid
from sklearn.ensemble import RandomForestRegressor
param_grid = controller.get_safe_param_grid('random_forest', library='sklearn')
print("Safe parameters:", param_grid)

# Nested cross-validation
results = controller.nested_cv(
    X_train, y_train,
    model=RandomForestRegressor(),
    param_grid=param_grid,
    outer_cv=5,
    inner_cv=3
)
print(f"Nested CV R²: {results['test_score_mean']:.3f}")
```

### 10. PCA (Dimensionality Reduction)

```python
from qsar_validation.pca_module import PCATransformer

# Retain 95% variance
pca = PCATransformer(n_components=0.95)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print(f"Original features: {X_train.shape[1]}")
print(f"PCA components: {pca.n_components_}")
print(f"Explained variance: {pca.explained_variance_ratio_.sum():.3f}")

# Fixed number of components
pca = PCATransformer(n_components=50)
X_train_pca = pca.fit_transform(X_train)
```

---

## Usage Example

### Complete Pipeline

```python
from utils.qsar_utils_no_leakage import quick_clean
from qsar_validation.splitting_strategies import AdvancedSplitter
from qsar_validation.feature_scaling import FeatureScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# 1. Clean data (removes duplicates, canonicalizes SMILES)
df_clean = quick_clean(df, smiles_col='SMILES', target_col='pIC50')

# 2. Scaffold split (prevents data leakage)
splitter = AdvancedSplitter()
train_idx, test_idx = splitter.scaffold_split(
    df_clean['SMILES'], 
    test_size=0.2
)

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# 3. Scale features (fit on train only)
scaler = FeatureScaler(method='standard')
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train and evaluate
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train_scaled, y_train)
score = r2_score(y_test, model.predict(X_test_scaled))
print(f"Test R²: {score:.3f}")
```

### Feature Selection

```python
from qsar_validation.feature_selection import FeatureSelector

selector = FeatureSelector()

# Remove low-variance features
X_var = selector.variance_threshold(X_train, threshold=0.01)

# Remove correlated features  
X_final = selector.correlation_filter(X_var, threshold=0.95)

print(f"Features: {X.shape[1]} → {X_final.shape[1]}")
```

---

## Key Principles

1. **Remove duplicates BEFORE splitting**
2. **Use scaffold-based splits** (not random)
3. **Fit scalers on train only** (no leakage)
4. **Generate features AFTER splitting**

---

## Troubleshooting

**RDKit not found:**
```bash
conda install -c conda-forge rdkit
```

**Permission denied:**
```bash
pip install --user git+https://github.com/bhatnira/Roy-QSAR-Generative-dev.git
```

**Import errors:**
```python
import sys
sys.path.insert(0, '/path/to/Roy-QSAR-Generative-dev/src')
```

---

## License

MIT License

## Citation

```bibtex
@software{roy_qsar_2024,
  author = {Roy Lab},
  title = {QSAR Validation Framework},
  year = {2024},
  url = {https://github.com/bhatnira/Roy-QSAR-Generative-dev}
}
```

## Support

Issues: https://github.com/bhatnira/Roy-QSAR-Generative-dev/issues
