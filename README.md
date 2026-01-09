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

# Scaffold split (recommended - prevents data leakage)
splitter = AdvancedSplitter(smiles_col='SMILES', strategy='scaffold')
train_idx, test_idx = splitter.split(df, test_size=0.2)

# Other strategies: 'temporal' (time-based), 'cluster' (similarity-based)
# splitter = AdvancedSplitter(strategy='temporal', date_col='Date')
# splitter = AdvancedSplitter(strategy='cluster', n_clusters=5)
```

### 3. Feature Scaling

```python
from qsar_validation.feature_scaling import FeatureScaler

# Standard scaling (fit on train only - prevents leakage)
scaler = FeatureScaler(method='standard')  # or 'minmax', 'robust'
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### 4. Feature Selection

```python
from qsar_validation.feature_selection import FeatureSelector

selector = FeatureSelector()

# Remove low-variance features
X_selected = selector.variance_threshold(X_train, threshold=0.01)
X_test_selected = selector.transform(X_test)

# Other methods: correlation_filter, model_based_selection, univariate_selection
```

### 5. Performance Validation

```python
from qsar_validation.performance_validation import PerformanceValidator

validator = PerformanceValidator()

# Calculate metrics (R², RMSE, MAE, Spearman)
metrics = validator.calculate_comprehensive_metrics(y_test, y_pred)
print(f"R²: {metrics['r2']:.3f}, RMSE: {metrics['rmse']:.3f}")

# Y-randomization test (negative control - should give ~0)
random_results = validator.y_randomization_test(X_train, y_train, model, n_iterations=100)
```

### 6. Dataset Quality Analysis

```python
from qsar_validation.dataset_quality_analysis import DatasetQualityAnalyzer

analyzer = DatasetQualityAnalyzer(smiles_col='SMILES', activity_col='pIC50')
report = analyzer.analyze(df)

print(f"Scaffolds: {report['n_scaffolds']}, Diversity: {report['diversity_ratio']:.2f}")
print(f"Warnings: {report['warnings']}")
```

### 7. Activity Cliffs Detection

```python
from qsar_validation.activity_cliffs_detection import ActivityCliffsDetector

# Set thresholds in constructor
detector = ActivityCliffsDetector(
    smiles_col='SMILES', 
    activity_col='pIC50',
    similarity_threshold=0.85,
    activity_threshold=1.0
)
cliffs = detector.detect_cliffs(df)
print(f"Activity cliffs found: {len(cliffs)}")
```

### 8. Uncertainty Estimation

```python
from qsar_validation.uncertainty_estimation import UncertaintyEstimator

estimator = UncertaintyEstimator(method='both')
estimator.fit(X_train, y_train, model)  # Fit on training data first
results = estimator.predict_with_uncertainty(X_test)
print(f"Mean uncertainty: {results['uncertainty'].mean():.3f}")
```

### 9. Model Complexity Control

```python
from qsar_validation.model_complexity_control import ModelComplexityController

controller = ModelComplexityController(n_samples=len(X_train), n_features=X_train.shape[1])
recommendations = controller.recommend_models()
param_grid = controller.get_safe_param_grid('random_forest', library='sklearn')
print("Recommended:", recommendations)
```

### 10. PCA (Dimensionality Reduction)

```python
from qsar_validation.pca_module import PCATransformer

pca = PCATransformer(n_components=0.95)  # Retain 95% variance (or use fixed number)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
print(f"Features: {X_train.shape[1]} → {pca.n_components_}")
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

## Advanced Validation Options

### (a) Nested Cross-Validation (Gold Standard for Low-Data QSAR)

Prevents implicit information leakage through hyperparameter tuning:

```python
from qsar_validation.model_complexity_control import ModelComplexityController
from sklearn.ensemble import RandomForestRegressor

controller = ModelComplexityController(n_samples=len(X_train), n_features=X_train.shape[1])

# Inner loop: hyperparameter tuning
# Outer loop: unbiased performance estimation
param_grid = controller.get_safe_param_grid('random_forest', library='sklearn')
results = controller.nested_cv(
    X_train, y_train,
    model=RandomForestRegressor(),
    param_grid=param_grid,
    outer_cv=5,  # Outer loop for performance
    inner_cv=3   # Inner loop for tuning
)
print(f"Nested CV R²: {results['test_score_mean']:.3f} ± {results['test_score_std']:.3f}")
```

**Key principle:** Any operation informed by labels (feature selection, scaling fit, hyperparameter tuning) must occur inside the training fold only.

### (b) Scaffold-Based or Similarity-Aware Splitting

Mimics real-world extrapolation by splitting compounds based on structure:

```python
from qsar_validation.splitting_strategies import AdvancedSplitter

# Option 1: Bemis-Murcko scaffolds (recommended)
splitter = AdvancedSplitter(strategy='scaffold')
train_idx, test_idx = splitter.split(df, test_size=0.2)

# Option 2: Clustering in fingerprint space
splitter = AdvancedSplitter(strategy='cluster', n_clusters=5)
train_idx, test_idx = splitter.split(df, test_size=0.2)
```

**Why:** Performance is lower but more realistic - tests generalization to new scaffolds.

### (c) Y-Randomization (Response Permutation)

Negative control test - detects chance correlations:

```python
from qsar_validation.performance_validation import PerformanceValidator

validator = PerformanceValidator()

# Shuffle activity labels - model should collapse to random performance
random_results = validator.y_randomization_test(
    X_train, y_train,
    model=model,
    n_iterations=100
)

print(f"Real model R²: {real_r2:.3f}")
print(f"Random R²: {random_results['mean_r2']:.3f} (should be ~0)")

if random_results['mean_r2'] > 0.3:
    print("⚠️ Warning: Model may be overfitting or finding chance correlations!")
```

**Expected:** Random R² should be near 0 - if high, model is capturing noise.

---

## Advanced Validation Options

### Option A: Nested (Double) Cross-Validation

**Gold standard for low-data QSAR** - Prevents implicit information leakage during hyperparameter tuning.

```python
from qsar_validation.model_complexity_control import ModelComplexityController

# Nested CV: Outer loop = performance, Inner loop = hyperparameter tuning
controller = ModelComplexityController(n_samples=len(X_train), n_features=X_train.shape[1])

param_grid = {
    'max_depth': [3, 5, 7],
    'n_estimators': [50, 100, 150]
}

results = controller.nested_cv(
    X_train, y_train,
    model_type='random_forest',
    param_grid=param_grid,
    outer_cv=5,  # Outer loop: unbiased performance
    inner_cv=3   # Inner loop: hyperparameter tuning
)
print(f"Nested CV R²: {results['test_score_mean']:.3f} ± {results['test_score_std']:.3f}")
```

**Why use it?** Any operation informed by labels (feature selection, hyperparameter tuning) must occur inside the training fold only. Nested CV ensures this.

### Option B: Scaffold-Based Splitting

**Mimics real-world extrapolation** - Compounds split by Bemis-Murcko scaffolds or fingerprint clustering.

```python
from qsar_validation.splitting_strategies import AdvancedSplitter

# Scaffold-based (recommended for diverse datasets)
splitter = AdvancedSplitter(smiles_col='SMILES', strategy='scaffold')
train_idx, test_idx = splitter.split(df, test_size=0.2)

# Cluster-based (for congeneric series)
splitter = AdvancedSplitter(strategy='cluster', n_clusters=5)
train_idx, test_idx = splitter.split(df, test_size=0.2)
```

**Why use it?** Performance is lower but more realistic - no scaffold overlap between train/test means the model must extrapolate to new chemical space.

### Option C: Y-Randomization (Response Permutation)

**Detects chance correlations** - Activity labels are shuffled; model should collapse to random performance.

```python
from qsar_validation.performance_validation import PerformanceValidator

validator = PerformanceValidator()

# Shuffle labels and check if performance drops to ~0
random_results = validator.y_randomization_test(
    X_train, y_train, 
    model, 
    n_iterations=100
)

print(f"Real model R²: {real_r2:.3f}")
print(f"Random R²: {random_results['mean_r2']:.3f} (should be ~0)")

if random_results['mean_r2'] < 0.3:
    print("✓ Model is not overfitting (random labels give poor performance)")
else:
    print("⚠ Potential overfitting detected!")
```

**Why use it?** If the model achieves good performance even with shuffled labels, it's memorizing patterns rather than learning true structure-activity relationships.

---

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
