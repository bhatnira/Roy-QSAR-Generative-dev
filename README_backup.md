# QSAR Models

**A Purely Modular QSAR Validation Framework with Pitfall Mitigation**

**Version 4.1.0 - Multi-Library Support**

A professional framework of **13 independent, composable modules** for QSAR validation with built-in mitigation for all common pitfalls. Perfect for the low-data regime (< 200 compounds). **Works with ANY ML library** - sklearn, XGBoost, LightGBM, PyTorch, TensorFlow, or custom models!

**We provide the building blocks, you build the workflow.**

---

## ⚠️ IMPORTANT: Correct Usage

**For correct import examples, see:** [`CORRECT_USAGE_GUIDE.md`](CORRECT_USAGE_GUIDE.md)

**Quick reference:**
```python
# ✅ CORRECT
from utils.qsar_utils_no_leakage import QSARDataProcessor
from qsar_validation.splitting_strategies import AdvancedSplitter

# ❌ WRONG (old examples)
from qsar_validation.duplicate_removal import DuplicateRemoval  # This doesn't exist!
```

---

## 🎉 NEW in v4.1.0: Multi-Library Support!

**No more sklearn lock-in!** The framework now works with:
- ✅ **Scikit-learn** (Ridge, RandomForest, SVM, etc.)
- ✅ **XGBoost** (XGBRegressor, XGBClassifier)
- ✅ **LightGBM** (LGBMRegressor, LGBMClassifier)
- ✅ **PyTorch** (custom neural networks)
- ✅ **TensorFlow/Keras** (Sequential, Functional API)
- ✅ **Custom models** (any class with fit/predict interface)

Same consistent API. Maximum flexibility. Your choice!

---

## 🧩 Framework Philosophy

> **"No magic. No automation. Just reliable tools."**
> 
> **"You build the pipeline. We provide the pipes."**

This framework provides **ONLY individual modules** - no all-in-one pipelines, no hidden automation, no forced workflows.

**You control:**
- ✅ Which modules to use
- ✅ When to use them  
- ✅ How to combine them
- ✅ Which ML library to use
- ✅ Your complete workflow

**We provide:**
- ✅ 13 independent, tested modules
- ✅ Support for 5+ ML libraries
- ✅ Data leakage prevention tools
- ✅ QSAR pitfall mitigation tools
- ✅ Validation analysis tools
- ✅ Best practices enforcement

---

## 📦 Available Modules

### Core Modules (Data Splitting & Feature Engineering)

| # | Module | Purpose | When to Use |
|---|--------|---------|-------------|
| 1 | `QSARDataProcessor` | Remove duplicate molecules & canonicalize SMILES | Before any data splitting |
| 2 | **`AdvancedSplitter`** | **Split data (3 strategies!)** | **Choose your splitting strategy** |
| 3 | **`FeatureScaler`** | **Scale features (no leakage!)** | **Fit on train fold only** |
| 4 | **`FeatureSelector`** | **Select features (nested CV!)** | **Within each CV fold** |
| 5 | **`PCATransformer`** | **Reduce dimensions (no leakage!)** | **Fit on train fold only** |

### Pitfall Mitigation Modules

| # | Module | Mitigates | What It Does |
|---|--------|-----------|--------------|
| 6 | **`DatasetQualityAnalyzer`** | Dataset bias, narrow chemical space | Analyzes scaffold diversity, chemical space coverage, activity distribution |
| 7 | **`ModelComplexityController`** ⭐ | Overfitting, excessive complexity | **Multi-library support!** Recommends models, restricts hyperparameters, nested CV (works with sklearn, XGBoost, LightGBM, PyTorch, TensorFlow) |
| 8 | **`PerformanceValidator`** | Improper CV, wrong metrics | Proper CV reporting, comprehensive metrics, baseline comparison |
| 9 | **`ActivityCliffsDetector`** | Activity cliffs, local instability | Identifies structure-activity cliffs, assesses severity |
| 10 | **`UncertaintyEstimator`** | Point predictions, no confidence | Provides uncertainty estimates, confidence intervals, applicability domain |

### Analysis & Reporting Modules

| # | Module | Purpose | When to Use |
|---|--------|---------|-------------|
| 11 | `CrossValidator` | Perform k-fold cross-validation | For model evaluation |
| 12 | `PerformanceMetrics` | Calculate comprehensive metrics | For performance analysis |
| 13 | `DatasetBiasAnalysis` | Detect dataset bias | Legacy - use DatasetQualityAnalyzer |

**Each module is independent. Use any, all, or none. Mix with your own code.**

---

## 🎯 Three Splitting Strategies Available!

The `AdvancedSplitter` module supports **three different splitting strategies** - choose the best one for your data:

| Strategy | When to Use | How It Works | Pros |
|----------|-------------|--------------|------|
| **Scaffold** ⭐ | Most QSAR tasks (RECOMMENDED) | Splits by Bemis-Murcko scaffolds | Industry standard, prevents scaffold leakage |
| **Temporal** 📅 | When you have date/time data | Train on older, test on newer | Simulates realistic deployment |
| **Cluster** 🔗 | Small datasets (< 100 compounds) | Clusters by fingerprint similarity | Good for diverse, small datasets |

```python
# Strategy 1: Scaffold-based (RECOMMENDED)
from qsar_validation.splitting_strategies import ScaffoldSplitter
splitter = ScaffoldSplitter(smiles_col='SMILES')
train_idx, val_idx, test_idx = splitter.split(df, test_size=0.2)

# Strategy 2: Time-based (when you have dates)
from qsar_validation.splitting_strategies import TemporalSplitter
splitter = TemporalSplitter(smiles_col='SMILES', date_col='Date')
train_idx, val_idx, test_idx = splitter.split(df, test_size=0.2)

# Strategy 3: Leave-cluster-out (for small datasets)
from qsar_validation.splitting_strategies import ClusterSplitter
splitter = ClusterSplitter(smiles_col='SMILES', n_clusters=5)
train_idx, val_idx, test_idx = splitter.split(df, test_size=0.2)
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/bhatnira/Roy-QSAR-Generative-dev.git
cd Roy-QSAR-Generative-dev

# Install core dependencies
pip install -r requirements.txt

# Optional: Install ML libraries you want to use
pip install scikit-learn  # For sklearn models
pip install xgboost       # For XGBoost models
pip install lightgbm      # For LightGBM models
pip install torch         # For PyTorch models
pip install tensorflow    # For TensorFlow/Keras models
```

### Minimal Example (3 Modules)

```python
# ✅ CORRECT IMPORTS
from utils.qsar_utils_no_leakage import QSARDataProcessor
from qsar_validation.splitting_strategies import AdvancedSplitter
from qsar_validation.metrics import PerformanceMetricsCalculator
import pandas as pd

# Load your data
df = pd.read_csv('my_data.csv')

# Module 1: Clean data (remove duplicates)
processor = QSARDataProcessor(smiles_col='SMILES', target_col='Activity')
df = processor.canonicalize_smiles(df)
df = processor.remove_duplicates(df, strategy='average')

# Module 2: Split by scaffolds (prevents data leakage!)
splitter = AdvancedSplitter()
splits = splitter.scaffold_split(
    df, 
    smiles_col='SMILES', 
    target_col='Activity',
    test_size=0.2
)
train_idx, test_idx = splits['train_idx'], splits['test_idx']

# YOUR CODE: Featurization, model training, predictions
# (You control this part completely)

# Module 3: Calculate metrics
metrics_calc = PerformanceMetricsCalculator()
results = metrics_calc.calculate_all_metrics(y_true, y_pred)
print(results)
```

### Multi-Library Example

```python
from qsar_validation.model_complexity_control import ModelComplexityController

# Create controller
controller = ModelComplexityController(
    n_samples=X_train.shape[0],
    n_features=X_train.shape[1]
)

# Get recommendations for ALL libraries
recommendations = controller.recommend_models()
print(f"Recommended sklearn models: {recommendations['sklearn']}")
print(f"Recommended XGBoost models: {recommendations['xgboost']}")
print(f"Recommended LightGBM models: {recommendations['lightgbm']}")

# Example 1: Use with sklearn
from sklearn.linear_model import Ridge
model = Ridge()
param_grid = controller.get_safe_param_grid('ridge', library='sklearn')
results = controller.nested_cv(X_train, y_train, model=model, 
                               param_grid=param_grid, library='sklearn')

# Example 2: Use with XGBoost
import xgboost as xgb
model = xgb.XGBRegressor()
param_grid = controller.get_safe_param_grid('xgboost', library='xgboost')
results = controller.nested_cv(X_train, y_train, model=model,
                               param_grid=param_grid, library='xgboost')

# Example 3: Use with LightGBM
import lightgbm as lgb
model = lgb.LGBMRegressor()
param_grid = controller.get_safe_param_grid('lightgbm', library='lightgbm')
results = controller.nested_cv(X_train, y_train, model=model,
                               param_grid=param_grid, library='lightgbm')
```

### Feature Engineering Example (Proper CV - No Leakage!)

```python
from qsar_validation.feature_scaling import FeatureScaler
from qsar_validation.feature_selection import FeatureSelector
from qsar_validation.pca_module import PCATransformer
from sklearn.model_selection import KFold

# CRITICAL: All feature engineering happens WITHIN each CV fold!
for train_idx, val_idx in KFold(n_splits=5).split(X_train):
    X_train_fold = X_train[train_idx]
    X_val_fold = X_train[val_idx]
    
    # Fit scaler on train fold only
    scaler = FeatureScaler(method='standard')
    scaler.fit(X_train_fold)
    X_train_scaled = scaler.transform(X_train_fold)
    X_val_scaled = scaler.transform(X_val_fold)
    
    # Select features on train fold only
    selector = FeatureSelector(method='univariate', n_features=50)
    selector.fit(X_train_scaled, y_train[train_idx])
    X_train_selected = selector.transform(X_train_scaled)
    X_val_selected = selector.transform(X_val_scaled)
    
    # Fit PCA on train fold only
    pca = PCATransformer(n_components=0.95)
    pca.fit(X_train_selected)
    X_train_pca = pca.transform(X_train_selected)
    X_val_pca = pca.transform(X_val_selected)
    
    # Train model on processed features
    model.fit(X_train_pca, y_train[train_idx])
    score = model.score(X_val_pca, y_train[val_idx])
```

**⚠️ CRITICAL:** Feature scaling, selection, and PCA must be fitted **WITHIN** each CV fold to prevent data leakage!

### QSAR Pitfalls Mitigation Example

```python
from qsar_validation.dataset_quality_analysis import DatasetQualityAnalyzer
from qsar_validation.model_complexity_control import ModelComplexityController
from qsar_validation.performance_validation import PerformanceValidator

# 1. Analyze dataset quality
analyzer = DatasetQualityAnalyzer(smiles_col='SMILES', activity_col='pIC50')
quality_report = analyzer.analyze(df)

# 2. Get model recommendations based on dataset size
controller = ModelComplexityController(
    n_samples=X_train.shape[0],
    n_features=X_train.shape[1]
)
recommendations = controller.recommend_models()

# 3. Run proper validation with controls
validator = PerformanceValidator()

# Cross-validation with proper reporting
cv_results = validator.cross_validate_properly(X_train, y_train, model)

# Y-randomization test (negative control)
random_test = validator.y_randomization_test(X_train, y_train, model)

# Baseline comparison
baseline_comparison = validator.compare_to_baseline(y_test, y_pred)
```

---

## 🌟 Multi-Library Support Details

### Universal Model Interface

The framework provides a **universal ModelWrapper** that works with ANY ML library:

```python
from qsar_validation.model_complexity_control import ModelWrapper

# Works with sklearn
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
wrapped = ModelWrapper(model)

# Works with XGBoost
import xgboost as xgb
model = xgb.XGBRegressor()
wrapped = ModelWrapper(model)

# Works with LightGBM
import lightgbm as lgb
model = lgb.LGBMRegressor()
wrapped = ModelWrapper(model)

# Works with PyTorch
import torch.nn as nn
model = MyPyTorchModel()
wrapped = ModelWrapper(model)

# Works with TensorFlow
from tensorflow import keras
model = keras.Sequential([...])
wrapped = ModelWrapper(model)

# Unified interface for all
wrapped.fit(X_train, y_train)
predictions = wrapped.predict(X_test)
```

### Library-Specific Recommendations

The `ModelComplexityController` provides **library-specific recommendations** based on your dataset size:

```python
controller = ModelComplexityController(n_samples=100, n_features=500)
recommendations = controller.recommend_models()

# For small datasets (100 samples, 500 features):
# recommendations['sklearn'] = ['Ridge', 'Lasso', 'ElasticNet']
# recommendations['xgboost'] = ['XGBRegressor (max_depth≤3)']
# recommendations['lightgbm'] = ['LGBMRegressor (num_leaves≤15)']
# recommendations['pytorch'] = ['Small MLP (1-2 hidden layers)']
# recommendations['tensorflow'] = ['Keras Sequential (shallow)']
# recommendations['avoid'] = ['Deep neural networks', 'RandomForest with >100 trees']
```

### Library-Specific Safe Parameters

Get **safe hyperparameter ranges** automatically adapted to your dataset size:

```python
# Sklearn Ridge - simple regularization range
param_grid = controller.get_safe_param_grid('ridge', library='sklearn')
# Returns: {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}

# XGBoost - restricted depth for small datasets
param_grid = controller.get_safe_param_grid('xgboost', library='xgboost')
# Returns: {'max_depth': [2, 3], 'learning_rate': [0.1, 0.01], 'n_estimators': [50, 100]}

# LightGBM - restricted leaves for small datasets
param_grid = controller.get_safe_param_grid('lightgbm', library='lightgbm')
# Returns: {'num_leaves': [7, 15], 'learning_rate': [0.1, 0.01], 'n_estimators': [50, 100]}

# PyTorch - small architectures for small datasets
param_grid = controller.get_safe_param_grid('pytorch_mlp', library='pytorch')
# Returns: {'hidden_sizes': [[32], [64]], 'learning_rate': [0.01, 0.001], 'dropout': [0.2, 0.3]}
```

### Universal Nested Cross-Validation

Run nested CV with **ANY model from ANY library**:

```python
# Works with sklearn
from sklearn.linear_model import Ridge
model = Ridge()
param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0]}
results = controller.nested_cv(X, y, model=model, param_grid=param_grid, 
                               library='sklearn', outer_cv=5, inner_cv=3)

# Works with XGBoost
import xgboost as xgb
model = xgb.XGBRegressor()
param_grid = {'max_depth': [3, 5], 'learning_rate': [0.1, 0.01]}
results = controller.nested_cv(X, y, model=model, param_grid=param_grid,
                               library='xgboost', outer_cv=5, inner_cv=3)

# Works with custom models
class MyCustomModel:
    def fit(self, X, y): ...
    def predict(self, X): ...

model = MyCustomModel()
param_grid = {'my_param': [1, 2, 3]}
results = controller.nested_cv(X, y, model=model, param_grid=param_grid,
                               library='custom', outer_cv=5, inner_cv=3)

# All return the same format
print(f"R²: {results['mean_r2']:.4f} ± {results['std_r2']:.4f}")
print(f"RMSE: {results['mean_rmse']:.4f} ± {results['std_rmse']:.4f}")
```

### Complete Multi-Library Example

See [`examples/multi_library_examples.py`](examples/multi_library_examples.py) for complete working examples with all libraries!

---

## 🛡️ Data Leakage Prevention

**Six modules** work together to prevent all types of data leakage:

### 1️⃣ DuplicateRemoval
**Prevents:** Duplicates appearing in both train and test sets

```python
from qsar_validation.duplicate_removal import DuplicateRemoval

remover = DuplicateRemoval(smiles_col='SMILES')
df = remover.remove_duplicates(df, strategy='average')
```

### 2️⃣ AdvancedSplitter
**Prevents:** Similar molecules in train and test sets

```python
from qsar_validation.splitting_strategies import ScaffoldSplitter

splitter = ScaffoldSplitter(smiles_col='SMILES')
train_idx, val_idx, test_idx = splitter.split(df, test_size=0.2)

# Verify no scaffold overlap
overlap = splitter.check_scaffold_overlap(train_idx, test_idx, df)
print(f"Scaffold overlap: {overlap}")  # Should be 0
```

### 3️⃣ FeatureScaler
**Prevents:** Using test set statistics to scale features
**CRITICAL:** Fit on train fold only!

```python
from qsar_validation.feature_scaling import FeatureScaler

# CORRECT: Within CV fold
for train_idx, val_idx in cv_folds:
    scaler = FeatureScaler(method='standard')
    scaler.fit(X_train[train_idx])  # ✓ Fit on train fold only
    X_train_scaled = scaler.transform(X_train[train_idx])
    X_val_scaled = scaler.transform(X_train[val_idx])
```

### 4️⃣ FeatureSelector
**Prevents:** Using validation data to select features
**CRITICAL:** Use nested CV!

```python
from qsar_validation.feature_selection import FeatureSelector

# CORRECT: Select features within CV fold
for train_idx, val_idx in cv_folds:
    selector = FeatureSelector(method='univariate', n_features=50)
    selector.fit(X_train[train_idx], y_train[train_idx])
    X_train_selected = selector.transform(X_train[train_idx])
    X_val_selected = selector.transform(X_train[val_idx])
```

### 5️⃣ PCATransformer
**Prevents:** Learning PCA from validation/test data
**CRITICAL:** Fit on train fold only!

```python
from qsar_validation.pca_module import PCATransformer

# CORRECT: Fit PCA within CV fold
for train_idx, val_idx in cv_folds:
    pca = PCATransformer(n_components=0.95)
    pca.fit(X_train[train_idx])  # ✓ Fit on train fold only
    X_train_pca = pca.transform(X_train[train_idx])
    X_val_pca = pca.transform(X_train[val_idx])
```

**⚠️ THE GOLDEN RULE:**
- **Scaling:** Fit on train fold only
- **Feature Selection:** Use nested CV
- **PCA:** Fit on train fold only
- **Never fit ANY transformation on validation or test data!**

---

## 📊 QSAR Pitfalls Covered

This framework helps mitigate **13 common QSAR pitfalls**:

| Pitfall | Mitigation Module | How It Helps |
|---------|------------------|--------------|
| 1. Dataset bias | `DatasetQualityAnalyzer` | Analyzes scaffold diversity, chemical space |
| 2. Data leakage | `AdvancedSplitter`, `FeatureScaler`, etc. | Proper splitting, no test data in training |
| 3. Overfitting | `ModelComplexityController` | Recommends appropriate model complexity |
| 4. Improper CV | `PerformanceValidator` | Proper nested CV, mean±std reporting |
| 5. Wrong metrics | `PerformanceValidator` | Comprehensive metrics, not just R² |
| 6. Activity cliffs | `ActivityCliffsDetector` | Identifies cliffs, warns about instability |
| 7. Narrow applicability | `UncertaintyEstimator` | Defines applicability domain |
| 8. No uncertainty | `UncertaintyEstimator` | Provides prediction confidence |
| 9. Cherry-picking | `PerformanceValidator` | Y-randomization, baseline comparison |
| 10. Small datasets | `ModelComplexityController` | Restricts model complexity for small N |
| 11. High dimensional | `FeatureSelector`, `PCATransformer` | Feature selection, dimensionality reduction |
| 12. Imbalanced data | `DatasetQualityAnalyzer` | Detects activity distribution issues |
| 13. External validation | `AdvancedSplitter` | Proper train/test/external splitting |

---

## 💡 Best Practices

### ✅ DO:
1. **Always remove duplicates BEFORE splitting**
2. **Use scaffold splitting for realistic estimates**
3. **Fit scalers/selectors on training data only**
4. **Use nested CV for hyperparameter tuning**
5. **Check dataset quality before modeling**
6. **Use multiple ML libraries and compare**
7. **Report mean ± std for CV metrics**
8. **Include y-randomization tests**
9. **Check for activity cliffs**
10. **Estimate prediction uncertainty**

### ❌ DON'T:
1. ~~Remove duplicates after splitting~~
2. ~~Use random splitting for QSAR~~
3. ~~Fit scaler/selector on all data before CV~~
4. ~~Use GridSearchCV without nested CV~~
5. ~~Skip dataset quality analysis~~
6. ~~Stick to one ML library~~
7. ~~Report only best CV score~~
8. ~~Skip negative controls~~
9. ~~Ignore activity cliffs~~
10. ~~Give point predictions without uncertainty~~

---

## 📚 Examples

### Complete Examples
- **Multi-library examples**: [`examples/multi_library_examples.py`](examples/multi_library_examples.py)
- **Splitting strategies**: [`examples/splitting_strategies_examples.py`](examples/splitting_strategies_examples.py)
- **Feature engineering**: [`examples/feature_engineering_examples.py`](examples/feature_engineering_examples.py)
- **Pitfall mitigation**: [`examples/pitfall_mitigation_examples.py`](examples/pitfall_mitigation_examples.py)

### Usage Patterns

**Pattern 1: Minimal (Leakage Prevention Only)**
```python
from qsar_validation.duplicate_removal import DuplicateRemoval
from qsar_validation.splitting_strategies import ScaffoldSplitter
from qsar_validation.feature_scaling import FeatureScaler

# 1. Remove duplicates
remover = DuplicateRemoval(smiles_col='SMILES')
df = remover.remove_duplicates(df)

# 2. Scaffold split
splitter = ScaffoldSplitter(smiles_col='SMILES')
train_idx, _, test_idx = splitter.split(df)

# 3. Scale properly
scaler = FeatureScaler(method='standard')
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Pattern 2: Complete Validation**
```python
# Use all modules for comprehensive validation
from qsar_validation.dataset_quality_analysis import DatasetQualityAnalyzer
from qsar_validation.model_complexity_control import ModelComplexityController
from qsar_validation.performance_validation import PerformanceValidator
from qsar_validation.activity_cliffs_detection import ActivityCliffsDetector
from qsar_validation.uncertainty_estimation import UncertaintyEstimator

# 1. Dataset quality
analyzer = DatasetQualityAnalyzer(smiles_col='SMILES', activity_col='pIC50')
quality = analyzer.analyze(df)

# 2. Model complexity
controller = ModelComplexityController(n_samples=len(X_train), n_features=X_train.shape[1])
recommendations = controller.recommend_models()

# 3. Proper validation
validator = PerformanceValidator()
cv_results = validator.cross_validate_properly(X_train, y_train, model)

# 4. Activity cliffs
cliff_detector = ActivityCliffsDetector(smiles_col='SMILES', activity_col='pIC50')
cliffs = cliff_detector.detect_cliffs(df)

# 5. Uncertainty estimation
uncertainty = UncertaintyEstimator()
pred_with_uncertainty = uncertainty.predict_with_uncertainty(model, X_test)
```

---

## � Example Notebooks

**Want to see the framework in action?** Check out our comprehensive Jupyter notebooks in the [`notebooks/`](notebooks/) folder!

### Available Notebooks:

1. **📚 DATA_LEAKAGE_FIX_EXAMPLE.ipynb**
   - Complete tutorial on data leakage prevention
   - Step-by-step guide with explanations
   - Perfect starting point for beginners

2. **🧪 Model_1** - Circular Fingerprints + H2O AutoML
   - Morgan fingerprints (1024 bits)
   - H2O AutoML with model interpretation
   - Comprehensive SHAP analysis

3. **🤖 Model_2** - ChEBERTa Embeddings + Linear Regression
   - Transformer-based molecular embeddings
   - Linear regression with proper validation
   - Demonstrates embeddings workflow

4. **📊 Model_3** - RDKit Features + H2O AutoML
   - RDKit molecular descriptors
   - H2O AutoML leaderboard
   - Feature importance analysis

5. **🎯 Model_4** - Gaussian Process + Bayesian Optimization
   - Morgan fingerprints with GP regression
   - Bayesian hyperparameter optimization
   - Uncertainty quantification

### Quick Start with Notebooks:

```bash
# Clone the repository
git clone https://github.com/bhatnira/Roy-QSAR-Generative-dev.git
cd Roy-QSAR-Generative-dev

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
cd notebooks
jupyter notebook
```

**The notebooks automatically detect the framework path!** Just open and run - no configuration needed.

See [`notebooks/README.md`](notebooks/README.md) for detailed instructions.

---

## �🔧 Requirements

### Core Requirements
- Python ≥ 3.8
- pandas
- numpy
- rdkit
- scipy

### ML Libraries (Optional - Install What You Need)
- scikit-learn (for sklearn models)
- xgboost (for XGBoost models)
- lightgbm (for LightGBM models)
- torch (for PyTorch models)
- tensorflow (for TensorFlow/Keras models)

```bash
# Install core only
pip install numpy pandas scipy rdkit

# Install with sklearn
pip install numpy pandas scipy rdkit scikit-learn

# Install with all ML libraries
pip install numpy pandas scipy rdkit scikit-learn xgboost lightgbm torch tensorflow
```

---

## 📞 Support & Contributing

- **Questions**: Open an issue on [GitHub](https://github.com/bhatnira/Roy-QSAR-Generative-dev/issues)
- **Examples**: See [`examples/`](examples/) folder
- **Contributing**: Contributions welcome! Each module is independent, making it easy to add new features.

---

## 📄 License

[Add your license here]

---

## 🌟 Framework Principles

> **"No magic. No automation. Just reliable tools."**
> 
> **"You build the pipeline. We provide the pipes."**
> 
> **"Your workflow, your rules, your ML library, our modules."**

We believe in:
- **Modularity** over monoliths
- **Flexibility** over convenience
- **Transparency** over magic
- **Choice** over lock-in

**You are the architect. We provide the building blocks.** 

---

## 🎓 Version History

- **v4.1.0** (Current): Multi-library support (sklearn, XGBoost, LightGBM, PyTorch, TensorFlow)
- **v4.0.0**: QSAR pitfalls mitigation modules
- **v3.0.0**: Feature engineering with leakage prevention
- **v2.0.0**: Three splitting strategies (scaffold, temporal, cluster)
- **v1.0.0**: Initial purely modular framework

---

**Remember: Each module is independent. Use what you need, ignore the rest! 🎯**
