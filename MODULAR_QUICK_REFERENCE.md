# Modular Components - Quick Reference Card

## 🎯 Choose Your Approach

### Option A: Full Pipeline (All-in-One)
```python
from qsar_validation import ModelAgnosticQSARPipeline

pipeline = ModelAgnosticQSARPipeline(featurizer=my_featurizer, model=my_model)
results = pipeline.fit_predict_validate(df)
```
**Use when:** You want everything automated with sensible defaults

---

### Option B: Individual Modules (Pick & Choose)
```python
# Import only what you need
from qsar_validation.duplicate_removal import DuplicateRemoval
from qsar_validation.scaffold_splitting import ScaffoldSplitter
# ... use them independently
```
**Use when:** You need custom workflow or specific components

---

## 📦 Available Modules

| # | Module | Purpose | Key Method |
|---|--------|---------|------------|
| 1 | `DuplicateRemoval` | Remove duplicate molecules | `remove_duplicates(df)` |
| 2 | `ScaffoldSplitter` | Split by scaffolds | `split(df, test_size=0.2)` |
| 3 | `FeatureScaler` | Scale features properly | `fit_transform(X_train)` |
| 4 | `CrossValidator` | Cross-validation | `cross_validate(model, X, y)` |
| 5 | `PerformanceMetrics` | Calculate metrics | `calculate_all_metrics(y_true, y_pred)` |
| 6 | `DatasetBiasAnalysis` | Detect bias | `analyze_bias(X_train, X_test, ...)` |
| 7 | `ModelComplexityAnalysis` | Analyze complexity | `analyze_complexity(model, X, y)` |

---

## 🚀 Common Use Cases

### Use Case 1: Just Remove Duplicates
```python
from qsar_validation.duplicate_removal import DuplicateRemoval

remover = DuplicateRemoval(smiles_col='SMILES')
clean_df = remover.remove_duplicates(df, strategy='average')
```

### Use Case 2: Just Scaffold Split
```python
from qsar_validation.scaffold_splitting import ScaffoldSplitter

splitter = ScaffoldSplitter(smiles_col='SMILES')
train_idx, val_idx, test_idx = splitter.split(df, test_size=0.2)
```

### Use Case 3: Just Calculate Metrics
```python
from qsar_validation.performance_metrics import PerformanceMetrics

metrics = PerformanceMetrics()
results = metrics.calculate_all_metrics(y_true, y_pred, set_name='Test')
```

### Use Case 4: Data Leakage Prevention Only (3 modules)
```python
from qsar_validation.duplicate_removal import DuplicateRemoval
from qsar_validation.scaffold_splitting import ScaffoldSplitter
from qsar_validation.feature_scaling import FeatureScaler

# 1. Remove duplicates BEFORE splitting
remover = DuplicateRemoval(smiles_col='SMILES')
df = remover.remove_duplicates(df)

# 2. Scaffold split
splitter = ScaffoldSplitter(smiles_col='SMILES')
train_idx, _, test_idx = splitter.split(df, test_size=0.2)

# 3. Scale using train stats only
scaler = FeatureScaler(method='standard')
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Now do your own modeling...
```

### Use Case 5: Custom Full Workflow
```python
from qsar_validation.duplicate_removal import DuplicateRemoval
from qsar_validation.scaffold_splitting import ScaffoldSplitter
from qsar_validation.feature_scaling import FeatureScaler
from qsar_validation.cross_validation import CrossValidator
from qsar_validation.performance_metrics import PerformanceMetrics

# 1. Clean data
remover = DuplicateRemoval(smiles_col='SMILES')
df = remover.remove_duplicates(df)

# 2. Split
splitter = ScaffoldSplitter(smiles_col='SMILES')
train_idx, _, test_idx = splitter.split(df, test_size=0.2)

# 3. Features (your code)
X_train, X_test, y_train, y_test = your_featurizer(df, train_idx, test_idx)

# 4. Scale
scaler = FeatureScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Cross-validate
cv = CrossValidator(n_folds=5)
cv_scores = cv.cross_validate(model, X_train, y_train)

# 6. Train & predict (your code)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 7. Metrics
metrics = PerformanceMetrics()
results = metrics.calculate_all_metrics(y_test, y_pred)
```

---

## 📋 Module Details

### 1️⃣ DuplicateRemoval

**Import:**
```python
from qsar_validation.duplicate_removal import DuplicateRemoval
```

**Methods:**
- `remove_duplicates(df, strategy='first')` - Remove duplicates
  - `strategy='first'`: Keep first occurrence
  - `strategy='average'`: Average activities
- `check_duplicates(df)` - Check if duplicates exist

**Example:**
```python
remover = DuplicateRemoval(smiles_col='SMILES')
clean_df = remover.remove_duplicates(df, strategy='average')
```

---

### 2️⃣ ScaffoldSplitter

**Import:**
```python
from qsar_validation.scaffold_splitting import ScaffoldSplitter
```

**Methods:**
- `split(df, test_size=0.2, val_size=0.1)` - Split by scaffolds
- `get_scaffold(smiles)` - Get Bemis-Murcko scaffold
- `check_scaffold_overlap(train_idx, test_idx, df)` - Check overlap

**Example:**
```python
splitter = ScaffoldSplitter(smiles_col='SMILES')
train_idx, val_idx, test_idx = splitter.split(df, test_size=0.2, val_size=0.1)
overlap = splitter.check_scaffold_overlap(train_idx, test_idx, df)
```

---

### 3️⃣ FeatureScaler

**Import:**
```python
from qsar_validation.feature_scaling import FeatureScaler
```

**Methods:**
- `fit(X)` - Fit scaler on data
- `transform(X)` - Transform data
- `fit_transform(X)` - Fit and transform

**Options:**
- `method='standard'` - StandardScaler (mean=0, std=1)
- `method='minmax'` - MinMaxScaler (range [0,1])
- `method='robust'` - RobustScaler (uses median/IQR)

**Example:**
```python
scaler = FeatureScaler(method='standard')
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

### 4️⃣ CrossValidator

**Import:**
```python
from qsar_validation.cross_validation import CrossValidator
```

**Methods:**
- `cross_validate(model, X, y, scoring='r2')` - Perform CV
- `get_folds(X)` - Get fold indices

**Example:**
```python
cv = CrossValidator(n_folds=5, random_state=42)
cv_scores = cv.cross_validate(model, X, y, scoring='r2')
print(f"CV: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
```

---

### 5️⃣ PerformanceMetrics

**Import:**
```python
from qsar_validation.performance_metrics import PerformanceMetrics
```

**Methods:**
- `calculate_all_metrics(y_true, y_pred, set_name='Test')` - All metrics
- `calculate_r2(y_true, y_pred)` - R² score
- `calculate_rmse(y_true, y_pred)` - RMSE
- `calculate_mae(y_true, y_pred)` - MAE
- `calculate_pearson_r(y_true, y_pred)` - Pearson correlation
- `calculate_spearman_rho(y_true, y_pred)` - Spearman correlation

**Example:**
```python
metrics = PerformanceMetrics()
results = metrics.calculate_all_metrics(y_true, y_pred, set_name='Test')
# Returns: {'Test_R2': 0.85, 'Test_RMSE': 0.5, 'Test_MAE': 0.4, ...}
```

---

### 6️⃣ DatasetBiasAnalysis

**Import:**
```python
from qsar_validation.dataset_bias_analysis import DatasetBiasAnalysis
```

**Methods:**
- `analyze_bias(X_train, X_test, y_train, y_test)` - Full bias analysis
- `check_activity_coverage(y_train, y_test)` - Activity range coverage

**Example:**
```python
analyzer = DatasetBiasAnalysis()
report = analyzer.analyze_bias(X_train, X_test, y_train, y_test)
# Returns: {'activity_range_train': [...], 'warnings': [...], ...}
```

---

### 7️⃣ ModelComplexityAnalysis

**Import:**
```python
from qsar_validation.model_complexity_analysis import ModelComplexityAnalysis
```

**Methods:**
- `analyze_complexity(model, X, y)` - Full complexity analysis
- `check_model_complexity(model, X, y)` - Get warnings

**Example:**
```python
analyzer = ModelComplexityAnalysis()
report = analyzer.analyze_complexity(model, X, y)
# Returns: {'n_samples': 100, 'sample_to_feature_ratio': 2.0, 'warnings': [...]}
```

---

## 🎨 Mix & Match Examples

### Minimal (Just Leakage Prevention)
```python
from qsar_validation.duplicate_removal import DuplicateRemoval
from qsar_validation.scaffold_splitting import ScaffoldSplitter

# Clean + Split
remover = DuplicateRemoval(smiles_col='SMILES')
df = remover.remove_duplicates(df)

splitter = ScaffoldSplitter(smiles_col='SMILES')
train_idx, _, test_idx = splitter.split(df)

# Your modeling...
```

### Standard (Leakage Prevention + Validation)
```python
from qsar_validation.duplicate_removal import DuplicateRemoval
from qsar_validation.scaffold_splitting import ScaffoldSplitter
from qsar_validation.feature_scaling import FeatureScaler
from qsar_validation.performance_metrics import PerformanceMetrics

# Full workflow with validation
```

### Complete (Everything)
```python
from qsar_validation import *

# Use all modules as needed
```

---

## 💡 Pro Tips

1. **Always remove duplicates BEFORE splitting** to prevent leakage
2. **Always fit scaler on training data only** to prevent leakage
3. **Use scaffold splitting** for better generalization estimation
4. **Check for bias** to ensure test set is representative
5. **Analyze complexity** to detect overfitting risk

---

## 📚 Documentation

- **Full Guide**: `MODULAR_USAGE_GUIDE.md`
- **Examples**: `examples/modular_examples.py`
- **API Docs**: Each module has detailed docstrings

---

## 🔗 Quick Links

- Full Pipeline: `MODEL_AGNOSTIC_README.md`
- Leakage Guide: `DATA_LEAKAGE_PREVENTION.md`
- Demonstration: `COMPREHENSIVE_DEMONSTRATION_REPORT.md`

---

## ✨ Remember

**You have complete freedom!**

- Use **full pipeline** for simplicity
- Use **individual modules** for control
- **Mix and match** as needed

Every module is independent. Pick what you need! 🚀
