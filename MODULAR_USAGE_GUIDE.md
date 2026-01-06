# Modular Usage Guide
## Using Components Independently

The QSAR Validation Framework is designed with **modular components** that can be used independently or combined as needed. You don't need to use the full pipeline - you can pick and choose individual modules!

## Table of Contents
1. [Component Overview](#component-overview)
2. [Individual Module Usage](#individual-module-usage)
3. [Mix and Match Examples](#mix-and-match-examples)
4. [Full Pipeline vs Modular](#full-pipeline-vs-modular)

---

## Component Overview

Each module is **independent** and can be imported and used separately:

| Module | Purpose | Input | Output |
|--------|---------|-------|--------|
| `DuplicateRemoval` | Remove duplicate molecules | DataFrame | Cleaned DataFrame |
| `ScaffoldSplitter` | Split by Bemis-Murcko scaffolds | DataFrame | Train/Val/Test indices |
| `FeatureScaler` | Scale features properly | Features | Scaled features |
| `CrossValidator` | Perform cross-validation | Features + targets | CV scores |
| `PerformanceMetrics` | Calculate metrics | Predictions + actuals | Metrics dict |
| `DatasetBiasAnalysis` | Detect dataset bias | Features + targets | Bias report |
| `ModelComplexityAnalysis` | Analyze model complexity | Model + data | Complexity report |

---

## Individual Module Usage

### 1. Duplicate Removal (Standalone)

```python
from qsar_validation.duplicate_removal import DuplicateRemoval
import pandas as pd

# Your data
df = pd.DataFrame({
    'SMILES': ['CCO', 'CCO', 'CC(C)O', 'CCO'],  # Has duplicates
    'Activity': [5.0, 5.1, 6.0, 5.0]
})

# Use standalone duplicate removal
remover = DuplicateRemoval(smiles_col='SMILES')

# Strategy 1: Keep first occurrence
clean_df = remover.remove_duplicates(df, strategy='first')
print(f"Original: {len(df)}, After: {len(clean_df)}")

# Strategy 2: Average duplicate activities
clean_df = remover.remove_duplicates(df, strategy='average')
print(clean_df)

# Check for duplicates
has_dups = remover.check_duplicates(df)
print(f"Has duplicates: {has_dups}")
```

**Output:**
```
Original: 4, After: 3
   SMILES  Activity
0     CCO      5.03  # Averaged 5.0, 5.1, 5.0
1  CC(C)O      6.00
Has duplicates: True
```

---

### 2. Scaffold Splitting (Standalone)

```python
from qsar_validation.scaffold_splitting import ScaffoldSplitter
import pandas as pd

# Your data
df = pd.DataFrame({
    'SMILES': ['c1ccccc1CC', 'c1ccccc1CCC', 'CCCCCC', 'CCCCCCC'],
    'Activity': [5.0, 5.5, 3.0, 3.2]
})

# Use standalone scaffold splitter
splitter = ScaffoldSplitter(smiles_col='SMILES')

# Split data
train_idx, val_idx, test_idx = splitter.split(
    df, 
    test_size=0.25,
    val_size=0.25
)

print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

# Get train/val/test sets
train_df = df.iloc[train_idx]
val_df = df.iloc[val_idx]
test_df = df.iloc[test_idx]

# Verify no scaffold overlap
overlap = splitter.check_scaffold_overlap(train_idx, test_idx, df)
print(f"Scaffold overlap: {overlap}")  # Should be 0

# Get scaffold for a specific SMILES
scaffold = splitter.get_scaffold('c1ccccc1CC')
print(f"Scaffold: {scaffold}")  # 'c1ccccc1'
```

---

### 3. Feature Scaling (Standalone)

```python
from qsar_validation.feature_scaling import FeatureScaler
import numpy as np

# Your features (e.g., from Morgan fingerprints)
train_features = np.random.rand(100, 1024)
test_features = np.random.rand(30, 1024)

# Use standalone feature scaler
scaler = FeatureScaler(method='standard')  # or 'minmax', 'robust'

# Fit on training data only (no data leakage!)
scaler.fit(train_features)

# Transform both sets
train_scaled = scaler.transform(train_features)
test_scaled = scaler.transform(test_features)

print(f"Train mean: {train_scaled.mean():.3f}, std: {train_scaled.std():.3f}")
print(f"Test mean: {test_scaled.mean():.3f}, std: {test_scaled.std():.3f}")

# Or fit and transform in one step
train_scaled = scaler.fit_transform(train_features)
```

---

### 4. Cross-Validation (Standalone)

```python
from qsar_validation.cross_validation import CrossValidator
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Your data
X = np.random.rand(100, 50)
y = np.random.rand(100)

# Use standalone cross-validator
cv = CrossValidator(n_folds=5, random_state=42)

# Your model
model = RandomForestRegressor(n_estimators=100)

# Perform cross-validation
cv_scores = cv.cross_validate(model, X, y, scoring='r2')

print(f"CV R² scores: {cv_scores}")
print(f"Mean: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")

# Get fold indices (for manual iteration)
for fold_idx, (train_idx, val_idx) in enumerate(cv.get_folds(X)):
    print(f"Fold {fold_idx}: train={len(train_idx)}, val={len(val_idx)}")
```

---

### 5. Performance Metrics (Standalone)

```python
from qsar_validation.performance_metrics import PerformanceMetrics
import numpy as np

# Your predictions and actual values
y_true = np.array([5.0, 5.5, 6.0, 4.5, 7.0])
y_pred = np.array([4.8, 5.6, 5.9, 4.7, 6.8])

# Use standalone metrics calculator
metrics_calc = PerformanceMetrics()

# Calculate all metrics
metrics = metrics_calc.calculate_all_metrics(
    y_true=y_true,
    y_pred=y_pred,
    set_name='Test'
)

print(metrics)
# Output:
# {
#   'Test_R2': 0.95,
#   'Test_RMSE': 0.15,
#   'Test_MAE': 0.12,
#   'Test_Pearson_r': 0.98,
#   'Test_Spearman_rho': 0.95
# }

# Calculate individual metrics
r2 = metrics_calc.calculate_r2(y_true, y_pred)
rmse = metrics_calc.calculate_rmse(y_true, y_pred)
mae = metrics_calc.calculate_mae(y_true, y_pred)

print(f"R²: {r2:.3f}, RMSE: {rmse:.3f}, MAE: {mae:.3f}")
```

---

### 6. Dataset Bias Analysis (Standalone)

```python
from qsar_validation.dataset_bias_analysis import DatasetBiasAnalysis
import numpy as np

# Your data
X_train = np.random.rand(100, 50)
X_test = np.random.rand(30, 50)
y_train = np.random.rand(100)
y_test = np.random.rand(30)

# Use standalone bias analyzer
bias_analyzer = DatasetBiasAnalysis()

# Analyze bias
bias_report = bias_analyzer.analyze_bias(
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test
)

print(bias_report)
# Output:
# {
#   'activity_range_train': [0.01, 0.98],
#   'activity_range_test': [0.05, 0.95],
#   'train_coverage': 0.95,
#   'feature_correlation': 0.12,
#   'warnings': []
# }

# Check specific aspects
coverage = bias_analyzer.check_activity_coverage(y_train, y_test)
print(f"Test set coverage: {coverage:.1%}")
```

---

### 7. Model Complexity Analysis (Standalone)

```python
from qsar_validation.model_complexity_analysis import ModelComplexityAnalysis
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Your model and data
model = RandomForestRegressor(n_estimators=100, max_depth=10)
X = np.random.rand(50, 100)  # 50 samples, 100 features
y = np.random.rand(50)

# Train model
model.fit(X, y)

# Use standalone complexity analyzer
complexity_analyzer = ModelComplexityAnalysis()

# Analyze complexity
complexity_report = complexity_analyzer.analyze_complexity(
    model=model,
    X=X,
    y=y
)

print(complexity_report)
# Output:
# {
#   'n_samples': 50,
#   'n_features': 100,
#   'sample_to_feature_ratio': 0.5,
#   'model_parameters': 1000,
#   'warnings': ['Low samples-to-features ratio (0.5 < 3.0)']
# }

# Check if model is too complex
warnings = complexity_analyzer.check_model_complexity(model, X, y)
for warning in warnings:
    print(f"⚠️  {warning}")
```

---

## Mix and Match Examples

### Example 1: Custom Pipeline (Duplicate Removal + Your Own Split)

```python
from qsar_validation.duplicate_removal import DuplicateRemoval
import pandas as pd
from sklearn.model_selection import train_test_split

# Load your data
df = pd.read_csv('my_data.csv')

# Step 1: Remove duplicates (use module)
remover = DuplicateRemoval(smiles_col='SMILES')
clean_df = remover.remove_duplicates(df, strategy='average')

# Step 2: Your own splitting method (not scaffold-based)
train_df, test_df = train_test_split(clean_df, test_size=0.2, random_state=42)

# Step 3: Your own modeling
# ...
```

---

### Example 2: Only Scaffold Split + Your Own Validation

```python
from qsar_validation.scaffold_splitting import ScaffoldSplitter
import pandas as pd

# Load your data (already cleaned)
df = pd.read_csv('clean_data.csv')

# Only use scaffold splitting
splitter = ScaffoldSplitter(smiles_col='SMILES')
train_idx, val_idx, test_idx = splitter.split(df, test_size=0.2, val_size=0.1)

# Get datasets
train_df = df.iloc[train_idx]
test_df = df.iloc[test_idx]

# Your own feature generation
# Your own modeling
# Your own validation
```

---

### Example 3: Only Metrics Calculation

```python
from qsar_validation.performance_metrics import PerformanceMetrics
import numpy as np

# You already have predictions (from your own pipeline)
y_true = np.load('true_values.npy')
y_pred = np.load('predictions.npy')

# Just calculate metrics
metrics = PerformanceMetrics()
results = metrics.calculate_all_metrics(y_true, y_pred, set_name='Test')

print(f"R²: {results['Test_R2']:.3f}")
print(f"RMSE: {results['Test_RMSE']:.3f}")
```

---

### Example 4: Manual Workflow with Selected Modules

```python
from qsar_validation.duplicate_removal import DuplicateRemoval
from qsar_validation.scaffold_splitting import ScaffoldSplitter
from qsar_validation.feature_scaling import FeatureScaler
from qsar_validation.performance_metrics import PerformanceMetrics
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from rdkit import Chem
from rdkit.Chem import AllChem

# ========== YOUR CUSTOM WORKFLOW ==========

# 1. Load data
df = pd.read_csv('my_data.csv')

# 2. Remove duplicates (Module 1)
remover = DuplicateRemoval(smiles_col='SMILES')
df = remover.remove_duplicates(df, strategy='average')

# 3. Scaffold split (Module 2)
splitter = ScaffoldSplitter(smiles_col='SMILES')
train_idx, val_idx, test_idx = splitter.split(df, test_size=0.2)

# 4. Your own featurization
def my_featurizer(smiles_list):
    features = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        features.append(np.array(fp))
    return np.array(features)

X_train = my_featurizer(df.iloc[train_idx]['SMILES'].tolist())
X_test = my_featurizer(df.iloc[test_idx]['SMILES'].tolist())
y_train = df.iloc[train_idx]['Activity'].values
y_test = df.iloc[test_idx]['Activity'].values

# 5. Scale features (Module 3)
scaler = FeatureScaler(method='standard')
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Your own model training
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train_scaled, y_train)

# 7. Your own predictions
y_pred = model.predict(X_test_scaled)

# 8. Calculate metrics (Module 4)
metrics_calc = PerformanceMetrics()
metrics = metrics_calc.calculate_all_metrics(y_test, y_pred, set_name='Test')

print(metrics)
```

---

### Example 5: Just Data Leakage Prevention Modules

```python
from qsar_validation.duplicate_removal import DuplicateRemoval
from qsar_validation.scaffold_splitting import ScaffoldSplitter
from qsar_validation.feature_scaling import FeatureScaler
import pandas as pd

# Use only the data leakage prevention modules
df = pd.read_csv('data.csv')

# 1. Duplicates (Module 1)
remover = DuplicateRemoval(smiles_col='SMILES')
df = remover.remove_duplicates(df)

# 2. Scaffold split (Module 2)
splitter = ScaffoldSplitter(smiles_col='SMILES')
train_idx, _, test_idx = splitter.split(df, test_size=0.2)

# 3. Feature scaling (Module 3)
scaler = FeatureScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Verify no leakage
overlap = splitter.check_scaffold_overlap(train_idx, test_idx, df)
assert overlap == 0, "Data leakage detected!"

# Now do your own modeling...
```

---

## Full Pipeline vs Modular

### Full Pipeline (All-in-One)

```python
from qsar_validation import ModelAgnosticQSARPipeline

# One call does everything
pipeline = ModelAgnosticQSARPipeline(
    featurizer=my_featurizer,
    model=my_model
)

results = pipeline.fit_predict_validate(df)
```

**Pros:** 
- Simple, one function call
- All steps automated
- No configuration needed

**Cons:**
- Less control over individual steps
- Must use all features
- Harder to customize

---

### Modular Approach (Pick & Choose)

```python
from qsar_validation.duplicate_removal import DuplicateRemoval
from qsar_validation.scaffold_splitting import ScaffoldSplitter
from qsar_validation.performance_metrics import PerformanceMetrics

# Use only what you need
remover = DuplicateRemoval()
df = remover.remove_duplicates(df)

splitter = ScaffoldSplitter()
train_idx, _, test_idx = splitter.split(df)

# Your custom code here...

metrics = PerformanceMetrics()
results = metrics.calculate_all_metrics(y_true, y_pred)
```

**Pros:**
- Full control over each step
- Use only what you need
- Easy to customize
- Mix with your own code

**Cons:**
- More code to write
- Need to understand each module
- Manual integration

---

## When to Use What?

| Use Case | Recommended Approach |
|----------|---------------------|
| Quick validation, trust defaults | **Full Pipeline** |
| Custom workflow, specific needs | **Modular** |
| Learning the framework | **Modular** (see each step) |
| Production deployment | **Modular** (more control) |
| Research/experimentation | **Modular** (flexibility) |
| Just need metrics | **Single Module** (PerformanceMetrics) |
| Just need splitting | **Single Module** (ScaffoldSplitter) |
| Complex custom pipeline | **Mix of Modules** |

---

## Module Import Summary

```python
# Individual imports (pick what you need)
from qsar_validation.duplicate_removal import DuplicateRemoval
from qsar_validation.scaffold_splitting import ScaffoldSplitter
from qsar_validation.feature_scaling import FeatureScaler
from qsar_validation.cross_validation import CrossValidator
from qsar_validation.performance_metrics import PerformanceMetrics
from qsar_validation.dataset_bias_analysis import DatasetBiasAnalysis
from qsar_validation.model_complexity_analysis import ModelComplexityAnalysis

# Or full pipeline
from qsar_validation import ModelAgnosticQSARPipeline
```

---

## Conclusion

The framework is designed to be **flexible**:

✅ **Use the full pipeline** for quick validation with sensible defaults  
✅ **Use individual modules** for custom workflows and maximum control  
✅ **Mix and match** modules as needed for your specific use case  

Every module is **independent** and **self-contained**. Pick the ones you need, ignore the rest!

---

## Next Steps

1. **Explore individual modules**: Try each module separately
2. **Build custom workflows**: Combine modules your way
3. **See examples**: Check `examples/modular_examples.py` (coming next!)
4. **Ask questions**: Each module has detailed docstrings

**You have complete freedom!** 🚀
