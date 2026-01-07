# QSAR Models

**A Purely Modular QSAR Validation Framework**

**Version 3.0.0 - Independent Modules, Maximum Flexibility**

A professional framework of **7 independent, composable modules** for QSAR validation. Perfect for the low-data regime (< 200 compounds). **We provide the building blocks, you build the workflow.**

## 🧩 Framework Philosophy

> **"No magic. No automation. Just reliable tools."**
> 
> **"You build the pipeline. We provide the pipes."**

This framework provides **ONLY individual modules** - no all-in-one pipelines, no hidden automation, no forced workflows.

**You control:**
- ✅ Which modules to use
- ✅ When to use them  
- ✅ How to combine them
- ✅ Your complete workflow

**We provide:**
- ✅ 7 independent, tested modules
- ✅ Clear documentation for each
- ✅ Examples of combinations
- ✅ Data leakage prevention tools
- ✅ Validation analysis tools

---

## 📦 Available Modules

| # | Module | Purpose | When to Use |
|---|--------|---------|-------------|
| 1 | `DuplicateRemoval` | Remove duplicate molecules | Before any data splitting |
| 2 | `ScaffoldSplitter` | Split by molecular scaffolds | For realistic generalization estimates |
| 3 | `FeatureScaler` | Scale features properly | When normalizing features (no leakage) |
| 4 | `CrossValidator` | Perform k-fold cross-validation | For model evaluation |
| 5 | `PerformanceMetrics` | Calculate comprehensive metrics | For performance analysis |
| 6 | `DatasetBiasAnalysis` | Detect dataset bias | To check data quality issues |
| 7 | `ModelComplexityAnalysis` | Analyze model complexity | To detect overfitting risk |

**Each module is independent. Use any, all, or none. Mix with your own code.**

---

## 🚀 Quick Start

### Minimal Example (Just 3 Modules)
```python
from qsar_validation.duplicate_removal import DuplicateRemoval
from qsar_validation.scaffold_splitting import ScaffoldSplitter
from qsar_validation.performance_metrics import PerformanceMetrics
import pandas as pd

# Load your data
df = pd.read_csv('my_data.csv')

# Module 1: Clean data
remover = DuplicateRemoval(smiles_col='SMILES')
df = remover.remove_duplicates(df, strategy='average')

# Module 2: Split by scaffolds
splitter = ScaffoldSplitter(smiles_col='SMILES')
train_idx, val_idx, test_idx = splitter.split(df, test_size=0.2)

# YOUR CODE: Featurization, model training, predictions
# (You control this part completely)

# Module 3: Calculate metrics
metrics = PerformanceMetrics()
results = metrics.calculate_all_metrics(y_true, y_pred, set_name='Test')
print(results)
```

That's it! Three modules, full control, no magic.

---

## 🎨 Usage Patterns

### Pattern 1: Data Leakage Prevention Only
Use just the essential modules to prevent data leakage:

```python
from qsar_validation.duplicate_removal import DuplicateRemoval
from qsar_validation.scaffold_splitting import ScaffoldSplitter
from qsar_validation.feature_scaling import FeatureScaler

# 1. Remove duplicates BEFORE splitting (prevents leakage)
remover = DuplicateRemoval(smiles_col='SMILES')
df = remover.remove_duplicates(df)

# 2. Scaffold-based split (prevents leakage)
splitter = ScaffoldSplitter(smiles_col='SMILES')
train_idx, _, test_idx = splitter.split(df, test_size=0.2)

# 3. Scale using train stats only (prevents leakage)
scaler = FeatureScaler(method='standard')
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Now do YOUR modeling with confidence - no data leakage!
```

### Pattern 2: Complete Validation Workflow
Use all 7 modules for comprehensive validation:

```python
from qsar_validation.duplicate_removal import DuplicateRemoval
from qsar_validation.scaffold_splitting import ScaffoldSplitter
from qsar_validation.feature_scaling import FeatureScaler
from qsar_validation.cross_validation import CrossValidator
from qsar_validation.performance_metrics import PerformanceMetrics
from qsar_validation.dataset_bias_analysis import DatasetBiasAnalysis
from qsar_validation.model_complexity_analysis import ModelComplexityAnalysis

# 1. Clean
remover = DuplicateRemoval()
df = remover.remove_duplicates(df)

# 2. Split
splitter = ScaffoldSplitter()
train_idx, val_idx, test_idx = splitter.split(df)

# 3. YOUR featurization code
X_train, y_train = your_featurizer(df.iloc[train_idx])
X_test, y_test = your_featurizer(df.iloc[test_idx])

# 4. Scale
scaler = FeatureScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Cross-validate
cv = CrossValidator(n_folds=5)
cv_scores = cv.cross_validate(model, X_train, y_train)

# 6. Train & predict (YOUR code)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 7. Metrics
metrics = PerformanceMetrics()
results = metrics.calculate_all_metrics(y_test, y_pred)

# 8. Bias analysis
bias_analyzer = DatasetBiasAnalysis()
bias_report = bias_analyzer.analyze_bias(X_train, X_test, y_train, y_test)

# 9. Complexity analysis
complexity_analyzer = ModelComplexityAnalysis()
complexity_report = complexity_analyzer.analyze_complexity(model, X_train, y_train)
```

### Pattern 3: Custom Workflow
Mix our modules with your own code:

```python
from qsar_validation.duplicate_removal import DuplicateRemoval
from qsar_validation.performance_metrics import PerformanceMetrics

# Use our duplicate removal
remover = DuplicateRemoval()
df = remover.remove_duplicates(df)

# YOUR CUSTOM SPLITTING (not using our ScaffoldSplitter)
train_df, test_df = my_custom_splitter(df)

# YOUR CUSTOM FEATURIZATION
X_train, y_train = my_custom_featurizer(train_df)
X_test, y_test = my_custom_featurizer(test_df)

# YOUR CUSTOM MODEL
model = my_custom_model()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Use our metrics
metrics = PerformanceMetrics()
results = metrics.calculate_all_metrics(y_test, y_pred)
```

### Pattern 4: Just One Module
Need just one thing? Use just one module:

```python
# Scenario: You just need to calculate metrics for existing predictions

from qsar_validation.performance_metrics import PerformanceMetrics
import numpy as np

# You already have predictions from your pipeline
y_true = np.load('true_values.npy')
y_pred = np.load('predictions.npy')

# Just calculate metrics
metrics = PerformanceMetrics()
results = metrics.calculate_all_metrics(y_true, y_pred, set_name='Test')

print(f"R²: {results['Test_R2']:.3f}")
print(f"RMSE: {results['Test_RMSE']:.3f}")
print(f"MAE: {results['Test_MAE']:.3f}")
```

---

## 🛡️ Data Leakage Prevention

Three modules work together to prevent all types of data leakage:

### 1️⃣ DuplicateRemoval
**Prevents:** Duplicates appearing in both train and test sets

```python
from qsar_validation.duplicate_removal import DuplicateRemoval

remover = DuplicateRemoval(smiles_col='SMILES')

# Strategy 1: Keep first occurrence
df = remover.remove_duplicates(df, strategy='first')

# Strategy 2: Average duplicate activities
df = remover.remove_duplicates(df, strategy='average')

# Check if duplicates exist
has_dups = remover.check_duplicates(df)
```

### 2️⃣ ScaffoldSplitter
**Prevents:** Similar molecules (same scaffold) in train and test sets

```python
from qsar_validation.scaffold_splitting import ScaffoldSplitter

splitter = ScaffoldSplitter(smiles_col='SMILES')

# Split by Bemis-Murcko scaffolds
train_idx, val_idx, test_idx = splitter.split(
    df, 
    test_size=0.2,
    val_size=0.1,
    random_state=42
)

# Verify no scaffold overlap
overlap = splitter.check_scaffold_overlap(train_idx, test_idx, df)
print(f"Scaffold overlap: {overlap}")  # Should be 0

# Get scaffold for a specific molecule
scaffold = splitter.get_scaffold('c1ccccc1CC')
```

### 3️⃣ FeatureScaler
**Prevents:** Using test set statistics to scale features

```python
from qsar_validation.feature_scaling import FeatureScaler

scaler = FeatureScaler(method='standard')  # or 'minmax', 'robust'

# CORRECT: Fit on train only, transform both
scaler.fit(X_train)  # Learn statistics from train only
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Or use fit_transform for train
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

## 📊 Validation Modules

### 4️⃣ CrossValidator
Perform k-fold cross-validation properly:

```python
from qsar_validation.cross_validation import CrossValidator
from sklearn.ensemble import RandomForestRegressor

cv = CrossValidator(n_folds=5, random_state=42)

model = RandomForestRegressor()
cv_scores = cv.cross_validate(model, X_train, y_train, scoring='r2')

print(f"CV R² scores: {cv_scores}")
print(f"Mean: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")

# Get fold indices for manual iteration
for fold_idx, (train_idx, val_idx) in enumerate(cv.get_folds(X_train)):
    print(f"Fold {fold_idx}: {len(train_idx)} train, {len(val_idx)} val")
```

### 5️⃣ PerformanceMetrics
Calculate comprehensive performance metrics:

```python
from qsar_validation.performance_metrics import PerformanceMetrics

metrics = PerformanceMetrics()

# All metrics at once
results = metrics.calculate_all_metrics(y_true, y_pred, set_name='Test')
# Returns: {'Test_R2': ..., 'Test_RMSE': ..., 'Test_MAE': ..., 
#           'Test_Pearson_r': ..., 'Test_Spearman_rho': ...}

# Individual metrics
r2 = metrics.calculate_r2(y_true, y_pred)
rmse = metrics.calculate_rmse(y_true, y_pred)
mae = metrics.calculate_mae(y_true, y_pred)
pearson = metrics.calculate_pearson_r(y_true, y_pred)
spearman = metrics.calculate_spearman_rho(y_true, y_pred)
```

### 6️⃣ DatasetBiasAnalysis
Detect dataset bias issues:

```python
from qsar_validation.dataset_bias_analysis import DatasetBiasAnalysis

analyzer = DatasetBiasAnalysis()

# Comprehensive bias analysis
report = analyzer.analyze_bias(
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test
)

# Check specific aspects
coverage = analyzer.check_activity_coverage(y_train, y_test)
print(f"Test activity coverage: {coverage:.1%}")

# Interpret warnings
if report['warnings']:
    print("⚠️ Dataset bias detected:")
    for warning in report['warnings']:
        print(f"  - {warning}")
```

### 7️⃣ ModelComplexityAnalysis
Analyze model complexity and overfitting risk:

```python
from qsar_validation.model_complexity_analysis import ModelComplexityAnalysis

analyzer = ModelComplexityAnalysis()

# Analyze model complexity
report = analyzer.analyze_complexity(
    model=model,
    X=X_train,
    y=y_train
)

print(f"Samples: {report['n_samples']}")
print(f"Features: {report['n_features']}")
print(f"Sample/Feature ratio: {report['sample_to_feature_ratio']:.2f}")

if report['warnings']:
    print("⚠️ Complexity warnings:")
    for warning in report['warnings']:
        print(f"  - {warning}")
```

---

## 📚 Documentation

### Module-Specific Guides
- **Quick Reference**: [`MODULAR_QUICK_REFERENCE.md`](MODULAR_QUICK_REFERENCE.md) - One-page reference for all modules
- **Complete Guide**: [`MODULAR_USAGE_GUIDE.md`](MODULAR_USAGE_GUIDE.md) - Comprehensive usage guide
- **Philosophy**: [`MODULAR_FRAMEWORK_PHILOSOPHY.md`](MODULAR_FRAMEWORK_PHILOSOPHY.md) - Framework design principles
- **Examples**: [`examples/modular_examples.py`](examples/modular_examples.py) - 10 working examples

### Technical Documentation
- **Data Leakage Guide**: [`DATA_LEAKAGE_PREVENTION.md`](DATA_LEAKAGE_PREVENTION.md) - Complete leakage prevention guide
- **Demonstration Report**: [`COMPREHENSIVE_DEMONSTRATION_REPORT.md`](COMPREHENSIVE_DEMONSTRATION_REPORT.md) - Framework validation
- **Demo Summary**: [`DEMO_SUMMARY.md`](DEMO_SUMMARY.md) - Quick summary of demo results

---

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/bhatnira/Roy-QSAR-Generative-dev.git
cd Roy-QSAR-Generative-dev

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

Or install directly from GitHub:
```bash
pip install git+https://github.com/bhatnira/Roy-QSAR-Generative-dev.git
```

### Requirements
- Python ≥ 3.8
- pandas
- numpy
- scikit-learn
- rdkit
- scipy

---

## 🎯 When to Use Each Module

| Your Need | Recommended Modules | Why |
|-----------|--------------------|----|
| Prevent data leakage | `DuplicateRemoval`, `ScaffoldSplitter`, `FeatureScaler` | Essential 3 for leakage prevention |
| Just calculate metrics | `PerformanceMetrics` | Quick evaluation of predictions |
| Just clean data | `DuplicateRemoval` | Remove duplicates before anything else |
| Just split data | `ScaffoldSplitter` | Realistic train/test splits |
| Evaluate model | `CrossValidator`, `PerformanceMetrics` | Comprehensive evaluation |
| Check data quality | `DatasetBiasAnalysis` | Detect bias issues |
| Check overfitting risk | `ModelComplexityAnalysis` | Analyze model/data complexity |
| Custom workflow | Any combination | Mix and match as needed |

---

## ⚙️ Module Parameters

### Quick Parameter Reference

**DuplicateRemoval:**
- `smiles_col`: Column name for SMILES strings
- `strategy`: 'first' | 'average' (how to handle duplicates)

**ScaffoldSplitter:**
- `smiles_col`: Column name for SMILES strings
- `test_size`: Fraction for test set (default 0.2)
- `val_size`: Fraction for validation set (default 0.1)
- `random_state`: Random seed for reproducibility

**FeatureScaler:**
- `method`: 'standard' | 'minmax' | 'robust'

**CrossValidator:**
- `n_folds`: Number of CV folds (default 5)
- `random_state`: Random seed

**PerformanceMetrics:**
- `set_name`: Name prefix for metrics (e.g., 'Test')

---

## 💡 Best Practices

### ✅ DO:
1. **Always remove duplicates BEFORE splitting**
2. **Use scaffold splitting for realistic estimates**
3. **Fit scaler on training data only**
4. **Use cross-validation on training set only**
5. **Check for data bias**
6. **Analyze model complexity**
7. **Verify zero scaffold overlap**
8. **Mix modules with your own code freely**

### ❌ DON'T:
1. ~~Remove duplicates after splitting~~
2. ~~Use random splitting for QSAR~~
3. ~~Fit scaler on all data~~
4. ~~Include test data in cross-validation~~
5. ~~Ignore bias warnings~~
6. ~~Skip complexity analysis~~
7. ~~Assume modules are compatible with everything~~

---

## 🎓 Learning Path

1. **Start Simple**: Use just one module (e.g., `PerformanceMetrics`)
2. **Add Leakage Prevention**: Add `DuplicateRemoval` and `ScaffoldSplitter`
3. **Add Validation**: Include `CrossValidator`
4. **Add Analysis**: Use `DatasetBiasAnalysis` and `ModelComplexityAnalysis`
5. **Customize**: Mix with your own code
6. **Advanced**: Build your own helper classes wrapping modules

---

## 🤝 Contributing

Contributions welcome! This modular design makes it easy to:
- Add new modules
- Improve existing modules
- Add examples
- Improve documentation

Each module is independent, so changes don't affect others.

---

## 📄 License

[Add your license here]

---

## 🔗 Links

- **GitHub**: https://github.com/bhatnira/Roy-QSAR-Generative-dev
- **Issues**: https://github.com/bhatnira/Roy-QSAR-Generative-dev/issues
- **Documentation**: See `/docs` folder

---

## 🌟 Framework Principles

> **"No magic. No automation. Just reliable tools."**
> 
> **"You build the pipeline. We provide the pipes."**
> 
> **"Your workflow, your rules, our modules."**

We believe in **modularity** over monoliths, **flexibility** over convenience, and **transparency** over magic.

**You are the architect. We provide the building blocks.** 🚀

---

## 📞 Support

- **Questions**: Open an issue on GitHub
- **Examples**: See `examples/modular_examples.py`
- **Documentation**: See `MODULAR_USAGE_GUIDE.md`
- **Philosophy**: See `MODULAR_FRAMEWORK_PHILOSOPHY.md`

**Remember: Each module is independent. Use what you need, ignore the rest!**
