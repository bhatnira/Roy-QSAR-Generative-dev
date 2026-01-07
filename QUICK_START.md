# Quick Reference: Notebook-Free QSAR Validation

## 30-Second Start

```python
import pandas as pd
from qsar_validation import run_comprehensive_validation

df = pd.read_csv('your_data.csv')
results = run_comprehensive_validation(df, 'SMILES', 'Activity')
```

## 5-Minute Complete Workflow

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit.Chem import AllChem

# Import validators
from qsar_validation import (
    DatasetBiasAnalyzer,
    ActivityCliffDetector,
    ModelComplexityAnalyzer,
    PerformanceMetricsCalculator,
    YRandomizationTester
)

# 1. Load data
df = pd.read_csv('data.csv')

# 2. Validate dataset
analyzer = DatasetBiasAnalyzer('SMILES', 'Activity')
diversity = analyzer.analyze_scaffold_diversity(df)
cliffs = ActivityCliffDetector('SMILES', 'Activity').detect_activity_cliffs(df)

# 3. Generate features (Morgan fingerprints example)
fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, 2048) 
       for s in df['SMILES']]
X = np.array([[int(b) for b in fp] for fp in fps])
y = df['Activity'].values

# 4. Check model complexity
ModelComplexityAnalyzer.analyze_complexity(len(X), X.shape[1], 'random_forest')

# 5. Train model (with low-data settings)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=50, max_depth=5, min_samples_leaf=5, 
                              max_features='sqrt', random_state=42)
model.fit(X_train, y_train)

# 6. Evaluate
y_pred = model.predict(X_test)
metrics = PerformanceMetricsCalculator.calculate_all_metrics(y_test, y_pred)

# 7. Check for overfitting
rand = YRandomizationTester.perform_y_randomization(X_train, y_train, model, n_iterations=10)

print(f"Test RMSE: {metrics['rmse']:.3f}, Y-random R²: {rand['r2_mean']:.3f}")
```

## Command-Line Usage

```bash
# Show validation checklist
python standalone_qsar_workflow.py --checklist

# Run complete workflow
python standalone_qsar_workflow.py --data mydata.csv

# Specify columns
python standalone_qsar_workflow.py --data mydata.csv --smiles SMILES --target pIC50

# Use Ridge regression instead of Random Forest
python standalone_qsar_workflow.py --data mydata.csv --model ridge

# Save results to file
python standalone_qsar_workflow.py --data mydata.csv --output results.txt
```

## Essential Modules

| Module | Purpose | Usage |
|--------|---------|-------|
| `DatasetBiasAnalyzer` | Scaffold diversity | `analyzer.analyze_scaffold_diversity(df)` |
| `ActivityCliffDetector` | Find activity cliffs | `detector.detect_activity_cliffs(df)` |
| `ModelComplexityAnalyzer` | Check n/p ratio | `analyze_complexity(n, p, 'rf')` |
| `PerformanceMetricsCalculator` | Calculate metrics | `calculate_all_metrics(y_true, y_pred)` |
| `YRandomizationTester` | Check overfitting | `perform_y_randomization(X, y, model)` |
| `AssayNoiseEstimator` | Experimental error | `estimate_experimental_error(df, col)` |

## Low-Data Model Settings (n < 200)

```python
# Random Forest (Recommended)
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(
    n_estimators=50,      # Not 100+
    max_depth=5,          # Limit depth
    min_samples_leaf=5,   # Regularization
    max_features='sqrt',  # Feature subsampling
    random_state=42
)

# Ridge Regression (Best for very low data)
from sklearn.linear_model import Ridge
model = Ridge(alpha=1.0)  # Always use regularization

# Gradient Boosting (Use with caution)
from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor(
    n_estimators=50,
    learning_rate=0.05,   # Slow learning
    max_depth=3,          # Shallow trees
    subsample=0.8,
    random_state=42
)
```

## Sample-to-Feature Ratio Guidelines

| Ratio | Recommendation |
|-------|---------------|
| < 5:1 | Use only simple linear models (Ridge/Lasso) |
| < 10:1 | Use regularized models, avoid complex ensembles |
| < 20:1 | Can use Random Forest with strict regularization |
| > 20:1 | More flexibility in model choice |

## Quick Checks

```python
# Check 1: Is my dataset diverse enough?
diversity_ratio = n_scaffolds / n_molecules
if diversity_ratio < 0.3:
    print("WARNING: Low diversity - limited applicability")

# Check 2: Is my RMSE realistic?
if rmse < 0.3:  # For log units (IC50/pIC50)
    print("WARNING: RMSE suspiciously low - check for leakage")

# Check 3: Is my model overfitting?
if y_random_r2 > 0.2:
    print("FAIL: Model is overfitting")

# Check 4: Do I beat baseline?
baseline = Ridge(alpha=1.0)
baseline.fit(X_train, y_train)
baseline_rmse = sqrt(mse(y_test, baseline.predict(X_test)))
if my_rmse > baseline_rmse:
    print("WARNING: Not beating simple baseline")
```

## Integration Patterns

### Pattern 1: Preprocessing Check
```python
def preprocess(data_path):
    df = pd.read_csv(data_path)
    results = run_comprehensive_validation(df, 'SMILES', 'Activity')
    
    if results['scaffold_diversity']['diversity_ratio'] < 0.3:
        raise ValueError("Dataset not diverse enough")
    
    return df
```

### Pattern 2: Post-Training Validation
```python
def validate_model(model, X_test, y_test, X_train, y_train):
    y_pred = model.predict(X_test)
    metrics = PerformanceMetricsCalculator.calculate_all_metrics(y_test, y_pred)
    rand = YRandomizationTester.perform_y_randomization(X_train, y_train, model)
    
    return metrics['rmse'] < 0.5 and rand['r2_mean'] < 0.2
```

### Pattern 3: Custom Validator
```python
class MyValidator:
    def __init__(self):
        self.results = {}
    
    def validate_all(self, df, X, y, model):
        # Dataset checks
        analyzer = DatasetBiasAnalyzer('SMILES', 'Activity')
        self.results['diversity'] = analyzer.analyze_scaffold_diversity(df)
        
        # Model checks
        ModelComplexityAnalyzer.analyze_complexity(len(X), X.shape[1], 'rf')
        
        return self.results
```

## Common Issues

**Import Error**: 
```python
# If package not installed, use:
import sys
sys.path.append('path/to/QSAR_Models/src')
from qsar_validation import ...
```

**RDKit Issues**:
```bash
conda install -c conda-forge rdkit
# or
pip install rdkit-pypi  # May not work on all systems
```

**Missing Dependencies**:
```bash
pip install pandas numpy scikit-learn scipy
```

## Expected Performance (IC50 data)

- **Good RMSE**: 0.4 - 0.6 log units
- **Suspicious RMSE**: < 0.3 log units (check for leakage)
- **Y-random R²**: Should be ≤ 0 (negative is good)
- **Scaffold diversity**: Should be > 0.3 for diverse dataset

## Files You Need

1. **USAGE_GUIDE.md** - Complete documentation
2. **standalone_qsar_workflow.py** - Ready-to-use script
3. **examples/** - Working examples
4. **src/qsar_validation/** - The validation package

## One-Liners for Common Tasks

```python
# Show validation checklist
from qsar_validation import print_comprehensive_validation_checklist
print_comprehensive_validation_checklist()

# Quick dataset check
from qsar_validation import run_comprehensive_validation
results = run_comprehensive_validation(df, 'SMILES', 'Activity')

# Check model complexity
from qsar_validation import ModelComplexityAnalyzer
ModelComplexityAnalyzer.analyze_complexity(150, 2048, 'random_forest')

# Calculate all metrics
from qsar_validation import PerformanceMetricsCalculator
metrics = PerformanceMetricsCalculator.calculate_all_metrics(y_true, y_pred)
```

---

**Remember**: This framework is designed to work **WITHOUT** notebooks. Use it in scripts, pipelines, or production code!
