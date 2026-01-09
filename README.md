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

## Core Modules

| Module | Import | Purpose |
|--------|--------|---------|
| **Data Cleaning** | `from utils.qsar_utils_no_leakage import quick_clean` | Remove duplicates, canonicalize SMILES |
| **Splitting** | `from qsar_validation.splitting_strategies import AdvancedSplitter` | Scaffold/temporal/cluster splits |
| **Scaling** | `from qsar_validation.feature_scaling import FeatureScaler` | StandardScaler (fit on train only) |
| **Selection** | `from qsar_validation.feature_selection import FeatureSelector` | Variance, correlation filtering |
| **Validation** | `from qsar_validation.performance_validation import PerformanceValidator` | Cross-validation, metrics |

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
