# Multi-Library Support (v4.1.0)

## Overview

The QSAR framework now supports **multiple ML libraries** - not just scikit-learn!

Users can now use:
- **Scikit-learn** (Ridge, RandomForest, SVM, etc.)
- **XGBoost** (XGBRegressor, XGBClassifier)
- **LightGBM** (LGBMRegressor, LGBMClassifier)
- **PyTorch** (custom neural networks)
- **TensorFlow/Keras** (Sequential, Functional API)
- **Custom models** (any class with fit/predict methods)

## What Changed?

### 1. Model Complexity Control (model_complexity_control.py)

**Before**: Only worked with sklearn models
```python
# Old API - sklearn only
controller.nested_cv(X, y, model_type='ridge')
```

**After**: Works with ANY ML library
```python
# New API - any library
from sklearn.linear_model import Ridge
model = Ridge()
param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0]}
results = controller.nested_cv(X, y, model=model, param_grid=param_grid, library='sklearn')

# Or XGBoost
import xgboost as xgb
model = xgb.XGBRegressor()
param_grid = {'max_depth': [3, 5], 'learning_rate': [0.1, 0.01]}
results = controller.nested_cv(X, y, model=model, param_grid=param_grid, library='xgboost')

# Or LightGBM
import lightgbm as lgb
model = lgb.LGBMRegressor()
param_grid = {'num_leaves': [15, 31], 'learning_rate': [0.1, 0.01]}
results = controller.nested_cv(X, y, model=model, param_grid=param_grid, library='lightgbm')
```

### 2. Universal ModelWrapper

New `ModelWrapper` class provides unified interface across all libraries:

```python
# Automatically detects library and wraps model
wrapped = ModelWrapper(any_model)

# Unified interface
wrapped.fit(X_train, y_train)
predictions = wrapped.predict(X_test)
cloned = wrapped.clone()
```

Handles library-specific quirks:
- PyTorch: tensor conversions, training mode
- TensorFlow: verbose settings, input reshaping
- XGBoost/LightGBM: eval_set, early stopping
- Sklearn: standard fit/predict

### 3. Library-Agnostic Metrics

New `calculate_metrics()` function doesn't depend on sklearn:

```python
metrics = calculate_metrics(y_true, y_pred)
# Returns: {'r2': ..., 'rmse': ..., 'mae': ...}
```

### 4. Library-Specific Recommendations

`recommend_models()` now provides recommendations for ALL libraries:

```python
recommendations = controller.recommend_models()

# Returns dict with keys:
# - 'sklearn': ['Ridge', 'Lasso', ...]
# - 'xgboost': ['XGBRegressor (max_depth≤3)', ...]
# - 'lightgbm': ['LGBMRegressor (num_leaves≤15)', ...]
# - 'pytorch': ['Small MLP (1-2 layers)', ...]
# - 'tensorflow': ['Keras Sequential', ...]
# - 'avoid': ['Deep neural nets', ...]
```

Recommendations adapt to dataset size:
- **ratio < 0.1**: sklearn linear models only
- **ratio < 0.5**: sklearn + shallow boosting
- **ratio < 2.0**: all libraries, moderate complexity
- **ratio ≥ 2.0**: all libraries, all complexity levels

### 5. Library-Specific Parameter Grids

`get_safe_param_grid()` now supports all libraries:

```python
# Sklearn Ridge
param_grid = controller.get_safe_param_grid('ridge', library='sklearn')
# Returns: {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}

# XGBoost
param_grid = controller.get_safe_param_grid('xgboost', library='xgboost')
# Returns: {'max_depth': [...], 'learning_rate': [...], 'n_estimators': [...]}

# LightGBM
param_grid = controller.get_safe_param_grid('lightgbm', library='lightgbm')
# Returns: {'num_leaves': [...], 'learning_rate': [...], 'n_estimators': [...]}

# PyTorch MLP
param_grid = controller.get_safe_param_grid('pytorch_mlp', library='pytorch')
# Returns: {'hidden_sizes': [[64], [128]], 'learning_rate': [...], 'dropout': [...]}

# TensorFlow
param_grid = controller.get_safe_param_grid('tensorflow', library='tensorflow')
# Returns: {'layers': [[64], [128]], 'learning_rate': [...], 'dropout': [...]}
```

Parameter grids automatically adjust to dataset size!

### 6. Universal Nested Cross-Validation

`nested_cv()` now works with any library:

```python
# Works with ANY model!
results = controller.nested_cv(
    X, y,
    model=any_model,           # sklearn, xgboost, lightgbm, pytorch, tensorflow
    param_grid=param_grid,     # library-specific parameters
    library='sklearn',         # 'sklearn', 'xgboost', 'lightgbm', 'pytorch', 'tensorflow'
    outer_cv=5,
    inner_cv=3
)

# Returns comprehensive results
print(f"R²: {results['mean_r2']:.4f} ± {results['std_r2']:.4f}")
print(f"RMSE: {results['mean_rmse']:.4f} ± {results['std_rmse']:.4f}")
print(f"Best params per fold: {results['best_params_per_fold']}")
```

### 7. Universal Complexity Assessment

`assess_model_complexity()` now handles all libraries:

```python
# Works with ANY trained model!
stats = controller.assess_model_complexity(trained_model, library='sklearn')

# Returns library-specific complexity metrics:
# - Sklearn linear: n_parameters, regularization
# - Sklearn trees: n_estimators, max_depth, estimated_parameters
# - XGBoost: n_estimators, max_depth, learning_rate
# - LightGBM: n_estimators, num_leaves, learning_rate
# - PyTorch: total_parameters, n_layers
# - TensorFlow: total_parameters, n_layers
```

## Architecture Changes

### No More Hard Dependencies!

**Before**:
```python
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
```

**After**:
```python
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

# sklearn, xgboost, lightgbm, torch, tensorflow are OPTIONAL
# Imported only when needed, with try/except blocks
```

### New Helper Functions

1. **`simple_kfold_split()`**: sklearn-independent K-fold splitting
2. **`simple_grid_search()`**: sklearn-independent grid search
3. **`calculate_metrics()`**: sklearn-independent metrics
4. **`ModelWrapper`**: universal model interface

All nested CV and hyperparameter tuning now uses these library-agnostic implementations!

## Usage Examples

### Example 1: Sklearn Ridge Regression

```python
from sklearn.linear_model import Ridge

# Setup
controller = ModelComplexityController(n_samples=100, n_features=500)

# Get safe parameters for this dataset size
param_grid = controller.get_safe_param_grid('ridge', library='sklearn')

# Create model
model = Ridge()

# Nested CV
results = controller.nested_cv(
    X, y,
    model=model,
    param_grid=param_grid,
    library='sklearn',
    outer_cv=5,
    inner_cv=3
)

print(f"R²: {results['mean_r2']:.4f} ± {results['std_r2']:.4f}")
```

### Example 2: XGBoost Regressor

```python
import xgboost as xgb

# Setup
controller = ModelComplexityController(n_samples=100, n_features=500)

# Get safe parameters (automatically adapts to small dataset)
param_grid = controller.get_safe_param_grid('xgboost', library='xgboost')
# For small dataset: {'max_depth': [2, 3], 'learning_rate': [0.1, 0.01], ...}

# Create model
model = xgb.XGBRegressor(random_state=42)

# Nested CV
results = controller.nested_cv(
    X, y,
    model=model,
    param_grid=param_grid,
    library='xgboost',
    outer_cv=5,
    inner_cv=3
)

print(f"R²: {results['mean_r2']:.4f} ± {results['std_r2']:.4f}")
```

### Example 3: LightGBM Regressor

```python
import lightgbm as lgb

# Setup
controller = ModelComplexityController(n_samples=100, n_features=500)

# Get safe parameters
param_grid = controller.get_safe_param_grid('lightgbm', library='lightgbm')
# For small dataset: {'num_leaves': [7, 15], 'learning_rate': [0.1, 0.01], ...}

# Create model
model = lgb.LGBMRegressor(random_state=42, verbose=-1)

# Nested CV
results = controller.nested_cv(
    X, y,
    model=model,
    param_grid=param_grid,
    library='lightgbm',
    outer_cv=5,
    inner_cv=3
)

print(f"R²: {results['mean_r2']:.4f} ± {results['std_r2']:.4f}")
```

### Example 4: PyTorch Neural Network

```python
import torch
import torch.nn as nn

# Define custom PyTorch model
class SimpleMLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden_sizes=[64], dropout=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# Setup
controller = ModelComplexityController(n_samples=100, n_features=500)

# Get safe parameters
param_grid = controller.get_safe_param_grid('pytorch_mlp', library='pytorch')
# For small dataset: {'hidden_sizes': [[32], [64]], 'dropout': [0.2, 0.3], ...}

# Create model
model = SimpleMLPRegressor(input_dim=500, hidden_sizes=[64], dropout=0.2)

# Nested CV
results = controller.nested_cv(
    X, y,
    model=model,
    param_grid=param_grid,
    library='pytorch',
    outer_cv=5,
    inner_cv=3
)

print(f"R²: {results['mean_r2']:.4f} ± {results['std_r2']:.4f}")
```

### Example 5: TensorFlow/Keras

```python
from tensorflow import keras

# Create Keras model builder function
def create_keras_model(layers=[64], learning_rate=0.001, dropout=0.2):
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(500,)))
    for units in layers:
        model.add(keras.layers.Dense(units, activation='relu'))
        model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.Dense(1))
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse'
    )
    return model

# Setup
controller = ModelComplexityController(n_samples=100, n_features=500)

# Get safe parameters
param_grid = controller.get_safe_param_grid('tensorflow', library='tensorflow')

# Create model
model = create_keras_model(layers=[64], learning_rate=0.001, dropout=0.2)

# Nested CV
results = controller.nested_cv(
    X, y,
    model=model,
    param_grid=param_grid,
    library='tensorflow',
    outer_cv=5,
    inner_cv=3
)

print(f"R²: {results['mean_r2']:.4f} ± {results['std_r2']:.4f}")
```

### Example 6: Custom Models

```python
# Define custom model with fit/predict interface
class CustomModel:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.coef_ = None
    
    def fit(self, X, y):
        # Your custom training logic
        self.coef_ = np.linalg.lstsq(X, y, rcond=None)[0]
        return self
    
    def predict(self, X):
        return X @ self.coef_

# Setup
controller = ModelComplexityController(n_samples=100, n_features=500)

# Define parameter grid
param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0]}

# Create model
model = CustomModel(alpha=1.0)

# Nested CV
results = controller.nested_cv(
    X, y,
    model=model,
    param_grid=param_grid,
    library='custom',
    outer_cv=5,
    inner_cv=3
)

print(f"R²: {results['mean_r2']:.4f} ± {results['std_r2']:.4f}")
```

## Benefits

1. **Maximum Flexibility**: Use ANY ML library you prefer
2. **No Vendor Lock-in**: Not forced to use sklearn
3. **Consistent Interface**: Same API across all libraries
4. **Library-Specific Optimization**: Parameter grids adapt to each library's strengths
5. **Dataset-Aware**: All recommendations consider dataset size
6. **Best Practices**: Nested CV, proper metrics, complexity control work everywhere

## Backwards Compatibility

The old sklearn-specific API still works through the new interface:

```python
# Old code using sklearn
from sklearn.linear_model import Ridge
model = Ridge()
param_grid = {'alpha': [0.01, 0.1, 1.0]}
results = controller.nested_cv(X, y, model=model, param_grid=param_grid, library='sklearn')
```

## Future Modules to Update

The following modules currently assume sklearn and should be updated in future versions:

1. **performance_validation.py**: `cross_validate_properly()`, `y_randomization_test()`
2. **uncertainty_estimation.py**: Ensemble uncertainty, dropout-based uncertainty
3. Feature engineering modules: Already library-agnostic (use numpy/pandas)

## Installation

### Minimal Installation (no ML libraries)
```bash
pip install numpy pandas scipy rdkit
```

### With Scikit-learn
```bash
pip install numpy pandas scipy rdkit scikit-learn
```

### With XGBoost
```bash
pip install numpy pandas scipy rdkit scikit-learn xgboost
```

### With LightGBM
```bash
pip install numpy pandas scipy rdkit scikit-learn lightgbm
```

### With PyTorch
```bash
pip install numpy pandas scipy rdkit torch
```

### With TensorFlow
```bash
pip install numpy pandas scipy rdkit tensorflow
```

### Everything
```bash
pip install numpy pandas scipy rdkit scikit-learn xgboost lightgbm torch tensorflow
```

## Version History

- **v4.1.0** (Current): Multi-library support added
- **v4.0.0**: QSAR pitfalls mitigation modules added
- **v3.0.0**: Feature engineering modules with leakage prevention
- **v2.0.0**: Three splitting strategies (scaffold, temporal, cluster)
- **v1.0.0**: Initial purely modular framework

## Philosophy

The framework follows these principles:

1. **Modularity**: Use only what you need
2. **Flexibility**: Choose your own ML library
3. **Best Practices**: Prevent common QSAR pitfalls
4. **Transparency**: Understand what's happening
5. **No Black Boxes**: Full control over every step

---

**Framework Version**: 4.1.0  
**Date**: January 2025  
**Status**: ✅ Complete - Ready for use with all major ML libraries
