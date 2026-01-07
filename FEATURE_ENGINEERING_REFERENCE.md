# Feature Engineering Reference - Data Leakage Prevention

**Quick reference for proper feature engineering in QSAR models**

---

## ⚠️ THE GOLDEN RULES

1. **Feature Scaling** → Fit on **TRAIN FOLD ONLY**
2. **Feature Selection** → Use **NESTED CV**
3. **PCA** → Fit on **TRAIN FOLD ONLY**

**NEVER fit ANY transformation on validation or test data!**

---

## 🎯 Three Splitting Strategies

### 1. Scaffold-Based (RECOMMENDED) ⭐

**When:** Most QSAR tasks  
**Why:** Industry standard, prevents scaffold leakage

```python
from qsar_validation.splitting_strategies import ScaffoldSplitter

splitter = ScaffoldSplitter(smiles_col='SMILES')
train_idx, val_idx, test_idx = splitter.split(df, test_size=0.2)
```

### 2. Temporal (Time-Based) 📅

**When:** You have date/time information  
**Why:** Simulates realistic deployment (train on past, predict future)

```python
from qsar_validation.splitting_strategies import TemporalSplitter

splitter = TemporalSplitter(smiles_col='SMILES', date_col='Date')
train_idx, val_idx, test_idx = splitter.split(df, test_size=0.2)
```

### 3. Leave-Cluster-Out 🔗

**When:** Small datasets (< 100 compounds)  
**Why:** Good for diverse, small datasets

```python
from qsar_validation.splitting_strategies import ClusterSplitter

splitter = ClusterSplitter(smiles_col='SMILES', n_clusters=5)
train_idx, val_idx, test_idx = splitter.split(df, test_size=0.2)
```

---

## 📏 Feature Scaling (Fit on Train Fold Only)

### Available Scalers
- `standard`: StandardScaler (mean=0, std=1)
- `minmax`: MinMaxScaler (range [0,1])
- `robust`: RobustScaler (uses median, robust to outliers)

### ✅ CORRECT Usage

```python
from qsar_validation.feature_scaling import FeatureScaler
from sklearn.model_selection import KFold

# Within each CV fold
for train_idx, val_idx in KFold(n_splits=5).split(X_train):
    scaler = FeatureScaler(method='standard')
    
    # Fit on THIS fold's training data only
    scaler.fit(X_train[train_idx])  # ✓
    
    # Transform train and val
    X_train_scaled = scaler.transform(X_train[train_idx])
    X_val_scaled = scaler.transform(X_train[val_idx])
    
    # Train model
    model.fit(X_train_scaled, y_train[train_idx])
```

### ❌ INCORRECT Usage (LEAKAGE!)

```python
# WRONG: Fitting on all training data before CV
scaler = FeatureScaler(method='standard')
scaler.fit(X_train)  # ❌ LEAKAGE!
X_scaled = scaler.transform(X_train)
cv_score = cross_val_score(model, X_scaled, y)  # ❌ LEAKAGE!
```

---

## 🎯 Feature Selection (Nested CV)

### Available Methods
- `variance`: Remove low-variance features
- `correlation`: Remove highly correlated features
- `model_based`: Use Random Forest importance
- `univariate`: Statistical tests (f_regression)

### ✅ CORRECT Usage

```python
from qsar_validation.feature_selection import FeatureSelector
from sklearn.model_selection import KFold

# Within each CV fold
for train_idx, val_idx in KFold(n_splits=5).split(X_train):
    selector = FeatureSelector(method='univariate', n_features=50)
    
    # Fit on THIS fold's training data only
    selector.fit(X_train[train_idx], y_train[train_idx])  # ✓
    
    # Transform train and val
    X_train_selected = selector.transform(X_train[train_idx])
    X_val_selected = selector.transform(X_train[val_idx])
    
    # Train model
    model.fit(X_train_selected, y_train[train_idx])
```

### ❌ INCORRECT Usage (LEAKAGE!)

```python
# WRONG: Selecting features before CV
selector = FeatureSelector(method='univariate', n_features=50)
selector.fit(X_train, y_train)  # ❌ LEAKAGE!
X_selected = selector.transform(X_train)
cv_score = cross_val_score(model, X_selected, y)  # ❌ LEAKAGE!
```

---

## 📊 PCA (Fit on Train Fold Only)

### Component Selection
- `int`: Keep exact number (e.g., 50)
- `float`: Keep enough for variance (e.g., 0.95 for 95%)
- `None`: Keep all components

### ✅ CORRECT Usage

```python
from qsar_validation.pca_module import PCATransformer
from sklearn.model_selection import KFold

# Within each CV fold
for train_idx, val_idx in KFold(n_splits=5).split(X_train):
    pca = PCATransformer(n_components=0.95)  # Keep 95% variance
    
    # Fit on THIS fold's training data only
    pca.fit(X_train[train_idx])  # ✓
    
    # Transform train and val
    X_train_pca = pca.transform(X_train[train_idx])
    X_val_pca = pca.transform(X_train[val_idx])
    
    # Train model
    model.fit(X_train_pca, y_train[train_idx])
```

### ❌ INCORRECT Usage (LEAKAGE!)

```python
# WRONG: Fitting PCA before CV
pca = PCATransformer(n_components=0.95)
pca.fit(X_train)  # ❌ LEAKAGE!
X_pca = pca.transform(X_train)
cv_score = cross_val_score(model, X_pca, y)  # ❌ LEAKAGE!
```

---

## 🔗 Complete Pipeline (No Leakage!)

Combine all three: scaling → selection → PCA

```python
from qsar_validation.feature_scaling import FeatureScaler
from qsar_validation.feature_selection import FeatureSelector
from qsar_validation.pca_module import PCATransformer
from sklearn.model_selection import KFold

# All steps within each CV fold
for train_idx, val_idx in KFold(n_splits=5).split(X_train):
    X_train_fold = X_train[train_idx]
    y_train_fold = y_train[train_idx]
    X_val_fold = X_train[val_idx]
    y_val_fold = y_train[val_idx]
    
    # Step 1: Scale features
    scaler = FeatureScaler(method='standard')
    scaler.fit(X_train_fold)
    X_train_scaled = scaler.transform(X_train_fold)
    X_val_scaled = scaler.transform(X_val_fold)
    
    # Step 2: Select features
    selector = FeatureSelector(method='univariate', n_features=50)
    selector.fit(X_train_scaled, y_train_fold)
    X_train_selected = selector.transform(X_train_scaled)
    X_val_selected = selector.transform(X_val_scaled)
    
    # Step 3: Apply PCA
    pca = PCATransformer(n_components=0.95)
    pca.fit(X_train_selected)
    X_train_pca = pca.transform(X_train_selected)
    X_val_pca = pca.transform(X_val_selected)
    
    # Step 4: Train model
    model.fit(X_train_pca, y_train_fold)
    score = model.score(X_val_pca, y_val_fold)
    
    print(f"Fold score: {score:.4f}")
```

---

## 📚 Examples

See comprehensive examples in:

- **Splitting Strategies:** `examples/splitting_strategies_examples.py`
  - Example 1: Scaffold-based splitting (recommended)
  - Example 2: Time-based splitting
  - Example 3: Leave-cluster-out splitting
  - Example 4: Compare all three strategies
  - Example 5: Decision tree for choosing strategy

- **Feature Engineering:** `examples/feature_engineering_examples.py`
  - Example 1: Feature scaling in CV
  - Example 2: Feature selection in nested CV
  - Example 3: PCA in CV
  - Example 4: Complete pipeline
  - Example 5: Compare different approaches

---

## 🚨 Common Mistakes

### Mistake 1: Fitting scaler on all data
```python
# ❌ WRONG
scaler.fit(X_train)  # Before CV loop
```
**Fix:** Fit within each CV fold

### Mistake 2: Selecting features before CV
```python
# ❌ WRONG
selector.fit(X_train, y_train)  # Before CV loop
```
**Fix:** Select features within each CV fold

### Mistake 3: Fitting PCA on all data
```python
# ❌ WRONG
pca.fit(X_train)  # Before CV loop
```
**Fix:** Fit PCA within each CV fold

### Mistake 4: Using test data for any fitting
```python
# ❌ WRONG
scaler.fit(np.vstack([X_train, X_test]))  # Including test data
```
**Fix:** NEVER use test data for fitting

---

## ✅ Checklist

Before running your model:

- [ ] Did you remove duplicates BEFORE splitting?
- [ ] Did you use scaffold/temporal/cluster splitting?
- [ ] Are you fitting scaler WITHIN each CV fold?
- [ ] Are you selecting features WITHIN each CV fold?
- [ ] Are you fitting PCA WITHIN each CV fold?
- [ ] Did you verify no test data is used for fitting?

If all checked: **No data leakage! ✓**

---

**For full documentation, see:** `README.md`

**For implementation details, see:**
- `src/qsar_validation/splitting_strategies.py`
- `src/qsar_validation/feature_scaling.py`
- `src/qsar_validation/feature_selection.py`
- `src/qsar_validation/pca_module.py`
