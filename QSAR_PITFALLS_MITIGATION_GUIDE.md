# QSAR Pitfalls Mitigation Guide

**Comprehensive guide to avoiding common QSAR modeling pitfalls**

---

## Overview

This framework provides **mitigation tools** for 13 common QSAR pitfalls. Each tool is an independent module that you can use as needed.

---

## 🎯 Quick Mitigation Checklist

Before publishing QSAR results:

- [ ] **Dataset Quality** → Run `DatasetQualityAnalyzer`
- [ ] **Model Complexity** → Use `ModelComplexityController`
- [ ] **Cross-Validation** → Use `PerformanceValidator.cross_validate_properly()`
- [ ] **Y-Randomization** → Run `PerformanceValidator.y_randomization_test()`
- [ ] **Activity Cliffs** → Run `ActivityCliffsDetector`
- [ ] **Uncertainty** → Use `UncertaintyEstimator`
- [ ] **Baseline** → Compare to `PerformanceValidator.compare_to_baseline()`
- [ ] **Report All Metrics** → R², RMSE, MAE, Spearman ρ
- [ ] **Fixed Seeds** → Set `random_state` everywhere
- [ ] **Document Splits** → Save train/val/test indices

---

## 📊 Pitfall-by-Pitfall Mitigation

### 1. Dataset Bias & Representativeness

**Problem:** Narrow chemical space, congeneric series, scaffold imbalance

**Mitigation:**
```python
from qsar_validation.dataset_quality_analysis import DatasetQualityAnalyzer

analyzer = DatasetQualityAnalyzer(
    smiles_col='SMILES',
    activity_col='pIC50'
)

# Comprehensive dataset analysis
report = analyzer.analyze(df)

# Get splitting recommendations
recommendations = analyzer.generate_split_recommendations(df)
```

**What it checks:**
- Scaffold diversity (Bemis-Murcko)
- Chemical space coverage (Tanimoto similarity)
- Activity distribution balance
- Sample size adequacy

**Warnings to watch for:**
- ⚠️ Scaffold diversity < 0.3 → Congeneric dataset
- ⚠️ Top scaffold > 50% → Dominated by single series
- ⚠️ Mean Tanimoto > 0.7 → Narrow chemical space
- ⚠️ Activity range < 2 log units → Limited range

**Actions:**
- Report scaffold diversity in paper
- Use appropriate splitting strategy
- Limit claims to studied chemical space
- Avoid claiming "general" QSAR model

---

### 2. Overfitting Due to Model Complexity

**Problem:** Too many parameters, deep models, excessive tuning

**Mitigation:**
```python
from qsar_validation.model_complexity_control import ModelComplexityController

controller = ModelComplexityController(
    n_samples=X_train.shape[0],
    n_features=X_train.shape[1]
)

# Get model recommendations
recommendations = controller.recommend_models()

# Get safe hyperparameter ranges
param_grid = controller.get_safe_param_grid('random_forest')

# Run nested CV
results = controller.nested_cv(X, y, model_type='ridge')
```

**What it checks:**
- Sample-to-feature ratio
- Model complexity vs dataset size
- Hyperparameter ranges

**Warnings to watch for:**
- ⚠️ Ratio < 0.1 → Only linear models!
- ⚠️ Ratio < 0.5 → Shallow models only
- ⚠️ High CV std → Overfitting

**Actions:**
- Use Ridge/Lasso for N < 100
- Limit tree depth: `max_depth ≤ 5` for small data
- Always use nested CV for hyperparameter tuning
- Report mean ± std from CV

---

### 3. Improper Cross-Validation Design

**Problem:** Random CV, CV for feature selection, reporting best fold

**Mitigation:**
```python
from qsar_validation.performance_validation import PerformanceValidator

validator = PerformanceValidator()

# Proper CV with complete reporting
cv_results = validator.cross_validate_properly(
    X_train, y_train, model, n_folds=5
)

# Reports mean ± std for R², RMSE, MAE, Spearman ρ
```

**Best practices:**
- Use scaffold-based CV (not random!)
- Feature selection → nested CV only
- Always report mean ± std
- Never report "best fold"

---

### 4. Measurement Noise & Assay Variability

**Problem:** Mixed assays, large experimental error, ignoring uncertainty

**Mitigation:**
```python
from qsar_validation.dataset_quality_analysis import DatasetQualityAnalyzer

analyzer = DatasetQualityAnalyzer(
    smiles_col='SMILES',
    activity_col='pIC50',
    assay_col='Assay_Type'  # If available
)

report = analyzer.analyze(df)
```

**What to check:**
- Multiple assay types → Report separately
- Expected experimental error (usually ±0.3-0.6 log units)
- RMSE < experimental error → Suspicious!

**Actions:**
- Convert to consistent units (pIC₅₀)
- Stratify by assay if multiple types
- Report expected assay error
- Don't claim RMSE = 0.1 if assay error = 0.5!

---

### 5. Activity Cliffs

**Problem:** Similar molecules, large activity differences

**Mitigation:**
```python
from qsar_validation.activity_cliffs_detection import ActivityCliffsDetector

detector = ActivityCliffsDetector(
    smiles_col='SMILES',
    activity_col='pIC50',
    similarity_threshold=0.85,
    activity_threshold=2.0
)

# Detect cliffs
cliffs_df = detector.detect_cliffs(df)

# Analyze implications
analysis = detector.analyze_cliffs(cliffs_df, df)

# Flag cliff molecules
df_flagged = detector.identify_cliff_regions(df, cliffs_df)
```

**What it detects:**
- Structurally similar pairs (Tanimoto > 0.85)
- Large activity differences (> 2 log units)
- Severity score

**Warnings:**
- ⚠️ Severity > 0.3 → Many cliffs
- Predictions near cliffs are unreliable
- Feature importance is unstable

**Actions:**
- Report cliff molecules
- Use Gaussian Process Regression
- Don't over-interpret feature importance
- Consider local models

---

### 6. Target Imbalance & Range Compression

**Problem:** Narrow activity range, clustered values, truncation

**Mitigation:**

Part of `DatasetQualityAnalyzer` (Pitfall #1 above)

**What to check:**
- Activity range < 2 log units → Narrow
- Skewness > 1.5 → Highly skewed
- Artificial cutoffs

**Actions:**
- Report activity distribution
- Use RMSE/MAE (not just R²)
- Avoid hard regression cutoffs

---

### 7. Descriptor & Fingerprint Limitations

**Problem:** High dimensionality, collinearity, non-interpretable

**Mitigation:**

Use feature engineering modules with proper leakage prevention:

```python
from qsar_validation.feature_scaling import FeatureScaler
from qsar_validation.feature_selection import FeatureSelector
from qsar_validation.pca_module import PCATransformer

# Within each CV fold:
for train_idx, val_idx in cv_folds:
    # Scale features
    scaler = FeatureScaler(method='standard')
    scaler.fit(X_train[train_idx])
    
    # Select features
    selector = FeatureSelector(method='univariate', n_features=50)
    selector.fit(X_train_scaled[train_idx], y_train[train_idx])
    
    # PCA if needed
    pca = PCATransformer(n_components=0.95)
    pca.fit(X_train_selected[train_idx])
```

**Best practices:**
- Prune correlated descriptors
- Use regularization (Ridge, Lasso)
- Feature selection in nested CV only
- Caution with SHAP interpretation

---

### 8. Poor Uncertainty Estimation

**Problem:** Point predictions only, no confidence intervals

**Mitigation:**
```python
from qsar_validation.uncertainty_estimation import UncertaintyEstimator

estimator = UncertaintyEstimator(method='both')

# Fit on training data
estimator.fit(X_train, y_train, model)

# Predict with uncertainty
results = estimator.predict_with_uncertainty(X_test)

# Analyze confidence
estimator.analyze_prediction_confidence(results, y_test)
```

**What it provides:**
- Ensemble uncertainty (Random Forest std)
- Distance-based uncertainty (nearest neighbor)
- Confidence intervals (95%)
- Applicability domain flags

**Actions:**
- Report predictions ± confidence intervals
- Flag out-of-domain predictions
- Use Gaussian Process for uncertainty
- Consider quantile regression

---

### 9. Improper Performance Metrics

**Problem:** R² only, no external test, no baseline

**Mitigation:**
```python
from qsar_validation.performance_validation import PerformanceValidator

validator = PerformanceValidator()

# Calculate ALL metrics
metrics = validator.calculate_comprehensive_metrics(y_test, y_pred)

# Compare to baseline
comparison = validator.compare_to_baseline(y_test, y_pred)
```

**Report these metrics:**
- ✓ R²
- ✓ RMSE
- ✓ MAE
- ✓ Spearman ρ
- ✓ Baseline comparison
- ✓ Y-randomization p-value

---

### 10. Lack of Baseline & Negative Controls

**Problem:** No simple model comparison, no randomization tests

**Mitigation:**
```python
from qsar_validation.performance_validation import PerformanceValidator

validator = PerformanceValidator()

# Y-randomization test
random_results = validator.y_randomization_test(
    X_train, y_train, model, n_iterations=100
)

# Baseline comparison
comparison = validator.compare_to_baseline(y_test, y_pred)
```

**What to report:**
- Y-randomization p-value
- Separation from random (σ)
- Comparison to Ridge regression baseline
- Improvement over mean prediction

**Pass criteria:**
- P-value < 0.05
- Separation > 2σ
- Beats baseline by > 0.05 R²

---

### 11. Interpretability Overclaiming

**Problem:** Overinterpreting SHAP, claiming mechanistic insights

**Mitigation:**

**Guidelines:**
- SHAP on correlated features → Unstable
- Fingerprints → Statistical associations, not mechanisms
- Activity cliffs → Feature importance unreliable

**Safe statements:**
- ✓ "Feature X is associated with higher activity"
- ✗ "Feature X causes higher activity"
- ✓ "Model suggests hypothesis for X"
- ✗ "Model proves mechanism of X"

**Actions:**
- Treat explanations as hypothesis-generating
- Validate mechanistically (not just computationally)
- Report instability in feature importance
- Avoid causal language

---

### 12. Reproducibility & Transparency

**Problem:** No code, no seeds, no dataset versioning

**Mitigation:**

**Checklist:**
- [ ] Fixed `random_state` everywhere
- [ ] Save train/val/test split indices
- [ ] Document preprocessing steps
- [ ] Share code (GitHub)
- [ ] Share data (if possible)
- [ ] Report all hyperparameters
- [ ] Report package versions

**Example:**
```python
# Set seeds
np.random.seed(42)

# Split with fixed seed
splitter = ScaffoldSplitter(smiles_col='SMILES')
train_idx, val_idx, test_idx = splitter.split(df, random_state=42)

# Save indices
np.save('train_indices.npy', train_idx)
np.save('val_indices.npy', val_idx)
np.save('test_indices.npy', test_idx)

# Document
print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
```

---

### 13. External Validity Overstatement

**Problem:** Claiming "general model", extrapolating beyond domain

**Mitigation:**

**Use uncertainty estimation + dataset quality analysis:**

```python
# Assess chemical space
analyzer = DatasetQualityAnalyzer(...)
report = analyzer.analyze(df)

# Check applicability domain
estimator = UncertaintyEstimator(...)
results = estimator.predict_with_uncertainty(X_new)

# Flag out-of-domain
n_out = (~results['in_domain']).sum()
if n_out > 0:
    print(f"⚠️ {n_out} predictions are outside training domain")
```

**Safe statements:**
- ✓ "Model applicable to [X] class of compounds"
- ✗ "General QSAR model for all compounds"
- ✓ "Validated on [Y] chemical space"
- ✗ "Can predict any molecule"

**Actions:**
- Define applicability domain
- Report training set chemical space
- Flag out-of-domain predictions
- Limit claims to validated space

---

## 📋 Complete Workflow Example

```python
from qsar_validation.dataset_quality_analysis import DatasetQualityAnalyzer
from qsar_validation.model_complexity_control import ModelComplexityController
from qsar_validation.performance_validation import PerformanceValidator
from qsar_validation.activity_cliffs_detection import ActivityCliffsDetector
from qsar_validation.uncertainty_estimation import UncertaintyEstimator
from qsar_validation.splitting_strategies import ScaffoldSplitter

# 1. Dataset Quality Analysis
print("Step 1: Analyzing dataset quality...")
quality_analyzer = DatasetQualityAnalyzer(smiles_col='SMILES', activity_col='pIC50')
quality_report = quality_analyzer.analyze(df)

# 2. Activity Cliffs Detection
print("\nStep 2: Detecting activity cliffs...")
cliff_detector = ActivityCliffsDetector(smiles_col='SMILES', activity_col='pIC50')
cliffs_df = cliff_detector.detect_cliffs(df)
cliff_analysis = cliff_detector.analyze_cliffs(cliffs_df, df)

# 3. Proper Splitting
print("\nStep 3: Splitting data...")
splitter = ScaffoldSplitter(smiles_col='SMILES')
train_idx, val_idx, test_idx = splitter.split(df, test_size=0.2, random_state=42)

# 4. Model Complexity Control
print("\nStep 4: Selecting appropriate model...")
complexity_controller = ModelComplexityController(
    n_samples=len(train_idx),
    n_features=X.shape[1]
)
model_recommendations = complexity_controller.recommend_models()

# 5. Nested CV with Y-Randomization
print("\nStep 5: Model validation...")
validator = PerformanceValidator()

# Cross-validation
cv_results = validator.cross_validate_properly(X_train, y_train, model, n_folds=5)

# Y-randomization test
random_results = validator.y_randomization_test(X_train, y_train, model, n_iterations=100)

# 6. Test Set Evaluation
print("\nStep 6: Test set evaluation...")
y_pred = model.predict(X_test)

# Comprehensive metrics
metrics = validator.calculate_comprehensive_metrics(y_test, y_pred, set_name='Test')

# Baseline comparison
comparison = validator.compare_to_baseline(y_test, y_pred)

# 7. Uncertainty Estimation
print("\nStep 7: Uncertainty estimation...")
uncertainty_estimator = UncertaintyEstimator(method='both')
uncertainty_estimator.fit(X_train, y_train, model)
uncertainty_results = uncertainty_estimator.predict_with_uncertainty(X_test)
uncertainty_estimator.analyze_prediction_confidence(uncertainty_results, y_test)

print("\n" + "="*80)
print("✓ Complete QSAR validation workflow finished!")
print("="*80)
```

---

## 📊 Reporting Template

Include these in your paper/report:

### Dataset Characteristics
- N samples (train/val/test split)
- N unique scaffolds
- Scaffold diversity score
- Chemical space coverage (mean Tanimoto)
- Activity range (min, max, mean ± std)
- Assay types (if multiple)

### Model Details
- Model type and complexity
- Hyperparameters (all of them!)
- Training procedure
- Random seeds used

### Performance Metrics
- Train/Val/Test R²
- Train/Val/Test RMSE
- Train/Val/Test MAE
- Spearman ρ
- All reported as mean ± std from CV

### Validation Controls
- Y-randomization p-value
- Separation from random (σ)
- Baseline comparison (ΔR²)
- Activity cliffs analysis
- Uncertainty estimates

### Limitations
- Chemical space applicability
- Activity cliff regions
- Out-of-domain predictions
- Experimental error bounds

---

## 🚨 Red Flags to Avoid

**Automatic rejection signals:**
- ❌ No cross-validation
- ❌ Random CV (not scaffold-based)
- ❌ No y-randomization test
- ❌ R² only (no RMSE/MAE)
- ❌ No baseline comparison
- ❌ RMSE < experimental error
- ❌ No code/data sharing
- ❌ Claiming "general" QSAR model
- ❌ Over-interpreting SHAP
- ❌ No reproducibility details

---

## ✅ Publication Checklist

Before submission:

- [ ] Dataset quality analyzed
- [ ] Scaffold diversity reported
- [ ] Activity cliffs identified
- [ ] Proper splitting strategy used
- [ ] Model complexity justified
- [ ] Nested CV performed
- [ ] Y-randomization test passed (p < 0.05)
- [ ] Baseline comparison done
- [ ] All metrics reported (R², RMSE, MAE, Spearman)
- [ ] Uncertainty estimates provided
- [ ] Applicability domain defined
- [ ] Fixed seeds documented
- [ ] Code shared
- [ ] Limitations clearly stated

---

For complete examples, see:
- `examples/splitting_strategies_examples.py`
- `examples/feature_engineering_examples.py`
