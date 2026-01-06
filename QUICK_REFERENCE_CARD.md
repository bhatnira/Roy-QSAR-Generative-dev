# 🎯 QSAR VALIDATION QUICK REFERENCE CARD

## ⚡ 30-Second Summary

**What Changed:**
- ✅ Added comprehensive validation to all 4 notebooks
- ✅ Created `qsar_validation_utils.py` module
- ✅ Addressed 13+ critical issues beyond data leakage

**What to Do Next:**
1. Run validation cells in each notebook
2. Update feature generation (per split!)
3. Update scaling (training only!)
4. Run baseline & y-randomization tests

---

## 📊 Critical Thresholds

| Metric | Threshold | Meaning |
|--------|-----------|---------|
| **Scaffold diversity** | > 0.3 | Good diversity |
| **Samples:Features** | > 10:1 | Adequate for complex models |
| **Samples:Features** | 5-10:1 | Need regularization |
| **Samples:Features** | < 5:1 | 🔴 High overfitting risk |
| **RMSE (IC₅₀)** | ~0.5 | Near theoretical limit |
| **RMSE (IC₅₀)** | < 0.3 | ⚠️ Suspicious (check leakage) |
| **R² (y-random)** | ≤ 0.0 | ✅ Model not overfitting |
| **R² (y-random)** | > 0.2 | 🔴 Overfitting detected |

---

## 🔍 Quick Validation Code

### Run This First
```python
# Import validation utilities
from qsar_validation_utils import (
    DatasetBiasAnalyzer, ActivityCliffDetector,
    ModelComplexityAnalyzer, YRandomizationTester,
    PerformanceMetricsCalculator, AssayNoiseEstimator
)
```

### Check Dataset Bias
```python
analyzer = DatasetBiasAnalyzer(smiles_col='Canonical SMILES', target_col='IC50 uM')
diversity_results = analyzer.analyze_scaffold_diversity(df)
activity_stats = analyzer.analyze_activity_distribution(df)
```

**Look for:**
- Diversity ratio < 0.3 → Congeneric series
- Top scaffold > 50% → High imbalance
- Activity range < 2x mean → Narrow range

### Check Model Complexity
```python
ModelComplexityAnalyzer.analyze_complexity(
    n_samples=len(train_data),
    n_features=n_features,
    model_type='random_forest'  # or 'linear', 'gaussian_process', etc.
)
```

**Look for:**
- Samples:Features < 5:1 → 🔴 Critical
- Samples:Features < 10:1 → 🟠 Warning

### Detect Activity Cliffs
```python
cliff_detector = ActivityCliffDetector(smiles_col='Canonical SMILES', target_col='IC50 uM')
cliff_df = cliff_detector.detect_activity_cliffs(
    df, similarity_threshold=0.85, activity_threshold=2.0
)
```

**If cliffs found:**
- Feature importance less reliable
- Consider Gaussian Process
- Use local models

### Estimate Assay Noise
```python
noise_estimator = AssayNoiseEstimator()
error = noise_estimator.estimate_experimental_error(df, 'IC50 uM')
```

**Remember:**
- IC₅₀ typical error: 0.3-0.6 log units
- Target RMSE: ~0.5 log units
- RMSE < assay error → suspicious

---

## 🎯 Workflow Checklist

### Before Training
- [ ] Canonicalize SMILES
- [ ] Remove exact duplicates (average replicates)
- [ ] **NEW:** Run scaffold diversity analysis
- [ ] **NEW:** Check activity distribution
- [ ] **NEW:** Analyze model complexity
- [ ] **NEW:** Detect activity cliffs
- [ ] Scaffold-based split
- [ ] Remove near-duplicates (Tanimoto ≥ 0.95)
- [ ] Verify zero SMILES overlap

### During Training
- [ ] Generate features **per split** (not before!)
- [ ] Fit scaler on **training only**
- [ ] Use appropriate regularization
- [ ] Use scaffold-based CV (not random)
- [ ] Nested CV for hyperparameter tuning

### After Training
- [ ] **NEW:** Calculate baseline (Ridge) performance
- [ ] **NEW:** Run y-randomization test (n=10)
- [ ] **NEW:** Calculate ALL metrics (RMSE, MAE, R², Spearman)
- [ ] **NEW:** Compare RMSE to assay error
- [ ] **NEW:** Check uncertainty estimates (if GP)
- [ ] **NEW:** Define applicability domain

---

## 📈 Expected Performance

### Typical Changes After Fixes

| Stage | R² | RMSE | Notes |
|-------|-----|------|-------|
| **Before (Random Split)** | 0.80-0.85 | 0.25-0.30 | ⚠️ Optimistic |
| **After (Scaffold Split)** | 0.55-0.70 | 0.40-0.55 | ✅ Realistic |
| **Near-Optimal IC₅₀** | 0.60-0.75 | ~0.50 | ✅ Excellent |

**This drop is EXPECTED and CORRECT!**

---

## ⚠️ Common Warnings & Fixes

### "Low scaffold diversity (< 0.3)"
```python
# Action: State in paper
"Dataset represents congeneric series with limited 
chemical diversity (diversity ratio = X.XX). Model 
applicability limited to similar scaffolds."
```

### "Samples:Features < 5:1"
```python
# Action: Use strong regularization
from sklearn.linear_model import Ridge, Lasso
model = Ridge(alpha=10.0)  # Increase alpha

# OR reduce features
from sklearn.decomposition import PCA
pca = PCA(n_components=min(n_samples//5, n_features))
```

### "Activity cliffs detected"
```python
# Action: Use Gaussian Process
from sklearn.gaussian_process import GaussianProcessRegressor
model = GaussianProcessRegressor(...)

# OR state limitation
"X activity cliffs limit local predictivity. 
Feature importance should be interpreted cautiously."
```

### "RMSE < 0.3 log units"
```python
# Action: Check for issues
1. Data leakage? (verify zero SMILES overlap)
2. Overfitting? (run y-randomization test)
3. Lucky split? (report CV metrics)

# If legitimate:
"RMSE (X.XX) below typical assay error (0.5) 
suggests highly optimized model or favorable 
dataset characteristics."
```

### "R² > 0.2 with y-randomization"
```python
# Action: Simplify model
1. Reduce features (PCA, feature selection)
2. Increase regularization (higher alpha)
3. Use simpler model (Ridge instead of RF)
4. Check samples:features ratio
```

---

## 📝 Paper Reporting Template

### Methods
```
Data were split using Bemis-Murcko scaffold-based 
splitting to prevent data leakage. Dataset contains 
X molecules across Y unique scaffolds (diversity 
ratio: Z). Feature scaling was performed using 
training set statistics only.

Model complexity (samples:features = X:Y) was 
controlled via [regularization method]. Cross-
validation used scaffold-based 5-fold splitting.

Performance was compared to Ridge regression baseline 
(RMSE = X.XX). Y-randomization test confirmed absence 
of overfitting (R² = X.XX ± X.XX, n=10).
```

### Results Table
```
|           | RMSE  | MAE   | R²    | Spearman ρ |
|-----------|-------|-------|-------|------------|
| Train     | X.XX  | X.XX  | X.XX  | X.XX       |
| Val (CV)  | X.XX±X| X.XX±X| X.XX±X| X.XX±X     |
| Test      | X.XX  | X.XX  | X.XX  | X.XX       |
| Baseline  | X.XX  | X.XX  | X.XX  | X.XX       |
```

### Discussion
```
Model RMSE (X.XX) is [near/within/above] typical 
IC₅₀ assay precision (~0.5 log units). Scaffold-
based validation ensures generalization to novel 
scaffolds. X activity cliffs were detected, 
[limiting/enabling] local predictivity.

Applicability domain: Predictions reliable for 
molecules with Tanimoto similarity ≥ 0.5 to 
training set.
```

---

## 🚀 Model-Specific Notes

### Model 1: Circular Fingerprints + H2O
- Features: 1024 bits
- Need: Regularization
- Generate: Per split
- Scale: Yes (training only)

### Model 2: ChEBERTa + Linear
- Features: 768 dim (pre-trained)
- Need: Less regularization (pre-trained)
- Generate: Per split
- Scale: **NO** (pre-trained embeddings)

### Model 3: RDKit + H2O
- Features: ~200 descriptors
- Need: Regularization (correlated)
- Generate: Per split
- Scale: Yes (training only)

### Model 4: Circular FP + GP
- Features: 1024 bits
- Need: Moderate (GP handles well)
- Generate: Per split
- Scale: Yes (training only)
- Bonus: Built-in uncertainty!

---

## 📚 Files Reference

| File | Purpose | Lines |
|------|---------|-------|
| `qsar_utils_no_leakage.py` | Core leakage prevention | 549 |
| `qsar_validation_utils.py` | Comprehensive validation | 800+ |
| `COMPREHENSIVE_VALIDATION_GUIDE.md` | Detailed guide | 60+ pages |
| `COMPLETE_VALIDATION_SUMMARY.md` | Full summary | This doc |
| `START_HERE.md` | Quick start | Quick ref |

---

## 🎓 Key Principles

1. **Scaffold split mandatory** - Tests true generalization
2. **Features per split** - Prevents leakage
3. **Scale on training only** - Prevents leakage
4. **Report all metrics** - Not just R²
5. **Compare to baseline** - Ridge regression
6. **Test y-randomization** - Detect overfitting
7. **Compare to assay error** - ~0.5 for IC₅₀
8. **State limitations** - Honest reporting
9. **Simpler often better** - Low-data regime
10. **Uncertainty matters** - Applicability domain

---

## ⏱️ Time Estimates

| Task | Time | Priority |
|------|------|----------|
| Run validation cells | 5 min | 🔴 Now |
| Update feature generation | 30 min | 🔴 Today |
| Update scaling | 10 min | 🔴 Today |
| Baseline comparison | 15 min | 🟠 This week |
| Y-randomization test | 30 min | 🟠 This week |
| Comprehensive metrics | 15 min | 🟠 This week |
| Uncertainty analysis | 1 hour | 🟡 Next week |
| Paper writing | 4-8 hours | 🟡 Publication |

---

## 🆘 Quick Help

**Performance too low?**
- Check if scaffold split is too harsh (expected!)
- Compare to baseline (are you better?)
- Check assay noise (~0.5 is theoretical limit)
- Try Gaussian Process (Model 4)

**Performance suspiciously high?**
- Verify zero SMILES overlap
- Run y-randomization test
- Check samples:features ratio
- Compare to assay precision

**Overfitting detected?**
- Reduce features (PCA)
- Increase regularization
- Simplify model
- Use simpler baseline

**Need help?**
- Read: `COMPREHENSIVE_VALIDATION_GUIDE.md`
- Check: Validation cell outputs
- Review: Model-specific notes above

---

**🎯 Remember: Lower performance with proper validation is BETTER than high performance with data leakage!**

---

*Quick Reference Card - January 2026*  
*Keep this handy while running validation!*
