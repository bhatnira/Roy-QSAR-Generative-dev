# 🛡️ Data Leakage Prevention for QSAR Models

## Overview

This repository contains fixed versions of QSAR models with comprehensive data leakage prevention. Data leakage is a **critical issue** in low-data QSAR that leads to **overly optimistic** performance estimates and models that fail in real-world applications.

## 📁 Files in This Repository

### Core Utilities
- **`qsar_utils_no_leakage.py`** - Comprehensive utilities for data leakage prevention
  - Scaffold-based splitting
  - Duplicate detection and removal
  - Feature scaling without leakage
  - Similarity analysis
  - Applicability domain estimation

### Example Notebook
- **`DATA_LEAKAGE_FIX_EXAMPLE.ipynb`** - Complete step-by-step guide
  - Demonstrates all leakage prevention techniques
  - Shows proper workflow from data loading to model evaluation
  - Includes visualization and reporting

### Original Models (with leakage issues)
1. `Model_1_circular_fingerprint_features_1024_H20_autoML_Model_Interpretation (1).ipynb`
2. `Model_2_ChEBERTa_embedding_linear_regression_no_interpretation (2).ipynb`
3. `Model_3_rdkit_features_H20_autoML.ipynb`
4. `Model_4_circular_fingerprint_features_1024_Gaussian_Process_Bayesian_Optimization_Model_Interpretation.ipynb`

---

## ⚠️ Critical Data Leakage Issues Found

### 1. Random Splitting (Most Critical)

**❌ What was wrong:**
```python
# Random split at row level
KFold(n_splits=5, shuffle=True, random_state=42)
```

**Problem:**
- Same or highly similar compounds in train AND test
- Same scaffold in both splits
- Replicates split across sets
- **Result:** Model sees test data during training!

**✅ Solution:**
```python
from qsar_utils_no_leakage import ScaffoldSplitter

splitter = ScaffoldSplitter(smiles_col='Canonical SMILES')
train_idx, val_idx, test_idx = splitter.scaffold_split(
    df, test_size=0.2, val_size=0.1, random_state=42
)
```

**Why it works:**
- Uses Bemis-Murcko scaffolds
- Entire scaffolds assigned to train/val/test (never split)
- Zero scaffold overlap between splits
- More realistic performance estimation

---

### 2. Duplicate and Near-Duplicate Molecules

**❌ What was wrong:**
- Duplicates removed AFTER splitting (or not at all)
- Same SMILES appearing in train and test
- Near-identical molecules (Tanimoto > 0.95) in different splits

**✅ Solution:**
```python
from qsar_utils_no_leakage import QSARDataProcessor

processor = QSARDataProcessor(smiles_col='Canonical SMILES', target_col='IC50 uM')

# Step 1: Canonicalize SMILES
df = processor.canonicalize_smiles(df)

# Step 2: Remove exact duplicates (average replicates) BEFORE splitting
df = processor.remove_duplicates(df, strategy='average')

# Step 3: After splitting, remove near-duplicates
train_idx, test_idx = processor.remove_near_duplicates(
    df, train_idx, test_idx, threshold=0.95
)
```

---

### 3. Feature Scaling Leakage (Often Missed!)

**❌ What was wrong:**
```python
# DON'T DO THIS - Uses test statistics!
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_all_scaled = scaler.fit_transform(X_all)  # ❌ LEAKAGE!

# Then split...
X_train, X_test = train_test_split(X_all_scaled)
```

**Problem:**
- Scaler learns mean/std from FULL dataset (including test)
- Test statistics leak into training
- Model has unfair advantage

**✅ Solution:**
```python
# FIT on training data ONLY
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # ✅ FIT

# TRANSFORM (not fit_transform!) validation and test
X_val_scaled = scaler.transform(X_val)  # ✅ TRANSFORM only
X_test_scaled = scaler.transform(X_test)  # ✅ TRANSFORM only
```

---

### 4. Target Transformation Leakage

**❌ What was wrong:**
```python
# Normalizing target using full dataset statistics
df['IC50_normalized'] = (df['IC50'] - df['IC50'].mean()) / df['IC50'].std()

# Then split...
```

**✅ Solution Option 1 (Recommended):**
```python
# Transform BEFORE splitting (pIC50 doesn't use dataset statistics)
def to_pIC50(ic50_uM):
    return -np.log10(ic50_uM * 1e-6)

df['pIC50'] = df['IC50 uM'].apply(to_pIC50)
# NOW split the data
```

**✅ Solution Option 2:**
```python
# If normalizing, fit on train only
from sklearn.preprocessing import StandardScaler

target_scaler = StandardScaler()
y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1))
y_val_scaled = target_scaler.transform(y_val.reshape(-1, 1))
y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1))
```

---

### 5. Cross-Validation Strategy

**❌ What was wrong:**
```python
# Random K-Fold
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
```

**✅ Solution:**
```python
from qsar_utils_no_leakage import ScaffoldSplitter

splitter = ScaffoldSplitter(smiles_col='Canonical SMILES')
cv_splits = splitter.scaffold_kfold(df, n_splits=5, random_state=42)

for fold, (train_idx, val_idx) in enumerate(cv_splits):
    # Train on this fold
    # Each fold has zero scaffold overlap
```

---

## 🔧 How to Fix Your Models

### Step-by-Step Guide

#### 1. Start with the Example Notebook
```bash
# Open the example notebook first
jupyter notebook DATA_LEAKAGE_FIX_EXAMPLE.ipynb
```

This notebook demonstrates:
- Complete workflow from start to finish
- All leakage prevention techniques
- Proper evaluation and reporting

#### 2. Apply to Your Models

For each of your four models, follow this template:

```python
# Import utilities
from qsar_utils_no_leakage import QSARDataProcessor, ScaffoldSplitter

# Initialize
processor = QSARDataProcessor(smiles_col='Canonical SMILES', target_col='IC50 uM')
splitter = ScaffoldSplitter(smiles_col='Canonical SMILES')

# Step 1: Clean data (BEFORE splitting!)
df = processor.canonicalize_smiles(df)
df = processor.remove_duplicates(df, strategy='average')

# Step 2: Scaffold-based split
train_idx, val_idx, test_idx = splitter.scaffold_split(df, test_size=0.2, val_size=0.1)

# Step 3: Remove near-duplicates
train_idx, test_idx = processor.remove_near_duplicates(df, train_idx, test_idx)

# Step 4: Analyze similarity (for reporting)
similarity_stats = processor.analyze_similarity(df, train_idx, test_idx)

# Step 5: Generate features for each split
# ... your feature generation code ...

# Step 6: Scale features (FIT on train only!)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # FIT
X_val_scaled = scaler.transform(X_val)  # TRANSFORM only
X_test_scaled = scaler.transform(X_test)  # TRANSFORM only

# Step 7: Train and evaluate
# ... your model training code ...
```

---

## 📊 What to Expect After Fixing Leakage

### Performance Changes

| Metric | Before (with leakage) | After (no leakage) | Change |
|--------|----------------------|-------------------|---------|
| Train R² | 0.95 | 0.85 | -10% |
| **Test R²** | **0.90** | **0.65** | **-25%** |
| Test RMSE | 0.3 | 0.5 | +67% |

**This is NORMAL and EXPECTED!**

### Why Performance Drops

1. **Model no longer sees similar molecules in test**
   - Scaffold-based split creates harder test set
   - More realistic of real-world performance

2. **No information leakage**
   - Test statistics don't influence training
   - Fair evaluation

3. **Better science**
   - Results are reproducible
   - Reviewers will appreciate rigor
   - Model is actually useful in practice

---

## 📝 Reporting for Publication

### What Reviewers Want to See

#### 1. Data Splitting Section
```markdown
### Data Splitting

We employed scaffold-based splitting using Bemis-Murcko scaffolds to prevent
data leakage. Entire molecular scaffolds were assigned exclusively to either
the training, validation, or test set, ensuring zero scaffold overlap between
splits.

**Split sizes:**
- Training: N=X molecules (Y%)
- Validation: N=X molecules (Y%)
- Test: N=X molecules (Y%)

**Scaffold overlap:** 0% (confirmed)
```

#### 2. Similarity Analysis
```markdown
### Train-Test Similarity

We analyzed the maximum Tanimoto similarity between test molecules and the
training set using Morgan fingerprints (radius=2, 2048 bits):

- Mean maximum similarity: 0.65 ± 0.12
- Median maximum similarity: 0.67
- Range: [0.35, 0.89]

[Include similarity distribution plot here]
```

#### 3. Leakage Prevention Steps
```markdown
### Data Leakage Prevention

The following steps were taken to prevent data leakage:

1. **Duplicate removal:** SMILES were canonicalized and exact duplicates were
   removed before splitting (N=X duplicates found and averaged).

2. **Near-duplicate removal:** Test molecules with Tanimoto similarity ≥0.95
   to any training molecule were removed (N=X removed).

3. **Feature scaling:** StandardScaler was fitted on training data only,
   then applied to validation and test sets.

4. **Cross-validation:** 5-fold scaffold-based cross-validation was used for
   hyperparameter tuning, ensuring no scaffold overlap between folds.

5. **Target transformation:** pIC50 transformation was applied before data
   splitting to avoid information leakage.
```

#### 4. Applicability Domain
```markdown
### Applicability Domain

We evaluated the applicability domain by calculating the average Tanimoto
similarity to the 5 nearest neighbors in the training set:

- X% of test molecules within AD (similarity ≥ 0.5)
- Y% of test molecules outside AD (similarity < 0.5)
```

---

## 🎯 Quick Reference Checklist

Before finalizing your QSAR model, verify:

### Data Preparation
- [ ] SMILES canonicalized
- [ ] Exact duplicates removed BEFORE splitting
- [ ] Duplicate measurement strategy documented (average/first/best)
- [ ] Invalid/missing values handled

### Splitting Strategy
- [ ] Scaffold-based splitting used (NOT random!)
- [ ] Test size appropriate (15-25%)
- [ ] Validation set included
- [ ] Scaffold overlap = 0% (verified)
- [ ] Near-duplicates removed (Tanimoto ≥ 0.95)

### Feature Engineering
- [ ] Features generated AFTER splitting (or per-split)
- [ ] Scaler/normalizer fitted on TRAIN only
- [ ] Scaler applied (not re-fitted) to val/test
- [ ] No feature selection on full dataset
- [ ] PCA (if used) fitted on train only

### Target Variable
- [ ] Transformation applied before splitting OR fitted on train only
- [ ] No target-derived features
- [ ] Target statistics from train only

### Cross-Validation
- [ ] Scaffold-based K-fold (NOT random!)
- [ ] Nested CV for hyperparameter tuning
- [ ] Proper pipeline in each fold

### Reporting
- [ ] Split strategy clearly described
- [ ] Train-test similarity analysis included
- [ ] Similarity distribution plot included
- [ ] Scaffold overlap reported (should be 0%)
- [ ] Applicability domain analyzed
- [ ] Model complexity justified for dataset size
- [ ] Leakage prevention steps documented

### External Validation (if available)
- [ ] Completely unseen compounds
- [ ] No SMILES overlap with training
- [ ] Different source/assay (ideally)
- [ ] Performance on external set reported

---

## 🚀 Model-Specific Instructions

### Model 1: Circular Fingerprints + H2O AutoML

**Key changes needed:**
1. Replace random split with scaffold split
2. Generate fingerprints per-split (not on full dataset)
3. Use scaffold-based CV for H2O AutoML
4. Ensure H2O doesn't use full dataset for any preprocessing

### Model 2: ChEBERTa Embeddings + Linear Regression

**Key changes needed:**
1. Scaffold-based split before generating embeddings
2. Generate embeddings per-split
3. Check if ChEBERTa model itself needs any preprocessing
4. Linear regression should be fine (no hyperparameters to leak)

### Model 3: RDKit Features + H2O AutoML

**Key changes needed:**
1. Scaffold-based split
2. Calculate RDKit descriptors per-split
3. Standardize descriptors (fit on train only!)
4. Use scaffold-based CV for AutoML

### Model 4: Circular Fingerprints + Gaussian Process + Bayesian Optimization

**Key changes needed:**
1. Scaffold-based split
2. Generate fingerprints per-split
3. Bayesian optimization hyperparameters should be tuned using scaffold-based CV
4. GP kernel hyperparameters fitted on train only

---

## 📚 Additional Resources

### Papers on Data Leakage in QSAR
1. Sheridan, R. P. (2013). "Time-Split Cross-Validation as a Method for Estimating the Goodness of Prospective Prediction." *J. Chem. Inf. Model.*
2. Ramsundar, B., et al. (2015). "Massively Multitask Networks for Drug Discovery." *arXiv:1502.02072*
3. Mayr, A., et al. (2018). "Large-scale comparison of machine learning methods for drug target prediction on ChEMBL." *Chem. Sci.*

### Best Practices Guidelines
- MoleculeNet benchmarking suite (scaffold splitting as standard)
- Therapeutics Data Commons (TDC) guidelines
- DeepChem best practices documentation

---

## 💡 Common Questions

### Q: Will my performance drop significantly?
**A:** Yes, typically 10-30% drop in test R². This is **expected and good**! It means you're getting realistic estimates.

### Q: Should I report both results?
**A:** Report only the corrected results. The old results with leakage are not scientifically valid.

### Q: What if my dataset is too small for scaffold splitting?
**A:** 
1. Use repeated scaffold-based CV (e.g., 5-fold repeated 10 times)
2. Consider cluster-based splitting as alternative
3. Report that the dataset may be too small for robust evaluation
4. Emphasize applicability domain in predictions

### Q: Can I use random splitting if I report it?
**A:** No. Random splitting with similar molecules in train/test is a methodological flaw, even if disclosed. Use scaffold-based splitting.

### Q: What about temporal/time-based splitting?
**A:** Excellent if you have time information! Even better than scaffold splitting for prospective prediction. Use it!

---

## 🛠️ Troubleshooting

### Issue: Scaffold splitting creates very unbalanced splits
**Solution:** 
- Increase dataset size if possible
- Use stratified scaffold splitting (preserve activity distribution)
- Report the imbalance and discuss limitations

### Issue: Too many near-duplicates being removed
**Solution:**
- Lower the threshold (try 0.90 instead of 0.95)
- Report how many were removed and why
- Consider if your dataset has too little chemical diversity

### Issue: Code is too slow for large datasets
**Solution:**
- Use multiprocessing for fingerprint generation
- Consider sampling for similarity analysis
- Cache computed fingerprints

---

## 📞 Support

If you encounter issues or have questions:
1. Check the example notebook first
2. Review this README thoroughly
3. Verify each step in the checklist
4. Check that you're using the latest version of utilities

---

## 🎓 Citation

If you use these utilities in your research, please cite:

```bibtex
@software{qsar_utils_no_leakage,
  title = {Data Leakage Prevention Utilities for QSAR Models},
  author = {Your Name},
  year = {2026},
  note = {Implements best practices for low-data QSAR modeling}
}
```

---

## ✅ Final Checklist Before Publication

- [ ] All models use scaffold-based splitting
- [ ] Zero scaffold overlap verified
- [ ] Feature scaling done correctly
- [ ] Train-test similarity analyzed and plotted
- [ ] Applicability domain reported
- [ ] All leakage prevention steps documented
- [ ] Performance drops acknowledged (if applicable)
- [ ] Code and data available for reproducibility
- [ ] External validation performed (if possible)

---

**Remember:** Lower performance with proper methodology is **infinitely better** than inflated performance with data leakage. Reviewers will appreciate the rigor, and your model will actually work in practice!

🎯 **Goal:** Publishable, reproducible, and reliable QSAR models that work in the real world.
