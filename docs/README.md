# 🧬 Comprehensive QSAR Validation Framework

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

> **Publication-ready QSAR models with comprehensive validation for low-data regimes**

This repository contains a complete framework for building, validating, and reporting QSAR (Quantitative Structure-Activity Relationship) models with proper data leakage prevention and comprehensive validation checks.

---

## 🚀 Quick Start

### New Users (Start Here!)

1. **Read the documentation**: [`README_COMPREHENSIVE.md`](README_COMPREHENSIVE.md) (10 minutes)
2. **Quick reference**: [`QUICK_REFERENCE_CARD.md`](QUICK_REFERENCE_CARD.md) (5 minutes)
3. **Run validation**: Open any notebook → Execute "Comprehensive Validation Analysis" section
4. **Navigate all resources**: [`INDEX.md`](INDEX.md)

### Experienced Users

```python
# Import validation utilities
from qsar_utils_no_leakage import QSARDataProcessor, ScaffoldSplitter
from qsar_validation_utils import (
    DatasetBiasAnalyzer, ActivityCliffDetector, 
    ModelComplexityAnalyzer, YRandomizationTester
)

# Run comprehensive validation
processor = QSARDataProcessor(smiles_col='SMILES', target_col='IC50')
df = processor.canonicalize_smiles(df)
df = processor.remove_duplicates(df, strategy='average')

# Scaffold-based split (prevents data leakage)
splitter = ScaffoldSplitter()
train_idx, val_idx, test_idx = splitter.scaffold_split(df, test_size=0.2, val_size=0.1)

# Analyze dataset
analyzer = DatasetBiasAnalyzer()
diversity_results = analyzer.analyze_scaffold_diversity(df)
cliff_df = ActivityCliffDetector().detect_activity_cliffs(df)
```

---

## 📁 Repository Contents

### 🎯 Core Utilities

| File | Description | Lines |
|------|-------------|-------|
| **`qsar_utils_no_leakage.py`** | Data leakage prevention (scaffold split, scaling, CV) | 549 |
| **`qsar_validation_utils.py`** | Comprehensive validation (13+ checks) | 800+ |

### 📓 QSAR Models (4 Complete Examples)

1. **Model 1**: Circular Fingerprints (1024-bit) + H2O AutoML
2. **Model 2**: ChEBERTa Pre-trained Embeddings + Linear Regression
3. **Model 3**: RDKit Molecular Descriptors + H2O AutoML
4. **Model 4**: Circular Fingerprints + Gaussian Process + Bayesian Optimization

All notebooks include comprehensive validation sections!

### 📚 Documentation (115+ Pages)

| File | Purpose | Read Time |
|------|---------|-----------|
| **[README_COMPREHENSIVE.md](README_COMPREHENSIVE.md)** | 📌 **START HERE** - Main entry point | 10 min |
| **[QUICK_REFERENCE_CARD.md](QUICK_REFERENCE_CARD.md)** | Quick lookup reference | 5 min |
| **[COMPREHENSIVE_VALIDATION_GUIDE.md](COMPREHENSIVE_VALIDATION_GUIDE.md)** | Detailed 13-issue guide | 1 hour |
| **[COMPLETE_VALIDATION_SUMMARY.md](COMPLETE_VALIDATION_SUMMARY.md)** | Implementation summary | 30 min |
| **[INDEX.md](INDEX.md)** | Navigation guide | 5 min |
| **[START_HERE.md](START_HERE.md)** | Data leakage quick start | 10 min |

---

## ✅ Issues Addressed (13+)

### 🔴 Critical Issues

1. ✅ **Data Leakage** - Scaffold-based splitting (Bemis-Murcko)
2. ✅ **Dataset Bias** - Scaffold diversity analysis (Gini coefficient)
3. ✅ **Model Overfitting** - Samples:features ratio control
4. ✅ **Improper Cross-Validation** - Scaffold-based CV (not random)

### 🟠 High Priority Issues

5. ✅ **Assay Noise** - Experimental error estimation (~0.5 log units for IC₅₀)
6. ✅ **Activity Cliffs** - SAR discontinuity detection
7. ✅ **Improper Metrics** - RMSE, MAE, R², Spearman ρ (not just R²)
8. ✅ **No Baseline** - Ridge regression comparison
9. ✅ **No Y-Randomization** - Overfitting test (should give R² ≤ 0)

### 🟡 Best Practices

10. ✅ **Uncertainty Estimation** - Gaussian Process confidence intervals
11. ✅ **Interpretability** - Guidelines to avoid overclaiming
12. ✅ **Reproducibility** - Code/data sharing templates
13. ✅ **Applicability Domain** - Similarity-based domain definition

---

## 🎯 Key Features

### Data Leakage Prevention
- ✅ Scaffold-based splitting (entire scaffolds in train OR test)
- ✅ Duplicate removal before splitting
- ✅ Near-duplicate detection (Tanimoto ≥ 0.95)
- ✅ Features scaled using training data only
- ✅ Verification checks (zero SMILES overlap)

### Comprehensive Validation
- ✅ Scaffold diversity analysis (congeneric series detection)
- ✅ Model complexity assessment (samples:features ratio)
- ✅ Activity cliff detection (SAR discontinuities)
- ✅ Assay noise estimation (realistic performance expectations)
- ✅ Y-randomization testing (overfitting detection)
- ✅ Baseline comparison (Ridge regression minimum)
- ✅ Comprehensive metrics (not just R²)

### Publication-Ready Templates
- ✅ Methods section templates
- ✅ Results table formats
- ✅ Discussion limitation statements
- ✅ Reproducibility checklists
- ✅ Performance reporting guidelines

---

## 📊 Expected Performance

### Typical Changes After Proper Validation

| Stage | R² | RMSE (log units) | Interpretation |
|-------|-----|------------------|----------------|
| **Before** (random split) | 0.80-0.85 | 0.25-0.30 | ⚠️ Optimistic (data leakage) |
| **After** (scaffold split) | 0.55-0.70 | 0.40-0.55 | ✅ Realistic generalization |
| **Near-optimal** (IC₅₀) | 0.60-0.75 | ~0.50 | ✅ Excellent (near assay limit) |

**Performance drop is EXPECTED and CORRECT!**
- Scaffold split tests true generalization to novel scaffolds
- RMSE ~0.5 is near theoretical limit for IC₅₀ assays
- Honest results → better peer reviews
- More reproducible and trustworthy science

---

## 🛠️ Installation

### Requirements

```bash
# Core dependencies
pip install numpy pandas rdkit scikit-learn scipy matplotlib

# For specific models:
pip install h2o  # Models 1 & 3 (H2O AutoML)
pip install transformers torch  # Model 2 (ChEBERTa)
pip install scikit-optimize  # Model 4 (Bayesian Optimization)
```

### Quick Setup

```bash
# Clone repository
git clone https://github.com/bhatnira/Roy-QSAR-Generative-dev.git
cd Roy-QSAR-Generative-dev

# Run validation on your data
python -c "
from qsar_utils_no_leakage import print_leakage_prevention_summary
from qsar_validation_utils import print_comprehensive_validation_checklist
print_leakage_prevention_summary()
print_comprehensive_validation_checklist()
"
```

---

## 📖 Usage Examples

### Example 1: Data Leakage Prevention

```python
from qsar_utils_no_leakage import QSARDataProcessor, ScaffoldSplitter

# Initialize processor
processor = QSARDataProcessor(smiles_col='SMILES', target_col='IC50')

# Clean data
df = processor.canonicalize_smiles(df)
df = processor.remove_duplicates(df, strategy='average')

# Scaffold-based split (prevents leakage)
splitter = ScaffoldSplitter()
train_idx, val_idx, test_idx = splitter.scaffold_split(df, test_size=0.2, val_size=0.1)

# Remove near-duplicates between splits
train_idx, test_idx = processor.remove_near_duplicates(df, train_idx, test_idx, threshold=0.95)

# Verify zero overlap (critical!)
train_smiles = set(df.iloc[train_idx]['SMILES'])
test_smiles = set(df.iloc[test_idx]['SMILES'])
assert len(train_smiles & test_smiles) == 0, "❌ Data leakage detected!"
print("✅ Zero SMILES overlap - no data leakage")
```

### Example 2: Comprehensive Validation

```python
from qsar_validation_utils import (
    DatasetBiasAnalyzer, ActivityCliffDetector,
    ModelComplexityAnalyzer, YRandomizationTester
)

# Analyze dataset bias
analyzer = DatasetBiasAnalyzer()
diversity_results = analyzer.analyze_scaffold_diversity(df)

# Check model complexity
ModelComplexityAnalyzer.analyze_complexity(
    n_samples=len(train_data),
    n_features=1024,
    model_type='random_forest'
)

# Detect activity cliffs
cliff_detector = ActivityCliffDetector()
cliff_df = cliff_detector.detect_activity_cliffs(df, similarity_threshold=0.85)

# Y-randomization test (after training)
tester = YRandomizationTester()
rand_results = tester.perform_y_randomization(X, y, model, n_iterations=10)
print(f"Y-random R²: {rand_results['r2_mean']:.3f} (should be ≤ 0)")
```

### Example 3: Proper Feature Scaling

```python
from sklearn.preprocessing import StandardScaler

# ❌ WRONG: Fitting on entire dataset
scaler = StandardScaler()
X_all_scaled = scaler.fit_transform(X_all)  # Data leakage!

# ✅ CORRECT: Fit on training data only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)    # Use training statistics
X_test_scaled = scaler.transform(X_test)  # Use training statistics
```

---

## 🎓 Critical Thresholds

| Metric | Threshold | Interpretation |
|--------|-----------|----------------|
| **Scaffold diversity** | > 0.3 | Good chemical diversity |
| **Scaffold diversity** | < 0.3 | ⚠️ Congeneric series |
| **Samples:Features** | > 10:1 | ✅ Can use complex models |
| **Samples:Features** | 5-10:1 | 🟠 Need regularization |
| **Samples:Features** | < 5:1 | 🔴 High overfitting risk |
| **RMSE (IC₅₀)** | ~0.5 | ✅ Near theoretical limit |
| **RMSE (IC₅₀)** | < 0.3 | ⚠️ Suspicious (check leakage) |
| **R² (y-random)** | ≤ 0.0 | ✅ Not overfitting |
| **R² (y-random)** | > 0.2 | 🔴 Overfitting detected |

---

## 📝 Citation

If you use this framework in your research, please cite:

```bibtex
@software{qsar_validation_framework_2026,
  title = {Comprehensive QSAR Validation Framework for Low-Data Regimes},
  author = {Bhatnira},
  year = {2026},
  url = {https://github.com/bhatnira/Roy-QSAR-Generative-dev},
  note = {Data leakage prevention and comprehensive validation for QSAR models}
}
```

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Additional validation checks
- More molecular descriptors
- Deep learning model examples
- External test set validation
- Web interface for validation

Please open an issue or submit a pull request.

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- **RDKit**: Open-source cheminformatics toolkit
- **scikit-learn**: Machine learning library
- **H2O.ai**: AutoML platform
- **Hugging Face**: Pre-trained molecular transformers

---

## 📞 Support

- **Documentation**: Start with [`README_COMPREHENSIVE.md`](README_COMPREHENSIVE.md)
- **Quick Help**: [`QUICK_REFERENCE_CARD.md`](QUICK_REFERENCE_CARD.md)
- **Issues**: [GitHub Issues](https://github.com/bhatnira/Roy-QSAR-Generative-dev/issues)
- **Discussions**: [GitHub Discussions](https://github.com/bhatnira/Roy-QSAR-Generative-dev/discussions)

---

## 🔑 Key Takeaways

1. **Scaffold split is mandatory** - Random split = data leakage
2. **RMSE ~0.5 is excellent** - For IC₅₀ (near assay precision)
3. **Performance will drop** - This is correct and expected!
4. **Report all metrics** - RMSE, MAE, R², Spearman (not just R²)
5. **Compare to baseline** - Ridge regression minimum
6. **Run y-randomization** - Should give R² ≤ 0
7. **Simpler often better** - For datasets with n < 200
8. **State limitations** - Honest reporting → better reviews

---

## 📊 Repository Statistics

- **Documentation**: 115+ pages
- **Code**: 1,350+ lines
- **Models**: 4 complete examples
- **Issues Addressed**: 13+ critical validation checks
- **Python Utilities**: 2 comprehensive modules
- **Notebooks**: 4 fully validated QSAR models

---

## 🚀 Getting Started Paths

### 🟢 Beginner Path (1-2 weeks)
1. Week 1: Read documentation, understand concepts
2. Week 2: Run validation cells, review warnings
3. Week 3: Update your models with proper validation
4. Week 4: Implement baselines and tests

### 🟡 Intermediate Path (1 week)
1. Day 1-2: Read documentation, understand validation
2. Day 3-4: Run validation, update workflows
3. Day 5-6: Implement all validation checks
4. Day 7: Write paper using templates

### 🔴 Advanced Path (2-3 days)
1. Day 1: Quick scan, run validation cells
2. Day 2: Update all workflows, implement checks
3. Day 3: Finalize models, prepare for publication

---

**🎯 Ready to build publication-quality QSAR models with comprehensive validation!**

**Star ⭐ this repository if you find it useful!**

---

*Last Updated: January 2026*  
*Comprehensive QSAR Validation Framework*
