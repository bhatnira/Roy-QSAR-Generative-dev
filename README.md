# QSAR Models

**A Modular, Model-Agnostic, and Reproducible Framework**

**Version 3.0.0 - Now Completely Model-Agnostic!**

A professional, modular, **notebook-free** framework that works with ANY model and ANY featurizer. Perfect for the low-data regime (< 200 compounds). **You choose the models and features, we handle ALL data leakage prevention and validation automatically.**

## 🎯 What's New in v3.0?

- **Model-Agnostic**: Works with ANY sklearn-compatible model (Random Forest, XGBoost, Neural Networks, SVR, your custom models, etc.)
- **Featurizer-Agnostic**: Works with ANY feature representation (Morgan, MACCS, RDKit descriptors, ChemBERTa embeddings, custom features, etc.)
- **Ultra Simple**: Just 5 lines of code for complete validation
- **Complete Freedom**: You have full control over models, features, and settings

## Key Features

### 1. 🛡️ **Data Leakage Prevention** (Automatic)
  - ✅ Scaffold-based splitting (Bemis-Murcko, zero overlap)
  - ✅ Automatic duplicate removal (before splitting)
  - ✅ Proper feature scaling (train stats only)
  - ✅ Feature selection (within CV loops)
  - ✅ Nested CV support (hyperparameter tuning)
  - ✅ No information leakage from test sets

### 2. 🎯 **Model-Agnostic** (NEW!)
  - ✅ Works with ANY sklearn-compatible model
  - ✅ Random Forest, XGBoost, Ridge, SVR, Neural Networks, etc.
  - ✅ Your custom models welcome!

### 3. 🧬 **Featurizer-Agnostic** (NEW!)
  - ✅ Works with ANY featurizer function
  - ✅ Morgan fingerprints, MACCS keys, RDKit descriptors
  - ✅ ChemBERTa embeddings, custom features, anything!

### 4. 📊 **Comprehensive Validation** (Automatic)
  - ✅ Cross-validation (no leakage)
  - ✅ Y-randomization tests
  - ✅ Activity cliff detection
  - ✅ Model complexity analysis
  - ✅ Experimental error estimation

### 5. 🚀 **Ultra Simple to Use**
```python
# Just 5 lines!
pipeline = ModelAgnosticQSARPipeline(
    featurizer=my_featurizer,  # Your choice
    model=my_model,             # Your choice
    smiles_col='SMILES',
    target_col='Activity'
)
results = pipeline.fit_predict_validate(df)
```

## Project Structure

```
QSAR_Models/
├── notebooks/              # Jupyter notebooks with model implementations
│   ├── Model_1_circular_fingerprint_features_1024_H20_autoML_Model_Interpretation.ipynb
│   ├── Model_2_ChEBERTa_embedding_linear_regression_no_interpretation.ipynb
│   ├── Model_3_rdkit_features_H20_autoML.ipynb
│   ├── Model_4_circular_fingerprint_features_1024_Gaussian_Process_Bayesian_Optimization_Model_Interpretation.ipynb
│   └── DATA_LEAKAGE_FIX_EXAMPLE.ipynb
│
├── src/                    # Source code
│   ├── qsar_validation/   # Modular validation package
│   │   ├── __init__.py
│   │   ├── dataset_analysis.py      # Dataset bias and scaffold diversity
│   │   ├── activity_cliffs.py       # Activity cliff detection
│   │   ├── model_complexity.py      # Model complexity analysis
│   │   ├── metrics.py              # Performance metrics
│   │   ├── randomization.py        # Y-randomization testing
│   │   ├── assay_noise.py          # Experimental error estimation
│   │   └── validation_runner.py    # Main orchestrator
│   │
│   └── utils/             # Utility functions
│       ├── qsar_utils_no_leakage.py      # Leakage-free utilities
│       └── qsar_validation_utils.py      # Legacy validation utils
│
├── docs/                   # Documentation
│   ├── README.md                          # Main documentation
│   ├── INDEX.md                           # Documentation index
│   ├── START_HERE.md                      # Getting started guide
│   ├── QUICK_START_FIX.md                 # Quick fixes for common issues
│   ├── COMPREHENSIVE_VALIDATION_GUIDE.md  # Complete validation guide
│   └── ... (more documentation files)
│
├── examples/              # Example scripts and use cases
│   ├── 01_basic_validation.py            # Basic validation example
│   └── 02_custom_workflow.py             # Custom workflow example
│
├── tests/                 # Unit tests
│   └── test_validation.py                # Test suite
│
├── setup.py              # Package installation configuration
├── requirements.txt      # Python dependencies
└── .gitignore           # Git ignore rules
```

## Features

### Comprehensive Validation Framework

- **Dataset Bias Analysis**: Scaffold diversity, chemical space coverage
- **Activity Cliff Detection**: Identify SAR discontinuities
- **Model Complexity Control**: Prevent overfitting in low-data regimes
- **Performance Metrics**: RMSE, MAE, R², Spearman rho with baselines
- **Y-Randomization Testing**: Detect spurious correlations
- **Assay Noise Estimation**: Context for achievable performance

### Data Leakage Prevention

- Scaffold-based splitting (not random)
- Proper cross-validation workflows
- Feature scaling within CV loops
- No information leakage from test sets

### Multiple Model Implementations

1. **Model 1**: Circular Fingerprints (1024-bit) + H2O AutoML
2. **Model 2**: ChEBERTa Embeddings + Linear Regression
3. **Model 3**: RDKit Descriptors + H2O AutoML
4. **Model 4**: Circular Fingerprints + Gaussian Process + Bayesian Optimization

## Installation

```bash
# Clone the repository
git clone https://github.com/bhatnira/Roy-QSAR-Generative-dev.git
cd Roy-QSAR-Generative-dev

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Quick Start

### Option 1: One-Line Validation (Minimal Code)

```python
import pandas as pd
from qsar_validation import run_comprehensive_validation

df = pd.read_csv('your_data.csv')
results = run_comprehensive_validation(df, smiles_col='SMILES', target_col='Activity')
```

### Option 2: Standalone Workflow Script

```bash
# Run complete workflow from command line
python standalone_qsar_workflow.py --data your_data.csv --smiles SMILES --target Activity

# Show validation checklist
python standalone_qsar_workflow.py --checklist
```

### Option 3: Complete Python Script

```python
from qsar_validation import (
    DatasetBiasAnalyzer,
    ActivityCliffDetector,
    ModelComplexityAnalyzer,
    PerformanceMetricsCalculator
)

# Load your data
import pandas as pd
df = pd.read_csv('your_data.csv')

# Step 1: Validate dataset
analyzer = DatasetBiasAnalyzer('SMILES', 'Activity')
diversity = analyzer.analyze_scaffold_diversity(df)

# Step 2: Check for activity cliffs
detector = ActivityCliffDetector('SMILES', 'Activity')
cliffs = detector.detect_activity_cliffs(df)

# Step 3: Train your model (example with sklearn)
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ... generate features (X) ...
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Use recommended settings for low data
model = RandomForestRegressor(
    n_estimators=50,        # Limited for low data
    max_depth=5,            # Prevent overfitting
    min_samples_leaf=5,
    max_features='sqrt',
    random_state=42
)
model.fit(X_train, y_train)

# Step 4: Evaluate with proper metrics
y_pred = model.predict(X_test)
metrics = PerformanceMetricsCalculator.calculate_all_metrics(
    y_test, y_pred, set_name="Test"
)
```

### Option 4: Integration into Your Pipeline

```python
# preprocessing.py
from qsar_validation import run_comprehensive_validation

def validate_before_modeling(df):
    results = run_comprehensive_validation(df, 'SMILES', 'pIC50')
    
    # Add your logic
    if results['scaffold_diversity']['diversity_ratio'] < 0.3:
        print("WARNING: Limited applicability domain")
    
    return results

# Then use in your existing pipeline
df = pd.read_csv('data.csv')
validation = validate_before_modeling(df)
# ... continue with your modeling ...
```

## Documentation

- **[DATA_LEAKAGE_PREVENTION.md](DATA_LEAKAGE_PREVENTION.md)** - **How data leakage is prevented**
- **[USAGE_GUIDE.md](USAGE_GUIDE.md)** - Complete notebook-free usage guide
- **[standalone_qsar_workflow.py](standalone_qsar_workflow.py)** - Ready-to-use standalone script
- **[QUICK_START.md](QUICK_START.md)** - Quick reference (30 seconds to 5 minutes)
- **[docs/README_DATA_LEAKAGE_FIX.md](docs/README_DATA_LEAKAGE_FIX.md)** - Detailed leakage fixes
- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** - Detailed structure guide
- **[examples/](examples/)** - Example scripts for different use cases

## Key Validation Principles

### Critical (Must Fix)
- Scaffold-based splitting (not random)
- Remove duplicates BEFORE splitting
- Scale features using training data only
- No feature selection on full dataset

### High Priority (Strongly Recommended)
- Estimate and report experimental error
- Detect and report activity cliffs
- Report multiple metrics (not just R²)
- Perform y-randomization tests

### Best Practices
- Analyze target distribution
- Provide uncertainty estimates
- Use proper baselines (Ridge regression)
- Report limitations honestly

## Expected Performance

When fixing data leakage issues, expect:

- **R² drop**: 0.80 -> 0.60 (or lower) - This is NORMAL and CORRECT
- **RMSE increase**: 0.3 -> 0.5 - More realistic for IC50 data
- **Scaffold split harder**: Tests true generalization

For IC50/EC50 assays:
- Typical experimental error: 0.3 - 0.6 log units
- RMSE ~0.5 is near theoretical limit
- RMSE < 0.3 may indicate overfitting

## Repository Information

- **Repository**: [Roy-QSAR-Generative-dev](https://github.com/bhatnira/Roy-QSAR-Generative-dev)
- **Owner**: bhatnira
- **Branch**: main
- **Version**: 2.0.0 (Modularized & Organized)

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{qsar_validation_framework,
  title = {QSAR Validation Framework: A Modular and Reproducible Framework},
  author = {Roy QSAR Group},
  year = {2026},
  url = {https://github.com/bhatnira/Roy-QSAR-Generative-dev}
}
```

## License

[Add your license here]

## Acknowledgments

This framework implements best practices from:
- Tropsha, A. (2010). Best Practices for QSAR Model Development
- OECD Principles for QSAR Validation
- Recent advances in scaffold-based CV and activity cliff analysis

---

**Version**: 2.0.0 (Modular & Reproducible)  
**Last Updated**: January 6, 2026
