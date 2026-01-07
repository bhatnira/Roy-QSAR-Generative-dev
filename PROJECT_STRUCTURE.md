# Project Structure Overview

This document describes the organization of the QSAR Validation Framework project.

## Directory Structure

```
QSAR_Models/
├── README.md                  # Main project documentation
├── requirements.txt           # Python dependencies
├── setup.py                  # Package installation configuration
├── .gitignore               # Git ignore rules
│
├── notebooks/               # Jupyter notebooks with model implementations
│   ├── Model_1_circular_fingerprint_features_1024_H20_autoML_Model_Interpretation.ipynb
│   ├── Model_2_ChEBERTa_embedding_linear_regression_no_interpretation.ipynb
│   ├── Model_3_rdkit_features_H20_autoML.ipynb
│   ├── Model_4_circular_fingerprint_features_1024_Gaussian_Process_Bayesian_Optimization_Model_Interpretation.ipynb
│   └── DATA_LEAKAGE_FIX_EXAMPLE.ipynb
│
├── src/                     # Source code (Python package)
│   ├── __init__.py
│   │
│   ├── qsar_validation/    # Modular validation package
│   │   ├── __init__.py                # Package initialization
│   │   ├── dataset_analysis.py        # Dataset bias and scaffold diversity
│   │   ├── activity_cliffs.py         # Activity cliff detection
│   │   ├── model_complexity.py        # Model complexity analysis
│   │   ├── metrics.py                 # Performance metrics calculation
│   │   ├── randomization.py           # Y-randomization testing
│   │   ├── assay_noise.py             # Experimental error estimation
│   │   └── validation_runner.py       # Main orchestrator
│   │
│   └── utils/              # Utility functions
│       ├── __init__.py
│       ├── qsar_utils_no_leakage.py   # Leakage-free utilities
│       └── qsar_validation_utils.py   # Legacy validation utilities
│
├── docs/                    # Documentation
│   ├── README.md                          # Main documentation (moved from root)
│   ├── INDEX.md                           # Documentation index
│   ├── START_HERE.md                      # Getting started guide
│   ├── QUICK_START_FIX.md                 # Quick fixes guide
│   ├── COMPREHENSIVE_VALIDATION_GUIDE.md  # Complete validation guide
│   ├── COMPLETE_VALIDATION_SUMMARY.md     # Validation summary
│   ├── QUICK_REFERENCE_CARD.md            # Quick reference card
│   ├── ALL_MODELS_FIXED_SUMMARY.md        # Summary of all model fixes
│   ├── MODEL_1_CHANGES_SUMMARY.md         # Model 1 specific changes
│   ├── README_DATA_LEAKAGE_FIX.md         # Data leakage prevention guide
│   └── README_COMPREHENSIVE.md            # Comprehensive documentation
│
├── examples/               # Example scripts and use cases
│   ├── 01_basic_validation.py            # Basic validation example
│   └── 02_custom_workflow.py             # Custom workflow example
│
└── tests/                 # Unit tests
    └── test_validation.py                # Validation package tests
```

## Module Descriptions

### Core Validation Package (`src/qsar_validation/`)

**dataset_analysis.py** (11K)
- `DatasetBiasAnalyzer` class
- Scaffold diversity analysis using Bemis-Murcko scaffolds
- Activity distribution analysis
- Chemical space coverage assessment
- Split quality evaluation

**activity_cliffs.py** (4.7K)
- `ActivityCliffDetector` class
- Detects pairs of similar molecules with large activity differences
- Uses Tanimoto similarity on Morgan fingerprints
- Identifies SAR discontinuities

**model_complexity.py** (5.3K)
- `ModelComplexityAnalyzer` class
- Analyzes sample-to-feature ratio
- Provides model-specific recommendations
- Prevents overfitting in low-data regimes

**metrics.py** (5.5K)
- `PerformanceMetricsCalculator` class
- Calculates RMSE, MAE, R², Pearson r, Spearman rho
- Baseline model comparison (Ridge regression)
- Proper cross-validation metrics

**randomization.py** (5.2K)
- `YRandomizationTester` class
- Y-scrambling test for detecting spurious correlations
- Tests if model memorizes noise
- Reports mean ± std across iterations

**assay_noise.py** (3.4K)
- `AssayNoiseEstimator` class
- Estimates experimental error from replicates
- Provides literature-based estimates for IC50/EC50
- Contextualizes model performance

**validation_runner.py** (7.7K)
- `run_comprehensive_validation()` function
- Orchestrates all validation checks
- `print_comprehensive_validation_checklist()` function
- Comprehensive validation checklist

### Utilities (`src/utils/`)

**qsar_utils_no_leakage.py** (20K)
- Data preprocessing without leakage
- Proper scaffold-based splitting
- Cross-validation utilities
- Feature scaling within CV loops

**qsar_validation_utils.py** (34K)
- Legacy validation utilities (monolithic version)
- Kept for backward compatibility
- Contains all validation functions in one file

## Usage Patterns

### Quick Start
```python
from src.qsar_validation import run_comprehensive_validation

results = run_comprehensive_validation(df, smiles_col='SMILES', target_col='IC50')
```

### Custom Workflow
```python
from src.qsar_validation import DatasetBiasAnalyzer, ActivityCliffDetector

analyzer = DatasetBiasAnalyzer('SMILES', 'IC50')
diversity = analyzer.analyze_scaffold_diversity(df)

detector = ActivityCliffDetector('SMILES', 'IC50')
cliffs = detector.detect_activity_cliffs(df)
```

### Model Evaluation
```python
from src.qsar_validation import PerformanceMetricsCalculator

metrics = PerformanceMetricsCalculator.calculate_all_metrics(
    y_true=y_test,
    y_pred=y_pred,
    set_name="Test Set"
)
```

## File Sizes

- **Notebooks**: ~5.4 MB total (4 model notebooks + 1 example)
- **Source Code**: ~50 KB (modular validation package)
- **Documentation**: ~150 KB (11 markdown files)
- **Total Project**: ~5.6 MB

## Installation

### Option 1: Editable Install (Recommended for Development)
```bash
cd QSAR_Models
pip install -e .
```

### Option 2: Direct Install
```bash
cd QSAR_Models
pip install .
```

### Option 3: From GitHub
```bash
pip install git+https://github.com/bhatnira/Roy-QSAR-Generative-dev.git
```

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src/qsar_validation

# Run specific test file
pytest tests/test_validation.py -v
```

## Version History

- **v2.0.0** (2026-01-06): Modularized package structure
  - Split monolithic file into 7 focused modules
  - Removed emojis for cleaner output
  - Added proper package structure with setup.py
  - Organized files into appropriate directories

- **v1.0.0** (2026-01-06): Initial comprehensive validation framework
  - Implemented all 13+ validation checks
  - Fixed data leakage issues
  - Added scaffold-based splitting

## Contributing

This is a research project. The code is organized to facilitate:

1. **Easy modification** - Each module has a single responsibility
2. **Testing** - Unit tests can target specific modules
3. **Documentation** - Each function has clear docstrings
4. **Extension** - New validators can be added as new modules

## License

[Add license information]

---

**Last Updated**: January 6, 2026
**Version**: 2.0.0
