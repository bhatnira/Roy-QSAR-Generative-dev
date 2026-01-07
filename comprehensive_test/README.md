# Comprehensive QSAR Framework Test

This folder contains a complete test of all 13 modules using a realistic QSAR dataset.

## Contents

1. **`generate_qsar_dataset.py`** - Generates realistic QSAR test data
2. **`test_all_modules.py`** - Comprehensive test of all modules
3. **`run_tests.sh`** - Shell script to run everything
4. **`qsar_test_dataset.csv`** - Generated test dataset (created by generate script)
5. **`test_results.txt`** - Test output (created by test script)

## What Gets Tested

### All 13 Modules

1. **DuplicateRemoval** - Removes duplicate molecules
2. **AdvancedSplitter** - Tests all 3 splitting strategies:
   - Scaffold-based splitting
   - Temporal splitting
   - Cluster-based splitting
3. **FeatureScaler** - Standard/MinMax/Robust scaling
4. **FeatureSelector** - Variance/Univariate feature selection
5. **PCATransformer** - Dimensionality reduction
6. **DatasetQualityAnalyzer** - Dataset quality assessment
7. **ModelComplexityController** - Multi-library model recommendations
8. **PerformanceValidator** - Proper CV, y-randomization, baseline comparison
9. **ActivityCliffsDetector** - Activity cliffs detection
10. **UncertaintyEstimator** - Prediction uncertainty
11. **CrossValidator** - K-fold cross-validation
12. **PerformanceMetrics** - Comprehensive metrics calculation
13. **DatasetBiasAnalysis** - Train/test bias detection

### Multi-Library Support

- **Scikit-learn** (Ridge Regression)
- **XGBoost** (XGBRegressor) - if installed
- **LightGBM** (LGBMRegressor) - if installed

### Data Leakage Prevention

- Duplicates removed before splitting
- Scaffold-based splitting (no scaffold overlap)
- Feature scaling fitted on train only
- Feature selection fitted on train only
- PCA fitted on train only

### QSAR Best Practices

- Nested cross-validation
- Y-randomization test
- Baseline comparison
- Activity cliffs detection
- Uncertainty estimation
- Dataset quality analysis

## How to Run

### Option 1: Run Everything

```bash
cd comprehensive_test
bash run_tests.sh
```

This will:
1. Generate the QSAR dataset
2. Run all module tests
3. Save results to `test_results.txt`

### Option 2: Step by Step

```bash
# Step 1: Generate dataset
python generate_qsar_dataset.py

# Step 2: Run tests
python test_all_modules.py
```

### Option 3: Run with output redirect

```bash
python generate_qsar_dataset.py
python test_all_modules.py > test_results.txt 2>&1
```

## Requirements

### Core Requirements (Minimum)
```bash
pip install pandas numpy scikit-learn scipy
```

### For Full Testing (Recommended)
```bash
pip install pandas numpy scikit-learn scipy rdkit xgboost lightgbm
```

### Optional (for RDKit features)
```bash
# If RDKit is not available, the test will use random features
conda install -c conda-forge rdkit
# or
pip install rdkit
```

## Expected Output

### Dataset Generation

```
Generating realistic QSAR dataset...
================================================================================

✓ Generated 158 molecules
✓ Saved to: qsar_test_dataset.csv

Dataset Statistics:
--------------------------------------------------------------------------------
  Total molecules: 158
  Unique SMILES: 150
  Duplicates: 8
  Activity range: 4.12 - 8.87
  Activity mean: 6.54 ± 0.92
  Date range: 2023-01-01 to 2024-12-30
```

### Module Testing

```
================================================================================
COMPREHENSIVE QSAR FRAMEWORK TEST
================================================================================

Testing ALL 13 modules with realistic QSAR dataset

Modules to test:
  1. DuplicateRemoval
  2. AdvancedSplitter (3 strategies)
  3. FeatureScaler
  ...
  13. DatasetBiasAnalysis

[... detailed test output for each module ...]

================================================================================
COMPREHENSIVE TEST COMPLETE!
================================================================================

Modules Tested:
  ✓  1. DuplicateRemoval
  ✓  2. AdvancedSplitter (3 strategies)
  ✓  3. FeatureScaler
  ...
  ✓ 13. DatasetBiasAnalysis

Multi-Library Support:
  ✓ Sklearn (Ridge)
  ✓ XGBoost
  ✓ LightGBM

Dataset:
  Total molecules: 150
  Train: 105, Val: 15, Test: 30
  Features: 384

================================================================================
✓ ALL TESTS PASSED!
================================================================================
```

## Test Dataset Details

### Molecule Scaffolds (10 different)

1. Benzodiazepines
2. Phenethylamines
3. Sulfonamides
4. Quinolines
5. Indoles
6. Pyridines
7. Piperazines
8. Morpholines
9. Thiophenes
10. Pyrimidines

### Dataset Characteristics

- **Size**: ~150 molecules (after duplicate removal)
- **Activity**: pIC50 values (range: 4.0 - 9.0)
- **Duplicates**: ~5% intentional duplicates (to test removal)
- **Time span**: 2 years of data (for temporal splitting)
- **Features**: Morgan fingerprints (512 bits)
- **Realistic**: Based on real drug-like molecules

### Splitting Results

- **Train**: ~70% (105 molecules)
- **Validation**: ~10% (15 molecules)
- **Test**: ~20% (30 molecules)
- **Scaffold overlap**: 0 (verified)

## Troubleshooting

### Import Errors

If you see import errors, make sure you're running from the correct directory:

```bash
cd /path/to/QSAR_Models/comprehensive_test
python test_all_modules.py
```

### RDKit Not Available

If RDKit is not installed, the test will automatically use random features instead:

```
⚠️  RDKit not available - using random features for demo
```

This is fine for testing the framework functionality, but for real QSAR work, install RDKit.

### XGBoost/LightGBM Not Available

If these libraries are not installed, those specific tests will be skipped:

```
⚠️  XGBoost not installed
⚠️  LightGBM not installed
```

The framework will still test sklearn models.

### Module Import Errors

If you see "module not found" errors, make sure the src directory is in your Python path:

```python
# This is handled automatically by the test script
import sys
from pathlib import Path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))
```

## What This Test Demonstrates

### ✅ All Modules Work

- Every module can be imported
- Every module's main functionality works
- No import conflicts
- No breaking errors

### ✅ Multi-Library Support

- Framework works with sklearn
- Framework works with XGBoost (if installed)
- Framework works with LightGBM (if installed)
- Same API across all libraries

### ✅ Data Leakage Prevention

- Duplicates removed before splitting
- No scaffold overlap between train/test
- Feature engineering fitted on train only
- Proper nested CV for feature selection

### ✅ QSAR Best Practices

- Three splitting strategies available
- Model complexity controlled by dataset size
- Proper cross-validation with mean±std
- Y-randomization test included
- Activity cliffs detected
- Uncertainty estimated
- Dataset bias analyzed

### ✅ Complete Workflow

- Data cleaning
- Data splitting (3 strategies)
- Feature engineering (scale, select, PCA)
- Model training (multi-library)
- Validation (CV, metrics)
- Pitfall mitigation (13 checks)
- Final evaluation

## Files Generated

After running the tests, you'll have:

1. **`qsar_test_dataset.csv`** - The generated QSAR dataset
2. **`test_results.txt`** - Complete test output (if redirected)
3. Console output showing all test results

## Time to Run

- **Dataset generation**: ~5 seconds
- **Module testing**: ~30-60 seconds (depending on CV folds)
- **Total**: ~1 minute

## Success Criteria

The test is successful if you see:

```
================================================================================
✓ ALL TESTS PASSED!
================================================================================

The framework is working correctly with all modules!
```

## Next Steps

After running this test successfully:

1. **Use the framework** with your own QSAR data
2. **Customize** the modules for your specific needs
3. **Extend** by adding your own modules
4. **Share** your results with the community

## Questions?

- Check the main README.md in the parent directory
- See the examples/ folder for more usage examples
- Open an issue on GitHub

---

**Status**: ✅ Ready to run  
**Last Updated**: January 6, 2026  
**Framework Version**: 4.1.0
