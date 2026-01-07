# QSAR Validation Framework v4.1.0

**A Purely Modular QSAR Validation Framework with Data Leakage Prevention**

Perfect for the low-data regime (< 200 compounds). **Works with ANY ML library** - sklearn, XGBoost, LightGBM, PyTorch, TensorFlow!

[![GitHub](https://img.shields.io/badge/GitHub-Roy--QSAR--Generative--dev-blue)](https://github.com/bhatnira/Roy-QSAR-Generative-dev)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

## Table of Contents

- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Framework Overview](#-framework-overview)
- [Available Modules](#-available-modules)
- [Usage Examples](#-usage-examples)
- [Example Notebooks](#-example-notebooks)
- [Google Colab Setup](#-google-colab-setup)
- [Complete Functionality List](#-complete-functionality-list)
- [Testing](#-testing)
- [Troubleshooting](#-troubleshooting)
- [Repository Structure](#-repository-structure)
- [Contributing](#-contributing)

---

## Quick Start

### Install Once, Use Anywhere (Recommended!)

```bash
# Option 1: Install from GitHub
pip install git+https://github.com/bhatnira/Roy-QSAR-Generative-dev.git

# Option 2: Install locally in editable mode (for development)
git clone https://github.com/bhatnira/Roy-QSAR-Generative-dev.git
cd Roy-QSAR-Generative-dev
pip install -e .
```

**Then use it anywhere - no more path juggling!**

```python
# No sys.path.insert() needed!
from utils.qsar_utils_no_leakage import quick_clean
from qsar_validation.splitting_strategies import AdvancedSplitter

df_clean = quick_clean(df, smiles_col='SMILES', target_col='pIC50')
```

---

### Alternative: Clone and Run (Without Installation)

```bash
# 1. Clone repository
git clone https://github.com/bhatnira/Roy-QSAR-Generative-dev.git
cd Roy-QSAR-Generative-dev

# 2. Install dependencies only
pip install -r requirements.txt

# 3. Use with sys.path in your scripts
import sys
sys.path.insert(0, '/path/to/Roy-QSAR-Generative-dev/src')
```

---

## Installation

### System Requirements

- **Python:** ≥ 3.8
- **OS:** macOS, Linux, Windows
- **RAM:** 4GB minimum (8GB recommended)

### Core Dependencies

```bash
# Required
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
rdkit>=2022.3.0

# Machine Learning (choose what you need)
scikit-learn>=1.0.0 # For sklearn models
xgboost>=1.5.0 # For XGBoost
lightgbm>=3.3.0 # For LightGBM
torch>=1.10.0 # For PyTorch models
tensorflow>=2.8.0 # For TensorFlow models

# Visualization
matplotlib>=3.4.0
seaborn>=0.11.0
```

### Installation Methods

#### Method 1: Direct from GitHub (Easiest)

```bash
pip install git+https://github.com/bhatnira/Roy-QSAR-Generative-dev.git
```

After installation:
```python
from utils.qsar_utils_no_leakage import quick_clean
from qsar_validation.splitting_strategies import AdvancedSplitter
# Ready to use!
```

#### Method 2: Local Editable Install (For Development)

```bash
git clone https://github.com/bhatnira/Roy-QSAR-Generative-dev.git
cd Roy-QSAR-Generative-dev
pip install -e .
```

Benefits:
- Changes to source code immediately reflected
- No need to reinstall after modifications
- Perfect for development and experimentation

#### Method 3: Clone Without Installation

```bash
git clone https://github.com/bhatnira/Roy-QSAR-Generative-dev.git
cd Roy-QSAR-Generative-dev
pip install -r requirements.txt
```

Then in your scripts:
```python
import sys
sys.path.insert(0, '/path/to/Roy-QSAR-Generative-dev/src')
from utils.qsar_utils_no_leakage import QSARDataProcessor
```

### Google Colab Installation

**Recommended: Install as package**
```python
!pip install git+https://github.com/bhatnira/Roy-QSAR-Generative-dev.git

# Then import normally
from utils.qsar_utils_no_leakage import quick_clean
```

**Alternative: Clone and use paths**
```python
!git clone https://github.com/bhatnira/Roy-QSAR-Generative-dev.git
import sys
sys.path.insert(0, '/content/Roy-QSAR-Generative-dev/src')
```

### Installing Optional Dependencies

```bash
# For H2O AutoML (Models 1 & 3)
pip install h2o

# For ChEBERTa embeddings (Model 2)
pip install transformers torch

# For Bayesian optimization (Model 4)
pip install scikit-optimize

# For Jupyter notebooks
pip install jupyter ipykernel notebook
```

### Verification

After installation, verify it works:

```python
import sys
sys.path.insert(0, '/path/to/Roy-QSAR-Generative-dev/src')

# Test imports
from utils.qsar_utils_no_leakage import quick_clean
from qsar_validation.splitting_strategies import AdvancedSplitter
from qsar_validation.feature_scaling import FeatureScaler

print(" Installation successful!")
```

### Troubleshooting Installation

**Issue: RDKit not found**
```bash
# RDKit requires conda
conda install -c conda-forge rdkit

# Or use rdkit-pypi (may have limitations)
pip install rdkit-pypi
```

**Issue: Permission denied**
```bash
# Add --user flag
pip install --user git+https://github.com/bhatnira/Roy-QSAR-Generative-dev.git
```

**Issue: Old version cached**
```bash
# Force reinstall
pip install --force-reinstall git+https://github.com/bhatnira/Roy-QSAR-Generative-dev.git
```

---

## Framework Overview

### Philosophy

> **"No magic. No automation. Just reliable tools."**
> 
> **"You build the pipeline. We provide the pipes."**

This framework provides **independent, composable modules** for QSAR validation:
- No forced workflows
- Use only what you need
- Mix with your own code
- Works with any ML library

### Key Features

- **Data Leakage Prevention** - Scaffold-based splitting, proper scaling
- **Modular Design** - 13+ independent modules
- **Multi-Library Support** - sklearn, XGBoost, LightGBM, PyTorch, TensorFlow
- **Comprehensive Validation** - Dataset quality, model complexity, performance metrics
- **Example Notebooks** - 5 complete working examples
- **Google Colab Ready** - Works out of the box

---

## Available Modules

### Core Data Processing

| Module | Import | Purpose |
|--------|--------|---------|
| **QSARDataProcessor** | `from utils.qsar_utils_no_leakage import QSARDataProcessor` | SMILES canonicalization, duplicate removal, near-duplicate detection |
| **quick_clean** NEW | `from utils.qsar_utils_no_leakage import quick_clean` | Simple data cleaning with basic reporting |
| **clean_qsar_data_with_report** NEW | `from utils.qsar_utils_no_leakage import clean_qsar_data_with_report` | Detailed cleaning with comprehensive CSV reports |

### Data Splitting (3 Strategies)

| Module | Import | Purpose |
|--------|--------|---------|
| **AdvancedSplitter** | `from qsar_validation.splitting_strategies import AdvancedSplitter` | Scaffold/temporal/cluster-based splitting |

### Feature Engineering

| Module | Import | Purpose |
|--------|--------|---------|
| **FeatureScaler** | `from qsar_validation.feature_scaling import FeatureScaler` | StandardScaler, MinMaxScaler, RobustScaler (fit on train only) |
| **FeatureSelector** | `from qsar_validation.feature_selection import FeatureSelector` | Variance, correlation, model-based selection |
| **PCATransformer** | `from qsar_validation.pca_module import PCATransformer` | Dimensionality reduction (fit on train only) |

### Validation & Analysis

| Module | Import | Purpose |
|--------|--------|---------|
| **DatasetQualityAnalyzer** | `from qsar_validation.dataset_quality_analysis import DatasetQualityAnalyzer` | Dataset size, diversity, chemical space analysis |
| **ModelComplexityController** | `from qsar_validation.model_complexity_control import ModelComplexityController` | Model recommendations, hyperparameter control, nested CV |
| **PerformanceValidator** | `from qsar_validation.performance_validation import PerformanceValidator` | Cross-validation, metrics, Y-randomization |
| **ActivityCliffsDetector** | `from qsar_validation.activity_cliffs_detection import ActivityCliffsDetector` | Activity cliff detection, SALI calculation |
| **UncertaintyEstimator** | `from qsar_validation.uncertainty_estimation import UncertaintyEstimator` | Prediction uncertainty, applicability domain |

---

## Installation

## Installation

### Requirements

- Python ≥ 3.8
- pandas, numpy, rdkit, scipy
- scikit-learn (optional, for sklearn models)
- xgboost, lightgbm (optional, for XGBoost/LightGBM)
- torch, tensorflow (optional, for neural networks)

### Method 1: Install as Package (Recommended)

```bash
# Install directly from GitHub
pip install git+https://github.com/bhatnira/Roy-QSAR-Generative-dev.git

# Or clone and install in editable mode (for development)
git clone https://github.com/bhatnira/Roy-QSAR-Generative-dev.git
cd Roy-QSAR-Generative-dev
pip install -e .
```

**After installation, use it anywhere:**
```python
from utils.qsar_utils_no_leakage import quick_clean
from qsar_validation.splitting_strategies import AdvancedSplitter
# No path setup needed!
```

### Method 2: Clone Without Installation

```bash
git clone https://github.com/bhatnira/Roy-QSAR-Generative-dev.git
cd Roy-QSAR-Generative-dev
pip install -r requirements.txt

# Then add to path in your scripts
import sys
sys.path.insert(0, '/path/to/Roy-QSAR-Generative-dev/src')
```

### Google Colab Installation

```python
# Recommended: Install as package
!pip install git+https://github.com/bhatnira/Roy-QSAR-Generative-dev.git

# Alternative: Clone and use paths
!git clone https://github.com/bhatnira/Roy-QSAR-Generative-dev.git
import sys
sys.path.insert(0, '/content/Roy-QSAR-Generative-dev/src')
```

 **Complete installation guide:** [INSTALL.md](INSTALL.md)

---

## Usage Examples

### Example 1: Data Cleaning & Splitting

```python
import pandas as pd

# After pip install, just import directly!
from utils.qsar_utils_no_leakage import QSARDataProcessor, quick_clean, clean_qsar_data_with_report
from qsar_validation.splitting_strategies import AdvancedSplitter

# Load data
df = pd.read_csv('your_data.csv')

# Step 1: Clean duplicates BEFORE splitting (Choose ONE method)

# Option A: Basic cleaning (manual control)
processor = QSARDataProcessor(smiles_col='SMILES', target_col='pIC50')
df_clean = processor.canonicalize_smiles(df)
df_clean = processor.remove_duplicates(df_clean, strategy='average')
print(f"Clean dataset: {len(df_clean)} molecules")

# Option B: Quick clean with basic reporting SIMPLE!
df_clean = quick_clean(df, smiles_col='SMILES', target_col='pIC50')

# Option C: Detailed clean with comprehensive CSV reports DETAILED!
df_clean, stats = clean_qsar_data_with_report(df, smiles_col='SMILES', target_col='pIC50')
# Generates: cleaning_report_invalid_smiles.csv, cleaning_report_duplicates.csv,
# cleaning_report_summary.csv, cleaned_dataset.csv

# Step 2: Scaffold-based split (prevents leakage!)
splitter = AdvancedSplitter()
splits = splitter.scaffold_split(
 df_clean, # Use cleaned data!
 smiles_col='SMILES',
 target_col='pIC50',
 test_size=0.2,
 val_size=0.1
)

train_idx = splits['train_idx']
val_idx = splits['val_idx']
test_idx = splits['test_idx']

print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
```

### Example 2: Feature Engineering (Proper Way - No Leakage!)

```python
from qsar_validation.feature_scaling import FeatureScaler
from qsar_validation.feature_selection import FeatureSelector

# CRITICAL: Generate features AFTER splitting
# (Your feature generation code here - fingerprints, descriptors, etc.)

# Split data into train/test
X_train = features[train_idx]
X_test = features[test_idx]
y_train = df.loc[train_idx, 'pIC50'].values
y_test = df.loc[test_idx, 'pIC50'].values

# Step 1: Scale features (fit on train only!)
scaler = FeatureScaler(method='standard')
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 2: Select features (fit on train only!)
selector = FeatureSelector(method='variance', threshold=0.01)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)

print(f"Features: {X_train.shape[1]} → {X_train_selected.shape[1]}")
```

### Example 3: Dataset Quality Analysis

```python
from qsar_validation.dataset_quality_analysis import DatasetQualityAnalyzer

analyzer = DatasetQualityAnalyzer(smiles_col='SMILES', activity_col='pIC50')
quality_report = analyzer.analyze(df)

print(f"Dataset Size: {quality_report['size_analysis']['n_samples']}")
print(f"Scaffold Diversity: {quality_report['diversity_analysis']['n_unique_scaffolds']}")
print(f"Activity Range: {quality_report['activity_analysis']['range']}")
print(f"Overall Quality Score: {quality_report['overall_score']}/10")
```

### Example 4: Model Complexity Control

```python
from qsar_validation.model_complexity_control import ModelComplexityController
from sklearn.ensemble import RandomForestRegressor

# Get recommendations based on dataset size
controller = ModelComplexityController(
 n_samples=len(X_train),
 n_features=X_train.shape[1]
)

# Check recommended models
recommendations = controller.recommend_models()
print("Recommended models:", recommendations['sklearn'])

# Get safe hyperparameter grid
model = RandomForestRegressor()
param_grid = controller.get_safe_param_grid('random_forest', library='sklearn')
print("Safe parameter grid:", param_grid)

# Run nested cross-validation
results = controller.nested_cv(X_train, y_train, model=model, param_grid=param_grid)
print(f"Nested CV R²: {results['test_score_mean']:.3f} ± {results['test_score_std']:.3f}")
```

### Example 5: Performance Validation

```python
from qsar_validation.performance_validation import PerformanceValidator

validator = PerformanceValidator()

# Cross-validation (proper scaffold-based folds)
cv_results = validator.cross_validate(model, X_train, y_train, cv=5)
print(f"CV R²: {cv_results['test_r2_mean']:.3f}")
print(f"CV RMSE: {cv_results['test_rmse_mean']:.3f}")

# Y-randomization test (negative control)
random_results = validator.y_randomization_test(model, X_train, y_train, n_iterations=10)
print(f"Random R²: {random_results['mean_r2']:.3f} (should be near 0)")

# Baseline comparison
y_pred = model.predict(X_test)
baseline = validator.compare_to_baseline(y_test, y_pred)
print(f"Better than mean predictor: {baseline['beats_mean']}")
```

### Example 6: Complete Workflow

```python
import pandas as pd
from utils.qsar_utils_no_leakage import QSARDataProcessor
from qsar_validation.splitting_strategies import AdvancedSplitter
from qsar_validation.feature_scaling import FeatureScaler
from qsar_validation.dataset_quality_analysis import DatasetQualityAnalyzer
from qsar_validation.model_complexity_control import ModelComplexityController
from qsar_validation.performance_validation import PerformanceValidator
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# 1. Load and clean data
df = pd.read_csv('qsar_data.csv')
processor = QSARDataProcessor(smiles_col='SMILES', target_col='pIC50')
df = processor.remove_duplicates(df, strategy='average')

# 2. Analyze dataset quality
analyzer = DatasetQualityAnalyzer(smiles_col='SMILES', activity_col='pIC50')
quality = analyzer.analyze(df)
print(f"Quality Score: {quality['overall_score']}/10")

# 3. Split data (scaffold-based)
splitter = AdvancedSplitter()
splits = splitter.scaffold_split(df, smiles_col='SMILES', target_col='pIC50', test_size=0.2)
train_idx, test_idx = splits['train_idx'], splits['test_idx']

# 4. Generate features (your code - fingerprints, descriptors, etc.)
# X = generate_features(df) # Your feature generation

# 5. Scale features
scaler = FeatureScaler()
X_train_scaled = scaler.fit_transform(X[train_idx])
X_test_scaled = scaler.transform(X[test_idx])

# 6. Get model recommendations
controller = ModelComplexityController(n_samples=len(train_idx), n_features=X.shape[1])
recommendations = controller.recommend_models()

# 7. Train model
model = RandomForestRegressor(n_estimators=100, max_depth=5)
y_train = df.loc[train_idx, 'pIC50'].values
y_test = df.loc[test_idx, 'pIC50'].values
model.fit(X_train_scaled, y_train)

# 8. Validate
validator = PerformanceValidator()
cv_results = validator.cross_validate(model, X_train_scaled, y_train, cv=5)
print(f"CV R²: {cv_results['test_r2_mean']:.3f}")

# 9. Test
y_pred = model.predict(X_test_scaled)
test_r2 = r2_score(y_test, y_pred)
print(f"Test R²: {test_r2:.3f}")
```

### Example 7: Data Cleaning with Reports NEW

```python
import pandas as pd

# After pip install, just import!
from utils.qsar_utils_no_leakage import quick_clean, clean_qsar_data_with_report

# Load data
df = pd.read_csv('your_data.csv')

# Option 1: Quick clean (basic reporting)
clean_df = quick_clean(df, smiles_col='SMILES', target_col='pIC50')
# Output:
# Original dataset: 500 molecules
# Invalid SMILES removed: 5 molecules
# Duplicates merged: 45 molecules
# Clean dataset: 450 molecules

# Option 2: Detailed clean (comprehensive CSV reports)
clean_df, stats = clean_qsar_data_with_report(
 df, 
 smiles_col='SMILES', 
 target_col='pIC50'
)

# Generated reports:
# cleaning_report_invalid_smiles.csv - List of molecules that failed canonicalization
# cleaning_report_duplicates.csv - Duplicate details with original/averaged values
# cleaning_report_summary.csv - High-level statistics
# cleaned_dataset.csv - Final clean dataset

# Access statistics
print(f"Invalid: {stats['invalid_count']}, Duplicates: {stats['duplicate_count']}")
```

---

## Example Notebooks

The `notebooks/` folder contains **5 complete working examples**:

### 1. DATA_LEAKAGE_FIX_EXAMPLE.ipynb 
**Complete tutorial on data leakage prevention**
- Before/after comparison
- Common mistakes explained
- Proper workflow demonstration
- **Start here if you're new!**

### 2. Model 1: Circular Fingerprints + H2O AutoML
- Morgan fingerprints (1024 bits)
- H2O AutoML integration
- SHAP interpretability
- Feature importance

### 3. Model 2: ChEBERTa Embeddings + Linear Regression
- Transformer-based molecular embeddings
- ChEBERTa integration
- Linear regression with validation
- Embedding visualization

### 4. Model 3: RDKit Features + H2O AutoML
- RDKit molecular descriptors (200+)
- Descriptor calculation pipeline
- H2O AutoML leaderboard
- Feature correlation analysis

### 5. Model 4: Gaussian Process + Bayesian Optimization
- Gaussian Process regression
- Bayesian hyperparameter optimization
- Uncertainty quantification
- Acquisition function visualization

### Running Notebooks

**Locally:**
```bash
cd notebooks
jupyter notebook
# Open any notebook → Run cells in order
```

**Google Colab:**
1. Upload notebook to Colab
2. Add setup cell (see Google Colab section below)
3. Run all cells

---

## Google Colab Setup

### Universal Setup (Works for All Notebooks)

**Copy-paste this into the FIRST cell:**

```python
# ==========================================
# GOOGLE COLAB SETUP - Copy to first cell
# ==========================================
import os
import sys

# Check if in Colab
try:
 import google.colab
 IN_COLAB = True
except:
 IN_COLAB = False

if IN_COLAB:
 print(" Setting up Google Colab environment...")
 
 # Clone repository
 if not os.path.exists('Roy-QSAR-Generative-dev'):
 !git clone -q https://github.com/bhatnira/Roy-QSAR-Generative-dev.git
 print(" Repository cloned")
 
 # Change directory
 os.chdir('Roy-QSAR-Generative-dev/notebooks')
 print(f" Working directory: {os.getcwd()}")
 
 # Install dependencies
 !pip install -q rdkit-pypi pandas numpy scikit-learn matplotlib seaborn xgboost
 print(" Dependencies installed")
 
 print(" Setup complete!\n")

# Add framework to path (works locally and in Colab)
repo_root = os.path.abspath(os.path.join(os.getcwd(), '..') if IN_COLAB else os.getcwd())
sys.path.insert(0, os.path.join(repo_root, 'src'))

# Import framework
from utils.qsar_utils_no_leakage import QSARDataProcessor
from qsar_validation.splitting_strategies import AdvancedSplitter
from qsar_validation.feature_scaling import FeatureScaler

print(" Framework loaded!")
```

### Optional: Mount Google Drive (For Your Data)

```python
from google.colab import drive
drive.mount('/content/drive')

# Load your data from Drive
import pandas as pd
df = pd.read_csv('/content/drive/MyDrive/your_data.csv')
```

### Notebook-Specific Dependencies

**For Model 1 & 3 (H2O AutoML):**
```python
!pip install -q h2o
```

**For Model 2 (ChEBERTa):**
```python
!pip install -q transformers torch
```

**For Model 4 (Bayesian Optimization):**
```python
!pip install -q scikit-optimize
```

---

## Complete Functionality List

### 1. Core Data Processing

**QSARDataProcessor**
- SMILES canonicalization
- Duplicate removal (exact matches)
- Near-duplicate detection (Tanimoto ≥ 0.95)
- Replicate averaging
- Data validation

### 2. Data Splitting (3 Strategies)

**AdvancedSplitter**
- **Scaffold-based** - Bemis-Murcko scaffolds (RECOMMENDED)
- **Temporal** - Time-based splitting
- **Cluster-based** - Fingerprint clustering
- Stratified splitting
- Train/val/test support

### 3. Feature Engineering

**FeatureScaler**
- StandardScaler (Z-score normalization)
- MinMaxScaler ([0,1] scaling)
- RobustScaler (outlier-resistant)
- Fit on train only (prevents leakage!)

**FeatureSelector**
- Variance threshold
- Correlation filter
- Univariate selection (F-test, mutual info)
- Model-based selection
- Recursive feature elimination
- Select K best

**PCATransformer**
- Variance-based component selection
- Number-based selection
- Explained variance reporting
- Fit on train only

### 4. Dataset Quality Analysis

**DatasetQualityAnalyzer**
- Dataset size analysis
- Scaffold diversity assessment
- Chemical space coverage
- Activity distribution analysis
- Overall quality scoring

### 5. Model Complexity Control

**ModelComplexityController**
- Model recommendations (based on dataset size)
- Safe hyperparameter grids
- Nested cross-validation
- Overfitting detection
- Multi-library support (sklearn, XGBoost, LightGBM, PyTorch, TensorFlow)

### 6. Performance Validation

**PerformanceValidator**
- Scaffold-based cross-validation
- Comprehensive metrics (R², RMSE, MAE, etc.)
- Y-randomization test (negative control)
- Baseline comparison
- Confidence intervals

### 7. Activity Analysis

**ActivityCliffsDetector**
- Activity cliff detection
- SALI calculation
- Severity assessment
- Pair identification
- Reliability scoring

**UncertaintyEstimator**
- Ensemble variance
- Bootstrap confidence intervals
- Applicability domain
- Prediction confidence scoring

### 8. Metrics & Reporting

**PerformanceMetricsCalculator**
- Regression metrics (R², RMSE, MAE, MSE, Spearman, Kendall)
- Classification metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- Statistical tests (permutation, bootstrap)
- Visualization (predicted vs actual, residuals, ROC, PR curves)

---

## Key Principles

### Data Leakage Prevention

1. **Remove duplicates BEFORE splitting**
 ```python
 df = processor.remove_duplicates(df) # Do this first!
 splits = splitter.scaffold_split(df) # Then split
 ```

2. **Use scaffold-based splits (not random!)**
 ```python
 # GOOD
 splits = splitter.scaffold_split(df, ...)
 
 # BAD
 train_test_split(X, y, random_state=42) # Can leak!
 ```

3. **Generate features AFTER splitting**
 ```python
 splits = splitter.scaffold_split(df)
 # Now generate features for train/test separately
 ```

4. **Fit scalers on train only**
 ```python
 # GOOD
 scaler.fit(X_train)
 X_train_scaled = scaler.transform(X_train)
 X_test_scaled = scaler.transform(X_test)
 
 # BAD
 scaler.fit(X_all) # Leaks information!
 ```

5. **Use proper cross-validation**
 ```python
 # GOOD
 validator.cross_validate(...) # Uses scaffold-based folds
 
 # BAD
 cross_val_score(..., cv=KFold()) # Random folds leak!
 ```

### Best Practices

- Always analyze dataset quality first
- Use nested CV for hyperparameter tuning
- Check for activity cliffs
- Report uncertainty when possible
- Compare to baselines
- Run Y-randomization tests

---

## Testing

### Quick Test

Verify everything is working:

```bash
cd Roy-QSAR-Generative-dev

# Create test script
cat > test_quick.py << 'EOF'
import sys
sys.path.insert(0, './src')

print("Testing QSAR Framework...")
try:
 from utils.qsar_utils_no_leakage import quick_clean
 from qsar_validation.splitting_strategies import AdvancedSplitter
 print(" Imports work!")
 print(" Framework is ready to use!")
except Exception as e:
 print(f" Error: {e}")
EOF

# Run test
python3 test_quick.py
```

### Comprehensive Test

Test all functionality:

```python
import sys
sys.path.insert(0, './src')
import pandas as pd
from utils.qsar_utils_no_leakage import quick_clean

# Create test dataset
test_data = pd.DataFrame({
 'SMILES': ['CCO', 'CC(C)O', 'CCO', 'c1ccccc1'], # With duplicate
 'pIC50': [5.5, 6.2, 5.5, 4.8]
})

print(f"Original: {len(test_data)} rows")
cleaned = quick_clean(test_data, 'SMILES', 'pIC50')
print(f"Cleaned: {len(cleaned)} rows")

if len(cleaned) < len(test_data):
 print(" Data cleaning works correctly!")
else:
 print(" Check data cleaning logic")
```

### Expected Output

```
Testing QSAR Framework...
 Imports work!
 Framework is ready to use!

Original: 4 rows
 Canonicalized 4 SMILES
 Found 2 duplicate molecules
 Averaged 2 replicates
 Final dataset: 3 unique molecules
Cleaned: 3 rows
 Data cleaning works correctly!
```

### Running Example Scripts

```bash
# Navigate to examples folder
cd examples

# Run basic validation example
python3 01_basic_validation.py

# Run custom workflow example
python3 02_custom_workflow.py

# Test data cleaning with reports
python3 data_cleaning_with_report.py
```

---

## Troubleshooting

### Issue: ModuleNotFoundError

```python
# Error: ModuleNotFoundError: No module named 'qsar_validation'

# Solution: Add framework to path
import sys
sys.path.insert(0, '/path/to/Roy-QSAR-Generative-dev/src')

# Or if in notebooks folder:
sys.path.insert(0, '../src')
```

### Issue: Wrong Import

```python
# WRONG (old examples, doesn't exist)
from qsar_validation.duplicate_removal import DuplicateRemoval

# CORRECT
from utils.qsar_utils_no_leakage import QSARDataProcessor
```

### Issue: RDKit Not Found (Google Colab)

```python
# Install rdkit-pypi (not rdkit)
!pip install rdkit-pypi
# Then restart runtime
```

### Common Import Errors - All Correct Imports:

```python
# Core utilities
from utils.qsar_utils_no_leakage import QSARDataProcessor

# Splitting
from qsar_validation.splitting_strategies import AdvancedSplitter

# Feature engineering
from qsar_validation.feature_scaling import FeatureScaler
from qsar_validation.feature_selection import FeatureSelector
from qsar_validation.pca_module import PCATransformer

# Analysis
from qsar_validation.dataset_quality_analysis import DatasetQualityAnalyzer
from qsar_validation.model_complexity_control import ModelComplexityController
from qsar_validation.performance_validation import PerformanceValidator
from qsar_validation.activity_cliffs_detection import ActivityCliffsDetector
from qsar_validation.uncertainty_estimation import UncertaintyEstimator
```

---

## Framework Statistics

- **Modules:** 13+ independent modules + 2 cleaning utilities
- **ML Libraries:** 5+ supported (sklearn, XGBoost, LightGBM, PyTorch, TensorFlow)
- **Splitting Strategies:** 3 (Scaffold, Temporal, Cluster)
- **Feature Selection Methods:** 6+
- **Metrics:** 20+ (regression + classification)
- **Example Notebooks:** 5 complete examples
- **Example Scripts:** 10+ ready-to-use examples
- **Lines of Code:** 10,000+ (fully tested)
- **Data Cleaning Reports:** 4 CSV reports generated automatically NEW

---

## Repository Structure

```
Roy-QSAR-Generative-dev/

 README.md # This file - comprehensive documentation
 setup.py # Package configuration for pip install
 requirements.txt # Python dependencies

 src/ # Source code
 qsar_validation/ # Core validation modules
 model_agnostic_pipeline.py
 splitting_strategies.py
 activity_cliffs_detection.py
 dataset_quality_analysis.py
 feature_scaling.py
 feature_selection.py
 model_complexity_control.py
 performance_validation.py
 uncertainty_estimation.py
 ... (9 more modules)
 
 utils/
 qsar_utils_no_leakage.py # Data cleaning utilities

 examples/ # Usage examples (8 scripts)
 01_basic_validation.py
 02_custom_workflow.py
 data_cleaning_with_report.py
 feature_engineering_examples.py
 model_agnostic_examples.py
 modular_examples.py
 multi_library_examples.py
 splitting_strategies_examples.py

 notebooks/ # Jupyter notebooks (5 complete examples)
 DATA_LEAKAGE_FIX_EXAMPLE.ipynb Start here!
 Model_1_circular_fingerprint_features_1024_H20_autoML_Model_Interpretation.ipynb
 Model_2_ChEBERTa_embedding_linear_regression_no_interpretation.ipynb
 Model_3_rdkit_features_H20_autoML.ipynb
 Model_4_circular_fingerprint_features_1024_Gaussian_Process_Bayesian_Optimization_Model_Interpretation.ipynb

 tests/ # Unit tests
 ... (test modules)

 comprehensive_test/ # Integration tests
 ... (comprehensive test suite)
```

### Key Files

- **README.md**: Complete documentation (you are here!)
- **setup.py**: Package configuration for `pip install`
- **requirements.txt**: List of all dependencies
- **src/**: All source code (modular design)
- **examples/**: Ready-to-run Python scripts
- **notebooks/**: Complete Jupyter notebook examples

---

## Contributing

Contributions welcome! Each module is independent, making it easy to:
- Add new modules
- Improve existing functionality
- Fix bugs
- Add examples

### Development Setup

```bash
git clone https://github.com/bhatnira/Roy-QSAR-Generative-dev.git
cd Roy-QSAR-Generative-dev
pip install -r requirements.txt

# Run tests
cd comprehensive_test
python test_all_modules_simple.py
```

---

## Support

- **Issues:** [GitHub Issues](https://github.com/bhatnira/Roy-QSAR-Generative-dev/issues)
- **Examples:** See `notebooks/` folder
- **Tests:** See `comprehensive_test/` folder

---

## Citation

If you use this framework in your research:

```
QSAR Validation Framework v4.1.0
https://github.com/bhatnira/Roy-QSAR-Generative-dev
```

---

## Version History

- **v4.1.0** (Current): Multi-library support, comprehensive validation
- **v4.0.0**: QSAR pitfalls mitigation modules
- **v3.0.0**: Feature engineering with leakage prevention
- **v2.0.0**: Three splitting strategies
- **v1.0.0**: Initial modular framework

---

**Remember:** Each module is independent. Use what you need, ignore the rest! 

**Questions?** Check the example notebooks or open an issue on GitHub!

**Ready to start?** Clone the repository and try `DATA_LEAKAGE_FIX_EXAMPLE.ipynb` first!

---

**Made with for the QSAR community**
