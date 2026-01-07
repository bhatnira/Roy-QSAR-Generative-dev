# Installation Guide

## 🚀 Install Once, Use Anywhere (Like RDKit!)

### Method 1: Install from Local Repository (Editable Mode)

```bash
# Clone the repository
git clone https://github.com/bhatnira/Roy-QSAR-Generative-dev.git
cd Roy-QSAR-Generative-dev

# Install in editable mode (recommended for development)
pip install -e .

# Or install normally
pip install .
```

### Method 2: Install Directly from GitHub

```bash
pip install git+https://github.com/bhatnira/Roy-QSAR-Generative-dev.git
```

---

## ✅ After Installation - Use It Anywhere!

Once installed, you can use it in **any Python script or Jupyter notebook** without touching paths:

```python
# No more sys.path.insert()!
# Just import like any other package

from utils.qsar_utils_no_leakage import QSARDataProcessor, quick_clean, clean_qsar_data_with_report
from qsar_validation.splitting_strategies import AdvancedSplitter
from qsar_validation.feature_scaling import FeatureScaler
from qsar_validation.model_complexity_control import ModelComplexityController

# Use it!
df_clean = quick_clean(df, smiles_col='SMILES', target_col='pIC50')
```

---

## 📦 What Gets Installed

After installation, you get access to:

### Core Modules
- `utils.qsar_utils_no_leakage` - Data cleaning, canonicalization, duplicates
- `qsar_validation.splitting_strategies` - Scaffold/temporal/cluster splits
- `qsar_validation.feature_scaling` - Proper feature scaling
- `qsar_validation.feature_selection` - Feature selection methods
- `qsar_validation.model_complexity_control` - Multi-library model control
- `qsar_validation.performance_validation` - Cross-validation, metrics
- `qsar_validation.dataset_quality_analysis` - Dataset quality checks
- `qsar_validation.activity_cliffs_detection` - Activity cliff analysis
- `qsar_validation.uncertainty_estimation` - Prediction uncertainty

### Functions Available
- `quick_clean(df, smiles_col, target_col)` - Simple cleaning
- `clean_qsar_data_with_report(df, smiles_col, target_col)` - Detailed cleaning with reports

---

## 🔧 Verify Installation

```python
# Check if installed correctly
import utils.qsar_utils_no_leakage as qsar
import qsar_validation

print("✅ QSAR Validation Framework installed successfully!")
print(f"Version: 4.1.0")

# Try a quick function
from utils.qsar_utils_no_leakage import quick_clean
help(quick_clean)
```

---

## 📍 Google Colab Installation

```python
# In Google Colab, install directly from GitHub
!pip install git+https://github.com/bhatnira/Roy-QSAR-Generative-dev.git

# Then use it normally
from utils.qsar_utils_no_leakage import quick_clean
df_clean = quick_clean(df, smiles_col='SMILES', target_col='pIC50')
```

---

## 🔄 Update to Latest Version

```bash
# If installed in editable mode
cd Roy-QSAR-Generative-dev
git pull origin main

# If installed from GitHub
pip install --upgrade git+https://github.com/bhatnira/Roy-QSAR-Generative-dev.git
```

---

## ❌ Uninstall

```bash
pip uninstall qsar-validation-framework
```

---

## 🐛 Troubleshooting

### Issue: ModuleNotFoundError after installation

**Solution:**
```bash
# Make sure you're in the right environment
pip list | grep qsar

# Reinstall if needed
pip uninstall qsar-validation-framework
pip install -e /path/to/Roy-QSAR-Generative-dev
```

### Issue: Import errors

**Solution:**
```python
# Verify installation
import sys
print(sys.path)

# Check if package is installed
import pkg_resources
print(pkg_resources.get_distribution("qsar-validation-framework").version)
```

---

## 💡 Development Mode

For developers who want to modify the code:

```bash
# Install in editable mode with dev dependencies
cd Roy-QSAR-Generative-dev
pip install -e ".[dev]"

# Now any changes to source code are immediately available
# No need to reinstall!
```

---

## ✨ That's It!

No more:
- ❌ `sys.path.insert()`
- ❌ Path juggling
- ❌ Import errors
- ❌ Folder confusion

Just:
- ✅ `pip install`
- ✅ Import anywhere
- ✅ Use like RDKit!

🎉 **Enjoy your installed QSAR framework!**
