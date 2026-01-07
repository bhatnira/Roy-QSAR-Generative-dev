# Installation and Testing Guide

## ✅ Repository Cleanup Status (January 7, 2026)

### Completed Actions:
- ✅ Removed 33 redundant .md files
- ✅ Consolidated documentation to README.md and INSTALL.md
- ✅ Moved 5 duplicate notebooks to notebooks/ folder
- ✅ Created pip-installable package structure (setup.py)
- ✅ Pushed all changes to GitHub (commit 9a647d1)

---

## 🧪 Testing Instructions

### Prerequisites Check

Run this to check your Python environment:
```bash
python3 --version  # Should be Python 3.8+
```

### Step 1: Install Dependencies

```bash
cd /Users/nb/Desktop/QSAR_Models
python3 -m pip install -r requirements.txt
```

**Note:** Some packages (h2o, rdkit, torch) are optional and may take time to install.

### Step 2: Install Package in Development Mode

```bash
python3 -m pip install -e .
```

This installs the package in editable mode, so changes to source code are immediately reflected.

### Step 3: Verify Installation

Run this Python code to test:

```python
import sys
sys.path.insert(0, '/Users/nb/Desktop/QSAR_Models/src')

# Test imports
from qsar_validation.model_agnostic_pipeline import QSARPipeline
from qsar_validation.splitting_strategies import RandomSplit, TemporalSplit
from utils.qsar_utils_no_leakage import quick_clean, clean_qsar_data_with_report

print("✅ All imports successful!")
```

### Step 4: Test Data Cleaning

```python
import pandas as pd
from utils.qsar_utils_no_leakage import quick_clean

# Create test data
data = pd.DataFrame({
    'SMILES': ['CCO', 'CC(C)O', 'CCO', 'c1ccccc1'],
    'Activity': [2.5, 3.1, 2.5, 1.8]
})

# Clean data
cleaned = quick_clean(data, 'SMILES', 'Activity')
print(f"Original: {len(data)} rows → Cleaned: {len(cleaned)} rows")
```

### Step 5: Run Example Workflow

```bash
cd /Users/nb/Desktop/QSAR_Models
python3 examples/01_basic_validation.py
```

---

## 📦 Package Usage (After Installation)

Once installed with `pip install -e .`, you can import anywhere:

```python
# No need for sys.path.insert() anymore!
from qsar_validation.model_agnostic_pipeline import QSARPipeline
from utils.qsar_utils_no_leakage import quick_clean
```

---

## 🔍 Troubleshooting

### Issue: ModuleNotFoundError for numpy/pandas
**Solution:** Install dependencies first:
```bash
python3 -m pip install numpy pandas scipy scikit-learn
```

### Issue: Cannot import from qsar_validation
**Solution:** Make sure you're in the right directory and use:
```python
import sys
sys.path.insert(0, '/path/to/QSAR_Models/src')
```

Or install the package:
```bash
cd /path/to/QSAR_Models
python3 -m pip install -e .
```

### Issue: RDKit not found
**Solution:** RDKit requires conda:
```bash
conda install -c conda-forge rdkit
```

---

## 📊 Repository Structure (After Cleanup)

```
QSAR_Models/
├── README.md                  # Main documentation
├── INSTALL.md                 # Installation guide
├── setup.py                   # Package configuration
├── requirements.txt           # Dependencies
├── src/
│   ├── qsar_validation/      # Core validation modules
│   └── utils/                 # Utility functions
├── examples/                  # Usage examples (8 files)
├── notebooks/                 # Jupyter notebooks (5 files)
├── tests/                     # Test suite
└── comprehensive_test/        # Comprehensive tests
```

---

## 🎯 Quick Test Command

Create and run this test script:

```python
# test_quick.py
import sys
sys.path.insert(0, '/Users/nb/Desktop/QSAR_Models/src')

print("Testing QSAR Framework...")
try:
    from utils.qsar_utils_no_leakage import quick_clean
    print("✅ Imports work!")
    print("✅ Framework is ready to use!")
except Exception as e:
    print(f"❌ Error: {e}")
```

Run it:
```bash
python3 test_quick.py
```

---

## ✨ Success Criteria

- [ ] All dependencies installed
- [ ] Package installs without errors  (`pip install -e .`)
- [ ] Core modules import successfully
- [ ] Example scripts run without errors
- [ ] Data cleaning functions work correctly

---

**Last Updated:** January 7, 2026  
**Repository:** Roy-QSAR-Generative-dev (bhatnira/main)  
**Version:** 4.1.0
