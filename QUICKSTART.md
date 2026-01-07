# Quick Start Guide - After Cloning from GitHub

Welcome! You've just cloned the **QSAR Validation Framework v4.1.0**. Here's how to get started in 5 minutes.

---

## ✅ Step 1: Verify You Have Everything

After cloning, your directory structure should look like this:

```
Roy-QSAR-Generative-dev/
├── README.md                   # Main documentation
├── requirements.txt            # Python dependencies
├── setup.py                    # Package installer
├── src/                        # Framework source code
│   ├── utils/                  # Core utilities
│   └── qsar_validation/        # Validation modules
├── notebooks/                  # Example notebooks ⭐
│   ├── README.md              # Notebook documentation
│   ├── DATA_LEAKAGE_FIX_EXAMPLE.ipynb
│   ├── Model_1_...ipynb
│   ├── Model_2_...ipynb
│   ├── Model_3_...ipynb
│   └── Model_4_...ipynb
├── comprehensive_test/         # Test suite
└── examples/                   # Example scripts
```

---

## 🚀 Step 2: Install Dependencies

```bash
# Make sure you're in the repository root
cd Roy-QSAR-Generative-dev

# Install all dependencies
pip install -r requirements.txt
```

**What gets installed:**
- Core: pandas, numpy, rdkit, scipy, matplotlib, seaborn
- ML: scikit-learn, xgboost (lightgbm is optional)
- Notebooks: jupyter

---

## 🎯 Step 3: Choose Your Path

### Path A: I Want to Run the Example Notebooks 📓

```bash
cd notebooks
jupyter notebook
```

1. Open `DATA_LEAKAGE_FIX_EXAMPLE.ipynb` first
2. Run all cells - it will work out of the box!
3. Explore the other 4 model notebooks

**The notebooks automatically find the framework** - no configuration needed!

### Path B: I Want to Use the Framework in My Own Code 💻

```python
# In your Python script or notebook
import sys
import os

# Add framework to path
sys.path.insert(0, '/path/to/Roy-QSAR-Generative-dev/src')

# Import what you need
from utils.qsar_utils_no_leakage import QSARDataProcessor
from qsar_validation.splitting_strategies import AdvancedSplitter
from qsar_validation.feature_scaling import FeatureScaler

# Use the modules
processor = QSARDataProcessor(smiles_col='SMILES')
# ... your code here
```

### Path C: I Want to Run the Tests 🧪

```bash
cd comprehensive_test
python test_all_modules_simple.py
```

This tests all 12 framework modules with synthetic QSAR data.

---

## 📚 Step 4: Learn the Framework

### Quick Reference:

**Data Processing:**
```python
from utils.qsar_utils_no_leakage import QSARDataProcessor

processor = QSARDataProcessor(smiles_col='SMILES', target_col='Activity')
df = processor.canonicalize_smiles(df)
df = processor.remove_duplicates(df, strategy='average')
```

**Data Splitting:**
```python
from qsar_validation.splitting_strategies import AdvancedSplitter

splitter = AdvancedSplitter()
splits = splitter.scaffold_split(
    df,
    smiles_col='SMILES',
    target_col='Activity',
    test_size=0.2,
    val_size=0.1
)
train_idx, val_idx, test_idx = splits['train_idx'], splits['val_idx'], splits['test_idx']
```

**Feature Scaling:**
```python
from qsar_validation.feature_scaling import FeatureScaler

scaler = FeatureScaler(method='standard')
X_train_scaled = scaler.fit_transform(X_train)  # Fit on train only!
X_test_scaled = scaler.transform(X_test)        # Transform test
```

**Model Validation:**
```python
from qsar_validation.performance_validation import PerformanceValidator

validator = PerformanceValidator()
cv_results = validator.cross_validate(model, X_train, y_train, cv=5)
```

---

## 🎓 Step 5: Read the Documentation

1. **Framework overview:** [`README.md`](README.md) (main file)
2. **Notebook guide:** [`notebooks/README.md`](notebooks/README.md)
3. **Framework integration:** [`notebooks/FRAMEWORK_INTEGRATION_SUMMARY.md`](notebooks/FRAMEWORK_INTEGRATION_SUMMARY.md)
4. **Test results:** [`comprehensive_test/TEST_SUMMARY.md`](comprehensive_test/TEST_SUMMARY.md)
5. **Status report:** [`FINAL_STATUS_REPORT.md`](FINAL_STATUS_REPORT.md)

---

## ⚡ Common Issues

### Issue 1: "ModuleNotFoundError: No module named 'utils'"

**Cause:** Framework path not in Python path.

**Solution:**
```python
import sys
import os
sys.path.insert(0, '/absolute/path/to/Roy-QSAR-Generative-dev/src')
```

Or if running from notebooks folder:
```python
import sys
import os
current_dir = os.getcwd()
repo_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, os.path.join(repo_root, 'src'))
```

### Issue 2: "No module named 'rdkit'"

**Cause:** RDKit not installed.

**Solution:**
```bash
pip install rdkit
# or
conda install -c conda-forge rdkit
```

### Issue 3: Jupyter not starting

**Cause:** Jupyter not installed.

**Solution:**
```bash
pip install jupyter
```

### Issue 4: "No module named 'xgboost'" (or lightgbm)

**Cause:** Optional ML libraries not installed.

**Solution:**
```bash
pip install xgboost lightgbm
```

---

## 🎯 What to Do Next

### If you're new to QSAR modeling:
1. ✅ Read `notebooks/DATA_LEAKAGE_FIX_EXAMPLE.ipynb`
2. ✅ Understand why data leakage matters
3. ✅ Try running one of the model notebooks
4. ✅ Adapt the workflow to your data

### If you're experienced with QSAR:
1. ✅ Review the module list in `README.md`
2. ✅ Pick the modules you need
3. ✅ Integrate into your existing workflow
4. ✅ Use the framework to validate your models

### If you want to contribute:
1. ✅ Run the test suite: `cd comprehensive_test && python test_all_modules_simple.py`
2. ✅ Understand the module structure in `src/`
3. ✅ Check open issues on GitHub
4. ✅ Submit a pull request!

---

## 🌟 Key Features to Know

### 1. Multi-Library Support
The framework works with:
- ✅ scikit-learn
- ✅ XGBoost
- ✅ LightGBM
- ✅ PyTorch
- ✅ TensorFlow/Keras
- ✅ Custom models

### 2. Modular Design
- Use only the modules you need
- No forced workflows
- Easy to integrate with existing code

### 3. Data Leakage Prevention
- Scaffold-based splitting
- Proper feature scaling (fit on train only)
- Near-duplicate detection (Tanimoto ≥ 0.95)
- Cross-validation with proper fold assignment

### 4. Comprehensive Validation
- Dataset quality analysis
- Model complexity control
- Activity cliff detection
- Uncertainty estimation
- Performance metrics

---

## 📧 Need Help?

1. **Check the documentation:**
   - Main README: [`README.md`](README.md)
   - Notebooks guide: [`notebooks/README.md`](notebooks/README.md)
   - Framework summary: [`notebooks/FRAMEWORK_INTEGRATION_SUMMARY.md`](notebooks/FRAMEWORK_INTEGRATION_SUMMARY.md)

2. **Run the examples:**
   - Notebooks: `notebooks/`
   - Test suite: `comprehensive_test/`

3. **Open an issue:**
   - GitHub Issues: https://github.com/bhatnira/Roy-QSAR-Generative-dev/issues

---

## ✨ Quick Tips

1. **Always remove duplicates BEFORE splitting**
   ```python
   df = processor.remove_duplicates(df, strategy='average')
   # THEN split
   ```

2. **Use scaffold-based splitting (not random!)**
   ```python
   splitter = AdvancedSplitter()
   splits = splitter.scaffold_split(...)  # ✅ Good
   # NOT: train_test_split(random_state=42)  # ❌ Bad
   ```

3. **Fit scalers on train only**
   ```python
   scaler.fit(X_train)           # ✅ Good
   X_train_scaled = scaler.transform(X_train)
   X_test_scaled = scaler.transform(X_test)
   
   # NOT: scaler.fit(X_all)  # ❌ Bad (data leakage!)
   ```

4. **Use proper cross-validation**
   ```python
   validator = PerformanceValidator()
   cv_results = validator.cross_validate(...)  # ✅ Good
   # NOT: random K-Fold  # ❌ Bad
   ```

---

## 🎉 You're Ready!

You now have:
- ✅ Framework installed
- ✅ Dependencies ready
- ✅ Notebooks available
- ✅ Documentation accessible
- ✅ Quick reference at hand

**Start with:** `notebooks/DATA_LEAKAGE_FIX_EXAMPLE.ipynb`

**Questions?** Check the documentation or open an issue!

---

**Happy Modeling! 🚀**
