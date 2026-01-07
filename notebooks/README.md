# QSAR Model Notebooks

This folder contains Jupyter notebooks demonstrating QSAR modeling with the **Framework v4.1.0** including comprehensive data leakage prevention.

---

## 📚 Available Notebooks

### 1. **DATA_LEAKAGE_FIX_EXAMPLE.ipynb**
Complete tutorial on data leakage prevention for QSAR models.
- Duplicate/near-duplicate removal
- Scaffold-based splitting
- Proper feature scaling
- Cross-validation best practices

### 2. **Model_1** - Circular Fingerprint + H2O AutoML
Morgan fingerprints (1024 bits) with H2O AutoML and model interpretation.

### 3. **Model_2** - ChEBERTa Embeddings + Linear Regression
Transformer-based molecular embeddings with linear regression.

### 4. **Model_3** - RDKit Features + H2O AutoML
RDKit molecular descriptors with H2O AutoML.

### 5. **Model_4** - Gaussian Process + Bayesian Optimization
Morgan fingerprints with Gaussian Process regression and Bayesian hyperparameter optimization.

---

## 🚀 Quick Start (After Cloning from GitHub)

### Step 1: Clone the Repository
```bash
git clone https://github.com/bhatnira/Roy-QSAR-Generative-dev.git
cd Roy-QSAR-Generative-dev
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run Notebooks
```bash
cd notebooks
jupyter notebook
```
or
```bash
jupyter lab
```

### Step 4: Open Any Notebook and Run!
The notebooks will **automatically detect** the framework path. Just run the first cell (imports) and you're ready to go!

---

## 🛡️ Framework v4.1.0 Features

### Core Modules Loaded in Each Notebook:

1. **QSARDataProcessor**
   - Duplicate removal (exact and near-duplicates)
   - SMILES canonicalization
   - Tanimoto similarity threshold: ≥ 0.95

2. **AdvancedSplitter**
   - Scaffold-based splitting (Bemis-Murcko)
   - Temporal splitting
   - Stratified splitting
   - Train/validation/test support

3. **FeatureScaler**
   - Proper scaling (fit on train only!)
   - StandardScaler, MinMaxScaler, RobustScaler
   - Prevents information leakage

4. **FeatureSelector**
   - Variance threshold
   - Correlation-based selection
   - Model-based selection
   - Prevents overfitting

5. **DatasetQualityAnalyzer**
   - Dataset size analysis
   - Chemical diversity assessment
   - Activity distribution checks
   - Chemical space coverage

6. **PerformanceValidator**
   - Scaffold-based cross-validation
   - Multiple metrics (R², RMSE, MAE)
   - Proper fold assignment

7. **ActivityCliffsDetector**
   - Activity cliff detection
   - Model reliability assessment
   - Structure-activity relationship analysis

8. **UncertaintyEstimator** (Model 4 only)
   - Uncertainty quantification
   - Confidence intervals
   - Prediction reliability

---

## 📖 How the Notebooks Work

### Automatic Path Detection
Each notebook automatically finds the framework using:

```python
# Auto-detect framework path (works when cloned from GitHub)
current_dir = os.path.dirname(os.path.abspath('__file__')) if '__file__' in dir() else os.getcwd()
repo_root = os.path.abspath(os.path.join(current_dir, '..'))
framework_path = os.path.join(repo_root, 'src')
```

**This means:**
- ✅ Works when cloned from GitHub
- ✅ Works in Jupyter Notebook
- ✅ Works in Jupyter Lab
- ✅ Works in VS Code
- ✅ Works in Google Colab (with minor adjustment)

---

## 🌐 Google Colab Setup

If you want to run these notebooks on Google Colab:

```python
# Add this cell at the beginning (before imports)
!git clone https://github.com/bhatnira/Roy-QSAR-Generative-dev.git
%cd Roy-QSAR-Generative-dev/notebooks

# Install dependencies
!pip install -r ../requirements.txt

# Then run the rest of the notebook normally
```

---

## 📊 Example Workflow

Here's a typical workflow using the framework:

```python
# 1. Import framework (run the import cell)
# Already done in first cell of each notebook!

# 2. Load your data
df = pd.read_excel('your_data.xlsx')

# 3. Remove duplicates BEFORE splitting
processor = QSARDataProcessor(smiles_col='SMILES', target_col='Activity')
df = processor.canonicalize_smiles(df)
df = processor.remove_duplicates(df, strategy='average')

# 4. Scaffold-based split
splitter = AdvancedSplitter()
splits = splitter.scaffold_split(
    df,
    smiles_col='SMILES',
    target_col='Activity',
    test_size=0.2,
    val_size=0.1
)
train_idx, val_idx, test_idx = splits['train_idx'], splits['val_idx'], splits['test_idx']

# 5. Generate features AFTER splitting
# (Your feature generation code here - fingerprints, descriptors, embeddings)

# 6. Scale features properly
scaler = FeatureScaler(method='standard')
X_train_scaled = scaler.fit_transform(X_train)  # Fit on train
X_val_scaled = scaler.transform(X_val)          # Transform val
X_test_scaled = scaler.transform(X_test)        # Transform test

# 7. Train your model
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 8. Validate with scaffold-based CV
validator = PerformanceValidator()
cv_results = validator.cross_validate(model, X_train_scaled, y_train, cv=5)

# 9. Evaluate on test set
test_preds = model.predict(X_test_scaled)
```

---

## ⚠️ Common Issues

### Issue: "ModuleNotFoundError: No module named 'utils'"
**Solution:** Make sure you're running from the `notebooks/` folder and the `src/` folder exists in the parent directory.

```bash
# Check your location
pwd  # Should end with /notebooks

# Check if src exists
ls ../src  # Should show utils/ and qsar_validation/
```

### Issue: Framework path not found
**Solution:** The notebook will show a warning. Make sure you cloned the complete repository:

```bash
git clone https://github.com/bhatnira/Roy-QSAR-Generative-dev.git
cd Roy-QSAR-Generative-dev/notebooks
```

### Issue: Missing dependencies
**Solution:** Install all required packages:

```bash
pip install -r requirements.txt
```

---

## 📦 Required Packages

The framework requires:
- pandas
- numpy
- scikit-learn
- rdkit
- matplotlib
- seaborn
- xgboost (optional)
- lightgbm (optional)
- h2o (optional, for H2O AutoML models)

Install via:
```bash
pip install -r ../requirements.txt
```

---

## 🧪 Testing

To verify the framework is working correctly:

```bash
cd ../comprehensive_test
python test_all_modules_simple.py
```

This runs comprehensive tests on all 12 framework modules with a synthetic QSAR dataset.

---

## 📝 Documentation

- **Framework Integration Summary**: `FRAMEWORK_INTEGRATION_SUMMARY.md`
- **Comprehensive Test Results**: `../comprehensive_test/TEST_SUMMARY.md`
- **Overall Status**: `../FINAL_STATUS_REPORT.md`

---

## 🤝 Contributing

If you find issues or want to contribute improvements:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📧 Support

For questions or issues:
- Check the documentation in `FRAMEWORK_INTEGRATION_SUMMARY.md`
- Review the example notebook: `DATA_LEAKAGE_FIX_EXAMPLE.ipynb`
- Examine the test suite: `../comprehensive_test/`

---

## 🎯 Key Principles

### Data Leakage Prevention:
1. ✅ Remove duplicates **BEFORE** splitting
2. ✅ Use scaffold-based splits (not random!)
3. ✅ Generate features **AFTER** splitting
4. ✅ Fit scalers on **train only**, transform val/test
5. ✅ Use scaffold-based CV (not random K-Fold)
6. ✅ Verify no SMILES overlap between sets

### Best Practices:
- Use realistic data splits (scaffold-based)
- Always check dataset quality
- Validate with proper cross-validation
- Monitor for activity cliffs
- Assess model complexity vs. dataset size
- Report uncertainty when possible

---

## 📄 License

See the main repository for license information.

---

## ⭐ Citation

If you use this framework in your research, please cite the repository:

```
Roy QSAR Generative Framework v4.1.0
https://github.com/bhatnira/Roy-QSAR-Generative-dev
```

---

**Happy Modeling! 🚀**
