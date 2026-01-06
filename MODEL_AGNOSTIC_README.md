# Model-Agnostic QSAR Validation Framework

## A Modular and Reproducible Framework

**Version 3.0.0 - Now Completely Model-Agnostic!**

### What's New in v3.0?

🎯 **Complete Freedom**: You choose ANY model, ANY featurizer
- Bring your own models (Random Forest, XGBoost, Neural Networks, anything!)
- Bring your own features (Morgan, MACCS, descriptors, embeddings, anything!)
- Framework handles ALL data leakage prevention and validation automatically

### Key Features

#### 1. 🛡️ **Data Leakage Prevention** (Automatic)
- ✅ Scaffold-based splitting (Bemis-Murcko, zero overlap guaranteed)
- ✅ Duplicate removal BEFORE splitting
- ✅ Feature scaling using ONLY training statistics
- ✅ Feature selection within CV loops only
- ✅ Nested CV support for hyperparameter tuning

#### 2. 🎯 **Model-Agnostic** (NEW!)
- Works with ANY sklearn-compatible model
- Random Forest, Ridge, XGBoost, SVR, Neural Networks, etc.
- Your custom models too!

#### 3. 🧬 **Featurizer-Agnostic** (NEW!)
- Works with ANY featurizer function
- Morgan fingerprints, MACCS keys, RDKit descriptors
- ChemBERTa embeddings, custom features, anything!

#### 4. 📊 **Comprehensive Validation** (Automatic)
- Cross-validation (no leakage)
- Y-randomization tests
- Activity cliff detection
- Model complexity analysis
- Experimental error estimation
- Performance metrics (R², RMSE, MAE, etc.)

#### 5. 🚀 **Ultra Simple to Use**
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

---

## 🚀 Quick Start

### Installation

```bash
pip install git+https://github.com/bhatnira/Roy-QSAR-Generative-dev.git
```

### Minimal Example (10 Lines!)

```python
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestRegressor
from qsar_validation import ModelAgnosticQSARPipeline

# 1. Load your data
df = pd.read_csv('your_data.csv')

# 2. Define your featurizer (any function: SMILES -> features)
def my_featurizer(smiles_list):
    fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, 1024) 
           for s in smiles_list]
    return np.array([np.array(fp) for fp in fps])

# 3. Choose your model (any sklearn-compatible model)
my_model = RandomForestRegressor(n_estimators=100, random_state=42)

# 4. Create pipeline (handles ALL validation + leakage prevention)
pipeline = ModelAgnosticQSARPipeline(
    featurizer=my_featurizer,
    model=my_model,
    smiles_col='SMILES',
    target_col='Activity'
)

# 5. Run everything!
results = pipeline.fit_predict_validate(df, verbose=True)

# Done! You now have:
# ✅ Data leakage-free split (scaffold-based)
# ✅ Trained model
# ✅ Complete validation metrics
# ✅ Activity cliff detection
# ✅ Y-randomization test
# ✅ All checks passed
```

**That's it!** The pipeline handles:
- Duplicate removal (before splitting)
- Scaffold-based splitting (zero overlap)
- Feature scaling (train statistics only)
- Model training
- Comprehensive validation
- Activity cliff detection
- Y-randomization tests
- Data leakage verification

---

## 📚 Examples

### Example 1: Random Forest + Morgan Fingerprints

```python
from sklearn.ensemble import RandomForestRegressor

def morgan_featurizer(smiles_list):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
        fps.append(np.array(fp))
    return np.array(fps)

model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)

pipeline = ModelAgnosticQSARPipeline(morgan_featurizer, model, 'SMILES', 'Activity')
results = pipeline.fit_predict_validate(df)
```

### Example 2: XGBoost + MACCS Keys

```python
from xgboost import XGBRegressor
from rdkit.Chem import MACCSkeys

def maccs_featurizer(smiles_list):
    from rdkit import Chem
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        fp = MACCSkeys.GenMACCSKeys(mol)
        fps.append(np.array(fp))
    return np.array(fps)

model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1)

pipeline = ModelAgnosticQSARPipeline(maccs_featurizer, model, 'SMILES', 'Activity')
results = pipeline.fit_predict_validate(df)
```

### Example 3: Ridge + RDKit Descriptors

```python
from sklearn.linear_model import Ridge
from rdkit.Chem import Descriptors

def descriptor_featurizer(smiles_list):
    from rdkit import Chem
    features = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        desc = [func(mol) for name, func in Descriptors.descList]
        features.append(desc)
    return np.array(features)

model = Ridge(alpha=1.0)

pipeline = ModelAgnosticQSARPipeline(descriptor_featurizer, model, 'SMILES', 'Activity')
results = pipeline.fit_predict_validate(df)
```

### Example 4: Neural Network + Custom Features

```python
from sklearn.neural_network import MLPRegressor

def custom_featurizer(smiles_list):
    # Combine multiple feature types!
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    
    features = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        
        # Morgan fingerprint
        morgan = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 512)
        
        # RDKit descriptors
        descriptors = [func(mol) for name, func in Descriptors.descList[:50]]
        
        # Concatenate
        combined = np.concatenate([np.array(morgan), descriptors])
        features.append(combined)
    
    return np.array(features)

model = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500)

pipeline = ModelAgnosticQSARPipeline(custom_featurizer, model, 'SMILES', 'Activity')
results = pipeline.fit_predict_validate(df)
```

### Example 5: Compare Multiple Models

```python
models = {
    'Random Forest': RandomForestRegressor(n_estimators=100),
    'XGBoost': XGBRegressor(n_estimators=100),
    'Ridge': Ridge(alpha=1.0),
    'SVR': SVR(kernel='rbf'),
    'Neural Net': MLPRegressor(hidden_layer_sizes=(128, 64))
}

results = {}
for name, model in models.items():
    pipeline = ModelAgnosticQSARPipeline(my_featurizer, model, 'SMILES', 'Activity')
    results[name] = pipeline.fit_predict_validate(df, verbose=False)
    
    print(f"{name}: R² = {results[name]['performance']['test']['r2']:.3f}")
```

---

## 🎨 Customization

### Custom Validation Configuration

```python
custom_config = {
    'use_scaffold_split': True,          # Scaffold-based splitting
    'remove_duplicates': True,            # Remove duplicates before split
    'scale_features': True,               # Scale features
    'detect_activity_cliffs': True,       # Detect activity cliffs
    'run_y_randomization': True,          # Run Y-randomization test
    'n_randomization_runs': 20,           # Number of randomization runs
    'cv_folds': 10,                       # Number of CV folds
    'test_size': 0.15,                    # Test set size (15%)
    'val_size': 0.05,                     # Validation set size (5%)
    'random_state': 42
}

pipeline = ModelAgnosticQSARPipeline(
    featurizer=my_featurizer,
    model=my_model,
    smiles_col='SMILES',
    target_col='Activity',
    validation_config=custom_config  # <- Custom settings
)
```

---

## 📊 What You Get

### Complete Results Dictionary

```python
results = pipeline.fit_predict_validate(df)

# Performance metrics
results['performance']['test']['r2']       # Test R²
results['performance']['test']['rmse']     # Test RMSE
results['performance']['train']['r2']      # Train R²

# Cross-validation
results['cross_validation']['cv_r2_mean']  # CV R² mean
results['cross_validation']['cv_r2_std']   # CV R² std

# Y-randomization
results['y_randomization']['r2_mean']      # Random R² mean

# Activity cliffs
results['activity_cliffs']                  # List of cliffs

# Dataset stats
results['dataset_stats']['scaffold_diversity']
results['dataset_stats']['activity_distribution']

# Data leakage checks
results['data_leakage_checks']             # All checks
```

### Results Summary

```python
summary = pipeline.get_results_summary()
print(summary)
```

Output:
```
       Category                    Metric              Value
0   Performance                  Train R²              0.950
1   Performance                 Train RMSE              0.250
2   Performance             Validation R²              0.750
3   Performance            Validation RMSE              0.450
4   Performance                   Test R²              0.720
5   Performance                  Test RMSE              0.480
6  Cross-Validation   CV R² (mean ± std)     0.730 ± 0.050
7  Y-Randomization  Random R² (mean ± std)   0.010 ± 0.080
8       Dataset                 N compounds                150
9       Dataset                 N scaffolds                 45
10   Data Split             Train/Val/Test          105/15/30
11   Data Split            Scaffold overlap                  0
```

### Make Predictions

```python
# Predict on new compounds
new_smiles = ['CCO', 'c1ccccc1', 'CC(=O)O']
predictions = pipeline.predict(new_smiles)
print(predictions)
# Output: [5.2, 6.8, 4.1]
```

---

## 🛡️ Data Leakage Prevention

The pipeline **automatically** prevents all 5 types of data leakage:

| Leakage Type | Prevention Method | Status |
|--------------|-------------------|--------|
| **Scaffold Leakage** | Bemis-Murcko scaffold-based splitting | ✅ Automatic |
| **Duplicate Leakage** | Remove duplicates BEFORE splitting | ✅ Automatic |
| **Feature Scaling** | Scalers fit ONLY on training data | ✅ Automatic |
| **Feature Selection** | Selection within CV loops only | ✅ User responsibility |
| **Hyperparameter Tuning** | Nested CV support provided | ✅ User responsibility |

See [DATA_LEAKAGE_PREVENTION.md](DATA_LEAKAGE_PREVENTION.md) for full details.

---

## 📁 Your Data Format

Just a simple CSV file:

```csv
SMILES,Activity
CCO,5.2
c1ccccc1,6.8
CC(=O)O,4.1
...
```

That's it! The pipeline handles everything else.

---

## 🎯 Why Use This Framework?

### ❌ Without This Framework:
```python
# You write 500+ lines of code
# Risk of data leakage
# Forget validation steps
# Inconsistent results
# Hard to reproduce
```

### ✅ With This Framework:
```python
# Just 5 lines of code
# Zero data leakage (automatic)
# Complete validation (automatic)
# Consistent results
# Fully reproducible
```

---

## 📖 Documentation

- **[USAGE_GUIDE.md](USAGE_GUIDE.md)**: Complete usage guide
- **[DATA_LEAKAGE_PREVENTION.md](DATA_LEAKAGE_PREVENTION.md)**: Data leakage prevention details
- **[QUICK_START.md](QUICK_START.md)**: Quick reference
- **[examples/](examples/)**: More example scripts

---

## 🎓 For Low-Data Regimes (< 200 compounds)

This framework is specifically designed for low-data QSAR:

✅ Scaffold-based splitting (prevents optimistic bias)
✅ Y-randomization (detects overfitting)
✅ Model complexity analysis (ensures appropriate model)
✅ Activity cliff detection (identifies SAR discontinuities)
✅ Experimental error estimation (realistic expectations)

---

## 🤝 Contributing

This is a research framework. Feel free to:
- Use it in your projects
- Modify for your needs
- Report issues
- Suggest improvements

---

## 📝 Citation

If you use this framework in your research, please cite:

```bibtex
@software{qsar_validation_framework,
  title = {Model-Agnostic QSAR Validation Framework},
  author = {QSAR Validation Team},
  year = {2026},
  version = {3.0.0},
  url = {https://github.com/bhatnira/Roy-QSAR-Generative-dev}
}
```

---

## 📞 Support

- **Documentation**: See files above
- **Examples**: See `examples/` folder
- **Issues**: Report on GitHub

---

## ⭐ Star This Repo!

If you find this framework useful, please star it on GitHub!

---

**Framework**: Model-Agnostic QSAR Validation  
**Version**: 3.0.0  
**License**: MIT  
**Maintained**: Yes (actively)

**Remember**: You bring the model and features, we handle everything else! 🚀
