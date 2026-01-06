# Model-Agnostic QSAR: Quick Start

## 5-Minute Tutorial

### Installation (30 seconds)

```bash
pip install git+https://github.com/bhatnira/Roy-QSAR-Generative-dev.git
```

---

### Complete Example (5 minutes)

```python
# ============================================================
# STEP 1: Import libraries
# ============================================================
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestRegressor
from qsar_validation import ModelAgnosticQSARPipeline

# ============================================================
# STEP 2: Load your data (any CSV with SMILES + activity)
# ============================================================
df = pd.read_csv('your_data.csv')
# Expected format:
#   SMILES,Activity
#   CCO,5.2
#   c1ccccc1,6.8
#   ...

# ============================================================
# STEP 3: Define YOUR featurizer (any function works!)
# ============================================================
def my_featurizer(smiles_list):
    """
    YOUR choice of features!
    - This example uses Morgan fingerprints
    - You can use MACCS, descriptors, embeddings, anything!
    """
    fingerprints = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
            fingerprints.append(np.array(fp))
        else:
            fingerprints.append(np.zeros(1024))
    return np.array(fingerprints)

# ============================================================
# STEP 4: Choose YOUR model (any sklearn-compatible model!)
# ============================================================
my_model = RandomForestRegressor(
    n_estimators=100,     # YOUR choice
    max_depth=10,         # YOUR choice
    random_state=42
)
# You can use: XGBRegressor, Ridge, SVR, MLPRegressor, anything!

# ============================================================
# STEP 5: Create pipeline (handles ALL validation + leakage)
# ============================================================
pipeline = ModelAgnosticQSARPipeline(
    featurizer=my_featurizer,     # YOUR featurizer
    model=my_model,                # YOUR model
    smiles_col='SMILES',           # YOUR column name
    target_col='Activity'          # YOUR column name
)

# ============================================================
# STEP 6: Run complete validation (ONE function call!)
# ============================================================
results = pipeline.fit_predict_validate(df, verbose=True)

# ============================================================
# DONE! The pipeline just did:
# ✅ Removed duplicates (before splitting - no leakage)
# ✅ Scaffold-based splitting (zero overlap - no leakage)
# ✅ Feature scaling (train stats only - no leakage)
# ✅ Model training
# ✅ Cross-validation (no leakage)
# ✅ Y-randomization test
# ✅ Activity cliff detection
# ✅ Performance metrics
# ✅ Data leakage verification
# ============================================================

# ============================================================
# STEP 7: Check results
# ============================================================
print(f"Test R²: {results['performance']['test']['r2']:.3f}")
print(f"Test RMSE: {results['performance']['test']['rmse']:.3f}")
print(f"Data leakage: {results['data_leakage_checks']['scaffold_overlap']} scaffolds overlap (should be 0)")

# ============================================================
# STEP 8: Make predictions on new compounds
# ============================================================
new_smiles = ['CCO', 'c1ccccc1', 'CC(=O)O']
predictions = pipeline.predict(new_smiles)
print(f"Predictions: {predictions}")

# ============================================================
# STEP 9: Get summary table
# ============================================================
summary = pipeline.get_results_summary()
print(summary)
```

---

## What Just Happened?

You just got:

✅ **Complete QSAR model** trained on your data  
✅ **Zero data leakage** (scaffold-based splitting)  
✅ **Comprehensive validation** (CV, Y-randomization, activity cliffs)  
✅ **Performance metrics** (R², RMSE, MAE, etc.)  
✅ **Ready to predict** on new compounds  

All with **just ~40 lines** of simple code!

---

## Try Different Models

### XGBoost
```python
from xgboost import XGBRegressor

model = XGBRegressor(n_estimators=100, max_depth=6)
pipeline = ModelAgnosticQSARPipeline(my_featurizer, model, 'SMILES', 'Activity')
results = pipeline.fit_predict_validate(df)
```

### Ridge Regression
```python
from sklearn.linear_model import Ridge

model = Ridge(alpha=1.0)
pipeline = ModelAgnosticQSARPipeline(my_featurizer, model, 'SMILES', 'Activity')
results = pipeline.fit_predict_validate(df)
```

### Neural Network
```python
from sklearn.neural_network import MLPRegressor

model = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500)
pipeline = ModelAgnosticQSARPipeline(my_featurizer, model, 'SMILES', 'Activity')
results = pipeline.fit_predict_validate(df)
```

---

## Try Different Features

### MACCS Keys
```python
from rdkit.Chem import MACCSkeys

def maccs_featurizer(smiles_list):
    fps = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        fp = MACCSkeys.GenMACCSKeys(mol)
        fps.append(np.array(fp))
    return np.array(fps)

pipeline = ModelAgnosticQSARPipeline(maccs_featurizer, my_model, 'SMILES', 'Activity')
```

### RDKit Descriptors
```python
from rdkit.Chem import Descriptors

def descriptor_featurizer(smiles_list):
    features = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        desc = [func(mol) for name, func in Descriptors.descList]
        features.append(desc)
    return np.array(features)

pipeline = ModelAgnosticQSARPipeline(descriptor_featurizer, my_model, 'SMILES', 'Activity')
```

### Custom Features (Combine Multiple!)
```python
def custom_featurizer(smiles_list):
    features = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        
        # Morgan fingerprints
        morgan = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 512)
        
        # Descriptors
        descriptors = [func(mol) for name, func in Descriptors.descList[:50]]
        
        # Concatenate
        combined = np.concatenate([np.array(morgan), descriptors])
        features.append(combined)
    
    return np.array(features)

pipeline = ModelAgnosticQSARPipeline(custom_featurizer, my_model, 'SMILES', 'Activity')
```

---

## Customize Validation Settings

```python
custom_config = {
    'use_scaffold_split': True,       # Scaffold-based splitting
    'remove_duplicates': True,         # Remove duplicates
    'scale_features': True,            # Scale features
    'detect_activity_cliffs': True,    # Detect cliffs
    'run_y_randomization': True,       # Y-randomization test
    'n_randomization_runs': 20,        # 20 runs (default: 10)
    'cv_folds': 10,                    # 10-fold CV (default: 5)
    'test_size': 0.15,                 # 15% test (default: 0.2)
    'val_size': 0.05,                  # 5% val (default: 0.1)
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

## Your Data Format

Just a simple CSV:

```csv
SMILES,Activity
CCO,5.2
c1ccccc1,6.8
CC(=O)O,4.1
CCCC,5.9
c1ccccc1O,6.2
...
```

Column names can be anything - you specify them in the pipeline!

---

## Next Steps

1. **More Examples**: See `examples/model_agnostic_examples.py`
2. **Full Guide**: See `MODEL_AGNOSTIC_README.md`
3. **Data Leakage**: See `DATA_LEAKAGE_PREVENTION.md`

---

## Questions?

**Q: What models can I use?**  
A: Any sklearn-compatible model (has `.fit()` and `.predict()` methods)

**Q: What features can I use?**  
A: Any function that converts SMILES list to numpy array

**Q: Is data leakage prevented?**  
A: YES! Automatic scaffold-based splitting, duplicate removal, proper scaling

**Q: How much code do I need?**  
A: ~5 lines minimum, ~40 lines for full example

**Q: Does it work with small datasets?**  
A: YES! Designed specifically for low-data regime (< 200 compounds)

---

**Happy Modeling! 🚀**

Remember: You bring the model and features, we handle everything else!
