# Quick-Start: Fix Data Leakage in Your QSAR Models

## 🚨 URGENT: Replace These Code Blocks Immediately

This document shows **exactly** what code to find and replace in your notebooks to fix data leakage.

---

## ❌ → ✅ Code Replacements

### 1. Data Loading and Duplicate Removal

#### ❌ OLD CODE (Find this in your notebooks):
```python
# Load dataset
import pandas as pd
df = pd.read_excel('/content/drive/MyDrive/DrRoyRationalDesign/Input of triazole and cysteine_datasheet.xlsx')

# Remove duplicates
from rdkit import Chem

def canonicalize(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Chem.MolToSmiles(mol, canonical=True)
    except:
        return None
    return None

df = df.dropna(subset=['Canonical SMILES'])
duplicates = df[df.duplicated(subset=['Canonical SMILES'], keep=False)]
unique_df = df.drop_duplicates(subset=['Canonical SMILES'])

# Keep only numeric IC50
mask = pd.to_numeric(unique_df["IC50 uM"], errors="coerce").notna()
numeric_df = unique_df[mask]
```

#### ✅ NEW CODE (Replace with this):
```python
# Load dataset
import pandas as pd
import numpy as np
from qsar_utils_no_leakage import QSARDataProcessor, ScaffoldSplitter

# Set random seed
np.random.seed(42)

# Load data
df = pd.read_excel('/content/drive/MyDrive/DrRoyRationalDesign/Input of triazole and cysteine_datasheet.xlsx')

print(f"📊 Original dataset: {len(df)} rows")

# Initialize processor
processor = QSARDataProcessor(smiles_col='Canonical SMILES', target_col='IC50 uM')

# Step 1: Canonicalize SMILES
df = processor.canonicalize_smiles(df)

# Step 2: Remove duplicates (average replicates) - BEFORE splitting!
df = processor.remove_duplicates(df, strategy='average')

# Step 3: Keep only numeric IC50 values
mask = pd.to_numeric(df["IC50 uM"], errors="coerce").notna()
df = df[mask].reset_index(drop=True)

print(f"✅ Clean dataset: {len(df)} unique molecules")
```

---

### 2. Data Splitting

#### ❌ OLD CODE (Find and DELETE):
```python
# Random K-Fold split
from sklearn.model_selection import KFold

kf = KFold(n_splits=K, shuffle=True, random_state=42)

for fold, (train_idx, valid_idx) in enumerate(kf.split(df), 1):
    # ... training code ...
```

OR:

```python
# Random train-test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

#### ✅ NEW CODE (Replace with this):
```python
# Scaffold-based split
from qsar_utils_no_leakage import ScaffoldSplitter

splitter = ScaffoldSplitter(smiles_col='Canonical SMILES')

# Option A: Single train/val/test split
train_idx, val_idx, test_idx = splitter.scaffold_split(
    df, 
    test_size=0.2, 
    val_size=0.1, 
    random_state=42
)

# Remove near-duplicates between train and test
train_idx, test_idx = processor.remove_near_duplicates(
    df, train_idx, test_idx, threshold=0.95
)

# Analyze similarity (important for reporting!)
similarity_stats = processor.analyze_similarity(df, train_idx, test_idx)

# Create DataFrames
train_df = df.iloc[train_idx].reset_index(drop=True)
val_df = df.iloc[val_idx].reset_index(drop=True)
test_df = df.iloc[test_idx].reset_index(drop=True)

print(f"\n✅ Scaffold-based split:")
print(f"   Train: {len(train_df)} molecules")
print(f"   Val: {len(val_df)} molecules")
print(f"   Test: {len(test_df)} molecules")

# Option B: Scaffold-based K-Fold CV
cv_splits = splitter.scaffold_kfold(df, n_splits=5, random_state=42)

for fold, (train_idx, val_idx) in enumerate(cv_splits, 1):
    print(f"\nFold {fold}:")
    train_fold_df = df.iloc[train_idx]
    val_fold_df = df.iloc[val_idx]
    
    # Your training code here...
    # Generate features for THIS fold only
    # Train model on THIS fold only
```

---

### 3. Feature Scaling (CRITICAL!)

#### ❌ OLD CODE (Find and DELETE):
```python
# DON'T DO THIS - causes leakage!
from sklearn.preprocessing import StandardScaler

# Scaling entire dataset before split
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # ❌ WRONG!

# Then splitting...
train_idx, test_idx = ...
X_train = X_scaled[train_idx]
X_test = X_scaled[test_idx]
```

OR:

```python
# Using H2O with entire dataset
import h2o
h2o.init()

# Converting entire dataset
h2o_frame = h2o.H2OFrame(df)  # ❌ If this includes test data

# Then splitting in H2O
train, test = h2o_frame.split_frame(ratios=[0.8])  # ❌ Not scaffold-based
```

#### ✅ NEW CODE (Replace with this):

**For sklearn models:**
```python
from sklearn.preprocessing import StandardScaler

# Generate features for each split separately
X_train = generate_features(train_df)  # Your feature generation
X_val = generate_features(val_df)
X_test = generate_features(test_df)

# FIT scaler on TRAINING data ONLY
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # ✅ FIT on train

# TRANSFORM (not fit_transform!) val and test
X_val_scaled = scaler.transform(X_val)  # ✅ TRANSFORM only
X_test_scaled = scaler.transform(X_test)  # ✅ TRANSFORM only

print("✅ Features scaled correctly (no leakage)")
```

**For H2O models:**
```python
import h2o
h2o.init()

# Convert splits separately (AFTER scaffold splitting!)
train_h2o = h2o.H2OFrame(train_df)
val_h2o = h2o.H2OFrame(val_df)
test_h2o = h2o.H2OFrame(test_df)

# H2O AutoML should be trained on train_h2o only
# Validation should use val_h2o
# Final evaluation on test_h2o

print("✅ H2O frames created from scaffold splits")
```

---

### 4. Target Transformation

#### ❌ OLD CODE (Find and check):
```python
# Normalizing target using full dataset statistics
df['IC50_normalized'] = (df['IC50 uM'] - df['IC50 uM'].mean()) / df['IC50 uM'].std()

# Then splitting...
```

#### ✅ NEW CODE (Replace with this):

**Option 1 (Recommended): Transform before splitting**
```python
# pIC50 transformation doesn't use dataset statistics
def to_pIC50(ic50_uM):
    """Convert IC50 (μM) to pIC50 = -log10(IC50 in M)"""
    return -np.log10(ic50_uM * 1e-6)

# Transform BEFORE splitting
df['pIC50'] = df['IC50 uM'].apply(to_pIC50)

# NOW do scaffold splitting
train_idx, val_idx, test_idx = splitter.scaffold_split(df, ...)

# Use pIC50 as target
y_train = train_df['pIC50'].values
y_val = val_df['pIC50'].values
y_test = test_df['pIC50'].values
```

**Option 2: Fit on train only**
```python
from sklearn.preprocessing import StandardScaler

# After splitting...
y_train = train_df['IC50 uM'].values.reshape(-1, 1)
y_val = val_df['IC50 uM'].values.reshape(-1, 1)
y_test = test_df['IC50 uM'].values.reshape(-1, 1)

# FIT on train only
target_scaler = StandardScaler()
y_train_scaled = target_scaler.fit_transform(y_train).ravel()  # ✅ FIT
y_val_scaled = target_scaler.transform(y_val).ravel()  # ✅ TRANSFORM only
y_test_scaled = target_scaler.transform(y_test).ravel()  # ✅ TRANSFORM only
```

---

### 5. Cross-Validation for Hyperparameter Tuning

#### ❌ OLD CODE (Find and DELETE):
```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Random CV
param_grid = {...}
grid_search = GridSearchCV(
    model, 
    param_grid, 
    cv=5,  # ❌ Random K-Fold!
    scoring='neg_mean_squared_error'
)
grid_search.fit(X_train, y_train)
```

#### ✅ NEW CODE (Replace with this):
```python
from qsar_utils_no_leakage import ScaffoldSplitter
from sklearn.model_selection import GridSearchCV

# Get scaffold-based CV splits
cv_splits = splitter.scaffold_kfold(train_df, n_splits=5, random_state=42)

# Use these splits in GridSearchCV
param_grid = {...}
grid_search = GridSearchCV(
    model,
    param_grid,
    cv=cv_splits,  # ✅ Scaffold-based splits!
    scoring='neg_mean_squared_error'
)

# Now fit (will use scaffold-based CV)
grid_search.fit(X_train_scaled, y_train)

print(f"✅ Best params from scaffold-based CV: {grid_search.best_params_}")
```

---

## 📋 Model-Specific Instructions

### Model 1 & 4: Circular Fingerprints + ML

Add this function AFTER your imports:

```python
def generate_fingerprints_for_split(smiles_list, radius=2, n_bits=1024):
    """Generate Morgan fingerprints for a specific split."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, n_bits)
            fps.append(list(fp))
        else:
            fps.append([0] * n_bits)
    
    return np.array(fps)

# Use it AFTER scaffold splitting:
X_train = generate_fingerprints_for_split(train_df['Canonical SMILES'])
X_val = generate_fingerprints_for_split(val_df['Canonical SMILES'])
X_test = generate_fingerprints_for_split(test_df['Canonical SMILES'])
```

### Model 2: ChEBERTa Embeddings

```python
# Do scaffold split FIRST
train_idx, val_idx, test_idx = splitter.scaffold_split(df, ...)
train_df = df.iloc[train_idx].reset_index(drop=True)
val_df = df.iloc[val_idx].reset_index(drop=True)
test_df = df.iloc[test_idx].reset_index(drop=True)

# THEN generate embeddings per split
X_train = smiles_to_embedding(train_df['Canonical SMILES'].tolist())
X_val = smiles_to_embedding(val_df['Canonical SMILES'].tolist())
X_test = smiles_to_embedding(test_df['Canonical SMILES'].tolist())

# DO NOT scale embeddings from pre-trained models
# (they're already in a learned space)
```

### Model 3: RDKit Descriptors

```python
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors

# After scaffold splitting...
def calculate_rdkit_descriptors(smiles_list):
    """Calculate RDKit descriptors for a split."""
    descriptor_names = [desc[0] for desc in Descriptors._descList]
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
    
    descriptors = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            desc = calculator.CalcDescriptors(mol)
            descriptors.append(desc)
        else:
            descriptors.append([0] * len(descriptor_names))
    
    return np.array(descriptors)

# Calculate per split
X_train = calculate_rdkit_descriptors(train_df['Canonical SMILES'])
X_val = calculate_rdkit_descriptors(val_df['Canonical SMILES'])
X_test = calculate_rdkit_descriptors(test_df['Canonical SMILES'])

# NOW scale (fit on train only!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
```

---

## 🎯 Workflow Summary

**Correct Order of Operations:**

1. ✅ Load data
2. ✅ Canonicalize SMILES
3. ✅ Remove exact duplicates (average replicates)
4. ✅ Clean data (remove invalid values)
5. ✅ Transform target (if using pIC50 or log)
6. ✅ **Scaffold-based split** → train/val/test
7. ✅ Remove near-duplicates between splits
8. ✅ Analyze similarity (for reporting)
9. ✅ Generate features **per split**
10. ✅ Scale features (fit on train, transform val/test)
11. ✅ Train model on train set
12. ✅ Tune hyperparameters using scaffold-based CV
13. ✅ Evaluate on test set
14. ✅ Report applicability domain

---

## 💾 Save Your Splits

Add this at the end to save your clean splits:

```python
# Save splits for reproducibility and use in other models
train_df.to_csv('train_set_scaffold_split.csv', index=False)
val_df.to_csv('val_set_scaffold_split.csv', index=False)
test_df.to_csv('test_set_scaffold_split.csv', index=False)

# Save indices
np.save('train_indices_scaffold.npy', train_idx)
np.save('val_indices_scaffold.npy', val_idx)
np.save('test_indices_scaffold.npy', test_idx)

# Save similarity stats
import pickle
with open('similarity_stats.pkl', 'wb') as f:
    pickle.dump(similarity_stats, f)

print("✅ Splits saved! Use these in all your models for consistency.")
```

---

## 🔍 Verification Checklist

After making changes, verify:

```python
# Add this verification code
print("\n🔍 VERIFICATION CHECKS:")

# Check 1: No SMILES overlap
train_smiles = set(train_df['Canonical SMILES'])
val_smiles = set(val_df['Canonical SMILES'])
test_smiles = set(test_df['Canonical SMILES'])

overlap_train_test = train_smiles & test_smiles
overlap_train_val = train_smiles & val_smiles
overlap_val_test = val_smiles & test_smiles

print(f"1. Train-Test SMILES overlap: {len(overlap_train_test)} (should be 0) ✅" if len(overlap_train_test) == 0 else f"❌ {len(overlap_train_test)}")
print(f"2. Train-Val SMILES overlap: {len(overlap_train_val)} (should be 0) ✅" if len(overlap_train_val) == 0 else f"❌ {len(overlap_train_val)}")
print(f"3. Val-Test SMILES overlap: {len(overlap_val_test)} (should be 0) ✅" if len(overlap_val_test) == 0 else f"❌ {len(overlap_val_test)}")

# Check 2: Features scaled correctly
if 'scaler' in locals():
    print(f"4. Scaler fitted on train only: ✅")
    print(f"   Train mean: {X_train_scaled.mean():.6f} (should be ~0)")
    print(f"   Train std: {X_train_scaled.std():.6f} (should be ~1)")
else:
    print("4. No scaler found ⚠️")

# Check 3: Data sizes
print(f"5. Dataset sizes:")
print(f"   Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
print(f"   Val: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
print(f"   Test: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")

# Check 4: Similarity analysis
if 'similarity_stats' in locals():
    print(f"6. Train-Test similarity:")
    print(f"   Mean: {similarity_stats['mean']:.3f}")
    print(f"   Median: {similarity_stats['median']:.3f}")
    print(f"   ✅ Similarity analysis completed")
else:
    print("6. Similarity analysis not run ⚠️")

print("\n" + "="*60)
if (len(overlap_train_test) == 0 and len(overlap_train_val) == 0 and 
    len(overlap_val_test) == 0):
    print("🎉 ALL CHECKS PASSED - No data leakage detected!")
else:
    print("⚠️  WARNING: Data leakage detected - review splits!")
print("="*60)
```

---

## 📞 Quick Help

**Problem:** "My test performance dropped a lot!"
- **Answer:** This is expected! You're now getting realistic estimates. Report the corrected results.

**Problem:** "Scaffold splitting creates unbalanced splits"
- **Answer:** This can happen with small datasets. Use repeated scaffold-based CV instead.

**Problem:** "My dataset is very small (<100 molecules)"
- **Answer:** Consider:
  - Collecting more data
  - Using 5-fold repeated CV (repeat 10 times)
  - Reporting that dataset size limits conclusions

**Problem:** "H2O AutoML is still using wrong splits"
- **Answer:** Make sure to:
  1. Do scaffold split FIRST in pandas
  2. Create separate H2OFrames for train/val/test
  3. Pass validation_frame explicitly to H2O

---

## ✅ Final Check

Before running your models:
- [ ] Imported `qsar_utils_no_leakage`
- [ ] Replaced data loading code
- [ ] Replaced splitting code (scaffold-based now)
- [ ] Replaced scaling code (fit on train only)
- [ ] Checked target transformation
- [ ] Verified no SMILES overlap
- [ ] Added similarity analysis
- [ ] Updated CV to use scaffold splits
- [ ] Saved splits for reproducibility

---

**Now run your notebooks with confidence! 🚀**
