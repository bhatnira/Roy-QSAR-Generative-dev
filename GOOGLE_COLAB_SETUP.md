# 🌐 Google Colab Setup Guide - QSAR Framework

## Quick Setup for Google Colab

Copy and paste these cells at the **beginning** of any notebook when running on Google Colab.

---

## 📋 Setup Cells for Google Colab

### Cell 1: Clone Repository and Install Dependencies

```python
# Cell 1: Setup - Clone repository and install dependencies
# Run this cell first when using Google Colab

import os
import sys

# Check if we're in Colab
try:
    import google.colab
    IN_COLAB = True
    print("✓ Running in Google Colab")
except:
    IN_COLAB = False
    print("✓ Running locally")

if IN_COLAB:
    print("\n" + "="*70)
    print("GOOGLE COLAB SETUP")
    print("="*70)
    
    # Clone the repository
    print("\n1️⃣ Cloning repository...")
    if not os.path.exists('Roy-QSAR-Generative-dev'):
        !git clone https://github.com/bhatnira/Roy-QSAR-Generative-dev.git
        print("✓ Repository cloned")
    else:
        print("✓ Repository already exists")
    
    # Change to repository directory
    os.chdir('Roy-QSAR-Generative-dev/notebooks')
    print(f"✓ Changed directory to: {os.getcwd()}")
    
    # Install dependencies
    print("\n2️⃣ Installing dependencies...")
    !pip install -q rdkit-pypi pandas numpy scikit-learn matplotlib seaborn
    print("✓ Core dependencies installed")
    
    # Optional: Install ML libraries
    print("\n3️⃣ Installing optional ML libraries...")
    !pip install -q xgboost lightgbm
    print("✓ XGBoost and LightGBM installed")
    
    print("\n" + "="*70)
    print("✅ SETUP COMPLETE - You can now run the notebook!")
    print("="*70 + "\n")
```

---

### Cell 2: Import Framework (Run After Setup)

```python
# Cell 2: Import QSAR Framework
# Run this cell after the setup cell

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add framework to path
current_dir = os.getcwd()
repo_root = os.path.abspath(os.path.join(current_dir, '..'))
framework_path = os.path.join(repo_root, 'src')

if framework_path not in sys.path:
    sys.path.insert(0, framework_path)
    print(f"✓ Framework loaded from: {framework_path}")

# Import core utilities
from utils.qsar_utils_no_leakage import QSARDataProcessor

# Import validation modules
from qsar_validation.splitting_strategies import AdvancedSplitter
from qsar_validation.feature_scaling import FeatureScaler
from qsar_validation.feature_selection import FeatureSelector
from qsar_validation.dataset_quality_analysis import DatasetQualityAnalyzer
from qsar_validation.performance_validation import PerformanceValidator
from qsar_validation.activity_cliffs_detection import ActivityCliffsDetector

print("✅ Framework modules loaded successfully!")
print("✅ Framework v4.1.0 - Multi-Library Support")
print("\n" + "="*70)
print("QSAR VALIDATION FRAMEWORK - MODULES LOADED:")
print("  ✓ QSARDataProcessor - Duplicate removal & data processing")
print("  ✓ AdvancedSplitter - Scaffold-based splitting")
print("  ✓ FeatureScaler - Proper feature scaling (fit on train only)")
print("  ✓ FeatureSelector - Feature selection to prevent overfitting")
print("  ✓ DatasetQualityAnalyzer - Dataset representativeness checks")
print("  ✓ PerformanceValidator - Cross-validation & metrics")
print("  ✓ ActivityCliffsDetector - Activity cliff analysis")
print("="*70)
```

---

## 🎯 Complete Colab Setup (All-in-One)

If you prefer a single cell that does everything:

```python
# ===== GOOGLE COLAB SETUP - ALL IN ONE =====
# Copy this entire cell to the top of your Colab notebook

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Check if we're in Colab
try:
    import google.colab
    IN_COLAB = True
except:
    IN_COLAB = False

if IN_COLAB:
    print("="*70)
    print("🌐 GOOGLE COLAB SETUP")
    print("="*70)
    
    # Step 1: Clone repository
    print("\n📦 Step 1: Cloning repository...")
    if not os.path.exists('Roy-QSAR-Generative-dev'):
        !git clone -q https://github.com/bhatnira/Roy-QSAR-Generative-dev.git
        print("✓ Repository cloned successfully")
    else:
        print("✓ Repository already exists")
    
    # Step 2: Change directory
    os.chdir('Roy-QSAR-Generative-dev/notebooks')
    print(f"✓ Working directory: {os.getcwd()}")
    
    # Step 3: Install dependencies
    print("\n📚 Step 2: Installing dependencies...")
    print("   (This may take 1-2 minutes...)")
    !pip install -q rdkit-pypi pandas numpy scikit-learn matplotlib seaborn xgboost lightgbm
    print("✓ All dependencies installed")
    
    # Step 4: Add framework to path
    repo_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
    framework_path = os.path.join(repo_root, 'src')
    if framework_path not in sys.path:
        sys.path.insert(0, framework_path)
    
    print("\n" + "="*70)
    print("✅ SETUP COMPLETE!")
    print("="*70)

# Import framework (works both locally and in Colab)
from utils.qsar_utils_no_leakage import QSARDataProcessor
from qsar_validation.splitting_strategies import AdvancedSplitter
from qsar_validation.feature_scaling import FeatureScaler
from qsar_validation.feature_selection import FeatureSelector
from qsar_validation.dataset_quality_analysis import DatasetQualityAnalyzer
from qsar_validation.performance_validation import PerformanceValidator
from qsar_validation.activity_cliffs_detection import ActivityCliffsDetector

print("\n✅ Framework v4.1.0 loaded successfully!")
print("\n📚 Available modules:")
print("  • QSARDataProcessor - Data cleaning & duplicate removal")
print("  • AdvancedSplitter - Scaffold/temporal/cluster splitting")
print("  • FeatureScaler - StandardScaler, MinMaxScaler, RobustScaler")
print("  • FeatureSelector - Feature selection methods")
print("  • DatasetQualityAnalyzer - Dataset quality assessment")
print("  • PerformanceValidator - Cross-validation & metrics")
print("  • ActivityCliffsDetector - Activity cliff detection")
print("\n🚀 Ready to start modeling!")
```

---

## 📊 For Each Specific Notebook

### For DATA_LEAKAGE_FIX_EXAMPLE.ipynb

```python
# Google Colab Setup for DATA_LEAKAGE_FIX_EXAMPLE.ipynb

!git clone https://github.com/bhatnira/Roy-QSAR-Generative-dev.git
%cd Roy-QSAR-Generative-dev/notebooks

# Install dependencies
!pip install -q rdkit-pypi pandas numpy scikit-learn matplotlib seaborn

# The notebook will auto-detect the framework path from here
print("✅ Setup complete! Now run the next cell (imports)")
```

---

### For Model 1 (Circular Fingerprints + H2O AutoML)

```python
# Google Colab Setup for Model 1

!git clone https://github.com/bhatnira/Roy-QSAR-Generative-dev.git
%cd Roy-QSAR-Generative-dev/notebooks

# Install dependencies
!pip install -q rdkit-pypi pandas numpy scikit-learn matplotlib seaborn
!pip install -q h2o  # For H2O AutoML

print("✅ Setup complete! Now run the import cell")
```

---

### For Model 2 (ChEBERTa Embeddings)

```python
# Google Colab Setup for Model 2 (ChEBERTa)

!git clone https://github.com/bhatnira/Roy-QSAR-Generative-dev.git
%cd Roy-QSAR-Generative-dev/notebooks

# Install dependencies
!pip install -q rdkit-pypi pandas numpy scikit-learn matplotlib seaborn
!pip install -q transformers torch  # For ChEBERTa

print("✅ Setup complete! Now run the import cell")
```

---

### For Model 3 (RDKit Features + H2O AutoML)

```python
# Google Colab Setup for Model 3

!git clone https://github.com/bhatnira/Roy-QSAR-Generative-dev.git
%cd Roy-QSAR-Generative-dev/notebooks

# Install dependencies
!pip install -q rdkit-pypi pandas numpy scikit-learn matplotlib seaborn
!pip install -q h2o  # For H2O AutoML

print("✅ Setup complete! Now run the import cell")
```

---

### For Model 4 (Gaussian Process + Bayesian Optimization)

```python
# Google Colab Setup for Model 4

!git clone https://github.com/bhatnira/Roy-QSAR-Generative-dev.git
%cd Roy-QSAR-Generative-dev/notebooks

# Install dependencies
!pip install -q rdkit-pypi pandas numpy scikit-learn matplotlib seaborn
!pip install -q scikit-optimize  # For Bayesian Optimization

print("✅ Setup complete! Now run the import cell")
```

---

## 🔧 Troubleshooting on Colab

### Issue 1: RDKit Import Error

```python
# If you get "No module named 'rdkit'", run this:
!pip install -q rdkit-pypi
# Then restart runtime: Runtime → Restart runtime
```

### Issue 2: Framework Not Found

```python
# Check current directory and framework path
import os
print("Current directory:", os.getcwd())
print("Framework path:", os.path.join(os.getcwd(), '..', 'src'))

# List files to verify structure
!ls -la ../src/
```

### Issue 3: Git Clone Fails

```python
# If clone fails, try with specific branch
!git clone -b main https://github.com/bhatnira/Roy-QSAR-Generative-dev.git

# Or update existing repo
%cd Roy-QSAR-Generative-dev
!git pull origin main
%cd notebooks
```

---

## 📱 Quick Reference Card

**Copy-paste this at the start of ANY notebook:**

```python
# 🚀 QUICK COLAB SETUP
try:
    import google.colab
    !git clone -q https://github.com/bhatnira/Roy-QSAR-Generative-dev.git 2>/dev/null || echo "Repo exists"
    %cd Roy-QSAR-Generative-dev/notebooks
    !pip install -q rdkit-pypi pandas numpy scikit-learn matplotlib seaborn xgboost
    print("✅ Colab setup complete!")
except:
    print("✅ Running locally")
```

---

## 🎓 Step-by-Step Instructions for Complete Beginners

### Step 1: Open Google Colab
1. Go to https://colab.research.google.com/
2. Click "New Notebook"

### Step 2: Add Setup Cell
1. In the first cell, paste this code:
```python
!git clone https://github.com/bhatnira/Roy-QSAR-Generative-dev.git
%cd Roy-QSAR-Generative-dev/notebooks
!pip install -q rdkit-pypi pandas numpy scikit-learn matplotlib seaborn
```

### Step 3: Run Setup Cell
1. Click the "Play" button (▶) or press Shift+Enter
2. Wait for it to complete (30-60 seconds)

### Step 4: Add Import Cell
1. In the next cell, paste the import code from above
2. Run it (Shift+Enter)

### Step 5: Continue with Your Analysis
1. Now you can run the rest of the notebook cells normally!

---

## 💡 Pro Tips

### Tip 1: Faster Installation
```python
# Use --no-deps for faster installation if you know dependencies are met
!pip install -q --no-deps rdkit-pypi
```

### Tip 2: Mount Google Drive (For Your Data)
```python
from google.colab import drive
drive.mount('/content/drive')

# Now you can access your data from Drive
import pandas as pd
df = pd.read_csv('/content/drive/MyDrive/your_data.csv')
```

### Tip 3: Save Results to Drive
```python
# Save your results
results.to_csv('/content/drive/MyDrive/qsar_results.csv', index=False)
print("✅ Results saved to Google Drive")
```

### Tip 4: Check GPU Availability (If Needed)
```python
# Check if GPU is available
import torch
print("GPU available:", torch.cuda.is_available())

# To enable GPU: Runtime → Change runtime type → GPU
```

---

## 📋 Complete Example: Full Colab Notebook Setup

Here's a complete example showing the entire flow:

```python
# ==========================================
# CELL 1: COLAB SETUP
# ==========================================
import os

# Clone and setup
!git clone https://github.com/bhatnira/Roy-QSAR-Generative-dev.git
%cd Roy-QSAR-Generative-dev/notebooks
!pip install -q rdkit-pypi pandas numpy scikit-learn matplotlib seaborn xgboost

print("✅ Setup complete!")

# ==========================================
# CELL 2: MOUNT GOOGLE DRIVE (OPTIONAL)
# ==========================================
from google.colab import drive
drive.mount('/content/drive')

# ==========================================
# CELL 3: IMPORT FRAMEWORK
# ==========================================
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add framework to path
sys.path.insert(0, '../src')

from utils.qsar_utils_no_leakage import QSARDataProcessor
from qsar_validation.splitting_strategies import AdvancedSplitter
from qsar_validation.feature_scaling import FeatureScaler
from qsar_validation.dataset_quality_analysis import DatasetQualityAnalyzer

print("✅ Framework loaded!")

# ==========================================
# CELL 4: LOAD YOUR DATA
# ==========================================
import pandas as pd

# Option 1: Load from Drive
# df = pd.read_csv('/content/drive/MyDrive/your_data.csv')

# Option 2: Load from URL
# df = pd.read_csv('https://your-url.com/data.csv')

# Option 3: Use sample data
df = pd.DataFrame({
    'SMILES': ['CCO', 'CCOCC', 'c1ccccc1'],
    'Activity': [5.2, 6.1, 7.3]
})

print(f"✅ Data loaded: {len(df)} molecules")

# ==========================================
# CELL 5: YOUR ANALYSIS STARTS HERE
# ==========================================
# Now use the framework as normal...
processor = QSARDataProcessor(smiles_col='SMILES')
df_clean = processor.remove_duplicates(df)
# ... continue with your analysis
```

---

## 🔗 Useful Colab Links

- **Official Colab:** https://colab.research.google.com/
- **Colab Tips:** https://colab.research.google.com/notebooks/basic_features_overview.ipynb
- **GitHub Integration:** https://colab.research.google.com/github/

---

## ✅ Quick Checklist

Before running your notebook on Colab:

- [ ] Added setup cell at the beginning
- [ ] Installed all required dependencies
- [ ] Changed to correct directory
- [ ] Imported framework successfully
- [ ] Can access your data (Drive/URL)
- [ ] Framework modules load without errors

---

**That's it! You're ready to use the QSAR framework on Google Colab! 🚀**
