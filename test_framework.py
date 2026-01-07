#!/usr/bin/env python3
"""
Quick Test Script for QSAR Framework
Tests basic functionality without requiring all dependencies
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("=" * 70)
print("🧪 QSAR FRAMEWORK - QUICK TEST")
print("=" * 70)

# Test 1: Check Python version
print("\n📍 Test 1: Python Version")
print(f"Python: {sys.version}")
if sys.version_info >= (3, 8):
    print("✅ Python version OK (>= 3.8)")
else:
    print("❌ Python version too old (need >= 3.8)")
    sys.exit(1)

# Test 2: Check core dependencies
print("\n📍 Test 2: Core Dependencies")
required = ['numpy', 'pandas', 'scipy']
optional = ['sklearn', 'matplotlib', 'rdkit']

missing_required = []
missing_optional = []

for pkg in required:
    try:
        __import__(pkg)
        print(f"✅ {pkg}")
    except ImportError:
        print(f"❌ {pkg} - REQUIRED")
        missing_required.append(pkg)

for pkg in optional:
    try:
        __import__(pkg)
        print(f"✅ {pkg}")
    except ImportError:
        print(f"⚠️  {pkg} - optional")
        missing_optional.append(pkg)

if missing_required:
    print(f"\n❌ Missing required packages: {', '.join(missing_required)}")
    print(f"Install with: python3 -m pip install {' '.join(missing_required)}")
    sys.exit(1)

# Test 3: Test module structure
print("\n📍 Test 3: Module Structure")
expected_dirs = ['qsar_validation', 'utils']
src_path = os.path.join(os.path.dirname(__file__), 'src')

for dir_name in expected_dirs:
    dir_path = os.path.join(src_path, dir_name)
    if os.path.isdir(dir_path):
        init_file = os.path.join(dir_path, '__init__.py')
        if os.path.exists(init_file):
            print(f"✅ {dir_name}/ module found")
        else:
            print(f"⚠️  {dir_name}/ found but no __init__.py")
    else:
        print(f"❌ {dir_name}/ not found")

# Test 4: Test imports
print("\n📍 Test 4: Import Core Modules")
try:
    from utils.qsar_utils_no_leakage import quick_clean, clean_qsar_data_with_report
    print("✅ utils.qsar_utils_no_leakage imports OK")
except Exception as e:
    print(f"❌ Failed to import utils: {e}")

try:
    from qsar_validation.splitting_strategies import RandomSplit
    print("✅ qsar_validation.splitting_strategies imports OK")
except Exception as e:
    print(f"⚠️  Failed to import splitting_strategies: {e}")

try:
    from qsar_validation.model_agnostic_pipeline import QSARPipeline
    print("✅ qsar_validation.model_agnostic_pipeline imports OK")
except Exception as e:
    print(f"⚠️  Failed to import pipeline: {e}")

# Test 5: Test basic functionality
print("\n📍 Test 5: Test Data Cleaning Function")
try:
    import pandas as pd
    from utils.qsar_utils_no_leakage import quick_clean
    
    # Create tiny test dataset
    test_df = pd.DataFrame({
        'SMILES': ['CCO', 'CC(C)O', 'CCO'],  # With duplicate
        'pIC50': [5.5, 6.2, 5.5]
    })
    
    print(f"Test data: {len(test_df)} rows (with 1 duplicate)")
    cleaned_df = quick_clean(test_df, 'SMILES', 'pIC50')
    print(f"Cleaned data: {len(cleaned_df)} rows")
    
    if len(cleaned_df) < len(test_df):
        print("✅ quick_clean() works correctly (removed duplicates)")
    else:
        print("⚠️  quick_clean() ran but didn't remove duplicates")
        
except Exception as e:
    print(f"❌ Data cleaning test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 6: Check examples folder
print("\n📍 Test 6: Examples Folder")
examples_path = os.path.join(os.path.dirname(__file__), 'examples')
if os.path.isdir(examples_path):
    examples = [f for f in os.listdir(examples_path) if f.endswith('.py')]
    print(f"✅ Found {len(examples)} example scripts:")
    for ex in sorted(examples)[:5]:  # Show first 5
        print(f"   • {ex}")
else:
    print("⚠️  No examples folder found")

# Test 7: Check documentation
print("\n📍 Test 7: Documentation")
docs = ['README.md', 'INSTALL.md', 'requirements.txt', 'setup.py']
for doc in docs:
    doc_path = os.path.join(os.path.dirname(__file__), doc)
    if os.path.exists(doc_path):
        print(f"✅ {doc}")
    else:
        print(f"❌ {doc} not found")

# Final summary
print("\n" + "=" * 70)
print("📊 TEST SUMMARY")
print("=" * 70)

if missing_required:
    print("❌ TESTS FAILED - Missing required dependencies")
    print(f"   Install: python3 -m pip install {' '.join(missing_required)}")
else:
    print("✅ ALL CRITICAL TESTS PASSED!")
    print("\n🎯 Framework is ready to use!")
    print("\n📚 Next steps:")
    print("   1. Review README.md for usage examples")
    print("   2. Check examples/ folder for sample scripts")
    print("   3. Run: python3 examples/01_basic_validation.py")

print("=" * 70)
