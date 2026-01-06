# 🚀 Quick Start: Run Your Fixed QSAR Models

## Order of Execution

### Step 1: Run Verification for All Models (Required First!)

Execute notebooks in this order to verify data leakage fixes:

```bash
1. Model_1_circular_fingerprint_features_1024_H20_autoML_Model_Interpretation (1).ipynb
   → Run cells 1-20 (up to verification)
   → Confirm: "ALL CHECKS PASSED"
   
2. Model_2_ChEBERTa_embedding_linear_regression_no_interpretation (2).ipynb
   → Run cells 1-15 (up to verification)
   → Confirm: "ALL CHECKS PASSED"
   
3. Model_3_rdkit_features_H20_autoML.ipynb
   → Run cells 1-20 (up to verification)
   → Confirm: "ALL CHECKS PASSED"
   
4. Model_4_circular_fingerprint_features_1024_Gaussian_Process_Bayesian_Optimization_Model_Interpretation.ipynb
   → Run cells 1-20 (up to verification)
   → Confirm: "ALL CHECKS PASSED"
```

### Step 2: Check Verification Output

For each model, you should see:

```
🔍 VERIFICATION: Checking for Data Leakage
======================================================================

1. SMILES Overlap: 0 total
   ✅ PASS

🎉 ALL CHECKS PASSED - No data leakage detected!

   Train: XXX | Val: XXX | Test: XXX
======================================================================
```

---

## What You'll See After Running

### New Outputs Generated:

1. **Similarity Distribution Plots:**
   - `train_test_similarity.png` (Model 1)
   - `model2_similarity.png` (Model 2)
   - `model3_similarity.png` (Model 3)
   - `model4_similarity.png` (Model 4)

2. **Console Output:**
   - Scaffold statistics
   - Near-duplicate counts
   - Similarity analysis
   - Verification results

3. **New Variables Created:**
   - `train_df_clean` - Training set
   - `val_df_clean` - Validation set
   - `test_df_clean` - Test set
   - `df2` - Updated to training set only
   - `similarity_stats` - Similarity statistics dictionary

---

## Quick Verification Checklist

For EACH model, check:

### ✅ Console Output Shows:
- [ ] "✓ Canonicalized X SMILES"
- [ ] "✓ Final dataset: X unique molecules"
- [ ] "✓ Found X unique scaffolds"
- [ ] "📊 Scaffold-based split:" with train/val/test counts
- [ ] "📊 Train-Test Similarity Analysis:" with statistics
- [ ] "🎉 ALL CHECKS PASSED - No data leakage detected!"

### ✅ No Error Messages About:
- [ ] Import errors (qsar_utils_no_leakage)
- [ ] Missing columns
- [ ] Data type issues
- [ ] Overlap warnings

### ✅ Files Created:
- [ ] Similarity distribution plot saved

---

## Common Questions

### Q: Do I need to run all models?
**A:** Yes, verify all four to ensure consistency. They should all use the same cleaned data.

### Q: Can I skip the verification cells?
**A:** No! These are critical to ensure no data leakage.

### Q: What if one model fails verification?
**A:** Stop and fix it before proceeding. Check the error message and re-run from the beginning.

### Q: Should I save the cleaned splits?
**A:** Yes! Add this cell after verification in Model 1:

```python
# Save cleaned splits for consistency across models
train_df_clean.to_csv('train_set_scaffold_split.csv', index=False)
val_df_clean.to_csv('val_set_scaffold_split.csv', index=False)
test_df_clean.to_csv('test_set_scaffold_split.csv', index=False)
```

Then load these in Models 2, 3, 4 for complete consistency.

---

## Next Steps After Verification

### For Each Model:

1. **Update Feature Generation**
   - Generate features SEPARATELY for train/val/test
   - Use the appropriate method for each model type

2. **Update Feature Scaling**
   - FIT scaler on train only
   - TRANSFORM val and test

3. **Update Model Training**
   - Train on training set only
   - Validate on validation set
   - Evaluate on test set

4. **Update Cross-Validation** (if used)
   - Use scaffold-based CV
   - Replace `KFold(shuffle=True)` with `splitter.scaffold_kfold()`

---

## Expected Timeline

- **Verification (all 4 models):** 5-10 minutes
- **Feature generation updates:** 30-60 minutes per model
- **Model training updates:** 30-60 minutes per model
- **Full re-training:** Depends on model complexity

---

## Success Indicators

You'll know everything is working when:

1. ✅ All models pass verification
2. ✅ Zero SMILES overlap in all models
3. ✅ Similarity plots generated for all models
4. ✅ Training/val/test splits are consistent
5. ✅ Feature generation works with split data
6. ✅ Models train successfully on training set
7. ✅ Performance is reported on test set

---

## If Something Goes Wrong

### Error: "ModuleNotFoundError: No module named 'qsar_utils_no_leakage'"

**Solution:**
```python
# Add at the top of the notebook
import sys
sys.path.append('/Users/nb/Desktop/QSAR_Models')

# Then import
from qsar_utils_no_leakage import QSARDataProcessor, ScaffoldSplitter
```

### Error: "KeyError: 'Canonical SMILES'"

**Solution:** Check that your data has the column 'Canonical SMILES'. If not, update:
```python
processor = QSARDataProcessor(smiles_col='YOUR_COLUMN_NAME', target_col='IC50 uM')
```

### Error: Verification shows overlap

**Solution:** Re-run from the beginning. Make sure:
1. Duplicates are removed before splitting
2. No code modifications between split and verification
3. Variables haven't been accidentally overwritten

---

## 📊 What to Compare

### Before vs After Performance:

Create this comparison table after running all models:

| Model | Before R² | After R² | Change | Status |
|-------|-----------|----------|--------|--------|
| 1     | ?         | ?        | ?      | ✅ Fixed |
| 2     | ?         | ?        | ?      | ✅ Fixed |
| 3     | ?         | ?        | ?      | ✅ Fixed |
| 4     | ?         | ?        | ?      | ✅ Fixed |

---

## 🎯 Today's Goal

✅ Verify all four models pass data leakage checks  
✅ Generate similarity distribution plots  
✅ Document train/val/test split sizes  
✅ Confirm zero scaffold overlap across all models  

**Tomorrow's Goal:**
- Update feature generation for all models
- Update model training sections
- Re-train all models
- Document new performance metrics

---

## 📞 Need Help?

Check these files:
1. `README_DATA_LEAKAGE_FIX.md` - Full documentation
2. `QUICK_START_FIX.md` - Code examples
3. `ALL_MODELS_FIXED_SUMMARY.md` - Comprehensive summary
4. `DATA_LEAKAGE_FIX_EXAMPLE.ipynb` - Complete working example

---

**Ready to start? Run Model 1 first and verify the output!** 🚀
