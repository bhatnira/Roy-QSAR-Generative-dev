"""
Validation Runner Module
=========================

Orchestrates comprehensive QSAR validation workflow.
"""

import pandas as pd
from typing import Dict

from .dataset_analysis import DatasetBiasAnalyzer
from .activity_cliffs import ActivityCliffDetector
from .assay_noise import AssayNoiseEstimator


def run_comprehensive_validation(df: pd.DataFrame, 
                                 smiles_col: str = 'Canonical SMILES',
                                 target_col: str = 'IC50 uM') -> Dict:
    """
    Run all validation checks in sequence.
    
    Args:
        df: DataFrame with SMILES and target columns
        smiles_col: Name of SMILES column (default: 'Canonical SMILES')
        target_col: Name of target column (default: 'IC50 uM')
        
    Returns:
        Dictionary with all validation results
    """
    results = {}
    
    # 1. Dataset bias analysis
    bias_analyzer = DatasetBiasAnalyzer(smiles_col, target_col)
    results['scaffold_diversity'] = bias_analyzer.analyze_scaffold_diversity(df)
    results['activity_distribution'] = bias_analyzer.analyze_activity_distribution(df)
    
    # 2. Activity cliffs
    cliff_detector = ActivityCliffDetector(smiles_col, target_col)
    results['activity_cliffs'] = cliff_detector.detect_activity_cliffs(df)
    
    # 3. Assay noise
    noise_estimator = AssayNoiseEstimator()
    results['experimental_error'] = noise_estimator.estimate_experimental_error(df, target_col)
    
    print("\n" + "=" * 70)
    print("✓ Comprehensive validation complete")
    print("=" * 70)
    
    return results


def print_comprehensive_validation_checklist():
    """
    Print comprehensive checklist for QSAR model validation.
    """
    checklist = """
    ╔═══════════════════════════════════════════════════════════════════════╗
    ║     COMPREHENSIVE QSAR VALIDATION CHECKLIST (Low-Data Regime)        ║
    ╚═══════════════════════════════════════════════════════════════════════╝
    
    🔴 CRITICAL (Must Fix)
    ─────────────────────────────────────────────────────────────────────────
    ☐ 1. Data Leakage Prevention
       • Scaffold-based splitting (not random)
       • Duplicates removed BEFORE splitting
       • Features scaled using training data only
       • No feature selection on full dataset
    
    ☐ 2. Dataset Bias & Representativeness
       • Scaffold diversity analyzed
       • Scaffold counts reported per split
       • Chemical space limitations acknowledged
       • Congeneric series identified
    
    ☐ 3. Model Complexity Control
       • Samples-to-features ratio ≥ 5 (preferably ≥ 10)
       • Regularization applied appropriately
       • Hyperparameter ranges restricted
       • Nested CV for hyperparameter tuning
    
    ☐ 4. Proper Cross-Validation
       • Scaffold-based CV (not random)
       • Report mean ± std across folds
       • No feature selection in CV loop
       • Proper pipeline in each fold
    
    🟠 HIGH PRIORITY (Strongly Recommended)
    ─────────────────────────────────────────────────────────────────────────
    ☐ 5. Assay Noise Consideration
       • Experimental error estimated/reported
       • RMSE compared to assay precision
       • Mixed assay types identified
       • Suspicious RMSE < 0.3 log units flagged
    
    ☐ 6. Activity Cliffs
       • Activity cliffs detected and reported
       • SAR discontinuities acknowledged
       • Local models considered if many cliffs
       • Feature importance interpreted cautiously
    
    ☐ 7. Proper Metrics & Baselines
       • RMSE, MAE, R², Spearman ρ all reported
       • Baseline model (Ridge) compared
       • External test set evaluated
       • Not just R² (can be misleading)
    
    ☐ 8. Y-Randomization Test
       • Y-scrambling performed (10+ iterations)
       • R² should be ≤ 0 with random targets
       • Results reported in supplementary
    
    🟡 MODERATE PRIORITY (Best Practices)
    ─────────────────────────────────────────────────────────────────────────
    ☐ 9. Target Distribution Analysis
       • Activity range reported
       • Distribution visualized
       • Narrow ranges acknowledged
       • Outliers investigated
    
    ☐ 10. Uncertainty Estimation
       • Prediction intervals provided (GPR/ensemble)
       • Applicability domain defined
       • Out-of-domain predictions flagged
    
    ☐ 11. Interpretability
       • Mechanistic claims avoided
       • SHAP/feature importance = hypothesis generation only
       • Correlation vs causation clear
       • Validated against known SAR
    
    ☐ 12. Reproducibility
       • All random seeds fixed
       • Code/data publicly available
       • Preprocessing steps documented
       • Software versions recorded
    
    ☐ 13. Honest Reporting
       • Applicability domain stated clearly
       • Limitations acknowledged
       • Performance drop with scaffold split noted
       • No cherry-picking of metrics
    
    ═══════════════════════════════════════════════════════════════════════════
    
    📊 EXPECTED PERFORMANCE CHANGES
    ─────────────────────────────────────────────────────────────────────────
    When fixing data leakage issues, expect:
    
    • R² drop: 0.80 → 0.60 (or lower)  ✓ This is NORMAL and CORRECT
    • RMSE increase: 0.3 → 0.5        ✓ More realistic
    • Scaffold split harder than random ✓ Tests generalization
    
    If performance stays very high after fixes → still may be issues
    
    ═══════════════════════════════════════════════════════════════════════════
    
    📌 KEY TAKEAWAYS FOR LOW-DATA QSAR
    ─────────────────────────────────────────────────────────────────────────
    1. Simpler models often outperform complex ones (n < 200)
    2. Scaffold split is mandatory for honest evaluation
    3. RMSE ≈ 0.5 log units is near theoretical limit for IC₅₀
    4. R² alone is misleading with narrow activity ranges
    5. Y-randomization test catches overfitting
    6. Activity cliffs limit local predictivity
    7. Report limitations honestly → better reviews
    
    ═══════════════════════════════════════════════════════════════════════════
    """
    print(checklist)


if __name__ == "__main__":
    print_comprehensive_validation_checklist()
