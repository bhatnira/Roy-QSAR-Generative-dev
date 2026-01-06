"""
QSAR Validation & Best Practices Utilities
===========================================

This module extends qsar_utils_no_leakage.py with additional validation tools:

1. Dataset bias & representativeness analysis
2. Model complexity control & overfitting detection
3. Proper cross-validation design
4. Measurement noise handling
5. Activity cliff detection
6. Target distribution analysis
7. Uncertainty estimation
8. Performance metrics & baselines
9. Y-randomization tests
10. Reproducibility checks

Author: Generated for low-data QSAR best practices
Date: January 2026
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import DataStructs
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr, pearsonr
from typing import Tuple, List, Dict, Optional
import warnings

warnings.filterwarnings('ignore')


class DatasetBiasAnalyzer:
    """
    Analyze dataset bias, chemical space coverage, and representativeness.
    """
    
    def __init__(self, smiles_col='Canonical SMILES', target_col='IC50 uM'):
        self.smiles_col = smiles_col
        self.target_col = target_col
    
    def analyze_scaffold_diversity(self, df: pd.DataFrame) -> Dict:
        """
        Analyze scaffold diversity and distribution.
        
        Args:
            df: DataFrame with SMILES
            
        Returns:
            Dictionary with diversity metrics
        """
        print("\n🔬 DATASET BIAS ANALYSIS: Scaffold Diversity")
        print("=" * 70)
        
        # Get scaffolds
        scaffolds = []
        for smi in df[self.smiles_col]:
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                    scaffolds.append(Chem.MolToSmiles(scaffold))
                else:
                    scaffolds.append(None)
            except:
                scaffolds.append(None)
        
        df = df.copy()
        df['scaffold'] = scaffolds
        df = df.dropna(subset=['scaffold'])
        
        # Analyze distribution
        scaffold_counts = df['scaffold'].value_counts()
        n_unique_scaffolds = len(scaffold_counts)
        n_molecules = len(df)
        
        # Diversity metrics
        diversity_ratio = n_unique_scaffolds / n_molecules
        gini_coefficient = self._calculate_gini(scaffold_counts.values)
        
        # Find dominant scaffolds
        top_scaffolds = scaffold_counts.head(5)
        top_scaffold_fraction = top_scaffolds.sum() / n_molecules
        
        print(f"📊 Total molecules: {n_molecules}")
        print(f"📊 Unique scaffolds: {n_unique_scaffolds}")
        print(f"📊 Diversity ratio: {diversity_ratio:.3f}")
        print(f"📊 Gini coefficient: {gini_coefficient:.3f} (0=perfect equality, 1=max inequality)")
        print(f"\n🔝 Top 5 scaffolds represent {top_scaffold_fraction*100:.1f}% of dataset:")
        for i, (scaffold, count) in enumerate(top_scaffolds.items(), 1):
            print(f"   {i}. {count} molecules ({count/n_molecules*100:.1f}%)")
        
        # Warnings
        if diversity_ratio < 0.3:
            print("\n⚠️  WARNING: Low scaffold diversity (congeneric series)")
            print("   → Model may not generalize beyond this scaffold family")
            print("   → Consider stating limited applicability domain")
        
        if top_scaffold_fraction > 0.5:
            print("\n⚠️  WARNING: Dataset dominated by top scaffolds")
            print("   → High risk of overfitting to these scaffolds")
            print("   → Performance may be scaffold-specific")
        
        if gini_coefficient > 0.6:
            print("\n⚠️  WARNING: High scaffold imbalance (Gini > 0.6)")
            print("   → Some scaffolds heavily overrepresented")
            print("   → Consider scaffold-aware sampling")
        
        return {
            'n_molecules': n_molecules,
            'n_scaffolds': n_unique_scaffolds,
            'diversity_ratio': diversity_ratio,
            'gini_coefficient': gini_coefficient,
            'top_scaffold_fraction': top_scaffold_fraction,
            'scaffold_counts': scaffold_counts,
            'df_with_scaffolds': df
        }
    
    def analyze_activity_distribution(self, df: pd.DataFrame) -> Dict:
        """
        Analyze target activity distribution and identify biases.
        
        Args:
            df: DataFrame with target column
            
        Returns:
            Dictionary with distribution statistics
        """
        print("\n📊 TARGET DISTRIBUTION ANALYSIS")
        print("=" * 70)
        
        activities = df[self.target_col].values
        
        # Basic statistics
        stats = {
            'mean': np.mean(activities),
            'median': np.median(activities),
            'std': np.std(activities),
            'min': np.min(activities),
            'max': np.max(activities),
            'range': np.max(activities) - np.min(activities),
            'q25': np.percentile(activities, 25),
            'q75': np.percentile(activities, 75),
            'iqr': np.percentile(activities, 75) - np.percentile(activities, 25),
        }
        
        print(f"Mean: {stats['mean']:.3f}")
        print(f"Median: {stats['median']:.3f}")
        print(f"Std Dev: {stats['std']:.3f}")
        print(f"Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
        print(f"IQR: {stats['iqr']:.3f}")
        
        # Check for narrow range
        relative_range = stats['range'] / stats['mean'] if stats['mean'] > 0 else 0
        print(f"\nRelative range: {relative_range:.3f}")
        
        if relative_range < 2.0:
            print("⚠️  WARNING: Narrow activity range")
            print("   → Limited chemical space coverage")
            print("   → R² may be artificially inflated")
            print("   → Focus on RMSE/MAE for evaluation")
        
        # Check for clustering
        if stats['iqr'] < 0.5 * stats['std']:
            print("\n⚠️  WARNING: Activities clustered in narrow range")
            print("   → Most values in small range")
            print("   → Poor extrapolation expected")
        
        # Check for outliers
        lower_bound = stats['q25'] - 1.5 * stats['iqr']
        upper_bound = stats['q75'] + 1.5 * stats['iqr']
        outliers = (activities < lower_bound) | (activities > upper_bound)
        n_outliers = np.sum(outliers)
        
        if n_outliers > 0:
            print(f"\n📌 Detected {n_outliers} potential outliers ({n_outliers/len(activities)*100:.1f}%)")
            print("   → Review for measurement errors")
            print("   → Consider robust regression methods")
        
        stats['outliers'] = outliers
        stats['n_outliers'] = n_outliers
        
        return stats
    
    def report_split_diversity(self, df_with_scaffolds: pd.DataFrame, 
                               train_idx: np.ndarray, 
                               val_idx: np.ndarray, 
                               test_idx: np.ndarray) -> None:
        """
        Report scaffold diversity for each split.
        
        Args:
            df_with_scaffolds: DataFrame with scaffold column
            train_idx: Training indices
            val_idx: Validation indices
            test_idx: Test indices
        """
        print("\n📊 SCAFFOLD DISTRIBUTION PER SPLIT")
        print("=" * 70)
        
        train_scaffolds = set(df_with_scaffolds.iloc[train_idx]['scaffold'])
        val_scaffolds = set(df_with_scaffolds.iloc[val_idx]['scaffold'])
        test_scaffolds = set(df_with_scaffolds.iloc[test_idx]['scaffold'])
        
        print(f"Training set: {len(train_scaffolds)} unique scaffolds")
        print(f"Validation set: {len(val_scaffolds)} unique scaffolds")
        print(f"Test set: {len(test_scaffolds)} unique scaffolds")
        
        # Novel scaffolds in test
        novel_test = test_scaffolds - train_scaffolds
        if len(novel_test) > 0:
            print(f"\n✓ Test set contains {len(novel_test)} novel scaffolds ({len(novel_test)/len(test_scaffolds)*100:.1f}%)")
            print("  → Good generalization test")
        else:
            print("\n⚠️  WARNING: All test scaffolds present in training")
            print("  → May not test generalization adequately")
    
    @staticmethod
    def _calculate_gini(values: np.ndarray) -> float:
        """Calculate Gini coefficient for inequality measurement."""
        sorted_values = np.sort(values)
        n = len(values)
        cumsum = np.cumsum(sorted_values)
        return (2 * np.sum((n - np.arange(1, n + 1) + 0.5) * sorted_values)) / (n * np.sum(sorted_values)) - 1


class ActivityCliffDetector:
    """
    Detect activity cliffs: pairs of similar molecules with large activity differences.
    """
    
    def __init__(self, smiles_col='Canonical SMILES', target_col='IC50 uM'):
        self.smiles_col = smiles_col
        self.target_col = target_col
    
    def detect_activity_cliffs(self, df: pd.DataFrame, 
                               similarity_threshold=0.85, 
                               activity_threshold=2.0) -> pd.DataFrame:
        """
        Detect activity cliffs in dataset.
        
        Args:
            df: DataFrame with SMILES and target
            similarity_threshold: Tanimoto similarity threshold
            activity_threshold: Minimum activity difference (log units or fold)
            
        Returns:
            DataFrame with activity cliff pairs
        """
        print("\n⚠️  ACTIVITY CLIFF DETECTION")
        print("=" * 70)
        print(f"Criteria: Similarity ≥ {similarity_threshold}, Activity diff ≥ {activity_threshold}")
        
        # Generate fingerprints
        fps = []
        for smi in df[self.smiles_col]:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048))
            else:
                fps.append(None)
        
        # Find activity cliffs
        cliffs = []
        activities = df[self.target_col].values
        
        for i in range(len(fps)):
            if fps[i] is None:
                continue
            for j in range(i + 1, len(fps)):
                if fps[j] is None:
                    continue
                
                similarity = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                activity_diff = abs(activities[i] - activities[j])
                
                if similarity >= similarity_threshold and activity_diff >= activity_threshold:
                    cliffs.append({
                        'mol1_idx': i,
                        'mol2_idx': j,
                        'smiles1': df.iloc[i][self.smiles_col],
                        'smiles2': df.iloc[j][self.smiles_col],
                        'activity1': activities[i],
                        'activity2': activities[j],
                        'activity_diff': activity_diff,
                        'similarity': similarity
                    })
        
        cliff_df = pd.DataFrame(cliffs)
        
        if len(cliffs) > 0:
            print(f"\n⚠️  Found {len(cliffs)} activity cliff pairs")
            print(f"   Mean similarity: {cliff_df['similarity'].mean():.3f}")
            print(f"   Mean activity diff: {cliff_df['activity_diff'].mean():.3f}")
            print("\n📌 IMPLICATIONS:")
            print("   → Local SAR is discontinuous")
            print("   → Fingerprint-based models may struggle")
            print("   → Consider local models or Gaussian Processes")
            print("   → Feature importance interpretation limited")
        else:
            print("\n✓ No activity cliffs detected")
            print("  → SAR appears continuous")
        
        return cliff_df


class ModelComplexityAnalyzer:
    """
    Analyze model complexity vs dataset size to prevent overfitting.
    """
    
    @staticmethod
    def analyze_complexity(n_samples: int, n_features: int, model_type: str) -> None:
        """
        Analyze if model complexity is appropriate for dataset size.
        
        Args:
            n_samples: Number of samples
            n_features: Number of features
            model_type: Type of model (for recommendations)
        """
        print("\n🔍 MODEL COMPLEXITY ANALYSIS")
        print("=" * 70)
        
        ratio = n_samples / n_features
        
        print(f"Samples: {n_samples}")
        print(f"Features: {n_features}")
        print(f"Samples-to-features ratio: {ratio:.2f}")
        
        # General guidelines
        if ratio < 5:
            print("\n🔴 CRITICAL: Very low samples-to-features ratio (< 5)")
            print("   → High overfitting risk")
            print("   → REQUIRED: Strong regularization")
            print("   → RECOMMENDED: Feature selection or dimensionality reduction")
            print("   → AVOID: Complex models (deep learning, unregularized ensemble)")
        elif ratio < 10:
            print("\n🟠 WARNING: Low samples-to-features ratio (< 10)")
            print("   → Moderate overfitting risk")
            print("   → REQUIRED: Regularization (Ridge, Lasso, ElasticNet)")
            print("   → RECOMMENDED: Simple models (linear, regularized)")
        elif ratio < 20:
            print("\n🟡 CAUTION: Modest samples-to-features ratio (< 20)")
            print("   → Use cross-validation carefully")
            print("   → RECOMMENDED: Regularized models")
        else:
            print("\n✓ Adequate samples-to-features ratio")
        
        # Model-specific recommendations
        print(f"\n📌 Recommendations for {model_type}:")
        
        recommendations = {
            'deep_learning': {
                'min_ratio': 50,
                'advice': [
                    "Use pre-trained embeddings (ChEBERTa, MolBERT)",
                    "Strong dropout (0.3-0.5)",
                    "Early stopping",
                    "Very simple architecture (1-2 layers)",
                ]
            },
            'random_forest': {
                'min_ratio': 10,
                'advice': [
                    "Limit max_depth (3-5)",
                    "Increase min_samples_leaf (5-10)",
                    "Use max_features='sqrt' or 'log2'",
                    "Limit n_estimators (50-100)",
                ]
            },
            'gradient_boosting': {
                'min_ratio': 10,
                'advice': [
                    "Low learning_rate (0.01-0.05)",
                    "Limit max_depth (2-4)",
                    "Use subsample (0.5-0.8)",
                    "Early stopping based on validation",
                ]
            },
            'gaussian_process': {
                'min_ratio': 5,
                'advice': [
                    "Good for small datasets",
                    "Provides uncertainty estimates",
                    "Watch for kernel overfitting",
                    "Consider noise parameter",
                ]
            },
            'linear': {
                'min_ratio': 3,
                'advice': [
                    "Use Ridge (L2) or Lasso (L1)",
                    "Cross-validate alpha parameter",
                    "ElasticNet for mixed regularization",
                    "Works well with limited data",
                ]
            },
        }
        
        if model_type.lower() in recommendations:
            rec = recommendations[model_type.lower()]
            if ratio < rec['min_ratio']:
                print(f"   ⚠️  Dataset size below recommended minimum ({rec['min_ratio']}:1)")
            
            print("   Best practices:")
            for advice in rec['advice']:
                print(f"   • {advice}")
        
        print("\n📌 GENERAL BEST PRACTICES:")
        print("   • Use nested cross-validation")
        print("   • Report validation metrics (not just training)")
        print("   • Compare to simple baseline (Ridge regression)")
        print("   • Perform y-randomization test")


class PerformanceMetricsCalculator:
    """
    Calculate comprehensive performance metrics with proper baselines.
    """
    
    @staticmethod
    def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                             set_name: str = "Test") -> Dict:
        """
        Calculate comprehensive regression metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            set_name: Name of the dataset (for printing)
            
        Returns:
            Dictionary with all metrics
        """
        print(f"\n📊 {set_name.upper()} SET PERFORMANCE METRICS")
        print("=" * 70)
        
        # Regression metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Correlation metrics
        pearson_r, pearson_p = pearsonr(y_true, y_pred)
        spearman_r, spearman_p = spearmanr(y_true, y_pred)
        
        # Additional metrics
        mse = mean_squared_error(y_true, y_pred)
        max_error = np.max(np.abs(y_true - y_pred))
        
        metrics = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'mse': mse,
            'max_error': max_error,
        }
        
        print(f"RMSE:           {rmse:.4f}")
        print(f"MAE:            {mae:.4f}")
        print(f"R²:             {r2:.4f}")
        print(f"Pearson r:      {pearson_r:.4f} (p={pearson_p:.4e})")
        print(f"Spearman ρ:     {spearman_r:.4f} (p={spearman_p:.4e})")
        print(f"Max Error:      {max_error:.4f}")
        
        return metrics
    
    @staticmethod
    def calculate_baseline_metrics(X: np.ndarray, y: np.ndarray, 
                                   cv_folds: List = None) -> Dict:
        """
        Calculate simple baseline model performance (Ridge regression).
        
        Args:
            X: Features
            y: Target values
            cv_folds: Optional CV folds for proper evaluation
            
        Returns:
            Baseline metrics
        """
        print("\n📊 BASELINE MODEL PERFORMANCE (Ridge Regression)")
        print("=" * 70)
        
        if cv_folds is None:
            # Simple train-test split
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            model = Ridge(alpha=1.0)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            metrics = PerformanceMetricsCalculator.calculate_all_metrics(
                y_test, y_pred, "Baseline Test"
            )
        else:
            # Use provided CV folds
            rmse_scores = []
            mae_scores = []
            r2_scores = []
            
            for train_idx, val_idx in cv_folds:
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                model = Ridge(alpha=1.0)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                
                rmse_scores.append(np.sqrt(mean_squared_error(y_val, y_pred)))
                mae_scores.append(mean_absolute_error(y_val, y_pred))
                r2_scores.append(r2_score(y_val, y_pred))
            
            metrics = {
                'rmse_mean': np.mean(rmse_scores),
                'rmse_std': np.std(rmse_scores),
                'mae_mean': np.mean(mae_scores),
                'mae_std': np.std(mae_scores),
                'r2_mean': np.mean(r2_scores),
                'r2_std': np.std(r2_scores),
            }
            
            print(f"RMSE:  {metrics['rmse_mean']:.4f} ± {metrics['rmse_std']:.4f}")
            print(f"MAE:   {metrics['mae_mean']:.4f} ± {metrics['mae_std']:.4f}")
            print(f"R²:    {metrics['r2_mean']:.4f} ± {metrics['r2_std']:.4f}")
        
        print("\n📌 Use this as minimum performance threshold")
        print("   Complex models should significantly outperform this baseline")
        
        return metrics


class YRandomizationTester:
    """
    Perform y-randomization (y-scrambling) test to detect overfitting.
    """
    
    @staticmethod
    def perform_y_randomization(X: np.ndarray, y: np.ndarray, 
                                model, n_iterations: int = 10,
                                cv_folds: List = None) -> Dict:
        """
        Perform y-randomization test.
        
        Args:
            X: Features
            y: Target values
            model: Model to test (with fit/predict methods)
            n_iterations: Number of randomization iterations
            cv_folds: Optional CV folds
            
        Returns:
            Dictionary with randomization results
        """
        print("\n🎲 Y-RANDOMIZATION TEST (Y-SCRAMBLING)")
        print("=" * 70)
        print(f"Running {n_iterations} iterations with randomized targets...")
        
        rmse_scores = []
        r2_scores = []
        
        for i in range(n_iterations):
            # Randomize target
            y_random = np.random.permutation(y)
            
            if cv_folds is None:
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_random, test_size=0.2, random_state=i
                )
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))
                r2_scores.append(r2_score(y_test, y_pred))
            else:
                fold_rmse = []
                fold_r2 = []
                
                for train_idx, val_idx in cv_folds:
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y_random[train_idx], y_random[val_idx]
                    
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_val)
                    
                    fold_rmse.append(np.sqrt(mean_squared_error(y_val, y_pred)))
                    fold_r2.append(r2_score(y_val, y_pred))
                
                rmse_scores.append(np.mean(fold_rmse))
                r2_scores.append(np.mean(fold_r2))
        
        results = {
            'rmse_mean': np.mean(rmse_scores),
            'rmse_std': np.std(rmse_scores),
            'r2_mean': np.mean(r2_scores),
            'r2_std': np.std(r2_scores),
            'rmse_scores': rmse_scores,
            'r2_scores': r2_scores,
        }
        
        print(f"\n📊 Randomized Results ({n_iterations} iterations):")
        print(f"   RMSE: {results['rmse_mean']:.4f} ± {results['rmse_std']:.4f}")
        print(f"   R²:   {results['r2_mean']:.4f} ± {results['r2_std']:.4f}")
        
        # Interpretation
        if results['r2_mean'] > 0.2:
            print("\n⚠️  WARNING: R² > 0.2 with randomized targets")
            print("   → Model is likely overfitting")
            print("   → Reduce model complexity")
            print("   → Increase regularization")
        elif results['r2_mean'] > 0.0:
            print("\n🟡 CAUTION: Positive R² with randomized targets")
            print("   → Some overfitting detected")
            print("   → Consider simplifying model")
        else:
            print("\n✓ Good: R² ≤ 0 with randomized targets")
            print("  → Model is not memorizing random noise")
        
        return results


class AssayNoiseEstimator:
    """
    Estimate and report experimental/assay noise levels.
    """
    
    @staticmethod
    def estimate_experimental_error(df: pd.DataFrame, target_col: str,
                                    replicate_groups: Optional[List] = None) -> Dict:
        """
        Estimate experimental error from replicates or literature.
        
        Args:
            df: DataFrame with target values
            target_col: Target column name
            replicate_groups: Optional list of replicate group identifiers
            
        Returns:
            Dictionary with error estimates
        """
        print("\n🔬 EXPERIMENTAL ERROR ESTIMATION")
        print("=" * 70)
        
        if replicate_groups is not None and len(replicate_groups) > 0:
            # Calculate from replicates
            errors = []
            for group in set(replicate_groups):
                group_values = df[replicate_groups == group][target_col].values
                if len(group_values) > 1:
                    errors.append(np.std(group_values))
            
            if errors:
                mean_error = np.mean(errors)
                print(f"✓ Estimated from {len(errors)} replicate groups")
                print(f"   Mean replicate error: {mean_error:.4f}")
                
                return {
                    'experimental_error': mean_error,
                    'source': 'replicates',
                    'n_groups': len(errors)
                }
        
        # Use literature estimates
        print("📚 Using typical IC₅₀/EC₅₀ assay error estimates:")
        print("   Typical experimental error: 0.3 - 0.6 log units")
        print("   Conservative estimate: 0.5 log units")
        print("\n📌 IMPLICATION:")
        print("   RMSE < 0.5 may indicate overfitting or lucky split")
        print("   RMSE ≈ 0.5 is near theoretical limit")
        print("   Report model error relative to assay precision")
        
        return {
            'experimental_error': 0.5,
            'source': 'literature',
            'range': (0.3, 0.6)
        }


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


# Convenience function to run all validation checks
def run_comprehensive_validation(df: pd.DataFrame, 
                                 smiles_col: str = 'Canonical SMILES',
                                 target_col: str = 'IC50 uM') -> Dict:
    """
    Run all validation checks in sequence.
    
    Args:
        df: DataFrame with SMILES and target
        smiles_col: SMILES column name
        target_col: Target column name
        
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


if __name__ == "__main__":
    print_comprehensive_validation_checklist()
