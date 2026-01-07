"""
Dataset Bias and Representativeness Analysis Module
===================================================

Analyzes chemical space coverage, scaffold diversity, and activity distribution.

Classes:
    DatasetBiasAnalyzer: Main analyzer for dataset characteristics
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from typing import Dict


class DatasetBiasAnalyzer:
    """
    Analyze dataset bias, chemical space coverage, and representativeness.
    
    Attributes:
        smiles_col (str): Name of SMILES column
        target_col (str): Name of target activity column
    """
    
    def __init__(self, smiles_col: str = 'Canonical SMILES', target_col: str = 'IC50 uM'):
        """Initialize analyzer with column names."""
        self.smiles_col = smiles_col
        self.target_col = target_col
    
    def analyze_scaffold_diversity(self, df: pd.DataFrame) -> Dict:
        """
        Analyze scaffold diversity and distribution using Bemis-Murcko scaffolds.
        
        Args:
            df: DataFrame with SMILES column
            
        Returns:
            Dictionary containing:
                - n_molecules: Total number of molecules
                - n_scaffolds: Number of unique scaffolds
                - diversity_ratio: Ratio of scaffolds to molecules
                - gini_coefficient: Inequality measure (0=equal, 1=unequal)
                - top_scaffold_fraction: Fraction represented by top 5 scaffolds
                - scaffold_counts: Series with scaffold frequencies
                - df_with_scaffolds: DataFrame with scaffold column added
        """
        print("\n[ANALYSIS] DATASET BIAS ANALYSIS: Scaffold Diversity")
        print("=" * 70)
        
        # Extract Bemis-Murcko scaffolds
        scaffolds = self._extract_scaffolds(df[self.smiles_col])
        
        df_analyzed = df.copy()
        df_analyzed['scaffold'] = scaffolds
        df_analyzed = df_analyzed.dropna(subset=['scaffold'])
        
        # Calculate diversity metrics
        scaffold_counts = df_analyzed['scaffold'].value_counts()
        metrics = self._calculate_diversity_metrics(scaffold_counts, len(df_analyzed))
        
        # Display results
        self._print_diversity_results(metrics, scaffold_counts)
        
        return {
            'n_molecules': metrics['n_molecules'],
            'n_scaffolds': metrics['n_scaffolds'],
            'diversity_ratio': metrics['diversity_ratio'],
            'gini_coefficient': metrics['gini_coefficient'],
            'top_scaffold_fraction': metrics['top_scaffold_fraction'],
            'scaffold_counts': scaffold_counts,
            'df_with_scaffolds': df_analyzed
        }
    
    def analyze_activity_distribution(self, df: pd.DataFrame) -> Dict:
        """
        Analyze target activity distribution and identify potential issues.
        
        Args:
            df: DataFrame with target column
            
        Returns:
            Dictionary with distribution statistics including:
                - Basic stats (mean, median, std, range, IQR)
                - Outlier detection results
                - Warning flags
        """
        print("\n[METRICS] TARGET DISTRIBUTION ANALYSIS")
        print("=" * 70)
        
        activities = df[self.target_col].values
        
        # Calculate statistics
        stats = self._calculate_activity_stats(activities)
        
        # Display and check for issues
        self._print_activity_results(stats, activities)
        
        return stats
    
    def report_split_diversity(self, df_with_scaffolds: pd.DataFrame, 
                               train_idx: np.ndarray, 
                               val_idx: np.ndarray, 
                               test_idx: np.ndarray) -> None:
        """
        Report scaffold diversity across train/validation/test splits.
        
        Args:
            df_with_scaffolds: DataFrame with scaffold column
            train_idx: Training set indices
            val_idx: Validation set indices
            test_idx: Test set indices
        """
        print("\n[METRICS] SCAFFOLD DISTRIBUTION PER SPLIT")
        print("=" * 70)
        
        train_scaffolds = set(df_with_scaffolds.iloc[train_idx]['scaffold'])
        val_scaffolds = set(df_with_scaffolds.iloc[val_idx]['scaffold'])
        test_scaffolds = set(df_with_scaffolds.iloc[test_idx]['scaffold'])
        
        print(f"Training set: {len(train_scaffolds)} unique scaffolds")
        print(f"Validation set: {len(val_scaffolds)} unique scaffolds")
        print(f"Test set: {len(test_scaffolds)} unique scaffolds")
        
        # Check for novel scaffolds
        novel_test = test_scaffolds - train_scaffolds
        if len(novel_test) > 0:
            print(f"\n[OK] Test set contains {len(novel_test)} novel scaffolds "
                  f"({len(novel_test)/len(test_scaffolds)*100:.1f}%)")
            print("  -> Good generalization test")
        else:
            print("\n[WARNING]  WARNING: All test scaffolds present in training")
            print("  -> May not test generalization adequately")
    
    # Private helper methods
    
    @staticmethod
    def _extract_scaffolds(smiles_series: pd.Series) -> list:
        """Extract Bemis-Murcko scaffolds from SMILES."""
        scaffolds = []
        for smi in smiles_series:
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                    scaffolds.append(Chem.MolToSmiles(scaffold))
                else:
                    scaffolds.append(None)
            except:
                scaffolds.append(None)
        return scaffolds
    
    @staticmethod
    def _calculate_diversity_metrics(scaffold_counts: pd.Series, n_molecules: int) -> Dict:
        """Calculate scaffold diversity metrics."""
        n_unique_scaffolds = len(scaffold_counts)
        diversity_ratio = n_unique_scaffolds / n_molecules
        gini_coefficient = DatasetBiasAnalyzer._calculate_gini(scaffold_counts.values)
        top_scaffold_fraction = scaffold_counts.head(5).sum() / n_molecules
        
        return {
            'n_molecules': n_molecules,
            'n_scaffolds': n_unique_scaffolds,
            'diversity_ratio': diversity_ratio,
            'gini_coefficient': gini_coefficient,
            'top_scaffold_fraction': top_scaffold_fraction
        }
    
    @staticmethod
    def _calculate_gini(values: np.ndarray) -> float:
        """Calculate Gini coefficient for inequality measurement."""
        sorted_values = np.sort(values)
        n = len(values)
        cumsum = np.cumsum(sorted_values)
        return (2 * np.sum((n - np.arange(1, n + 1) + 0.5) * sorted_values)) / (n * np.sum(sorted_values)) - 1
    
    def _print_diversity_results(self, metrics: Dict, scaffold_counts: pd.Series) -> None:
        """Print diversity analysis results with warnings."""
        print(f"[METRICS] Total molecules: {metrics['n_molecules']}")
        print(f"[METRICS] Unique scaffolds: {metrics['n_scaffolds']}")
        print(f"[METRICS] Diversity ratio: {metrics['diversity_ratio']:.3f}")
        print(f"[METRICS] Gini coefficient: {metrics['gini_coefficient']:.3f} "
              "(0=perfect equality, 1=max inequality)")
        print(f"\n🔝 Top 5 scaffolds represent {metrics['top_scaffold_fraction']*100:.1f}% of dataset:")
        
        for i, (scaffold, count) in enumerate(scaffold_counts.head(5).items(), 1):
            print(f"   {i}. {count} molecules ({count/metrics['n_molecules']*100:.1f}%)")
        
        # Display warnings
        if metrics['diversity_ratio'] < 0.3:
            print("\n[WARNING]  WARNING: Low scaffold diversity (congeneric series)")
            print("   -> Model may not generalize beyond this scaffold family")
            print("   -> Consider stating limited applicability domain")
        
        if metrics['top_scaffold_fraction'] > 0.5:
            print("\n[WARNING]  WARNING: Dataset dominated by top scaffolds")
            print("   -> High risk of overfitting to these scaffolds")
            print("   -> Performance may be scaffold-specific")
        
        if metrics['gini_coefficient'] > 0.6:
            print("\n[WARNING]  WARNING: High scaffold imbalance (Gini > 0.6)")
            print("   -> Some scaffolds heavily overrepresented")
            print("   -> Consider scaffold-aware sampling")
    
    def _calculate_activity_stats(self, activities: np.ndarray) -> Dict:
        """Calculate activity distribution statistics."""
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
        
        # Detect outliers
        lower_bound = stats['q25'] - 1.5 * stats['iqr']
        upper_bound = stats['q75'] + 1.5 * stats['iqr']
        outliers = (activities < lower_bound) | (activities > upper_bound)
        stats['outliers'] = outliers
        stats['n_outliers'] = np.sum(outliers)
        
        return stats
    
    def _print_activity_results(self, stats: Dict, activities: np.ndarray) -> None:
        """Print activity distribution results with warnings."""
        print(f"Mean: {stats['mean']:.3f}")
        print(f"Median: {stats['median']:.3f}")
        print(f"Std Dev: {stats['std']:.3f}")
        print(f"Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
        print(f"IQR: {stats['iqr']:.3f}")
        
        # Check for narrow range
        relative_range = stats['range'] / stats['mean'] if stats['mean'] > 0 else 0
        print(f"\nRelative range: {relative_range:.3f}")
        
        if relative_range < 2.0:
            print("[WARNING]  WARNING: Narrow activity range")
            print("   -> Limited chemical space coverage")
            print("   -> R² may be artificially inflated")
            print("   -> Focus on RMSE/MAE for evaluation")
        
        # Check for clustering
        if stats['iqr'] < 0.5 * stats['std']:
            print("\n[WARNING]  WARNING: Activities clustered in narrow range")
            print("   -> Most values in small range")
            print("   -> Poor extrapolation expected")
        
        # Report outliers
        if stats['n_outliers'] > 0:
            print(f"\n[NOTE] Detected {stats['n_outliers']} potential outliers "
                  f"({stats['n_outliers']/len(activities)*100:.1f}%)")
            print("   -> Review for measurement errors")
            print("   -> Consider robust regression methods")
