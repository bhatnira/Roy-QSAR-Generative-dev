"""
Dataset Quality Analysis Module
================================

Mitigates:
1. Dataset Bias & Representativeness
4. Measurement Noise & Assay Variability
6. Target Imbalance & Range Compression

Provides tools to:
- Analyze scaffold diversity
- Detect narrow chemical space
- Identify assay inconsistencies
- Check activity distribution balance
- Report dataset statistics

Author: QSAR Validation Framework
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from collections import Counter
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import DataStructs
from rdkit.Chem import AllChem


class DatasetQualityAnalyzer:
    """
    Comprehensive dataset quality analysis.
    
    Detects:
    - Narrow chemical space (congeneric series)
    - Scaffold imbalance
    - Activity range issues
    - Assay variability problems
    
    Parameters
    ----------
    smiles_col : str
        Column name for SMILES strings
    
    activity_col : str
        Column name for activity values
    
    assay_col : str, optional
        Column name for assay type/protocol
    
    Examples
    --------
    >>> analyzer = DatasetQualityAnalyzer(
    ...     smiles_col='SMILES',
    ...     activity_col='pIC50'
    ... )
    >>> 
    >>> report = analyzer.analyze(df)
    >>> print(report['scaffold_diversity'])
    >>> print(report['warnings'])
    """
    
    def __init__(
        self,
        smiles_col: str = 'SMILES',
        activity_col: str = 'pIC50',
        assay_col: Optional[str] = None
    ):
        self.smiles_col = smiles_col
        self.activity_col = activity_col
        self.assay_col = assay_col
    
    def analyze(self, df: pd.DataFrame) -> Dict:
        """
        Run comprehensive dataset quality analysis.
        
        Returns
        -------
        report : dict
            Comprehensive quality report with warnings
        """
        print("\n" + "="*80)
        print("DATASET QUALITY ANALYSIS")
        print("="*80)
        
        report = {}
        warnings = []
        
        # 1. Scaffold diversity analysis
        print("\n1. Scaffold Diversity Analysis")
        print("-" * 80)
        scaffold_stats = self._analyze_scaffold_diversity(df)
        report['scaffold_diversity'] = scaffold_stats
        
        if scaffold_stats['diversity_score'] < 0.3:
            warnings.append("⚠️  LOW SCAFFOLD DIVERSITY: Dataset is congeneric (diversity < 0.3)")
        if scaffold_stats['top_scaffold_fraction'] > 0.5:
            warnings.append(f"⚠️  DOMINANT SCAFFOLD: Top scaffold represents {scaffold_stats['top_scaffold_fraction']*100:.1f}% of data")
        
        # 2. Chemical space coverage
        print("\n2. Chemical Space Analysis")
        print("-" * 80)
        space_stats = self._analyze_chemical_space(df)
        report['chemical_space'] = space_stats
        
        if space_stats['mean_tanimoto'] > 0.7:
            warnings.append(f"⚠️  NARROW CHEMICAL SPACE: High similarity (mean Tanimoto = {space_stats['mean_tanimoto']:.3f})")
        
        # 3. Activity distribution
        print("\n3. Activity Distribution Analysis")
        print("-" * 80)
        activity_stats = self._analyze_activity_distribution(df)
        report['activity_distribution'] = activity_stats
        
        if activity_stats['range'] < 2.0:
            warnings.append(f"⚠️  NARROW ACTIVITY RANGE: Range = {activity_stats['range']:.2f} log units (< 2.0)")
        if activity_stats['skewness'] > 1.5 or activity_stats['skewness'] < -1.5:
            warnings.append(f"⚠️  SKEWED DISTRIBUTION: Skewness = {activity_stats['skewness']:.2f}")
        
        # 4. Assay variability (if available)
        if self.assay_col and self.assay_col in df.columns:
            print("\n4. Assay Variability Analysis")
            print("-" * 80)
            assay_stats = self._analyze_assay_variability(df)
            report['assay_variability'] = assay_stats
            
            if len(assay_stats['assay_types']) > 3:
                warnings.append(f"⚠️  MULTIPLE ASSAYS: {len(assay_stats['assay_types'])} different assays detected")
        
        # 5. Sample size assessment
        print("\n5. Sample Size Assessment")
        print("-" * 80)
        size_stats = self._analyze_sample_size(df)
        report['sample_size'] = size_stats
        
        if size_stats['total_samples'] < 100:
            warnings.append(f"⚠️  SMALL DATASET: Only {size_stats['total_samples']} samples (< 100)")
        
        report['warnings'] = warnings
        report['quality_score'] = self._calculate_quality_score(report)
        
        # Print summary
        self._print_summary(report)
        
        return report
    
    def _analyze_scaffold_diversity(self, df: pd.DataFrame) -> Dict:
        """Analyze Bemis-Murcko scaffold diversity."""
        scaffolds = []
        
        for smiles in df[self.smiles_col]:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                    scaffold_smiles = Chem.MolToSmiles(scaffold)
                    scaffolds.append(scaffold_smiles)
                else:
                    scaffolds.append(None)
            except:
                scaffolds.append(None)
        
        scaffold_counts = Counter([s for s in scaffolds if s is not None])
        n_unique = len(scaffold_counts)
        n_total = len([s for s in scaffolds if s is not None])
        
        diversity_score = n_unique / n_total if n_total > 0 else 0
        
        top_scaffolds = scaffold_counts.most_common(5)
        top_scaffold_fraction = top_scaffolds[0][1] / n_total if top_scaffolds else 0
        
        print(f"  Total scaffolds: {n_unique}")
        print(f"  Total molecules: {n_total}")
        print(f"  Diversity score: {diversity_score:.3f}")
        print(f"  Top scaffold represents: {top_scaffold_fraction*100:.1f}% of data")
        print(f"\n  Top 5 scaffolds:")
        for i, (scaffold, count) in enumerate(top_scaffolds[:5], 1):
            print(f"    {i}. {count} molecules ({count/n_total*100:.1f}%)")
        
        return {
            'n_unique_scaffolds': n_unique,
            'n_total_molecules': n_total,
            'diversity_score': diversity_score,
            'top_scaffold_fraction': top_scaffold_fraction,
            'top_scaffolds': top_scaffolds
        }
    
    def _analyze_chemical_space(self, df: pd.DataFrame) -> Dict:
        """Analyze chemical space coverage using Tanimoto similarity."""
        # Sample if too large
        sample_size = min(500, len(df))
        df_sample = df.sample(n=sample_size, random_state=42)
        
        fps = []
        for smiles in df_sample[self.smiles_col]:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                    fps.append(fp)
            except:
                pass
        
        if len(fps) < 2:
            return {'error': 'Insufficient valid molecules'}
        
        # Calculate pairwise Tanimoto similarities
        similarities = []
        for i in range(min(100, len(fps))):
            for j in range(i+1, min(100, len(fps))):
                sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                similarities.append(sim)
        
        mean_sim = np.mean(similarities)
        median_sim = np.median(similarities)
        
        print(f"  Sampled: {len(fps)} molecules")
        print(f"  Mean Tanimoto similarity: {mean_sim:.3f}")
        print(f"  Median Tanimoto similarity: {median_sim:.3f}")
        
        if mean_sim > 0.7:
            print(f"  ⚠️  Chemical space is NARROW (mean similarity > 0.7)")
        elif mean_sim > 0.5:
            print(f"  ⚙️  Chemical space is MODERATE (0.5 < mean similarity < 0.7)")
        else:
            print(f"  ✓  Chemical space is DIVERSE (mean similarity < 0.5)")
        
        return {
            'n_compared': len(fps),
            'mean_tanimoto': mean_sim,
            'median_tanimoto': median_sim,
            'interpretation': 'narrow' if mean_sim > 0.7 else 'moderate' if mean_sim > 0.5 else 'diverse'
        }
    
    def _analyze_activity_distribution(self, df: pd.DataFrame) -> Dict:
        """Analyze activity value distribution."""
        activities = df[self.activity_col].dropna()
        
        mean_val = activities.mean()
        std_val = activities.std()
        min_val = activities.min()
        max_val = activities.max()
        range_val = max_val - min_val
        
        # Calculate percentiles
        q25, q50, q75 = activities.quantile([0.25, 0.5, 0.75])
        
        # Calculate skewness
        from scipy.stats import skew
        skewness = skew(activities)
        
        print(f"  N samples: {len(activities)}")
        print(f"  Mean ± Std: {mean_val:.2f} ± {std_val:.2f}")
        print(f"  Range: [{min_val:.2f}, {max_val:.2f}] ({range_val:.2f} log units)")
        print(f"  Quartiles: Q1={q25:.2f}, Q2={q50:.2f}, Q3={q75:.2f}")
        print(f"  Skewness: {skewness:.2f}")
        
        if range_val < 2.0:
            print(f"  ⚠️  NARROW RANGE: < 2 log units")
        
        if abs(skewness) > 1.5:
            print(f"  ⚠️  HIGHLY SKEWED DISTRIBUTION")
        
        return {
            'n_samples': len(activities),
            'mean': mean_val,
            'std': std_val,
            'min': min_val,
            'max': max_val,
            'range': range_val,
            'q25': q25,
            'q50': q50,
            'q75': q75,
            'skewness': skewness
        }
    
    def _analyze_assay_variability(self, df: pd.DataFrame) -> Dict:
        """Analyze variability across assay types."""
        assay_types = df[self.assay_col].unique()
        
        stats_by_assay = {}
        for assay in assay_types:
            assay_data = df[df[self.assay_col] == assay][self.activity_col]
            stats_by_assay[assay] = {
                'n': len(assay_data),
                'mean': assay_data.mean(),
                'std': assay_data.std()
            }
        
        print(f"  Number of assay types: {len(assay_types)}")
        print(f"\n  Statistics by assay:")
        for assay, stats in stats_by_assay.items():
            print(f"    {assay}: N={stats['n']}, Mean={stats['mean']:.2f}, Std={stats['std']:.2f}")
        
        return {
            'n_assay_types': len(assay_types),
            'assay_types': list(assay_types),
            'stats_by_assay': stats_by_assay
        }
    
    def _analyze_sample_size(self, df: pd.DataFrame) -> Dict:
        """Assess whether sample size is adequate."""
        n_total = len(df)
        n_features_typical = 1024  # Typical fingerprint size
        
        ratio = n_total / n_features_typical
        
        print(f"  Total samples: {n_total}")
        print(f"  Typical features: {n_features_typical}")
        print(f"  Sample-to-feature ratio: {ratio:.2f}")
        
        if ratio < 0.1:
            print(f"  ⚠️  VERY LOW: Risk of severe overfitting")
            recommendation = "Use very simple models (Ridge, Linear)"
        elif ratio < 0.5:
            print(f"  ⚠️  LOW: Use simple models only")
            recommendation = "Use simple models (Ridge, Random Forest with low depth)"
        else:
            print(f"  ✓  ADEQUATE: Can use moderate complexity models")
            recommendation = "Can use moderate complexity models"
        
        return {
            'total_samples': n_total,
            'typical_features': n_features_typical,
            'sample_feature_ratio': ratio,
            'recommendation': recommendation
        }
    
    def _calculate_quality_score(self, report: Dict) -> float:
        """Calculate overall quality score (0-1)."""
        score = 0.0
        
        # Scaffold diversity (0-0.3)
        if 'scaffold_diversity' in report:
            score += min(report['scaffold_diversity']['diversity_score'], 0.3)
        
        # Chemical space diversity (0-0.3)
        if 'chemical_space' in report and 'mean_tanimoto' in report['chemical_space']:
            # Lower similarity = higher score
            space_score = (1 - report['chemical_space']['mean_tanimoto']) * 0.3
            score += space_score
        
        # Activity range (0-0.2)
        if 'activity_distribution' in report:
            range_score = min(report['activity_distribution']['range'] / 5.0, 1.0) * 0.2
            score += range_score
        
        # Sample size (0-0.2)
        if 'sample_size' in report:
            size_score = min(report['sample_size']['total_samples'] / 200, 1.0) * 0.2
            score += size_score
        
        return score
    
    def _print_summary(self, report: Dict):
        """Print summary and warnings."""
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        
        quality_score = report.get('quality_score', 0)
        print(f"\nOverall Quality Score: {quality_score:.2f} / 1.00")
        
        if quality_score > 0.7:
            print("✓ HIGH QUALITY dataset")
        elif quality_score > 0.5:
            print("⚙️  MODERATE QUALITY dataset")
        else:
            print("⚠️  LOW QUALITY dataset - results may not generalize well")
        
        if report['warnings']:
            print(f"\n⚠️  {len(report['warnings'])} WARNING(S):")
            for warning in report['warnings']:
                print(f"  {warning}")
        else:
            print("\n✓ No major quality issues detected")
        
        print("\n" + "="*80)
    
    def generate_split_recommendations(self, df: pd.DataFrame) -> Dict:
        """
        Recommend splitting strategy based on dataset characteristics.
        
        Returns
        -------
        recommendations : dict
            Splitting strategy recommendations
        """
        report = self.analyze(df)
        
        recommendations = {
            'primary_strategy': None,
            'alternative_strategy': None,
            'reasoning': []
        }
        
        # Check scaffold diversity
        diversity = report['scaffold_diversity']['diversity_score']
        n_scaffolds = report['scaffold_diversity']['n_unique_scaffolds']
        
        if diversity < 0.3:
            recommendations['primary_strategy'] = 'cluster'
            recommendations['reasoning'].append(
                "Low scaffold diversity (congeneric) → Use cluster-based splitting"
            )
        elif n_scaffolds >= 20:
            recommendations['primary_strategy'] = 'scaffold'
            recommendations['reasoning'].append(
                f"Good scaffold diversity ({n_scaffolds} scaffolds) → Use scaffold-based splitting"
            )
        else:
            recommendations['primary_strategy'] = 'cluster'
            recommendations['reasoning'].append(
                f"Moderate scaffold count ({n_scaffolds}) → Use cluster-based splitting"
            )
        
        # Check sample size
        n_samples = report['sample_size']['total_samples']
        if n_samples < 100:
            recommendations['alternative_strategy'] = 'cluster'
            recommendations['reasoning'].append(
                f"Small dataset ({n_samples} samples) → Consider leave-cluster-out CV"
            )
        
        return recommendations


def demonstrate_quality_analysis():
    """Demonstrate dataset quality analysis."""
    print("\n" + "="*80)
    print("DATASET QUALITY ANALYSIS DEMONSTRATION")
    print("="*80)
    
    # Create synthetic dataset
    np.random.seed(42)
    
    # Simulate congeneric series (low diversity)
    base_scaffold = 'c1ccccc1'
    smiles_list = [f"{base_scaffold}C{'C'*i}N" for i in range(50)]
    smiles_list += [f"{base_scaffold}C{'C'*i}O" for i in range(30)]
    smiles_list += ["c1ccc(cc1)C", "c1ccc(cc1)N", "c1ccc(cc1)O"] * 10
    
    # Simulate narrow activity range
    activities = np.random.normal(6.5, 0.3, len(smiles_list))
    
    df = pd.DataFrame({
        'SMILES': smiles_list,
        'pIC50': activities
    })
    
    print(f"\nAnalyzing synthetic dataset:")
    print(f"  {len(df)} molecules")
    print(f"  Activity range: {activities.min():.2f} - {activities.max():.2f}")
    
    # Run analysis
    analyzer = DatasetQualityAnalyzer(smiles_col='SMILES', activity_col='pIC50')
    report = analyzer.analyze(df)
    
    # Get recommendations
    print("\n" + "="*80)
    print("SPLITTING STRATEGY RECOMMENDATIONS")
    print("="*80)
    recommendations = analyzer.generate_split_recommendations(df)
    print(f"\nPrimary strategy: {recommendations['primary_strategy']}")
    print(f"\nReasoning:")
    for reason in recommendations['reasoning']:
        print(f"  - {reason}")


if __name__ == '__main__':
    demonstrate_quality_analysis()
