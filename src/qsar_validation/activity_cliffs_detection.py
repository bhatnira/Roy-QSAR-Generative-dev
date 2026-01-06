"""
Activity Cliffs Detection Module
=================================

Mitigates:
5. Activity Cliffs (Major QSAR Limitation)

Provides tools to:
- Identify activity cliffs
- Quantify cliff severity
- Report local prediction instability
- Recommend modeling strategies

Activity cliffs = structurally similar molecules with large activity differences.
These are fundamental QSAR limitations.

Author: QSAR Validation Framework
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs


class ActivityCliffsDetector:
    """
    Detect and analyze activity cliffs.
    
    Activity cliffs are pairs of molecules that are:
    - Structurally similar (high Tanimoto similarity)
    - But have large activity differences
    
    Parameters
    ----------
    smiles_col : str
        Column name for SMILES
    
    activity_col : str
        Column name for activity values
    
    similarity_threshold : float
        Tanimoto similarity threshold (default 0.85)
    
    activity_threshold : float
        Activity difference threshold in log units (default 2.0)
    
    Examples
    --------
    >>> detector = ActivityCliffsDetector(
    ...     smiles_col='SMILES',
    ...     activity_col='pIC50'
    ... )
    >>> 
    >>> cliffs = detector.detect_cliffs(df)
    >>> report = detector.analyze_cliffs(cliffs)
    """
    
    def __init__(
        self,
        smiles_col: str = 'SMILES',
        activity_col: str = 'pIC50',
        similarity_threshold: float = 0.85,
        activity_threshold: float = 2.0
    ):
        self.smiles_col = smiles_col
        self.activity_col = activity_col
        self.similarity_threshold = similarity_threshold
        self.activity_threshold = activity_threshold
    
    def detect_cliffs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect activity cliffs in dataset.
        
        Returns
        -------
        cliffs_df : pd.DataFrame
            DataFrame with cliff pairs
        """
        print(f"\n{'='*80}")
        print("ACTIVITY CLIFFS DETECTION")
        print(f"{'='*80}")
        print(f"\nSettings:")
        print(f"  Similarity threshold: {self.similarity_threshold}")
        print(f"  Activity threshold: {self.activity_threshold} log units")
        
        # Calculate fingerprints
        print(f"\nCalculating fingerprints...")
        fps = []
        valid_indices = []
        
        for idx, smiles in enumerate(df[self.smiles_col]):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                    fps.append(fp)
                    valid_indices.append(idx)
            except:
                pass
        
        print(f"  Valid molecules: {len(fps)}/{len(df)}")
        
        # Find cliffs
        print(f"\nSearching for activity cliffs...")
        cliffs = []
        
        for i in range(len(fps)):
            for j in range(i + 1, len(fps)):
                # Calculate similarity
                sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                
                if sim >= self.similarity_threshold:
                    # Check activity difference
                    idx_i = valid_indices[i]
                    idx_j = valid_indices[j]
                    
                    act_i = df.iloc[idx_i][self.activity_col]
                    act_j = df.iloc[idx_j][self.activity_col]
                    act_diff = abs(act_i - act_j)
                    
                    if act_diff >= self.activity_threshold:
                        cliffs.append({
                            'mol1_idx': idx_i,
                            'mol2_idx': idx_j,
                            'mol1_smiles': df.iloc[idx_i][self.smiles_col],
                            'mol2_smiles': df.iloc[idx_j][self.smiles_col],
                            'mol1_activity': act_i,
                            'mol2_activity': act_j,
                            'activity_diff': act_diff,
                            'similarity': sim
                        })
        
        cliffs_df = pd.DataFrame(cliffs)
        
        print(f"\n✓ Found {len(cliffs_df)} activity cliff pairs")
        
        return cliffs_df
    
    def analyze_cliffs(self, cliffs_df: pd.DataFrame, df: pd.DataFrame) -> Dict:
        """
        Analyze activity cliffs and their implications.
        
        Parameters
        ----------
        cliffs_df : pd.DataFrame
            Output from detect_cliffs()
        
        df : pd.DataFrame
            Original dataset
        
        Returns
        -------
        analysis : dict
            Cliff analysis results
        """
        print(f"\n{'='*80}")
        print("ACTIVITY CLIFFS ANALYSIS")
        print(f"{'='*80}")
        
        if len(cliffs_df) == 0:
            print("\n✓ No activity cliffs detected")
            return {'n_cliffs': 0, 'cliffs_ratio': 0.0}
        
        n_cliffs = len(cliffs_df)
        n_molecules = len(df)
        
        # Molecules involved in cliffs
        cliff_molecules = set()
        cliff_molecules.update(cliffs_df['mol1_idx'])
        cliff_molecules.update(cliffs_df['mol2_idx'])
        n_cliff_molecules = len(cliff_molecules)
        
        # Statistics
        mean_sim = cliffs_df['similarity'].mean()
        mean_act_diff = cliffs_df['activity_diff'].mean()
        max_act_diff = cliffs_df['activity_diff'].max()
        
        # Severity score
        severity = (n_cliff_molecules / n_molecules) * (mean_act_diff / self.activity_threshold)
        
        print(f"\nCliff Statistics:")
        print(f"  Number of cliff pairs: {n_cliffs}")
        print(f"  Molecules involved: {n_cliff_molecules}/{n_molecules} ({n_cliff_molecules/n_molecules*100:.1f}%)")
        print(f"  Mean similarity: {mean_sim:.3f}")
        print(f"  Mean activity difference: {mean_act_diff:.2f} log units")
        print(f"  Max activity difference: {max_act_diff:.2f} log units")
        print(f"  Severity score: {severity:.3f}")
        
        # Interpretation
        print(f"\n{'='*80}")
        print("IMPLICATIONS FOR QSAR MODELING")
        print(f"{'='*80}")
        
        if severity < 0.1:
            print("\n✓ LOW: Activity cliffs are minimal")
            print("  → Standard QSAR approaches should work well")
            recommendation = "standard"
        elif severity < 0.3:
            print("\n⚙️  MODERATE: Some activity cliffs present")
            print("  → Consider:")
            print("    - Local models (k-NN, GPR)")
            print("    - Ensemble methods")
            print("    - Report cliff molecules separately")
            recommendation = "local_models"
        else:
            print("\n⚠️  HIGH: Many activity cliffs detected")
            print("  → QSAR models will struggle with these regions")
            print("  → Recommendations:")
            print("    - Use Gaussian Process Regression (handles uncertainty)")
            print("    - Build separate models for cliff regions")
            print("    - Report predictions near cliffs as uncertain")
            print("    - DO NOT over-interpret feature importance")
            recommendation = "gpr_or_separate"
        
        # Get top cliffs
        print(f"\n{'='*80}")
        print("TOP 5 ACTIVITY CLIFFS")
        print(f"{'='*80}")
        
        top_cliffs = cliffs_df.nlargest(5, 'activity_diff')
        for idx, cliff in top_cliffs.iterrows():
            print(f"\nCliff {idx + 1}:")
            print(f"  Similarity: {cliff['similarity']:.3f}")
            print(f"  Activity difference: {cliff['activity_diff']:.2f} log units")
            print(f"  Mol 1: {cliff['mol1_activity']:.2f}")
            print(f"  Mol 2: {cliff['mol2_activity']:.2f}")
        
        return {
            'n_cliffs': n_cliffs,
            'n_cliff_molecules': n_cliff_molecules,
            'cliffs_ratio': n_cliff_molecules / n_molecules,
            'mean_similarity': mean_sim,
            'mean_activity_diff': mean_act_diff,
            'max_activity_diff': max_act_diff,
            'severity_score': severity,
            'recommendation': recommendation,
            'top_cliffs': top_cliffs.to_dict('records')
        }
    
    def identify_cliff_regions(
        self,
        df: pd.DataFrame,
        cliffs_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Identify molecules that are in cliff regions.
        
        These molecules may have unreliable predictions.
        
        Parameters
        ----------
        df : pd.DataFrame
            Original dataset
        
        cliffs_df : pd.DataFrame
            Output from detect_cliffs()
        
        Returns
        -------
        df_with_flags : pd.DataFrame
            Original df with 'in_cliff_region' column
        """
        # Get molecules involved in cliffs
        cliff_molecules = set()
        cliff_molecules.update(cliffs_df['mol1_idx'])
        cliff_molecules.update(cliffs_df['mol2_idx'])
        
        # Add flag
        df_copy = df.copy()
        df_copy['in_cliff_region'] = df_copy.index.isin(cliff_molecules)
        
        n_cliff = df_copy['in_cliff_region'].sum()
        
        print(f"\n✓ Flagged {n_cliff} molecules in cliff regions")
        print(f"  → Predictions for these may be unreliable")
        
        return df_copy


def demonstrate_cliff_detection():
    """Demonstrate activity cliffs detection."""
    print("\n" + "="*80)
    print("ACTIVITY CLIFFS DETECTION DEMONSTRATION")
    print("="*80)
    
    # Create synthetic dataset with activity cliffs
    np.random.seed(42)
    
    # Base molecules (similar structures)
    base_smiles = [
        'c1ccccc1C',
        'c1ccccc1CC',
        'c1ccccc1CCC',
    ]
    
    # Add variants with different activities
    smiles_list = []
    activities = []
    
    # Cluster 1: Similar structures, similar activities (no cliff)
    for i in range(10):
        smiles_list.append(f'c1ccccc1C{"C"*i}')
        activities.append(6.0 + np.random.normal(0, 0.2))
    
    # Cluster 2: Similar structures, VERY different activities (cliff!)
    smiles_list.append('c1ccc(cc1)CN')
    activities.append(4.0)  # Low activity
    
    smiles_list.append('c1ccc(cc1)CO')  # Very similar
    activities.append(8.5)  # High activity - CLIFF!
    
    # Cluster 3: More diverse structures
    for i in range(8):
        smiles_list.append(f'c1cc(ccc1){"N" if i % 2 == 0 else "O"}')
        activities.append(5.5 + np.random.normal(0, 0.5))
    
    df = pd.DataFrame({
        'SMILES': smiles_list,
        'pIC50': activities
    })
    
    print(f"\nSynthetic dataset: {len(df)} molecules")
    print(f"Activity range: {min(activities):.2f} - {max(activities):.2f}")
    
    # Detect cliffs
    detector = ActivityCliffsDetector(
        smiles_col='SMILES',
        activity_col='pIC50',
        similarity_threshold=0.80,
        activity_threshold=2.0
    )
    
    cliffs_df = detector.detect_cliffs(df)
    
    if len(cliffs_df) > 0:
        analysis = detector.analyze_cliffs(cliffs_df, df)
        df_flagged = detector.identify_cliff_regions(df, cliffs_df)
    
    print("\n" + "="*80)
    print("✓ Demonstration complete!")
    print("="*80)


if __name__ == '__main__':
    demonstrate_cliff_detection()
