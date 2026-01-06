"""
Activity Cliff Detection Module
================================

Detects activity cliffs (similar molecules with large activity differences).
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from typing import Optional


class ActivityCliffDetector:
    """
    Detect activity cliffs: pairs of structurally similar molecules with 
    large activity differences, which indicate SAR discontinuities.
    """
    
    def __init__(self, smiles_col: str = 'Canonical SMILES', target_col: str = 'IC50 uM'):
        """Initialize detector with column names."""
        self.smiles_col = smiles_col
        self.target_col = target_col
    
    def detect_activity_cliffs(self, 
                               df: pd.DataFrame, 
                               similarity_threshold: float = 0.85, 
                               activity_threshold: float = 2.0) -> pd.DataFrame:
        """
        Detect activity cliffs in dataset.
        
        Args:
            df: DataFrame with SMILES and target columns
            similarity_threshold: Minimum Tanimoto similarity (default 0.85)
            activity_threshold: Minimum activity difference in log units (default 2.0 = 100-fold)
            
        Returns:
            DataFrame with detected activity cliff pairs containing:
                - mol1_idx, mol2_idx: Indices of the molecule pair
                - smiles1, smiles2: SMILES strings
                - activity1, activity2: Activity values
                - activity_diff: Absolute difference
                - similarity: Tanimoto similarity
        """
        print("\n[WARNING]  ACTIVITY CLIFF DETECTION")
        print("=" * 70)
        print(f"Criteria: Similarity >= {similarity_threshold}, "
              f"Activity diff >= {activity_threshold}")
        
        # Generate Morgan fingerprints
        fps = self._generate_fingerprints(df[self.smiles_col])
        activities = df[self.target_col].values
        
        # Find cliff pairs
        cliffs = self._find_cliffs(df, fps, activities, similarity_threshold, activity_threshold)
        cliff_df = pd.DataFrame(cliffs)
        
        # Report results
        self._print_results(cliff_df)
        
        return cliff_df
    
    def _generate_fingerprints(self, smiles_series: pd.Series) -> list:
        """Generate Morgan fingerprints for all molecules."""
        fps = []
        for smi in smiles_series:
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048))
                else:
                    fps.append(None)
            except:
                fps.append(None)
        return fps
    
    def _find_cliffs(self, df: pd.DataFrame, fps: list, activities: np.ndarray,
                     sim_threshold: float, act_threshold: float) -> list:
        """Find all activity cliff pairs."""
        cliffs = []
        
        for i in range(len(fps)):
            if fps[i] is None:
                continue
            for j in range(i + 1, len(fps)):
                if fps[j] is None:
                    continue
                
                similarity = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                activity_diff = abs(activities[i] - activities[j])
                
                if similarity >= sim_threshold and activity_diff >= act_threshold:
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
        
        return cliffs
    
    def _print_results(self, cliff_df: pd.DataFrame) -> None:
        """Print activity cliff detection results."""
        if len(cliff_df) > 0:
            print(f"\n[WARNING]  Found {len(cliff_df)} activity cliff pairs")
            print(f"   Mean similarity: {cliff_df['similarity'].mean():.3f}")
            print(f"   Mean activity diff: {cliff_df['activity_diff'].mean():.3f}")
            print("\n[NOTE] IMPLICATIONS:")
            print("   -> Local SAR is discontinuous")
            print("   -> Fingerprint-based models may struggle")
            print("   -> Consider local models or Gaussian Processes")
            print("   -> Feature importance interpretation limited")
        else:
            print("\n[OK] No activity cliffs detected")
            print("  -> SAR appears continuous")
