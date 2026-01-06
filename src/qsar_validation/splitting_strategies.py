"""
Advanced Splitting Strategies Module
=====================================

Provides multiple splitting strategies for QSAR model validation:

1. Scaffold-based splitting (recommended for most cases)
   - Uses Bemis-Murcko scaffolds
   - Entire scaffolds assigned to train/val/test
   - Prevents scaffold leakage

2. Time-based splitting (when temporal data available)
   - Train on older compounds
   - Test on newer compounds
   - Simulates realistic deployment

3. Leave-cluster-out splitting (for small datasets)
   - Clusters compounds by fingerprint similarity
   - Holds out entire clusters
   - Tests generalization to dissimilar compounds

Author: QSAR Validation Framework
License: MIT
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import AllChem
from sklearn.cluster import KMeans, AgglomerativeClustering
from collections import defaultdict


class AdvancedSplitter:
    """
    Advanced splitting strategies for QSAR validation.
    
    Supports three splitting modes:
    1. 'scaffold' - Scaffold-based splitting (recommended)
    2. 'temporal' - Time-based splitting 
    3. 'cluster' - Leave-cluster-out splitting
    
    Parameters
    ----------
    smiles_col : str
        Column name containing SMILES strings
        
    strategy : str
        Splitting strategy: 'scaffold', 'temporal', or 'cluster'
        Default: 'scaffold'
        
    date_col : str, optional
        Column name containing dates (required for 'temporal' strategy)
        
    n_clusters : int, optional
        Number of clusters for 'cluster' strategy (default: 5)
        
    Examples
    --------
    # Scaffold-based (recommended)
    splitter = AdvancedSplitter(smiles_col='SMILES', strategy='scaffold')
    train_idx, val_idx, test_idx = splitter.split(df, test_size=0.2)
    
    # Time-based
    splitter = AdvancedSplitter(smiles_col='SMILES', strategy='temporal', date_col='Date')
    train_idx, val_idx, test_idx = splitter.split(df, test_size=0.2)
    
    # Cluster-based
    splitter = AdvancedSplitter(smiles_col='SMILES', strategy='cluster', n_clusters=5)
    train_idx, val_idx, test_idx = splitter.split(df, test_size=0.2)
    """
    
    def __init__(
        self,
        smiles_col: str = 'SMILES',
        strategy: str = 'scaffold',
        date_col: Optional[str] = None,
        n_clusters: int = 5
    ):
        self.smiles_col = smiles_col
        self.strategy = strategy.lower()
        self.date_col = date_col
        self.n_clusters = n_clusters
        
        # Validate strategy
        valid_strategies = ['scaffold', 'temporal', 'cluster']
        if self.strategy not in valid_strategies:
            raise ValueError(f"Strategy must be one of {valid_strategies}, got '{self.strategy}'")
        
        # Validate temporal strategy requirements
        if self.strategy == 'temporal' and date_col is None:
            raise ValueError("date_col must be provided for temporal splitting strategy")
        
        self.scaffold_dict = {}  # Cache scaffolds
        
    def split(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data using the selected strategy.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing SMILES and target values
            
        test_size : float
            Fraction of data for test set (0.0 to 1.0)
            
        val_size : float
            Fraction of data for validation set (0.0 to 1.0)
            
        random_state : int
            Random seed for reproducibility
            
        Returns
        -------
        train_idx : np.ndarray
            Indices for training set
            
        val_idx : np.ndarray
            Indices for validation set
            
        test_idx : np.ndarray
            Indices for test set
        """
        if self.strategy == 'scaffold':
            return self._scaffold_split(df, test_size, val_size, random_state)
        elif self.strategy == 'temporal':
            return self._temporal_split(df, test_size, val_size)
        elif self.strategy == 'cluster':
            return self._cluster_split(df, test_size, val_size, random_state)
    
    # ========================================================================
    # STRATEGY 1: Scaffold-Based Splitting (RECOMMENDED)
    # ========================================================================
    
    def _scaffold_split(
        self,
        df: pd.DataFrame,
        test_size: float,
        val_size: float,
        random_state: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split by Bemis-Murcko scaffolds.
        
        Entire scaffolds are assigned to train/val/test to prevent leakage.
        This is the RECOMMENDED strategy for most QSAR tasks.
        """
        print(f"\n🔬 Scaffold-Based Splitting")
        print(f"  Strategy: Bemis-Murcko scaffolds")
        print(f"  Entire scaffolds assigned to train/val/test")
        print(f"  Prevents scaffold leakage ✓")
        
        # Get scaffolds for all compounds
        scaffolds = {}
        scaffold_sets = defaultdict(list)
        
        for idx, smiles in enumerate(df[self.smiles_col]):
            scaffold = self._get_scaffold(smiles)
            scaffolds[idx] = scaffold
            scaffold_sets[scaffold].append(idx)
        
        n_scaffolds = len(scaffold_sets)
        print(f"\n  Total compounds: {len(df)}")
        print(f"  Unique scaffolds: {n_scaffolds}")
        
        # Shuffle scaffolds
        np.random.seed(random_state)
        scaffold_list = list(scaffold_sets.keys())
        np.random.shuffle(scaffold_list)
        
        # Assign scaffolds to sets
        n_total = len(df)
        n_test_target = int(n_total * test_size)
        n_val_target = int(n_total * val_size)
        
        test_idx = []
        val_idx = []
        train_idx = []
        
        for scaffold in scaffold_list:
            indices = scaffold_sets[scaffold]
            
            if len(test_idx) < n_test_target:
                test_idx.extend(indices)
            elif len(val_idx) < n_val_target:
                val_idx.extend(indices)
            else:
                train_idx.extend(indices)
        
        # Convert to numpy arrays
        train_idx = np.array(train_idx)
        val_idx = np.array(val_idx)
        test_idx = np.array(test_idx)
        
        # Get unique scaffolds in each set
        train_scaffolds = set(scaffolds[i] for i in train_idx)
        test_scaffolds = set(scaffolds[i] for i in test_idx)
        overlap = train_scaffolds & test_scaffolds
        
        print(f"\n  Split sizes:")
        print(f"    Train: {len(train_idx)} compounds ({len(train_scaffolds)} scaffolds)")
        print(f"    Val:   {len(val_idx)} compounds")
        print(f"    Test:  {len(test_idx)} compounds ({len(test_scaffolds)} scaffolds)")
        print(f"\n  Scaffold overlap: {len(overlap)} ✓ {'(ZERO - No leakage!)' if len(overlap) == 0 else '⚠️  WARNING: Overlap detected!'}")
        
        return train_idx, val_idx, test_idx
    
    def _get_scaffold(self, smiles: str) -> str:
        """Get Bemis-Murcko scaffold for a SMILES string."""
        if smiles in self.scaffold_dict:
            return self.scaffold_dict[smiles]
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                scaffold = smiles  # Fallback to original SMILES
            else:
                scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
            
            self.scaffold_dict[smiles] = scaffold
            return scaffold
        except:
            self.scaffold_dict[smiles] = smiles
            return smiles
    
    def get_scaffold(self, smiles: str) -> str:
        """Public method to get scaffold for a single SMILES."""
        return self._get_scaffold(smiles)
    
    # ========================================================================
    # STRATEGY 2: Time-Based Splitting
    # ========================================================================
    
    def _temporal_split(
        self,
        df: pd.DataFrame,
        test_size: float,
        val_size: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split by date/time.
        
        Train on older compounds, test on newer compounds.
        Simulates realistic deployment scenario.
        """
        print(f"\n📅 Time-Based Splitting")
        print(f"  Strategy: Temporal ordering")
        print(f"  Train on older → Test on newer")
        print(f"  Simulates realistic deployment ✓")
        
        if self.date_col not in df.columns:
            raise ValueError(f"Date column '{self.date_col}' not found in DataFrame")
        
        # Sort by date
        df_sorted = df.sort_values(by=self.date_col).reset_index(drop=True)
        
        n_total = len(df_sorted)
        n_test = int(n_total * test_size)
        n_val = int(n_total * val_size)
        n_train = n_total - n_test - n_val
        
        # Assign indices (oldest to newest)
        train_idx = np.arange(0, n_train)
        val_idx = np.arange(n_train, n_train + n_val)
        test_idx = np.arange(n_train + n_val, n_total)
        
        # Get date ranges
        train_dates = df_sorted.iloc[train_idx][self.date_col]
        test_dates = df_sorted.iloc[test_idx][self.date_col]
        
        print(f"\n  Split sizes:")
        print(f"    Train: {len(train_idx)} compounds")
        print(f"    Val:   {len(val_idx)} compounds")
        print(f"    Test:  {len(test_idx)} compounds")
        print(f"\n  Date ranges:")
        print(f"    Train: {train_dates.min()} to {train_dates.max()}")
        print(f"    Test:  {test_dates.min()} to {test_dates.max()}")
        
        # Map back to original DataFrame indices
        original_indices = df_sorted.index.values
        train_idx = original_indices[train_idx]
        val_idx = original_indices[val_idx]
        test_idx = original_indices[test_idx]
        
        return train_idx, val_idx, test_idx
    
    # ========================================================================
    # STRATEGY 3: Leave-Cluster-Out Splitting
    # ========================================================================
    
    def _cluster_split(
        self,
        df: pd.DataFrame,
        test_size: float,
        val_size: float,
        random_state: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split by molecular similarity clusters.
        
        Clusters compounds by fingerprint similarity, then holds out entire clusters.
        Good for small datasets and testing generalization to dissimilar compounds.
        """
        print(f"\n🔗 Leave-Cluster-Out Splitting")
        print(f"  Strategy: Fingerprint-based clustering")
        print(f"  Clusters: {self.n_clusters}")
        print(f"  Hold out entire clusters ✓")
        
        # Generate fingerprints
        print(f"\n  Generating fingerprints...")
        fps = []
        valid_indices = []
        
        for idx, smiles in enumerate(df[self.smiles_col]):
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                fps.append(np.array(fp))
                valid_indices.append(idx)
        
        fps = np.array(fps)
        valid_indices = np.array(valid_indices)
        
        print(f"  Valid compounds: {len(fps)}/{len(df)}")
        
        # Cluster compounds
        print(f"  Clustering compounds...")
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=random_state, n_init=10)
        clusters = kmeans.fit_predict(fps)
        
        # Organize by cluster
        cluster_dict = defaultdict(list)
        for idx, cluster_id in zip(valid_indices, clusters):
            cluster_dict[cluster_id].append(idx)
        
        # Shuffle clusters
        np.random.seed(random_state)
        cluster_ids = list(cluster_dict.keys())
        np.random.shuffle(cluster_ids)
        
        # Assign clusters to sets
        n_total = len(valid_indices)
        n_test_target = int(n_total * test_size)
        n_val_target = int(n_total * val_size)
        
        test_idx = []
        val_idx = []
        train_idx = []
        
        for cluster_id in cluster_ids:
            indices = cluster_dict[cluster_id]
            
            if len(test_idx) < n_test_target:
                test_idx.extend(indices)
            elif len(val_idx) < n_val_target:
                val_idx.extend(indices)
            else:
                train_idx.extend(indices)
        
        # Convert to numpy arrays
        train_idx = np.array(train_idx)
        val_idx = np.array(val_idx)
        test_idx = np.array(test_idx)
        
        # Get cluster distribution
        train_clusters = set(clusters[np.isin(valid_indices, train_idx)])
        test_clusters = set(clusters[np.isin(valid_indices, test_idx)])
        
        print(f"\n  Split sizes:")
        print(f"    Train: {len(train_idx)} compounds ({len(train_clusters)} clusters)")
        print(f"    Val:   {len(val_idx)} compounds")
        print(f"    Test:  {len(test_idx)} compounds ({len(test_clusters)} clusters)")
        print(f"\n  Cluster overlap: ZERO (by design) ✓")
        
        return train_idx, val_idx, test_idx
    
    # ========================================================================
    # Validation Methods
    # ========================================================================
    
    def check_scaffold_overlap(
        self,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        df: pd.DataFrame
    ) -> int:
        """
        Check for scaffold overlap between train and test sets.
        
        Only applicable for scaffold-based splitting.
        
        Returns
        -------
        int
            Number of overlapping compounds (should be 0)
        """
        if self.strategy != 'scaffold':
            print(f"⚠️  Scaffold overlap check not applicable for '{self.strategy}' strategy")
            return 0
        
        train_scaffolds = set()
        test_scaffolds = set()
        
        for idx in train_idx:
            smiles = df.iloc[idx][self.smiles_col]
            scaffold = self._get_scaffold(smiles)
            train_scaffolds.add(scaffold)
        
        for idx in test_idx:
            smiles = df.iloc[idx][self.smiles_col]
            scaffold = self._get_scaffold(smiles)
            test_scaffolds.add(scaffold)
        
        overlap_scaffolds = train_scaffolds & test_scaffolds
        
        if overlap_scaffolds:
            # Count compounds with overlapping scaffolds
            overlap_count = 0
            for idx in test_idx:
                smiles = df.iloc[idx][self.smiles_col]
                scaffold = self._get_scaffold(smiles)
                if scaffold in train_scaffolds:
                    overlap_count += 1
            return overlap_count
        
        return 0
    
    def get_split_info(self) -> Dict[str, str]:
        """Get information about the current splitting strategy."""
        info = {
            'strategy': self.strategy,
            'smiles_column': self.smiles_col
        }
        
        if self.strategy == 'temporal':
            info['date_column'] = self.date_col
        elif self.strategy == 'cluster':
            info['n_clusters'] = self.n_clusters
        
        return info


# ============================================================================
# Convenience Classes (Backward Compatibility)
# ============================================================================

class ScaffoldSplitter(AdvancedSplitter):
    """
    Scaffold-based splitter (backward compatibility).
    
    Convenience class that defaults to scaffold-based splitting.
    """
    def __init__(self, smiles_col: str = 'SMILES'):
        super().__init__(smiles_col=smiles_col, strategy='scaffold')


class TemporalSplitter(AdvancedSplitter):
    """
    Time-based splitter (convenience class).
    
    Train on older compounds, test on newer compounds.
    """
    def __init__(self, smiles_col: str = 'SMILES', date_col: str = 'Date'):
        super().__init__(smiles_col=smiles_col, strategy='temporal', date_col=date_col)


class ClusterSplitter(AdvancedSplitter):
    """
    Leave-cluster-out splitter (convenience class).
    
    Good for small datasets and testing generalization.
    """
    def __init__(self, smiles_col: str = 'SMILES', n_clusters: int = 5):
        super().__init__(smiles_col=smiles_col, strategy='cluster', n_clusters=n_clusters)
