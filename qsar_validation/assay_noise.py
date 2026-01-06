"""
Assay Noise Estimation Module
==============================

Estimates experimental/assay noise levels to contextualize model performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List


class AssayNoiseEstimator:
    """
    Estimate and report experimental/assay noise levels.
    Helps contextualize model performance against achievable limits.
    """
    
    @staticmethod
    def estimate_experimental_error(df: pd.DataFrame, target_col: str,
                                    replicate_groups: Optional[List] = None) -> Dict:
        """
        Estimate experimental error from replicates or use literature values.
        
        Args:
            df: DataFrame containing target values
            target_col: Name of target column
            replicate_groups: Optional list/array of replicate group identifiers
            
        Returns:
            Dictionary with error estimates and metadata
        """
        print("\n🔬 EXPERIMENTAL ERROR ESTIMATION")
        print("=" * 70)
        
        # Try to calculate from replicates if available
        if replicate_groups is not None and len(replicate_groups) > 0:
            result = AssayNoiseEstimator._estimate_from_replicates(
                df, target_col, replicate_groups
            )
            if result is not None:
                return result
        
        # Fall back to literature estimates
        return AssayNoiseEstimator._use_literature_estimates()
    
    @staticmethod
    def _estimate_from_replicates(df: pd.DataFrame, target_col: str, 
                                 replicate_groups: List) -> Optional[Dict]:
        """Estimate error from replicate measurements."""
        errors = []
        
        for group in set(replicate_groups):
            group_values = df[replicate_groups == group][target_col].values
            if len(group_values) > 1:
                errors.append(np.std(group_values))
        
        if errors:
            mean_error = np.mean(errors)
            print(f"✓ Estimated from {len(errors)} replicate groups")
            print(f"   Mean replicate error: {mean_error:.4f}")
            
            AssayNoiseEstimator._print_implications(mean_error)
            
            return {
                'experimental_error': mean_error,
                'source': 'replicates',
                'n_groups': len(errors)
            }
        
        return None
    
    @staticmethod
    def _use_literature_estimates() -> Dict:
        """Use typical literature estimates for assay error."""
        print("📚 Using typical IC₅₀/EC₅₀ assay error estimates:")
        print("   Typical experimental error: 0.3 - 0.6 log units")
        print("   Conservative estimate: 0.5 log units")
        
        AssayNoiseEstimator._print_implications(0.5)
        
        return {
            'experimental_error': 0.5,
            'source': 'literature',
            'range': (0.3, 0.6)
        }
    
    @staticmethod
    def _print_implications(error: float) -> None:
        """Print implications of experimental error."""
        print("\n📌 IMPLICATION:")
        if error <= 0.3:
            print(f"   RMSE < {error:.1f} may indicate overfitting or lucky split")
        else:
            print(f"   RMSE < {error:.1f} may indicate overfitting or lucky split")
        print(f"   RMSE ≈ {error:.1f} is near theoretical limit")
        print("   Report model error relative to assay precision")
