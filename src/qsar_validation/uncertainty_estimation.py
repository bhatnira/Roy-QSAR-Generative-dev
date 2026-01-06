"""
Uncertainty Estimation Module
==============================

Mitigates:
8. Poor Uncertainty Estimation

Provides tools to:
- Estimate prediction uncertainty
- Calculate confidence intervals
- Assess applicability domain
- Flag out-of-domain predictions

Author: QSAR Validation Framework
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import NearestNeighbors
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs


class UncertaintyEstimator:
    """
    Estimate prediction uncertainty and applicability domain.
    
    Provides:
    - Ensemble-based uncertainty (Random Forest std)
    - Distance-based uncertainty (nearest neighbor distance)
    - Confidence intervals
    - Applicability domain flags
    
    Parameters
    ----------
    method : str
        'ensemble' or 'distance' or 'both'
    
    threshold_percentile : float
        Percentile for applicability domain (default 90)
    
    Examples
    --------
    >>> estimator = UncertaintyEstimator(method='both')
    >>> 
    >>> # Fit on training data
    >>> estimator.fit(X_train, y_train, model)
    >>> 
    >>> # Predict with uncertainty
    >>> predictions = estimator.predict_with_uncertainty(X_test)
    """
    
    def __init__(
        self,
        method: str = 'both',
        threshold_percentile: float = 90
    ):
        self.method = method
        self.threshold_percentile = threshold_percentile
        self.is_fitted = False
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        model = None
    ):
        """
        Fit uncertainty estimator on training data.
        
        Parameters
        ----------
        X_train : np.ndarray
            Training features
        
        y_train : np.ndarray
            Training targets
        
        model : sklearn model, optional
            Trained model (required for some methods)
        """
        print(f"\n{'='*80}")
        print("UNCERTAINTY ESTIMATOR - FIT")
        print(f"{'='*80}")
        
        self.X_train = X_train
        self.y_train = y_train
        self.model = model
        
        # Fit nearest neighbors for distance-based uncertainty
        if self.method in ['distance', 'both']:
            print("\nFitting nearest neighbors...")
            self.nn_model = NearestNeighbors(n_neighbors=5, metric='euclidean')
            self.nn_model.fit(X_train)
            
            # Calculate training set distances for threshold
            train_distances, _ = self.nn_model.kneighbors(X_train)
            self.train_mean_distances = train_distances.mean(axis=1)
            self.distance_threshold = np.percentile(
                self.train_mean_distances,
                self.threshold_percentile
            )
            print(f"  Applicability domain threshold: {self.distance_threshold:.4f}")
        
        self.is_fitted = True
        print("\n✓ Uncertainty estimator fitted")
    
    def predict_with_uncertainty(
        self,
        X_test: np.ndarray
    ) -> Dict:
        """
        Predict with uncertainty estimates.
        
        Parameters
        ----------
        X_test : np.ndarray
            Test features
        
        Returns
        -------
        results : dict
            Predictions with uncertainty estimates
        """
        if not self.is_fitted:
            raise RuntimeError("Call .fit() first!")
        
        print(f"\n{'='*80}")
        print("PREDICTION WITH UNCERTAINTY")
        print(f"{'='*80}")
        
        results = {
            'predictions': None,
            'uncertainty': None,
            'in_domain': None,
            'confidence_lower': None,
            'confidence_upper': None
        }
        
        # Ensemble-based uncertainty (Random Forest)
        if self.method in ['ensemble', 'both']:
            if isinstance(self.model, RandomForestRegressor):
                print("\nCalculating ensemble uncertainty...")
                
                # Get predictions from each tree
                tree_predictions = np.array([
                    tree.predict(X_test) for tree in self.model.estimators_
                ])
                
                # Mean and std across trees
                predictions = tree_predictions.mean(axis=0)
                uncertainty = tree_predictions.std(axis=0)
                
                results['predictions'] = predictions
                results['uncertainty'] = uncertainty
                
                # 95% confidence intervals
                results['confidence_lower'] = predictions - 1.96 * uncertainty
                results['confidence_upper'] = predictions + 1.96 * uncertainty
                
                print(f"  Mean uncertainty: {uncertainty.mean():.4f}")
                print(f"  Max uncertainty: {uncertainty.max():.4f}")
            else:
                print("\n⚠️  Ensemble uncertainty requires Random Forest model")
        
        # Distance-based uncertainty (applicability domain)
        if self.method in ['distance', 'both']:
            print("\nCalculating distance-based uncertainty...")
            
            distances, _ = self.nn_model.kneighbors(X_test)
            mean_distances = distances.mean(axis=1)
            
            # Flag molecules outside applicability domain
            in_domain = mean_distances <= self.distance_threshold
            
            results['in_domain'] = in_domain
            results['distances'] = mean_distances
            
            n_out = (~in_domain).sum()
            pct_out = (n_out / len(X_test)) * 100
            
            print(f"  Molecules in domain: {in_domain.sum()}/{len(X_test)} ({100-pct_out:.1f}%)")
            print(f"  Molecules OUT of domain: {n_out} ({pct_out:.1f}%)")
            
            if n_out > 0:
                print(f"\n  ⚠️  {n_out} predictions are outside training domain!")
                print(f"     → These predictions may be unreliable")
        
        return results
    
    def analyze_prediction_confidence(
        self,
        results: Dict,
        y_true: Optional[np.ndarray] = None
    ):
        """
        Analyze prediction confidence.
        
        Parameters
        ----------
        results : dict
            Output from predict_with_uncertainty()
        
        y_true : np.ndarray, optional
            True values (for validation)
        """
        print(f"\n{'='*80}")
        print("PREDICTION CONFIDENCE ANALYSIS")
        print(f"{'='*80}")
        
        if results['uncertainty'] is not None:
            uncertainty = results['uncertainty']
            
            print(f"\nUncertainty Statistics:")
            print(f"  Mean: {uncertainty.mean():.4f}")
            print(f"  Median: {np.median(uncertainty):.4f}")
            print(f"  25th percentile: {np.percentile(uncertainty, 25):.4f}")
            print(f"  75th percentile: {np.percentile(uncertainty, 75):.4f}")
            print(f"  Max: {uncertainty.max():.4f}")
            
            # Flag high uncertainty predictions
            high_uncertainty_threshold = np.percentile(uncertainty, 75)
            high_uncertainty = uncertainty > high_uncertainty_threshold
            n_high = high_uncertainty.sum()
            
            print(f"\n  {n_high} predictions have high uncertainty (>75th percentile)")
            
            # If true values available, check calibration
            if y_true is not None:
                predictions = results['predictions']
                errors = np.abs(predictions - y_true)
                
                # Correlation between uncertainty and error
                from scipy.stats import spearmanr
                corr, p_value = spearmanr(uncertainty, errors)
                
                print(f"\n  Uncertainty vs Error correlation: {corr:.3f} (p={p_value:.4f})")
                
                if corr > 0.3 and p_value < 0.05:
                    print(f"  ✓ Uncertainty is well-calibrated")
                else:
                    print(f"  ⚠️  Uncertainty may not be well-calibrated")
        
        if results['in_domain'] is not None:
            in_domain = results['in_domain']
            
            print(f"\n{'='*80}")
            print("APPLICABILITY DOMAIN")
            print(f"{'='*80}")
            
            n_in = in_domain.sum()
            n_out = (~in_domain).sum()
            
            print(f"\nIn domain: {n_in} ({n_in/len(in_domain)*100:.1f}%)")
            print(f"Out of domain: {n_out} ({n_out/len(in_domain)*100:.1f}%)")
            
            if y_true is not None and results['predictions'] is not None:
                predictions = results['predictions']
                
                # Compare errors for in-domain vs out-of-domain
                errors_in = np.abs(predictions[in_domain] - y_true[in_domain])
                errors_out = np.abs(predictions[~in_domain] - y_true[~in_domain]) if n_out > 0 else []
                
                print(f"\nMean Absolute Error:")
                print(f"  In domain: {errors_in.mean():.4f}")
                if len(errors_out) > 0:
                    print(f"  Out of domain: {errors_out.mean():.4f}")
                    
                    if errors_out.mean() > errors_in.mean() * 1.5:
                        print(f"\n  ⚠️  Out-of-domain predictions are significantly worse!")


def demonstrate_uncertainty_estimation():
    """Demonstrate uncertainty estimation."""
    print("\n" + "="*80)
    print("UNCERTAINTY ESTIMATION DEMONSTRATION")
    print("="*80)
    
    # Generate synthetic data
    np.random.seed(42)
    X_train = np.random.randn(200, 50)
    y_train = X_train[:, 0] + 0.5 * X_train[:, 1] + np.random.randn(200) * 0.5
    
    X_test_in = np.random.randn(50, 50)  # In domain
    X_test_out = np.random.randn(20, 50) * 3 + 5  # Out of domain!
    X_test = np.vstack([X_test_in, X_test_out])
    
    y_test_in = X_test_in[:, 0] + 0.5 * X_test_in[:, 1] + np.random.randn(50) * 0.5
    y_test_out = X_test_out[:, 0] + 0.5 * X_test_out[:, 1] + np.random.randn(20) * 0.5
    y_test = np.hstack([y_test_in, y_test_out])
    
    # Train Random Forest
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Create uncertainty estimator
    estimator = UncertaintyEstimator(method='both', threshold_percentile=90)
    estimator.fit(X_train, y_train, model)
    
    # Predict with uncertainty
    results = estimator.predict_with_uncertainty(X_test)
    
    # Analyze confidence
    estimator.analyze_prediction_confidence(results, y_test)
    
    print("\n" + "="*80)
    print("✓ Demonstration complete!")
    print("="*80)


if __name__ == '__main__':
    demonstrate_uncertainty_estimation()
