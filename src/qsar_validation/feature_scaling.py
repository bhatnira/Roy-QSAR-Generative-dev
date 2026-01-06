"""
Feature Scaling Module (Data Leakage Prevention)
================================================

Provides proper feature scaling that prevents data leakage:

CRITICAL RULES:
1. StandardScaler / MinMaxScaler → fit on TRAIN ONLY
2. Transform train, val, and test using train statistics
3. Never fit on validation or test data
4. Never fit on combined train+val+test data

Supported scalers:
- StandardScaler (mean=0, std=1) - RECOMMENDED for most cases
- MinMaxScaler (range [0,1]) - Good for bounded algorithms
- RobustScaler (uses median/IQR) - Good for outliers

Author: QSAR Validation Framework
License: MIT
"""

import numpy as np
import pandas as pd
from typing import Optional, Literal
from sklearn.preprocessing import StandardScaler as SKStandardScaler
from sklearn.preprocessing import MinMaxScaler as SKMinMaxScaler
from sklearn.preprocessing import RobustScaler as SKRobustScaler


class FeatureScaler:
    """
    Safe feature scaling that prevents data leakage.
    
    CRITICAL: Always fit on training data only, then transform all sets.
    
    Parameters
    ----------
    method : str
        Scaling method: 'standard', 'minmax', or 'robust'
        
        - 'standard': StandardScaler (mean=0, std=1) - RECOMMENDED
        - 'minmax': MinMaxScaler (range [0,1])
        - 'robust': RobustScaler (uses median/IQR, robust to outliers)
    
    Examples
    --------
    CORRECT Usage (No Data Leakage):
    >>> scaler = FeatureScaler(method='standard')
    >>> scaler.fit(X_train)  # Fit on TRAIN ONLY!
    >>> X_train_scaled = scaler.transform(X_train)
    >>> X_val_scaled = scaler.transform(X_val)
    >>> X_test_scaled = scaler.transform(X_test)
    
    Or using fit_transform:
    >>> scaler = FeatureScaler(method='standard')
    >>> X_train_scaled = scaler.fit_transform(X_train)  # Fit AND transform train
    >>> X_test_scaled = scaler.transform(X_test)         # Only transform test
    
    INCORRECT Usage (Data Leakage!):
    >>> # WRONG: Fitting on all data
    >>> scaler.fit(np.vstack([X_train, X_test]))  # ❌ LEAKAGE!
    >>> 
    >>> # WRONG: Fitting on test data
    >>> scaler.fit(X_test)  # ❌ LEAKAGE!
    """
    
    def __init__(self, method: Literal['standard', 'minmax', 'robust'] = 'standard'):
        self.method = method.lower()
        
        # Validate method
        valid_methods = ['standard', 'minmax', 'robust']
        if self.method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}, got '{self.method}'")
        
        # Initialize scaler
        if self.method == 'standard':
            self.scaler = SKStandardScaler()
        elif self.method == 'minmax':
            self.scaler = SKMinMaxScaler()
        elif self.method == 'robust':
            self.scaler = SKRobustScaler()
        
        self.is_fitted = False
        self.feature_names = None
        self.n_features = None
    
    def fit(self, X: np.ndarray, feature_names: Optional[list] = None) -> 'FeatureScaler':
        """
        Fit scaler on TRAINING DATA ONLY.
        
        ⚠️  CRITICAL: Only call this on training data!
        
        Parameters
        ----------
        X : np.ndarray
            Training features (n_samples, n_features)
            
        feature_names : list, optional
            Names of features (for logging)
        
        Returns
        -------
        self : FeatureScaler
            Fitted scaler
        """
        if self.is_fitted:
            print("⚠️  Warning: Scaler already fitted. Refitting on new data...")
        
        print(f"\n🔧 Fitting {self.method.upper()} scaler")
        print(f"  Data shape: {X.shape}")
        print(f"  ✓ Fitting on TRAIN data only (no leakage)")
        
        # Fit scaler
        self.scaler.fit(X)
        self.is_fitted = True
        self.feature_names = feature_names
        self.n_features = X.shape[1]
        
        # Log statistics
        if self.method == 'standard':
            print(f"  Train mean: {X.mean():.3f}")
            print(f"  Train std:  {X.std():.3f}")
        elif self.method == 'minmax':
            print(f"  Train min: {X.min():.3f}")
            print(f"  Train max: {X.max():.3f}")
        elif self.method == 'robust':
            print(f"  Train median: {np.median(X):.3f}")
        
        return self
    
    def transform(self, X: np.ndarray, set_name: str = 'Data') -> np.ndarray:
        """
        Transform features using TRAIN statistics.
        
        Use this to transform validation and test sets.
        
        Parameters
        ----------
        X : np.ndarray
            Features to transform (n_samples, n_features)
            
        set_name : str
            Name of dataset (for logging): 'Train', 'Val', 'Test'
        
        Returns
        -------
        X_scaled : np.ndarray
            Scaled features
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler not fitted! Call .fit() first on training data.")
        
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.n_features}, got {X.shape[1]}"
            )
        
        print(f"  Transforming {set_name}: {X.shape}")
        
        X_scaled = self.scaler.transform(X)
        
        # Log transformed statistics
        print(f"    Scaled mean: {X_scaled.mean():.3f}, std: {X_scaled.std():.3f}")
        
        return X_scaled
    
    def fit_transform(self, X: np.ndarray, feature_names: Optional[list] = None) -> np.ndarray:
        """
        Fit on training data AND transform it.
        
        ⚠️  Use ONLY on training data!
        
        Parameters
        ----------
        X : np.ndarray
            Training features (n_samples, n_features)
            
        feature_names : list, optional
            Names of features
        
        Returns
        -------
        X_scaled : np.ndarray
            Scaled training features
        """
        self.fit(X, feature_names)
        return self.transform(X, set_name='Train')
    
    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled features back to original scale.
        
        Parameters
        ----------
        X_scaled : np.ndarray
            Scaled features
        
        Returns
        -------
        X_original : np.ndarray
            Features in original scale
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler not fitted!")
        
        return self.scaler.inverse_transform(X_scaled)
    
    def get_params(self) -> dict:
        """Get scaler parameters."""
        if not self.is_fitted:
            return {'fitted': False}
        
        params = {
            'method': self.method,
            'fitted': True,
            'n_features': self.n_features
        }
        
        if self.method == 'standard':
            params['mean'] = self.scaler.mean_
            params['std'] = self.scaler.scale_
        elif self.method == 'minmax':
            params['min'] = self.scaler.data_min_
            params['max'] = self.scaler.data_max_
        elif self.method == 'robust':
            params['center'] = self.scaler.center_
            params['scale'] = self.scaler.scale_
        
        return params


def demonstrate_correct_usage():
    """
    Demonstrate CORRECT usage of FeatureScaler (no data leakage).
    """
    print("\n" + "="*80)
    print("CORRECT USAGE: Feature Scaling Without Data Leakage")
    print("="*80)
    
    # Simulate data
    np.random.seed(42)
    X_train = np.random.rand(100, 50)
    X_val = np.random.rand(30, 50)
    X_test = np.random.rand(20, 50)
    
    print(f"\n📊 Data:")
    print(f"  Train: {X_train.shape}")
    print(f"  Val:   {X_val.shape}")
    print(f"  Test:  {X_test.shape}")
    
    # CORRECT: Fit on train only
    print("\n✅ CORRECT APPROACH:")
    print("="*80)
    
    scaler = FeatureScaler(method='standard')
    
    # Step 1: Fit on TRAIN ONLY
    print("\nStep 1: Fit on TRAIN ONLY")
    scaler.fit(X_train)
    
    # Step 2: Transform all sets using TRAIN statistics
    print("\nStep 2: Transform all sets using TRAIN statistics")
    X_train_scaled = scaler.transform(X_train, set_name='Train')
    X_val_scaled = scaler.transform(X_val, set_name='Val')
    X_test_scaled = scaler.transform(X_test, set_name='Test')
    
    print("\n✓ No data leakage!")
    print("✓ Val and Test scaled using Train statistics only")
    
    # Verify
    print("\n📈 Verification:")
    print(f"  Train scaled: mean={X_train_scaled.mean():.3f}, std={X_train_scaled.std():.3f}")
    print(f"  Val scaled:   mean={X_val_scaled.mean():.3f}, std={X_val_scaled.std():.3f}")
    print(f"  Test scaled:  mean={X_test_scaled.mean():.3f}, std={X_test_scaled.std():.3f}")
    
    # Alternative: fit_transform
    print("\n" + "="*80)
    print("ALTERNATIVE: Using fit_transform()")
    print("="*80)
    
    scaler2 = FeatureScaler(method='standard')
    X_train_scaled = scaler2.fit_transform(X_train)  # Fit AND transform
    X_test_scaled = scaler2.transform(X_test, set_name='Test')  # Only transform
    
    print("\n✓ Same result, more concise!")


def demonstrate_incorrect_usage():
    """
    Demonstrate INCORRECT usage that causes data leakage (DON'T DO THIS!).
    """
    print("\n" + "="*80)
    print("❌ INCORRECT USAGE: Data Leakage Examples (DON'T DO THIS!)")
    print("="*80)
    
    np.random.seed(42)
    X_train = np.random.rand(100, 50)
    X_test = np.random.rand(20, 50)
    
    print("\n❌ WRONG APPROACH 1: Fitting on all data")
    print("-"*80)
    print("# This causes DATA LEAKAGE!")
    print("scaler.fit(np.vstack([X_train, X_test]))  # ❌ LEAKAGE!")
    print("\nWhy wrong: Test data statistics influence the scaling")
    
    print("\n❌ WRONG APPROACH 2: Fitting on test data")
    print("-"*80)
    print("# This causes DATA LEAKAGE!")
    print("scaler.fit(X_test)  # ❌ LEAKAGE!")
    print("\nWhy wrong: Using test data for any fitting is leakage")
    
    print("\n❌ WRONG APPROACH 3: Scaling before splitting")
    print("-"*80)
    print("# This causes DATA LEAKAGE!")
    print("X_all_scaled = scaler.fit_transform(X_all)  # ❌ LEAKAGE!")
    print("train, test = split(X_all_scaled)")
    print("\nWhy wrong: Train/test split happens AFTER seeing all data")
    
    print("\n💡 REMEMBER: Always fit on TRAIN ONLY, then transform all sets!")


if __name__ == '__main__':
    demonstrate_correct_usage()
    print("\n" + "="*80 + "\n")
    demonstrate_incorrect_usage()
    print("\n" + "="*80 + "\n")
