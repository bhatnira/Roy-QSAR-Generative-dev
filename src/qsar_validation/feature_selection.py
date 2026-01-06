"""
Feature Selection Module (Data Leakage Prevention)
==================================================

Provides proper feature selection that prevents data leakage:

CRITICAL RULES:
1. Feature selection → NESTED CV ONLY
2. Selection must happen within each CV fold
3. Never select features on validation/test data
4. Never select features using all data before CV

Supported methods:
- Variance threshold (remove low-variance features)
- Correlation-based (remove highly correlated features)
- Model-based (e.g., feature importance from Random Forest)
- Univariate (e.g., f_regression, mutual_info)

Author: QSAR Validation Framework
License: MIT
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Literal
from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    f_regression,
    mutual_info_regression,
    RFE
)
from sklearn.ensemble import RandomForestRegressor


class FeatureSelector:
    """
    Safe feature selection that prevents data leakage.
    
    CRITICAL: Feature selection must be done in NESTED CV!
    
    For proper cross-validation workflow:
    1. Split data into train/test
    2. Within training set, use nested CV
    3. In each CV fold:
        a. Select features on CV train fold
        b. Apply selection to CV val fold
        c. Train model and evaluate
    
    Parameters
    ----------
    method : str
        Selection method:
        - 'variance': Remove low-variance features
        - 'correlation': Remove highly correlated features
        - 'model_based': Use model feature importance
        - 'univariate': Statistical tests (f_regression)
        
    n_features : int or float, optional
        Number of features to select:
        - int: Exact number (e.g., 50)
        - float: Fraction (e.g., 0.5 for 50%)
        - None: Automatic selection
    
    Examples
    --------
    CORRECT Usage (Nested CV):
    >>> selector = FeatureSelector(method='variance', n_features=100)
    >>> 
    >>> # For each CV fold:
    >>> for train_idx, val_idx in cv_folds:
    >>>     X_train_fold = X_train[train_idx]
    >>>     
    >>>     # Fit selector on THIS fold's training data
    >>>     selector.fit(X_train_fold, y_train[train_idx])
    >>>     
    >>>     # Transform THIS fold's train and val
    >>>     X_train_selected = selector.transform(X_train_fold)
    >>>     X_val_selected = selector.transform(X_train[val_idx])
    >>>     
    >>>     # Train model on selected features
    >>>     model.fit(X_train_selected, y_train[train_idx])
    
    INCORRECT Usage (Data Leakage!):
    >>> # WRONG: Selecting features before CV
    >>> selector.fit(X_train)  # ❌ Uses all training data
    >>> X_selected = selector.transform(X_train)
    >>> cv_score = cross_val_score(model, X_selected, y)  # ❌ LEAKAGE!
    """
    
    def __init__(
        self,
        method: Literal['variance', 'correlation', 'model_based', 'univariate'] = 'variance',
        n_features: Optional[int] = None,
        threshold: float = 0.01
    ):
        self.method = method.lower()
        self.n_features = n_features
        self.threshold = threshold
        
        # Validate method
        valid_methods = ['variance', 'correlation', 'model_based', 'univariate']
        if self.method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}, got '{self.method}'")
        
        self.selector = None
        self.is_fitted = False
        self.selected_features = None
        self.n_features_in = None
        self.n_features_out = None
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'FeatureSelector':
        """
        Fit feature selector on FOLD TRAINING DATA ONLY.
        
        ⚠️  CRITICAL: Only call this within CV fold on fold's training data!
        
        Parameters
        ----------
        X : np.ndarray
            Features from CURRENT CV FOLD training set
            
        y : np.ndarray, optional
            Target values (required for some methods)
        
        Returns
        -------
        self : FeatureSelector
            Fitted selector
        """
        self.n_features_in = X.shape[1]
        
        print(f"\n🔍 Fitting {self.method.upper()} feature selector")
        print(f"  Input features: {self.n_features_in}")
        print(f"  ⚠️  Remember: Use in NESTED CV to prevent leakage!")
        
        if self.method == 'variance':
            self._fit_variance(X)
        elif self.method == 'correlation':
            self._fit_correlation(X)
        elif self.method == 'model_based':
            if y is None:
                raise ValueError("y required for model-based selection")
            self._fit_model_based(X, y)
        elif self.method == 'univariate':
            if y is None:
                raise ValueError("y required for univariate selection")
            self._fit_univariate(X, y)
        
        self.is_fitted = True
        self.n_features_out = len(self.selected_features)
        
        print(f"  Selected features: {self.n_features_out}")
        print(f"  Reduction: {self.n_features_in} → {self.n_features_out} ({self.n_features_out/self.n_features_in*100:.1f}%)")
        
        return self
    
    def _fit_variance(self, X: np.ndarray):
        """Remove low-variance features."""
        selector = VarianceThreshold(threshold=self.threshold)
        selector.fit(X)
        self.selected_features = np.where(selector.get_support())[0]
        self.selector = selector
    
    def _fit_correlation(self, X: np.ndarray):
        """Remove highly correlated features."""
        # Compute correlation matrix
        corr_matrix = np.corrcoef(X.T)
        
        # Find highly correlated pairs
        upper_tri = np.triu(np.ones_like(corr_matrix), k=1).astype(bool)
        to_drop = set()
        
        for i in range(len(corr_matrix)):
            for j in range(i + 1, len(corr_matrix)):
                if abs(corr_matrix[i, j]) > 0.95:  # High correlation threshold
                    to_drop.add(j)
        
        # Keep features not in drop list
        self.selected_features = np.array([i for i in range(X.shape[1]) if i not in to_drop])
    
    def _fit_model_based(self, X: np.ndarray, y: np.ndarray):
        """Select features using Random Forest importance."""
        if self.n_features is None:
            k = int(X.shape[1] * 0.5)  # Select 50% by default
        elif isinstance(self.n_features, float):
            k = int(X.shape[1] * self.n_features)
        else:
            k = min(self.n_features, X.shape[1])
        
        # Train Random Forest
        rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        # Get feature importances
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1][:k]
        
        self.selected_features = np.sort(indices)
    
    def _fit_univariate(self, X: np.ndarray, y: np.ndarray):
        """Select features using univariate statistical tests."""
        if self.n_features is None:
            k = int(X.shape[1] * 0.5)
        elif isinstance(self.n_features, float):
            k = int(X.shape[1] * self.n_features)
        else:
            k = min(self.n_features, X.shape[1])
        
        selector = SelectKBest(score_func=f_regression, k=k)
        selector.fit(X, y)
        
        self.selected_features = np.where(selector.get_support())[0]
        self.selector = selector
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform features by selecting only fitted features.
        
        Parameters
        ----------
        X : np.ndarray
            Features to transform
        
        Returns
        -------
        X_selected : np.ndarray
            Selected features only
        """
        if not self.is_fitted:
            raise RuntimeError("Selector not fitted! Call .fit() first.")
        
        if X.shape[1] != self.n_features_in:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.n_features_in}, got {X.shape[1]}"
            )
        
        return X[:, self.selected_features]
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Fit selector and transform features.
        
        ⚠️  Use ONLY within CV fold on fold's training data!
        """
        self.fit(X, y)
        return self.transform(X)
    
    def get_selected_feature_indices(self) -> np.ndarray:
        """Get indices of selected features."""
        if not self.is_fitted:
            raise RuntimeError("Selector not fitted!")
        return self.selected_features


class NestedCVFeatureSelector:
    """
    Helper class for proper nested CV with feature selection.
    
    This ensures feature selection happens correctly within each CV fold.
    
    Example
    -------
    >>> cv_selector = NestedCVFeatureSelector(
    >>>     selector_method='variance',
    >>>     n_features=100,
    >>>     n_folds=5
    >>> )
    >>> 
    >>> cv_scores = cv_selector.nested_cross_validate(X_train, y_train, model)
    """
    
    def __init__(
        self,
        selector_method: str = 'variance',
        n_features: Optional[int] = None,
        n_folds: int = 5,
        random_state: int = 42
    ):
        self.selector_method = selector_method
        self.n_features = n_features
        self.n_folds = n_folds
        self.random_state = random_state
    
    def nested_cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model,
        metric: str = 'r2'
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Perform nested cross-validation with feature selection.
        
        In each fold:
        1. Select features on fold's training data
        2. Transform fold's validation data
        3. Train model and evaluate
        
        Parameters
        ----------
        X : np.ndarray
            Training features
            
        y : np.ndarray
            Training targets
            
        model : sklearn model
            Model to evaluate
            
        metric : str
            Evaluation metric ('r2', 'rmse', 'mae')
        
        Returns
        -------
        scores : np.ndarray
            CV scores for each fold
            
        selected_features_per_fold : list
            Selected feature indices for each fold
        """
        from sklearn.model_selection import KFold
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        
        print(f"\n🔁 Nested CV with Feature Selection")
        print(f"  Method: {self.selector_method}")
        print(f"  Folds: {self.n_folds}")
        print(f"  Features: {X.shape[1]}")
        
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        scores = []
        selected_features_list = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            print(f"\n  Fold {fold_idx + 1}/{self.n_folds}")
            
            # Get fold data
            X_train_fold = X[train_idx]
            y_train_fold = y[train_idx]
            X_val_fold = X[val_idx]
            y_val_fold = y[val_idx]
            
            # CRITICAL: Select features on THIS fold's training data only
            selector = FeatureSelector(method=self.selector_method, n_features=self.n_features)
            selector.fit(X_train_fold, y_train_fold)
            
            # Transform using selected features
            X_train_selected = selector.transform(X_train_fold)
            X_val_selected = selector.transform(X_val_fold)
            
            # Train model
            model_fold = model.__class__(**model.get_params())  # Clone model
            model_fold.fit(X_train_selected, y_train_fold)
            
            # Predict
            y_pred = model_fold.predict(X_val_selected)
            
            # Calculate metric
            if metric == 'r2':
                score = r2_score(y_val_fold, y_pred)
            elif metric == 'rmse':
                score = np.sqrt(mean_squared_error(y_val_fold, y_pred))
            elif metric == 'mae':
                score = mean_absolute_error(y_val_fold, y_pred)
            
            scores.append(score)
            selected_features_list.append(selector.get_selected_feature_indices())
            
            print(f"    Selected: {len(selector.selected_features)} features")
            print(f"    {metric.upper()}: {score:.4f}")
        
        scores = np.array(scores)
        
        print(f"\n  ✓ Nested CV complete")
        print(f"  Mean {metric.upper()}: {scores.mean():.4f} ± {scores.std():.4f}")
        
        return scores, selected_features_list


def demonstrate_correct_nested_cv():
    """Demonstrate CORRECT feature selection in nested CV."""
    print("\n" + "="*80)
    print("✅ CORRECT: Feature Selection in Nested CV")
    print("="*80)
    
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import KFold
    
    # Simulate data
    np.random.seed(42)
    X = np.random.rand(100, 200)  # 200 features
    y = np.random.rand(100)
    
    print(f"\nData: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Setup
    model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    print("\n🔁 Starting Nested CV...")
    
    fold_scores = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\nFold {fold_idx + 1}/5")
        
        # Get fold data
        X_train_fold = X[train_idx]
        y_train_fold = y[train_idx]
        X_val_fold = X[val_idx]
        y_val_fold = y[val_idx]
        
        # CRITICAL: Fit selector on THIS fold's training data
        selector = FeatureSelector(method='variance', n_features=50)
        selector.fit(X_train_fold)
        
        # Transform both train and val for this fold
        X_train_selected = selector.transform(X_train_fold)
        X_val_selected = selector.transform(X_val_fold)
        
        print(f"  Selected: {X_train_selected.shape[1]} features")
        
        # Train model on selected features
        model.fit(X_train_selected, y_train_fold)
        
        # Evaluate
        from sklearn.metrics import r2_score
        y_pred = model.predict(X_val_selected)
        score = r2_score(y_val_fold, y_pred)
        fold_scores.append(score)
        
        print(f"  R² score: {score:.4f}")
    
    print(f"\n✅ Nested CV complete!")
    print(f"Mean R²: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
    print(f"\n✓ No data leakage - features selected within each fold!")


def demonstrate_incorrect_usage():
    """Demonstrate INCORRECT feature selection (causes leakage)."""
    print("\n" + "="*80)
    print("❌ INCORRECT: Feature Selection Before CV (DATA LEAKAGE!)")
    print("="*80)
    
    print("\n❌ WRONG APPROACH: Selecting features before CV")
    print("-"*80)
    print("""
# This causes DATA LEAKAGE!

# WRONG: Select features on all training data
selector = FeatureSelector()
selector.fit(X_train)  # ❌ Uses ALL training data
X_selected = selector.transform(X_train)

# WRONG: Then do CV on pre-selected features
cv_scores = cross_val_score(model, X_selected, y_train)  # ❌ LEAKAGE!

Why wrong:
- Feature selection sees all training data
- CV folds don't independently select features
- Model sees information from validation folds
- Overestimates performance
    """)
    
    print("\n✅ CORRECT APPROACH:")
    print("-"*80)
    print("""
# Feature selection WITHIN each CV fold

for train_idx, val_idx in cv_folds:
    # Fit selector on THIS fold's training data only
    selector.fit(X_train[train_idx])  # ✓ Only this fold
    
    # Transform this fold
    X_train_selected = selector.transform(X_train[train_idx])
    X_val_selected = selector.transform(X_train[val_idx])
    
    # Train and evaluate
    model.fit(X_train_selected, y_train[train_idx])
    score = model.score(X_val_selected, y_train[val_idx])
    """)


if __name__ == '__main__':
    demonstrate_correct_nested_cv()
    print("\n" + "="*80 + "\n")
    demonstrate_incorrect_usage()
    print("\n" + "="*80 + "\n")
