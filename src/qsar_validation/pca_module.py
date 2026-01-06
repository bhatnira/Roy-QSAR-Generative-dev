"""
PCA Module (Data Leakage Prevention)
====================================

Provides proper PCA dimensionality reduction that prevents data leakage:

CRITICAL RULES:
1. PCA → fit on TRAIN FOLD ONLY
2. Never fit PCA on validation or test data
3. Never fit PCA on all data before CV
4. Fit PCA within each CV fold

PCA must be fitted on training data only, then applied to validation/test data.
This ensures that the principal components are learned from training data and
don't "see" the validation/test data distribution.

Author: QSAR Validation Framework
License: MIT
"""

import numpy as np
from typing import Optional, Union
from sklearn.decomposition import PCA


class PCATransformer:
    """
    Safe PCA that prevents data leakage.
    
    CRITICAL: PCA must be fitted on TRAIN FOLD ONLY!
    
    For proper cross-validation workflow:
    1. Split data into train/test
    2. Within training set, use nested CV
    3. In each CV fold:
        a. Fit PCA on CV train fold
        b. Transform CV train and val folds
        c. Train model and evaluate
    
    Parameters
    ----------
    n_components : int, float, or None
        Number of components to keep:
        - int: Keep exact number (e.g., 50)
        - float: Keep enough to explain variance (e.g., 0.95 for 95%)
        - None: Keep all components
    
    whiten : bool, default=False
        Whether to whiten (scale) components
    
    Examples
    --------
    CORRECT Usage (Nested CV):
    >>> pca = PCATransformer(n_components=50)
    >>> 
    >>> # For each CV fold:
    >>> for train_idx, val_idx in cv_folds:
    >>>     X_train_fold = X_train[train_idx]
    >>>     
    >>>     # Fit PCA on THIS fold's training data
    >>>     pca.fit(X_train_fold)
    >>>     
    >>>     # Transform THIS fold's train and val
    >>>     X_train_pca = pca.transform(X_train_fold)
    >>>     X_val_pca = pca.transform(X_train[val_idx])
    >>>     
    >>>     # Train model on PCA features
    >>>     model.fit(X_train_pca, y_train[train_idx])
    
    INCORRECT Usage (Data Leakage!):
    >>> # WRONG: Fitting PCA before CV
    >>> pca.fit(X_train)  # ❌ Uses all training data
    >>> X_pca = pca.transform(X_train)
    >>> cv_score = cross_val_score(model, X_pca, y)  # ❌ LEAKAGE!
    """
    
    def __init__(
        self,
        n_components: Optional[Union[int, float]] = None,
        whiten: bool = False,
        random_state: int = 42
    ):
        self.n_components = n_components
        self.whiten = whiten
        self.random_state = random_state
        
        self.pca = None
        self.is_fitted = False
        self.n_features_in = None
        self.n_components_out = None
        self.explained_variance_ratio = None
    
    def fit(self, X: np.ndarray) -> 'PCATransformer':
        """
        Fit PCA on FOLD TRAINING DATA ONLY.
        
        ⚠️  CRITICAL: Only call this within CV fold on fold's training data!
        
        Parameters
        ----------
        X : np.ndarray
            Features from CURRENT CV FOLD training set
        
        Returns
        -------
        self : PCATransformer
            Fitted PCA transformer
        """
        self.n_features_in = X.shape[1]
        
        print(f"\n📊 Fitting PCA")
        print(f"  Input features: {self.n_features_in}")
        print(f"  ⚠️  Remember: Fit on TRAIN FOLD ONLY to prevent leakage!")
        
        # Create and fit PCA
        self.pca = PCA(
            n_components=self.n_components,
            whiten=self.whiten,
            random_state=self.random_state
        )
        self.pca.fit(X)
        
        self.is_fitted = True
        self.n_components_out = self.pca.n_components_
        self.explained_variance_ratio = self.pca.explained_variance_ratio_
        
        # Calculate cumulative variance explained
        cumulative_variance = np.cumsum(self.explained_variance_ratio)
        
        print(f"  PCA components: {self.n_components_out}")
        print(f"  Variance explained: {cumulative_variance[-1]*100:.2f}%")
        print(f"  Reduction: {self.n_features_in} → {self.n_components_out} ({self.n_components_out/self.n_features_in*100:.1f}%)")
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform features using fitted PCA.
        
        Parameters
        ----------
        X : np.ndarray
            Features to transform
        
        Returns
        -------
        X_pca : np.ndarray
            PCA-transformed features
        """
        if not self.is_fitted:
            raise RuntimeError("PCA not fitted! Call .fit() first.")
        
        if X.shape[1] != self.n_features_in:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.n_features_in}, got {X.shape[1]}"
            )
        
        return self.pca.transform(X)
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit PCA and transform features.
        
        ⚠️  Use ONLY within CV fold on fold's training data!
        """
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_pca: np.ndarray) -> np.ndarray:
        """
        Transform PCA features back to original space.
        
        Useful for visualization or interpreting PCA components.
        """
        if not self.is_fitted:
            raise RuntimeError("PCA not fitted!")
        
        return self.pca.inverse_transform(X_pca)
    
    def get_explained_variance_ratio(self) -> np.ndarray:
        """Get variance explained by each component."""
        if not self.is_fitted:
            raise RuntimeError("PCA not fitted!")
        return self.explained_variance_ratio
    
    def get_cumulative_variance(self) -> np.ndarray:
        """Get cumulative variance explained."""
        if not self.is_fitted:
            raise RuntimeError("PCA not fitted!")
        return np.cumsum(self.explained_variance_ratio)


def demonstrate_correct_pca_usage():
    """Demonstrate CORRECT PCA usage in nested CV."""
    print("\n" + "="*80)
    print("✅ CORRECT: PCA in Nested CV")
    print("="*80)
    
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import KFold
    from sklearn.metrics import r2_score
    
    # Simulate data
    np.random.seed(42)
    X = np.random.rand(100, 200)  # 200 features
    y = np.random.rand(100)
    
    print(f"\nData: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Setup
    model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    print("\n🔁 Starting Nested CV with PCA...")
    
    fold_scores = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\nFold {fold_idx + 1}/5")
        
        # Get fold data
        X_train_fold = X[train_idx]
        y_train_fold = y[train_idx]
        X_val_fold = X[val_idx]
        y_val_fold = y[val_idx]
        
        # CRITICAL: Fit PCA on THIS fold's training data
        pca = PCATransformer(n_components=0.95)  # Keep 95% variance
        pca.fit(X_train_fold)
        
        # Transform both train and val for this fold
        X_train_pca = pca.transform(X_train_fold)
        X_val_pca = pca.transform(X_val_fold)
        
        print(f"  PCA components: {X_train_pca.shape[1]}")
        print(f"  Variance: {pca.get_cumulative_variance()[-1]*100:.1f}%")
        
        # Train model on PCA features
        model.fit(X_train_pca, y_train_fold)
        
        # Evaluate
        y_pred = model.predict(X_val_pca)
        score = r2_score(y_val_fold, y_pred)
        fold_scores.append(score)
        
        print(f"  R² score: {score:.4f}")
    
    print(f"\n✅ Nested CV complete!")
    print(f"Mean R²: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
    print(f"\n✓ No data leakage - PCA fitted within each fold!")


def demonstrate_incorrect_usage():
    """Demonstrate INCORRECT PCA usage (causes leakage)."""
    print("\n" + "="*80)
    print("❌ INCORRECT: PCA Before CV (DATA LEAKAGE!)")
    print("="*80)
    
    print("\n❌ WRONG APPROACH: Fitting PCA before CV")
    print("-"*80)
    print("""
# This causes DATA LEAKAGE!

# WRONG: Fit PCA on all training data
pca = PCATransformer(n_components=50)
pca.fit(X_train)  # ❌ Uses ALL training data
X_pca = pca.transform(X_train)

# WRONG: Then do CV on PCA features
cv_scores = cross_val_score(model, X_pca, y_train)  # ❌ LEAKAGE!

Why wrong:
- PCA sees all training data
- Principal components learned from all data
- CV folds don't independently learn PCA
- Validation folds "see" information from training folds
- Overestimates performance
    """)
    
    print("\n✅ CORRECT APPROACH:")
    print("-"*80)
    print("""
# PCA WITHIN each CV fold

for train_idx, val_idx in cv_folds:
    # Fit PCA on THIS fold's training data only
    pca.fit(X_train[train_idx])  # ✓ Only this fold
    
    # Transform this fold
    X_train_pca = pca.transform(X_train[train_idx])
    X_val_pca = pca.transform(X_train[val_idx])
    
    # Train and evaluate
    model.fit(X_train_pca, y_train[train_idx])
    score = model.score(X_val_pca, y_train[val_idx])
    """)


def compare_pca_strategies():
    """Compare different PCA component selection strategies."""
    print("\n" + "="*80)
    print("📊 PCA Component Selection Strategies")
    print("="*80)
    
    np.random.seed(42)
    X = np.random.rand(100, 200)
    
    print("\nData: 100 samples, 200 features")
    
    strategies = [
        ("Fixed number", 50),
        ("95% variance", 0.95),
        ("90% variance", 0.90),
        ("All components", None)
    ]
    
    for name, n_components in strategies:
        print(f"\n{name} (n_components={n_components}):")
        
        pca = PCATransformer(n_components=n_components)
        pca.fit(X)
        
        print(f"  Components kept: {pca.n_components_out}")
        print(f"  Variance explained: {pca.get_cumulative_variance()[-1]*100:.2f}%")
        print(f"  Reduction: {X.shape[1]} → {pca.n_components_out} ({pca.n_components_out/X.shape[1]*100:.1f}%)")


def plot_pca_variance(X: np.ndarray, max_components: int = 50):
    """
    Plot cumulative variance explained by PCA components.
    
    Helps decide how many components to keep.
    """
    print("\n" + "="*80)
    print("📈 PCA Variance Analysis")
    print("="*80)
    
    pca = PCATransformer(n_components=min(max_components, X.shape[1]))
    pca.fit(X)
    
    cumulative = pca.get_cumulative_variance()
    
    print(f"\nComponents analyzed: {len(cumulative)}")
    print(f"\nVariance thresholds:")
    
    for threshold in [0.80, 0.90, 0.95, 0.99]:
        n_comp = np.argmax(cumulative >= threshold) + 1
        print(f"  {threshold*100:.0f}% variance: {n_comp} components")
    
    print(f"\n💡 Recommendation:")
    print(f"  - For efficiency: {np.argmax(cumulative >= 0.90) + 1} components (90% variance)")
    print(f"  - For accuracy: {np.argmax(cumulative >= 0.95) + 1} components (95% variance)")


if __name__ == '__main__':
    demonstrate_correct_pca_usage()
    print("\n" + "="*80 + "\n")
    demonstrate_incorrect_usage()
    print("\n" + "="*80 + "\n")
    compare_pca_strategies()
    
    # Demonstrate variance analysis
    np.random.seed(42)
    X_demo = np.random.rand(100, 200)
    plot_pca_variance(X_demo)
    print("\n" + "="*80 + "\n")
