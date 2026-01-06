"""
Performance Validation Module
==============================

Mitigates:
3. Improper Cross-Validation Design
9. Improper Performance Metrics
10. Lack of Baseline & Negative Controls

Provides tools to:
- Implement proper CV strategies
- Calculate comprehensive metrics
- Run baseline comparisons
- Perform y-randomization tests
- Report results correctly

Author: QSAR Validation Framework
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.model_selection import KFold
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error
)
from scipy.stats import spearmanr, pearsonr


class PerformanceValidator:
    """
    Comprehensive performance validation with proper controls.
    
    Provides:
    - Multiple metrics (R², RMSE, MAE, Spearman ρ)
    - Y-randomization tests
    - Baseline model comparisons
    - Proper CV reporting (mean ± std)
    
    Examples
    --------
    >>> validator = PerformanceValidator()
    >>> 
    >>> # Calculate all metrics
    >>> metrics = validator.calculate_comprehensive_metrics(y_true, y_pred)
    >>> 
    >>> # Run y-randomization test
    >>> random_results = validator.y_randomization_test(X, y, model, n_iterations=100)
    >>> 
    >>> # Compare to baseline
    >>> comparison = validator.compare_to_baseline(y_true, y_pred, baseline_pred)
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
    
    def calculate_comprehensive_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        set_name: str = 'Test'
    ) -> Dict:
        """
        Calculate comprehensive regression metrics.
        
        Parameters
        ----------
        y_true : np.ndarray
            True values
        
        y_pred : np.ndarray
            Predicted values
        
        set_name : str
            Name of dataset (e.g., 'Test', 'Train', 'Val')
        
        Returns
        -------
        metrics : dict
            All calculated metrics
        """
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        # Spearman correlation (rank-based, robust to outliers)
        spearman_rho, spearman_p = spearmanr(y_true, y_pred)
        
        # Pearson correlation
        pearson_r, pearson_p = pearsonr(y_true, y_pred)
        
        metrics = {
            f'{set_name}_R2': r2,
            f'{set_name}_RMSE': rmse,
            f'{set_name}_MAE': mae,
            f'{set_name}_Spearman_rho': spearman_rho,
            f'{set_name}_Spearman_p': spearman_p,
            f'{set_name}_Pearson_r': pearson_r,
            f'{set_name}_Pearson_p': pearson_p,
            f'{set_name}_N': len(y_true)
        }
        
        if self.verbose:
            print(f"\n{set_name} Set Metrics:")
            print(f"  N = {len(y_true)}")
            print(f"  R² = {r2:.4f}")
            print(f"  RMSE = {rmse:.4f}")
            print(f"  MAE = {mae:.4f}")
            print(f"  Spearman ρ = {spearman_rho:.4f} (p = {spearman_p:.4e})")
            print(f"  Pearson r = {pearson_r:.4f} (p = {pearson_p:.4e})")
        
        return metrics
    
    def cross_validate_properly(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model,
        n_folds: int = 5,
        random_state: int = 42
    ) -> Dict:
        """
        Perform proper cross-validation with complete reporting.
        
        Reports mean ± std for all metrics.
        
        Parameters
        ----------
        X : np.ndarray
            Features
        
        y : np.ndarray
            Targets
        
        model : sklearn model
            Model to evaluate
        
        n_folds : int
            Number of CV folds
        
        Returns
        -------
        cv_results : dict
            Mean ± std for all metrics
        """
        print(f"\n{'='*80}")
        print(f"CROSS-VALIDATION ({n_folds} folds)")
        print(f"{'='*80}")
        
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        
        fold_metrics = {
            'R2': [],
            'RMSE': [],
            'MAE': [],
            'Spearman_rho': []
        }
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train = X[train_idx]
            y_train = y[train_idx]
            X_val = X[val_idx]
            y_val = y[val_idx]
            
            # Train model
            model_clone = model.__class__(**model.get_params())
            model_clone.fit(X_train, y_train)
            
            # Predict
            y_pred = model_clone.predict(X_val)
            
            # Calculate metrics
            r2 = r2_score(y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            mae = mean_absolute_error(y_val, y_pred)
            spearman_rho, _ = spearmanr(y_val, y_pred)
            
            fold_metrics['R2'].append(r2)
            fold_metrics['RMSE'].append(rmse)
            fold_metrics['MAE'].append(mae)
            fold_metrics['Spearman_rho'].append(spearman_rho)
            
            if self.verbose:
                print(f"\nFold {fold_idx + 1}/{n_folds}:")
                print(f"  R² = {r2:.4f}, RMSE = {rmse:.4f}, MAE = {mae:.4f}")
        
        # Calculate mean ± std
        cv_results = {}
        for metric_name, values in fold_metrics.items():
            cv_results[f'CV_{metric_name}_mean'] = np.mean(values)
            cv_results[f'CV_{metric_name}_std'] = np.std(values)
        
        print(f"\n{'='*80}")
        print("CV RESULTS (mean ± std)")
        print(f"{'='*80}")
        print(f"R²:         {cv_results['CV_R2_mean']:.4f} ± {cv_results['CV_R2_std']:.4f}")
        print(f"RMSE:       {cv_results['CV_RMSE_mean']:.4f} ± {cv_results['CV_RMSE_std']:.4f}")
        print(f"MAE:        {cv_results['CV_MAE_mean']:.4f} ± {cv_results['CV_MAE_std']:.4f}")
        print(f"Spearman ρ: {cv_results['CV_Spearman_rho_mean']:.4f} ± {cv_results['CV_Spearman_rho_std']:.4f}")
        
        return cv_results
    
    def y_randomization_test(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model,
        n_iterations: int = 100,
        n_folds: int = 5,
        random_state: int = 42
    ) -> Dict:
        """
        Perform y-randomization test (negative control).
        
        Tests if model performance is due to real signal or chance.
        
        Parameters
        ----------
        X : np.ndarray
            Features
        
        y : np.ndarray
            True targets
        
        model : sklearn model
            Model to test
        
        n_iterations : int
            Number of randomization iterations
        
        n_folds : int
            CV folds per iteration
        
        Returns
        -------
        results : dict
            Y-randomization test results
        """
        print(f"\n{'='*80}")
        print(f"Y-RANDOMIZATION TEST ({n_iterations} iterations)")
        print(f"{'='*80}")
        print("\nTesting if model learns real signal or just noise...")
        
        np.random.seed(random_state)
        
        # First, get real performance
        print("\n1. Real model performance:")
        real_cv = self.cross_validate_properly(X, y, model, n_folds=n_folds, random_state=random_state)
        real_r2 = real_cv['CV_R2_mean']
        
        # Run randomization iterations
        print(f"\n2. Running {n_iterations} randomization iterations...")
        random_r2_scores = []
        
        for i in range(n_iterations):
            # Shuffle y values
            y_random = np.random.permutation(y)
            
            # Cross-validate with shuffled y
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state+i)
            fold_scores = []
            
            for train_idx, val_idx in kf.split(X):
                X_train = X[train_idx]
                y_train_random = y_random[train_idx]
                X_val = X[val_idx]
                y_val_random = y_random[val_idx]
                
                model_clone = model.__class__(**model.get_params())
                model_clone.fit(X_train, y_train_random)
                
                y_pred = model_clone.predict(X_val)
                r2 = r2_score(y_val_random, y_pred)
                fold_scores.append(r2)
            
            random_r2_scores.append(np.mean(fold_scores))
            
            if (i + 1) % 20 == 0:
                print(f"  Completed {i + 1}/{n_iterations} iterations")
        
        # Statistics
        random_mean = np.mean(random_r2_scores)
        random_std = np.std(random_r2_scores)
        random_max = np.max(random_r2_scores)
        
        # Calculate p-value (how often random does better)
        p_value = np.sum(np.array(random_r2_scores) >= real_r2) / n_iterations
        
        # Calculate separation
        if random_std > 0:
            separation = (real_r2 - random_mean) / random_std
        else:
            separation = np.inf if real_r2 > random_mean else 0
        
        print(f"\n{'='*80}")
        print("Y-RANDOMIZATION RESULTS")
        print(f"{'='*80}")
        print(f"\nReal model R²:            {real_r2:.4f}")
        print(f"Random model R² (mean):   {random_mean:.4f} ± {random_std:.4f}")
        print(f"Random model R² (max):    {random_max:.4f}")
        print(f"\nSeparation (σ):           {separation:.2f}")
        print(f"P-value:                  {p_value:.4f}")
        
        # Interpretation
        print(f"\n{'='*80}")
        print("INTERPRETATION")
        print(f"{'='*80}")
        
        if p_value > 0.05:
            print("⚠️  FAILED: Model does NOT significantly outperform random!")
            print("   → Model is likely learning noise, not real signal")
            interpretation = "failed"
        elif separation < 2.0:
            print("⚠️  WEAK: Model barely outperforms random")
            print("   → Weak signal, results may not be robust")
            interpretation = "weak"
        elif separation < 5.0:
            print("⚙️  MODERATE: Model moderately outperforms random")
            print("   → Some signal present, but could be stronger")
            interpretation = "moderate"
        else:
            print("✓  STRONG: Model strongly outperforms random")
            print("   → Clear signal, model learning real relationships")
            interpretation = "strong"
        
        return {
            'real_r2': real_r2,
            'random_mean_r2': random_mean,
            'random_std_r2': random_std,
            'random_max_r2': random_max,
            'separation_sigma': separation,
            'p_value': p_value,
            'interpretation': interpretation,
            'random_scores': random_r2_scores
        }
    
    def compare_to_baseline(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        baseline_pred: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Compare model to baseline.
        
        If no baseline provided, uses mean prediction.
        
        Parameters
        ----------
        y_true : np.ndarray
            True values
        
        y_pred : np.ndarray
            Model predictions
        
        baseline_pred : np.ndarray, optional
            Baseline predictions (if None, uses mean)
        
        Returns
        -------
        comparison : dict
            Comparison results
        """
        print(f"\n{'='*80}")
        print("BASELINE COMPARISON")
        print(f"{'='*80}")
        
        # Model metrics
        model_r2 = r2_score(y_true, y_pred)
        model_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # Baseline metrics
        if baseline_pred is None:
            # Use mean as baseline
            baseline_pred = np.full_like(y_true, y_true.mean())
            baseline_name = "Mean baseline"
        else:
            baseline_name = "Provided baseline"
        
        baseline_r2 = r2_score(y_true, baseline_pred)
        baseline_rmse = np.sqrt(mean_squared_error(y_true, baseline_pred))
        
        # Calculate improvements
        r2_improvement = model_r2 - baseline_r2
        rmse_improvement = baseline_rmse - model_rmse  # Lower is better
        rmse_improvement_pct = (rmse_improvement / baseline_rmse) * 100
        
        print(f"\n{baseline_name}:")
        print(f"  R² = {baseline_r2:.4f}")
        print(f"  RMSE = {baseline_rmse:.4f}")
        
        print(f"\nYour model:")
        print(f"  R² = {model_r2:.4f}")
        print(f"  RMSE = {model_rmse:.4f}")
        
        print(f"\nImprovement:")
        print(f"  ΔR² = {r2_improvement:+.4f}")
        print(f"  ΔRMSE = {rmse_improvement:+.4f} ({rmse_improvement_pct:+.1f}%)")
        
        if r2_improvement < 0.05:
            print(f"\n⚠️  Model barely beats baseline (ΔR² < 0.05)")
        elif r2_improvement < 0.2:
            print(f"\n⚙️  Modest improvement over baseline")
        else:
            print(f"\n✓  Good improvement over baseline")
        
        return {
            'model_r2': model_r2,
            'model_rmse': model_rmse,
            'baseline_r2': baseline_r2,
            'baseline_rmse': baseline_rmse,
            'r2_improvement': r2_improvement,
            'rmse_improvement': rmse_improvement,
            'rmse_improvement_pct': rmse_improvement_pct
        }


def demonstrate_performance_validation():
    """Demonstrate performance validation tools."""
    print("\n" + "="*80)
    print("PERFORMANCE VALIDATION DEMONSTRATION")
    print("="*80)
    
    from sklearn.linear_model import Ridge
    
    # Generate synthetic data with weak signal
    np.random.seed(42)
    X = np.random.randn(200, 100)
    # Weak linear relationship + noise
    y = 0.5 * X[:, 0] + 0.3 * X[:, 1] + np.random.randn(200) * 2
    
    # Split data
    X_train, X_test = X[:150], X[150:]
    y_train, y_test = y[:150], y[150:]
    
    # Train model
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Create validator
    validator = PerformanceValidator()
    
    # 1. Comprehensive metrics
    print("\n1. Comprehensive Metrics")
    print("-" * 80)
    metrics = validator.calculate_comprehensive_metrics(y_test, y_pred, set_name='Test')
    
    # 2. Cross-validation
    print("\n\n2. Proper Cross-Validation")
    print("-" * 80)
    cv_results = validator.cross_validate_properly(X_train, y_train, model, n_folds=5)
    
    # 3. Y-randomization test
    print("\n\n3. Y-Randomization Test")
    print("-" * 80)
    y_random_results = validator.y_randomization_test(
        X_train, y_train, model, n_iterations=50, n_folds=5
    )
    
    # 4. Baseline comparison
    print("\n\n4. Baseline Comparison")
    print("-" * 80)
    comparison = validator.compare_to_baseline(y_test, y_pred)
    
    print("\n" + "="*80)
    print("✓ Demonstration complete!")
    print("="*80)


if __name__ == '__main__':
    demonstrate_performance_validation()
