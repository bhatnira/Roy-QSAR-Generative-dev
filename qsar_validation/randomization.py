"""
Y-Randomization Testing Module
===============================

Performs y-randomization (y-scrambling) tests to detect overfitting.
"""

import numpy as np
from typing import Dict, List, Optional
from sklearn.metrics import mean_squared_error, r2_score


class YRandomizationTester:
    """
    Perform y-randomization (y-scrambling) test to detect overfitting.
    If model performs well with randomized targets, it's memorizing noise.
    """
    
    @staticmethod
    def perform_y_randomization(X: np.ndarray, y: np.ndarray, 
                                model, n_iterations: int = 10,
                                cv_folds: Optional[List] = None) -> Dict:
        """
        Perform y-randomization test.
        
        Args:
            X: Feature matrix
            y: Target values (will be randomized)
            model: Model instance with fit/predict methods
            n_iterations: Number of randomization iterations (default: 10)
            cv_folds: Optional list of (train_idx, val_idx) tuples for CV
            
        Returns:
            Dictionary with randomization results including mean/std metrics
        """
        print("\n🎲 Y-RANDOMIZATION TEST (Y-SCRAMBLING)")
        print("=" * 70)
        print(f"Running {n_iterations} iterations with randomized targets...")
        
        # Run iterations
        rmse_scores, r2_scores = YRandomizationTester._run_iterations(
            X, y, model, n_iterations, cv_folds
        )
        
        # Compile results
        results = {
            'rmse_mean': np.mean(rmse_scores),
            'rmse_std': np.std(rmse_scores),
            'r2_mean': np.mean(r2_scores),
            'r2_std': np.std(r2_scores),
            'rmse_scores': rmse_scores,
            'r2_scores': r2_scores,
        }
        
        # Print and interpret results
        YRandomizationTester._print_results(results, n_iterations)
        YRandomizationTester._interpret_results(results)
        
        return results
    
    @staticmethod
    def _run_iterations(X: np.ndarray, y: np.ndarray, model, 
                       n_iterations: int, cv_folds: Optional[List]) -> tuple:
        """Run randomization iterations."""
        rmse_scores = []
        r2_scores = []
        
        for i in range(n_iterations):
            # Randomize target
            y_random = np.random.permutation(y)
            
            if cv_folds is None:
                rmse, r2 = YRandomizationTester._single_split(X, y_random, model, i)
            else:
                rmse, r2 = YRandomizationTester._cross_validation(
                    X, y_random, model, cv_folds
                )
            
            rmse_scores.append(rmse)
            r2_scores.append(r2)
        
        return rmse_scores, r2_scores
    
    @staticmethod
    def _single_split(X: np.ndarray, y_random: np.ndarray, 
                     model, seed: int) -> tuple:
        """Evaluate with single train-test split."""
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_random, test_size=0.2, random_state=seed
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        return rmse, r2
    
    @staticmethod
    def _cross_validation(X: np.ndarray, y_random: np.ndarray, 
                         model, cv_folds: List) -> tuple:
        """Evaluate with cross-validation."""
        fold_rmse = []
        fold_r2 = []
        
        for train_idx, val_idx in cv_folds:
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y_random[train_idx], y_random[val_idx]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            
            fold_rmse.append(np.sqrt(mean_squared_error(y_val, y_pred)))
            fold_r2.append(r2_score(y_val, y_pred))
        
        return np.mean(fold_rmse), np.mean(fold_r2)
    
    @staticmethod
    def _print_results(results: Dict, n_iterations: int) -> None:
        """Print randomization results."""
        print(f"\n📊 Randomized Results ({n_iterations} iterations):")
        print(f"   RMSE: {results['rmse_mean']:.4f} ± {results['rmse_std']:.4f}")
        print(f"   R²:   {results['r2_mean']:.4f} ± {results['r2_std']:.4f}")
    
    @staticmethod
    def _interpret_results(results: Dict) -> None:
        """Interpret and provide guidance on results."""
        r2_mean = results['r2_mean']
        
        if r2_mean > 0.2:
            print("\n⚠️  WARNING: R² > 0.2 with randomized targets")
            print("   → Model is likely overfitting")
            print("   → Reduce model complexity")
            print("   → Increase regularization")
        elif r2_mean > 0.0:
            print("\n🟡 CAUTION: Positive R² with randomized targets")
            print("   → Some overfitting detected")
            print("   → Consider simplifying model")
        else:
            print("\n✓ Good: R² ≤ 0 with randomized targets")
            print("  → Model is not memorizing random noise")
