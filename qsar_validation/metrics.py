"""
Performance Metrics Module
===========================

Calculates comprehensive performance metrics with proper baselines.
"""

import numpy as np
from typing import Dict, List, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge


class PerformanceMetricsCalculator:
    """
    Calculate comprehensive performance metrics with proper baselines.
    """
    
    @staticmethod
    def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                             set_name: str = "Test") -> Dict:
        """
        Calculate comprehensive regression metrics.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            set_name: Name of the dataset (for display)
            
        Returns:
            Dictionary containing all calculated metrics
        """
        print(f"\n📊 {set_name.upper()} SET PERFORMANCE METRICS")
        print("=" * 70)
        
        # Calculate all metrics
        metrics = PerformanceMetricsCalculator._compute_metrics(y_true, y_pred)
        
        # Display results
        PerformanceMetricsCalculator._print_metrics(metrics)
        
        return metrics
    
    @staticmethod
    def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Compute all metrics."""
        # Regression metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        max_error = np.max(np.abs(y_true - y_pred))
        
        # Correlation metrics
        pearson_r, pearson_p = pearsonr(y_true, y_pred)
        spearman_r, spearman_p = spearmanr(y_true, y_pred)
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'mse': mse,
            'max_error': max_error,
        }
    
    @staticmethod
    def _print_metrics(metrics: Dict) -> None:
        """Print metrics in formatted output."""
        print(f"RMSE:           {metrics['rmse']:.4f}")
        print(f"MAE:            {metrics['mae']:.4f}")
        print(f"R²:             {metrics['r2']:.4f}")
        print(f"Pearson r:      {metrics['pearson_r']:.4f} (p={metrics['pearson_p']:.4e})")
        print(f"Spearman ρ:     {metrics['spearman_r']:.4f} (p={metrics['spearman_p']:.4e})")
        print(f"Max Error:      {metrics['max_error']:.4f}")
    
    @staticmethod
    def calculate_baseline_metrics(X: np.ndarray, y: np.ndarray, 
                                   cv_folds: Optional[List] = None) -> Dict:
        """
        Calculate simple baseline model performance using Ridge regression.
        
        Args:
            X: Feature matrix
            y: Target values
            cv_folds: Optional list of (train_idx, val_idx) tuples for CV
            
        Returns:
            Dictionary with baseline metrics
        """
        print("\n📊 BASELINE MODEL PERFORMANCE (Ridge Regression)")
        print("=" * 70)
        
        if cv_folds is None:
            metrics = PerformanceMetricsCalculator._baseline_train_test(X, y)
        else:
            metrics = PerformanceMetricsCalculator._baseline_cv(X, y, cv_folds)
        
        print("\n📌 Use this as minimum performance threshold")
        print("   Complex models should significantly outperform this baseline")
        
        return metrics
    
    @staticmethod
    def _baseline_train_test(X: np.ndarray, y: np.ndarray) -> Dict:
        """Calculate baseline with simple train-test split."""
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        return PerformanceMetricsCalculator.calculate_all_metrics(
            y_test, y_pred, "Baseline Test"
        )
    
    @staticmethod
    def _baseline_cv(X: np.ndarray, y: np.ndarray, cv_folds: List) -> Dict:
        """Calculate baseline with cross-validation."""
        rmse_scores = []
        mae_scores = []
        r2_scores = []
        
        for train_idx, val_idx in cv_folds:
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model = Ridge(alpha=1.0)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            
            rmse_scores.append(np.sqrt(mean_squared_error(y_val, y_pred)))
            mae_scores.append(mean_absolute_error(y_val, y_pred))
            r2_scores.append(r2_score(y_val, y_pred))
        
        metrics = {
            'rmse_mean': np.mean(rmse_scores),
            'rmse_std': np.std(rmse_scores),
            'mae_mean': np.mean(mae_scores),
            'mae_std': np.std(mae_scores),
            'r2_mean': np.mean(r2_scores),
            'r2_std': np.std(r2_scores),
        }
        
        print(f"RMSE:  {metrics['rmse_mean']:.4f} ± {metrics['rmse_std']:.4f}")
        print(f"MAE:   {metrics['mae_mean']:.4f} ± {metrics['mae_std']:.4f}")
        print(f"R²:    {metrics['r2_mean']:.4f} ± {metrics['r2_std']:.4f}")
        
        return metrics
