"""
Model Complexity Control Module
================================

Mitigates:
2. Overfitting Due to Model Complexity
8. Poor Uncertainty Estimation

Provides tools to:
- Assess model complexity vs dataset size
- Recommend appropriate model types
- Control hyperparameter ranges
- Implement nested CV
- Provide uncertainty estimates

Author: QSAR Validation Framework
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error


class ModelComplexityController:
    """
    Control model complexity to prevent overfitting.
    
    Provides:
    - Model recommendations based on dataset size
    - Restricted hyperparameter ranges
    - Nested CV implementation
    - Complexity assessment
    
    Parameters
    ----------
    n_samples : int
        Number of training samples
    
    n_features : int
        Number of features
    
    Examples
    --------
    >>> controller = ModelComplexityController(
    ...     n_samples=150,
    ...     n_features=1024
    ... )
    >>> 
    >>> # Get model recommendations
    >>> recommendations = controller.recommend_models()
    >>> 
    >>> # Get safe hyperparameter ranges
    >>> param_grid = controller.get_safe_param_grid('random_forest')
    >>> 
    >>> # Run nested CV
    >>> results = controller.nested_cv(X, y, model_type='ridge')
    """
    
    def __init__(
        self,
        n_samples: int,
        n_features: int,
        verbose: bool = True
    ):
        self.n_samples = n_samples
        self.n_features = n_features
        self.verbose = verbose
        self.complexity_ratio = n_samples / n_features
        
        if verbose:
            self._print_dataset_summary()
    
    def _print_dataset_summary(self):
        """Print dataset size summary."""
        print("\n" + "="*80)
        print("MODEL COMPLEXITY CONTROL")
        print("="*80)
        print(f"\nDataset size:")
        print(f"  Samples: {self.n_samples}")
        print(f"  Features: {self.n_features}")
        print(f"  Sample/Feature ratio: {self.complexity_ratio:.3f}")
        
        if self.complexity_ratio < 0.1:
            print(f"  ⚠️  VERY LOW - High overfitting risk!")
        elif self.complexity_ratio < 0.5:
            print(f"  ⚠️  LOW - Use simple models only")
        elif self.complexity_ratio < 2.0:
            print(f"  ⚙️  MODERATE - Use moderate complexity")
        else:
            print(f"  ✓  GOOD - Can use complex models")
    
    def recommend_models(self) -> Dict[str, str]:
        """
        Recommend appropriate model types based on dataset size.
        
        Returns
        -------
        recommendations : dict
            Model recommendations with reasoning
        """
        print("\n" + "-"*80)
        print("MODEL RECOMMENDATIONS")
        print("-"*80)
        
        recommendations = {
            'primary': [],
            'avoid': [],
            'reasoning': []
        }
        
        ratio = self.complexity_ratio
        
        if ratio < 0.1:
            # Very low data regime
            recommendations['primary'] = ['Ridge', 'Lasso', 'ElasticNet']
            recommendations['avoid'] = ['Random Forest', 'XGBoost', 'Neural Networks', 'SVM']
            recommendations['reasoning'].append(
                f"Sample/feature ratio = {ratio:.3f} < 0.1: Only linear models recommended"
            )
            print("\n  ⚠️  VERY LOW DATA REGIME")
            print("  ✓ Recommended: Ridge, Lasso, ElasticNet")
            print("  ✗ Avoid: Random Forest, XGBoost, Neural Networks")
            
        elif ratio < 0.5:
            # Low data regime
            recommendations['primary'] = ['Ridge', 'Random Forest (shallow)']
            recommendations['avoid'] = ['Deep Random Forest', 'XGBoost', 'Neural Networks']
            recommendations['reasoning'].append(
                f"Sample/feature ratio = {ratio:.3f} < 0.5: Simple models recommended"
            )
            print("\n  ⚠️  LOW DATA REGIME")
            print("  ✓ Recommended: Ridge, Random Forest (max_depth ≤ 5)")
            print("  ✗ Avoid: Deep trees, XGBoost, Neural Networks")
            
        elif ratio < 2.0:
            # Moderate data regime
            recommendations['primary'] = ['Ridge', 'Random Forest', 'Gradient Boosting (conservative)']
            recommendations['avoid'] = ['Deep Neural Networks']
            recommendations['reasoning'].append(
                f"Sample/feature ratio = {ratio:.3f} < 2.0: Moderate complexity OK"
            )
            print("\n  ⚙️  MODERATE DATA REGIME")
            print("  ✓ Recommended: Ridge, Random Forest, Gradient Boosting (conservative)")
            print("  ⚠️  Caution: Deep Neural Networks (requires careful regularization)")
            
        else:
            # Good data regime
            recommendations['primary'] = ['All models']
            recommendations['avoid'] = []
            recommendations['reasoning'].append(
                f"Sample/feature ratio = {ratio:.3f} ≥ 2.0: Can use complex models"
            )
            print("\n  ✓  GOOD DATA REGIME")
            print("  ✓ Can use: All model types")
        
        print("\n  📌 Remember: In low-data QSAR, simpler is often better!")
        
        return recommendations
    
    def get_safe_param_grid(self, model_type: str) -> Dict:
        """
        Get safe hyperparameter ranges based on dataset size.
        
        Parameters
        ----------
        model_type : str
            Model type: 'ridge', 'random_forest', 'gradient_boosting'
        
        Returns
        -------
        param_grid : dict
            Safe hyperparameter grid for GridSearchCV
        """
        print(f"\n" + "-"*80)
        print(f"SAFE HYPERPARAMETER RANGES: {model_type.upper()}")
        print("-"*80)
        
        ratio = self.complexity_ratio
        
        if model_type.lower() == 'ridge':
            param_grid = {
                'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
            }
            print("  Ridge regularization:")
            print("    alpha: [0.01, 0.1, 1.0, 10.0, 100.0]")
            
        elif model_type.lower() == 'random_forest':
            if ratio < 0.5:
                # Low data: shallow trees only
                param_grid = {
                    'n_estimators': [50, 100],
                    'max_depth': [3, 5],
                    'min_samples_split': [10, 20],
                    'min_samples_leaf': [5, 10]
                }
                print("  ⚠️  Low data regime - SHALLOW trees:")
            elif ratio < 2.0:
                # Moderate data: moderate depth
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15],
                    'min_samples_split': [5, 10],
                    'min_samples_leaf': [2, 5]
                }
                print("  ⚙️  Moderate data regime - MODERATE trees:")
            else:
                # Good data: can go deeper
                param_grid = {
                    'n_estimators': [100, 200, 500],
                    'max_depth': [10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 5]
                }
                print("  ✓  Good data regime - Can use deeper trees:")
            
            for param, values in param_grid.items():
                print(f"    {param}: {values}")
        
        elif model_type.lower() == 'gradient_boosting':
            if ratio < 0.5:
                print("  ⚠️  NOT RECOMMENDED for this dataset size!")
                print("  Use Ridge or shallow Random Forest instead")
                return {}
            elif ratio < 2.0:
                param_grid = {
                    'n_estimators': [50, 100],
                    'max_depth': [3, 5],
                    'learning_rate': [0.01, 0.1],
                    'subsample': [0.8]
                }
                print("  ⚙️  Conservative settings:")
            else:
                param_grid = {
                    'n_estimators': [100, 200, 500],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'subsample': [0.8, 1.0]
                }
                print("  ✓  Standard settings:")
            
            for param, values in param_grid.items():
                print(f"    {param}: {values}")
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        print("\n  📌 These ranges prevent excessive overfitting")
        
        return param_grid
    
    def nested_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_type: str = 'ridge',
        outer_cv: int = 5,
        inner_cv: int = 3,
        random_state: int = 42
    ) -> Dict:
        """
        Perform nested cross-validation.
        
        Outer loop: Performance estimation
        Inner loop: Hyperparameter tuning
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        
        y : np.ndarray
            Target values
        
        model_type : str
            'ridge' or 'random_forest'
        
        outer_cv : int
            Number of outer CV folds
        
        inner_cv : int
            Number of inner CV folds
        
        Returns
        -------
        results : dict
            Nested CV results
        """
        print("\n" + "="*80)
        print(f"NESTED CROSS-VALIDATION: {model_type.upper()}")
        print("="*80)
        print(f"\nOuter CV: {outer_cv} folds (performance estimation)")
        print(f"Inner CV: {inner_cv} folds (hyperparameter tuning)")
        
        # Get safe parameter grid
        param_grid = self.get_safe_param_grid(model_type)
        
        if not param_grid:
            raise ValueError(f"Model type '{model_type}' not recommended for this dataset")
        
        # Setup model
        if model_type.lower() == 'ridge':
            base_model = Ridge()
        elif model_type.lower() == 'random_forest':
            base_model = RandomForestRegressor(random_state=random_state, n_jobs=-1)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Outer CV loop
        outer_kf = KFold(n_splits=outer_cv, shuffle=True, random_state=random_state)
        
        outer_scores = []
        outer_rmse = []
        best_params_per_fold = []
        
        print("\n" + "-"*80)
        
        for fold_idx, (train_idx, test_idx) in enumerate(outer_kf.split(X)):
            print(f"\nOuter Fold {fold_idx + 1}/{outer_cv}")
            
            X_train_outer = X[train_idx]
            y_train_outer = y[train_idx]
            X_test_outer = X[test_idx]
            y_test_outer = y[test_idx]
            
            # Inner CV loop: hyperparameter tuning
            inner_kf = KFold(n_splits=inner_cv, shuffle=True, random_state=random_state)
            
            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=inner_kf,
                scoring='r2',
                n_jobs=-1
            )
            
            grid_search.fit(X_train_outer, y_train_outer)
            
            # Best model from inner CV
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_params_per_fold.append(best_params)
            
            print(f"  Best params: {best_params}")
            
            # Evaluate on outer test fold
            y_pred = best_model.predict(X_test_outer)
            score = r2_score(y_test_outer, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test_outer, y_pred))
            
            outer_scores.append(score)
            outer_rmse.append(rmse)
            
            print(f"  R²: {score:.4f}")
            print(f"  RMSE: {rmse:.4f}")
        
        # Summary
        mean_r2 = np.mean(outer_scores)
        std_r2 = np.std(outer_scores)
        mean_rmse = np.mean(outer_rmse)
        std_rmse = np.std(outer_rmse)
        
        print("\n" + "="*80)
        print("NESTED CV RESULTS")
        print("="*80)
        print(f"\nR²:   {mean_r2:.4f} ± {std_r2:.4f}")
        print(f"RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")
        
        # Check for overfitting signs
        if std_r2 > 0.2:
            print(f"\n⚠️  High variance in R² (std = {std_r2:.4f})")
            print("   Consider:")
            print("   - Using simpler model")
            print("   - More regularization")
            print("   - More data")
        
        return {
            'mean_r2': mean_r2,
            'std_r2': std_r2,
            'mean_rmse': mean_rmse,
            'std_rmse': std_rmse,
            'fold_scores': outer_scores,
            'fold_rmse': outer_rmse,
            'best_params_per_fold': best_params_per_fold
        }
    
    def assess_model_complexity(
        self,
        model: Any,
        model_type: str
    ) -> Dict:
        """
        Assess complexity of a trained model.
        
        Parameters
        ----------
        model : sklearn model
            Trained model
        
        model_type : str
            'ridge', 'random_forest', etc.
        
        Returns
        -------
        complexity_stats : dict
            Model complexity statistics
        """
        print("\n" + "="*80)
        print("MODEL COMPLEXITY ASSESSMENT")
        print("="*80)
        
        stats = {
            'model_type': model_type,
            'n_samples': self.n_samples,
            'n_features': self.n_features
        }
        
        if model_type.lower() == 'ridge':
            stats['n_parameters'] = self.n_features + 1  # Weights + bias
            stats['regularization'] = model.alpha
            
            print(f"\nModel: Ridge Regression")
            print(f"  Parameters: {stats['n_parameters']}")
            print(f"  Regularization (alpha): {stats['regularization']}")
            print(f"  Effective complexity: Controlled by alpha")
            
        elif model_type.lower() == 'random_forest':
            stats['n_estimators'] = model.n_estimators
            stats['max_depth'] = model.max_depth
            stats['n_leaves_per_tree'] = 2 ** model.max_depth if model.max_depth else 'unlimited'
            
            # Rough estimate of parameters
            if model.max_depth:
                params_per_tree = 2 ** model.max_depth
                stats['estimated_parameters'] = params_per_tree * model.n_estimators
            else:
                stats['estimated_parameters'] = 'unknown (unlimited depth)'
            
            print(f"\nModel: Random Forest")
            print(f"  Trees: {stats['n_estimators']}")
            print(f"  Max depth: {stats['max_depth']}")
            print(f"  Estimated parameters: {stats['estimated_parameters']}")
        
        # Calculate complexity ratio
        if isinstance(stats.get('n_parameters'), (int, float)):
            complexity_ratio = self.n_samples / stats['n_parameters']
            stats['samples_per_parameter'] = complexity_ratio
            
            print(f"\nComplexity Assessment:")
            print(f"  Samples per parameter: {complexity_ratio:.3f}")
            
            if complexity_ratio < 1.0:
                print(f"  ⚠️  SEVERE: More parameters than samples!")
            elif complexity_ratio < 5.0:
                print(f"  ⚠️  HIGH RISK: Very few samples per parameter")
            elif complexity_ratio < 10.0:
                print(f"  ⚙️  MODERATE: Adequate samples per parameter")
            else:
                print(f"  ✓  LOW RISK: Many samples per parameter")
        
        return stats


def demonstrate_complexity_control():
    """Demonstrate model complexity control."""
    print("\n" + "="*80)
    print("MODEL COMPLEXITY CONTROL DEMONSTRATION")
    print("="*80)
    
    # Simulate low-data scenario
    np.random.seed(42)
    X = np.random.randn(100, 500)  # 100 samples, 500 features
    y = np.random.randn(100)
    
    # Create controller
    controller = ModelComplexityController(
        n_samples=X.shape[0],
        n_features=X.shape[1]
    )
    
    # Get recommendations
    recommendations = controller.recommend_models()
    
    # Get safe parameters for Ridge
    param_grid = controller.get_safe_param_grid('ridge')
    
    # Run nested CV
    results = controller.nested_cv(X, y, model_type='ridge', outer_cv=5, inner_cv=3)
    
    print("\n" + "="*80)
    print("✓ Demonstration complete!")
    print("="*80)


if __name__ == '__main__':
    demonstrate_complexity_control()
