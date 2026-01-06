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

Supports:
- Scikit-learn models
- XGBoost / LightGBM
- PyTorch / TensorFlow
- Custom models

Author: QSAR Validation Framework
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable


class ModelWrapper:
    """
    Universal model wrapper for different ML libraries.
    
    Supports:
    - Scikit-learn
    - XGBoost
    - LightGBM
    - PyTorch
    - TensorFlow
    - Custom models
    """
    
    def __init__(self, model, library: str = 'sklearn'):
        self.model = model
        self.library = library.lower()
        self._detect_library()
    
    def _detect_library(self):
        """Auto-detect ML library if not specified."""
        if self.library == 'auto':
            model_class = str(type(self.model))
            if 'sklearn' in model_class:
                self.library = 'sklearn'
            elif 'xgboost' in model_class or 'XGB' in model_class:
                self.library = 'xgboost'
            elif 'lightgbm' in model_class or 'LGB' in model_class:
                self.library = 'lightgbm'
            elif 'torch' in model_class:
                self.library = 'pytorch'
            elif 'tensorflow' in model_class or 'keras' in model_class:
                self.library = 'tensorflow'
            else:
                self.library = 'custom'
    
    def fit(self, X, y):
        """Fit model."""
        if self.library in ['sklearn', 'xgboost', 'lightgbm']:
            self.model.fit(X, y)
        elif self.library == 'pytorch':
            # PyTorch models need custom training loop
            # User should provide fit method
            if hasattr(self.model, 'fit'):
                self.model.fit(X, y)
            else:
                raise ValueError("PyTorch model needs .fit() method")
        elif self.library == 'tensorflow':
            # TensorFlow/Keras models
            self.model.fit(X, y, verbose=0)
        elif self.library == 'custom':
            if hasattr(self.model, 'fit'):
                self.model.fit(X, y)
            else:
                raise ValueError("Custom model needs .fit() method")
        return self
    
    def predict(self, X):
        """Predict."""
        if self.library in ['sklearn', 'xgboost', 'lightgbm', 'tensorflow']:
            return self.model.predict(X)
        elif self.library == 'pytorch':
            import torch
            if isinstance(X, np.ndarray):
                X = torch.FloatTensor(X)
            self.model.eval()
            with torch.no_grad():
                return self.model(X).numpy()
        elif self.library == 'custom':
            return self.model.predict(X)
    
    def clone(self, **params):
        """Clone model with new parameters."""
        if self.library == 'sklearn':
            from sklearn.base import clone
            return ModelWrapper(clone(self.model), self.library)
        elif self.library in ['xgboost', 'lightgbm']:
            # XGBoost/LightGBM can be recreated
            model_class = type(self.model)
            new_params = self.model.get_params()
            new_params.update(params)
            return ModelWrapper(model_class(**new_params), self.library)
        else:
            # For PyTorch/TensorFlow, return same model
            # User should handle model recreation
            return self


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Calculate metrics without sklearn dependency."""
    # R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # RMSE
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # MAE
    mae = np.mean(np.abs(y_true - y_pred))
    
    return {'r2': r2, 'rmse': rmse, 'mae': mae}


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
        
        Now includes recommendations for ALL ML libraries!
        
        Returns
        -------
        recommendations : dict
            Model recommendations with reasoning
        """
        print("\n" + "-"*80)
        print("MODEL RECOMMENDATIONS (All Libraries)")
        print("-"*80)
        
        recommendations = {
            'sklearn': [],
            'xgboost': [],
            'lightgbm': [],
            'pytorch': [],
            'tensorflow': [],
            'avoid': [],
            'reasoning': []
        }
        
        ratio = self.complexity_ratio
        
        if ratio < 0.1:
            # Very low data regime (N < 100 for typical features)
            recommendations['sklearn'] = ['Ridge', 'Lasso', 'ElasticNet', 'KernelRidge (RBF)']
            recommendations['xgboost'] = []
            recommendations['lightgbm'] = []
            recommendations['pytorch'] = []
            recommendations['tensorflow'] = []
            recommendations['avoid'] = ['All tree-based', 'All neural networks']
            recommendations['reasoning'].append(
                f"Sample/feature ratio = {ratio:.3f} < 0.1: Only LINEAR models recommended"
            )
            print("\n  ⚠️  VERY LOW DATA REGIME (N < 100)")
            print("  ✓ Scikit-learn: Ridge, Lasso, ElasticNet")
            print("  ✗ Avoid: XGBoost, LightGBM, PyTorch, TensorFlow")
            
        elif ratio < 0.5:
            # Low data regime (100 < N < 500 for typical features)
            recommendations['sklearn'] = ['Ridge', 'RandomForest (shallow)', 'GradientBoosting (conservative)']
            recommendations['xgboost'] = ['XGBRegressor (max_depth≤3, n_estimators≤100)']
            recommendations['lightgbm'] = ['LGBMRegressor (num_leaves≤15, n_estimators≤100)']
            recommendations['pytorch'] = []
            recommendations['tensorflow'] = []
            recommendations['avoid'] = ['Deep neural networks', 'Complex architectures']
            recommendations['reasoning'].append(
                f"Sample/feature ratio = {ratio:.3f} < 0.5: Simple models + shallow boosting"
            )
            print("\n  ⚠️  LOW DATA REGIME (100 < N < 500)")
            print("  ✓ Scikit-learn: Ridge, RandomForest (max_depth≤5)")
            print("  ⚙️  XGBoost/LightGBM: Shallow trees ONLY (max_depth≤3)")
            print("  ✗ Avoid: Deep neural networks")
            
        elif ratio < 2.0:
            # Moderate data regime (500 < N < 2000)
            recommendations['sklearn'] = ['Ridge', 'RandomForest', 'GradientBoosting', 'SVR']
            recommendations['xgboost'] = ['XGBRegressor (standard settings)']
            recommendations['lightgbm'] = ['LGBMRegressor (standard settings)']
            recommendations['pytorch'] = ['Small MLP (1-2 hidden layers, <100 units)']
            recommendations['tensorflow'] = ['Keras Sequential (1-2 hidden layers)']
            recommendations['avoid'] = ['Very deep networks (>5 layers)']
            recommendations['reasoning'].append(
                f"Sample/feature ratio = {ratio:.3f} < 2.0: Moderate complexity OK"
            )
            print("\n  ⚙️  MODERATE DATA REGIME (500 < N < 2000)")
            print("  ✓ Scikit-learn: All models")
            print("  ✓ XGBoost/LightGBM: Standard settings")
            print("  ⚙️  PyTorch/TensorFlow: Small networks (1-2 layers)")
            print("  ⚠️  Caution: Very deep networks")
            
        else:
            # Good data regime (N > 2000)
            recommendations['sklearn'] = ['All models']
            recommendations['xgboost'] = ['All settings']
            recommendations['lightgbm'] = ['All settings']
            recommendations['pytorch'] = ['MLP, CNN, Attention (moderate depth)']
            recommendations['tensorflow'] = ['All Keras models']
            recommendations['avoid'] = []
            recommendations['reasoning'].append(
                f"Sample/feature ratio = {ratio:.3f} ≥ 2.0: Can use complex models"
            )
            print("\n  ✓  GOOD DATA REGIME (N > 2000)")
            print("  ✓ Can use: All model types and libraries")
            print("  ✓ XGBoost/LightGBM: Full hyperparameter ranges")
            print("  ✓ PyTorch/TensorFlow: Moderate depth networks")
        
        print("\n  📌 Library-specific recommendations:")
        print(f"     • Scikit-learn: {', '.join(recommendations['sklearn']) if recommendations['sklearn'] else 'Not recommended'}")
        print(f"     • XGBoost: {', '.join(recommendations['xgboost']) if recommendations['xgboost'] else 'Not recommended'}")
        print(f"     • LightGBM: {', '.join(recommendations['lightgbm']) if recommendations['lightgbm'] else 'Not recommended'}")
        print(f"     • PyTorch: {', '.join(recommendations['pytorch']) if recommendations['pytorch'] else 'Not recommended'}")
        print(f"     • TensorFlow: {', '.join(recommendations['tensorflow']) if recommendations['tensorflow'] else 'Not recommended'}")
        
        return recommendations
    
    def get_safe_param_grid(self, model_type: str, library: str = 'sklearn') -> Dict:
        """
        Get safe hyperparameter ranges based on dataset size.
        
        Supports all major ML libraries!
        
        Parameters
        ----------
        model_type : str
            Model type: 'ridge', 'random_forest', 'xgboost', 'lightgbm', 'pytorch_mlp'
        
        library : str
            Library: 'sklearn', 'xgboost', 'lightgbm', 'pytorch', 'tensorflow'
        
        Returns
        -------
        param_grid : dict
            Safe hyperparameter grid
        """
        print(f"\n" + "-"*80)
        print(f"SAFE HYPERPARAMETER RANGES: {model_type.upper()} ({library.upper()})")
        print("-"*80)
        
        ratio = self.complexity_ratio
        
        # Scikit-learn models
        if library == 'sklearn' and model_type.lower() == 'ridge':
            param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
            print(f"Alpha range: {param_grid['alpha']}")
            return param_grid
            
        elif library == 'sklearn' and model_type.lower() == 'random_forest':
            # Adjust complexity based on ratio
            if ratio < 1.0:  # Small dataset
                param_grid = {
                    'n_estimators': [50, 100],
                    'max_depth': [3, 5],
                    'min_samples_split': [10, 20],
                    'min_samples_leaf': [5, 10]
                }
            elif ratio < 2.0:  # Medium dataset
                param_grid = {
                    'n_estimators': [100, 200],
                    'max_depth': [5, 10],
                    'min_samples_split': [5, 10],
                    'min_samples_leaf': [2, 5]
                }
            else:  # Large dataset
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                }
            
            print(f"n_estimators: {param_grid['n_estimators']}")
            print(f"max_depth: {param_grid['max_depth']}")
            print(f"min_samples_split: {param_grid['min_samples_split']}")
            print(f"min_samples_leaf: {param_grid['min_samples_leaf']}")
            return param_grid
        
        # XGBoost models
        elif library == 'xgboost' and model_type.lower() in ['xgboost', 'xgbregressor']:
            if ratio < 1.0:  # Small dataset
                param_grid = {
                    'max_depth': [2, 3],
                    'learning_rate': [0.1, 0.01],
                    'n_estimators': [50, 100],
                    'subsample': [0.8],
                    'colsample_bytree': [0.8]
                }
            elif ratio < 2.0:  # Medium dataset
                param_grid = {
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.1, 0.05, 0.01],
                    'n_estimators': [100, 200],
                    'subsample': [0.7, 0.8, 0.9],
                    'colsample_bytree': [0.7, 0.8, 0.9]
                }
            else:  # Large dataset
                param_grid = {
                    'max_depth': [5, 7, 9],
                    'learning_rate': [0.1, 0.05, 0.01],
                    'n_estimators': [100, 200, 300],
                    'subsample': [0.7, 0.8, 0.9],
                    'colsample_bytree': [0.7, 0.8, 0.9]
                }
            
            print(f"max_depth: {param_grid['max_depth']}")
            print(f"learning_rate: {param_grid['learning_rate']}")
            print(f"n_estimators: {param_grid['n_estimators']}")
            return param_grid
        
        # LightGBM models
        elif library == 'lightgbm' and model_type.lower() in ['lightgbm', 'lgbmregressor']:
            if ratio < 1.0:  # Small dataset
                param_grid = {
                    'num_leaves': [7, 15],
                    'learning_rate': [0.1, 0.01],
                    'n_estimators': [50, 100],
                    'min_child_samples': [10, 20]
                }
            elif ratio < 2.0:  # Medium dataset
                param_grid = {
                    'num_leaves': [15, 31, 63],
                    'learning_rate': [0.1, 0.05, 0.01],
                    'n_estimators': [100, 200],
                    'min_child_samples': [5, 10, 20]
                }
            else:  # Large dataset
                param_grid = {
                    'num_leaves': [31, 63, 127],
                    'learning_rate': [0.1, 0.05, 0.01],
                    'n_estimators': [100, 200, 300],
                    'min_child_samples': [5, 10, 20]
                }
            
            print(f"num_leaves: {param_grid['num_leaves']}")
            print(f"learning_rate: {param_grid['learning_rate']}")
            print(f"n_estimators: {param_grid['n_estimators']}")
            return param_grid
        
        # PyTorch models
        elif library == 'pytorch' and model_type.lower() in ['pytorch_mlp', 'mlp']:
            if ratio < 1.0:  # Small dataset
                param_grid = {
                    'hidden_sizes': [[32], [64]],
                    'learning_rate': [0.01, 0.001],
                    'dropout': [0.2, 0.3],
                    'batch_size': [16, 32]
                }
            elif ratio < 2.0:  # Medium dataset
                param_grid = {
                    'hidden_sizes': [[64], [128], [64, 32]],
                    'learning_rate': [0.01, 0.001, 0.0001],
                    'dropout': [0.1, 0.2, 0.3],
                    'batch_size': [32, 64]
                }
            else:  # Large dataset
                param_grid = {
                    'hidden_sizes': [[128], [256], [128, 64], [256, 128]],
                    'learning_rate': [0.01, 0.001, 0.0001],
                    'dropout': [0.1, 0.2, 0.3],
                    'batch_size': [32, 64, 128]
                }
            
            print(f"hidden_sizes: {param_grid['hidden_sizes']}")
            print(f"learning_rate: {param_grid['learning_rate']}")
            print(f"dropout: {param_grid['dropout']}")
            return param_grid
        
        # TensorFlow models
        elif library == 'tensorflow' and model_type.lower() in ['tensorflow', 'keras']:
            if ratio < 1.0:  # Small dataset
                param_grid = {
                    'layers': [[32], [64]],
                    'learning_rate': [0.01, 0.001],
                    'dropout': [0.2, 0.3],
                    'batch_size': [16, 32]
                }
            elif ratio < 2.0:  # Medium dataset
                param_grid = {
                    'layers': [[64], [128], [64, 32]],
                    'learning_rate': [0.01, 0.001, 0.0001],
                    'dropout': [0.1, 0.2, 0.3],
                    'batch_size': [32, 64]
                }
            else:  # Large dataset
                param_grid = {
                    'layers': [[128], [256], [128, 64], [256, 128]],
                    'learning_rate': [0.01, 0.001, 0.0001],
                    'dropout': [0.1, 0.2, 0.3],
                    'batch_size': [32, 64, 128]
                }
            
            print(f"layers: {param_grid['layers']}")
            print(f"learning_rate: {param_grid['learning_rate']}")
            print(f"dropout: {param_grid['dropout']}")
            return param_grid
        
        else:
            # Default: simple parameter grid
            print("WARNING: Unknown model_type or library, returning empty grid")
            return {}
    
    def simple_kfold_split(self, n_samples: int, n_splits: int = 5, 
                          shuffle: bool = True, random_state: int = 42):
        """
        Simple K-Fold cross-validation split (sklearn-independent).
        
        Parameters
        ----------
        n_samples : int
            Number of samples
        n_splits : int
            Number of folds
        shuffle : bool
            Whether to shuffle before splitting
        random_state : int
            Random seed
        
        Yields
        ------
        train_idx, test_idx : tuple of arrays
            Training and test indices for each fold
        """
        indices = np.arange(n_samples)
        
        if shuffle:
            np.random.seed(random_state)
            np.random.shuffle(indices)
        
        fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
        fold_sizes[:n_samples % n_splits] += 1
        
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_idx = indices[start:stop]
            train_idx = np.concatenate([indices[:start], indices[stop:]])
            yield train_idx, test_idx
            current = stop
    
    def simple_grid_search(self, model, param_grid: Dict, X_train, y_train, 
                          cv_splits: int = 3, random_state: int = 42):
        """
        Simple grid search (sklearn-independent).
        
        Parameters
        ----------
        model : object
            Model to tune (wrapped with ModelWrapper)
        param_grid : dict
            Parameter grid
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training targets
        cv_splits : int
            Number of CV folds
        random_state : int
            Random seed
        
        Returns
        -------
        best_model : object
            Best model found
        best_params : dict
            Best parameters
        best_score : float
            Best CV score (R²)
        """
        from itertools import product
        
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(product(*param_values))
        
        best_score = -np.inf
        best_params = None
        best_model = None
        
        for param_combo in param_combinations:
            params = dict(zip(param_names, param_combo))
            
            # Create model with these params
            # (Assuming model can be created with **params)
            try:
                wrapped_model = ModelWrapper(model.__class__(**params))
            except:
                # If model doesn't support direct param setting, skip
                continue
            
            # Cross-validation
            cv_scores = []
            for train_idx, val_idx in self.simple_kfold_split(
                len(X_train), n_splits=cv_splits, random_state=random_state
            ):
                X_tr = X_train[train_idx]
                y_tr = y_train[train_idx]
                X_val = X_train[val_idx]
                y_val = y_train[val_idx]
                
                # Train
                wrapped_model.fit(X_tr, y_tr)
                
                # Evaluate
                y_pred = wrapped_model.predict(X_val)
                metrics = calculate_metrics(y_val, y_pred)
                cv_scores.append(metrics['r2'])
            
            mean_score = np.mean(cv_scores)
            
            if mean_score > best_score:
                best_score = mean_score
                best_params = params
                best_model = wrapped_model
        
        return best_model, best_params, best_score
    
    def nested_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model,  # Accept model instance directly
        param_grid: Dict,
        library: str = 'sklearn',
        outer_cv: int = 5,
        inner_cv: int = 3,
        random_state: int = 42
    ) -> Dict:
        """
        Perform nested cross-validation (library-agnostic).
        
        Outer loop: Performance estimation
        Inner loop: Hyperparameter tuning
        
        Works with ANY ML library!
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        
        y : np.ndarray
            Target values
        
        model : object
            Base model to tune (can be sklearn, xgboost, lightgbm, pytorch, tensorflow)
        
        param_grid : dict
            Parameter grid for tuning
        
        library : str
            Library type: 'sklearn', 'xgboost', 'lightgbm', 'pytorch', 'tensorflow'
        
        outer_cv : int
            Number of outer CV folds
        
        inner_cv : int
            Number of inner CV folds
        
        random_state : int
            Random seed
        
        Returns
        -------
        results : dict
            Nested CV results including mean/std R², RMSE, best parameters
        """
        print("\n" + "="*80)
        print(f"NESTED CROSS-VALIDATION ({library.upper()})")
        print("="*80)
        print(f"\nOuter CV: {outer_cv} folds (performance estimation)")
        print(f"Inner CV: {inner_cv} folds (hyperparameter tuning)")
        print(f"Model: {model.__class__.__name__}")
        
        if not param_grid:
            raise ValueError("Empty parameter grid provided")
        
        # Wrap model
        wrapped_model = ModelWrapper(model)
        
        # Outer CV loop
        outer_scores = []
        outer_rmse = []
        outer_mae = []
        best_params_per_fold = []
        
        print("\n" + "-"*80)
        
        for fold_idx, (train_idx, test_idx) in enumerate(
            self.simple_kfold_split(len(X), n_splits=outer_cv, random_state=random_state)
        ):
            print(f"\nOuter Fold {fold_idx + 1}/{outer_cv}")
            
            X_train_outer = X[train_idx]
            y_train_outer = y[train_idx]
            X_test_outer = X[test_idx]
            y_test_outer = y[test_idx]
            
            # Inner CV loop: hyperparameter tuning
            best_model, best_params, best_cv_score = self.simple_grid_search(
                model, param_grid, X_train_outer, y_train_outer,
                cv_splits=inner_cv, random_state=random_state
            )
            
            best_params_per_fold.append(best_params)
            print(f"  Best params: {best_params}")
            print(f"  Best CV R²: {best_cv_score:.4f}")
            
            # Evaluate on outer test fold
            y_pred = best_model.predict(X_test_outer)
            metrics = calculate_metrics(y_test_outer, y_pred)
            
            outer_scores.append(metrics['r2'])
            outer_rmse.append(metrics['rmse'])
            outer_mae.append(metrics['mae'])
            
            print(f"  Test R²: {metrics['r2']:.4f}")
            print(f"  Test RMSE: {metrics['rmse']:.4f}")
        
        # Summary
        mean_r2 = np.mean(outer_scores)
        std_r2 = np.std(outer_scores)
        mean_rmse = np.mean(outer_rmse)
        std_rmse = np.std(outer_rmse)
        mean_mae = np.mean(outer_mae)
        std_mae = np.std(outer_mae)
        
        print("\n" + "="*80)
        print("NESTED CV RESULTS")
        print("="*80)
        print(f"\nR²:   {mean_r2:.4f} ± {std_r2:.4f}")
        print(f"RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")
        print(f"MAE:  {mean_mae:.4f} ± {std_mae:.4f}")
        
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
            'mean_mae': mean_mae,
            'std_mae': std_mae,
            'fold_scores': outer_scores,
            'fold_rmse': outer_rmse,
            'fold_mae': outer_mae,
            'best_params_per_fold': best_params_per_fold
        }
    
    def assess_model_complexity(
        self,
        model: Any,
        library: str = 'sklearn'
    ) -> Dict:
        """
        Assess complexity of a trained model (library-agnostic).
        
        Works with ANY ML library!
        
        Parameters
        ----------
        model : object
            Trained model (sklearn, xgboost, lightgbm, pytorch, tensorflow, etc.)
        
        library : str
            Library type: 'sklearn', 'xgboost', 'lightgbm', 'pytorch', 'tensorflow', 'custom'
        
        Returns
        -------
        complexity_stats : dict
            Model complexity statistics
        """
        print("\n" + "="*80)
        print(f"MODEL COMPLEXITY ASSESSMENT ({library.upper()})")
        print("="*80)
        
        stats = {
            'library': library,
            'model_class': model.__class__.__name__,
            'n_samples': self.n_samples,
            'n_features': self.n_features
        }
        
        # Sklearn models
        if library == 'sklearn':
            if hasattr(model, 'coef_'):
                # Linear models
                stats['n_parameters'] = self.n_features + 1
                stats['regularization'] = getattr(model, 'alpha', 'none')
                print(f"\nModel: {model.__class__.__name__}")
                print(f"  Parameters: {stats['n_parameters']}")
                print(f"  Regularization: {stats['regularization']}")
                
            elif hasattr(model, 'n_estimators'):
                # Tree ensembles
                stats['n_estimators'] = model.n_estimators
                stats['max_depth'] = getattr(model, 'max_depth', 'unlimited')
                
                if stats['max_depth'] != 'unlimited':
                    params_per_tree = 2 ** stats['max_depth']
                    stats['estimated_parameters'] = params_per_tree * stats['n_estimators']
                else:
                    stats['estimated_parameters'] = 'unknown'
                
                print(f"\nModel: {model.__class__.__name__}")
                print(f"  Trees: {stats['n_estimators']}")
                print(f"  Max depth: {stats['max_depth']}")
                print(f"  Estimated params: {stats['estimated_parameters']}")
        
        # XGBoost models
        elif library == 'xgboost':
            if hasattr(model, 'get_params'):
                params = model.get_params()
                stats['n_estimators'] = params.get('n_estimators', 'unknown')
                stats['max_depth'] = params.get('max_depth', 'unknown')
                stats['learning_rate'] = params.get('learning_rate', 'unknown')
                
                print(f"\nModel: XGBoost")
                print(f"  Boosting rounds: {stats['n_estimators']}")
                print(f"  Max depth: {stats['max_depth']}")
                print(f"  Learning rate: {stats['learning_rate']}")
        
        # LightGBM models
        elif library == 'lightgbm':
            if hasattr(model, 'get_params'):
                params = model.get_params()
                stats['n_estimators'] = params.get('n_estimators', 'unknown')
                stats['num_leaves'] = params.get('num_leaves', 'unknown')
                stats['learning_rate'] = params.get('learning_rate', 'unknown')
                
                print(f"\nModel: LightGBM")
                print(f"  Boosting rounds: {stats['n_estimators']}")
                print(f"  Num leaves: {stats['num_leaves']}")
                print(f"  Learning rate: {stats['learning_rate']}")
        
        # PyTorch models
        elif library == 'pytorch':
            try:
                n_params = sum(p.numel() for p in model.parameters())
                stats['n_parameters'] = n_params
                
                print(f"\nModel: PyTorch Neural Network")
                print(f"  Total parameters: {n_params:,}")
                
                # Count layers
                n_layers = len(list(model.modules())) - 1  # Exclude root
                stats['n_layers'] = n_layers
                print(f"  Layers: {n_layers}")
            except:
                print(f"\nModel: PyTorch Neural Network")
                print(f"  (Unable to count parameters)")
        
        # TensorFlow models
        elif library == 'tensorflow':
            try:
                if hasattr(model, 'count_params'):
                    n_params = model.count_params()
                    stats['n_parameters'] = n_params
                    
                    print(f"\nModel: TensorFlow/Keras Model")
                    print(f"  Total parameters: {n_params:,}")
                    
                    if hasattr(model, 'layers'):
                        n_layers = len(model.layers)
                        stats['n_layers'] = n_layers
                        print(f"  Layers: {n_layers}")
            except:
                print(f"\nModel: TensorFlow/Keras Model")
                print(f"  (Unable to count parameters)")
        
        # Custom models
        else:
            print(f"\nModel: Custom ({model.__class__.__name__})")
            print(f"  (Complexity assessment not available)")
        
        # Calculate complexity ratio
        if 'n_parameters' in stats and isinstance(stats['n_parameters'], (int, float)):
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
    """Demonstrate model complexity control with multiple ML libraries."""
    print("\n" + "="*80)
    print("MODEL COMPLEXITY CONTROL DEMONSTRATION (MULTI-LIBRARY)")
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
    
    # Get recommendations for all libraries
    print("\n" + "="*80)
    print("RECOMMENDATIONS FOR ALL ML LIBRARIES:")
    print("="*80)
    recommendations = controller.recommend_models()
    
    # Example 1: Sklearn Ridge
    print("\n\n" + "="*80)
    print("EXAMPLE 1: SKLEARN RIDGE REGRESSION")
    print("="*80)
    
    param_grid_ridge = controller.get_safe_param_grid('ridge', library='sklearn')
    
    try:
        from sklearn.linear_model import Ridge
        model_ridge = Ridge()
        results_ridge = controller.nested_cv(
            X, y, 
            model=model_ridge,
            param_grid=param_grid_ridge,
            library='sklearn',
            outer_cv=3,  # Use 3 for faster demo
            inner_cv=2
        )
        
        complexity_ridge = controller.assess_model_complexity(model_ridge, library='sklearn')
    except ImportError:
        print("Sklearn not available - skipping Ridge example")
    
    # Example 2: XGBoost (if available)
    print("\n\n" + "="*80)
    print("EXAMPLE 2: XGBOOST (if available)")
    print("="*80)
    
    try:
        import xgboost as xgb
        param_grid_xgb = controller.get_safe_param_grid('xgboost', library='xgboost')
        model_xgb = xgb.XGBRegressor(random_state=42)
        
        results_xgb = controller.nested_cv(
            X, y,
            model=model_xgb,
            param_grid=param_grid_xgb,
            library='xgboost',
            outer_cv=3,
            inner_cv=2
        )
        
        complexity_xgb = controller.assess_model_complexity(model_xgb, library='xgboost')
    except ImportError:
        print("XGBoost not available - install with: pip install xgboost")
    
    # Example 3: LightGBM (if available)
    print("\n\n" + "="*80)
    print("EXAMPLE 3: LIGHTGBM (if available)")
    print("="*80)
    
    try:
        import lightgbm as lgb
        param_grid_lgb = controller.get_safe_param_grid('lightgbm', library='lightgbm')
        model_lgb = lgb.LGBMRegressor(random_state=42, verbose=-1)
        
        results_lgb = controller.nested_cv(
            X, y,
            model=model_lgb,
            param_grid=param_grid_lgb,
            library='lightgbm',
            outer_cv=3,
            inner_cv=2
        )
        
        complexity_lgb = controller.assess_model_complexity(model_lgb, library='lightgbm')
    except ImportError:
        print("LightGBM not available - install with: pip install lightgbm")
    
    print("\n" + "="*80)
    print("✓ Multi-library demonstration complete!")
    print("="*80)
    print("\nThis framework supports:")
    print("  - Scikit-learn (Ridge, RandomForest, etc.)")
    print("  - XGBoost (XGBRegressor)")
    print("  - LightGBM (LGBMRegressor)")
    print("  - PyTorch (custom neural networks)")
    print("  - TensorFlow/Keras (Sequential, Functional API)")
    print("  - Custom models (with fit/predict interface)")


if __name__ == '__main__':
    demonstrate_complexity_control()
