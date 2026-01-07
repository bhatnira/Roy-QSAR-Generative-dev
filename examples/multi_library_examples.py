"""
Multi-Library Support Examples
==============================

Demonstrate how to use the QSAR framework with different ML libraries.

This shows that the framework is NOT limited to sklearn - you can use:
- Scikit-learn
- XGBoost
- LightGBM
- PyTorch
- TensorFlow/Keras
- Custom models

All with the same consistent API!
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from qsar_validation.model_complexity_control import ModelComplexityController


def example_1_sklearn():
    """Example 1: Sklearn Ridge Regression."""
    print("\n" + "="*80)
    print("EXAMPLE 1: SKLEARN RIDGE REGRESSION")
    print("="*80)
    
    try:
        from sklearn.linear_model import Ridge
        from sklearn.ensemble import RandomForestRegressor
        
        # Generate sample data (small dataset)
        np.random.seed(42)
        X = np.random.randn(100, 500)  # 100 samples, 500 features
        y = np.random.randn(100)
        
        # Create controller
        controller = ModelComplexityController(
            n_samples=X.shape[0],
            n_features=X.shape[1]
        )
        
        # Check recommendations
        recommendations = controller.recommend_models()
        print("\nRecommended sklearn models:")
        for model_name in recommendations.get('sklearn', []):
            print(f"  ✓ {model_name}")
        
        # Get safe parameter grid
        param_grid = controller.get_safe_param_grid('ridge', library='sklearn')
        print(f"\nSafe parameter grid: {param_grid}")
        
        # Create model
        model = Ridge()
        
        # Run nested CV (reduced folds for speed)
        print("\nRunning nested CV...")
        results = controller.nested_cv(
            X, y,
            model=model,
            param_grid=param_grid,
            library='sklearn',
            outer_cv=3,  # Use 3 for faster demo
            inner_cv=2   # Use 2 for faster demo
        )
        
        # Assess complexity
        model.fit(X, y)  # Fit for assessment
        complexity = controller.assess_model_complexity(model, library='sklearn')
        
        print("\n✓ Sklearn Ridge example completed successfully!")
        
    except ImportError as e:
        print(f"Sklearn not available: {e}")
        print("Install with: pip install scikit-learn")


def example_2_xgboost():
    """Example 2: XGBoost Regressor."""
    print("\n" + "="*80)
    print("EXAMPLE 2: XGBOOST REGRESSOR")
    print("="*80)
    
    try:
        import xgboost as xgb
        
        # Generate sample data
        np.random.seed(42)
        X = np.random.randn(100, 500)
        y = np.random.randn(100)
        
        # Create controller
        controller = ModelComplexityController(
            n_samples=X.shape[0],
            n_features=X.shape[1]
        )
        
        # Check recommendations
        recommendations = controller.recommend_models()
        print("\nRecommended XGBoost models:")
        for model_name in recommendations.get('xgboost', []):
            print(f"  ✓ {model_name}")
        
        # Get safe parameter grid (adapted for small dataset)
        param_grid = controller.get_safe_param_grid('xgboost', library='xgboost')
        print(f"\nSafe parameter grid for XGBoost:")
        for param, values in param_grid.items():
            print(f"  {param}: {values}")
        
        # Create model
        model = xgb.XGBRegressor(random_state=42)
        
        # Run nested CV
        print("\nRunning nested CV...")
        results = controller.nested_cv(
            X, y,
            model=model,
            param_grid=param_grid,
            library='xgboost',
            outer_cv=3,
            inner_cv=2
        )
        
        # Assess complexity
        model.fit(X, y)
        complexity = controller.assess_model_complexity(model, library='xgboost')
        
        print("\n✓ XGBoost example completed successfully!")
        
    except ImportError as e:
        print(f"XGBoost not available: {e}")
        print("Install with: pip install xgboost")


def example_3_lightgbm():
    """Example 3: LightGBM Regressor."""
    print("\n" + "="*80)
    print("EXAMPLE 3: LIGHTGBM REGRESSOR")
    print("="*80)
    
    try:
        import lightgbm as lgb
        
        # Generate sample data
        np.random.seed(42)
        X = np.random.randn(100, 500)
        y = np.random.randn(100)
        
        # Create controller
        controller = ModelComplexityController(
            n_samples=X.shape[0],
            n_features=X.shape[1]
        )
        
        # Check recommendations
        recommendations = controller.recommend_models()
        print("\nRecommended LightGBM models:")
        for model_name in recommendations.get('lightgbm', []):
            print(f"  ✓ {model_name}")
        
        # Get safe parameter grid
        param_grid = controller.get_safe_param_grid('lightgbm', library='lightgbm')
        print(f"\nSafe parameter grid for LightGBM:")
        for param, values in param_grid.items():
            print(f"  {param}: {values}")
        
        # Create model
        model = lgb.LGBMRegressor(random_state=42, verbose=-1)
        
        # Run nested CV
        print("\nRunning nested CV...")
        results = controller.nested_cv(
            X, y,
            model=model,
            param_grid=param_grid,
            library='lightgbm',
            outer_cv=3,
            inner_cv=2
        )
        
        # Assess complexity
        model.fit(X, y)
        complexity = controller.assess_model_complexity(model, library='lightgbm')
        
        print("\n✓ LightGBM example completed successfully!")
        
    except ImportError as e:
        print(f"LightGBM not available: {e}")
        print("Install with: pip install lightgbm")


def example_4_pytorch():
    """Example 4: PyTorch Neural Network."""
    print("\n" + "="*80)
    print("EXAMPLE 4: PYTORCH NEURAL NETWORK")
    print("="*80)
    
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        
        # Define custom PyTorch model with fit/predict interface
        class SimpleMLPRegressor(nn.Module):
            """Simple MLP for regression with sklearn-like interface."""
            
            def __init__(self, input_dim, hidden_sizes=[64], dropout=0.2, 
                        learning_rate=0.001, epochs=100):
                super().__init__()
                self.learning_rate = learning_rate
                self.epochs = epochs
                
                layers = []
                prev_dim = input_dim
                for hidden_dim in hidden_sizes:
                    layers.append(nn.Linear(prev_dim, hidden_dim))
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(dropout))
                    prev_dim = hidden_dim
                layers.append(nn.Linear(prev_dim, 1))
                
                self.model = nn.Sequential(*layers)
                self.optimizer = None
            
            def forward(self, x):
                return self.model(x)
            
            def fit(self, X, y):
                """Fit the model (sklearn-like interface)."""
                self.train()
                
                # Convert to tensors
                if not isinstance(X, torch.Tensor):
                    X = torch.FloatTensor(X)
                if not isinstance(y, torch.Tensor):
                    y = torch.FloatTensor(y).reshape(-1, 1)
                
                # Setup optimizer
                self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
                criterion = nn.MSELoss()
                
                # Training loop
                for epoch in range(self.epochs):
                    self.optimizer.zero_grad()
                    outputs = self(X)
                    loss = criterion(outputs, y)
                    loss.backward()
                    self.optimizer.step()
                
                return self
            
            def predict(self, X):
                """Predict (sklearn-like interface)."""
                self.eval()
                
                with torch.no_grad():
                    if not isinstance(X, torch.Tensor):
                        X = torch.FloatTensor(X)
                    predictions = self(X)
                    return predictions.numpy().flatten()
        
        # Generate sample data
        np.random.seed(42)
        X = np.random.randn(100, 500).astype(np.float32)
        y = np.random.randn(100).astype(np.float32)
        
        # Create controller
        controller = ModelComplexityController(
            n_samples=X.shape[0],
            n_features=X.shape[1]
        )
        
        # Check recommendations
        recommendations = controller.recommend_models()
        print("\nRecommended PyTorch architectures:")
        for model_name in recommendations.get('pytorch', []):
            print(f"  ✓ {model_name}")
        
        # Get safe parameter grid
        param_grid = controller.get_safe_param_grid('pytorch_mlp', library='pytorch')
        print(f"\nSafe parameter grid for PyTorch:")
        for param, values in param_grid.items():
            print(f"  {param}: {values}")
        
        # Create model (use smaller network for small dataset)
        model = SimpleMLPRegressor(
            input_dim=500, 
            hidden_sizes=[32],  # Small network for small dataset
            dropout=0.2,
            learning_rate=0.001,
            epochs=50  # Fewer epochs for demo
        )
        
        print("\nNote: Full nested CV with PyTorch is slow for demonstration")
        print("Showing model assessment instead...")
        
        # Fit and assess
        model.fit(X, y)
        complexity = controller.assess_model_complexity(model, library='pytorch')
        
        print("\n✓ PyTorch example completed successfully!")
        
    except ImportError as e:
        print(f"PyTorch not available: {e}")
        print("Install with: pip install torch")


def example_5_tensorflow():
    """Example 5: TensorFlow/Keras."""
    print("\n" + "="*80)
    print("EXAMPLE 5: TENSORFLOW/KERAS")
    print("="*80)
    
    try:
        import tensorflow as tf
        from tensorflow import keras
        
        # Disable TF warnings
        tf.get_logger().setLevel('ERROR')
        
        # Define Keras model builder
        def create_keras_model(input_dim=500, layers=[64], dropout=0.2, 
                              learning_rate=0.001):
            """Create Keras Sequential model."""
            model = keras.Sequential()
            model.add(keras.layers.Input(shape=(input_dim,)))
            
            for units in layers:
                model.add(keras.layers.Dense(units, activation='relu'))
                model.add(keras.layers.Dropout(dropout))
            
            model.add(keras.layers.Dense(1))
            
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                loss='mse'
            )
            
            return model
        
        # Generate sample data
        np.random.seed(42)
        X = np.random.randn(100, 500).astype(np.float32)
        y = np.random.randn(100).astype(np.float32)
        
        # Create controller
        controller = ModelComplexityController(
            n_samples=X.shape[0],
            n_features=X.shape[1]
        )
        
        # Check recommendations
        recommendations = controller.recommend_models()
        print("\nRecommended TensorFlow architectures:")
        for model_name in recommendations.get('tensorflow', []):
            print(f"  ✓ {model_name}")
        
        # Get safe parameter grid
        param_grid = controller.get_safe_param_grid('tensorflow', library='tensorflow')
        print(f"\nSafe parameter grid for TensorFlow:")
        for param, values in param_grid.items():
            print(f"  {param}: {values}")
        
        # Create model
        model = create_keras_model(
            input_dim=500,
            layers=[32],  # Small network for small dataset
            dropout=0.2,
            learning_rate=0.001
        )
        
        print("\nNote: Full nested CV with TensorFlow is slow for demonstration")
        print("Showing model assessment instead...")
        
        # Fit and assess
        model.fit(X, y, epochs=50, verbose=0)
        complexity = controller.assess_model_complexity(model, library='tensorflow')
        
        print("\n✓ TensorFlow example completed successfully!")
        
    except ImportError as e:
        print(f"TensorFlow not available: {e}")
        print("Install with: pip install tensorflow")


def example_6_custom():
    """Example 6: Custom Model."""
    print("\n" + "="*80)
    print("EXAMPLE 6: CUSTOM MODEL")
    print("="*80)
    
    # Define custom model
    class CustomLinearRegressor:
        """Custom linear regression with regularization."""
        
        def __init__(self, alpha=1.0):
            self.alpha = alpha
            self.coef_ = None
            self.intercept_ = 0.0
        
        def fit(self, X, y):
            """Fit using ridge regression formula."""
            n_samples, n_features = X.shape
            
            # Add intercept
            X_with_intercept = np.column_stack([np.ones(n_samples), X])
            
            # Ridge regression: (X'X + αI)^(-1) X'y
            XtX = X_with_intercept.T @ X_with_intercept
            XtX += self.alpha * np.eye(n_features + 1)
            Xty = X_with_intercept.T @ y
            
            params = np.linalg.solve(XtX, Xty)
            
            self.intercept_ = params[0]
            self.coef_ = params[1:]
            
            return self
        
        def predict(self, X):
            """Predict using learned coefficients."""
            return X @ self.coef_ + self.intercept_
    
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(100, 500)
    y = np.random.randn(100)
    
    # Create controller
    controller = ModelComplexityController(
        n_samples=X.shape[0],
        n_features=X.shape[1]
    )
    
    print("\nUsing custom linear regressor with ridge regularization")
    
    # Define parameter grid
    param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
    print(f"\nParameter grid: {param_grid}")
    
    # Create model
    model = CustomLinearRegressor(alpha=1.0)
    
    # Run nested CV
    print("\nRunning nested CV...")
    results = controller.nested_cv(
        X, y,
        model=model,
        param_grid=param_grid,
        library='custom',
        outer_cv=3,
        inner_cv=2
    )
    
    # Assess complexity
    model.fit(X, y)
    complexity = controller.assess_model_complexity(model, library='custom')
    
    print("\n✓ Custom model example completed successfully!")


def compare_all_libraries():
    """Compare all available libraries on the same dataset."""
    print("\n" + "="*80)
    print("LIBRARY COMPARISON")
    print("="*80)
    print("\nComparing all available ML libraries on the same dataset...")
    
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(200, 100)  # Larger dataset for comparison
    true_coef = np.random.randn(100)
    y = X @ true_coef + 0.1 * np.random.randn(200)
    
    # Create controller
    controller = ModelComplexityController(
        n_samples=X.shape[0],
        n_features=X.shape[1]
    )
    
    results_comparison = {}
    
    # Try sklearn
    try:
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=1.0)
        model.fit(X, y)
        pred = model.predict(X)
        r2 = 1 - np.sum((y - pred)**2) / np.sum((y - y.mean())**2)
        results_comparison['sklearn Ridge'] = r2
        print(f"  ✓ sklearn Ridge: R² = {r2:.4f}")
    except ImportError:
        print(f"  ✗ sklearn not available")
    
    # Try XGBoost
    try:
        import xgboost as xgb
        model = xgb.XGBRegressor(max_depth=3, n_estimators=100, random_state=42)
        model.fit(X, y)
        pred = model.predict(X)
        r2 = 1 - np.sum((y - pred)**2) / np.sum((y - y.mean())**2)
        results_comparison['XGBoost'] = r2
        print(f"  ✓ XGBoost: R² = {r2:.4f}")
    except ImportError:
        print(f"  ✗ XGBoost not available")
    
    # Try LightGBM
    try:
        import lightgbm as lgb
        model = lgb.LGBMRegressor(num_leaves=15, n_estimators=100, random_state=42, verbose=-1)
        model.fit(X, y)
        pred = model.predict(X)
        r2 = 1 - np.sum((y - pred)**2) / np.sum((y - y.mean())**2)
        results_comparison['LightGBM'] = r2
        print(f"  ✓ LightGBM: R² = {r2:.4f}")
    except ImportError:
        print(f"  ✗ LightGBM not available")
    
    print("\n" + "-"*80)
    print("Summary:")
    if results_comparison:
        best_library = max(results_comparison, key=results_comparison.get)
        print(f"  Best library: {best_library} (R² = {results_comparison[best_library]:.4f})")
        print(f"\nAll libraries work with the same framework API!")
    else:
        print("  No ML libraries available. Install with:")
        print("  pip install scikit-learn xgboost lightgbm")


if __name__ == '__main__':
    print("\n" + "="*80)
    print("MULTI-LIBRARY SUPPORT EXAMPLES")
    print("="*80)
    print("\nDemonstrating that the QSAR framework works with ANY ML library!")
    print("\nLibraries supported:")
    print("  - Scikit-learn (Ridge, RandomForest, etc.)")
    print("  - XGBoost (XGBRegressor)")
    print("  - LightGBM (LGBMRegressor)")
    print("  - PyTorch (custom neural networks)")
    print("  - TensorFlow/Keras (Sequential, Functional API)")
    print("  - Custom models (any class with fit/predict)")
    
    # Run all examples
    example_1_sklearn()
    example_2_xgboost()
    example_3_lightgbm()
    example_4_pytorch()
    example_5_tensorflow()
    example_6_custom()
    compare_all_libraries()
    
    print("\n" + "="*80)
    print("✓ ALL EXAMPLES COMPLETED!")
    print("="*80)
    print("\nKey takeaway: The framework is library-agnostic!")
    print("You can use ANY ML library you prefer with the same consistent API.")
    print("\nThe framework handles:")
    print("  - Library-specific parameter grids")
    print("  - Library-specific complexity assessment")
    print("  - Library-specific recommendations")
    print("  - Unified interface (ModelWrapper)")
    print("  - Consistent metrics (calculate_metrics)")
    print("  - Universal nested CV")
    print("\nNo vendor lock-in. Maximum flexibility. Your choice!")
