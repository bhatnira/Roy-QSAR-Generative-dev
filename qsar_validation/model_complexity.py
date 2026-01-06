"""
Model Complexity Analysis Module
=================================

Analyzes model complexity vs dataset size to prevent overfitting.
"""

from typing import Dict


class ModelComplexityAnalyzer:
    """
    Analyze model complexity vs dataset size to prevent overfitting.
    Provides guidelines for appropriate model selection based on sample-to-feature ratio.
    """
    
    @staticmethod
    def analyze_complexity(n_samples: int, n_features: int, model_type: str) -> None:
        """
        Analyze if model complexity is appropriate for dataset size.
        
        Args:
            n_samples: Number of samples in dataset
            n_features: Number of features
            model_type: Type of model being used (for specific recommendations)
        """
        print("\n[INFO] MODEL COMPLEXITY ANALYSIS")
        print("=" * 70)
        
        ratio = n_samples / n_features
        
        print(f"Samples: {n_samples}")
        print(f"Features: {n_features}")
        print(f"Samples-to-features ratio: {ratio:.2f}")
        
        # General guidelines based on ratio
        ModelComplexityAnalyzer._print_general_guidelines(ratio)
        
        # Model-specific recommendations
        ModelComplexityAnalyzer._print_model_recommendations(model_type, ratio)
        
        # General best practices
        ModelComplexityAnalyzer._print_best_practices()
    
    @staticmethod
    def _print_general_guidelines(ratio: float) -> None:
        """Print general guidelines based on sample-to-feature ratio."""
        if ratio < 5:
            print("\n[CRITICAL] CRITICAL: Very low samples-to-features ratio (< 5)")
            print("   -> High overfitting risk")
            print("   -> REQUIRED: Strong regularization")
            print("   -> RECOMMENDED: Feature selection or dimensionality reduction")
            print("   -> AVOID: Complex models (deep learning, unregularized ensemble)")
        elif ratio < 10:
            print("\n[HIGH PRIORITY] WARNING: Low samples-to-features ratio (< 10)")
            print("   -> Moderate overfitting risk")
            print("   -> REQUIRED: Regularization (Ridge, Lasso, ElasticNet)")
            print("   -> RECOMMENDED: Simple models (linear, regularized)")
        elif ratio < 20:
            print("\n[MODERATE] CAUTION: Modest samples-to-features ratio (< 20)")
            print("   -> Use cross-validation carefully")
            print("   -> RECOMMENDED: Regularized models")
        else:
            print("\n[OK] Adequate samples-to-features ratio")
    
    @staticmethod
    def _print_model_recommendations(model_type: str, ratio: float) -> None:
        """Print model-specific recommendations."""
        print(f"\n[NOTE] Recommendations for {model_type}:")
        
        recommendations = {
            'deep_learning': {
                'min_ratio': 50,
                'advice': [
                    "Use pre-trained embeddings (ChEBERTa, MolBERT)",
                    "Strong dropout (0.3-0.5)",
                    "Early stopping",
                    "Very simple architecture (1-2 layers)",
                ]
            },
            'random_forest': {
                'min_ratio': 10,
                'advice': [
                    "Limit max_depth (3-5)",
                    "Increase min_samples_leaf (5-10)",
                    "Use max_features='sqrt' or 'log2'",
                    "Limit n_estimators (50-100)",
                ]
            },
            'gradient_boosting': {
                'min_ratio': 10,
                'advice': [
                    "Low learning_rate (0.01-0.05)",
                    "Limit max_depth (2-4)",
                    "Use subsample (0.5-0.8)",
                    "Early stopping based on validation",
                ]
            },
            'gaussian_process': {
                'min_ratio': 5,
                'advice': [
                    "Good for small datasets",
                    "Provides uncertainty estimates",
                    "Watch for kernel overfitting",
                    "Consider noise parameter",
                ]
            },
            'linear': {
                'min_ratio': 3,
                'advice': [
                    "Use Ridge (L2) or Lasso (L1)",
                    "Cross-validate alpha parameter",
                    "ElasticNet for mixed regularization",
                    "Works well with limited data",
                ]
            },
        }
        
        model_key = model_type.lower()
        if model_key in recommendations:
            rec = recommendations[model_key]
            if ratio < rec['min_ratio']:
                print(f"   [WARNING]  Dataset size below recommended minimum ({rec['min_ratio']}:1)")
            
            print("   Best practices:")
            for advice in rec['advice']:
                print(f"   • {advice}")
        else:
            print(f"   No specific recommendations for '{model_type}'")
            print("   Follow general guidelines above")
    
    @staticmethod
    def _print_best_practices() -> None:
        """Print general best practices."""
        print("\n[NOTE] GENERAL BEST PRACTICES:")
        print("   • Use nested cross-validation")
        print("   • Report validation metrics (not just training)")
        print("   • Compare to simple baseline (Ridge regression)")
        print("   • Perform y-randomization test")
