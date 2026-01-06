"""
Example: Custom Validation Workflow
====================================

This example shows how to use individual validation modules
for a custom validation workflow.
"""

from src.qsar_validation import (
    DatasetBiasAnalyzer,
    ActivityCliffDetector,
    ModelComplexityAnalyzer,
    PerformanceMetricsCalculator,
    YRandomizationTester
)

def custom_validation_workflow(df, X, y, model):
    """
    Custom validation workflow using individual modules.
    
    Args:
        df: DataFrame with SMILES and activity
        X: Feature matrix
        y: Target values
        model: Trained model instance
    """
    
    print("=" * 70)
    print("CUSTOM VALIDATION WORKFLOW")
    print("=" * 70)
    
    # Step 1: Analyze dataset bias
    print("\n[STEP 1] Dataset Bias Analysis")
    bias_analyzer = DatasetBiasAnalyzer('SMILES', 'Activity')
    diversity = bias_analyzer.analyze_scaffold_diversity(df)
    distribution = bias_analyzer.analyze_activity_distribution(df)
    
    # Step 2: Check for activity cliffs
    print("\n[STEP 2] Activity Cliff Detection")
    cliff_detector = ActivityCliffDetector('SMILES', 'Activity')
    cliffs = cliff_detector.detect_activity_cliffs(df)
    
    # Step 3: Analyze model complexity
    print("\n[STEP 3] Model Complexity Analysis")
    ModelComplexityAnalyzer.analyze_complexity(
        n_samples=len(X),
        n_features=X.shape[1],
        model_type='random_forest'
    )
    
    # Step 4: Calculate performance metrics
    print("\n[STEP 4] Performance Metrics")
    y_pred = model.predict(X)
    metrics = PerformanceMetricsCalculator.calculate_all_metrics(
        y_true=y,
        y_pred=y_pred,
        set_name="Test"
    )
    
    # Step 5: Y-randomization test
    print("\n[STEP 5] Y-Randomization Test")
    rand_results = YRandomizationTester.perform_y_randomization(
        X=X,
        y=y,
        model=model,
        n_iterations=10
    )
    
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    
    return {
        'diversity': diversity,
        'cliffs': cliffs,
        'metrics': metrics,
        'randomization': rand_results
    }


if __name__ == "__main__":
    print("Example script - integrate with your model training pipeline")
