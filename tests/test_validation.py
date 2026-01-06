"""
Unit Tests for QSAR Validation Package
=======================================

Run with: pytest tests/
"""

import pytest
import numpy as np
import pandas as pd
from rdkit import Chem

# Test fixtures
@pytest.fixture
def sample_dataframe():
    """Create sample dataset for testing."""
    smiles = [
        'CCO',  # ethanol
        'CC(C)O',  # isopropanol
        'CCCC',  # butane
        'c1ccccc1',  # benzene
        'c1ccccc1O',  # phenol
    ]
    activities = [2.5, 2.8, 1.2, 3.5, 3.2]
    
    return pd.DataFrame({
        'SMILES': smiles,
        'Activity': activities
    })


@pytest.fixture
def sample_features():
    """Create sample feature matrix."""
    return np.random.rand(100, 50)


@pytest.fixture
def sample_targets():
    """Create sample target values."""
    return np.random.rand(100)


# Tests for DatasetBiasAnalyzer
def test_dataset_bias_analyzer_import():
    """Test that DatasetBiasAnalyzer can be imported."""
    from src.qsar_validation import DatasetBiasAnalyzer
    assert DatasetBiasAnalyzer is not None


def test_scaffold_diversity(sample_dataframe):
    """Test scaffold diversity analysis."""
    from src.qsar_validation import DatasetBiasAnalyzer
    
    analyzer = DatasetBiasAnalyzer('SMILES', 'Activity')
    result = analyzer.analyze_scaffold_diversity(sample_dataframe)
    
    assert 'n_molecules' in result
    assert 'n_scaffolds' in result
    assert 'diversity_ratio' in result
    assert result['n_molecules'] == len(sample_dataframe)


# Tests for ActivityCliffDetector
def test_activity_cliff_detector_import():
    """Test that ActivityCliffDetector can be imported."""
    from src.qsar_validation import ActivityCliffDetector
    assert ActivityCliffDetector is not None


# Tests for ModelComplexityAnalyzer
def test_model_complexity_analyzer_import():
    """Test that ModelComplexityAnalyzer can be imported."""
    from src.qsar_validation import ModelComplexityAnalyzer
    assert ModelComplexityAnalyzer is not None


def test_complexity_analysis():
    """Test model complexity analysis."""
    from src.qsar_validation import ModelComplexityAnalyzer
    
    # Should not raise exception
    ModelComplexityAnalyzer.analyze_complexity(
        n_samples=100,
        n_features=50,
        model_type='random_forest'
    )


# Tests for PerformanceMetricsCalculator
def test_metrics_calculator_import():
    """Test that PerformanceMetricsCalculator can be imported."""
    from src.qsar_validation import PerformanceMetricsCalculator
    assert PerformanceMetricsCalculator is not None


def test_metrics_calculation(sample_targets):
    """Test performance metrics calculation."""
    from src.qsar_validation import PerformanceMetricsCalculator
    
    y_true = sample_targets
    y_pred = sample_targets + np.random.randn(len(sample_targets)) * 0.1
    
    metrics = PerformanceMetricsCalculator.calculate_all_metrics(
        y_true, y_pred, set_name="Test"
    )
    
    assert 'rmse' in metrics
    assert 'mae' in metrics
    assert 'r2' in metrics
    assert 'pearson_r' in metrics


# Integration test
def test_comprehensive_validation(sample_dataframe):
    """Test comprehensive validation function."""
    from src.qsar_validation import run_comprehensive_validation
    
    results = run_comprehensive_validation(
        sample_dataframe,
        smiles_col='SMILES',
        target_col='Activity'
    )
    
    assert 'scaffold_diversity' in results
    assert 'activity_distribution' in results
    assert 'activity_cliffs' in results
    assert 'experimental_error' in results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
