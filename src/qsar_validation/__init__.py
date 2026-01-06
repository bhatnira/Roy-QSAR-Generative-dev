"""
QSAR Validation Package
=======================

A modular, model-agnostic, and featurizer-agnostic framework for comprehensive QSAR validation.

Modules:
--------
- dataset_analysis: Dataset bias and representativeness analysis
- activity_cliffs: Activity cliff detection
- model_complexity: Model complexity and overfitting control
- metrics: Performance metrics calculation
- randomization: Y-randomization testing
- assay_noise: Experimental error estimation
- validation_runner: Main validation orchestrator
- model_agnostic_pipeline: Complete model-agnostic pipeline (NEW!)

Usage:
------
    # NEW: Model-Agnostic Pipeline (Recommended)
    from qsar_validation import ModelAgnosticQSARPipeline
    
    pipeline = ModelAgnosticQSARPipeline(
        featurizer=my_featurizer_function,  # Your choice!
        model=my_sklearn_model,             # Your choice!
        smiles_col='SMILES',
        target_col='Activity'
    )
    results = pipeline.fit_predict_validate(df)
    
    # Traditional: Individual modules
    from qsar_validation import run_comprehensive_validation
    results = run_comprehensive_validation(df, smiles_col='SMILES', target_col='IC50')

Author: QSAR Validation Framework
Date: January 2026
Version: 3.0.0 (Model-Agnostic)
"""

__version__ = "3.0.0"
__author__ = "QSAR Validation Framework"

# Import main classes and functions for easy access
from .dataset_analysis import DatasetBiasAnalyzer
from .activity_cliffs import ActivityCliffDetector
from .model_complexity import ModelComplexityAnalyzer
from .metrics import PerformanceMetricsCalculator
from .randomization import YRandomizationTester
from .assay_noise import AssayNoiseEstimator
from .validation_runner import (
    run_comprehensive_validation,
    print_comprehensive_validation_checklist
)
from .model_agnostic_pipeline import ModelAgnosticQSARPipeline

__all__ = [
    # Model-Agnostic Pipeline (NEW - Recommended)
    'ModelAgnosticQSARPipeline',
    
    # Individual Modules
    'DatasetBiasAnalyzer',
    'ActivityCliffDetector',
    'ModelComplexityAnalyzer',
    'PerformanceMetricsCalculator',
    'YRandomizationTester',
    'AssayNoiseEstimator',
    'run_comprehensive_validation',
    'print_comprehensive_validation_checklist',
]
