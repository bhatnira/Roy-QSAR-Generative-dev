"""
QSAR Validation Package
=======================

A modular framework for comprehensive QSAR model validation.

Modules:
--------
- dataset_analysis: Dataset bias and representativeness analysis
- activity_cliffs: Activity cliff detection
- model_complexity: Model complexity and overfitting control
- metrics: Performance metrics calculation
- randomization: Y-randomization testing
- assay_noise: Experimental error estimation
- validation_runner: Main validation orchestrator

Usage:
------
    from qsar_validation import run_validation, DatasetAnalyzer
    
    # Quick validation
    results = run_validation(df, smiles_col='SMILES', target_col='IC50')
    
    # Or use individual modules
    from qsar_validation.dataset_analysis import DatasetBiasAnalyzer
    analyzer = DatasetBiasAnalyzer()
    diversity = analyzer.analyze_scaffold_diversity(df)

Author: QSAR Validation Framework
Date: January 2026
Version: 2.0.0 (Modularized)
"""

__version__ = "2.0.0"
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

__all__ = [
    'DatasetBiasAnalyzer',
    'ActivityCliffDetector',
    'ModelComplexityAnalyzer',
    'PerformanceMetricsCalculator',
    'YRandomizationTester',
    'AssayNoiseEstimator',
    'run_comprehensive_validation',
    'print_comprehensive_validation_checklist',
]
