#!/usr/bin/env python3
"""
QSAR Validation Framework - Standalone Workflow
================================================

A complete, notebook-free workflow for QSAR model validation
in the low-data regime (< 200 compounds).

Usage:
    python standalone_qsar_workflow.py --data your_data.csv --smiles SMILES --target Activity

Requirements:
    - pandas, numpy, scikit-learn, rdkit, scipy
    - qsar_validation package (from this repository)
"""

import argparse
import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit.Chem import AllChem

# Import validation framework
try:
    from qsar_validation import (
        DatasetBiasAnalyzer,
        ActivityCliffDetector,
        ModelComplexityAnalyzer,
        PerformanceMetricsCalculator,
        YRandomizationTester,
        AssayNoiseEstimator,
        print_comprehensive_validation_checklist
    )
except ImportError:
    # Fallback to relative import if not installed
    sys.path.append('src')
    from qsar_validation import (
        DatasetBiasAnalyzer,
        ActivityCliffDetector,
        ModelComplexityAnalyzer,
        PerformanceMetricsCalculator,
        YRandomizationTester,
        AssayNoiseEstimator,
        print_comprehensive_validation_checklist
    )


class QSARWorkflow:
    """Complete QSAR modeling workflow with validation."""
    
    def __init__(self, data_path, smiles_col='SMILES', target_col='Activity'):
        """
        Initialize workflow.
        
        Args:
            data_path: Path to CSV file with SMILES and activity
            smiles_col: Name of SMILES column
            target_col: Name of target/activity column
        """
        self.data_path = data_path
        self.smiles_col = smiles_col
        self.target_col = target_col
        self.df = None
        self.X = None
        self.y = None
        self.model = None
        self.results = {}
    
    def load_data(self):
        """Load and inspect dataset."""
        print("\n" + "=" * 70)
        print("STEP 1: LOAD DATASET")
        print("=" * 70)
        
        self.df = pd.read_csv(self.data_path)
        print(f"\nDataset loaded: {len(self.df)} compounds")
        print(f"Columns: {list(self.df.columns)}")
        
        # Check required columns
        if self.smiles_col not in self.df.columns:
            raise ValueError(f"SMILES column '{self.smiles_col}' not found!")
        if self.target_col not in self.df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found!")
        
        # Basic stats
        print(f"\nTarget statistics:")
        print(f"  Mean: {self.df[self.target_col].mean():.3f}")
        print(f"  Std:  {self.df[self.target_col].std():.3f}")
        print(f"  Range: [{self.df[self.target_col].min():.3f}, {self.df[self.target_col].max():.3f}]")
        
        return self.df
    
    def validate_dataset(self):
        """Run comprehensive dataset validation."""
        print("\n" + "=" * 70)
        print("STEP 2: DATASET VALIDATION")
        print("=" * 70)
        
        # Scaffold diversity
        print("\n[2.1] Scaffold Diversity Analysis")
        analyzer = DatasetBiasAnalyzer(self.smiles_col, self.target_col)
        diversity = analyzer.analyze_scaffold_diversity(self.df)
        self.results['scaffold_diversity'] = diversity
        
        # Activity distribution
        print("\n[2.2] Activity Distribution Analysis")
        distribution = analyzer.analyze_activity_distribution(self.df)
        self.results['activity_distribution'] = distribution
        
        # Activity cliffs
        print("\n[2.3] Activity Cliff Detection")
        cliff_detector = ActivityCliffDetector(self.smiles_col, self.target_col)
        cliffs = cliff_detector.detect_activity_cliffs(self.df)
        self.results['activity_cliffs'] = cliffs
        
        # Experimental error
        print("\n[2.4] Experimental Error Estimation")
        noise_estimator = AssayNoiseEstimator()
        exp_error = noise_estimator.estimate_experimental_error(self.df, self.target_col)
        self.results['experimental_error'] = exp_error
        
        return self.results
    
    def generate_features(self, fp_radius=2, fp_bits=2048):
        """
        Generate molecular fingerprints.
        
        Args:
            fp_radius: Morgan fingerprint radius
            fp_bits: Number of bits in fingerprint
        """
        print("\n" + "=" * 70)
        print("STEP 3: FEATURE GENERATION")
        print("=" * 70)
        
        print(f"\nGenerating Morgan fingerprints (radius={fp_radius}, bits={fp_bits})...")
        
        fps = []
        valid_indices = []
        invalid_count = 0
        
        for idx, smi in enumerate(self.df[self.smiles_col]):
            mol = Chem.MolFromSmiles(smi)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, fp_radius, fp_bits)
                fps.append([int(b) for b in fp])
                valid_indices.append(idx)
            else:
                invalid_count += 1
        
        self.X = np.array(fps)
        self.df = self.df.iloc[valid_indices].reset_index(drop=True)
        self.y = self.df[self.target_col].values
        
        print(f"\nFeatures generated: {self.X.shape}")
        print(f"Valid molecules: {len(valid_indices)}")
        if invalid_count > 0:
            print(f"[WARNING] {invalid_count} invalid SMILES removed")
        
        return self.X, self.y
    
    def check_model_complexity(self, model_type='random_forest'):
        """Check if model complexity is appropriate."""
        print("\n" + "=" * 70)
        print("STEP 4: MODEL COMPLEXITY ANALYSIS")
        print("=" * 70)
        
        ModelComplexityAnalyzer.analyze_complexity(
            n_samples=len(self.X),
            n_features=self.X.shape[1],
            model_type=model_type
        )
    
    def train_baseline(self, X_train, X_test, y_train, y_test):
        """Train baseline Ridge regression model."""
        print("\n[5.1] Baseline Model (Ridge Regression)")
        
        baseline = Ridge(alpha=1.0)
        baseline.fit(X_train, y_train)
        y_pred = baseline.predict(X_test)
        
        metrics = PerformanceMetricsCalculator.calculate_all_metrics(
            y_test, y_pred, set_name="Baseline Test"
        )
        
        self.results['baseline_metrics'] = metrics
        return baseline, metrics
    
    def train_model(self, model_type='random_forest', test_size=0.2, random_state=42):
        """
        Train QSAR model with proper validation.
        
        Args:
            model_type: Type of model ('random_forest', 'ridge', 'linear')
            test_size: Test set proportion
            random_state: Random seed for reproducibility
        """
        print("\n" + "=" * 70)
        print("STEP 5: MODEL TRAINING")
        print("=" * 70)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        
        print(f"\nTrain size: {len(X_train)}")
        print(f"Test size:  {len(X_test)}")
        
        # Train baseline first
        baseline, baseline_metrics = self.train_baseline(
            X_train, X_test, y_train, y_test
        )
        
        # Train main model
        print(f"\n[5.2] Main Model ({model_type})")
        
        if model_type == 'random_forest':
            # Regularized Random Forest for low data
            self.model = RandomForestRegressor(
                n_estimators=50,
                max_depth=5,
                min_samples_leaf=5,
                max_features='sqrt',
                random_state=random_state
            )
        elif model_type == 'ridge':
            self.model = Ridge(alpha=1.0)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        
        metrics = PerformanceMetricsCalculator.calculate_all_metrics(
            y_test, y_pred, set_name="Test"
        )
        
        self.results['test_metrics'] = metrics
        self.results['X_train'] = X_train
        self.results['y_train'] = y_train
        self.results['X_test'] = X_test
        self.results['y_test'] = y_test
        
        return self.model, metrics
    
    def perform_y_randomization(self, n_iterations=10):
        """Perform y-randomization test to check for overfitting."""
        print("\n" + "=" * 70)
        print("STEP 6: Y-RANDOMIZATION TEST")
        print("=" * 70)
        
        rand_results = YRandomizationTester.perform_y_randomization(
            X=self.results['X_train'],
            y=self.results['y_train'],
            model=self.model,
            n_iterations=n_iterations
        )
        
        self.results['y_randomization'] = rand_results
        return rand_results
    
    def print_summary(self):
        """Print comprehensive summary of results."""
        print("\n" + "=" * 70)
        print("FINAL SUMMARY")
        print("=" * 70)
        
        # Dataset info
        print("\n[DATASET]")
        print(f"  Total compounds: {len(self.df)}")
        print(f"  Unique scaffolds: {self.results['scaffold_diversity']['n_scaffolds']}")
        print(f"  Diversity ratio: {self.results['scaffold_diversity']['diversity_ratio']:.3f}")
        print(f"  Activity cliffs: {len(self.results['activity_cliffs'])}")
        
        # Model performance
        print("\n[MODEL PERFORMANCE]")
        test_metrics = self.results['test_metrics']
        baseline_metrics = self.results['baseline_metrics']
        
        print(f"  Test RMSE:     {test_metrics['rmse']:.3f}")
        print(f"  Test R²:       {test_metrics['r2']:.3f}")
        print(f"  Baseline RMSE: {baseline_metrics['rmse']:.3f}")
        print(f"  Baseline R²:   {baseline_metrics['r2']:.3f}")
        
        # Y-randomization
        print("\n[OVERFITTING CHECK]")
        rand = self.results['y_randomization']
        print(f"  Y-random R²: {rand['r2_mean']:.3f} ± {rand['r2_std']:.3f}")
        
        # Experimental error comparison
        exp_error = self.results['experimental_error']['experimental_error']
        print("\n[EXPERIMENTAL ERROR]")
        print(f"  Estimated error: {exp_error:.3f}")
        print(f"  Model RMSE:      {test_metrics['rmse']:.3f}")
        
        # Warnings
        print("\n[WARNINGS]")
        warnings_found = False
        
        if test_metrics['rmse'] < exp_error:
            print("  [WARNING] RMSE below experimental error - possible overfitting!")
            warnings_found = True
        
        if rand['r2_mean'] > 0.2:
            print("  [WARNING] High y-random R² - model is overfitting!")
            warnings_found = True
        
        if self.results['scaffold_diversity']['diversity_ratio'] < 0.3:
            print("  [WARNING] Low scaffold diversity - limited applicability!")
            warnings_found = True
        
        if not warnings_found:
            print("  No major warnings detected.")
        
        print("\n" + "=" * 70)
    
    def save_results(self, output_path='qsar_results.txt'):
        """Save results to file."""
        with open(output_path, 'w') as f:
            f.write("QSAR Validation Results\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Dataset: {self.data_path}\n")
            f.write(f"Compounds: {len(self.df)}\n")
            f.write(f"Features: {self.X.shape[1]}\n\n")
            
            f.write("Test Metrics:\n")
            for key, value in self.results['test_metrics'].items():
                f.write(f"  {key}: {value}\n")
        
        print(f"\nResults saved to: {output_path}")
    
    def run_complete_workflow(self, model_type='random_forest'):
        """Run the complete workflow."""
        self.load_data()
        self.validate_dataset()
        self.generate_features()
        self.check_model_complexity(model_type)
        self.train_model(model_type)
        self.perform_y_randomization()
        self.print_summary()
        
        return self.results


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description='QSAR Validation Framework - Standalone Workflow',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python standalone_qsar_workflow.py --data mydata.csv
  python standalone_qsar_workflow.py --data mydata.csv --smiles SMILES --target pIC50
  python standalone_qsar_workflow.py --data mydata.csv --model ridge --output results.txt
  python standalone_qsar_workflow.py --checklist
        """
    )
    
    parser.add_argument('--data', type=str, help='Path to CSV file with SMILES and activity')
    parser.add_argument('--smiles', type=str, default='SMILES', help='SMILES column name (default: SMILES)')
    parser.add_argument('--target', type=str, default='Activity', help='Target column name (default: Activity)')
    parser.add_argument('--model', type=str, default='random_forest', 
                       choices=['random_forest', 'ridge'],
                       help='Model type (default: random_forest)')
    parser.add_argument('--output', type=str, help='Output file for results')
    parser.add_argument('--checklist', action='store_true', help='Print validation checklist and exit')
    
    args = parser.parse_args()
    
    # Show checklist
    if args.checklist:
        print_comprehensive_validation_checklist()
        return
    
    # Validate arguments
    if not args.data:
        parser.error("--data is required (or use --checklist to show validation checklist)")
    
    # Run workflow
    print("\n" + "=" * 70)
    print("QSAR VALIDATION FRAMEWORK - STANDALONE WORKFLOW")
    print("=" * 70)
    
    workflow = QSARWorkflow(
        data_path=args.data,
        smiles_col=args.smiles,
        target_col=args.target
    )
    
    try:
        results = workflow.run_complete_workflow(model_type=args.model)
        
        if args.output:
            workflow.save_results(args.output)
        
        print("\n[SUCCESS] Workflow completed!")
        
    except Exception as e:
        print(f"\n[ERROR] Workflow failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
