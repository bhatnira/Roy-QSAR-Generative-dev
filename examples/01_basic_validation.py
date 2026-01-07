"""
Example: Basic QSAR Validation
================================

This example demonstrates how to use the validation framework
to check your QSAR dataset before model building.
"""

import pandas as pd
from src.qsar_validation import run_comprehensive_validation

def main():
    """Run comprehensive validation on example dataset."""
    
    # Load your dataset
    # df = pd.read_csv('your_data.csv')
    
    # For this example, create synthetic data
    print("Loading dataset...")
    # Replace with actual data loading
    
    # Run comprehensive validation
    print("\nRunning comprehensive validation...")
    results = run_comprehensive_validation(
        df,
        smiles_col='Canonical SMILES',
        target_col='IC50 uM'
    )
    
    # Results contain:
    # - scaffold_diversity: Diversity metrics
    # - activity_distribution: Distribution analysis
    # - activity_cliffs: Detected cliffs
    # - experimental_error: Error estimates
    
    print("\nValidation complete! Review the output above.")
    print("\nKey results:")
    print(f"  Number of unique scaffolds: {results['scaffold_diversity']['n_scaffolds']}")
    print(f"  Diversity ratio: {results['scaffold_diversity']['diversity_ratio']:.3f}")
    print(f"  Activity cliffs detected: {len(results['activity_cliffs'])}")
    
    return results


if __name__ == "__main__":
    # Uncomment when you have actual data
    # results = main()
    print("Example script - modify with your data paths")
