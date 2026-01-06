"""
Splitting Strategies Examples
==============================

Demonstrates the three splitting strategies:
1. Scaffold-based (recommended for most cases)
2. Time-based (when temporal data available)
3. Leave-cluster-out (for small datasets)

Each example shows:
- When to use the strategy
- How to implement it
- How to validate the split
- Pros and cons
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Import all three splitting strategies
from qsar_validation.splitting_strategies import (
    AdvancedSplitter,
    ScaffoldSplitter,
    TemporalSplitter,
    ClusterSplitter
)


# ============================================================================
# Example 1: Scaffold-Based Splitting (RECOMMENDED)
# ============================================================================

def example_1_scaffold_splitting():
    """
    Scaffold-based splitting using Bemis-Murcko scaffolds.
    
    WHEN TO USE:
    - Most QSAR modeling tasks (RECOMMENDED)
    - When you want to test generalization to new scaffolds
    - When you have compounds with diverse scaffolds
    
    PROS:
    - Realistic test of generalization
    - Prevents scaffold leakage
    - Industry standard
    
    CONS:
    - May result in uneven split sizes
    - Requires RDKit
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: Scaffold-Based Splitting (RECOMMENDED)")
    print("="*80)
    
    # Sample data with different scaffolds
    df = pd.DataFrame({
        'SMILES': [
            'c1ccccc1CC',      # Benzene scaffold
            'c1ccccc1CCC',     # Benzene scaffold
            'c1ccccc1CCCC',    # Benzene scaffold
            'c1ccncc1CC',      # Pyridine scaffold
            'c1ccncc1CCC',     # Pyridine scaffold
            'CCCCCC',          # Aliphatic
            'CCCCCCC',         # Aliphatic
            'c1ccc2ccccc2c1C', # Naphthalene scaffold
        ],
        'Activity': [5.0, 5.5, 6.0, 7.0, 7.5, 3.0, 3.5, 8.0]
    })
    
    print(f"\nDataset: {len(df)} compounds")
    
    # Method 1: Using AdvancedSplitter with strategy='scaffold'
    print("\n--- Method 1: AdvancedSplitter ---")
    splitter = AdvancedSplitter(smiles_col='SMILES', strategy='scaffold')
    train_idx, val_idx, test_idx = splitter.split(df, test_size=0.25, val_size=0.25)
    
    # Method 2: Using convenience class ScaffoldSplitter
    print("\n--- Method 2: ScaffoldSplitter (convenience) ---")
    splitter = ScaffoldSplitter(smiles_col='SMILES')
    train_idx, val_idx, test_idx = splitter.split(df, test_size=0.25, val_size=0.25)
    
    # Verify no scaffold overlap
    overlap = splitter.check_scaffold_overlap(train_idx, test_idx, df)
    print(f"\n✓ Validation: Scaffold overlap = {overlap} (should be 0)")
    
    # Show which scaffolds are in train vs test
    print("\n--- Scaffold Distribution ---")
    train_smiles = df.iloc[train_idx]['SMILES']
    test_smiles = df.iloc[test_idx]['SMILES']
    
    print("Train scaffolds:")
    for smiles in train_smiles:
        scaffold = splitter.get_scaffold(smiles)
        print(f"  {smiles:20s} → {scaffold}")
    
    print("\nTest scaffolds:")
    for smiles in test_smiles:
        scaffold = splitter.get_scaffold(smiles)
        print(f"  {smiles:20s} → {scaffold}")
    
    print("\n✅ Use scaffold splitting for most QSAR tasks!")


# ============================================================================
# Example 2: Time-Based Splitting
# ============================================================================

def example_2_temporal_splitting():
    """
    Time-based splitting for compounds with dates.
    
    WHEN TO USE:
    - When you have date/time information
    - When modeling drug discovery pipeline
    - When testing prospective prediction
    
    PROS:
    - Simulates realistic deployment
    - Tests temporal generalization
    - Natural for time-series data
    
    CONS:
    - Requires date information
    - May have temporal bias
    - Older compounds may differ from newer ones
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Time-Based Splitting")
    print("="*80)
    
    # Sample data with dates (e.g., synthesis dates or assay dates)
    base_date = datetime(2020, 1, 1)
    dates = [base_date + timedelta(days=i*30) for i in range(10)]
    
    df = pd.DataFrame({
        'SMILES': [
            'CCO', 'CCC', 'CCCC', 'CCCCC', 'CCCCCC',
            'c1ccccc1', 'c1ccccc1C', 'c1ccccc1CC', 'c1ccccc1CCC', 'c1ccncc1'
        ],
        'Activity': [4.0, 4.2, 4.5, 4.8, 5.0, 6.0, 6.2, 6.5, 6.8, 7.0],
        'Date': dates
    })
    
    print(f"\nDataset: {len(df)} compounds")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    # Method 1: Using AdvancedSplitter with strategy='temporal'
    print("\n--- Method 1: AdvancedSplitter ---")
    splitter = AdvancedSplitter(
        smiles_col='SMILES',
        strategy='temporal',
        date_col='Date'
    )
    train_idx, val_idx, test_idx = splitter.split(df, test_size=0.2, val_size=0.2)
    
    # Method 2: Using convenience class TemporalSplitter
    print("\n--- Method 2: TemporalSplitter (convenience) ---")
    splitter = TemporalSplitter(smiles_col='SMILES', date_col='Date')
    train_idx, val_idx, test_idx = splitter.split(df, test_size=0.2, val_size=0.2)
    
    # Visualize temporal split
    print("\n--- Temporal Distribution ---")
    print(f"Train (oldest): {len(train_idx)} compounds")
    for idx in train_idx[:3]:
        print(f"  {df.iloc[idx]['Date']} - {df.iloc[idx]['SMILES']}")
    
    print(f"\nTest (newest): {len(test_idx)} compounds")
    for idx in test_idx[:3]:
        print(f"  {df.iloc[idx]['Date']} - {df.iloc[idx]['SMILES']}")
    
    print("\n✅ Use temporal splitting when you have date information!")


# ============================================================================
# Example 3: Leave-Cluster-Out Splitting
# ============================================================================

def example_3_cluster_splitting():
    """
    Leave-cluster-out splitting using fingerprint similarity.
    
    WHEN TO USE:
    - Small datasets (< 100 compounds)
    - When testing generalization to dissimilar compounds
    - When you don't have clear scaffolds
    
    PROS:
    - Works with small datasets
    - Tests generalization to dissimilar compounds
    - Doesn't require scaffold identification
    
    CONS:
    - Computationally expensive (fingerprint generation + clustering)
    - Cluster assignment can be arbitrary
    - May need to tune n_clusters
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Leave-Cluster-Out Splitting")
    print("="*80)
    
    # Sample data (diverse compounds)
    df = pd.DataFrame({
        'SMILES': [
            # Aliphatics
            'CCCC', 'CCCCC', 'CCCCCC',
            # Aromatics
            'c1ccccc1', 'c1ccccc1C', 'c1ccccc1CC',
            # Heterocycles
            'c1ccncc1', 'c1ccncc1C', 'c1cnccn1',
            # Mixed
            'CCOc1ccccc1', 'CCCc1ccccc1', 'c1ccc(O)cc1'
        ],
        'Activity': [3.0, 3.2, 3.5, 6.0, 6.2, 6.5, 7.0, 7.2, 7.5, 5.0, 5.5, 6.8]
    })
    
    print(f"\nDataset: {len(df)} compounds")
    
    # Method 1: Using AdvancedSplitter with strategy='cluster'
    print("\n--- Method 1: AdvancedSplitter ---")
    splitter = AdvancedSplitter(
        smiles_col='SMILES',
        strategy='cluster',
        n_clusters=3
    )
    train_idx, val_idx, test_idx = splitter.split(df, test_size=0.25, val_size=0.25)
    
    # Method 2: Using convenience class ClusterSplitter
    print("\n--- Method 2: ClusterSplitter (convenience) ---")
    splitter = ClusterSplitter(smiles_col='SMILES', n_clusters=3)
    train_idx, val_idx, test_idx = splitter.split(df, test_size=0.25, val_size=0.25)
    
    print("\n✅ Use cluster splitting for small, diverse datasets!")


# ============================================================================
# Example 4: Comparing All Three Strategies
# ============================================================================

def example_4_compare_strategies():
    """
    Compare all three splitting strategies on the same dataset.
    
    Shows how different strategies lead to different splits.
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Comparing All Three Strategies")
    print("="*80)
    
    # Generate sample data
    np.random.seed(42)
    base_date = datetime(2020, 1, 1)
    
    df = pd.DataFrame({
        'SMILES': [
            'c1ccccc1', 'c1ccccc1C', 'c1ccccc1CC',  # Benzene
            'c1ccncc1', 'c1ccncc1C',                # Pyridine
            'CCCC', 'CCCCC', 'CCCCCC',              # Aliphatic
            'c1cnccn1', 'c1cncnc1',                 # Other heterocycles
        ],
        'Activity': np.random.rand(10) * 10,
        'Date': [base_date + timedelta(days=i*30) for i in range(10)]
    })
    
    print(f"\nDataset: {len(df)} compounds\n")
    
    # Strategy 1: Scaffold
    print("="*60)
    print("STRATEGY 1: Scaffold-Based")
    print("="*60)
    splitter1 = ScaffoldSplitter()
    train1, val1, test1 = splitter1.split(df, test_size=0.2, val_size=0.2)
    
    # Strategy 2: Temporal
    print("\n" + "="*60)
    print("STRATEGY 2: Time-Based")
    print("="*60)
    splitter2 = TemporalSplitter(date_col='Date')
    train2, val2, test2 = splitter2.split(df, test_size=0.2, val_size=0.2)
    
    # Strategy 3: Cluster
    print("\n" + "="*60)
    print("STRATEGY 3: Leave-Cluster-Out")
    print("="*60)
    splitter3 = ClusterSplitter(n_clusters=3)
    train3, val3, test3 = splitter3.split(df, test_size=0.2, val_size=0.2)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY: Split Size Comparison")
    print("="*60)
    print(f"{'Strategy':<20} {'Train':<10} {'Val':<10} {'Test':<10}")
    print("-"*60)
    print(f"{'Scaffold':<20} {len(train1):<10} {len(val1):<10} {len(test1):<10}")
    print(f"{'Temporal':<20} {len(train2):<10} {len(val2):<10} {len(test2):<10}")
    print(f"{'Cluster':<20} {len(train3):<10} {len(val3):<10} {len(test3):<10}")


# ============================================================================
# Example 5: Choosing the Right Strategy
# ============================================================================

def example_5_choosing_strategy():
    """
    Decision tree for choosing the right splitting strategy.
    """
    print("\n" + "="*80)
    print("EXAMPLE 5: How to Choose the Right Strategy")
    print("="*80)
    
    decision_tree = """
    
    📊 DECISION TREE FOR SPLITTING STRATEGY
    ═══════════════════════════════════════
    
    START: What type of data do you have?
    
    ┌─────────────────────────────────────────────────────────────┐
    │ Do you have date/time information for compounds?            │
    └─────────────────────────────────────────────────────────────┘
                    │
                    ├─ YES → ⏰ Use TEMPORAL SPLITTING
                    │         • Simulates realistic deployment
                    │         • Train on older, test on newer
                    │         • Best for time-series data
                    │
                    └─ NO → Continue below
    
    ┌─────────────────────────────────────────────────────────────┐
    │ Do you have > 100 compounds with diverse scaffolds?         │
    └─────────────────────────────────────────────────────────────┘
                    │
                    ├─ YES → 🔬 Use SCAFFOLD SPLITTING (RECOMMENDED)
                    │         • Industry standard
                    │         • Tests generalization to new scaffolds
                    │         • Prevents data leakage
                    │         • Best for most QSAR tasks
                    │
                    └─ NO → Continue below
    
    ┌─────────────────────────────────────────────────────────────┐
    │ Do you have < 100 compounds or few distinct scaffolds?      │
    └─────────────────────────────────────────────────────────────┘
                    │
                    └─ YES → 🔗 Use CLUSTER SPLITTING
                              • Good for small datasets
                              • Tests generalization to dissimilar compounds
                              • Works when scaffolds are limited
    
    
    📋 SUMMARY TABLE
    ═══════════════════════════════════════════════════════════════
    
    Strategy          When to Use                  Pros                 Cons
    ────────────────────────────────────────────────────────────────────────
    Scaffold          Most QSAR tasks              Realistic,           May have uneven
                      > 100 compounds              Industry standard    split sizes
                      Diverse scaffolds            Prevents leakage
    
    Temporal          Have date info               Simulates            Requires dates
                      Time-series data             deployment           May have bias
                      Drug discovery pipeline      Prospective test
    
    Cluster           Small datasets               Works with           Computationally
                      < 100 compounds              small data           expensive
                      Few scaffolds                Tests dissimilarity  Arbitrary clusters
    
    
    💡 RECOMMENDATIONS
    ═══════════════════════════════════════════════════════════════
    
    1. DEFAULT: Use scaffold splitting for most cases
    
    2. TEMPORAL DATA: Use temporal splitting if you have dates
    
    3. SMALL DATASETS: Use cluster splitting for < 100 compounds
    
    4. VALIDATION: Always check for data leakage after splitting!
    
    5. COMBINATION: You can use multiple strategies for comparison
    
    """
    
    print(decision_tree)


# ============================================================================
# MAIN: Run All Examples
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("SPLITTING STRATEGIES EXAMPLES")
    print("Demonstrating three splitting approaches")
    print("="*80)
    
    examples = [
        ("1", "Scaffold-Based Splitting (RECOMMENDED)", example_1_scaffold_splitting),
        ("2", "Time-Based Splitting", example_2_temporal_splitting),
        ("3", "Leave-Cluster-Out Splitting", example_3_cluster_splitting),
        ("4", "Compare All Three Strategies", example_4_compare_strategies),
        ("5", "How to Choose the Right Strategy", example_5_choosing_strategy),
    ]
    
    print("\nAvailable examples:")
    for num, title, _ in examples:
        print(f"  {num}. {title}")
    
    print("\n" + "="*80)
    choice = input("\nEnter example number to run (or 'all' for all examples): ").strip()
    
    if choice.lower() == 'all':
        for num, title, func in examples:
            func()
    elif choice in [ex[0] for ex in examples]:
        for num, title, func in examples:
            if num == choice:
                func()
                break
    else:
        print("Invalid choice. Running all examples...")
        for num, title, func in examples:
            func()
    
    print("\n" + "="*80)
    print("EXAMPLES COMPLETE")
    print("="*80)
    print("\n💡 Key Takeaways:")
    print("   • Scaffold splitting: RECOMMENDED for most QSAR tasks")
    print("   • Temporal splitting: Use when you have date information")
    print("   • Cluster splitting: Use for small datasets (< 100 compounds)")
    print("   • Always validate your split to prevent data leakage!")
    print("="*80 + "\n")
