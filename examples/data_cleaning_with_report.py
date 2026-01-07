"""
Enhanced Data Cleaning with Detailed Reporting

This script demonstrates how to:
1. Canonicalize SMILES and identify invalid molecules
2. Remove duplicates with averaging strategy
3. Generate comprehensive cleaning reports (CSV files)
4. Track all changes with detailed statistics

Output files:
- cleaning_report_invalid_smiles.csv: List of molecules that couldn't be canonicalized
- cleaning_report_duplicates.csv: Details of all duplicates found and how they were merged
- cleaning_report_summary.csv: High-level statistics
- cleaned_dataset.csv: Final clean dataset ready for modeling

Usage:
    # Quick version (no detailed reports)
    from examples.data_cleaning_with_report import quick_clean
    clean_df = quick_clean(df, smiles_col='Canonical SMILES', target_col='PIC50')
    
    # Detailed version (with reports)
    from examples.data_cleaning_with_report import clean_qsar_data_with_report
    clean_df, stats = clean_qsar_data_with_report(df, smiles_col='Canonical SMILES', target_col='PIC50')
"""

import pandas as pd
from utils.qsar_utils_no_leakage import QSARDataProcessor
from qsar_validation.splitting_strategies import AdvancedSplitter


def quick_clean(df, smiles_col='Canonical SMILES', target_col='PIC50'):
    """
    Quick data cleaning without detailed reports.
    
    This is the simple version - just cleans the data and prints basic info.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with SMILES and activity data
    smiles_col : str
        Column name containing SMILES strings
    target_col : str
        Column name containing activity values
    
    Returns
    -------
    pd.DataFrame
        Cleaned dataframe
    
    Example
    -------
    >>> import pandas as pd
    >>> from examples.data_cleaning_with_report import quick_clean
    >>> 
    >>> df = pd.read_csv('your_data.csv')
    >>> clean_df = quick_clean(df, smiles_col='Canonical SMILES', target_col='PIC50')
    >>> print(f"Clean dataset: {len(clean_df)} molecules")
    """
    # Initialize processor
    processor = QSARDataProcessor(smiles_col=smiles_col, target_col=target_col)
    
    # Store original count
    original_count = len(df)
    
    # Step 1: Canonicalize SMILES
    df_clean = processor.canonicalize_smiles(df)
    canonical_count = len(df_clean)
    invalid_removed = original_count - canonical_count
    
    # Step 2: Remove duplicates
    df_clean = processor.remove_duplicates(df_clean, strategy='average')
    final_count = len(df_clean)
    duplicates_merged = canonical_count - final_count
    
    # Print summary
    print(f"Original dataset: {original_count} molecules")
    print(f"Invalid SMILES removed: {invalid_removed} molecules")
    print(f"Duplicates merged: {duplicates_merged} molecules")
    print(f"Clean dataset: {final_count} molecules")
    
    return df_clean


def clean_qsar_data_with_report(df, smiles_col='Canonical SMILES', target_col='PIC50'):
    """
    Clean QSAR dataset and generate comprehensive reports.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with SMILES and activity data
    smiles_col : str
        Column name containing SMILES strings
    target_col : str
        Column name containing activity values
    
    Returns
    -------
    pd.DataFrame
        Cleaned dataframe
    dict
        Dictionary with cleaning statistics
    """
    
    # Initialize processor
    processor = QSARDataProcessor(smiles_col=smiles_col, target_col=target_col)
    
    # Store original dataset info
    original_count = len(df)
    original_smiles = df[smiles_col].tolist()
    
    print("="*80)
    print("DATA CLEANING REPORT")
    print("="*80)
    
    # Step 1: Canonicalize SMILES
    print(f"\n📊 Original dataset: {original_count} molecules")
    
    df_clean = processor.canonicalize_smiles(df)
    canonical_count = len(df_clean)
    invalid_count = original_count - canonical_count
    
    print(f"\n🧹 Step 1: SMILES Canonicalization")
    print(f"   ✓ Valid SMILES: {canonical_count} molecules")
    print(f"   ✗ Invalid SMILES removed: {invalid_count} molecules")
    
    # Identify which SMILES failed canonicalization
    if invalid_count > 0:
        valid_smiles = set(df_clean[smiles_col].tolist())
        invalid_smiles = [smi for smi in original_smiles if smi not in valid_smiles]
        
        print(f"\n❌ INVALID SMILES (could not be canonicalized):")
        print("-" * 80)
        for i, smi in enumerate(invalid_smiles[:20], 1):  # Show first 20
            print(f"   {i}. {smi}")
        if len(invalid_smiles) > 20:
            print(f"   ... and {len(invalid_smiles) - 20} more")
        
        # Save to file
        pd.DataFrame({'Invalid_SMILES': invalid_smiles}).to_csv(
            'cleaning_report_invalid_smiles.csv', index=False
        )
        print(f"\n💾 Saved full list to: cleaning_report_invalid_smiles.csv")
    
    # Step 2: Remove duplicates
    print(f"\n🔁 Step 2: Duplicate Removal (strategy='average')")
    
    # Store pre-duplicate info
    before_dedup = df_clean.copy()
    before_dedup_count = len(before_dedup)
    
    df_clean = processor.remove_duplicates(df_clean, strategy='average')
    after_dedup_count = len(df_clean)
    duplicate_count = before_dedup_count - after_dedup_count
    
    print(f"   ✓ Unique molecules: {after_dedup_count}")
    print(f"   🔁 Duplicates merged: {duplicate_count} molecules")
    
    # Identify duplicates and show details
    if duplicate_count > 0:
        # Find which SMILES were duplicated
        smiles_counts = before_dedup[smiles_col].value_counts()
        duplicated_smiles = smiles_counts[smiles_counts > 1]
        
        print(f"\n🔍 DUPLICATE DETAILS:")
        print("-" * 80)
        
        duplicate_report = []
        
        for smi in duplicated_smiles.index[:20]:  # Show first 20
            dup_rows = before_dedup[before_dedup[smiles_col] == smi]
            activities = dup_rows[target_col].tolist()
            averaged = df_clean[df_clean[smiles_col] == smi][target_col].values[0]
            
            print(f"\n   SMILES: {smi}")
            print(f"   Occurrences: {len(dup_rows)}")
            print(f"   Original {target_col} values: {activities}")
            print(f"   Averaged {target_col}: {averaged:.4f}")
            
            duplicate_report.append({
                'Canonical_SMILES': smi,
                'Occurrences': len(dup_rows),
                f'Original_{target_col}_Values': str(activities),
                f'Averaged_{target_col}': averaged,
                'Std_Dev': dup_rows[target_col].std()
            })
        
        if len(duplicated_smiles) > 20:
            print(f"\n   ... and {len(duplicated_smiles) - 20} more duplicated SMILES")
        
        # Save duplicate report
        pd.DataFrame(duplicate_report).to_csv(
            'cleaning_report_duplicates.csv', index=False
        )
        print(f"\n💾 Saved duplicate report to: cleaning_report_duplicates.csv")
    
    # Final summary
    print("\n" + "="*80)
    print("CLEANING SUMMARY")
    print("="*80)
    print(f"Original dataset:        {original_count} molecules")
    print(f"Invalid SMILES removed:  {invalid_count} molecules ({invalid_count/original_count*100:.1f}%)")
    print(f"Duplicates merged:       {duplicate_count} molecules ({duplicate_count/original_count*100:.1f}%)")
    print(f"Clean dataset:           {after_dedup_count} molecules ({after_dedup_count/original_count*100:.1f}%)")
    print(f"Total removed:           {original_count - after_dedup_count} molecules")
    print("="*80)
    
    # Save clean dataset
    df_clean.to_csv('cleaned_dataset.csv', index=False)
    print(f"\n💾 Saved clean dataset to: cleaned_dataset.csv")
    
    # Create comprehensive report
    report = {
        'Metric': [
            'Original molecules',
            'Invalid SMILES',
            'Valid molecules',
            'Duplicates found',
            'Final clean dataset',
            'Total removed',
            '% Data retained'
        ],
        'Count': [
            original_count,
            invalid_count,
            canonical_count,
            duplicate_count,
            after_dedup_count,
            original_count - after_dedup_count,
            f"{after_dedup_count/original_count*100:.1f}%"
        ]
    }
    
    pd.DataFrame(report).to_csv('cleaning_report_summary.csv', index=False)
    print(f"💾 Saved summary report to: cleaning_report_summary.csv")
    
    print("\n✅ Data cleaning complete!")
    
    # Return statistics
    stats = {
        'original_count': original_count,
        'invalid_count': invalid_count,
        'duplicate_count': duplicate_count,
        'final_count': after_dedup_count
    }
    
    return df_clean, stats


# Example usage
if __name__ == "__main__":
    print("="*80)
    print("DATA CLEANING SCRIPT - TWO METHODS AVAILABLE")
    print("="*80)
    
    print("\n📋 METHOD 1: Quick Clean (Simple, no reports)")
    print("-" * 80)
    print("import pandas as pd")
    print("from examples.data_cleaning_with_report import quick_clean")
    print("")
    print("df = pd.read_csv('your_data.csv')")
    print("clean_df = quick_clean(df, smiles_col='Canonical SMILES', target_col='PIC50')")
    print("print(f'Clean dataset: {len(clean_df)} molecules')")
    
    print("\n" + "="*80)
    print("\n📊 METHOD 2: Detailed Clean (With comprehensive reports)")
    print("-" * 80)
    print("import pandas as pd")
    print("from examples.data_cleaning_with_report import clean_qsar_data_with_report")
    print("")
    print("df = pd.read_csv('your_data.csv')")
    print("clean_df, stats = clean_qsar_data_with_report(")
    print("    df, ")
    print("    smiles_col='Canonical SMILES', ")
    print("    target_col='PIC50'")
    print(")")
    print("")
    print("# Generates 4 CSV files:")
    print("# - cleaning_report_invalid_smiles.csv")
    print("# - cleaning_report_duplicates.csv")
    print("# - cleaning_report_summary.csv")
    print("# - cleaned_dataset.csv")
    
    print("\n" + "="*80)
    print("For a working example with real data, see: comprehensive_test/")
    print("="*80)
