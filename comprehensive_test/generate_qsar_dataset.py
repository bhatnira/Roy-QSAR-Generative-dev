"""
Generate Realistic QSAR Dataset for Testing
============================================

Creates a synthetic but realistic QSAR dataset with:
- Valid SMILES strings
- Activity values (pIC50)
- Date information
- Duplicate molecules
- Chemical diversity
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Real drug-like molecules (simplified SMILES)
BASE_MOLECULES = [
    # Benzodiazepines (scaffold 1)
    "c1ccc2c(c1)C(=O)N(C)c1ccccc1N2",
    "c1ccc2c(c1)C(=O)NCc1ccccc1N2",
    "c1ccc2c(c1)C(=O)N(CC)c1ccccc1N2",
    "Clc1ccc2c(c1)C(=O)Nc1ccccc1N2",
    "Fc1ccc2c(c1)C(=O)Nc1ccccc1N2",
    
    # Phenethylamines (scaffold 2)
    "c1ccc(cc1)CCN",
    "c1ccc(cc1)CC(C)N",
    "COc1ccc(cc1)CCN",
    "c1ccc(c(c1)OC)CCN",
    "Cc1ccc(cc1)CCN",
    
    # Sulfonamides (scaffold 3)
    "c1ccc(cc1)S(=O)(=O)N",
    "Cc1ccc(cc1)S(=O)(=O)N",
    "c1ccc(cc1)S(=O)(=O)Nc2ccccc2",
    "Nc1ccc(cc1)S(=O)(=O)N",
    
    # Quinolines (scaffold 4)
    "c1ccc2c(c1)cccn2",
    "Cc1cc2ccccc2nc1",
    "c1ccc2c(c1)nccc2O",
    "Clc1ccc2c(c1)cccn2",
    
    # Indoles (scaffold 5)
    "c1ccc2c(c1)[nH]cc2",
    "Cc1c[nH]c2ccccc12",
    "c1ccc2c(c1)cc[nH]2",
    "Nc1ccc2c(c1)[nH]cc2",
    
    # Pyridines (scaffold 6)
    "c1ccncc1",
    "Cc1ccncc1",
    "c1cc(C)ncc1",
    "c1cc(N)ncc1",
    "Clc1ccncc1",
    
    # Piperazines (scaffold 7)
    "C1CNCCN1",
    "c1ccc(cc1)N2CCNCC2",
    "CC1CNCCN1",
    "c1ccc(cc1)C2CNCCN2",
    
    # Morpholines (scaffold 8)
    "C1COCCN1",
    "c1ccc(cc1)N2CCOCC2",
    "CC1COCCN1",
    
    # Thiophenes (scaffold 9)
    "c1ccsc1",
    "Cc1ccsc1",
    "c1cc(C)sc1",
    "Clc1ccsc1",
    
    # Pyrimidines (scaffold 10)
    "c1cncnc1",
    "Cc1cncnc1",
    "Nc1cncnc1",
    "c1c(N)ncnc1",
]

def generate_variants(base_smiles, n_variants=3):
    """Generate variants of a molecule."""
    variants = [base_smiles]
    
    # Add some simple modifications
    modifications = [
        lambda s: s.replace("C", "C(C)", 1) if "C" in s else s,
        lambda s: s.replace("c1", "c1c", 1) if "c1" in s else s,
        lambda s: s + "C" if len(s) < 30 else s,
    ]
    
    for i, mod in enumerate(modifications[:n_variants-1]):
        try:
            variants.append(mod(base_smiles))
        except:
            variants.append(base_smiles)
    
    return variants[:n_variants]

def generate_activity(smiles, base_activity=6.5, noise=0.5):
    """
    Generate realistic activity values.
    Activity correlates weakly with molecular properties.
    """
    # Simple heuristic: longer molecules tend to have different activity
    length_effect = (len(smiles) - 25) * 0.05
    
    # Presence of certain groups affects activity
    halogen_effect = 0.3 if any(x in smiles for x in ['Cl', 'Br', 'F']) else 0
    nitrogen_effect = 0.2 * smiles.count('N')
    oxygen_effect = 0.15 * smiles.count('O')
    
    # Base activity with effects and noise
    activity = base_activity + length_effect + halogen_effect + nitrogen_effect + oxygen_effect
    activity += np.random.normal(0, noise)
    
    # pIC50 typically ranges from 4 to 9
    activity = np.clip(activity, 4.0, 9.0)
    
    return round(activity, 2)

def generate_qsar_dataset(n_molecules=150, duplicate_rate=0.05):
    """
    Generate a complete QSAR dataset.
    
    Parameters
    ----------
    n_molecules : int
        Target number of molecules
    duplicate_rate : float
        Fraction of molecules to duplicate (with slight variation)
    
    Returns
    -------
    df : pd.DataFrame
        QSAR dataset with SMILES, pIC50, Date, etc.
    """
    
    molecules = []
    
    # Generate molecules by creating variants of base molecules
    while len(molecules) < n_molecules:
        base = np.random.choice(BASE_MOLECULES)
        variants = generate_variants(base, n_variants=2)
        molecules.extend(variants)
    
    # Trim to exact size
    molecules = molecules[:n_molecules]
    
    # Generate activities
    activities = [generate_activity(smiles) for smiles in molecules]
    
    # Generate dates (spread over 2 years)
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=np.random.randint(0, 730)) for _ in range(len(molecules))]
    
    # Create DataFrame
    df = pd.DataFrame({
        'SMILES': molecules,
        'pIC50': activities,
        'Date': dates,
        'Compound_ID': [f'CMPD_{i:04d}' for i in range(len(molecules))]
    })
    
    # Add some duplicates (intentional)
    n_duplicates = int(len(df) * duplicate_rate)
    if n_duplicates > 0:
        duplicate_indices = np.random.choice(df.index, n_duplicates, replace=False)
        duplicates = df.loc[duplicate_indices].copy()
        # Slightly modify activity for duplicates (experimental noise)
        duplicates['pIC50'] = duplicates['pIC50'] + np.random.normal(0, 0.1, len(duplicates))
        duplicates['Date'] = duplicates['Date'] + pd.Timedelta(days=30)
        duplicates['Compound_ID'] = [f'CMPD_DUP_{i:04d}' for i in range(len(duplicates))]
        
        df = pd.concat([df, duplicates], ignore_index=True)
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Add some metadata
    df['Molecular_Weight'] = [len(smiles) * 5 + np.random.randint(-10, 10) for smiles in df['SMILES']]
    df['LogP'] = [len(smiles) * 0.1 + np.random.normal(0, 0.5) for smiles in df['SMILES']]
    
    return df

if __name__ == '__main__':
    print("Generating realistic QSAR dataset...")
    print("=" * 80)
    
    # Generate dataset
    df = generate_qsar_dataset(n_molecules=150, duplicate_rate=0.05)
    
    # Save to CSV
    output_file = 'qsar_test_dataset.csv'
    df.to_csv(output_file, index=False)
    
    print(f"\n✓ Generated {len(df)} molecules")
    print(f"✓ Saved to: {output_file}")
    
    # Show statistics
    print("\nDataset Statistics:")
    print("-" * 80)
    print(f"  Total molecules: {len(df)}")
    print(f"  Unique SMILES: {df['SMILES'].nunique()}")
    print(f"  Duplicates: {len(df) - df['SMILES'].nunique()}")
    print(f"  Activity range: {df['pIC50'].min():.2f} - {df['pIC50'].max():.2f}")
    print(f"  Activity mean: {df['pIC50'].mean():.2f} ± {df['pIC50'].std():.2f}")
    print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    # Show first few rows
    print("\nFirst 10 molecules:")
    print("-" * 80)
    print(df[['Compound_ID', 'SMILES', 'pIC50', 'Date']].head(10).to_string(index=False))
    
    print("\n" + "=" * 80)
    print("✓ Dataset generation complete!")
