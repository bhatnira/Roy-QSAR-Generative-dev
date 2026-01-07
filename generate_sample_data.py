"""
Generate diverse sample QSAR data for demonstration
"""
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

# Set seed for reproducibility
np.random.seed(42)

# Create diverse SMILES with different scaffolds
smiles_list = [
    # Aromatic compounds (benzene derivatives)
    'c1ccccc1', 'c1ccc(C)cc1', 'c1ccc(O)cc1', 'c1ccc(N)cc1', 'c1ccc(Cl)cc1',
    'c1ccc(F)cc1', 'c1ccc(Br)cc1', 'c1ccc(I)cc1', 'c1ccc(C(=O)O)cc1', 'c1ccc(C(=O)N)cc1',
    'c1ccc(C(F)(F)F)cc1', 'c1ccc(C#N)cc1', 'c1ccc(OC)cc1', 'c1ccc(N(C)C)cc1', 'c1ccc(S)cc1',
    
    # Heterocycles (pyridine, pyrimidine)
    'c1cccnc1', 'c1ccncc1', 'c1ncccn1', 'c1ncncc1', 'c1cnccn1',
    
    # Naphthalene derivatives
    'c1ccc2ccccc2c1', 'c1ccc2c(c1)ccc1ccccc12', 'c1ccc2cc3ccccc3cc2c1',
    
    # Aliphatic compounds (alkanes, alcohols)
    'CCCC', 'CCCCC', 'CCCCCC', 'CCCCCCC', 'CCCCCCCC',
    'CCCCO', 'CCCCCO', 'CCCCCCO', 'CCCCCCCO', 'CCCCCCCCO',
    'CC(C)C', 'CC(C)CC', 'CC(C)CCC', 'CC(C)CCCC', 'CC(C)C(C)C',
    
    # Carboxylic acids
    'CC(=O)O', 'CCC(=O)O', 'CCCC(=O)O', 'CC(C)C(=O)O', 'CC(C)(C)C(=O)O',
    
    # Ethers
    'CCOC', 'CCCOC', 'CC(C)OC', 'CCOCCOC', 'c1ccc(OCC)cc1',
    
    # Amines
    'CCN', 'CCCN', 'CC(C)N', 'c1ccc(NC)cc1', 'c1ccc(N(CC)CC)cc1',
    
    # Substituted benzenes (multi-substituted)
    'c1cc(C)c(O)cc1', 'c1cc(Cl)c(Cl)cc1', 'c1cc(F)c(F)c(F)c1',
    
    # Biphenyl derivatives
    'c1ccc(c2ccccc2)cc1', 'c1ccc(c2ccc(C)cc2)cc1', 'c1ccc(c2ccc(O)cc2)cc1',
    
    # Cycloalkanes
    'C1CCCC1', 'C1CCCCC1', 'C1CCCCCC1',
    
    # Ketones
    'CC(=O)C', 'CCC(=O)C', 'CC(=O)CC', 'c1ccc(C(=O)C)cc1',
    
    # Esters
    'CC(=O)OC', 'CCC(=O)OC', 'c1ccc(C(=O)OC)cc1',
    
    # Amides
    'CC(=O)N', 'CCC(=O)N', 'CC(=O)NC', 'c1ccc(C(=O)NC)cc1',
    
    # Sulfonyl compounds
    'c1ccc(S(=O)(=O)N)cc1', 'c1ccc(S(=O)(=O)C)cc1',
    
    # Phosphates
    'c1ccc(P(=O)(O)O)cc1',
    
    # Additional diverse structures
    'c1ccc(C(C)C)cc1', 'c1ccc(C(C)(C)C)cc1', 'c1ccc(CC)cc1', 'c1ccc(CCC)cc1',
    'c1ccc(CCCC)cc1', 'c1ccc(C(=O)NCC)cc1', 'c1ccc(C(=O)NCCC)cc1',
]

# Generate activities with structure-based patterns
activities = []
for smiles in smiles_list:
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        # Base activity
        activity = 5.0
        
        # Add contribution from molecular weight
        mw = Descriptors.MolWt(mol)
        activity += (mw - 100) / 50
        
        # Add contribution from aromatic rings
        n_aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
        activity += n_aromatic_rings * 0.8
        
        # Add contribution from halogens
        n_halogens = sum([len(mol.GetSubstructMatches(Chem.MolFromSmarts(f'[{x}]'))) 
                         for x in ['F', 'Cl', 'Br', 'I']])
        activity += n_halogens * 0.5
        
        # Add some noise
        activity += np.random.normal(0, 0.3)
        
        activities.append(round(activity, 1))
    else:
        activities.append(5.0)

# Create DataFrame
df = pd.DataFrame({
    'SMILES': smiles_list,
    'Activity': activities
})

# Save to CSV
df.to_csv('sample_data.csv', index=False)

print(f"Generated {len(df)} compounds")
print(f"Unique SMILES: {df['SMILES'].nunique()}")
print(f"Activity range: [{df['Activity'].min():.2f}, {df['Activity'].max():.2f}]")
print("Saved to: sample_data.csv")
