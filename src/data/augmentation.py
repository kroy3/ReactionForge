"""
Data Augmentation for Chemical Reactions
=========================================
SMILES enumeration and perturbation strategies.
"""

from rdkit import Chem
import random


def enumerate_smiles(smiles, n=5):
    """
    Generate different SMILES representations of the same molecule.
    
    Args:
        smiles: Input SMILES string
        n: Number of variations to generate
        
    Returns:
        list: Different SMILES strings representing the same molecule
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [smiles]
    
    smiles_list = []
    for _ in range(n):
        smiles_enum = Chem.MolToSmiles(mol, doRandom=True)
        smiles_list.append(smiles_enum)
    
    return list(set(smiles_list))  # Remove duplicates


def perturb_conditions(conditions, noise_level=0.05):
    """
    Add small perturbations to reaction conditions for augmentation.
    
    Args:
        conditions: Dict of reaction conditions
        noise_level: Magnitude of perturbations
        
    Returns:
        dict: Perturbed conditions
    """
    perturbed = conditions.copy()
    
    # Add Gaussian noise to continuous variables
    for key in ['temperature', 'time', 'concentration']:
        if key in perturbed:
            noise = random.gauss(0, noise_level * perturbed[key])
            perturbed[key] = max(0, perturbed[key] + noise)
    
    return perturbed
