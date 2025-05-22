from rdkit import Chem
from rdkit.Chem import rdchem

def is_valid_smiles(smi: str) -> bool:
    """
    Returns True iff RDKit can parse + sanitize this SMILES.
    """
    try:
        mol = Chem.MolFromSmiles(smi, sanitize=False)
        if mol is None:
            return False
        Chem.SanitizeMol(mol)
        return True
    except (Chem.rdchem.MolSanitizeException, ValueError):
        return False
    except Exception:
        return False

def canonical_smiles(smi: str) -> str:
    """
    Returns the canonical SMILES for a valid molecule, or empty string.
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol, canonical=True)

def inchikey(smi: str) -> str:
    """
    Returns the InChIKey for a valid molecule, or empty string.
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return ""
    # requires RDKit built with InChI support
    return Chem.MolToInchiKey(mol)

def are_same_molecule(smi1: str, smi2: str) -> bool:
    """
    Compare two SMILES via InChIKey (more robust) or canonical SMILES.
    """
    key1 = inchikey(smi1)
    key2 = inchikey(smi2)
    if key1 and key2:
        return key1 == key2
    # fallback to canonical SMILES
    return canonical_smiles(smi1) == canonical_smiles(smi2)

if __name__ == "__main__":
    test_smiles = [
        "CC(=O)OC1=CC=CC=C1C(=O)O",
        "OC(=O)c1ccccc1OC=O",           # same as above but different ordering
        "C1CCCCC1",                     
        "C1CCCCC1C",                   
    ]

    # validity check
    for smi in test_smiles:
        print(f"SMILES: {smi}, Valid: {is_valid_smiles(smi)}")

    print("\nComparisons:")
    pairs = [
        ("CC(=O)OC1=CC=CC=C1C(=O)O", "OC(=O)c1ccccc1OC=O"),
        ("C1CCCCC1", "C1CCCCC1C"),
    ]
    for a, b in pairs:
        print(f"{a}  vs  {b} → same? {are_same_molecule(a,b)}")