import torch

def bond_length_loss(pred_positions, bond_indices, target_lengths):
    """
    Calculate loss for bond length deviations from ideal values.
    
    Args:
        pred_positions: [num_atoms, 3] tensor of atomic coordinates
        bond_indices: [num_bonds, 2] tensor of atom indices forming bonds
        target_lengths: [num_bonds] tensor of ideal bond lengths (in Å)
        
    Returns:
        Scalar tensor with mean squared deviation from ideal bond lengths
        
    Math:
        For each bond between atoms i and j:
        1. Calculate the Euclidean distance ||r_i - r_j||
        2. Compute squared difference from ideal length
        3. Average across all bonds
    """
    # Extract coordinates of atom pairs involved in bonds
    atom_pairs = pred_positions[bond_indices]  # [num_bonds, 2, 3]
    
    # Calculate actual bond lengths
    predicted_lengths = torch.norm(atom_pairs[:, 0] - atom_pairs[:, 1], dim=1)
    
    # Mean squared error between predicted and target bond lengths
    return torch.mean((predicted_lengths - target_lengths) ** 2)