def bond_length_loss(pred_positions, bond_indices, target_lengths):
    """
    pred_positions: [num_atoms, 3] (tensor)
    bond_indices: [num_bonds, 2] (tensor of atom indices)
    target_lengths: [num_bonds] (tensor of ideal lengths)
    """
    atom_pairs = pred_positions[bond_indices]  # [num_bonds, 2, 3]
    predicted_lengths = torch.norm(atom_pairs[:, 0] - atom_pairs[:, 1], dim=1)
    return torch.mean((predicted_lengths - target_lengths) ** 2)