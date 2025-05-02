def steric_clash_loss(positions, vdw_radii, threshold=0.75):
    """
    positions: [num_atoms, 3]
    vdw_radii: [num_atoms] (van der Waals radii)
    """
    dist_matrix = torch.cdist(positions, positions)
    mask = torch.eye(len(positions), dtype=torch.bool)  # Exclude self
    sum_vdw = vdw_radii.unsqueeze(0) + vdw_radii.unsqueeze(1)
    clash_scores = torch.relu(threshold * sum_vdw - dist_matrix) * ~mask
    return torch.sum(clash_scores ** 2)