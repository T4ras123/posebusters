def chirality_loss(positions, chiral_centers):
    """
    chiral_centers: List of tuples (center_idx, [neighbor_indices])
    """
    loss = 0.0
    for center_idx, neighbors in chiral_centers:
        vec = positions[neighbors] - positions[center_idx]
        vol = torch.det(vec)  # Volume of tetrahedron
        loss += torch.relu(-vol)  # Penalize incorrect handedness
    return loss