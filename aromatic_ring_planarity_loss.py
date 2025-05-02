def ring_planarity_loss(ring_positions):
    """
    ring_positions: [batch_size, num_ring_atoms, 3] (tensor)
    """
    centroid = torch.mean(ring_positions, dim=1, keepdim=True)
    centered = ring_positions - centroid
    cov = torch.bmm(centered.transpose(1, 2), centered)  # Covariance matrix
    _, _, v = torch.svd(cov)  # Singular value decomposition
    normals = v[:, :, -1]  # Last eigenvector = plane normal
    distances = torch.einsum('bij,bj->bi', centered, normals)
    return torch.mean(distances ** 2)