import torch

def ring_planarity_loss(ring_positions):
    """
    Calculate loss for deviations from ring planarity.
    
    Args:
        ring_positions: [batch_size, num_ring_atoms, 3] tensor of ring atom coordinates
                        Each batch element is a different ring
        
    Returns:
        Scalar tensor with mean squared distance from best-fit plane
        
    Math:
        For each ring:
        1. Calculate centroid of the ring atoms
        2. Center coordinates by subtracting centroid
        3. Compute covariance matrix of centered coordinates
        4. Use SVD to find the best-fit plane (normal = eigenvector with smallest eigenvalue)
        5. Calculate squared distances from each atom to the plane (dot product with normal)
    """
    # Calculate centroids for each ring
    centroid = torch.mean(ring_positions, dim=1, keepdim=True)  # [batch_size, 1, 3]
    
    # Center coordinates
    centered = ring_positions - centroid  # [batch_size, num_ring_atoms, 3]
    
    # Compute covariance matrix for each ring
    cov = torch.bmm(centered.transpose(1, 2), centered)  # [batch_size, 3, 3]
    
    # Perform SVD - the last column of v contains the normal to the best-fit plane
    # (corresponds to the smallest singular value)
    _, _, v = torch.svd(cov)  # v: [batch_size, 3, 3]
    normals = v[:, :, -1]  # [batch_size, 3]
    
    # Calculate distances from points to the plane using dot product
    distances = torch.einsum('bij,bj->bi', centered, normals)
    
    # Mean squared distance from plane
    return torch.mean(distances ** 2)