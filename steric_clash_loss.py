import torch

def steric_clash_loss(positions, vdw_radii, threshold=0.75):
    """
    Calculate loss for steric clashes (overlapping atoms).
    
    Args:
        positions: [num_atoms, 3] tensor of atomic coordinates
        vdw_radii: [num_atoms] tensor of van der Waals radii
        threshold: Scaling factor for sum of radii (default: 0.75)
                   Lower values allow more overlap before penalizing
        
    Returns:
        Scalar tensor with sum of squared clash penalties
        
    Math:
        For each pair of atoms i and j:
        1. Calculate pairwise distances between all atoms
        2. Determine minimum allowed distance as threshold × (r_i + r_j)
        3. Compute clash score as ReLU(min_distance - actual_distance)
        4. Square and sum all positive clash scores
    """
    # Calculate pairwise distances between all atoms
    dist_matrix = torch.cdist(positions, positions)  # [num_atoms, num_atoms]
    
    # Create a mask to exclude self-interactions
    mask = torch.eye(len(positions), dtype=torch.bool, device=positions.device)
    
    # Calculate minimum allowed distances (threshold × sum of vdW radii)
    sum_vdw = vdw_radii.unsqueeze(0) + vdw_radii.unsqueeze(1)  # [num_atoms, num_atoms]
    min_allowed_dist = threshold * sum_vdw
    
    # Calculate clash scores: positive when atoms are too close
    # ReLU ensures only overlapping atoms contribute to the loss
    clash_scores = torch.relu(min_allowed_dist - dist_matrix)
    
    # Apply mask to exclude self-interactions
    clash_scores = clash_scores * (~mask)
    
    # Return sum of squared clash scores
    return torch.sum(clash_scores ** 2)