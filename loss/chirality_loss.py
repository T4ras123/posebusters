import torch

def chirality_loss(positions, chiral_centers):
    """
    Calculate loss for enforcing correct tetrahedral stereochemistry at chiral centers.
    
    Args:
        positions: [num_atoms, 3] tensor containing atomic coordinates
        chiral_centers: List of tuples (center_idx, [neighbor_indices]), where:
            - center_idx: index of the chiral atom
            - neighbor_indices: list of 4 indices for the connected atoms
            
    Returns:
        Scalar tensor containing the chirality loss
        
    Math:
        For each chiral center with neighbors (a,b,c,d):
        1. Calculate vectors from center to neighbors
        2. Compute signed volume using determinant
        3. Penalize volumes with incorrect sign using ReLU
    """
    loss = 0.0
    for center_idx, neighbors in chiral_centers:
        if len(neighbors) != 4:
            raise ValueError(f"Chiral center at index {center_idx} needs exactly 4 neighbors, got {len(neighbors)}")
        
        # Calculate vectors from chiral center to its neighbors
        vec = positions[neighbors] - positions[center_idx].unsqueeze(0)
        
        # Compute signed volume of tetrahedron (determinant of 3×3 matrix)
        # First three vectors define the handedness
        vol = torch.det(vec[:3])
        
        # Penalize incorrect handedness (negative volume)
        loss += torch.relu(-vol)  # Will be positive when vol is negative
    
    return loss