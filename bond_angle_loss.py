import torch

def bond_angle_loss(positions, angle_indices, target_angles_rad):
    """
    Calculate loss for bond angle deviations from ideal values.
    
    Args:
        positions: [num_atoms, 3] tensor of atomic coordinates
        angle_indices: [num_angles, 3] tensor of atom indices i-j-k forming angles
                       (j is the central atom)
        target_angles_rad: [num_angles] tensor of ideal angles in radians
        
    Returns:
        Scalar tensor with mean squared deviation from ideal bond angles
        
    Math:
        For each angle i-j-k:
        1. Calculate vectors a = r_i - r_j and b = r_k - r_j
        2. Compute cos(θ) = (a·b)/(|a|·|b|)
        3. Apply arccos to get the angle in radians
        4. Compute squared difference from ideal angle
    """
    # Vectors from central atom to outer atoms
    a = positions[angle_indices[:, 0]] - positions[angle_indices[:, 1]]
    b = positions[angle_indices[:, 2]] - positions[angle_indices[:, 1]]
    
    # Calculate cosine of angles using dot product
    # Adding small epsilon (1e-6) to prevent division by zero
    cos_angles = torch.sum(a * b, dim=1) / (
        torch.norm(a, dim=1) * torch.norm(b, dim=1) + 1e-6)
    
    # Clamp values to valid range for arccos [-1, 1]
    cos_angles = torch.clamp(cos_angles, -1.0, 1.0)
    
    # Calculate angles in radians
    predicted_angles = torch.acos(cos_angles)
    
    # Mean squared error between predicted and target angles
    return torch.mean((predicted_angles - target_angles_rad) ** 2)