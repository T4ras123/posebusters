def bond_angle_loss(positions, angle_indices, target_angles_rad):
    """
    angle_indices: [num_angles, 3] (tensor of atom indices i-j-k)
    target_angles_rad: [num_angles] (ideal angles in radians)
    """
    a = positions[angle_indices[:, 0]] - positions[angle_indices[:, 1]]
    b = positions[angle_indices[:, 2]] - positions[angle_indices[:, 1]]
    cos_angles = torch.sum(a * b, dim=1) / (
        torch.norm(a, dim=1) * torch.norm(b, dim=1) + 1e-6)
    predicted_angles = torch.acos(cos_angles)
    return torch.mean((predicted_angles - target_angles_rad) ** 2)