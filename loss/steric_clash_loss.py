import torch

def steric_clash_loss(positions, vdw_radii, bond_indices=None, threshold=0.75):
    """
    Calculate loss for steric clashes (overlapping atoms), *excluding* bonded pairs.
    
    Args:
        positions: [N,3] tensor
        vdw_radii: [N] tensor
        bond_indices: optional [num_bonds,2] tensor to mask out bonded pairs
        threshold: float
        
    Returns:
        sum of squared clash penalties over nonbonded atom pairs
    """
    # pairwise distances
    dist = torch.cdist(positions, positions)          # [N,N]
    N = positions.size(0)
    # mask self
    eye = torch.eye(N, dtype=torch.bool, device=positions.device)
    
    # min allowed distance
    sum_vdw = vdw_radii.unsqueeze(0) + vdw_radii.unsqueeze(1)
    min_allowed = threshold * sum_vdw
    
    # raw clash
    clash = torch.relu(min_allowed - dist)
    
    # exclude self
    clash = clash.masked_fill(eye, 0.0)
    
    # exclude bonded pairs
    if bond_indices is not None:
        b0, b1 = bond_indices[:,0], bond_indices[:,1]
        clash[b0, b1] = 0
        clash[b1, b0] = 0
    
    return torch.sum(clash**2)