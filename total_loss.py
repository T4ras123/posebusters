import torch
from .loss.bond_length_loss import bond_length_loss
from .loss.bond_angle_loss import bond_angle_loss
from .loss.aromatic_ring_planarity_loss import ring_planarity_loss
from .loss.steric_clash_loss import steric_clash_loss
from .loss.chirality_loss import chirality_loss

def total_loss(positions, bonds, angles, rings, vdw_radii, chiral_centers):
    """
    Compute the total molecular geometry validation loss.
    
    Args:
        positions: [num_atoms, 3] tensor of atomic coordinates
        bonds: Object with attributes:
            - indices: [num_bonds, 2] tensor of atom indices forming bonds
            - target_lengths: [num_bonds] tensor of ideal bond lengths
        angles: Object with attributes:
            - indices: [num_angles, 3] tensor of atom indices forming angles
            - target_radians: [num_angles] tensor of ideal angles in radians
        rings: [num_rings, ring_size, 3] tensor of ring atom coordinates
        vdw_radii: [num_atoms] tensor of van der Waals radii
        chiral_centers: List of tuples (center_idx, [neighbor_indices])
        
    Returns:
        Scalar tensor representing the weighted sum of all loss components
    """
    loss = 0.0
    # Bond length component (highest weight)
    loss += 1.0 * bond_length_loss(positions, bonds.indices, bonds.target_lengths)
    # Bond angle component
    loss += 0.5 * bond_angle_loss(positions, angles.indices, angles.target_radians)
    # Ring planarity component
    loss += 0.3 * ring_planarity_loss(positions[rings])
    # Steric clash prevention component
    loss += 0.2 * steric_clash_loss(positions, vdw_radii)
    # Chirality preservation component
    loss += 0.2 * chirality_loss(positions, chiral_centers)
    return loss