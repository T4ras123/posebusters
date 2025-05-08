import torch
from loss.bond_length_loss import bond_length_loss
from loss.bond_angle_loss import bond_angle_loss
from loss.aromatic_ring_planarity_loss import ring_planarity_loss
from loss.steric_clash_loss import steric_clash_loss
from loss.chirality_loss import chirality_loss

def total_loss(positions, bonds, angles, rings, vdw_radii, chiral_centers):
    loss = 0.0
    loss += bond_length_loss(positions, bonds.indices, bonds.target_lengths) * 1.0
    loss += bond_angle_loss(positions, angles.indices, angles.target_radians) * 0.5

    if rings.numel() > 0:
        ring_pos = positions[rings]                # [R, ring_size,3]
        loss += ring_planarity_loss(ring_pos) * 0.3

    # pass bonds.indices so bonded pairs aren’t counted as clashes
    loss += steric_clash_loss(positions, vdw_radii, bonds.indices) * 0.2

    if chiral_centers:
        loss += chirality_loss(positions, chiral_centers) * 0.2

    return loss