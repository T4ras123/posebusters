def total_loss(positions, bonds, angles, rings, vdw_radii, chiral_centers):
    loss = 0.0
    loss += 1.0 * bond_length_loss(positions, bonds.indices, bonds.target_lengths)
    loss += 0.5 * bond_angle_loss(positions, angles.indices, angles.target_radians)
    loss += 0.3 * ring_planarity_loss(positions[rings])
    loss += 0.2 * steric_clash_loss(positions, vdw_radii)
    loss += 0.2 * chirality_loss(positions, chiral_centers)
    return loss