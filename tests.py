import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from total_loss import total_loss
from collections import namedtuple

def test_methane():
    """Test the molecular geometry validation on methane (CH₄)."""
    # Define methane structure (in Angstroms)
    # Carbon at origin, hydrogens at vertices of a tetrahedron
    positions = torch.tensor([
        [0.000,  0.000,  0.000],  # C
        [0.629,  0.629,  0.629],  # H
        [-0.629, -0.629,  0.629],  # H
        [-0.629,  0.629, -0.629],  # H
        [0.629, -0.629, -0.629],  # H
    ], dtype=torch.float32)
    
    # Define bonds (C-H bonds)
    Bond = namedtuple('Bond', ['indices', 'target_lengths'])
    bonds = Bond(
        indices=torch.tensor([
            [0, 1], [0, 2], [0, 3], [0, 4]
        ]),
        target_lengths=torch.tensor([1.09, 1.09, 1.09, 1.09])  # C-H bond ~1.09 Å
    )
    
    # Define angles (H-C-H angles)
    Angle = namedtuple('Angle', ['indices', 'target_radians'])
    # Tetrahedral angle is approximately 109.5 degrees = 1.91 radians
    tetrahedral_angle = 109.5 * np.pi / 180.0
    angles = Angle(
        indices=torch.tensor([
            [1, 0, 2], [1, 0, 3], [1, 0, 4],
            [2, 0, 3], [2, 0, 4], [3, 0, 4]
        ]),
        target_radians=torch.tensor([
            tetrahedral_angle, tetrahedral_angle, tetrahedral_angle,
            tetrahedral_angle, tetrahedral_angle, tetrahedral_angle
        ])
    )
    
    rings = torch.empty((0,0), dtype=torch.long)
    vdw_radii = torch.tensor([1.70,1.20,1.20,1.20,1.20])
    chiral_centers = []   
    loss = total_loss(positions, bonds, angles, rings, vdw_radii, chiral_centers)
    print("Methane loss:", loss.item())
    
    # Create slightly distorted methane
    distorted_positions = positions.clone()
    distorted_positions[1, 0] += 0.2  
    
    # Calculate loss for distorted methane
    distorted_loss = total_loss(distorted_positions, bonds, angles, rings, vdw_radii, chiral_centers)
    print(f"Distorted methane loss: {distorted_loss.item():.6f}")
    print(f"Loss increase due to distortion: {(distorted_loss - loss).item():.6f}")
    
    return positions, distorted_positions

def test_benzene():
    """Test the molecular geometry validation on benzene (C₆H₆)."""
    # Create benzene ring coordinates
    ring_radius = 1.40  # C-C distance in benzene is ~1.40 Å
    angle_step = 2 * np.pi / 6  # 60 degrees
    
    # Carbon atoms in a planar hexagon
    c_positions = []
    for i in range(6):
        angle = i * angle_step
        c_positions.append([ring_radius * np.cos(angle), 
                            ring_radius * np.sin(angle),
                            0.0])
    
    # Hydrogen atoms extending outward from each carbon
    h_positions = []
    for i in range(6):
        angle = i * angle_step
        h_positions.append([(ring_radius + 1.09) * np.cos(angle), 
                            (ring_radius + 1.09) * np.sin(angle),
                            0.0])
    
    # Combine all atoms (6 carbon + 6 hydrogen)
    positions = torch.tensor(c_positions + h_positions, dtype=torch.float32)
    
    # Define bonds (C-C and C-H bonds)
    c_c_bonds = [[i, (i+1)%6] for i in range(6)]  # Carbon ring
    c_h_bonds = [[i, i+6] for i in range(6)]      # Carbon-hydrogen bonds
    
    Bond = namedtuple('Bond', ['indices', 'target_lengths'])
    bonds = Bond(
        indices=torch.tensor(c_c_bonds + c_h_bonds),
        target_lengths=torch.tensor([1.40] * 6 + [1.09] * 6)  # C-C and C-H bonds
    )
    
    # Define angles (C-C-C and C-C-H angles)
    c_c_c_angles = [[i, (i+1)%6, (i+2)%6] for i in range(6)]  # Carbon ring angles
    c_c_h_angles = [[(i-1)%6, i, i+6] for i in range(6)]      # H-C-C angles
    
    Angle = namedtuple('Angle', ['indices', 'target_radians'])
    angles = Angle(
        indices=torch.tensor(c_c_c_angles + c_c_h_angles),
        target_radians=torch.tensor([120 * np.pi/180] * 6 + [120 * np.pi/180] * 6)
    )
    
    # Define aromatic ring
    # We only care about carbon positions for ring planarity
    rings = torch.tensor([[0, 1, 2, 3, 4, 5]], dtype=torch.long) 
    
    # van der Waals radii
    vdw_radii = torch.tensor([1.70] * 6 + [1.20] * 6)  # C and H
    
    # No chiral centers in benzene
    chiral_centers = []
    
    # Calculate total loss
    loss = total_loss(positions, bonds, angles, rings, vdw_radii, chiral_centers)
    print(f"\nBenzene geometry validation loss: {loss.item():.6f}")
    
    # Create distorted benzene by moving one carbon out of plane
    distorted_positions = positions.clone()
    distorted_positions[0, 2] += 0.5  # Move first carbon atom 0.5 Å in z direction
    
    # Calculate loss for distorted benzene
    distorted_loss = total_loss(distorted_positions, bonds, angles, rings, vdw_radii, chiral_centers)
    print(f"Distorted benzene loss: {distorted_loss.item():.6f}")
    print(f"Loss increase due to distortion: {(distorted_loss - loss).item():.6f}")
    
    return positions, distorted_positions

def visualize_molecules(methane, methane_distorted, benzene, benzene_distorted):
    """Visualize molecule positions in 3D plots"""
    fig = plt.figure(figsize=(15, 8))
    
    # Plot methane
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.scatter(methane[:1, 0], methane[:1, 1], methane[:1, 2], c='black', s=100, label='C')
    ax1.scatter(methane[1:, 0], methane[1:, 1], methane[1:, 2], c='whitesmoke', edgecolors='gray', s=80, label='H')
    for i in range(1, 5):
        ax1.plot([methane[0, 0], methane[i, 0]], 
                 [methane[0, 1], methane[i, 1]],
                 [methane[0, 2], methane[i, 2]], 'k-')
    ax1.set_title('Methane (CH₄)')
    ax1.legend()
    
    # Plot distorted methane
    ax2 = fig.add_subplot(222, projection='3d')
    ax2.scatter(methane_distorted[:1, 0], methane_distorted[:1, 1], methane_distorted[:1, 2], c='black', s=100, label='C')
    ax2.scatter(methane_distorted[1:, 0], methane_distorted[1:, 1], methane_distorted[1:, 2], c='whitesmoke', edgecolors='gray', s=80, label='H')
    for i in range(1, 5):
        ax2.plot([methane_distorted[0, 0], methane_distorted[i, 0]], 
                 [methane_distorted[0, 1], methane_distorted[i, 1]],
                 [methane_distorted[0, 2], methane_distorted[i, 2]], 'k-')
    ax2.set_title('Distorted Methane')
    ax2.legend()
    
    # Plot benzene
    ax3 = fig.add_subplot(223, projection='3d')
    ax3.scatter(benzene[:6, 0], benzene[:6, 1], benzene[:6, 2], c='black', s=100, label='C')
    ax3.scatter(benzene[6:, 0], benzene[6:, 1], benzene[6:, 2], c='whitesmoke', edgecolors='gray', s=80, label='H')
    # Draw carbon ring
    for i in range(6):
        ax3.plot([benzene[i, 0], benzene[(i+1)%6, 0]], 
                 [benzene[i, 1], benzene[(i+1)%6, 1]],
                 [benzene[i, 2], benzene[(i+1)%6, 2]], 'k-')
    # Draw C-H bonds
    for i in range(6):
        ax3.plot([benzene[i, 0], benzene[i+6, 0]], 
                 [benzene[i, 1], benzene[i+6, 1]],
                 [benzene[i, 2], benzene[i+6, 2]], 'k-')
    ax3.set_title('Benzene (C₆H₆)')
    ax3.legend()
    
    # Plot distorted benzene
    ax4 = fig.add_subplot(224, projection='3d')
    ax4.scatter(benzene_distorted[:6, 0], benzene_distorted[:6, 1], benzene_distorted[:6, 2], c='black', s=100, label='C')
    ax4.scatter(benzene_distorted[6:, 0], benzene_distorted[6:, 1], benzene_distorted[6:, 2], c='whitesmoke', edgecolors='gray', s=80, label='H')
    # Draw carbon ring
    for i in range(6):
        ax4.plot([benzene_distorted[i, 0], benzene_distorted[(i+1)%6, 0]], 
                 [benzene_distorted[i, 1], benzene_distorted[(i+1)%6, 1]],
                 [benzene_distorted[i, 2], benzene_distorted[(i+1)%6, 2]], 'k-')
    # Draw C-H bonds
    for i in range(6):
        ax4.plot([benzene_distorted[i, 0], benzene_distorted[i+6, 0]], 
                 [benzene_distorted[i, 1], benzene_distorted[i+6, 1]],
                 [benzene_distorted[i, 2], benzene_distorted[i+6, 2]], 'k-')
    ax4.set_title('Distorted Benzene')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('molecule_geometries.png')
    plt.show()

if __name__ == "__main__":
    print("Testing PoseBusters molecular geometry validation...")
    methane, methane_distorted = test_methane()
    benzene, benzene_distorted = test_benzene()
    
    visualize_molecules(methane.numpy(), methane_distorted.numpy(), 
                        benzene.numpy(), benzene_distorted.numpy())
    
    print("\nTest completed. Visualization saved as 'molecule_geometries.png'")

# Output:

# Testing PoseBusters molecular geometry validation...
# Methane loss: 0.001050713355652988
# Distorted methane loss: 0.006740
# Loss increase due to distortion: 0.005689

# Benzene geometry validation loss: 0.038393
# Distorted benzene loss: 0.042363
# Loss increase due to distortion: 0.003970
