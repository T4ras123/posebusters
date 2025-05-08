# PoseBusters: Differentiable Molecular Geometry Validation

PoseBusters provides a suite of differentiable loss functions for validating and optimizing 3D molecular geometries in PyTorch. These losses enforce chemically meaningful constraints, making them suitable for deep learning models that generate or refine atomic coordinates.

---

## Overview

PoseBusters is designed to ensure that predicted molecular structures are physically and chemically plausible. It does so by penalizing deviations from ideal bond lengths, bond angles, aromatic ring planarity, steric clashes, and incorrect chirality. All losses are fully differentiable and can be used as part of a neural network's training objective.

---

## Loss Components

### 1. Bond Length Loss (`bond_length_loss`)
- **Purpose:** Penalizes deviations of predicted bond lengths from ideal values.
- **Input:**
  - `pred_positions`: `[num_atoms, 3]` tensor of atomic coordinates
  - `bond_indices`: `[num_bonds, 2]` tensor of atom index pairs
  - `target_lengths`: `[num_bonds]` tensor of ideal bond lengths (Å)
- **Mathematics:**
  - For each bond between atoms i and j:
    - Compute the Euclidean distance: \( d_{ij} = \|r_i - r_j\| \)
    - Loss: \( (d_{ij} - d_{ij}^{\text{ideal}})^2 \)
  - The final loss is the mean squared deviation over all bonds.

### 2. Bond Angle Loss (`bond_angle_loss`)
- **Purpose:** Penalizes deviations of predicted bond angles from ideal values.
- **Input:**
  - `positions`: `[num_atoms, 3]` tensor of atomic coordinates
  - `angle_indices`: `[num_angles, 3]` tensor of atom indices (i, j, k), with j as the central atom
  - `target_angles_rad`: `[num_angles]` tensor of ideal angles (radians)
- **Mathematics:**
  - For each angle i-j-k:
    - Vectors: \( a = r_i - r_j,\ b = r_k - r_j \)
    - Cosine: \( \cos\theta = \frac{a \cdot b}{\|a\|\|b\|} \)
    - Angle: \( \theta = \arccos(\cos\theta) \)
    - Loss: \( (\theta - \theta_{\text{ideal}})^2 \)
  - The final loss is the mean squared deviation over all angles.

### 3. Aromatic Ring Planarity Loss (`ring_planarity_loss`)
- **Purpose:** Enforces planarity of aromatic rings by penalizing the mean squared distance of ring atoms from the best-fit plane.
- **Input:**
  - `ring_positions`: `[batch_size, num_ring_atoms, 3]` tensor of ring atom coordinates (each batch element is a different ring)
- **Mathematics:**
  - For each ring:
    1. Calculate centroid: \( c = \frac{1}{N} \sum_i r_i \)
    2. Center coordinates: \( r'_i = r_i - c \)
    3. Compute covariance matrix: \( C = (r')^T r' \)
    4. Use SVD to find the best-fit plane (normal = eigenvector with smallest eigenvalue)
    5. Compute squared distances from each atom to the plane: \( d_i = (r'_i \cdot n)^2 \)
  - The final loss is the mean squared distance from the plane, averaged over all rings and atoms.

### 4. Steric Clash Loss (`steric_clash_loss`)
- **Purpose:** Penalizes steric clashes by applying a penalty to non-bonded atom pairs whose distance is less than a threshold times the sum of their van der Waals radii.
- **Input:**
  - `positions`: `[N, 3]` tensor of atomic coordinates
  - `vdw_radii`: `[N]` tensor of van der Waals radii (Å)
  - `bond_indices`: Optional `[num_bonds, 2]` tensor to mask out bonded pairs
  - `threshold`: Float (default 0.75)
- **Mathematics:**
  - For each non-bonded atom pair (i, j):
    - Minimum allowed distance: \( d_{ij}^{\text{min}} = \text{threshold} \times (R_i + R_j) \)
    - Clash: \( \text{clash}_{ij} = \text{ReLU}(d_{ij}^{\text{min}} - \|r_i - r_j\|) \)
    - Loss: \( \sum_{i<j} \text{clash}_{ij}^2 \)
  - Bonded pairs and self-pairs are excluded from the sum.

### 5. Chirality Loss (`chirality_loss`)
- **Purpose:** Enforces correct tetrahedral stereochemistry at chiral centers by penalizing incorrect handedness.
- **Input:**
  - `positions`: `[num_atoms, 3]` tensor of atomic coordinates
  - `chiral_centers`: List of tuples `(center_idx, [neighbor_indices])`, where `neighbor_indices` is a list of 4 atom indices bonded to the chiral center
- **Mathematics:**
  - For each chiral center:
    1. Calculate vectors from center to neighbors: \( v_i = r_{n_i} - r_{\text{center}} \) for i=1..4
    2. Compute signed volume: \( V = \det([v_1, v_2, v_3]) \)
    3. Penalize negative volume (incorrect handedness): \( \text{Loss} = \text{ReLU}(-V) \)
  - The final loss is the sum over all chiral centers.

### 6. Total Loss (`total_loss`)
- **Purpose:** Aggregates all the above losses with configurable weights. Losses are excluded if not applicable (e.g., no rings or chiral centers).
- **Input:**
  - `positions`: `[num_atoms, 3]` tensor of atomic coordinates
  - `bonds`: Named tuple or object with `indices` and `target_lengths`
  - `angles`: Named tuple or object with `indices` and `target_radians`
  - `rings`: `[num_rings, num_ring_atoms]` tensor of atom indices for each aromatic ring (empty if none)
  - `vdw_radii`: `[num_atoms]` tensor of van der Waals radii (Å)
  - `chiral_centers`: List of tuples as above
- **Mathematics:**
  - \(
    \text{Total Loss} = w_1 \cdot \text{Bond Length Loss} + w_2 \cdot \text{Bond Angle Loss} + w_3 \cdot \text{Ring Planarity Loss} + w_4 \cdot \text{Steric Clash Loss} + w_5 \cdot \text{Chirality Loss}
    \)
  - Default weights: 1.0, 0.5, 0.3, 0.2, 0.2 (see `total_loss.py`)

---

## Required Inputs (Summary)

- `positions`: `[num_atoms, 3]` tensor of atomic coordinates
- `bonds`: Named tuple or object with:
  - `indices`: `[num_bonds, 2]` tensor of atom index pairs
  - `target_lengths`: `[num_bonds]` tensor of ideal bond lengths (Å)
- `angles`: Named tuple or object with:
  - `indices`: `[num_angles, 3]` tensor of atom indices (i, j, k), with j as the central atom
  - `target_radians`: `[num_angles]` tensor of ideal bond angles (radians)
- `rings`: `[num_rings, num_ring_atoms]` tensor of atom indices for each aromatic ring (empty if none)
- `vdw_radii`: `[num_atoms]` tensor of van der Waals radii (Å)
- `chiral_centers`: List of tuples, each with:
  - `center_idx`: Index of the chiral atom
  - `neighbor_indices`: List of 4 indices for atoms bonded to the chiral center

---

## Example Usage

Below is a minimal example for methane (CH₄) and benzene (C₆H₆):

```python
import torch
from collections import namedtuple
from total_loss import total_loss

# Example for methane (CH₄)
positions = torch.tensor([
    [0.000,  0.000,  0.000],  # C
    [0.629,  0.629,  0.629],  # H
    [-0.629, -0.629,  0.629], # H
    [-0.629,  0.629, -0.629], # H
    [0.629, -0.629, -0.629],  # H
], dtype=torch.float32)

Bond = namedtuple('Bond', ['indices', 'target_lengths'])
bonds = Bond(
    indices=torch.tensor([[0, 1], [0, 2], [0, 3], [0, 4]]),
    target_lengths=torch.tensor([1.09, 1.09, 1.09, 1.09])
)

Angle = namedtuple('Angle', ['indices', 'target_radians'])
tetrahedral_angle = 109.5 * torch.pi / 180.0
angles = Angle(
    indices=torch.tensor([
        [1, 0, 2], [1, 0, 3], [1, 0, 4],
        [2, 0, 3], [2, 0, 4], [3, 0, 4]
    ]),
    target_radians=torch.tensor([tetrahedral_angle] * 6)
)

rings = torch.empty((0,0), dtype=torch.long)
vdw_radii = torch.tensor([1.70,1.20,1.20,1.20,1.20])
chiral_centers = []

loss = total_loss(positions, bonds, angles, rings, vdw_radii, chiral_centers)
print("Methane loss:", loss.item())
```

For more complex molecules (e.g., benzene), see `tests.py` for full examples including aromatic rings and visualization.

---

## Notes
- All loss functions are fully differentiable and suitable for use in PyTorch model training.
- Losses are only computed for the relevant features present in the molecule (e.g., no ring loss if no rings).
- The weights in `total_loss` can be adjusted as needed for your application.
- For more details, see the code in the `loss/` directory and the test cases in `tests.py`.
