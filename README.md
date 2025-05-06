
## Loss Components

### 1. Bond Length Loss
Ensures bonds are close to their ideal lengths based on chemistry tables.

```
Loss = (predicted_length - target_length)²
```

### 2. Bond Angle Loss
Enforces proper angles between three bonded atoms (i-j-k).

```
Loss = (predicted_angle - target_angle)²
```

### 3. Aromatic Ring Planarity Loss
Ensures all atoms in a ring lie on a plane.

```
Loss = distances_from_best_fit_plane²
```

### 4. Steric Clash Loss
Prevents atoms from overlapping by checking distances against van der Waals radii.

```
Clash = ReLU(threshold × (r_i + r_j) - distance_ij)²
```

### 5. Chirality Loss
Enforces correct tetrahedral stereochemistry at chiral centers.

```
Loss = ReLU(-volume)  # Penalizes incorrect handedness
```

## Mathematical Foundation

### Bond Length Loss
For bonds between atoms i and j:
- Loss = (1/N_bonds) * Σ(||r_i - r_j|| - d_ideal)²
- Where r_i is the position of atom i and d_ideal is the target bond length

### Bond Angle Loss
For angle θ_ijk formed by atoms i-j-k:
- cosθ_ijk = ((r_i - r_j) · (r_k - r_j)) / (||r_i - r_j|| · ||r_k - r_j||)
- Loss = (1/N_angles) * Σ(θ_ijk - θ_ideal)²

### Ring Planarity Loss
For a ring with atoms i=1,...,N:
1. Calculate centroid: c = (1/N) * Σr_i
2. Find best-fit plane via SVD on centered coordinates
3. Loss = (1/N) * Σ((r_i - c) · normal)²

### Steric Clash Loss
For non-bonded atoms i and j:
- Clash = Σ ReLU(γ(R_i + R_j) - ||r_i - r_j||)²
- Where R_i is the van der Waals radius of atom i and γ is a threshold (typically 0.75)

### Chirality Loss
For a chiral center with neighbors a,b,c,d:
- Volume = det([r_a-r_center, r_b-r_center, r_c-r_center])
- Loss = ReLU(-Volume) to penalize incorrect handedness

## Requirements

- PyTorch
- NumPy (for certain utility functions)

