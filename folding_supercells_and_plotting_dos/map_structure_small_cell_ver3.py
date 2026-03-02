#!/usr/bin/env python3
# fold_supercell_to_primitive.py
# Robust periodic Voronoi folding (no k-means).
# Usage: python fold_supercell_to_primitive.py av_supercell.xyz monoclinic_fully_relaxed.vasp out.vasp

import sys, numpy as np
from ase import Atoms
from ase.io import read, write
import trimerorder as to
from pymatgen.io.ase import AseAtomsAdaptor

def wrap01(x):
    """Wrap fractional coords to [0,1)."""
    return np.mod(x, 1.0)

def pbc_delta(a, b):
    """Minimum-image delta (fractional): a - b on the 3-torus."""
    d = a - b
    return d - np.round(d)

def periodic_nearest_seed(points, seeds):
    """
    Assign each fractional point to nearest seed with minimum-image metric.
    points: (N,3), seeds: (M,3) in [0,1)
    Returns: idx (N,), list_of_lists indices per seed
    """
    # Compute distances to all seeds efficiently
    N, M = len(points), len(seeds)
    # Expand arrays to (N,M,3)
    P = points[:, None, :]          # (N,1,3)
    S = seeds[None, :, :]           # (1,M,3)
    d = pbc_delta(P, S)             # (N,M,3)
    dist2 = np.sum(d*d, axis=2)     # (N,M)
    idx = np.argmin(dist2, axis=1)  # nearest seed index per point

    per_seed = [[] for _ in range(M)]
    for i, j in enumerate(idx):
        per_seed[j].append(i)
    return idx, per_seed

def update_centroids(points, seeds, per_seed):
    """
    Update each seed to the mean of assigned points using minimum-image deltas.
    Returns new_seeds (wrapped).
    """
    new_seeds = seeds.copy()
    for j, ids in enumerate(per_seed):
        if len(ids) == 0:
            # keep original seed if nothing assigned
            continue
        # deltas from seed_j to points, under PBC
        d = pbc_delta(points[ids], seeds[j])
        mean_d = d.mean(axis=0)
        new_seeds[j] = wrap01(seeds[j] + mean_d)
    return new_seeds

def fold_species(points_frac, seeds_frac, n_iter=3):
    """Iteratively assign + update centroids on torus."""
    seeds = seeds_frac.copy()
    for _ in range(n_iter):
        _, per_seed = periodic_nearest_seed(points_frac, seeds)
        seeds = update_centroids(points_frac, seeds, per_seed)
    # Final assignment (for diagnostics)
    idx, per_seed = periodic_nearest_seed(points_frac, seeds)
    return seeds, idx, per_seed

def build_folded(prim, seeds_Ru, seeds_P):
    """Compose folded primitive cell with seeds in primitive order."""
    symbols = prim.get_chemical_symbols()
    prim_frac = prim.get_scaled_positions(wrap=True)
    # Reorder seeds to match primitive symbol order
    out_frac = np.zeros_like(prim_frac)
    ru_iter = iter(seeds_Ru)
    p_iter  = iter(seeds_P)
    for i, s in enumerate(symbols):
        out_frac[i] = next(ru_iter) if s == 'Ru' else next(p_iter)
    folded = Atoms(symbols=symbols, cell=prim.cell, pbc=True)
    folded.set_scaled_positions(out_frac)
    return folded

def min_pair_distance(atoms):
    """Compute minimal pair distance (Å) under PBC for diagnostics."""
    pos = atoms.get_positions()
    cell = atoms.cell
    # brute-force small N (primitive)
    from itertools import combinations
    mind = 1e9
    for i, j in combinations(range(len(pos)), 2):
        d = atoms.get_distance(i, j, mic=True)
        mind = min(mind, d)
    return mind

def main():
    if len(sys.argv) < 4:
        print("Usage: python fold_supercell_to_primitive.py BIG.xyz prim.vasp out.vasp")
        sys.exit(1)

    big_fn, prim_fn, out_fn = sys.argv[1:4]
    big  = read(big_fn)

    print(big)
    prim = read(prim_fn)

    # Use the *target primitive cell* to define fractional coords
    dummy = Atoms(positions=big.get_positions(), cell=prim.cell, pbc=True)
    frac_big = wrap01(dummy.get_scaled_positions(wrap=False))

    sym_big = np.array(big.get_chemical_symbols())
    sym_prim = np.array(prim.get_chemical_symbols())
    prim_frac = prim.get_scaled_positions(wrap=True)

    # Split by species using actual symbols (robust to ordering)
    mask_Ru_big = (sym_big == 'Ru')
    mask_P_big  = (sym_big == 'P')
    mask_Ru_prim = (sym_prim == 'Ru')
    mask_P_prim  = (sym_prim == 'P')

    pts_Ru = frac_big[mask_Ru_big]
    pts_P  = frac_big[mask_P_big]
    seeds_Ru0 = prim_frac[mask_Ru_prim]
    seeds_P0  = prim_frac[mask_P_prim]

    # Iterative periodic Voronoi folding
    seeds_Ru, idx_Ru, per_Ru = fold_species(pts_Ru, seeds_Ru0, n_iter=4)
    seeds_P,  idx_P,  per_P  = fold_species(pts_P,  seeds_P0,  n_iter=4)

    # Build and write
    folded = build_folded(prim, seeds_Ru, seeds_P)
    write(out_fn, folded, format='vasp')

    # Diagnostics
    dmin = min_pair_distance(folded)
    print(f"[ok] Wrote {out_fn}.  Min pair distance: {dmin:.3f} Å")
    # Warn if any primitive site got zero assignments (can happen if species count mismatch)
    zero_ru = sum(1 for ids in per_Ru if len(ids)==0)
    zero_p  = sum(1 for ids in per_P  if len(ids)==0)
    if zero_ru or zero_p:
        print(f"[warn] Empty clusters -> Ru:{zero_ru}, P:{zero_p}. Check species counts or target cell.")



    #reference cell
    #mono_ref_cell=Poscar.from_file("monoclinic_fully_relaxed.vasp","r")
    trimer_direction = np.dot([1,1,0], folded.cell)
    print(trimer_direction)
    trimer_direction=trimer_direction/(np.dot(trimer_direction,trimer_direction)**0.5)
    print(f"trimer direction to check {trimer_direction}")


    #mix-mash in ase and pymatgen atom object
    folded_pymatgen = AseAtomsAdaptor.get_structure(folded)
    bond_dist=to.get_trimer_bond_distribution(folded_pymatgen,trimer_direction,72)
    #print(bond_dist)
    bond_dist=np.array(bond_dist)
    #print(bond_dist)
    to.plot(bond_dist,label="folded_cell", xlabel="bond dist",bins=10) 

    
    #Also compare with the original large cell
    big_pymatgen = AseAtomsAdaptor.get_structure(big)
    bond_dist=to.get_trimer_bond_distribution(big_pymatgen,trimer_direction,1296)
    #print(bond_dist)
    bond_dist=np.array(bond_dist)
    #print(bond_dist)
    to.plot(bond_dist,label="big_cell", xlabel="bond dist",bins=10) 

#finally after we have folded back the large unitcell, I want to see the bond angle and bond lenght distributions along the trimer (which is so far the quantity we are focussed on)


if __name__ == "__main__":
    main()


