#!/usr/bin/env python3
"""
Compute space correlation function using fast ASE-based order parameter calculation.
No dependency on trimerorder module.
"""

import sys
import numpy as np
from pathlib import Path
from ase import Atoms
from ase.io import read
from ase.neighborlist import NeighborList
import time
import argparse
from tqdm import tqdm


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Compute space correlation from MD trajectories using fast method.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('-parent', '--parent-dir', required=True, type=str,
                        help='Parent directory containing temperature folders')
    
    parser.add_argument('-i', '--input-pattern', required=True, type=str,
                        help='Pattern to match XYZ files')
    
    parser.add_argument('-o', '--output-dir', required=True, type=str,
                        help='Output directory for NPZ files')
    
    parser.add_argument('-prefix', '--file-prefix', default='space_corr', type=str,
                        help='Prefix for output files (default: space_corr)')
    
    parser.add_argument('-subdir', '--subdirectory', default='lammps_out', type=str,
                        help='Subdirectory in temperature folders (default: lammps_out)')
    
    parser.add_argument('-start', '--start-frame', default=1990, type=int,
                        help='Starting frame index (default: 1990)')
    
    parser.add_argument('-step', '--frame-step', default=102, type=int,
                        help='Step between frames (default: 102)')
    
    parser.add_argument('-delta', '--delta-d', default=1.0, type=float,
                        help='Bin size for correlation function (default: 1.0 Å)')
    
    parser.add_argument('-temps', '--temperatures', nargs='+', default=None,
                        help='Specific temperatures to compute (default: all)')
    
    parser.add_argument('--max-distance', default=30.0, type=float,
                        help='Maximum distance for correlation (default: 30 Å)')
    
    parser.add_argument('--parallel', action='store_true',
                        help='Compute correlation ALONG [110] (parallel). Default is PERPENDICULAR.')
    
    return parser.parse_args()


def identify_ru_110_pairs(atoms):
    """
    Identify Ru-Ru pairs along [110] direction.
    Returns neighbor graph for computing order parameters.
    """
    symbols = np.array(atoms.get_chemical_symbols())
    ru_mask = symbols == 'Ru'
    ru_indices = np.where(ru_mask)[0]
    
    # Get [110] direction
    cell = atoms.get_cell()
    direction_110_frac = np.array([1.0, 1.0, 0.0])
    direction_110_real = cell.T @ direction_110_frac
    direction_110_unit = direction_110_real / np.linalg.norm(direction_110_real)
    
    # Build neighbor list
    cutoff_nearest = 4.0
    nl = NeighborList([cutoff_nearest/2] * len(atoms), self_interaction=False, bothways=False)
    nl.update(atoms)
    
    # Build neighbor graph
    ru_neighbor_graph = {}
    
    for ru_idx in ru_indices:
        indices, offsets = nl.get_neighbors(ru_idx)
        
        # Filter for Ru neighbors
        ru_neighbor_mask = np.isin(indices, ru_indices)
        if not np.any(ru_neighbor_mask):
            continue
        
        ru_neighbors = indices[ru_neighbor_mask]
        ru_offsets = offsets[ru_neighbor_mask]
        
        # Calculate positions with PBC
        current_pos = atoms.positions[ru_idx]
        neighbor_positions = atoms.positions[ru_neighbors].copy()
        for i, offset in enumerate(ru_offsets):
            neighbor_positions[i] += offset @ cell
        
        # Calculate displacements
        displacements = neighbor_positions - current_pos
        distances = np.linalg.norm(displacements, axis=1)
        
        # Project onto [110]
        projections = np.dot(displacements, direction_110_unit)
        alignments = np.abs(projections) / (distances + 1e-10)
        
        # Filter aligned neighbors
        aligned_mask = (alignments > 0.9) & (distances < cutoff_nearest)
        
        if not np.any(aligned_mask):
            continue
        
        aligned_neighbors = ru_neighbors[aligned_mask]
        aligned_projections = projections[aligned_mask]
        
        # Separate forward/backward neighbors
        forward_mask = aligned_projections > 0
        backward_mask = aligned_projections < 0
        
        neighbors_list = []
        
        # Get nearest forward neighbor
        forward_indices = np.where(forward_mask)[0]
        if len(forward_indices) > 0:
            neighbors_list.append(aligned_neighbors[forward_indices[0]])
        
        # Get nearest backward neighbor
        backward_indices = np.where(backward_mask)[0]
        if len(backward_indices) > 0:
            neighbors_list.append(aligned_neighbors[backward_indices[0]])
        
        if neighbors_list:
            ru_neighbor_graph[ru_idx] = neighbors_list
    
    return ru_neighbor_graph, direction_110_unit


def compute_dimer_order(atoms, ru_neighbor_graph, direction_110_unit):
    """
    Compute dimer order parameter for each Ru atom.
    Dimer order = d_23 - d_12 (difference between consecutive bond lengths).
    """
    cell = atoms.get_cell()
    dimer_orders = []
    ru_indices_list = []
    ru_positions_list = []
    
    for ru_idx, neighbor_indices in ru_neighbor_graph.items():
        if len(neighbor_indices) < 1:
            continue
        
        pos_i = atoms.positions[ru_idx]
        
        # Find forward neighbor
        forward_neighbor = None
        d_12 = None
        
        for neighbor_idx in neighbor_indices:
            pos_j = atoms.positions[neighbor_idx]
            
            # Vector with PBC
            vec_ij = pos_j - pos_i
            vec_ij = vec_ij - cell.T @ np.round(np.linalg.solve(cell.T, vec_ij))
            
            projection = np.dot(vec_ij, direction_110_unit)
            
            if projection > 0:  # Forward
                forward_neighbor = neighbor_idx
                d_12 = np.linalg.norm(vec_ij)
                break
        
        if forward_neighbor is None:
            continue
        
        # Find forward neighbor of j
        if forward_neighbor not in ru_neighbor_graph:
            continue
        
        j_neighbors = ru_neighbor_graph[forward_neighbor]
        if len(j_neighbors) < 1:
            continue
        
        pos_j = atoms.positions[forward_neighbor]
        
        d_23 = None
        for neighbor_idx in j_neighbors:
            if neighbor_idx == ru_idx:
                continue
            
            pos_k = atoms.positions[neighbor_idx]
            
            # Vector with PBC
            vec_jk = pos_k - pos_j
            vec_jk = vec_jk - cell.T @ np.round(np.linalg.solve(cell.T, vec_jk))
            
            projection = np.dot(vec_jk, direction_110_unit)
            
            if projection > 0:
                d_23 = np.linalg.norm(vec_jk)
                break
        
        if d_23 is None:
            continue
        
        # Compute dimer order
        #dimer_order = abs(d_23 - d_12) #no absolute value
        dimer_order = abs(d_23 - d_12)
        dimer_orders.append(dimer_order)
        ru_indices_list.append(ru_idx)
        ru_positions_list.append(pos_i.copy())
    
    return np.array(dimer_orders), np.array(ru_indices_list), np.array(ru_positions_list)


def compute_spatial_correlation_parallel(atoms, dimer_orders, ru_indices_list, ru_neighbor_graph,
                                        direction_110_unit, delta_d=1.0, max_distance=40.0):
    """
    Compute spatial correlation function C(r) ALONG [110] direction.
    This measures how the dimerization pattern persists along [110] chains.
    
    Uses efficient chain-following methodology:
    - For each Ru atom, follow the chain along [110]
    - Compute correlation with atoms along the same chain
    - Distance = projection along [110]
    """
    from scipy.ndimage import gaussian_filter
    
    # Create mapping from atom index to order parameter
    index_to_order = {}
    for idx, ru_idx in enumerate(ru_indices_list):
        index_to_order[ru_idx] = dimer_orders[idx]
    
    # Setup bins
    n_bins = int(max_distance / delta_d + 0.5) + 1
    correlation_sum = np.zeros(n_bins)
    count = np.zeros(n_bins)
    
    cell = atoms.get_cell()
    
    # For each Ru atom, follow the chain along [110]
    for i, ru_idx in enumerate(ru_indices_list):
        pos_i = atoms.positions[ru_idx]
        order_i = dimer_orders[i]
        
        # Self-correlation (r=0)
        correlation_sum[0] += order_i * order_i
        count[0] += 1
        
        # Follow chain forward and backward from this atom
        # Build a list of atoms along the chain using neighbor graph
        chain_atoms = set([ru_idx])
        to_explore = [ru_idx]
        explored = set()
        
        # Breadth-first search along [110] neighbors (within max_distance)
        while to_explore:
            current_idx = to_explore.pop(0)
            if current_idx in explored:
                continue
            explored.add(current_idx)
            
            # Check distance from original atom
            if current_idx != ru_idx:
                pos_current = atoms.positions[current_idx]
                vec = pos_current - pos_i
                vec = vec - cell.T @ np.round(np.linalg.solve(cell.T, vec))
                
                # Project along [110]
                #dist_along_110 = abs(np.dot(vec, direction_110_unit))
                #instead of the projection, use the real distance; this slightly increase the value of correlation length
                dist_along_110 = np.linalg.norm(vec)
                
                if dist_along_110 > max_distance:
                    continue
                
                # This atom is along the chain within max_distance
                chain_atoms.add(current_idx)
                
                # Compute correlation
                if current_idx in index_to_order:
                    bin_idx = int(dist_along_110 / delta_d + 0.5)
                    if bin_idx < n_bins:
                        order_j = index_to_order[current_idx]
                        correlation_sum[bin_idx] += order_i * order_j
                        count[bin_idx] += 1
            
            # Add [110] neighbors to explore
            if current_idx in ru_neighbor_graph:
                for neighbor_idx in ru_neighbor_graph[current_idx]:
                    if neighbor_idx not in explored:
                        to_explore.append(neighbor_idx)
    
    # Average
    mask = count > 0
    correlation = np.zeros(n_bins)
    correlation[mask] = correlation_sum[mask] / count[mask]
    
    # Apply Gaussian smoothing
    #do not apply Gaussian filter to the raw data; this can be done in the plotting script if necessary                  
    #correlation = gaussian_filter(correlation, sigma=1.0)
    
    return correlation


def compute_spatial_correlation(atoms, dimer_orders, ru_indices_list, ru_neighbor_graph, 
                               direction_110_unit, delta_d=1.0, max_distance=20.0):
    """
    Compute spatial correlation function C(r) PERPENDICULAR to [110] direction.
    Optimized to match parallel case speed.
    
    This matches trimerorder.get_space_correlation_all():
    - Includes ALL Ru neighbors within max_distance
    - EXCLUDES neighbors along [110] chains  
    - Computes correlation in perpendicular directions
    """
    from scipy.ndimage import gaussian_filter
    
    # Setup bins
    n_bins = int(max_distance / delta_d + 0.5) + 1
    correlation_sum = np.zeros(n_bins)
    count = np.zeros(n_bins)
    
    cell = atoms.get_cell()
    n_ru = len(ru_indices_list)
    
    # Convert to numpy array if needed
    ru_indices_array = np.array(ru_indices_list)
    
    # Create mapping from atom index to position in array
    index_to_pos = {idx: i for i, idx in enumerate(ru_indices_array)}
    
    # Get all Ru positions
    ru_positions = atoms.positions[ru_indices_array]
    
    # Pre-compute [110] neighbor exclusion pairs (i, j) to skip
    # This is the key optimization!
    exclusion_pairs = set()
    for i in range(n_ru):
        ru_idx = ru_indices_array[i]
        if ru_idx in ru_neighbor_graph:
            for neighbor_idx in ru_neighbor_graph[ru_idx]:
                if neighbor_idx in index_to_pos:
                    j = index_to_pos[neighbor_idx]
                    # Store both directions
                    exclusion_pairs.add((min(i, j), max(i, j)))
    
    # Self-correlation (r=0)
    correlation_sum[0] = np.sum(dimer_orders * dimer_orders)
    count[0] = n_ru
    
    # For each pair (i, j) with j > i
    for i in range(n_ru):
        order_i = dimer_orders[i]
        pos_i = ru_positions[i]
        
        for j in range(i + 1, n_ru):
            # Skip if this pair is along [110]
            if (i, j) in exclusion_pairs:
                continue
            
            pos_j = ru_positions[j]
            
            # Distance with PBC
            vec_ij = pos_j - pos_i
            vec_ij = vec_ij - cell.T @ np.round(np.linalg.solve(cell.T, vec_ij))
            distance = np.linalg.norm(vec_ij)
            
            if distance < 0.1 or distance > max_distance:
                continue
            
            # Bin index
            bin_idx = int(distance / delta_d + 0.5)
            if bin_idx >= n_bins:
                continue
            
            order_j = dimer_orders[j]
            
            # Add to correlation (count both i-j and j-i)
            corr_val = order_i * order_j
            correlation_sum[bin_idx] += 2 * corr_val
            count[bin_idx] += 2
    
    # Average
    mask = count > 0
    correlation = np.zeros(n_bins)
    correlation[mask] = correlation_sum[mask] / count[mask]
    
    # Apply Gaussian smoothing
    correlation = gaussian_filter(correlation, sigma=1.0)
    
    return correlation


def process_temperature(temp_dir, args):
    """Process a single temperature directory."""
    # Find XYZ file
    xyz_dir = temp_dir / args.subdirectory
    if not xyz_dir.exists():
        print(f"  ✗ Directory not found: {xyz_dir}")
        return None
    
    pattern = f"*{args.input_pattern}*.xyz"
    matching_files = list(xyz_dir.glob(pattern))
    
    if len(matching_files) == 0:
        print(f"  ✗ No files matching pattern '{pattern}' found")
        return None
    elif len(matching_files) > 1:
        print(f"  ⚠ Multiple files found, using first: {matching_files[0].name}")
    
    xyz_file = matching_files[0]
    print(f"  Reading: {xyz_file.name}")
    
    # Read frames
    index_str = f"{args.start_frame}:"
    frames = read(str(xyz_file), index=index_str)
    print(f"  Total frames available: {len(frames)}")
    
    # Get neighbor graph from first frame
    print(f"  Building neighbor graph from first frame...")
    first_atoms = frames[0]
    ru_neighbor_graph, direction_110_unit = identify_ru_110_pairs(first_atoms)
    print(f"  Found {len(ru_neighbor_graph)} Ru atoms with neighbors")
    
    # Compute spatial correlation for each frame
    print(f"  Computing spatial correlations...")
    
    all_correlations = []
    
    for step, atoms in enumerate(tqdm(frames, desc="  Processing frames", unit="frame")):
        if step % args.frame_step == 0:
            # Compute order parameters
            dimer_orders, ru_indices, ru_positions = compute_dimer_order(
                atoms, ru_neighbor_graph, direction_110_unit
            )
            
            # Compute spatial correlation (parallel or perpendicular)
            if args.parallel:
                corr = compute_spatial_correlation_parallel(
                    atoms, dimer_orders, ru_indices, ru_neighbor_graph, direction_110_unit,
                    delta_d=args.delta_d, 
                    max_distance=args.max_distance
                )
            else:
                corr = compute_spatial_correlation(
                    atoms, dimer_orders, ru_indices, ru_neighbor_graph, direction_110_unit,
                    delta_d=args.delta_d, 
                    max_distance=args.max_distance
                )
            
            all_correlations.append(corr)
    
    # Average over frames
    avg_correlation = np.mean(all_correlations, axis=0)
    
    print(f"  Processed {len(all_correlations)} frames")
    
    return avg_correlation


def main():
    """Main execution function."""
    args = parse_arguments()
    
    print("="*70)
    print("SPACE CORRELATION COMPUTATION (FAST METHOD)")
    print("="*70)
    print(f"\nParent directory: {args.parent_dir}")
    print(f"File pattern: *{args.input_pattern}*.xyz")
    print(f"Output directory: {args.output_dir}")
    print(f"Bin size (Δd): {args.delta_d} Å")
    print(f"Correlation direction: {'PARALLEL (along [110])' if args.parallel else 'PERPENDICULAR (to [110])'}")
    
    # Define temperatures
    if args.temperatures is not None:
        temperatures = args.temperatures
    else:
        temperatures = [
            "10K", "50K", "100K", "150K", "200K", "250K", "300K", "310K",
            "320K", "330K", "340K", "350K", "360K", "370K", "380K", "390K", "400K",
            "450K", "500K", "550K", "600K", "650K", "700K"
        ]
    
    print(f"\nTemperatures to process: {len(temperatures)}")
    print(f"{'='*70}\n")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each temperature
    results = {}
    
    for temp in temperatures:
        print(f"{'='*70}")
        print(f"Processing: {temp}")
        print(f"{'='*70}")
        
        start_time = time.time()
        
        temp_dir = Path(args.parent_dir) / temp
        if not temp_dir.exists():
            print(f"  ✗ Temperature directory not found: {temp_dir}")
            continue
        
        try:
            corr = process_temperature(temp_dir, args)
            
            if corr is not None:
                # Create distance array
                distances = np.arange(len(corr)) * args.delta_d
                
                # Save result
                output_file = output_dir / f"{args.file_prefix}_{temp}.npz"
                np.savez(
                    output_file,
                    temperature=temp,
                    correlation=corr,
                    distances=distances,
                    delta_d=args.delta_d,
                    correlation_direction='parallel' if args.parallel else 'perpendicular'
                )
                
                elapsed = time.time() - start_time
                print(f"  ✓ Completed in {elapsed:.1f} seconds")
                print(f"  ✓ Saved: {output_file.name}")
                print(f"  Correlation range: [{corr.min():.4f}, {corr.max():.4f}]")
                print(f"  Distance range: [0, {distances[-1]:.1f}] Å")
                
                results[temp] = output_file
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
        
        print()
    
    print(f"{'='*70}")
    print(f"COMPLETED: {len(results)}/{len(temperatures)} temperatures processed")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
