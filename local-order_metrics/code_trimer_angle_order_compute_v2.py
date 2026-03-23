import numpy as np
import pandas as pd
from ase import Atoms
from ase.neighborlist import NeighborList
import re
import os
from tqdm import tqdm
import warnings
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import gc
import psutil
warnings.filterwarnings('ignore')

def get_memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024 / 1024

def parse_single_snapshot(lines, start_idx):
    """Parse a single snapshot from XYZ file lines"""
    i = start_idx
    
    if i >= len(lines) or not lines[i].strip().isdigit():
        return None
    
    n_atoms = int(lines[i].strip())
    i += 1
    
    if i >= len(lines):
        return None
    
    header = lines[i].strip()
    i += 1
    
    # Extract timestep and lattice
    timestep_match = re.search(r'Timestep=(\d+)', header)
    timestep = int(timestep_match.group(1)) if timestep_match else 0
    
    lattice_match = re.search(r'Lattice="([^"]+)"', header)
    if lattice_match:
        lattice_str = lattice_match.group(1)
        lattice_values = np.fromstring(lattice_str, sep=' ', dtype=np.float64)
        cell = lattice_values.reshape(3, 3)
    else:
        cell = np.eye(3, dtype=np.float64) * 50.0
    
    if i + n_atoms > len(lines):
        return None
    
    atomic_lines = lines[i:i + n_atoms]
    
    symbols = []
    positions = np.empty((n_atoms, 3), dtype=np.float64)
    atom_ids = np.empty(n_atoms, dtype=np.int32)
    
    for idx, line in enumerate(atomic_lines):
        parts = line.split()
        if len(parts) >= 5:
            symbols.append(parts[0])
            positions[idx] = [float(parts[1]), float(parts[2]), float(parts[3])]
            atom_ids[idx] = int(parts[4])
    
    return {
        'timestep': timestep,
        'n_atoms': n_atoms,
        'symbols': symbols,
        'positions': positions,
        'cell': cell,
        'atom_ids': atom_ids,
        'next_idx': i + n_atoms
    }

def identify_ru_110_pairs(first_snapshot):
    """
    Identify all Ru-Ru pairs along [110] direction from the first snapshot.
    This is the CORRECT method from the original distance computation code.
    """
    print("Identifying Ru-Ru pairs along [110] direction from first snapshot...")
    
    atoms = Atoms(
        symbols=first_snapshot['symbols'],
        positions=first_snapshot['positions'],
        cell=first_snapshot['cell'],
        pbc=True
    )
    
    # Get Ru indices and IDs
    symbols = np.array(atoms.get_chemical_symbols())
    ru_mask = symbols == 'Ru'
    ru_indices = np.where(ru_mask)[0]
    ru_ids = first_snapshot['atom_ids'][ru_indices]
    
    print(f"Found {len(ru_indices)} Ru atoms")
    
    # Get [110] direction in real space
    cell = atoms.get_cell()
    direction_110_frac = np.array([1.0, 1.0, 0.0])
    direction_110_real = cell.T @ direction_110_frac
    direction_110_unit = direction_110_real / np.linalg.norm(direction_110_real)
    
    print(f"[110] direction vector: {direction_110_unit}")
    
    # Build neighbor list
    cutoff_nearest = 4.0
    nl = NeighborList([cutoff_nearest/2] * len(atoms), self_interaction=False, bothways=False)
    nl.update(atoms)
    
    # For each Ru atom, find its neighbors along [110]
    ru_neighbor_graph = {}
    
    for ru_idx, ru_id in zip(ru_indices, ru_ids):
        indices, offsets = nl.get_neighbors(ru_idx)
        
        # Filter for Ru neighbors only
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
        
        # Calculate displacements and distances
        displacements = neighbor_positions - current_pos
        distances = np.linalg.norm(displacements, axis=1)
        
        # Project displacements onto [110] direction
        projections = np.dot(displacements, direction_110_unit)
        alignments = np.abs(projections) / (distances + 1e-10)
        
        # Filter for neighbors aligned with [110]
        aligned_mask = (alignments > 0.9) & (distances < cutoff_nearest)
        
        if not np.any(aligned_mask):
            continue
        
        aligned_neighbors = ru_neighbors[aligned_mask]
        aligned_distances = distances[aligned_mask]
        aligned_projections = projections[aligned_mask]
        
        # Sort by distance
        sorted_indices = np.argsort(aligned_distances)
        
        # Separate into forward and backward neighbors
        forward_mask = aligned_projections > 0
        backward_mask = aligned_projections < 0
        
        neighbors_for_this_atom = []
        
        # Get nearest forward neighbor
        forward_indices = sorted_indices[forward_mask[sorted_indices]]
        if len(forward_indices) > 0:
            neighbor_idx = aligned_neighbors[forward_indices[0]]
            neighbor_id = first_snapshot['atom_ids'][neighbor_idx]
            neighbors_for_this_atom.append(neighbor_id)
        
        # Get nearest backward neighbor
        backward_indices = sorted_indices[backward_mask[sorted_indices]]
        if len(backward_indices) > 0:
            neighbor_idx = aligned_neighbors[backward_indices[0]]
            neighbor_id = first_snapshot['atom_ids'][neighbor_idx]
            neighbors_for_this_atom.append(neighbor_id)
        
        if neighbors_for_this_atom:
            ru_neighbor_graph[ru_id] = neighbors_for_this_atom
    
    # Build unique pairs from the neighbor graph
    ru_pairs = set()
    for atom_id, neighbor_ids in ru_neighbor_graph.items():
        for neighbor_id in neighbor_ids:
            if atom_id < neighbor_id:
                ru_pairs.add((atom_id, neighbor_id))
            else:
                ru_pairs.add((neighbor_id, atom_id))
    
    ru_pairs = list(ru_pairs)
    
    print(f"Identified {len(ru_pairs)} unique Ru-Ru pairs along [110]")
    
    return ru_pairs, ru_neighbor_graph

def find_angle(vector1, vector2):
    """Find angle between two vectors in degrees"""
    dot = np.dot(vector1, vector2)
    mag1 = np.linalg.norm(vector1)
    mag2 = np.linalg.norm(vector2)
    
    if mag1 < 1e-10 or mag2 < 1e-10:
        return 0.0
    
    theta = np.arccos(np.clip(dot / (mag1 * mag2), -1.0, 1.0)) * 180 / np.pi
    
    # Get deviation from 180 degrees
    theta = min(theta, 180 - theta)
    
    # Determine sign using cross product (z-component)
    zcross = vector1[0] * vector2[1] - vector1[1] * vector2[0]
    if zcross < 0:
        theta = -theta
    
    return theta

def compute_trimer_angle_for_snapshot(snapshot, ru_pairs, ru_neighbor_graph):
    """
    Compute trimer angle order by following chains along [110] direction.
    For each Ru atom, follow forward along [110]: i → j → k
    Compute angle between vectors (i→j) and (j→k)
    """
    atoms = Atoms(
        symbols=snapshot['symbols'],
        positions=snapshot['positions'],
        cell=snapshot['cell'],
        pbc=True
    )
    
    cell = atoms.get_cell()
    atom_id_to_idx = {aid: idx for idx, aid in enumerate(snapshot['atom_ids'])}
    
    # Get [110] direction
    direction_110_frac = np.array([1.0, 1.0, 0.0])
    direction_110_real = cell.T @ direction_110_frac
    direction_110_unit = direction_110_real / np.linalg.norm(direction_110_real)
    
    trimer_angles = []
    ru_atom_ids = []
    
    # For each Ru atom that has neighbors in the graph
    for ru_id, neighbor_ids in ru_neighbor_graph.items():
        if ru_id not in atom_id_to_idx:
            continue
        
        if len(neighbor_ids) < 1:
            continue
        
        idx_i = atom_id_to_idx[ru_id]
        pos_i = atoms.positions[idx_i]
        
        # Find forward neighbor j
        j_id = None
        vec_ij = None
        
        for neighbor_id in neighbor_ids:
            if neighbor_id not in atom_id_to_idx:
                continue
            
            idx_j = atom_id_to_idx[neighbor_id]
            pos_j = atoms.positions[idx_j]
            
            vec = pos_j - pos_i
            vec = vec - cell.T @ np.round(np.linalg.solve(cell.T, vec))
            
            projection = np.dot(vec, direction_110_unit)
            
            if projection > 0:
                j_id = neighbor_id
                vec_ij = vec
                break
        
        if j_id is None:
            continue
        
        # Find forward neighbor k of j
        if j_id not in ru_neighbor_graph:
            continue
        
        j_neighbors = ru_neighbor_graph[j_id]
        k_id = None
        vec_jk = None
        
        idx_j = atom_id_to_idx[j_id]
        pos_j = atoms.positions[idx_j]
        
        for neighbor_id in j_neighbors:
            if neighbor_id == ru_id:
                continue
            
            if neighbor_id not in atom_id_to_idx:
                continue
            
            idx_k = atom_id_to_idx[neighbor_id]
            pos_k = atoms.positions[idx_k]
            
            vec = pos_k - pos_j
            vec = vec - cell.T @ np.round(np.linalg.solve(cell.T, vec))
            
            projection = np.dot(vec, direction_110_unit)
            
            if projection > 0:
                k_id = neighbor_id
                vec_jk = vec
                break
        
        if k_id is None or vec_jk is None:
            continue
        
        # Calculate angle between vec_ij and vec_jk
        angle = find_angle(vec_ij, vec_jk)
        
        trimer_angles.append(angle)
        ru_atom_ids.append(ru_id)
    
    return trimer_angles, ru_atom_ids

def identify_ru_atoms_and_direction(first_snapshot):
    """Identify Ru-Ru pairs along [110] using the correct original method"""
    return identify_ru_110_pairs(first_snapshot)

def assign_populations_from_reference(ref_csv_file, pop1_range=None, pop2_range=None, pop3_range=None, output_dir='.'):
    """
    Assign Ru atoms to 3 populations based on their trimer angle values at reference temperature.
    Uses histogram peaks and valleys to automatically detect population boundaries.
    """
    print(f"Assigning populations from reference file: {ref_csv_file}")
    
    df = pd.read_csv(ref_csv_file)
    
    # Get ALL angle measurements (not per-atom averages) for finding peaks
    all_angles = df['trimer_angle'].values
    
    # Also get per-atom averages for assignment
    atom_angles = df.groupby('ru_atom_id')['trimer_angle'].mean()
    
    print(f"Analyzing {len(all_angles)} measurements to identify 3 populations...")
    
    # Auto-detect ranges if not provided
    if pop1_range is None or pop2_range is None or pop3_range is None:
        from scipy.signal import find_peaks
        from scipy.ndimage import gaussian_filter1d
        import matplotlib.pyplot as plt
        
        min_angle = all_angles.min()
        max_angle = all_angles.max()
        print(f"Trimer angle range: {min_angle:.3f} to {max_angle:.3f}")
        
        # Create high-resolution histogram using ALL measurements
        n_bins = 200
        hist, bin_edges = np.histogram(all_angles, bins=n_bins, density=False)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = bin_centers[1] - bin_centers[0]
        
        # Smooth histogram
        hist_smooth = gaussian_filter1d(hist.astype(float), sigma=2)
        
        # Find ALL peaks first (no distance constraint)
        peaks, peak_properties = find_peaks(hist_smooth, prominence=2, height=5)
        
        print(f"Found {len(peaks)} potential peaks")
        
        if len(peaks) >= 3:
            # Get peak positions and heights
            peak_positions = bin_centers[peaks]
            peak_heights = hist_smooth[peaks]
            
            # Sort peaks by height (prominence) and take top 3
            sorted_by_height = np.argsort(peak_heights)[-3:]
            top_3_peaks = peaks[sorted_by_height]
            
            # Sort these 3 peaks by position (left to right)
            top_3_sorted = sorted(top_3_peaks, key=lambda i: bin_centers[i])
            
            peak1_idx, peak2_idx, peak3_idx = top_3_sorted
            peak1_pos = bin_centers[peak1_idx]
            peak2_pos = bin_centers[peak2_idx]
            peak3_pos = bin_centers[peak3_idx]
            
            print(f"\nSelected 3 highest peaks at: {peak1_pos:.2f}°, {peak2_pos:.2f}°, {peak3_pos:.2f}°")
            print(f"Peak indices in histogram: {peak1_idx}, {peak2_idx}, {peak3_idx}")
            
            # Find valleys (minima) between peaks
            valley1_idx = peak1_idx + np.argmin(hist_smooth[peak1_idx:peak2_idx])
            valley1 = bin_centers[valley1_idx]
            
            valley2_idx = peak2_idx + np.argmin(hist_smooth[peak2_idx:peak3_idx])
            valley2 = bin_centers[valley2_idx]
            
            print(f"Valley indices: {valley1_idx}, {valley2_idx}")
            print(f"Population boundaries at: {valley1:.2f}°, {valley2:.2f}°")
            
            # Define population ranges
            pop1_range = (min_angle - 0.5, valley1)
            pop2_range = (valley1, valley2)
            pop3_range = (valley2, max_angle + 0.5)
            
        else:
            print(f"Warning: Only found {len(peaks)} peaks. Using fallback.")
            # Fallback to equal division
            range_width = (max_angle - min_angle) / 3
            pop1_range = (min_angle, min_angle + range_width)
            pop2_range = (min_angle + range_width, min_angle + 2 * range_width)
            pop3_range = (min_angle + 2 * range_width, max_angle)
            peak1_pos, peak2_pos, peak3_pos = None, None, None
            valley1, valley2 = pop1_range[1], pop2_range[1]
        
        # Create diagnostic plot showing ALL measurements
        print("Creating diagnostic histogram plot...")
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Plot histogram
        ax.bar(bin_centers, hist, width=bin_width*0.8, alpha=0.5, color='gray', edgecolor='black', label='Histogram (all measurements)')
        ax.plot(bin_centers, hist_smooth, 'b-', linewidth=2.5, label='Smoothed histogram')
        
        # Mark all detected peaks
        if len(peaks) > 0:
            ax.plot(bin_centers[peaks], hist_smooth[peaks], 'go', markersize=10, label=f'{len(peaks)} detected peaks', zorder=5)
        
        # Mark the selected 3 peaks
        if peak1_pos is not None:
            y_max = hist_smooth.max()
            ax.axvline(peak1_pos, color='red', linestyle='--', linewidth=3, label=f'Pop 1 peak: {peak1_pos:.2f}°', zorder=4)
            ax.axvline(peak2_pos, color='red', linestyle='--', linewidth=3, label=f'Pop 2 peak: {peak2_pos:.2f}°', zorder=4)
            ax.axvline(peak3_pos, color='red', linestyle='--', linewidth=3, label=f'Pop 3 peak: {peak3_pos:.2f}°', zorder=4)
        
        # Mark the boundaries
        ax.axvline(valley1, color='blue', linestyle=':', linewidth=2.5, label=f'Boundary 1: {valley1:.2f}°', zorder=3)
        ax.axvline(valley2, color='blue', linestyle=':', linewidth=2.5, label=f'Boundary 2: {valley2:.2f}°', zorder=3)
        
        ax.set_xlabel('Trimer Angle (degrees)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=14, fontweight='bold')
        ax.set_title(f'Trimer Angle Distribution at 10K ({len(all_angles):,} measurements)', fontsize=16, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        
        diag_file = os.path.join(output_dir, 'trimer_angle_population_diagnostic.png')
        plt.tight_layout()
        plt.savefig(diag_file, dpi=150)
        plt.close()
        print(f"Saved diagnostic plot to {diag_file}")
    
    print(f"Auto-detected ranges:")
    print(f"  Population 1: {pop1_range[0]:.3f} to {pop1_range[1]:.3f}")
    print(f"  Population 2: {pop2_range[0]:.3f} to {pop2_range[1]:.3f}")
    print(f"  Population 3: {pop3_range[0]:.3f} to {pop3_range[1]:.3f}")
    
    # Assign populations
    population_assignment = {}
    
    for ru_id, angle in atom_angles.items():
        if pop1_range[0] <= angle <= pop1_range[1]:
            population_assignment[ru_id] = 1
        elif pop2_range[0] <= angle <= pop2_range[1]:
            population_assignment[ru_id] = 2
        elif pop3_range[0] <= angle <= pop3_range[1]:
            population_assignment[ru_id] = 3
        else:
            # Default to closest population
            if angle < pop1_range[0]:
                population_assignment[ru_id] = 1
            elif angle > pop3_range[1]:
                population_assignment[ru_id] = 3
            else:
                # Between pop1 and pop2, or between pop2 and pop3
                if angle < pop2_range[0]:
                    population_assignment[ru_id] = 1
                else:
                    population_assignment[ru_id] = 3
    
    # Compute statistics for each population
    pop1_atoms = [ru_id for ru_id, pop in population_assignment.items() if pop == 1]
    pop2_atoms = [ru_id for ru_id, pop in population_assignment.items() if pop == 2]
    pop3_atoms = [ru_id for ru_id, pop in population_assignment.items() if pop == 3]
    
    pop1_angles = atom_angles[pop1_atoms]
    pop2_angles = atom_angles[pop2_atoms]
    pop3_angles = atom_angles[pop3_atoms]
    
    print("=== Population Assignment Results ===")
    print(f"Total Ru atoms: {len(atom_angles)}")
    print(f"Population 1: {len(pop1_atoms)} atoms ({100*len(pop1_atoms)/len(atom_angles):.1f}%)")
    print(f"  Mean: {pop1_angles.mean():.3f}")
    print(f"  Std:  {pop1_angles.std():.3f}")
    print(f"Population 2: {len(pop2_atoms)} atoms ({100*len(pop2_atoms)/len(atom_angles):.1f}%)")
    print(f"  Mean: {pop2_angles.mean():.3f}")
    print(f"  Std:  {pop2_angles.std():.3f}")
    print(f"Population 3: {len(pop3_atoms)} atoms ({100*len(pop3_atoms)/len(atom_angles):.1f}%)")
    print(f"  Mean: {pop3_angles.mean():.3f}")
    print(f"  Std:  {pop3_angles.std():.3f}")
    
    # Sanity check
    print("=== Sanity Check ===")
    print(f"Pop1 + Pop2 + Pop3 = {len(pop1_atoms) + len(pop2_atoms) + len(pop3_atoms)} / {len(atom_angles)}")
    print(f"Assigned fraction: {100*(len(pop1_atoms) + len(pop2_atoms) + len(pop3_atoms))/len(atom_angles):.1f}%")
    
    # Save population assignment
    pop_df = pd.DataFrame([
        {'ru_atom_id': ru_id, 'population': pop}
        for ru_id, pop in population_assignment.items()
    ])
    pop_file = os.path.join(output_dir, 'trimer_angle_population_assignment.csv')
    pop_df.to_csv(pop_file, index=False)
    print(f"Saved population assignment to {pop_file}")
    
    ref_stats = {
        'pop1_mean': pop1_angles.mean(),
        'pop1_std': pop1_angles.std(),
        'pop1_count': len(pop1_atoms),
        'pop2_mean': pop2_angles.mean(),
        'pop2_std': pop2_angles.std(),
        'pop2_count': len(pop2_atoms),
        'pop3_mean': pop3_angles.mean(),
        'pop3_std': pop3_angles.std(),
        'pop3_count': len(pop3_atoms)
    }
    
    return population_assignment, ref_stats

def compute_population_statistics_for_temperature(csv_file, population_assignment):
    """
    Compute statistics for each population at a given temperature.
    Uses mode (most frequent value) instead of mean to track peak positions.
    """
    from scipy.stats import gaussian_kde
    
    df = pd.read_csv(csv_file)
    
    # Add population column based on ru_atom_id
    df['population'] = df['ru_atom_id'].map(population_assignment)
    
    # Filter out atoms not in any population
    df = df[df['population'].notna()]
    
    # Compute statistics for each population
    stats = {}
    
    for pop in [1, 2, 3]:
        pop_data = df[df['population'] == pop]['trimer_angle'].values
        
        if len(pop_data) > 100:
            # Use KDE to find mode (peak of distribution)
            kde = gaussian_kde(pop_data, bw_method='scott')
            
            # Evaluate KDE on a grid to find maximum
            pop_min, pop_max = pop_data.min(), pop_data.max()
            x_grid = np.linspace(pop_min, pop_max, 500)
            kde_values = kde(x_grid)
            
            # Mode is the x value where KDE is maximum
            mode_idx = np.argmax(kde_values)
            pop_mode = x_grid[mode_idx]
        else:
            # Fallback to median for small populations
            pop_mode = np.median(pop_data)
        
        stats[f'pop{pop}_mean'] = pop_mode  # Store mode as "mean" for compatibility
        stats[f'pop{pop}_std'] = pop_data.std()
        stats[f'pop{pop}_count'] = len(pop_data)
    
    stats['total_measurements'] = len(df)
    
    return stats

def process_trajectory_chunk(chunk_data):
    """Process a chunk of the trajectory file"""
    trajectory_path, start_pos, end_pos, ru_pairs, ru_neighbor_graph = chunk_data
    
    # Read chunk
    with open(trajectory_path, 'rb') as f:
        f.seek(start_pos)
        chunk_bytes = f.read(end_pos - start_pos)
        chunk_text = chunk_bytes.decode('utf-8', errors='ignore')
    
    lines = chunk_text.split('\n')
    
    # Parse snapshots
    results = []
    i = 0
    while i < len(lines):
        snapshot = parse_single_snapshot(lines, i)
        if snapshot is None:
            i += 1
            continue
        
        # Compute trimer angle for this snapshot
        trimer_angles, ru_atom_ids = compute_trimer_angle_for_snapshot(snapshot, ru_pairs, ru_neighbor_graph)
        
        results.append({
            'timestep': snapshot['timestep'],
            'trimer_angles': trimer_angles,
            'ru_atom_ids': ru_atom_ids
        })
        
        i = snapshot['next_idx']
    
    return results

def find_snapshot_positions(trajectory_path):
    """Find byte positions of each snapshot in the file"""
    positions = []
    with open(trajectory_path, 'r') as f:
        pos = 0
        while True:
            line = f.readline()
            if not line:
                break
            
            if line.strip().isdigit():
                positions.append(pos)
                n_atoms = int(line.strip())
                f.readline()
                for _ in range(n_atoms):
                    f.readline()
            
            pos = f.tell()
    
    return positions

def compute_trimer_angle_streaming(trajectory_path, n_processes=8):
    """
    Main function to compute trimer angle order.
    Uses streaming to handle large files efficiently.
    """
    print(f"Processing trajectory: {trajectory_path}")
    print(f"Using {n_processes} processes")
    
    # Step 1: Read first snapshot
    print("\nStep 1: Reading first snapshot...")
    with open(trajectory_path, 'r') as f:
        lines = []
        n_atoms_line = f.readline()
        lines.append(n_atoms_line)
        n_atoms = int(n_atoms_line.strip())
        
        for _ in range(n_atoms + 1):
            lines.append(f.readline())
    
    first_snapshot = parse_single_snapshot(lines, 0)
    
    # Step 2: Identify Ru-Ru pairs along [110]
    print("\nStep 2: Identifying Ru-Ru pairs along [110]...")
    ru_pairs, ru_neighbor_graph = identify_ru_110_pairs(first_snapshot)
    ru_pairs_set = set(ru_pairs)
    
    # Step 3: Find all snapshot positions
    print("\nStep 3: Finding snapshot positions...")
    snapshot_positions = find_snapshot_positions(trajectory_path)
    print(f"Found {len(snapshot_positions)} snapshots")
    
    # Step 4: Create chunks for parallel processing
    print("\nStep 4: Processing trajectory in parallel...")
    file_size = os.path.getsize(trajectory_path)
    
    n_chunks = min(n_processes * 4, len(snapshot_positions))
    chunk_size = len(snapshot_positions) // n_chunks
    
    chunks = []
    for i in range(n_chunks):
        start_idx = i * chunk_size
        if i == n_chunks - 1:
            end_idx = len(snapshot_positions)
        else:
            end_idx = (i + 1) * chunk_size
        
        start_pos = snapshot_positions[start_idx]
        end_pos = file_size if end_idx >= len(snapshot_positions) else snapshot_positions[end_idx]
        
        chunks.append((trajectory_path, start_pos, end_pos, ru_pairs_set, ru_neighbor_graph))
    
    # Process chunks in parallel
    all_results = []
    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        futures = {executor.submit(process_trajectory_chunk, chunk): i 
                  for i, chunk in enumerate(chunks)}
        
        with tqdm(total=len(chunks), desc="Processing chunks") as pbar:
            for future in as_completed(futures):
                chunk_results = future.result()
                all_results.extend(chunk_results)
                pbar.update(1)
    
    # Sort by timestep
    all_results.sort(key=lambda x: x['timestep'])
    
    return all_results

def save_trimer_angle_to_csv(results, temperature, output_dir='.'):
    """Save computed trimer angles to CSV file"""
    data_rows = []
    
    for result in results:
        timestep = result['timestep']
        time_ps = timestep * 50 / 1000
        trimer_angles = result['trimer_angles']
        ru_atom_ids = result['ru_atom_ids']
        
        for trimer_angle, ru_id in zip(trimer_angles, ru_atom_ids):
            data_rows.append({
                'timestep': timestep,
                'time_ps': time_ps,
                'trimer_angle': trimer_angle,
                'ru_atom_id': ru_id,
                'temperature_K': temperature
            })
    
    df = pd.DataFrame(data_rows)
    
    csv_filename = os.path.join(output_dir, f'trimer_angle_{temperature}K.csv')
    df.to_csv(csv_filename, index=False)
    print(f"\nSaved {len(df)} trimer angle measurements to {csv_filename}")
    
    if len(df) > 0:
        print(f"\nStatistics:")
        print(f"  Mean trimer angle: {df['trimer_angle'].mean():.2f}°")
        print(f"  Std deviation: {df['trimer_angle'].std():.2f}°")
        print(f"  Min: {df['trimer_angle'].min():.2f}°")
        print(f"  Max: {df['trimer_angle'].max():.2f}°")
        print(f"  Unique Ru atoms: {df['ru_atom_id'].nunique()}")
    
    return csv_filename

def analyze_single_temperature(trajectory_path, temperature, n_processes=8):
    """Analyze trimer angle order for a single temperature"""
    print(f"\n{'='*60}")
    print(f"Analyzing {temperature}K")
    print(f"{'='*60}")
    
    start_time = time.time()
    print(f"Initial memory usage: {get_memory_usage():.2f} GB")
    
    results = compute_trimer_angle_streaming(trajectory_path, n_processes)
    csv_filename = save_trimer_angle_to_csv(results, temperature)
    
    total_time = time.time() - start_time
    print(f"\nTotal computation time: {total_time:.2f} seconds")
    print(f"Final memory usage: {get_memory_usage():.2f} GB")
    
    return csv_filename

def analyze_statistics_across_temperatures(ref_temperature=50, pop1_range=None, pop2_range=None, pop3_range=None, output_dir='.'):
    """
    Analyze 3 populations across all temperatures based on reference temperature assignment.
    """
    print("\n" + "="*60)
    print("Population Analysis Across Temperatures")
    print("="*60)
    
    # Get reference CSV file
    ref_csv_file = os.path.join(output_dir, f'trimer_angle_{ref_temperature}K.csv')
    
    # Assign populations from reference
    population_assignment, ref_stats = assign_populations_from_reference(
        ref_csv_file, pop1_range, pop2_range, pop3_range, output_dir
    )
    
    # Find all temperature CSV files
    import glob
    csv_files = glob.glob(os.path.join(output_dir, 'trimer_angle_*K.csv'))
    temperatures = []
    for f in csv_files:
        match = re.search(r'(\d+)K\.csv$', f)
        if match:
            temperatures.append(int(match.group(1)))
    
    temperatures = sorted(temperatures)
    print(f"\nFound {len(temperatures)} temperature files: {temperatures}")
    
    # Compute statistics for each temperature
    all_stats = []
    for temp in temperatures:
        csv_file = os.path.join(output_dir, f'trimer_angle_{temp}K.csv')
        
        if not os.path.exists(csv_file):
            print(f"Warning: Missing file for {temp}K")
            continue
        
        # Check if file is empty
        if os.path.getsize(csv_file) < 100:
            print(f"Warning: Empty file for {temp}K, skipping")
            continue
        
        stats = compute_population_statistics_for_temperature(csv_file, population_assignment)
        all_stats.append({
            'temperature': temp,
            **stats
        })
        
        print(f"{temp}K:")
        print(f"  Pop 1: {stats['pop1_mean']:.4f} ± {stats['pop1_std']:.4f}  ({stats['pop1_count']} measurements)")
        print(f"  Pop 2: {stats['pop2_mean']:.4f} ± {stats['pop2_std']:.4f}  ({stats['pop2_count']} measurements)")
        print(f"  Pop 3: {stats['pop3_mean']:.4f} ± {stats['pop3_std']:.4f}  ({stats['pop3_count']} measurements)")
    
    # Save all statistics
    stats_df = pd.DataFrame(all_stats)
    stats_file = os.path.join(output_dir, 'trimer_angle_population_statistics.csv')
    stats_df.to_csv(stats_file, index=False)
    print("="*60)
    print(f"Saved population statistics to {stats_file}")
    print("="*60)

def main():
    """Main function to compute trimer angle order"""
    temperature_paths = {
        # Uncomment to add more temperatures:
        50: "../../50K/lammps_out/rup_traj_sampled-10_100ps.xyz",
        100: "../../100K/lammps_out/rup_traj_sampled-10_100ps.xyz",
        110: "../../110K/lammps_out/rup_traj_sampled-10_100ps.xyz",
        120: "../../120K/lammps_out/rup_traj_sampled-10_100ps.xyz",
        130: "../../130K/lammps_out/rup_traj_sampled-10_100ps.xyz",
        140: "../../140K/lammps_out/rup_traj_sampled-10_100ps.xyz",               
        150: "../../150K/lammps_out/rup_traj_sampled-10_100ps.xyz",
        160: "../../160K/lammps_out/rup_traj_sampled-10_100ps.xyz",
        170: "../../170K/lammps_out/rup_traj_sampled-10_100ps.xyz",
        180: "../../180K/lammps_out/rup_traj_sampled-10_100ps.xyz",
        190: "../../190K/lammps_out/rup_traj_sampled-10_100ps.xyz",                
        200: "../../200K/lammps_out/rup_traj_sampled-10_100ps.xyz",
        250: "../../250K/lammps_out/rup_traj_sampled-10_100ps.xyz",
        260: "../../260K/lammps_out/rup_traj_sampled-10_100ps.xyz",
        270: "../../270K/lammps_out/rup_traj_sampled-10_100ps.xyz",
        280: "../../280K/lammps_out/rup_traj_sampled-10_100ps.xyz",
        290: "../../290K/lammps_out/rup_traj_sampled-10_100ps.xyz",                
        300: "../../300K/lammps_out/rup_traj_sampled-10_100ps.xyz",
        310: "../../310K/lammps_out/rup_traj_sampled-10_100ps.xyz",
        320: "../../320K/lammps_out/rup_traj_sampled-10_100ps.xyz",
        330: "../../330K/lammps_out/rup_traj_sampled-10_100ps.xyz",
        340: "../../340K/lammps_out/rup_traj_sampled-10_100ps.xyz",
        350: "../../350K/lammps_out/rup_traj_sampled-10_100ps.xyz",
        400: "../../400K/lammps_out/rup_traj_sampled-10_100ps.xyz",
        450: "../../450K/lammps_out/rup_traj_sampled-10_100ps.xyz",
        500: "../../500K/lammps_out/rup_traj_sampled-10_100ps.xyz",
        550: "../../550K/lammps_out/rup_traj_sampled-10_100ps.xyz",
        600: "../../600K/lammps_out/rup_traj_sampled-10_100ps.xyz",
        650: "../../650K/lammps_out/rup_traj_sampled-10_100ps.xyz",
        700: "../../700K/lammps_out/rup_traj_sampled-10_100ps.xyz",
        # Add your temperature paths here
    }
    
    n_processes = 8
    
    print("Starting trimer angle order computation...")
    print(f"System info: {mp.cpu_count()} CPUs available, using {n_processes} processes")
    
    csv_files = []
    total_start = time.time()
    
    # Step 1: Compute trimer angle for all temperatures
    for temperature, path in temperature_paths.items():
        if os.path.exists(path):
            csv_file = analyze_single_temperature(path, temperature, n_processes)
            if csv_file:
                csv_files.append(csv_file)
        else:
            print(f"Warning: File not found for {temperature}K: {path}")
    
    total_time = time.time() - total_start
    
    print(f"\n{'='*60}")
    print(f"Completed trimer angle computation in {total_time:.1f} seconds!")
    print(f"Generated {len(csv_files)} CSV files")
    
    # Step 2: Compute statistics
    print("\n" + "="*60)
    print("Step 2: Computing statistics...")
    print("="*60)
    
    analyze_statistics_across_temperatures(output_dir='.')
    
    print("\n" + "="*60)
    print("All processing complete!")
    print("="*60)

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
