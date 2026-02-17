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

def compute_dimer_order_for_snapshot(snapshot, ru_pairs, ru_neighbor_graph):
    """
    Compute dimer bond order by following chains along [110] direction.
    For each Ru atom, follow forward along [110]: i → j → k
    Dimer order = d_23 - d_12 where d_12 = distance(i,j) and d_23 = distance(j,k)
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
    
    dimer_orders = []
    ru_atom_ids = []
    
    # For each Ru atom that has neighbors in the graph
    for ru_id, neighbor_ids in ru_neighbor_graph.items():
        if ru_id not in atom_id_to_idx:
            continue
        
        if len(neighbor_ids) < 1:  # Need at least one neighbor
            continue
        
        idx_i = atom_id_to_idx[ru_id]
        pos_i = atoms.positions[idx_i]
        
        # Find forward neighbor (positive projection along [110])
        forward_neighbor = None
        d_12 = None
        
        for neighbor_id in neighbor_ids:
            if neighbor_id not in atom_id_to_idx:
                continue
            
            idx_j = atom_id_to_idx[neighbor_id]
            pos_j = atoms.positions[idx_j]
            
            # Vector from i to j with PBC
            vec_ij = pos_j - pos_i
            vec_ij = vec_ij - cell.T @ np.round(np.linalg.solve(cell.T, vec_ij))
            
            # Check if this is in forward direction
            projection = np.dot(vec_ij, direction_110_unit)
            
            if projection > 0:  # Forward direction
                forward_neighbor = neighbor_id
                d_12 = np.linalg.norm(vec_ij)
                break
        
        if forward_neighbor is None:
            continue
        
        # Now find the forward neighbor of j (which is k)
        if forward_neighbor not in ru_neighbor_graph:
            continue
        
        j_neighbors = ru_neighbor_graph[forward_neighbor]
        
        if len(j_neighbors) < 1:
            continue
        
        idx_j = atom_id_to_idx[forward_neighbor]
        pos_j = atoms.positions[idx_j]
        
        # Find forward neighbor of j (atom k)
        forward_neighbor_of_j = None
        d_23 = None
        
        for neighbor_id in j_neighbors:
            if neighbor_id == ru_id:  # Skip if it's the original atom i
                continue
            
            if neighbor_id not in atom_id_to_idx:
                continue
            
            idx_k = atom_id_to_idx[neighbor_id]
            pos_k = atoms.positions[idx_k]
            
            # Vector from j to k with PBC
            vec_jk = pos_k - pos_j
            vec_jk = vec_jk - cell.T @ np.round(np.linalg.solve(cell.T, vec_jk))
            
            # Check if this is in forward direction
            projection = np.dot(vec_jk, direction_110_unit)
            
            if projection > 0:  # Forward direction
                forward_neighbor_of_j = neighbor_id
                d_23 = np.linalg.norm(vec_jk)
                break
        
        if forward_neighbor_of_j is None or d_23 is None:
            continue
        
        # Compute dimer order
        dimer_order = d_23 - d_12
        dimer_orders.append(dimer_order)
        ru_atom_ids.append(ru_id)
    
    return dimer_orders, ru_atom_ids

def identify_ru_atoms_and_direction(first_snapshot):
    """Identify Ru-Ru pairs along [110] using the correct original method"""
    return identify_ru_110_pairs(first_snapshot)

def assign_populations_from_reference(ref_csv_file, 
                                     pop1_range=None, 
                                     pop2_range=None,
                                     pop3_range=None):
    """
    Assign Ru atoms to populations based on their dimer order in the reference temperature.
    For dimer order, we expect 3 distinct populations at low temperature.
    """
    print(f"\nAssigning populations from reference file: {ref_csv_file}")
    
    # Load reference data
    df = pd.read_csv(ref_csv_file)
    first_timestep = df['timestep'].min()
    first_frame = df[df['timestep'] == first_timestep]
    
    dimer_orders_first_frame = first_frame['dimer_order'].values
    ru_atom_ids = first_frame['ru_atom_id'].values
    
    # If ranges not provided, determine automatically from histogram
    if pop1_range is None or pop2_range is None or pop3_range is None:
        print("\nAnalyzing dimer order distribution to identify 3 populations...")
        hist, bins = np.histogram(dimer_orders_first_frame, bins=100)
        print(f"Dimer order range: {dimer_orders_first_frame.min():.3f} to {dimer_orders_first_frame.max():.3f}")
        
        # Simple approach: divide into three equal ranges (can be refined)
        min_val = dimer_orders_first_frame.min()
        max_val = dimer_orders_first_frame.max()
        range_size = (max_val - min_val) / 3
        
        pop1_range = (min_val, min_val + range_size)
        pop2_range = (min_val + range_size, min_val + 2*range_size)
        pop3_range = (min_val + 2*range_size, max_val)
        
        print(f"Auto-detected ranges:")
        print(f"  Population 1: {pop1_range[0]:.3f} to {pop1_range[1]:.3f}")
        print(f"  Population 2: {pop2_range[0]:.3f} to {pop2_range[1]:.3f}")
        print(f"  Population 3: {pop3_range[0]:.3f} to {pop3_range[1]:.3f}")
    else:
        print(f"Using provided ranges:")
        print(f"  Population 1: {pop1_range[0]:.3f} to {pop1_range[1]:.3f}")
        print(f"  Population 2: {pop2_range[0]:.3f} to {pop2_range[1]:.3f}")
        print(f"  Population 3: {pop3_range[0]:.3f} to {pop3_range[1]:.3f}")
    
    # Assign populations
    population_assignment = {}
    pop1_indices = []
    pop2_indices = []
    pop3_indices = []
    
    for i, (ru_id, dimer_val) in enumerate(zip(ru_atom_ids, dimer_orders_first_frame)):
        if pop1_range[0] <= dimer_val <= pop1_range[1]:
            population_assignment[ru_id] = 1
            pop1_indices.append(i)
        elif pop2_range[0] <= dimer_val <= pop2_range[1]:
            population_assignment[ru_id] = 2
            pop2_indices.append(i)
        elif pop3_range[0] <= dimer_val <= pop3_range[1]:
            population_assignment[ru_id] = 3
            pop3_indices.append(i)
    
    # Calculate statistics
    pop1_values = dimer_orders_first_frame[pop1_indices]
    pop2_values = dimer_orders_first_frame[pop2_indices]
    pop3_values = dimer_orders_first_frame[pop3_indices]
    
    stats = {
        'total_atoms': len(ru_atom_ids),
        'pop1_count': len(pop1_indices),
        'pop2_count': len(pop2_indices),
        'pop3_count': len(pop3_indices),
        'pop1_mean': pop1_values.mean() if len(pop1_values) > 0 else 0,
        'pop1_std': pop1_values.std() if len(pop1_values) > 0 else 0,
        'pop2_mean': pop2_values.mean() if len(pop2_values) > 0 else 0,
        'pop2_std': pop2_values.std() if len(pop2_values) > 0 else 0,
        'pop3_mean': pop3_values.mean() if len(pop3_values) > 0 else 0,
        'pop3_std': pop3_values.std() if len(pop3_values) > 0 else 0,
    }
    
    print(f"\n=== Population Assignment Results ===")
    print(f"Total Ru atoms: {stats['total_atoms']}")
    print(f"\nPopulation 1: {stats['pop1_count']} atoms ({100*stats['pop1_count']/stats['total_atoms']:.1f}%)")
    print(f"  Mean: {stats['pop1_mean']:.3f}")
    print(f"  Std:  {stats['pop1_std']:.3f}")
    print(f"\nPopulation 2: {stats['pop2_count']} atoms ({100*stats['pop2_count']/stats['total_atoms']:.1f}%)")
    print(f"  Mean: {stats['pop2_mean']:.3f}")
    print(f"  Std:  {stats['pop2_std']:.3f}")
    print(f"\nPopulation 3: {stats['pop3_count']} atoms ({100*stats['pop3_count']/stats['total_atoms']:.1f}%)")
    print(f"  Mean: {stats['pop3_mean']:.3f}")
    print(f"  Std:  {stats['pop3_std']:.3f}")
    
    assigned_total = stats['pop1_count'] + stats['pop2_count'] + stats['pop3_count']
    print(f"\n=== Sanity Check ===")
    print(f"Pop1 + Pop2 + Pop3 = {assigned_total} / {stats['total_atoms']}")
    print(f"Assigned fraction: {100*assigned_total/stats['total_atoms']:.1f}%")
    
    return population_assignment, stats

def compute_population_statistics_for_temperature(csv_file, population_assignment):
    """Compute statistics for each population at a given temperature"""
    df = pd.read_csv(csv_file)
    temperature = df['temperature_K'].iloc[0]
    
    # Assign population to each measurement
    df['population'] = df['ru_atom_id'].map(population_assignment).fillna(0).astype(int)
    
    # Calculate statistics
    pop1_data = df[df['population'] == 1]['dimer_order']
    pop2_data = df[df['population'] == 2]['dimer_order']
    pop3_data = df[df['population'] == 3]['dimer_order']
    
    stats = {
        'temperature': temperature,
        'total_measurements': len(df),
        'pop1_count': len(pop1_data),
        'pop1_mean': pop1_data.mean() if len(pop1_data) > 0 else np.nan,
        'pop1_std': pop1_data.std() if len(pop1_data) > 0 else np.nan,
        'pop2_count': len(pop2_data),
        'pop2_mean': pop2_data.mean() if len(pop2_data) > 0 else np.nan,
        'pop2_std': pop2_data.std() if len(pop2_data) > 0 else np.nan,
        'pop3_count': len(pop3_data),
        'pop3_mean': pop3_data.mean() if len(pop3_data) > 0 else np.nan,
        'pop3_std': pop3_data.std() if len(pop3_data) > 0 else np.nan,
    }
    
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
        
        # Compute dimer order for this snapshot
        dimer_orders, ru_atom_ids = compute_dimer_order_for_snapshot(snapshot, ru_pairs, ru_neighbor_graph)
        
        results.append({
            'timestep': snapshot['timestep'],
            'dimer_orders': dimer_orders,
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

def compute_dimer_order_streaming(trajectory_path, n_processes=8):
    """
    Main function to compute dimer bond order.
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
    ru_pairs_set = set(ru_pairs)  # Convert to set for O(1) lookup
    
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

def save_dimer_order_to_csv(results, temperature, output_dir='.'):
    """Save computed dimer orders to CSV file"""
    data_rows = []
    
    for result in results:
        timestep = result['timestep']
        time_ps = timestep * 50 / 1000
        dimer_orders = result['dimer_orders']
        ru_atom_ids = result['ru_atom_ids']
        
        for dimer_order, ru_id in zip(dimer_orders, ru_atom_ids):
            data_rows.append({
                'timestep': timestep,
                'time_ps': time_ps,
                'dimer_order': dimer_order,
                'ru_atom_id': ru_id,
                'temperature_K': temperature
            })
    
    df = pd.DataFrame(data_rows)
    
    csv_filename = os.path.join(output_dir, f'dimer_order_{temperature}K.csv')
    df.to_csv(csv_filename, index=False)
    print(f"\nSaved {len(df)} dimer order measurements to {csv_filename}")
    
    if len(df) > 0:
        print(f"\nStatistics:")
        print(f"  Mean dimer order: {df['dimer_order'].mean():.4f}")
        print(f"  Std deviation: {df['dimer_order'].std():.4f}")
        print(f"  Min: {df['dimer_order'].min():.4f}")
        print(f"  Max: {df['dimer_order'].max():.4f}")
        print(f"  Unique Ru atoms: {df['ru_atom_id'].nunique()}")
    
    return csv_filename

def analyze_single_temperature(trajectory_path, temperature, n_processes=8):
    """Analyze dimer bond order for a single temperature"""
    print(f"\n{'='*60}")
    print(f"Analyzing {temperature}K")
    print(f"{'='*60}")
    
    start_time = time.time()
    print(f"Initial memory usage: {get_memory_usage():.2f} GB")
    
    results = compute_dimer_order_streaming(trajectory_path, n_processes)
    csv_filename = save_dimer_order_to_csv(results, temperature)
    
    total_time = time.time() - start_time
    print(f"\nTotal computation time: {total_time:.2f} seconds")
    print(f"Final memory usage: {get_memory_usage():.2f} GB")
    
    return csv_filename

def analyze_populations_across_temperatures(output_dir='.', 
                                           ref_temperature=10,
                                           pop1_range=None,
                                           pop2_range=None,
                                           pop3_range=None):
    """
    Analyze population evolution across all computed temperatures.
    """
    print("\n" + "="*60)
    print("Population Analysis Across Temperatures")
    print("="*60)
    
    ref_csv = os.path.join(output_dir, f'dimer_order_{ref_temperature}K.csv')
    if not os.path.exists(ref_csv):
        print(f"Error: Reference temperature file not found: {ref_csv}")
        return
    
    population_assignment, ref_stats = assign_populations_from_reference(
        ref_csv, pop1_range, pop2_range, pop3_range
    )
    
    # Save population assignment
    pop_assign_file = os.path.join(output_dir, 'dimer_population_assignment.csv')
    pop_df = pd.DataFrame([
        {'ru_atom_id': ru_id, 'population': pop}
        for ru_id, pop in population_assignment.items()
    ])
    pop_df.to_csv(pop_assign_file, index=False)
    print(f"\nSaved population assignment to {pop_assign_file}")
    
    # Find all temperature files
    import glob
    csv_files = glob.glob(os.path.join(output_dir, 'dimer_order_*K.csv'))
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
        csv_file = os.path.join(output_dir, f'dimer_order_{temp}K.csv')
        
        if not os.path.exists(csv_file):
            print(f"Warning: Missing file for {temp}K")
            continue
        
        # Check if file is empty
        if os.path.getsize(csv_file) < 100:  # Less than 100 bytes means essentially empty
            print(f"Warning: Empty file for {temp}K, skipping")
            continue
        
        stats = compute_population_statistics_for_temperature(csv_file, population_assignment)
        all_stats.append(stats)
        
        print(f"\n{temp}K:")
        print(f"  Pop 1: {stats['pop1_mean']:.4f} ± {stats['pop1_std']:.4f}  ({stats['pop1_count']} measurements)")
        print(f"  Pop 2: {stats['pop2_mean']:.4f} ± {stats['pop2_std']:.4f}  ({stats['pop2_count']} measurements)")
        print(f"  Pop 3: {stats['pop3_mean']:.4f} ± {stats['pop3_std']:.4f}  ({stats['pop3_count']} measurements)")
    
    # Save statistics
    stats_df = pd.DataFrame(all_stats)
    stats_df = stats_df.sort_values('temperature')
    stats_file = os.path.join(output_dir, 'dimer_population_statistics.csv')
    stats_df.to_csv(stats_file, index=False)
    print(f"\n{'='*60}")
    print(f"Saved population statistics to {stats_file}")
    print(f"{'='*60}")

def main():
    """Main function to compute dimer bond order"""
    temperature_paths = {
        10: "../../10K/lammps_out/rup_traj_sampled-50_100ps.xyz",
        # 50: "../../50K/lammps_out/rup_traj_sampled-50_100ps.xyz",
        # 100: "../../100K/lammps_out/rup_traj_sampled-50_100ps.xyz",
        # 150: "../../150K/lammps_out/rup_traj_sampled-50_100ps.xyz",
        # 200: "../../200K/lammps_out/rup_traj_sampled-50_100ps.xyz",
        # 250: "../../250K/lammps_out/rup_traj_sampled-50_100ps.xyz",
        # 300: "../../300K/lammps_out/rup_traj_sampled-50_100ps.xyz",
        # 310: "../../310K/lammps_out/rup_traj_sampled-50_100ps.xyz",
        # 320: "../../320K/lammps_out/rup_traj_sampled-50_100ps.xyz",
        # 330: "../../330K/lammps_out/rup_traj_sampled-50_100ps.xyz",
        # 340: "../../340K/lammps_out/rup_traj_sampled-50_100ps.xyz",
        # 350: "../../350K/lammps_out/rup_traj_sampled-50_100ps.xyz",
        # 360: "../../360K/lammps_out/rup_traj_sampled-50_100ps.xyz",
        # 370: "../../370K/lammps_out/rup_traj_sampled-50_100ps.xyz",
        # 380: "../../380K/lammps_out/rup_traj_sampled-50_100ps.xyz",
        # 390: "../../390K/lammps_out/rup_traj_sampled-50_100ps.xyz",
        # 400: "../../400K/lammps_out/rup_traj_sampled-50_100ps.xyz",
        # 450: "../../450K/lammps_out/rup_traj_sampled-50_100ps.xyz",
        # 500: "../../500K/lammps_out/rup_traj_sampled-50_100ps.xyz",
        # 550: "../../550K/lammps_out/rup_traj_sampled-50_100ps.xyz",
        # 600: "../../600K/lammps_out/rup_traj_sampled-50_100ps.xyz",
        # 650: "../../650K/lammps_out/rup_traj_sampled-50_100ps.xyz",
        # 700: "../../700K/lammps_out/rup_traj_sampled-50_100ps.xyz",
        # Add your temperature paths here
    }
    
    n_processes = 8
    
    print("Starting dimer bond order computation...")
    print(f"System info: {mp.cpu_count()} CPUs available, using {n_processes} processes")
    
    csv_files = []
    total_start = time.time()
    
    # Step 1: Compute dimer order for all temperatures
    for temperature, path in temperature_paths.items():
        if os.path.exists(path):
            csv_file = analyze_single_temperature(path, temperature, n_processes)
            if csv_file:
                csv_files.append(csv_file)
        else:
            print(f"Warning: File not found for {temperature}K: {path}")
    
    total_time = time.time() - total_start
    
    print(f"\n{'='*60}")
    print(f"Completed dimer order computation in {total_time:.1f} seconds!")
    print(f"Generated {len(csv_files)} CSV files")
    
    # Step 2: Analyze populations
    print("\n" + "="*60)
    print("Step 2: Analyzing populations...")
    print("="*60)
    
    analyze_populations_across_temperatures(
        output_dir='.',
        ref_temperature=10,
        pop1_range=None,  # Auto-detect from data
        pop2_range=None,
        pop3_range=None
    )
    
    print("\n" + "="*60)
    print("All processing complete!")
    print("="*60)

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
