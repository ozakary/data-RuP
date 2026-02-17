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
    Handles periodic boundaries to ensure continuous chains wrap correctly.
    """
    print("Identifying Ru-Ru pairs along [110] direction from first snapshot...")
    
    # Create ASE Atoms object
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
    
    # Build neighbor list - use larger cutoff to ensure we find periodic connections
    cutoff_nearest = 4.0  # Increased to catch periodic boundary connections
    nl = NeighborList([cutoff_nearest/2] * len(atoms), self_interaction=False, bothways=False)
    nl.update(atoms)
    
    # For each Ru atom, find its two nearest neighbors along [110]
    # This ensures every Ru participates in the network
    ru_neighbor_graph = {}  # id -> list of neighbor ids along [110]
    
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
        
        # Filter for neighbors reasonably aligned with [110]
        # and within reasonable distance
        aligned_mask = (alignments > 0.9) & (distances < cutoff_nearest)
        
        if not np.any(aligned_mask):
            continue
        
        aligned_neighbors = ru_neighbors[aligned_mask]
        aligned_distances = distances[aligned_mask]
        aligned_projections = projections[aligned_mask]
        
        # Sort by distance and take the two nearest (one in each direction if possible)
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
    
    # Analyze connectivity
    atoms_with_neighbors = len(ru_neighbor_graph)
    total_connections = sum(len(neighbors) for neighbors in ru_neighbor_graph.values())
    avg_connections = total_connections / atoms_with_neighbors if atoms_with_neighbors > 0 else 0
    
    print(f"Atoms with neighbors along [110]: {atoms_with_neighbors}/{len(ru_indices)}")
    print(f"Average connections per atom: {avg_connections:.2f}")
    print(f"Identified {len(ru_pairs)} unique Ru-Ru pairs along [110]")
    print(f"Ratio of bonds to Ru atoms: {len(ru_pairs)/len(ru_indices):.3f}")
    
    # Validate pair distances
    pair_distances = []
    for id1, id2 in ru_pairs[:20]:  # Check first 20 pairs
        idx1 = np.where(first_snapshot['atom_ids'] == id1)[0][0]
        idx2 = np.where(first_snapshot['atom_ids'] == id2)[0][0]
        
        pos1 = atoms.positions[idx1]
        pos2 = atoms.positions[idx2]
        
        # Calculate distance with PBC
        vector = pos2 - pos1
        vector = vector - cell.T @ np.round(np.linalg.solve(cell.T, vector))
        distance = np.linalg.norm(vector)
        pair_distances.append(distance)
    
    if pair_distances:
        print(f"Sample pair distances (first 20): {np.mean(pair_distances):.3f} ± {np.std(pair_distances):.3f} Å")
        print(f"Range: {np.min(pair_distances):.3f} - {np.max(pair_distances):.3f} Å")
    
    return ru_pairs

def assign_populations_from_reference(ref_csv_file, 
                                     pop1_range=(2.8, 3.2), 
                                     pop2_range=(3.4, 3.8)):
    """
    Assign atom pairs to populations based on their distances in the reference temperature.
    
    Parameters:
    -----------
    ref_csv_file : str
        Path to CSV file for reference temperature (e.g., 10K)
    pop1_range : tuple
        (min, max) distance range for population 1 in Angstroms
    pop2_range : tuple
        (min, max) distance range for population 2 in Angstroms
    
    Returns:
    --------
    population_assignment : dict
        Dictionary mapping (atom_id1, atom_id2) to population (1 or 2)
    stats : dict
        Statistics about the populations
    """
    print(f"\nAssigning populations from reference file: {ref_csv_file}")
    print(f"Population 1 range: {pop1_range[0]:.2f} - {pop1_range[1]:.2f} Å")
    print(f"Population 2 range: {pop2_range[0]:.2f} - {pop2_range[1]:.2f} Å")
    
    # Load reference data - need atom IDs to track pairs
    # First, we need to read the pairs file to get atom IDs
    pairs_file = ref_csv_file.replace('distances', 'pairs').replace('.csv', '.txt')
    
    if not os.path.exists(pairs_file):
        raise FileNotFoundError(f"Pairs file not found: {pairs_file}")
    
    # Read pairs
    atom_pairs = []
    with open(pairs_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) == 2:
                atom_pairs.append((int(parts[0]), int(parts[1])))
    
    print(f"Loaded {len(atom_pairs)} atom pairs from {pairs_file}")
    
    # Read distances from first frame only
    df = pd.read_csv(ref_csv_file)
    first_timestep = df['timestep'].min()
    first_frame = df[df['timestep'] == first_timestep]
    
    distances_first_frame = first_frame['distance_angstrom'].values
    
    if len(distances_first_frame) != len(atom_pairs):
        print(f"Warning: Number of distances ({len(distances_first_frame)}) != number of pairs ({len(atom_pairs)})")
        print("Will use the minimum of the two...")
        n_pairs = min(len(distances_first_frame), len(atom_pairs))
        distances_first_frame = distances_first_frame[:n_pairs]
        atom_pairs = atom_pairs[:n_pairs]
    
    # Assign populations
    population_assignment = {}
    pop1_indices = []
    pop2_indices = []
    unassigned_indices = []
    
    for i, ((id1, id2), dist) in enumerate(zip(atom_pairs, distances_first_frame)):
        pair_key = (id1, id2)
        
        if pop1_range[0] <= dist <= pop1_range[1]:
            population_assignment[pair_key] = 1
            pop1_indices.append(i)
        elif pop2_range[0] <= dist <= pop2_range[1]:
            population_assignment[pair_key] = 2
            pop2_indices.append(i)
        else:
            # Distances outside both ranges are unassigned
            unassigned_indices.append(i)
    
    # Calculate statistics
    pop1_distances = distances_first_frame[pop1_indices]
    pop2_distances = distances_first_frame[pop2_indices]
    
    stats = {
        'total_pairs': len(atom_pairs),
        'pop1_count': len(pop1_indices),
        'pop2_count': len(pop2_indices),
        'unassigned_count': len(unassigned_indices),
        'pop1_mean': pop1_distances.mean() if len(pop1_distances) > 0 else 0,
        'pop1_std': pop1_distances.std() if len(pop1_distances) > 0 else 0,
        'pop1_min': pop1_distances.min() if len(pop1_distances) > 0 else 0,
        'pop1_max': pop1_distances.max() if len(pop1_distances) > 0 else 0,
        'pop2_mean': pop2_distances.mean() if len(pop2_distances) > 0 else 0,
        'pop2_std': pop2_distances.std() if len(pop2_distances) > 0 else 0,
        'pop2_min': pop2_distances.min() if len(pop2_distances) > 0 else 0,
        'pop2_max': pop2_distances.max() if len(pop2_distances) > 0 else 0,
    }
    
    print(f"\n=== Population Assignment Results ===")
    print(f"Total Ru-Ru pairs along [110]: {stats['total_pairs']}")
    print(f"\nPopulation 1 (short): {stats['pop1_count']} pairs ({100*stats['pop1_count']/stats['total_pairs']:.1f}%)")
    print(f"  Mean: {stats['pop1_mean']:.3f} Å")
    print(f"  Std:  {stats['pop1_std']:.3f} Å")
    print(f"  Range: {stats['pop1_min']:.3f} - {stats['pop1_max']:.3f} Å")
    
    print(f"\nPopulation 2 (long):  {stats['pop2_count']} pairs ({100*stats['pop2_count']/stats['total_pairs']:.1f}%)")
    print(f"  Mean: {stats['pop2_mean']:.3f} Å")
    print(f"  Std:  {stats['pop2_std']:.3f} Å")
    print(f"  Range: {stats['pop2_min']:.3f} - {stats['pop2_max']:.3f} Å")
    
    print(f"\nUnassigned: {stats['unassigned_count']} pairs ({100*stats['unassigned_count']/stats['total_pairs']:.1f}%)")
    
    # Sanity check
    assigned_total = stats['pop1_count'] + stats['pop2_count']
    print(f"\n=== Sanity Check ===")
    print(f"Pop1 + Pop2 = {assigned_total} (should be ≈ {stats['total_pairs']} accounting for unassigned)")
    print(f"Assigned fraction: {100*assigned_total/stats['total_pairs']:.1f}%")
    
    if stats['unassigned_count'] > 0:
        unassigned_dists = distances_first_frame[unassigned_indices]
        print(f"\nUnassigned distances range: {unassigned_dists.min():.3f} - {unassigned_dists.max():.3f} Å")
        print("These are likely P-Ru or other non-[110] bonds")
    
    return population_assignment, stats, atom_pairs

def compute_population_statistics_for_temperature(csv_file, pairs_file, population_assignment):
    """
    Compute statistics for each population at a given temperature.
    """
    # Load atom pairs
    atom_pairs = []
    with open(pairs_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) == 2:
                atom_pairs.append((int(parts[0]), int(parts[1])))
    
    # Load distance data
    df = pd.read_csv(csv_file)
    
    # Get temperature
    temperature = df['temperature_K'].iloc[0]
    
    # Group distances by pair and frame
    n_frames = df['timestep'].nunique()
    n_pairs_per_frame = len(df) // n_frames
    
    # Assign population to each distance measurement
    populations = []
    for i in range(len(df)):
        pair_idx = i % n_pairs_per_frame
        if pair_idx < len(atom_pairs):
            pair = atom_pairs[pair_idx]
            pop = population_assignment.get(pair, 0)  # 0 for unassigned
            populations.append(pop)
        else:
            populations.append(0)
    
    df['population'] = populations
    
    # Calculate statistics
    pop1_data = df[df['population'] == 1]['distance_angstrom']
    pop2_data = df[df['population'] == 2]['distance_angstrom']
    
    stats = {
        'temperature': temperature,
        'total_measurements': len(df),
        'pop1_count': len(pop1_data),
        'pop1_mean': pop1_data.mean() if len(pop1_data) > 0 else np.nan,
        'pop1_std': pop1_data.std() if len(pop1_data) > 0 else np.nan,
        'pop1_min': pop1_data.min() if len(pop1_data) > 0 else np.nan,
        'pop1_max': pop1_data.max() if len(pop1_data) > 0 else np.nan,
        'pop2_count': len(pop2_data),
        'pop2_mean': pop2_data.mean() if len(pop2_data) > 0 else np.nan,
        'pop2_std': pop2_data.std() if len(pop2_data) > 0 else np.nan,
        'pop2_min': pop2_data.min() if len(pop2_data) > 0 else np.nan,
        'pop2_max': pop2_data.max() if len(pop2_data) > 0 else np.nan,
    }
    
    return stats

def process_trajectory_chunk(chunk_data):
    """Process a chunk of the trajectory file"""
    trajectory_path, start_pos, end_pos, ru_pairs = chunk_data
    
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
        
        # Compute distances for this snapshot
        atoms = Atoms(
            symbols=snapshot['symbols'],
            positions=snapshot['positions'],
            cell=snapshot['cell'],
            pbc=True
        )
        
        atom_id_to_idx = {aid: idx for idx, aid in enumerate(snapshot['atom_ids'])}
        
        distances = []
        for id1, id2 in ru_pairs:
            if id1 not in atom_id_to_idx or id2 not in atom_id_to_idx:
                continue
            
            idx1 = atom_id_to_idx[id1]
            idx2 = atom_id_to_idx[id2]
            
            pos1 = atoms.positions[idx1]
            pos2 = atoms.positions[idx2]
            
            # PBC distance
            vector = pos2 - pos1
            vector = vector - atoms.cell.T @ np.round(np.linalg.solve(atoms.cell.T, vector))
            distance = np.linalg.norm(vector)
            distances.append(distance)
        
        results.append({
            'timestep': snapshot['timestep'],
            'distances': distances
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
                # Skip this snapshot
                n_atoms = int(line.strip())
                f.readline()  # Skip header
                for _ in range(n_atoms):
                    f.readline()
            
            pos = f.tell()
    
    return positions

def compute_ru_110_distances_streaming(trajectory_path, n_processes=8):
    """
    Main function to compute Ru-Ru distances along [110].
    Uses streaming to handle large files efficiently.
    """
    print(f"Processing trajectory: {trajectory_path}")
    print(f"Using {n_processes} processes")
    
    # Step 1: Read only the first snapshot to identify Ru pairs
    print("\nStep 1: Reading first snapshot...")
    with open(trajectory_path, 'r') as f:
        lines = []
        # Read just enough lines for the first snapshot
        n_atoms_line = f.readline()
        lines.append(n_atoms_line)
        n_atoms = int(n_atoms_line.strip())
        
        # Read header and atomic data
        for _ in range(n_atoms + 1):
            lines.append(f.readline())
    
    first_snapshot = parse_single_snapshot(lines, 0)
    
    # Step 2: Identify Ru-Ru pairs along [110]
    print("\nStep 2: Identifying Ru-Ru pairs along [110]...")
    ru_pairs = identify_ru_110_pairs(first_snapshot)
    
    # Step 3: Find all snapshot positions in the file
    print("\nStep 3: Finding snapshot positions...")
    snapshot_positions = find_snapshot_positions(trajectory_path)
    print(f"Found {len(snapshot_positions)} snapshots")
    
    # Step 4: Create chunks for parallel processing
    print("\nStep 4: Processing trajectory in parallel...")
    file_size = os.path.getsize(trajectory_path)
    
    # Create chunks
    n_chunks = min(n_processes * 4, len(snapshot_positions))  # More chunks for better load balancing
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
        
        chunks.append((trajectory_path, start_pos, end_pos, ru_pairs))
    
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
    
    return all_results, ru_pairs

def save_distances_to_csv(results, temperature, ru_pairs, output_dir='.'):
    """Save computed distances to CSV file"""
    # Flatten all distances
    data_rows = []
    
    for result in results:
        timestep = result['timestep']
        time_ps = timestep * 50 / 1000  # Convert to ps
        distances = result['distances']
        
        for distance in distances:
            data_rows.append({
                'timestep': timestep,
                'time_ps': time_ps,
                'distance_angstrom': distance,
                'temperature_K': temperature
            })
    
    df = pd.DataFrame(data_rows)
    
    # Save to CSV
    csv_filename = os.path.join(output_dir, f'ru_110_distances_{temperature}K.csv')
    df.to_csv(csv_filename, index=False)
    print(f"\nSaved {len(df)} distance measurements to {csv_filename}")
    
    # Save pair information
    pairs_filename = os.path.join(output_dir, f'ru_110_pairs_{temperature}K.txt')
    with open(pairs_filename, 'w') as f:
        f.write(f"# Ru-Ru pairs along [110] direction\n")
        f.write(f"# Total pairs: {len(ru_pairs)}\n")
        for id1, id2 in sorted(ru_pairs):
            f.write(f"{id1} {id2}\n")
    print(f"Saved pair information to {pairs_filename}")
    
    # Print statistics
    if len(df) > 0:
        print(f"\nStatistics:")
        print(f"  Mean distance: {df['distance_angstrom'].mean():.3f} Å")
        print(f"  Std deviation: {df['distance_angstrom'].std():.3f} Å")
        print(f"  Min distance: {df['distance_angstrom'].min():.3f} Å")
        print(f"  Max distance: {df['distance_angstrom'].max():.3f} Å")
        print(f"  Measurements per snapshot: {len(ru_pairs)}")
    
    return csv_filename, pairs_filename

def analyze_single_temperature(trajectory_path, temperature, n_processes=8):
    """Analyze Ru-Ru [110] distances for a single temperature"""
    print(f"\n{'='*60}")
    print(f"Analyzing {temperature}K")
    print(f"{'='*60}")
    
    start_time = time.time()
    print(f"Initial memory usage: {get_memory_usage():.2f} GB")
    
    # Compute distances with streaming approach
    results, ru_pairs = compute_ru_110_distances_streaming(trajectory_path, n_processes)
    
    # Save to CSV
    csv_filename, pairs_filename = save_distances_to_csv(results, temperature, ru_pairs)
    
    total_time = time.time() - start_time
    print(f"\nTotal computation time: {total_time:.2f} seconds")
    print(f"Final memory usage: {get_memory_usage():.2f} GB")
    
    return csv_filename, pairs_filename

def analyze_populations_across_temperatures(output_dir='.', 
                                           ref_temperature=10,
                                           pop1_range=(2.8, 3.2),
                                           pop2_range=(3.4, 3.8)):
    """
    Analyze population evolution across all computed temperatures.
    Must be run after all distance computations are complete.
    """
    print("\n" + "="*60)
    print("Population Analysis Across Temperatures")
    print("="*60)
    
    # Find reference temperature file
    ref_csv = os.path.join(output_dir, f'ru_110_distances_{ref_temperature}K.csv')
    if not os.path.exists(ref_csv):
        print(f"Error: Reference temperature file not found: {ref_csv}")
        print("Please compute distances for the reference temperature first.")
        return
    
    # Assign populations from reference temperature
    population_assignment, ref_stats, atom_pairs = assign_populations_from_reference(
        ref_csv, pop1_range, pop2_range
    )
    
    # Save population assignment
    pop_assign_file = os.path.join(output_dir, 'population_assignment.csv')
    pop_df = pd.DataFrame([
        {'atom_i': pair[0], 'atom_j': pair[1], 'population': pop}
        for pair, pop in population_assignment.items()
    ])
    pop_df.to_csv(pop_assign_file, index=False)
    print(f"\nSaved population assignment to {pop_assign_file}")
    
    # Find all temperature files
    import glob
    csv_files = glob.glob(os.path.join(output_dir, 'ru_110_distances_*K.csv'))
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
        csv_file = os.path.join(output_dir, f'ru_110_distances_{temp}K.csv')
        pairs_file = os.path.join(output_dir, f'ru_110_pairs_{temp}K.txt')
        
        if not os.path.exists(csv_file) or not os.path.exists(pairs_file):
            print(f"Warning: Missing files for {temp}K")
            continue
        
        stats = compute_population_statistics_for_temperature(
            csv_file, pairs_file, population_assignment
        )
        all_stats.append(stats)
        
        print(f"\n{temp}K:")
        print(f"  Pop 1: {stats['pop1_mean']:.3f} ± {stats['pop1_std']:.3f} Å  ({stats['pop1_count']} measurements)")
        print(f"  Pop 2: {stats['pop2_mean']:.3f} ± {stats['pop2_std']:.3f} Å  ({stats['pop2_count']} measurements)")
    
    # Save statistics
    stats_df = pd.DataFrame(all_stats)
    stats_df = stats_df.sort_values('temperature')
    stats_file = os.path.join(output_dir, 'population_statistics.csv')
    stats_df.to_csv(stats_file, index=False)
    print(f"\n{'='*60}")
    print(f"Saved population statistics to {stats_file}")
    print(f"{'='*60}")

def main():
    """Main function to compute Ru-Ru [110] distances"""
    # Define temperatures to analyze
    temperature_paths = {
        # Uncomment to add more temperatures:
        10: "../../10K/lammps_out/rup_traj_sampled-50_100ps.xyz",
        50: "../../50K/lammps_out/rup_traj_sampled-50_100ps.xyz",
        100: "../../100K/lammps_out/rup_traj_sampled-50_100ps.xyz",
        150: "../../150K/lammps_out/rup_traj_sampled-50_100ps.xyz",
        200: "../../200K/lammps_out/rup_traj_sampled-50_100ps.xyz",
        250: "../../250K/lammps_out/rup_traj_sampled-50_100ps.xyz",
        300: "../../300K/lammps_out/rup_traj_sampled-50_100ps.xyz",
        310: "../../310K/lammps_out/rup_traj_sampled-50_100ps.xyz",
        320: "../../320K/lammps_out/rup_traj_sampled-50_100ps.xyz",
        330: "../../330K/lammps_out/rup_traj_sampled-50_100ps.xyz",
        340: "../../340K/lammps_out/rup_traj_sampled-50_100ps.xyz",
        350: "../../350K/lammps_out/rup_traj_sampled-50_100ps.xyz",
        360: "../../360K/lammps_out/rup_traj_sampled-50_100ps.xyz",
        370: "../../370K/lammps_out/rup_traj_sampled-50_100ps.xyz",
        380: "../../380K/lammps_out/rup_traj_sampled-50_100ps.xyz",
        390: "../../390K/lammps_out/rup_traj_sampled-50_100ps.xyz",
        400: "../../400K/lammps_out/rup_traj_sampled-50_100ps.xyz",
        450: "../../450K/lammps_out/rup_traj_sampled-50_100ps.xyz",
        500: "../../500K/lammps_out/rup_traj_sampled-50_100ps.xyz",
        550: "../../550K/lammps_out/rup_traj_sampled-50_100ps.xyz",
        600: "../../600K/lammps_out/rup_traj_sampled-50_100ps.xyz",
        650: "../../650K/lammps_out/rup_traj_sampled-50_100ps.xyz",
        700: "../../700K/lammps_out/rup_traj_sampled-50_100ps.xyz",
        1000: "../../1000K/lammps_out/rup_traj_sampled-50_100ps.xyz",        
    }
    
    n_processes = 8  # Adjustable
    
    print("Starting Ru-Ru [110] distance computation with population tracking...")
    print(f"System info: {mp.cpu_count()} CPUs available, using {n_processes} processes")
    
    csv_files = []
    total_start = time.time()
    
    # Step 1: Compute distances for all temperatures
    for temperature, path in temperature_paths.items():
        if os.path.exists(path):
            csv_file, pairs_file = analyze_single_temperature(path, temperature, n_processes)
            if csv_file:
                csv_files.append(csv_file)
        else:
            print(f"Warning: File not found for {temperature}K: {path}")
    
    total_time = time.time() - total_start
    
    print(f"\n{'='*60}")
    print(f"Completed distance computation in {total_time:.1f} seconds!")
    print(f"Generated {len(csv_files)} CSV files")
    
    # Step 2: Analyze populations
    print("\n" + "="*60)
    print("Step 2: Analyzing populations...")
    print("="*60)
    
    analyze_populations_across_temperatures(
        output_dir='.',
        ref_temperature=10,  # Reference temperature for population assignment
        pop1_range=(2.8, 3.2),  # Population 1: short distances
        pop2_range=(3.4, 3.8)   # Population 2: long distances
    )
    
    print("\n" + "="*60)
    print("All processing complete!")
    print("="*60)

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
