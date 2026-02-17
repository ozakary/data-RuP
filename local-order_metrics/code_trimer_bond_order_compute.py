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

def compute_trimer_order_for_snapshot(snapshot, ru_pairs, ru_neighbor_graph):
    """
    Compute trimer bond order by following chains along [110] direction.
    For each Ru atom, follow forward along [110]: i → j → k → l
    Get d_12, d_23, d_34, then compute trimer order = (max - min) / average
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
    
    trimer_orders = []
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
        d_12 = None
        
        for neighbor_id in neighbor_ids:
            if neighbor_id not in atom_id_to_idx:
                continue
            
            idx_j = atom_id_to_idx[neighbor_id]
            pos_j = atoms.positions[idx_j]
            
            vec_ij = pos_j - pos_i
            vec_ij = vec_ij - cell.T @ np.round(np.linalg.solve(cell.T, vec_ij))
            
            projection = np.dot(vec_ij, direction_110_unit)
            
            if projection > 0:
                j_id = neighbor_id
                d_12 = np.linalg.norm(vec_ij)
                break
        
        if j_id is None:
            continue
        
        # Find forward neighbor k of j
        if j_id not in ru_neighbor_graph:
            continue
        
        j_neighbors = ru_neighbor_graph[j_id]
        k_id = None
        d_23 = None
        
        idx_j = atom_id_to_idx[j_id]
        pos_j = atoms.positions[idx_j]
        
        for neighbor_id in j_neighbors:
            if neighbor_id == ru_id:
                continue
            
            if neighbor_id not in atom_id_to_idx:
                continue
            
            idx_k = atom_id_to_idx[neighbor_id]
            pos_k = atoms.positions[idx_k]
            
            vec_jk = pos_k - pos_j
            vec_jk = vec_jk - cell.T @ np.round(np.linalg.solve(cell.T, vec_jk))
            
            projection = np.dot(vec_jk, direction_110_unit)
            
            if projection > 0:
                k_id = neighbor_id
                d_23 = np.linalg.norm(vec_jk)
                break
        
        if k_id is None:
            continue
        
        # Find forward neighbor l of k
        if k_id not in ru_neighbor_graph:
            continue
        
        k_neighbors = ru_neighbor_graph[k_id]
        l_id = None
        d_34 = None
        
        idx_k = atom_id_to_idx[k_id]
        pos_k = atoms.positions[idx_k]
        
        for neighbor_id in k_neighbors:
            if neighbor_id == j_id:
                continue
            
            if neighbor_id not in atom_id_to_idx:
                continue
            
            idx_l = atom_id_to_idx[neighbor_id]
            pos_l = atoms.positions[idx_l]
            
            vec_kl = pos_l - pos_k
            vec_kl = vec_kl - cell.T @ np.round(np.linalg.solve(cell.T, vec_kl))
            
            projection = np.dot(vec_kl, direction_110_unit)
            
            if projection > 0:
                l_id = neighbor_id
                d_34 = np.linalg.norm(vec_kl)
                break
        
        if l_id is None or d_34 is None:
            continue
        
        # Compute trimer order: (max - min) / average
        trimers = [d_12, d_23, d_34]
        avg_dist = np.mean(trimers)
        
        if avg_dist > 1e-6:
            trimer_order = (max(trimers) - min(trimers)) / avg_dist
            trimer_orders.append(trimer_order)
            ru_atom_ids.append(ru_id)
    
    return trimer_orders, ru_atom_ids

def identify_ru_atoms_and_direction(first_snapshot):
    """Identify Ru-Ru pairs along [110] using the correct original method"""
    return identify_ru_110_pairs(first_snapshot)

def compute_population_statistics_for_temperature(csv_file):
    """Compute statistics for the single population at a given temperature"""
    df = pd.read_csv(csv_file)
    temperature = df['temperature_K'].iloc[0]
    
    trimer_data = df['trimer_order']
    
    stats = {
        'temperature': temperature,
        'total_measurements': len(df),
        'mean': trimer_data.mean(),
        'std': trimer_data.std(),
        'min': trimer_data.min(),
        'max': trimer_data.max(),
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
        
        # Compute trimer order for this snapshot
        trimer_orders, ru_atom_ids = compute_trimer_order_for_snapshot(snapshot, ru_pairs, ru_neighbor_graph)
        
        results.append({
            'timestep': snapshot['timestep'],
            'trimer_orders': trimer_orders,
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

def compute_trimer_order_streaming(trajectory_path, n_processes=8):
    """
    Main function to compute trimer bond order.
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

def save_trimer_order_to_csv(results, temperature, output_dir='.'):
    """Save computed trimer orders to CSV file"""
    data_rows = []
    
    for result in results:
        timestep = result['timestep']
        time_ps = timestep * 50 / 1000
        trimer_orders = result['trimer_orders']
        ru_atom_ids = result['ru_atom_ids']
        
        for trimer_order, ru_id in zip(trimer_orders, ru_atom_ids):
            data_rows.append({
                'timestep': timestep,
                'time_ps': time_ps,
                'trimer_order': trimer_order,
                'ru_atom_id': ru_id,
                'temperature_K': temperature
            })
    
    df = pd.DataFrame(data_rows)
    
    csv_filename = os.path.join(output_dir, f'trimer_order_{temperature}K.csv')
    df.to_csv(csv_filename, index=False)
    print(f"\nSaved {len(df)} trimer order measurements to {csv_filename}")
    
    if len(df) > 0:
        print(f"\nStatistics:")
        print(f"  Mean trimer order: {df['trimer_order'].mean():.4f}")
        print(f"  Std deviation: {df['trimer_order'].std():.4f}")
        print(f"  Min: {df['trimer_order'].min():.4f}")
        print(f"  Max: {df['trimer_order'].max():.4f}")
        print(f"  Unique Ru atoms: {df['ru_atom_id'].nunique()}")
    
    return csv_filename

def analyze_single_temperature(trajectory_path, temperature, n_processes=8):
    """Analyze trimer bond order for a single temperature"""
    print(f"\n{'='*60}")
    print(f"Analyzing {temperature}K")
    print(f"{'='*60}")
    
    start_time = time.time()
    print(f"Initial memory usage: {get_memory_usage():.2f} GB")
    
    results = compute_trimer_order_streaming(trajectory_path, n_processes)
    csv_filename = save_trimer_order_to_csv(results, temperature)
    
    total_time = time.time() - start_time
    print(f"\nTotal computation time: {total_time:.2f} seconds")
    print(f"Final memory usage: {get_memory_usage():.2f} GB")
    
    return csv_filename

def analyze_statistics_across_temperatures(output_dir='.'):
    """
    Compute statistics across all temperatures.
    For trimer order, there's only 1 distribution, so we just compute mean/std at each T.
    """
    print("\n" + "="*60)
    print("Statistics Analysis Across Temperatures")
    print("="*60)
    
    # Find all temperature files
    import glob
    csv_files = glob.glob(os.path.join(output_dir, 'trimer_order_*K.csv'))
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
        csv_file = os.path.join(output_dir, f'trimer_order_{temp}K.csv')
        
        if not os.path.exists(csv_file):
            print(f"Warning: Missing file for {temp}K")
            continue
        
        # Check if file is empty
        if os.path.getsize(csv_file) < 100:
            print(f"Warning: Empty file for {temp}K, skipping")
            continue
        
        stats = compute_population_statistics_for_temperature(csv_file)
        all_stats.append(stats)
        
        print(f"\n{temp}K:")
        print(f"  Mean: {stats['mean']:.4f} ± {stats['std']:.4f}  ({stats['total_measurements']} measurements)")
    
    # Save statistics
    stats_df = pd.DataFrame(all_stats)
    stats_df = stats_df.sort_values('temperature')
    stats_file = os.path.join(output_dir, 'trimer_population_statistics.csv')
    stats_df.to_csv(stats_file, index=False)
    print(f"\n{'='*60}")
    print(f"Saved statistics to {stats_file}")
    print(f"{'='*60}")

def main():
    """Main function to compute trimer bond order"""
    temperature_paths = {
        # 10: "../../10K/lammps_out/rup_traj_sampled-50_100ps.xyz",
        # 50: "../../50K/lammps_out/rup_traj_sampled-50_100ps.xyz",
        # 100: "../../100K/lammps_out/rup_traj_sampled-50_100ps.xyz",
        # 150: "../../150K/lammps_out/rup_traj_sampled-50_100ps.xyz",
        # 200: "../../200K/lammps_out/rup_traj_sampled-50_100ps.xyz",
        # 250: "../../250K/lammps_out/rup_traj_sampled-50_100ps.xyz",
        # 300: "../../300K/lammps_out/rup_traj_sampled-50_100ps.xyz",
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
        # Add your temperature paths here
    }
    
    n_processes = 8
    
    print("Starting trimer bond order computation...")
    print(f"System info: {mp.cpu_count()} CPUs available, using {n_processes} processes")
    
    csv_files = []
    total_start = time.time()
    
    # Step 1: Compute trimer order for all temperatures
    for temperature, path in temperature_paths.items():
        if os.path.exists(path):
            csv_file = analyze_single_temperature(path, temperature, n_processes)
            if csv_file:
                csv_files.append(csv_file)
        else:
            print(f"Warning: File not found for {temperature}K: {path}")
    
    total_time = time.time() - total_start
    
    print(f"\n{'='*60}")
    print(f"Completed trimer order computation in {total_time:.1f} seconds!")
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
