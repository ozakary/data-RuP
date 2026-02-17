import numpy as np
from ase.io import read, write
from ase import Atoms
from scipy.spatial.distance import pdist
from tqdm import tqdm

def generate_average_structure(temp_dir, temp_value):
    """
    Generate average structure - correct method for NPT simulations
    """
    trajectory_file = f"../../{temp_dir}/lammps_out/rup_traj_sampled-50_100ps.xyz"
    
    print(f"\nProcessing {temp_dir} ({temp_value}K):")
    
    # Read trajectory
    trajectory = read(trajectory_file, index=':')
    print(f"  Loaded {len(trajectory)} frames")
    
    n_atoms = len(trajectory[0])
    n_frames = len(trajectory)
    
    # Collect all fractional coordinates
    all_scaled = np.zeros((n_frames, n_atoms, 3))
    
    print("  Collecting fractional coordinates...")
    for i, frame in enumerate(tqdm(trajectory, desc=f"  {temp_value}K", ncols=80, leave=False)):
        all_scaled[i] = frame.get_scaled_positions()
    
    # Unwrap: ensure continuous trajectories in fractional space
    print("  Unwrapping trajectories...")
    for i in range(1, n_frames):
        # Calculate displacement from previous frame
        delta = all_scaled[i] - all_scaled[i-1]
        
        # Atoms that jumped across boundary (|delta| > 0.5)
        # Correct by ±1.0 to keep them on same side
        jumps = np.round(delta)
        all_scaled[i] = all_scaled[i-1] + (delta - jumps)
    
    # Average fractional coordinates
    avg_scaled = np.mean(all_scaled, axis=0)
    
    # Wrap back to [0, 1) range
    avg_scaled = avg_scaled % 1.0
    
    # Average cell
    sum_cell = np.zeros((3, 3))
    for frame in trajectory:
        sum_cell += frame.cell.array
    avg_cell = sum_cell / n_frames
    
    # Create average structure
    avg_structure = Atoms(
        symbols=trajectory[0].get_chemical_symbols(),
        scaled_positions=avg_scaled,
        cell=avg_cell,
        pbc=True
    )
    
    # Verify minimum distance
    dists = pdist(avg_structure.get_positions())
    min_dist = np.min(dists)
    
    output_file = f"../../{temp_dir}/lammps_out/rup_traj_sampled-50_100ps_average_structure_{temp_value}K.xyz"
    write(output_file, avg_structure)
    
    status = "✓ OK" if min_dist >= 1.7 else "⚠️  WARNING"
    print(f"  Minimum distance: {min_dist:.3f} Å {status}")
    print(f"  Cell: a={avg_structure.cell.cellpar()[0]:.3f}, b={avg_structure.cell.cellpar()[1]:.3f}, c={avg_structure.cell.cellpar()[2]:.3f}")
    print(f"  Saved to: {output_file}")
    
    return avg_structure

# Process all temperatures
temperatures = {
    '10K': 10,
    '50K': 50,
    '100K': 100,
    '150K': 150,
    '200K': 200,
    '250K': 250,
    '300K': 300,
    '310K': 310,
    '320K': 320,
    '330K': 330,
    '340K': 340,
    '350K': 350,
    '360K': 360,
    '370K': 370,
    '380K': 380,
    '390K': 390,
    '400K': 400,
    '450K': 450,
    '500K': 500,
    '550K': 550,
    '600K': 600,
    '650K': 650,
    '700K': 700
}

print("="*50)
print("STEP 1: Generating Average Structures (NPT-corrected)")
print("="*50)

average_structures = {}
for temp_dir, temp_value in temperatures.items():
    average_structures[temp_dir] = generate_average_structure(temp_dir, temp_value)

print("\n" + "="*50)
print("Summary:")
print("="*50)
for temp_dir, temp_value in temperatures.items():
    avg = average_structures[temp_dir]
    dists = pdist(avg.get_positions())
    min_dist = np.min(dists)
    status = "✓" if min_dist >= 1.7 else "⚠️"
    print(f"  {status} {temp_dir}: min_dist = {min_dist:.3f} Å")

print("\nNext step: Use MACE to calculate forces for these structures")
