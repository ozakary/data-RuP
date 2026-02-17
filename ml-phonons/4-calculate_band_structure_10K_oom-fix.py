import numpy as np
from phonopy import Phonopy
from phonopy.interface.phonopy_yaml import read_cell_yaml
from phonopy.file_IO import parse_FORCE_CONSTANTS
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

def calculate_band_structure(temp_dir, temp_value):
    """
    Calculate and save phonon band structure for monoclinic RuP system
    """
    print(f"\nCalculating band structure for {temp_dir} ({temp_value}K):")

    # Read the cell 
    cell = read_cell_yaml(f"./{temp_dir}/phonopy_{temp_value}K_with_forces.yaml")
    supercell_matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    phonon = Phonopy(cell, supercell_matrix)

    # Load force constants
    fc_file = f"./{temp_dir}/FORCE_CONSTANTS"
    if os.path.exists(fc_file):
        force_constants = parse_FORCE_CONSTANTS(filename=fc_file)
        phonon.force_constants = force_constants
        print("  ✓ Force constants loaded from FORCE_CONSTANTS file")
    else:
        print("  ERROR: FORCE_CONSTANTS file not found!")
        return None

    # Define a simple continuous path for monoclinic structure
    # Using fractional coordinates in reciprocal space
    print("\n  Defining band path...")
    print("  Note: Using a simplified continuous path since structure has P1 symmetry")

    # Create a simple path that samples the Brillouin zone
    # This is a reasonable choice for a low-symmetry structure
    npoints = 26  # Points per segment

    path_segments = [
        # Segment 1: Gamma to X (along a*)
        (np.array([0.0, 0.0, 0.0]), np.array([0.5, 0.0, 0.0]), 'X'),
        # Segment 2: X to Y (along b*)
        (np.array([0.5, 0.0, 0.0]), np.array([0.5, 0.5, 0.0]), 'Y'),
        # Segment 3: Y to Gamma
        (np.array([0.5, 0.5, 0.0]), np.array([0.0, 0.0, 0.0]), '$\\Gamma$'),
        # Segment 4: Gamma to Z (along c*)
        (np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.5]), 'Z'),
        # Segment 5: Z to R (corner)
        (np.array([0.0, 0.0, 0.5]), np.array([0.5, 0.5, 0.5]), 'R'),
        # Segment 6: R to Gamma
        (np.array([0.5, 0.5, 0.5]), np.array([0.0, 0.0, 0.0]), '$\\Gamma$'),
    ]

    # Get atom indices for Ru atoms BEFORE the loop
    symbols = cell.symbols
    ru_indices = [i for i, sym in enumerate(symbols) if sym == 'Ru']
    print(f"  Found {len(ru_indices)} Ru atoms")

    # Calculate band structure for each segment
    all_distances = []
    all_frequencies = []
    all_ru_weights = []  # Store weights directly, not eigenvectors!
    all_qpoints = []
    distance_offset = 0

    print("  Calculating phonon frequencies along path...")

    for i, (start, end, label) in enumerate(path_segments):
        print(f"    Segment {i+1}/{len(path_segments)}: → {label}")

        # Generate q-points along this segment
        qpoints_segment = np.array([start + t * (end - start)
                                    for t in np.linspace(0, 1, npoints)])

        # Calculate frequencies and Ru projections for this segment
        frequencies_segment = []
        ru_weights_segment = []

        for q in tqdm(qpoints_segment, desc=f"      q-points", unit="q", leave=False):
            # Set q-point
            phonon.run_qpoints([q], with_eigenvectors=True)
            freq = phonon.get_qpoints_dict()['frequencies'][0]
            eigvec = phonon.get_qpoints_dict()['eigenvectors'][0]

            frequencies_segment.append(freq)
            
            # Calculate Ru projection IMMEDIATELY for this q-point
            n_bands = len(freq)
            ru_weight_q = np.zeros(n_bands)
            
            for ib in range(n_bands):
                eigv = eigvec[ib]
                ru_contribution = np.sum(np.abs(eigv[ru_indices])**2)
                total_norm = np.sum(np.abs(eigv)**2)
                ru_weight_q[ib] = ru_contribution / total_norm if total_norm > 0 else 0
            
            ru_weights_segment.append(ru_weight_q)
            # eigvec is automatically deleted after this iteration

        frequencies_segment = np.array(frequencies_segment)
        ru_weights_segment = np.array(ru_weights_segment)

        # Calculate distances along this segment
        segment_length = np.linalg.norm(phonon.primitive.cell.T @ (end - start))
        distances_segment = np.linspace(0, segment_length, npoints) + distance_offset
        distance_offset = distances_segment[-1]

        all_distances.append(distances_segment)
        all_frequencies.append(frequencies_segment)
        all_ru_weights.append(ru_weights_segment)
        all_qpoints.append(qpoints_segment)

    # Concatenate all segments
    distances = np.concatenate(all_distances)
    frequencies = np.concatenate(all_frequencies, axis=0)
    ru_weights = np.concatenate(all_ru_weights, axis=0)

    print(f"\n  Total: {len(distances)} q-points calculated")
    print(f"  Frequencies shape: {frequencies.shape}")
    print(f"  Number of bands: {frequencies.shape[1]}")
    print("  ✓ Ru projections calculated")

    # Determine special point positions for x-axis labels
    special_points = [all_distances[0][0]]  # Start with Gamma
    special_labels = ['$\\Gamma$']

    for i in range(len(path_segments)):
        special_points.append(all_distances[i][-1])
        special_labels.append(path_segments[i][2])

    # Plot band structure with Ru projection
    print("  Creating plots...")
    fig, ax = plt.subplots(figsize=(10, 6))

    for ib in tqdm(range(n_bands), desc="  Plotting bands", unit="band", leave=False):
        scatter = ax.scatter(
            distances,
            frequencies[:, ib],
            c=ru_weights[:, ib],
            s=10,
            cmap='Reds',
            vmin=0,
            vmax=1,
            alpha=0.7,
            rasterized=True
        )

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Ru character', fontsize=12)

    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_ylabel('Frequency (THz)', fontsize=14)
    ax.set_xlabel('', fontsize=14)

    for xp in special_points:
        ax.axvline(x=xp, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

    ax.set_xticks(special_points)
    ax.set_xticklabels(special_labels, fontsize=12)
    ax.set_xlim(distances[0], distances[-1])

    plt.tight_layout()

    output_file = f"./{temp_dir}/phonon_band_Ru_projection_{temp_value}K.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Band structure with Ru projection saved to {output_file}")

    # Simple plot without projection
    fig, ax = plt.subplots(figsize=(10, 6))

    for ib in tqdm(range(n_bands), desc="  Plotting simple bands", unit="band", leave=False):
        ax.plot(distances, frequencies[:, ib], 'b-', linewidth=0.5, alpha=0.7)

    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_ylabel('Frequency (THz)', fontsize=14)
    ax.set_xlabel('', fontsize=14)

    for xp in special_points:
        ax.axvline(x=xp, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

    ax.set_xticks(special_points)
    ax.set_xticklabels(special_labels, fontsize=12)
    ax.set_xlim(distances[0], distances[-1])

    plt.tight_layout()

    output_file_simple = f"./{temp_dir}/phonon_band_{temp_value}K.png"
    plt.savefig(output_file_simple, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Simple band structure saved to {output_file_simple}")

    # Save data
    npz_file = f"./{temp_dir}/band_data_{temp_value}K.npz"
    np.savez(
        npz_file,
        distances=distances,
        frequencies=frequencies,
        ru_weights=ru_weights,
        special_points=special_points,
        special_labels=special_labels
    )

    print(f"  ✓ Band data saved to {npz_file}")

    # Print frequency statistics
    print(f"\n  Frequency Statistics:")
    print(f"    Min frequency: {np.min(frequencies):.4f} THz")
    print(f"    Max frequency: {np.max(frequencies):.4f} THz")
    print(f"    Number of negative frequencies: {np.sum(frequencies < 0)}")
    if np.sum(frequencies < 0) > 0:
        print(f"    Most negative frequency: {np.min(frequencies):.4f} THz")

    return phonon

# Process your temperatures
temperatures = {
    '10K': 10,
}

print("="*50)
print("STEP 4: Phonon Band Structure with Custom Path")
print("="*50)
print(f"Total temperatures to process: {len(temperatures)}")
print("="*50)

phonon_objects = {}
for idx, (temp_dir, temp_value) in enumerate(temperatures.items(), 1):
    print(f"\n[{idx}/{len(temperatures)}] Processing {temp_dir}...")
    try:
        phonon = calculate_band_structure(temp_dir, temp_value)
        if phonon is not None:
            phonon_objects[temp_dir] = phonon
            print(f"✓ {temp_dir} completed successfully")
    except Exception as e:
        print(f"\n✗ ERROR processing {temp_dir}: {e}")
        import traceback
        traceback.print_exc()
        continue

print("\n" + "="*50)
print("COMPLETED!")
print("="*50)
print(f"\nSuccessfully processed: {len(phonon_objects)}/{len(temperatures)} temperatures")
if phonon_objects:
    print("\nCompleted temperatures:")
    for temp in phonon_objects.keys():
        print(f"  ✓ {temp}")
print("\nX-axis path: Γ → X → Y → Γ → Z → R → Γ")
print("This provides a continuous path sampling the Brillouin zone.")
