import numpy as np
from ase.io import read
from phonopy import Phonopy
from phonopy.interface.phonopy_yaml import read_cell_yaml
from phonopy.file_IO import write_FORCE_CONSTANTS
from tqdm import tqdm
import os

def collect_forces_and_compute_phonons(temp_dir, temp_value):
    """
    Collect forces from MACE calculations and compute phonon properties
    """
    print(f"\nProcessing {temp_dir} ({temp_value}K):")
    
    # Read the cell information from YAML
    cell = read_cell_yaml(f"./{temp_dir}/phonopy_{temp_value}K.yaml")
    
    # Supercell matrix (same as what you used in step 1: 1x1x1, no supercell)
    supercell_matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    
    # Recreate the Phonopy object
    phonon = Phonopy(cell, supercell_matrix)
    
    # Re-generate displacements (should be identical to before)
    print("  Regenerating displacements...")
    phonon.generate_displacements(distance=0.02)
    n_expected_displacements = len(phonon.displacements)
    print(f"  Expected {n_expected_displacements} displaced structures")
    
    # Path to MACE output file (multi-frame XYZ)
    displaced_dir = f"./{temp_dir}/displaced_structures"
    output_file = f"{displaced_dir}/all_displaced_structures_out.xyz"
    
    if not os.path.exists(output_file):
        raise FileNotFoundError(f"MACE output file not found: {output_file}")
    
    # Read all structures from multi-frame XYZ
    print(f"  Loading structures from {output_file}...")
    all_structures = read(output_file, index=':')
    print(f"  Found {len(all_structures)} structures with forces")
    
    if len(all_structures) != n_expected_displacements:
        raise ValueError(
            f"Number of structures ({len(all_structures)}) does not match "
            f"expected displacements ({n_expected_displacements})"
        )
    
    # Collect forces from all displaced structures
    print("  Collecting forces from MACE calculations...")
    forces_list = []
    
    for i, atoms in enumerate(tqdm(all_structures, desc="  Reading forces", unit="structure")):
        if 'MACE_forces' not in atoms.arrays:
            raise ValueError(f"No MACE_forces found in structure {i}")
        
        forces = atoms.arrays['MACE_forces']
        
        # Verify displacement index if available
        if 'displacement_index' in atoms.info:
            expected_idx = atoms.info['displacement_index']
            if expected_idx != i:
                print(f"  WARNING: Structure {i} has displacement_index={expected_idx}")
        
        forces_list.append(forces)
    
    # Convert to numpy array
    forces_array = np.array(forces_list)
    print(f"  Forces array shape: {forces_array.shape}")
    print(f"  Expected shape: ({n_expected_displacements}, {len(cell.symbols)}, 3)")
    
    # Set forces in Phonopy
    phonon.forces = forces_array
    
    # Produce force constants
    print("  Computing force constants...")
    phonon.produce_force_constants()
    print("  ✓ Force constants calculated")
    
    # Save force constants to separate file
    fc_file = f"./{temp_dir}/FORCE_CONSTANTS"
    write_FORCE_CONSTANTS(phonon.force_constants, filename=fc_file)
    print(f"  ✓ Force constants saved to {fc_file}")
    
    # Save force constants in YAML
    yaml_file = f"./{temp_dir}/phonopy_{temp_value}K_with_forces.yaml"
    phonon.save(yaml_file)
    print(f"  ✓ Phonopy data with forces saved to {yaml_file}")
    
    # Print some statistics
    print(f"\n  Force Constants Statistics:")
    print(f"    Shape: {phonon.force_constants.shape}")
    print(f"    Max |FC|: {np.max(np.abs(phonon.force_constants)):.6f} eV/Å²")
    print(f"    Mean |FC|: {np.mean(np.abs(phonon.force_constants)):.6f} eV/Å²")
    
    return phonon

# Process your temperature
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
    '700K': 700,
}

print("="*50)
print("STEP 3: Collecting Forces and Computing Force Constants")
print("="*50)

phonon_objects = {}
for temp_dir, temp_value in temperatures.items():
    try:
        phonon = collect_forces_and_compute_phonons(temp_dir, temp_value)
        phonon_objects[temp_dir] = phonon
    except Exception as e:
        print(f"\nERROR processing {temp_dir}: {e}")
        import traceback
        traceback.print_exc()
        continue

print("\n" + "="*50)
print("COMPLETED - Force Constants Computed!")
print("="*50)

if phonon_objects:
    print("\nSuccessfully processed temperatures:")
    for temp_dir in phonon_objects.keys():
        print(f"  ✓ {temp_dir}")
else:
    print("\nWARNING: No temperatures were successfully processed!")
