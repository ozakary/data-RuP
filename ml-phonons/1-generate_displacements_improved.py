import numpy as np
from ase import Atoms
from ase.io import read, write
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from tqdm import tqdm
import os

def generate_phonopy_displacements(temp_dir, temp_value, displacement_distance=0.02):
    """
    Generate displaced structures using Phonopy for finite displacement method
    Saves all displacements to a single multi-frame XYZ file
    """
    # Read the MACE-calculated average structure
    avg_structure_file = f"./{temp_dir}/rup_traj_sampled-50_100ps_average_structure_{temp_value}K.xyz"
    
    print(f"\nProcessing {temp_dir} ({temp_value}K):")
    
    # Read the structure
    structure = read(avg_structure_file)
    print(f"  Loaded structure with {len(structure)} atoms")
    
    # Convert to PhonopyAtoms
    phonopy_atoms = PhonopyAtoms(
        symbols=structure.get_chemical_symbols(),
        positions=structure.get_positions(),
        cell=structure.cell.array,
        pbc=True
    )
    
    # Create Phonopy object (no supercell - use the structure as is)
    phonon = Phonopy(phonopy_atoms, supercell_matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    
    # Generate displacements with progress indication
    print("  Generating displacements...")
    from tqdm import tqdm as tqdm_base
    with tqdm_base(total=1, desc="  Computing displacements", unit="step", leave=False) as pbar:
        phonon.generate_displacements(distance=displacement_distance)
        pbar.update(1)
    
    print(f"  Generated {len(phonon.displacements)} displacements")
    print(f"  Displacement distance: {displacement_distance} Å")
    
    # Create output directory
    displaced_dir = f"./{temp_dir}/displaced_structures"
    os.makedirs(displaced_dir, exist_ok=True)
    
    supercells = phonon.supercells_with_displacements
    
    # Prepare output file for all displaced structures
    output_file = f"{displaced_dir}/all_displaced_structures.xyz"
    
    # Convert and write displaced structures in batches
    print(f"  Converting and writing displaced structures...")
    batch_size = 500  # Write in batches to avoid memory issues
    
    for batch_start in tqdm(range(0, len(supercells), batch_size), 
                            desc="  Processing batches", 
                            unit="batch"):
        batch_end = min(batch_start + batch_size, len(supercells))
        batch_atoms = []
        
        for i in range(batch_start, batch_end):
            displaced_cell = supercells[i]
            # Convert back to ASE Atoms
            displaced_atoms = Atoms(
                symbols=displaced_cell.symbols,
                positions=displaced_cell.positions,
                cell=displaced_cell.cell,
                pbc=True
            )
            
            # Add frame number to info for tracking
            displaced_atoms.info['displacement_index'] = i
            batch_atoms.append(displaced_atoms)
        
        # Write batch (append mode after first batch)
        if batch_start == 0:
            write(output_file, batch_atoms)
        else:
            write(output_file, batch_atoms, append=True)
    
    # Save the Phonopy object for later use
    phonon.save(f"./{temp_dir}/phonopy_{temp_value}K.yaml")
    
    print(f"  ✓ Saved {len(supercells)} displaced structures to {output_file}")
    print(f"  ✓ Saved Phonopy data to phonopy_{temp_value}K.yaml")
    
    return phonon, len(supercells)

# Generate displacements for all temperatures
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
print("STEP 1: Generating Displaced Structures with Phonopy")
print("="*50)

phonon_objects = {}
for temp_dir, temp_value in temperatures.items():
    phonon, n_displacements = generate_phonopy_displacements(temp_dir, temp_value, displacement_distance=0.02)
    phonon_objects[temp_dir] = phonon

print("\n" + "="*50)
print("Summary:")
print("="*50)
print("Displaced structures created for each temperature:")
for temp_dir, temp_value in temperatures.items():
    print(f"  {temp_dir}: ./{temp_dir}/displaced_structures/all_displaced_structures.xyz")

print("\n" + "="*50)
print("NEXT STEPS:")
print("="*50)
print("1. Run MACE calculations on the multi-frame XYZ file")
print("2. After MACE calculations, we'll collect forces and build force constants")
print("\nExpected MACE output:")
print("  Input:  all_displaced_structures.xyz (multi-frame)")
print("  Output: all_displaced_structures_out.xyz (multi-frame with forces)")
