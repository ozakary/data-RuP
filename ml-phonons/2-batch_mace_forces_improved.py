#!/usr/bin/env python3
"""
Batch MACE force calculation for Phonopy displaced structures
Processes multi-frame XYZ file containing all displaced structures
"""

import argparse
import os
from mace.calculators import MACECalculator
from ase.io import read, write
import numpy as np
from tqdm import tqdm
import torch

# Set up argument parser
parser = argparse.ArgumentParser(description='Batch MACE force calculation for displaced structures')
parser.add_argument('-m', '--model', type=str, required=True,
                    help='Path to MACE model file (.model or .pt)')
parser.add_argument('-i', '--input', type=str, required=True,
                    help='Input multi-frame XYZ file')
parser.add_argument('-o', '--output', type=str, required=True,
                    help='Output multi-frame XYZ file')
parser.add_argument('-dtype', type=str, default='float64',
                    choices=['float32', 'float64'], 
                    help='Data type (default: float64)')
parser.add_argument('--resume', action='store_true',
                    help='Resume from existing output file')
parser.add_argument('--save-frequency', type=int, default=1,
                    help='Save output every N structures (default: 1)')

args = parser.parse_args()

print("="*60)
print("BATCH MACE FORCE CALCULATION FOR PHONOPY")
print("="*60)
print(f"Model: {args.model}")
print(f"Data type: {args.dtype}")
print(f"Input file: {args.input}")
print(f"Output file: {args.output}")
print(f"Resume mode: {args.resume}")
print(f"Save frequency: every {args.save_frequency} structure(s)")
print("-" * 60)

# Check if input file exists
if not os.path.exists(args.input):
    print(f"ERROR: Input file '{args.input}' not found")
    exit(1)

# Load all structures from multi-frame XYZ
print("Loading displaced structures from multi-frame XYZ file...")
all_structures = read(args.input, index=':')
print(f"Loaded {len(all_structures)} displaced structures")
print("-" * 60)

# Load the MACE model once
print("Loading MACE model...")

# Check for GPU availability
if torch.cuda.is_available():
    device = 'cuda'
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
else:
    device = 'cpu'
    print("No GPU detected, using CPU")

# Load model directly to device
calc = MACECalculator(
    model_paths=args.model,
    device=device,
    default_dtype=args.dtype
)

print(f"Model loaded successfully on {device}")
print("-" * 60)

# Determine starting index for resume
start_idx = 0
if args.resume and os.path.exists(args.output):
    try:
        existing_structures = read(args.output, index=':')
        start_idx = len(existing_structures)
        print(f"Resume mode: Found {start_idx} existing structures")
        print(f"Continuing from structure {start_idx}...")
    except Exception as e:
        print(f"Warning: Could not read existing output file: {e}")
        print("Starting from beginning...")
        start_idx = 0
else:
    if os.path.exists(args.output) and not args.resume:
        print(f"Warning: Output file exists but --resume not specified")
        print(f"Output file will be overwritten")

print("-" * 60)

# Process all structures
print("Processing displaced structures...")

try:
    for i, atoms in enumerate(tqdm(all_structures[start_idx:], 
                                    desc="Calculating forces", 
                                    unit="structure",
                                    initial=start_idx,
                                    total=len(all_structures))):
        actual_idx = start_idx + i
        
        try:
            # Set calculator
            atoms.calc = calc
            
            # Calculate energy, forces, and stress
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
            stress = atoms.get_stress(voigt=False)  # 3x3 tensor in eV/Å³
            
            # Store results
            atoms.info['MACE_energy'] = energy
            atoms.arrays['MACE_forces'] = forces
            atoms.info['MACE_stress'] = stress.flatten()
            
            # Preserve displacement index if it exists
            if 'displacement_index' in atoms.info:
                atoms.info['displacement_index'] = atoms.info['displacement_index']
            else:
                atoms.info['displacement_index'] = actual_idx
            
            # Remove calculator before saving
            atoms.calc = None
            
            # Save structure immediately or at specified frequency
            if (i + 1) % args.save_frequency == 0 or (actual_idx + 1) == len(all_structures):
                # Append to output file
                if actual_idx == 0:
                    # First structure - create new file
                    write(args.output, atoms, format='extxyz')
                else:
                    # Append to existing file
                    write(args.output, atoms, format='extxyz', append=True)
            
        except Exception as e:
            print(f"\nERROR processing structure {actual_idx}: {e}")
            raise
    
    print("\nCalculation completed successfully!")
    
except KeyboardInterrupt:
    print("\n\nInterrupted by user.")
    print(f"Progress saved up to structure {actual_idx}")
    print(f"Use --resume flag to continue from where you left off")
    exit(1)

print("\n" + "="*60)
print("COMPLETED")
print("="*60)

# Read final output to get statistics
final_structures = read(args.output, index=':')
print(f"Total structures processed: {len(final_structures)}")

if final_structures:
    energies = [atoms.info['MACE_energy'] for atoms in final_structures]
    print(f"  Energy range: {min(energies):.4f} to {max(energies):.4f} eV")
    forces = [atoms.arrays['MACE_forces'] for atoms in final_structures]
    max_forces = [np.max(np.abs(f)) for f in forces]
    print(f"  Max force magnitude: {max(max_forces):.4f} eV/Å")

print("\nNext step: Collect forces and compute phonons with Phonopy")
