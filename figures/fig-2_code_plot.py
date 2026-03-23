#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import matplotlib.ticker as ticker
import re

# Import the figure_formatting module (optional, remove if not needed)
try:
    import figure_formatting as ff
    ff.set_rcParams(ff.master_formatting)
except:
    print("Warning: figure_formatting module not found, using default matplotlib settings")

def parse_xyz_file(filename):
    """
    Parse XYZ file and extract energy, forces, and stress tensor data
    """
    energies = []
    forces_all = []  # All force components
    stress_all = []  # All stress tensor components
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        # First line: number of atoms
        try:
            n_atoms = int(lines[i].strip())
        except:
            i += 1
            continue
        
        # Second line: contains Lattice, energy, stress, etc.
        if i + 1 < len(lines):
            header_line = lines[i + 1]
            
            # Extract energy
            energy_match = re.search(r'energy=([-\d.]+)', header_line)
            if energy_match:
                energies.append(float(energy_match.group(1)))
            
            # Extract stress tensor (6 components for symmetric tensor)
            stress_match = re.search(r'stress="([^"]+)"', header_line)
            if stress_match:
                stress_values = [float(x) for x in stress_match.group(1).split()]
                stress_all.extend(stress_values)
        
        # Next n_atoms lines: atom positions and forces
        for j in range(n_atoms):
            if i + 2 + j < len(lines):
                atom_line = lines[i + 2 + j].strip()
                if atom_line:
                    parts = atom_line.split()
                    if len(parts) >= 7:  # species + 3 pos + 3 forces
                        # Extract forces (last 3 columns)
                        fx, fy, fz = float(parts[4]), float(parts[5]), float(parts[6])
                        forces_all.extend([fx, fy, fz])
        
        # Move to next structure
        i += n_atoms + 2
    
    return np.array(energies), np.array(forces_all), np.array(stress_all)

# Parse the XYZ file
print("Parsing XYZ file...")
energies, forces, stresses = parse_xyz_file("./mono_and_ortho_all-temps_NpT_PBE-D4_sampled-10fs-traj_ML-dataset.xyz")

print(f"\nData extracted:")
print(f"  Number of structures: {len(energies)}")
print(f"  Number of force components: {len(forces)}")
print(f"  Number of stress components: {len(stresses)}")

#==============================================================================
# PLOT 1: ENERGY DISTRIBUTION (WITH BROKEN AXIS)
#==============================================================================
# First, analyze the energy distribution to determine if we need broken axis
energy_sorted = np.sort(energies)
energy_range = energy_sorted[-1] - energy_sorted[0]

# Check for gap in distribution (simple heuristic: find largest gap)
gaps = np.diff(energy_sorted)
max_gap_idx = np.argmax(gaps)
max_gap = gaps[max_gap_idx]

print(f"\nEnergy distribution analysis:")
print(f"  Range: {energy_sorted[0]:.3f} to {energy_sorted[-1]:.3f} eV")
print(f"  Maximum gap: {max_gap:.3f} eV at index {max_gap_idx}")

# Use broken axis if gap is significant (>10% of range)
use_broken_axis = max_gap > 0.1 * energy_range

if use_broken_axis:
    # Find the gap location
    gap_start = energy_sorted[max_gap_idx]
    gap_end = energy_sorted[max_gap_idx + 1]
    
    # Split data into two groups
    energy_low = energies[energies <= gap_start]
    energy_high = energies[energies >= gap_end]
    
    print(f"  Using broken axis: gap from {gap_start:.3f} to {gap_end:.3f} eV")
    print(f"  Low energy group: {len(energy_low)} structures")
    print(f"  High energy group: {len(energy_high)} structures")
    
    # Create figure with two subplots side by side
    fig1, (ax1_left, ax1_right) = plt.subplots(1, 2, figsize=(4, 4))
    
    # Plot histogram for low energies (left subplot)
    counts_low, _, _ = ax1_left.hist(energy_low, bins=30, color='mediumaquamarine', 
                                      alpha=0.8, edgecolor='black', linewidth=0.25)
    
    # Plot histogram for high energies (right subplot)
    counts_high, _, _ = ax1_right.hist(energy_high, bins=30, color='mediumaquamarine', 
                                        alpha=0.8, edgecolor='black', linewidth=0.25)
    
    # Make y-axes the same scale
    max_count = max(counts_low.max(), counts_high.max())
    ax1_left.set_ylim(0, max_count * 1.1)
    ax1_right.set_ylim(0, max_count * 1.1)
    
    # Set x-limits with some padding
    ax1_left.set_xlim(energy_low.min() - 1, gap_start + 1)
    ax1_right.set_xlim(gap_end - 1, energy_high.max() + 1)
    
    # ADD LINEAR LOCATOR HERE - RIGHT AFTER set_xlim:
    ax1_left.xaxis.set_major_locator(ticker.LinearLocator(numticks=2))
    ax1_left.yaxis.set_major_locator(ticker.LinearLocator(numticks=4))
    ax1_right.xaxis.set_major_locator(ticker.LinearLocator(numticks=2))

    
    # Hide the spines between the two axes
    ax1_left.spines['right'].set_visible(False)
    ax1_right.spines['left'].set_visible(False)
    
    # Hide y-axis ticks and labels on the right subplot
    ax1_right.yaxis.set_visible(False)
    
    # Add diagonal break lines
    d = 0.035
    kwargs = dict(transform=ax1_left.transAxes, color='k', clip_on=False, linewidth=1)
    ax1_left.plot((1-d, 1+d), (-d, +d), **kwargs)
    
    kwargs.update(transform=ax1_right.transAxes)
    ax1_right.plot((-d, +d), (-d, +d), **kwargs)
    
    
    # Format y-axis to use scientific notation
    ax1_left.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax1_left.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    # Format x-axis
    ax1_left.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax1_right.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    
    # Rotate x-axis ticks by 45 degrees
    plt.setp(ax1_left.xaxis.get_majorticklabels(), rotation=45)
    plt.setp(ax1_right.xaxis.get_majorticklabels(), rotation=45)
    
    # Add a centered x-label for the entire figure
    fig1.text(0.6, 0, r'$E$ / eV', ha='center', va='bottom')
    
    # Set labels
    ax1_left.set_ylabel(r'count')
  
    # Adjust layout and save
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.4)
    fig1.savefig('energy_distribution.svg', format='svg', dpi=300, bbox_inches='tight')
    print("\nSaved: energy_distribution.svg")
    plt.show()
    
else:
    # Standard single plot without broken axis
    fig1, ax1 = plt.subplots(figsize=(4, 4))
    
    # Plot histogram
    ax1.hist(energies, bins=50, color='mediumaquamarine', alpha=0.8, edgecolor='black', linewidth=0.25)
    
    # Set labels
    ax1.set_xlabel(r'$E$ / eV')
    ax1.set_ylabel(r'count')
       
    # Save plot
    plt.tight_layout()
    fig1.savefig('energy_distribution.svg', format='svg', dpi=300, bbox_inches='tight')
    print("\nSaved: energy_distribution.svg")
    plt.show()

# Statistics
print(f"\nEnergy statistics:")
print(f"  Range: {energies.min():.3f} to {energies.max():.3f} eV")
print(f"  Mean: {energies.mean():.3f} eV")
print(f"  Std: {energies.std():.3f} eV")

#==============================================================================
# PLOT 2: FORCES DISTRIBUTION
#==============================================================================
fig2, ax2 = plt.subplots(figsize=(4, 4))

# Plot histogram
ax2.hist(forces, bins=50, color='#F3B6A5', alpha=0.8, edgecolor='black', linewidth=0.25)

# Set labels
ax2.set_xlabel(r'$\vec{f}$ / eV.Å$^{-1}$')
ax2.set_ylabel(r'count')

# Format y-axis to use scientific notation
ax2.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax2.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

# Set limits
ax2.set_xlim(forces.min(), forces.max())
ax2.xaxis.set_major_locator(ticker.LinearLocator(numticks=4))
ax2.tick_params(axis='x', rotation=45)
ax2.yaxis.set_major_locator(ticker.LinearLocator(numticks=4))

# Statistics
print(f"\nForce statistics:")
print(f"  Range: {forces.min():.3f} to {forces.max():.3f} eV/Å")
print(f"  Mean: {forces.mean():.3f} eV/Å")
print(f"  Std: {forces.std():.3f} eV/Å")

# Save plot
plt.tight_layout()
fig2.savefig('forces_distribution.svg', format='svg', dpi=300, bbox_inches='tight')
print("Saved: forces_distribution.svg")
plt.show()

#==============================================================================
# PLOT 3: STRESS TENSOR DISTRIBUTION
#==============================================================================
fig3, ax3 = plt.subplots(figsize=(4, 4))

# Plot histogram
ax3.hist(stresses, bins=50, color='violet', alpha=0.8, edgecolor='black', linewidth=0.25)

# Set labels
ax3.set_xlabel(r'$\boldsymbol{s}$ / eV.Å$^{-3}$')
ax3.set_ylabel(r'count')

# Format y-axis to use scientific notation
ax3.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
ax3.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax3.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))

# Set limits
ax3.set_xlim(stresses.min(), stresses.max())
ax3.xaxis.set_major_locator(ticker.LinearLocator(numticks=4))
ax3.tick_params(axis='x', rotation=45)
ax3.yaxis.set_major_locator(ticker.LinearLocator(numticks=4))

# Statistics
print(f"\nStress statistics:")
print(f"  Range: {stresses.min():.6f} to {stresses.max():.6f} eV/Å³")
print(f"  Mean: {stresses.mean():.6f} eV/Å³")
print(f"  Std: {stresses.std():.6f} eV/Å³")

# Save plot
plt.tight_layout()
fig3.savefig('stress_distribution.svg', format='svg', dpi=300, bbox_inches='tight')
print("Saved: stress_distribution.svg")
plt.show()

print("\n=== All plots saved successfully ===")
print("Files created:")
print("  - energy_distribution.svg")
print("  - forces_distribution.svg")
print("  - stress_distribution.svg")
