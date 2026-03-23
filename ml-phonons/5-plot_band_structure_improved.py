#!/usr/bin/env python3
"""
Comprehensive phonon analysis plotting from saved data
Includes: Band structure, DOS, and projected DOS
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
import os

from matplotlib.ticker import LinearLocator, FormatStrFormatter

# Try to import custom formatting
try:
    import figure_formatting_v2 as ff
    ff.set_rcParams(ff.master_formatting)
    print("Using custom figure formatting")
except ImportError:
    print("Using default matplotlib settings")
    # Set some reasonable defaults
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.linewidth'] = 1.0


def calculate_dos(frequencies, sigma=0.05, energy_range=None, n_points=1000):
    """
    Calculate phonon density of states using Gaussian broadening
    
    Parameters:
    -----------
    frequencies : array, shape (n_qpoints, n_bands)
        Phonon frequencies in THz
    sigma : float
        Gaussian broadening parameter in THz
    energy_range : tuple or None
        (min, max) energy range in THz. If None, auto-determined
    n_points : int
        Number of points for DOS energy grid
    
    Returns:
    --------
    energies : array
        Energy grid
    dos : array
        Density of states
    """
    # Flatten all frequencies
    all_freqs = frequencies.flatten()
    
    # Determine energy range
    if energy_range is None:
        freq_min = np.min(all_freqs)
        freq_max = np.max(all_freqs)
        # Add some padding
        padding = (freq_max - freq_min) * 0.05
        energy_range = (freq_min - padding, freq_max + padding)
    
    energies = np.linspace(energy_range[0], energy_range[1], n_points)
    dos = np.zeros(n_points)
    
    # Gaussian broadening
    print(f"    Calculating DOS with sigma={sigma} THz...")
    for freq in tqdm(all_freqs, desc="    Broadening", unit="mode", leave=False):
        dos += np.exp(-((energies - freq) ** 2) / (2 * sigma ** 2))
    
    # Normalize
    dos /= (sigma * np.sqrt(2 * np.pi))
    dos /= frequencies.shape[0]  # Normalize by number of q-points
    
    return energies, dos


def plot_band_and_dos(temp_dir, temp_value, y_limit=None, dos_sigma=0.05, 
                      recalculate_dos=False, color='blue'):
    """
    Create plots with band structure and DOS - with caching for fast iteration
    
    Parameters:
    -----------
    temp_dir : str
        Temperature directory
    temp_value : int
        Temperature value
    y_limit : tuple or None
        (ymin, ymax) for frequency axis
    dos_sigma : float
        Gaussian broadening for DOS calculation (THz)
    recalculate_dos : bool
        Force recalculation of DOS even if cache exists
    color : str or tuple
        Color for bands and DOS (matplotlib color)
    """
    print(f"\nProcessing {temp_dir} ({temp_value}K)...")
    
    # Load band structure data
    data_file = f"./{temp_dir}/band_data_{temp_value}K.npz"
    if not os.path.exists(data_file):
        print(f"  ERROR: Data file not found: {data_file}")
        return
    
    data = np.load(data_file)
    distances = data['distances']
    frequencies = data['frequencies']
    special_points = data['special_points']
    special_labels = data['special_labels']
    
    n_qpoints, n_bands = frequencies.shape
    print(f"  Loaded {n_qpoints} q-points with {n_bands} bands")
    
    # Cache file for DOS
    dos_cache_file = f"./{temp_dir}/dos_cache_{temp_value}K_sigma{dos_sigma}.npz"
    
    # Calculate or load DOS
    if os.path.exists(dos_cache_file) and not recalculate_dos:
        print(f"  Loading DOS from cache: {dos_cache_file}")
        dos_data = np.load(dos_cache_file)
        dos_energies = dos_data['dos_energies']
        dos = dos_data['dos']
    else:
        print("  Calculating DOS (this will be cached)...")
        dos_energies, dos = calculate_dos(frequencies, sigma=dos_sigma)
        
        # Save to cache
        np.savez(dos_cache_file,
                 dos_energies=dos_energies,
                 dos=dos)
        print(f"  ✓ DOS cached to {dos_cache_file}")
    
    # Auto-determine y-limits if not provided
    if y_limit is None:
        freq_min = np.min(frequencies)
        freq_max = np.max(frequencies)
        y_limit = (freq_min - 0.5, min(freq_max, 10))
    
    # ============================================
    # PLOT 1: Band structure + Total DOS
    # ============================================
    print("  Creating band structure + DOS plot...")
    fig = plt.figure(figsize=(4.25, 3.75))
    gs = GridSpec(1, 2, width_ratios=[3, 1], wspace=0.15)
    
    # Band structure panel
    ax_band = fig.add_subplot(gs[0])
    
    for ib in range(n_bands):
        ax_band.plot(distances, frequencies[:, ib], '-', 
                    linewidth=0.1, alpha=0.75, rasterized=True, color=color)
    
    ax_band.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.75)
    ax_band.set_ylabel('Frequency / THz')
    ax_band.set_xlabel('')
    
    # Set number of ticks
    ax_band.yaxis.set_major_locator(LinearLocator(numticks=6))
    
    for xp in special_points:
        ax_band.axvline(x=xp, color='gray', linestyle='-', linewidth=1.5, alpha=0.75)
    
    ax_band.set_xticks(special_points)
    ax_band.set_xticklabels(special_labels)
    ax_band.set_xlim(distances[0], distances[-1])
    ax_band.set_ylim(y_limit)
    
    # DOS panel
    ax_dos = fig.add_subplot(gs[1], sharey=ax_band)
    ax_dos.plot(dos, dos_energies, '-', linewidth=1.5, color=color)
    ax_dos.fill_betweenx(dos_energies, 0, dos, alpha=0.3, color=color)
    ax_dos.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.75)
    ax_dos.set_xlabel('DOS')# / states·THz$^{-1}$')
    ax_dos.set_xlim(0, 2000)
    ax_dos.set_ylim(y_limit)
    ax_dos.tick_params(axis='y', labelleft=False)  # Hide y-tick labels on DOS panel only
    
    ax_dos.yaxis.set_major_locator(LinearLocator(numticks=6))
    
    plt.tight_layout()
    output_file = f"./{temp_dir}/phonon_band_dos_{temp_value}K.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"  ✓ Band + DOS saved to {output_file}")
    
    # ============================================
    # Print statistics
    # ============================================
    print(f"\n  Statistics:")
    print(f"    Frequency range: {np.min(frequencies):.3f} to {np.max(frequencies):.3f} THz")
    print(f"    Negative frequencies: {np.sum(frequencies < 0)}")
    if np.sum(frequencies < 0) > 0:
        print(f"    Most negative: {np.min(frequencies):.3f} THz")


def compare_temperatures(temp_dict, y_limit=None, dos_sigma=0.05):
    """
    Create comparison plots across multiple temperatures using cached DOS
    
    Parameters:
    -----------
    temp_dict : dict
        Dictionary of {temp_dir: temp_value}
    y_limit : tuple or None
        (ymin, ymax) for frequency axis
    dos_sigma : float
        Gaussian broadening for DOS (THz)
    """
    print("\nCreating temperature comparison plot...")
    
    # Load all DOS data from cache
    all_dos_data = {}
    for temp_dir, temp_value in temp_dict.items():
        dos_cache_file = f"./{temp_dir}/dos_cache_{temp_value}K_sigma{dos_sigma}.npz"
        if os.path.exists(dos_cache_file):
            dos_data = np.load(dos_cache_file)
            all_dos_data[temp_value] = {
                'dos_energies': dos_data['dos_energies'],
                'dos': dos_data['dos']
            }
    
    if len(all_dos_data) < 2:
        print("  Need at least 2 temperatures for comparison")
        return
    
    # Sort by temperature
    temps_sorted = sorted(all_dos_data.keys())
    
    # Auto-determine y-limits from first temperature if not provided
    if y_limit is None:
        first_temp = temps_sorted[0]
        dos_energies = all_dos_data[first_temp]['dos_energies']
        y_limit = (np.min(dos_energies), min(np.max(dos_energies), 10))
    
    # ============================================
    # PLOT: DOS comparison across temperatures
    # ============================================
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Color map for temperatures
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(temps_sorted)))
    
    for i, temp in enumerate(temps_sorted):
        dos_energies = all_dos_data[temp]['dos_energies']
        dos = all_dos_data[temp]['dos']
        
        ax.plot(dos, dos_energies, linewidth=1.5, 
                label=f'{temp} K', color=colors[i], alpha=0.8)
    
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('DOS')# / states·THz$^{-1}$')
    ax.set_ylabel('Frequency / THz')
    ax.set_ylim(y_limit)
    ax.legend(loc='best', frameon=False)#, fontsize=8)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    output_file = "./phonon_dos_temperature_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"  ✓ Temperature comparison saved to {output_file}")


if __name__ == "__main__":
    # =================================================================
    # CONFIGURATION - Edit these
    # =================================================================
    
    temperatures = {
        '50K': 50,
        '100K': 100,
        '110K': 110,
        '120K': 120,
        '130K': 130,
        '140K': 140,
        '150K': 150,
        '160K': 160,
        '170K': 170,
        '180K': 180,
        '190K': 190,
        '200K': 200,
        '250K': 250,
        '260K': 260,
        '270K': 270,
        '280K': 280,
        '290K': 290,
        '300K': 300,
        '310K': 310,
        '320K': 320,
        '330K': 330,
        '340K': 340,
        '350K': 350,
        '400K': 400,
        '450K': 450,
        '500K': 500,
        '550K': 550,                
        '600K': 600,
        '650K': 650,        
        '700K': 700,        
    }
    
    # Plot settings - change these freely, plotting is fast after first run!
    y_limit = (-4.0, 16.0)          # Frequency range (THz)
    dos_sigma = 0.05           # DOS broadening (THz)
    recalculate_dos = False    # Set True to force recalculation
    
    # =================================================================
    
    print("="*60)
    print("PHONON BAND STRUCTURE + DOS ANALYSIS (with caching)")
    print("="*60)
    print(f"Processing {len(temperatures)} temperatures")
    print(f"DOS will be cached. After first run, plotting is instant!")
    print(f"To recalculate DOS: set recalculate_dos=True")
    print("="*60)
    
    # Create color map for temperatures (coolwarm)
    temps_sorted = sorted(temperatures.values())
    cmap = plt.cm.coolwarm
    colors = [cmap(i / (len(temps_sorted) - 1)) for i in range(len(temps_sorted))]
    
    # Map each temperature to its color
    temp_colors = {}
    for i, temp_val in enumerate(temps_sorted):
        temp_colors[temp_val] = colors[i]
    
    for temp_dir, temp_value in temperatures.items():
        try:
            plot_band_and_dos(
                temp_dir, 
                temp_value,
                y_limit=y_limit,
                dos_sigma=dos_sigma,
                recalculate_dos=recalculate_dos,
                color=temp_colors[temp_value]
            )
        except Exception as e:
            print(f"\n  ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    if len(temperatures) > 1:
        try:
            compare_temperatures(temperatures, y_limit=y_limit, dos_sigma=dos_sigma)
        except Exception as e:
            print(f"\n  ERROR in comparison: {e}")
    
    print("\n" + "="*60)
    print("COMPLETED!")
    print("="*60)
    print("\nOutput files per temperature:")
    print("  - phonon_band_dos_[T]K.png  (band structure + total DOS)")
    print("    Colors: coolwarm colormap (blue=cold, red=hot)")
    print("\nComparison file (multi-temperature):")
    print("  - phonon_dos_temperature_comparison.png  (total DOS overlay)")
    print("\nCache files (in each temperature directory):")
    print("  - dos_cache_[T]K_sigma[value].npz  (DOS data for fast re-plotting)")
