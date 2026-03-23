#!/usr/bin/env python3
"""
Fix population tracking for trimer angle analysis:
1. Identify 4 populations (not 3) at reference temperature
2. Track populations smoothly across temperatures using mode matching
3. Compute mode (not mean) for each population
"""

import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import os
import glob

def find_populations_at_reference(csv_file, n_populations=4):
    df = pd.read_csv(csv_file)
    all_angles = df['trimer_angle'].values
    atom_angles = df.groupby('ru_atom_id')['trimer_angle'].mean()

    print(f"Analyzing {len(all_angles):,} measurements")
    print(f"Angle range: {all_angles.min():.2f}° to {all_angles.max():.2f}°")

    n_bins = 200
    hist, bin_edges = np.histogram(all_angles, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    hist_smooth = gaussian_filter1d(hist.astype(float), sigma=3)

    # --- True peaks ---
    peaks, _ = find_peaks(hist_smooth,
                          prominence=hist_smooth.max() * 0.05,
                          height=hist_smooth.max() * 0.05)

    # --- Shoulder detection via 2nd derivative zero-crossings ---
    d1 = np.gradient(hist_smooth)
    d2 = np.gradient(d1)
    inflection_indices = np.where(np.diff(np.sign(d2)))[0]

    shoulder_candidates = []
    for idx in inflection_indices:
        # A shoulder: inflection point where d1 doesn't change sign (no true valley/peak nearby)
        window = 15
        local_d1 = d1[max(0, idx - window):min(len(d1), idx + window)]
        is_monotonic = np.all(local_d1 >= -0.5) or np.all(local_d1 <= 0.5)
        above_threshold = hist_smooth[idx] > hist_smooth.max() * 0.1
        not_near_peak = all(abs(idx - p) > 10 for p in peaks)

        if is_monotonic and above_threshold and not_near_peak:
            shoulder_candidates.append(idx)

    print(f"\nFound {len(peaks)} true peaks and {len(shoulder_candidates)} shoulder(s)")

    # Combine and sort all candidate peak positions
    all_peak_indices = sorted(set(list(peaks) + shoulder_candidates))

    # Select top n_populations by histogram height
    all_peak_heights = hist_smooth[all_peak_indices]
    top_indices = np.argsort(all_peak_heights)[-n_populations:]
    top_peaks = sorted([all_peak_indices[i] for i in top_indices])
    peak_positions = bin_centers[top_peaks]

    print(f"\nSelected {n_populations} features (peaks + shoulders):")
    for i, pos in enumerate(peak_positions):
        label = "shoulder" if top_peaks[i] in shoulder_candidates else "peak"
        print(f"  Population {i+1}: {pos:.2f}°  [{label}]")

    # --- Valleys between selected features ---
    valleys = []
    for i in range(len(top_peaks) - 1):
        segment = hist_smooth[top_peaks[i]:top_peaks[i+1]]
        valley_idx = top_peaks[i] + np.argmin(segment)
        valleys.append(bin_centers[valley_idx])

    print(f"\nBoundaries at: {', '.join([f'{v:.2f}°' for v in valleys])}")

    # Build ranges
    ranges = []
    ranges.append((all_angles.min() - 1, valleys[0]))
    for i in range(len(valleys) - 1):
        ranges.append((valleys[i], valleys[i+1]))
    ranges.append((valleys[-1], all_angles.max() + 1))

    # Atom assignment
    population_assignment = {}
    for ru_id, angle in atom_angles.items():
        for pop_idx, (low, high) in enumerate(ranges, 1):
            if low <= angle <= high:
                population_assignment[ru_id] = pop_idx
                break

    print(f"\nPopulation assignment:")
    for pop in range(1, n_populations + 1):
        count = sum(1 for p in population_assignment.values() if p == pop)
        print(f"  Pop {pop}: {count} atoms ({100*count/len(population_assignment):.1f}%)")

    # Diagnostic plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    ax = axes[0]
    ax.bar(bin_centers, hist, width=(bin_centers[1]-bin_centers[0])*0.8,
           alpha=0.5, color='gray', edgecolor='black')
    ax.plot(bin_centers, hist_smooth, 'b-', linewidth=2.5, label='Smoothed')
    colors = ['red', 'orange', 'green', 'purple']
    for i, (pos, pidx) in enumerate(zip(peak_positions, top_peaks)):
        label = "shoulder" if pidx in shoulder_candidates else "peak"
        ax.axvline(pos, color=colors[i], linestyle='--', linewidth=2.5,
                   label=f'Pop {i+1}: {pos:.2f}° [{label}]')
    for valley in valleys:
        ax.axvline(valley, color='blue', linestyle=':', linewidth=2)
    ax.set_xlabel('Trimer Angle (°)', fontsize=13)
    ax.set_ylabel('Frequency', fontsize=13)
    ax.set_title('Histogram with Peak + Shoulder Detection', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # 2nd derivative panel — shows where shoulders are
    ax2 = axes[1]
    ax2.plot(bin_centers, d2, 'k-', linewidth=1.5, label='2nd derivative')
    ax2.axhline(0, color='gray', linestyle='--')
    for idx in shoulder_candidates:
        ax2.axvline(bin_centers[idx], color='orange', linestyle='--',
                    linewidth=2, label=f'Shoulder at {bin_centers[idx]:.2f}°')
    ax2.set_xlabel('Trimer Angle (°)', fontsize=13)
    ax2.set_ylabel('d²(hist)/dx²', fontsize=13)
    ax2.set_title('2nd Derivative — Zero crossings reveal shoulders', fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('population_diagnostic_4pop.png', dpi=200)
    plt.close()
    print(f"\nSaved diagnostic plot to population_diagnostic_4pop.png")

    return population_assignment, ranges, peak_positions

def compute_mode_for_population(angles):
    """Compute mode (peak of distribution) using KDE"""
    if len(angles) < 50:
        return np.median(angles), angles.std()
    
    kde = gaussian_kde(angles, bw_method='scott')
    x_min, x_max = angles.min(), angles.max()
    x_grid = np.linspace(x_min, x_max, 500)
    kde_values = kde(x_grid)
    mode = x_grid[np.argmax(kde_values)]
    
    return mode, angles.std()

def match_populations_to_previous(current_modes, previous_modes, n_populations):
    """
    Match current population modes to previous temperature's modes.
    Returns mapping: current_pop_idx -> previous_pop_idx
    """
    mapping = {}
    used = set()
    
    # For each current population, find closest previous population
    for curr_idx in range(n_populations):
        best_match = None
        best_distance = float('inf')
        
        for prev_idx in range(n_populations):
            if prev_idx in used:
                continue
            distance = abs(current_modes[curr_idx] - previous_modes[prev_idx])
            if distance < best_distance:
                best_distance = distance
                best_match = prev_idx
        
        mapping[curr_idx] = best_match
        used.add(best_match)
    
    return mapping

def process_all_temperatures(temperatures, angle_ranges, n_populations=4):
    """
    Process all temperatures by computing mode within each angle range.
    Does NOT use fixed population assignment - computes fresh at each temperature.
    
    Args:
        temperatures: List of temperatures to process
        angle_ranges: List of (min, max) tuples defining angle ranges for each population
        n_populations: Number of populations
    """
    print(f"\n{'='*60}")
    print(f"Processing all temperatures with angle-range-based mode tracking")
    print(f"{'='*60}")
    
    print(f"\nUsing fixed angle ranges from 10K:")
    for i, (low, high) in enumerate(angle_ranges, 1):
        print(f"  Pop {i}: {low:.2f}° to {high:.2f}°")
    
    all_stats = []
    previous_modes = None
    
    for temp in temperatures:
        csv_file = f'./trimer_angle_{temp}K.csv'
        if not os.path.exists(csv_file):
            print(f"Warning: {csv_file} not found")
            continue
        
        print(f"\nProcessing {temp}K...")
        df = pd.read_csv(csv_file)
        all_angles = df['trimer_angle'].values
        
        # Compute mode for each angle range (NOT based on atom assignment!)
        current_modes = []
        current_stds = []
        current_counts = []
        
        for pop_idx, (low, high) in enumerate(angle_ranges, 1):
            # Get all measurements in this angle range
            mask = (all_angles >= low) & (all_angles <= high)
            pop_angles = all_angles[mask]
            
            if len(pop_angles) > 50:
                mode, std = compute_mode_for_population(pop_angles)
                current_modes.append(mode)
                current_stds.append(std)
                current_counts.append(len(pop_angles))
            else:
                current_modes.append(np.nan)
                current_stds.append(np.nan)
                current_counts.append(0)
        
        # Match to previous temperature to prevent crossing
        if previous_modes is not None:
            # Check if any modes have crossed
            mapping = match_populations_to_previous(current_modes, previous_modes, n_populations)
            
            # Reorder if needed
            if mapping != {i: i for i in range(n_populations)}:
                print(f"  Detected mode crossing, reordering: {mapping}")
                current_modes_reordered = [current_modes[k] for k, v in sorted(mapping.items(), key=lambda x: x[1])]
                current_stds_reordered = [current_stds[k] for k, v in sorted(mapping.items(), key=lambda x: x[1])]
                current_counts_reordered = [current_counts[k] for k, v in sorted(mapping.items(), key=lambda x: x[1])]
                
                # Also reorder the angle ranges for future temperatures
                angle_ranges_reordered = [angle_ranges[k] for k, v in sorted(mapping.items(), key=lambda x: x[1])]
                angle_ranges = angle_ranges_reordered
                
                current_modes = current_modes_reordered
                current_stds = current_stds_reordered
                current_counts = current_counts_reordered
        
        # Store stats
        stats = {'temperature': temp}
        for i in range(n_populations):
            stats[f'pop{i+1}_mode'] = current_modes[i]
            stats[f'pop{i+1}_std'] = current_stds[i]
            stats[f'pop{i+1}_count'] = current_counts[i]
        stats['total_measurements'] = len(all_angles)
        all_stats.append(stats)
        
        # Print summary
        for i in range(n_populations):
            print(f"  Pop {i+1}: mode={current_modes[i]:.2f}° ± {current_stds[i]:.2f}° ({current_counts[i]:,} measurements)")
        
        previous_modes = current_modes
    
    return pd.DataFrame(all_stats)

def main():
    # Find all temperature files
    csv_files = sorted(glob.glob('./trimer_angle_*K.csv'))
    temperatures = [int(f.split('_')[-1].replace('K.csv', '')) for f in csv_files]
    temperatures = sorted(temperatures)
    
    print(f"Found {len(temperatures)} temperature files: {temperatures}")
    
    # Use 50K as reference
    ref_csv = './trimer_angle_50K.csv'
    
    # Find 4 populations at reference temperature
    population_assignment, angle_ranges, peak_positions = find_populations_at_reference(ref_csv, n_populations=4)
    
    # Save population assignment (just for reference/diagnostics)
    pop_df = pd.DataFrame([
        {'ru_atom_id': ru_id, 'population': pop}
        for ru_id, pop in population_assignment.items()
    ])
    pop_df.to_csv('trimer_angle_population_assignment_4pop.csv', index=False)
    print(f"\nSaved: trimer_angle_population_assignment_4pop.csv (for reference only)")
    
    # Process all temperatures using ANGLE RANGES (not fixed atom assignment!)
    print(f"\n{'='*60}")
    print("IMPORTANT: Computing modes from angle ranges at each temperature")
    print("NOT using fixed atom assignments!")
    print(f"{'='*60}")
    stats_df = process_all_temperatures(temperatures, angle_ranges, n_populations=4)
    
    # Save statistics
    stats_df.to_csv('trimer_angle_population_statistics_4pop.csv', index=False)
    print(f"\nSaved: trimer_angle_population_statistics_4pop.csv")
    
    print(f"\n{'='*60}")
    print("Complete! Use these files with the plotting code.")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
