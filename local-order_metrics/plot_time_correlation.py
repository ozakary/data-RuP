#!/usr/bin/env python3
"""
Plot time correlation functions and extract characteristic times.
Creates 3 separate plots: correlation curves, relaxation times, oscillation periods.
"""

import re
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

from matplotlib.ticker import LinearLocator, FormatStrFormatter

# Import figure formatting if available
try:
    import figure_formatting_v2 as ff
    ff.set_rcParams(ff.master_formatting)
except ImportError:
    print("Figure formatting module not found. Using default matplotlib settings.")


# ============================================================================
# Core Analysis Functions
# ============================================================================

def tail_baseline(y, last_frac=0.05, min_tail=5):
    """Estimate baseline from tail of correlation function."""
    n = len(y)
    m = max(min_tail, int(last_frac * n))
    base = float(np.mean(y[-m:]))
    if abs(base) < 5e-5:
        base = 0
    return base


def zero_crossings(z):
    """Find zero crossing indices."""
    s = np.sign(z)
    idx = []
    for k in range(1, len(z)):
        if s[k] == 0 or s[k] != s[k-1]:
            idx.append(k)
    return np.array(idx, int)


def period_from_crossings(t, zc_idx, average_over=20):
    """Estimate period from zero crossings."""
    if len(zc_idx) < 3:
        return np.nan
    
    full = []
    for k in range(len(zc_idx) - 2):
        full.append(t[zc_idx[k+2]] - t[zc_idx[k]])
    
    if not full:
        return np.nan
    
    full = np.array(full[:average_over], float)
    T = float(np.mean(full))
    
    return T if T > 0 else np.nan


def envelope_tau_from_peaks(y, dt, baseline=None, max_peaks=10, min_pts=6):
    """Estimate relaxation time from envelope of peaks."""
    if baseline is None:
        baseline = tail_baseline(y)
    
    z = y - baseline
    a = np.abs(z)
    
    # Find peaks
    pk = [i for i in range(1, len(a)-1) if a[i] > a[i-1] and a[i] > a[i+1]]
    
    if len(pk) < min_pts:
        return np.nan
    
    pk = pk[:max_peaks]
    t = np.array(pk) * dt
    amp = a[pk]
    
    mask = amp > 0
    if mask.sum() < min_pts:
        return np.nan
    
    t, amp = t[mask], amp[mask]
    lnA = np.log(amp)
    
    # Linear fit: lnA = lnA0 - t/τ
    A = np.vstack([np.ones_like(t), t]).T
    (lnA0, m), *_ = np.linalg.lstsq(A, lnA, rcond=None)
    
    tau = -1.0 / m if m != 0 else np.nan
    
    return float(tau) if np.isfinite(tau) and tau > 0 else np.nan


# ============================================================================
# Data Loading
# ============================================================================

def load_correlation_data(input_dir, file_prefix='time_corr', selected_temps=None):
    """Load time correlation data from individual NPZ files."""
    input_dir = Path(input_dir)
    
    if not input_dir.exists():
        print(f"✗ Input directory not found: {input_dir}")
        sys.exit(1)
    
    print(f"Loading data from: {input_dir.absolute()}")
    print(f"Looking for files: {file_prefix}_*.npz")
    
    pattern = f"{file_prefix}_*.npz"
    npz_files = sorted(input_dir.glob(pattern))
    
    if len(npz_files) == 0:
        print(f"✗ No files matching pattern '{pattern}' found")
        sys.exit(1)
    
    print(f"Found {len(npz_files)} NPZ files")
    
    temperatures = []
    correlations = []
    
    for npz_file in npz_files:
        # Extract temperature from filename
        temp_str = npz_file.stem.replace(f"{file_prefix}_", "")
        
        if selected_temps is not None and temp_str not in selected_temps:
            continue
        
        data = np.load(npz_file, allow_pickle=True)
        
        temp = str(data['temperature'])
        corr = data['correlation']
        
        temperatures.append(temp)
        correlations.append(corr)
        
        print(f"  ✓ Loaded: {npz_file.name} ({temp})")
    
    if len(temperatures) == 0:
        print(f"✗ No temperature data loaded!")
        sys.exit(1)
    
    print(f"\nLoaded {len(temperatures)} temperatures")
    
    return temperatures, correlations


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_time_correlation(temperatures, correlations, args):
    """Plot time correlation curves."""
    fig, ax = plt.subplots(figsize=(7.0, 7.0))
    ax.grid(True, axis="y", which="both", alpha=0.5, linewidth=2.0)
    
    # Sort by temperature
    temp_numeric = np.array([float(t.rstrip('K')) for t in temperatures])
    order = np.argsort(temp_numeric)
    
    nT = len(temperatures)
    colors = plt.cm.coolwarm(np.linspace(0, 1, nT))
    
    for idx, i in enumerate(order):
        temp = temperatures[i]
        y = correlations[i]
        t = np.arange(len(y)) * args.dt
        
        color = colors[idx]
        ax.plot(t, y, linewidth=2.5, color=color, label=temp)
    
    ax.set_xlabel(rf't / {args.unit}')
    ax.set_ylabel(r'$C(t)$')
    ax.set_xlim(args.xmin, args.xmax)
    ax.set_ylim(-0.05, 0.3)
    
    # Set number of ticks
    ax.xaxis.set_major_locator(LinearLocator(numticks=5))
    ax.yaxis.set_major_locator(LinearLocator(numticks=8))
    
    if args.ylim is not None:
        ax.set_ylim(args.ylim)
    
    ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=4)
    
    output_file = f"{args.output_prefix}_correlation.svg"
    plt.savefig(output_file, bbox_inches='tight')
    print(f"\n✓ Saved: {output_file}")
    
    if args.show:
        plt.show()
    plt.close()


def plot_relaxation_time(temperatures, tau_values, args):
    """Plot relaxation time vs temperature."""
    fig, ax = plt.subplots(figsize=(5.25, 5.25))
    #ax.grid(True, axis="both", which="both", alpha=0.25, linewidth=0.8)
    
    # Sort by temperature
    temp_numeric = np.array([float(t.rstrip('K')) for t in temperatures])
    order = np.argsort(temp_numeric)
    
    temps_sorted = temp_numeric[order]
    tau_sorted = np.array(tau_values)[order]
    
    nT = len(temperatures)
    colors = plt.cm.coolwarm(np.linspace(0, 1, nT))
    
    # Line plot
    ax.plot(temps_sorted, tau_sorted, '-o', lw=1.6, ms=10, color='0.2', zorder=2)
    
    # Colored points
    for idx, (T, tau) in enumerate(zip(temps_sorted, tau_sorted)):
        ax.scatter([T], [tau], s=60, color=colors[idx], 
                  edgecolors='white', linewidths=0.6, zorder=3)
    
    ax.set_xlabel(r'$T$ / K')
    ax.set_ylabel(r'$\tau$ / ' + args.unit)
    
    ax.set_xlim(0, 750)
    ax.set_ylim(-100, 2000)
    
    
    # Set number of ticks
    ax.xaxis.set_major_locator(LinearLocator(numticks=4))
    ax.yaxis.set_major_locator(LinearLocator(numticks=6))
    
    output_file = f"{args.output_prefix}_relaxation_time.svg"
    plt.savefig(output_file, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    
    if args.show:
        plt.show()
    plt.close()


def plot_oscillation_period(temperatures, period_values, args):
    """Plot oscillation period vs temperature."""
    fig, ax = plt.subplots(figsize=(5.25, 5.25))
    #ax.grid(True, axis="both", which="both", alpha=0.25, linewidth=0.8)
    
    # Sort by temperature
    temp_numeric = np.array([float(t.rstrip('K')) for t in temperatures])
    order = np.argsort(temp_numeric)
    
    temps_sorted = temp_numeric[order]
    period_sorted = np.array(period_values)[order]
    
    nT = len(temperatures)
    colors = plt.cm.coolwarm(np.linspace(0, 1, nT))
    
    # Line plot
    ax.plot(temps_sorted, period_sorted, '-o', lw=1.6, ms=10, color='0.2', zorder=2)
    
    # Colored points
    for idx, (T, period) in enumerate(zip(temps_sorted, period_sorted)):
        ax.scatter([T], [period], s=60, color=colors[idx],
                  edgecolors='white', linewidths=0.6, zorder=3)
    
    ax.set_xlabel(r'$T$ / K')
    ax.set_ylabel(r'$t_{\mathrm{osc}}$ / ' + args.unit)
    
    
    ax.set_xlim(0, 750)
    ax.set_ylim(0, 3)    


    # Set number of ticks
    ax.xaxis.set_major_locator(LinearLocator(numticks=4))
    ax.yaxis.set_major_locator(LinearLocator(numticks=6))
    
    output_file = f"{args.output_prefix}_oscillation_period.svg"
    plt.savefig(output_file, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    
    if args.show:
        plt.show()
    plt.close()


# ============================================================================
# Main
# ============================================================================

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Plot time correlation functions and characteristic times.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('-i', '--input-dir', required=True, type=str,
                        help='Input directory containing NPZ files')
    
    parser.add_argument('-o', '--output-prefix', required=True, type=str,
                        help='Output prefix for plot files')
    
    parser.add_argument('-prefix', '--file-prefix', default='time_corr', type=str,
                        help='Prefix of NPZ files to load (default: time_corr)')
    
    parser.add_argument('-temps', '--temperatures', nargs='+', default=None,
                        help='Specific temperatures to plot')
    
    parser.add_argument('--dt', type=float, default=0.1, 
                        help='Time step (default: 0.1)')
    
    parser.add_argument('--unit', default='ps', type=str,
                        help='Time unit (default: ps)')
    
    parser.add_argument('--xmin', type=float, default=0,
                        help='X-axis minimum (default: 0)')
    
    parser.add_argument('--xmax', type=float, default=None,
                        help='X-axis maximum (default: auto)')
    
    parser.add_argument('--ylim', nargs=2, type=float, default=None,
                        help='Y-axis limits for correlation plot')
    
    parser.add_argument('--baseline-frac', type=float, default=0.10,
                        help='Fraction of tail for baseline (default: 0.10)')
    
    parser.add_argument('--no-baseline', action='store_true',
                        help='Do not subtract baseline')
    
    parser.add_argument('--show', action='store_true',
                        help='Show plots interactively')
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()
    
    print("="*70)
    print("TIME CORRELATION PLOTTING")
    print("="*70)
    
    # Load data
    temperatures, correlations = load_correlation_data(
        args.input_dir, args.file_prefix, args.temperatures
    )
    
    # Compute characteristic times
    print("\nComputing characteristic times...")
    print(f"Baseline subtraction: {'No' if args.no_baseline else f'Yes (tail fraction={args.baseline_frac})'}")
    
    tau_values = []
    period_values = []
    
    for temp, y in zip(temperatures, correlations):
        dt = args.dt
        t = np.arange(len(y)) * dt
        
        # Compute baseline
        if args.no_baseline:
            base = 0.0
            z = y
        else:
            base = tail_baseline(y, last_frac=args.baseline_frac, min_tail=5)
            z = y - base
        
        print(f"  {temp}: baseline={base:.6f}")
        
        # Compute relaxation time (pass baseline=0 since already subtracted)
        tau = envelope_tau_from_peaks(z, dt, baseline=0.0)
        tau_values.append(tau)
        
        # Compute oscillation period
        zc_idx = zero_crossings(z)
        T_est = period_from_crossings(t, zc_idx, average_over=20)
        period_values.append(T_est)
        
        print(f"    τ={tau:.2f} {args.unit}, T_osc={T_est:.2f} {args.unit}")
    
    # Create plots
    print("\nGenerating plots...")
    plot_time_correlation(temperatures, correlations, args)
    plot_relaxation_time(temperatures, tau_values, args)
    plot_oscillation_period(temperatures, period_values, args)
    
    print("\n✓ All plots saved!")
    print("="*70)


if __name__ == "__main__":
    main()
