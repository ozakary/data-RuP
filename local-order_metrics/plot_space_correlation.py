#!/usr/bin/env python3
"""
Plot space correlation functions and extract correlation length.
Creates 2 separate plots: correlation curves and correlation length vs temperature.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import curve_fit

from matplotlib.ticker import LinearLocator, FormatStrFormatter

# Import figure formatting if available
try:
    import figure_formatting_v2 as ff
    ff.set_rcParams(ff.master_formatting)
except ImportError:
    print("Figure formatting module not found. Using default matplotlib settings.")


# ============================================================================
# Analysis Functions
# ============================================================================

def smooth_correlation(y, method='savgol', window_frac=0.25):
    """
    Smooth correlation function.
    
    Parameters:
    -----------
    y : ndarray
        Correlation function
    method : str
        Smoothing method ('savgol' or 'gaussian')
    window_frac : float
        Fraction of data length for window size
    
    Returns:
    --------
    y_smooth : ndarray
        Smoothed correlation
    """
    if method == 'savgol':
        window = max(5, int(len(y) * window_frac))
        if window % 2 == 0:
            window += 1
        window = min(window, len(y) - 1)
        
        y_smooth = savgol_filter(y, window, polyorder=2)
    else:
        from scipy.ndimage import gaussian_filter1d
        sigma = len(y) * window_frac / 4
        y_smooth = gaussian_filter1d(y, sigma=sigma)
    
    return y_smooth


def exponential_decay(r, xi, A, C):
    """Exponential decay function: C(r) = A * exp(-r/xi) + C"""
    return A * np.exp(-r / xi) + C


def extract_correlation_length(distances, correlation, method='envelope', threshold=0.1):
    """
    Extract correlation length from space correlation function.
    
    Parameters:
    -----------
    distances : ndarray
        Distance array
    correlation : ndarray
        Correlation function
    method : str
        Method to extract correlation length
        'envelope': fit exponential to envelope of peaks
        'direct': fit exponential to smoothed data
        'first_zero': distance where correlation first crosses zero (for oscillatory data)
        'decay_threshold': distance where |C(r)| < threshold * C(0)
    threshold : float
        Threshold fraction for decay_threshold method (default: 0.1 = 10%)
    
    Returns:
    --------
    xi : float
        Correlation length
    """
    # Smooth the correlation
    corr_smooth = smooth_correlation(correlation, method='savgol', window_frac=0.25)
    
    if method == 'first_zero':
        # Find first zero crossing (good for oscillatory data)
        for i in range(1, len(corr_smooth)):
            if corr_smooth[i] <= 0 and corr_smooth[0] > 0:
                return distances[i]
        return np.nan
    
    elif method == 'decay_threshold':
        # Find where correlation decays to threshold fraction of initial value
        threshold_value = threshold * abs(corr_smooth[0])
        for i in range(1, len(corr_smooth)):
            if abs(corr_smooth[i]) < threshold_value:
                return distances[i]
        return np.nan
    
    elif method == 'envelope':
        # Find peaks
        peaks, properties = find_peaks(corr_smooth, prominence=0.001, height=0.001)
        
        if len(peaks) < 3:
            # Fall back to decay threshold
            threshold = 0.1 * abs(corr_smooth[0])
            for i in range(1, len(corr_smooth)):
                if abs(corr_smooth[i]) < threshold:
                    return distances[i]
            return np.nan
        
        # Use first several peaks for fit
        n_peaks = min(10, len(peaks))
        r_peaks = distances[peaks[:n_peaks]]
        c_peaks = corr_smooth[peaks[:n_peaks]]
        
        # Fit exponential envelope
        try:
            # Initial guess
            xi_guess = r_peaks[-1] / 2
            A_guess = c_peaks[0]
            C_guess = 0.0
            
            popt, _ = curve_fit(
                exponential_decay, r_peaks, c_peaks,
                p0=[xi_guess, A_guess, C_guess],
                bounds=([0, 0, -1], [np.inf, np.inf, 1]),
                maxfev=5000
            )
            
            xi = popt[0]
            
        except:
            # Fall back to decay threshold
            threshold = 0.1 * abs(corr_smooth[0])
            for i in range(1, len(corr_smooth)):
                if abs(corr_smooth[i]) < threshold:
                    return distances[i]
            xi = np.nan
    
    else:  # direct fit
        # Use data up to where correlation becomes small
        mask = (corr_smooth > 0.01) & (distances < 20)
        
        if mask.sum() < 5:
            return np.nan
        
        r_fit = distances[mask]
        c_fit = corr_smooth[mask]
        
        try:
            xi_guess = r_fit[-1] / 2
            popt, _ = curve_fit(
                exponential_decay, r_fit, c_fit,
                p0=[xi_guess, c_fit[0], 0.0],
                bounds=([0, 0, -1], [np.inf, np.inf, 1]),
                maxfev=5000
            )
            
            xi = popt[0]
            
        except:
            xi = np.nan
    
    return xi if np.isfinite(xi) and xi > 0 else np.nan


# ============================================================================
# Data Loading
# ============================================================================

def load_correlation_data(input_dir, file_prefix='space_corr', selected_temps=None):
    """Load space correlation data from individual NPZ files."""
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
    distances_list = []
    
    for npz_file in npz_files:
        # Extract temperature from filename
        temp_str = npz_file.stem.replace(f"{file_prefix}_", "")
        
        if selected_temps is not None and temp_str not in selected_temps:
            continue
        
        data = np.load(npz_file, allow_pickle=True)
        
        temp = str(data['temperature'])
        corr = data['correlation']
        distances = data['distances']
        
        temperatures.append(temp)
        correlations.append(corr)
        distances_list.append(distances)
        
        print(f"  ✓ Loaded: {npz_file.name} ({temp})")
    
    if len(temperatures) == 0:
        print(f"✗ No temperature data loaded!")
        sys.exit(1)
    
    print(f"\nLoaded {len(temperatures)} temperatures")
    
    return temperatures, correlations, distances_list


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_space_correlation(temperatures, correlations, distances_list, args):
    """Plot space correlation curves."""
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
        r = distances_list[i]
        
        # Smooth for plotting
        if args.smooth:
            y = smooth_correlation(y, method='savgol', window_frac=0.25)
        
        color = colors[idx]
        ax.plot(r, y, linewidth=2.5, color=color, label=temp)
    
    ax.set_xlabel(r'$r$ / Å')
    ax.set_ylabel(r'$C_{\bot}(r)$') # perp = \bot and para = \parallel
    ax.set_xlim(args.xmin, args.xmax)
    ax.set_ylim(-0.025, 0.15)
    
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


def plot_correlation_length(temperatures, xi_values, args):
    """Plot correlation length vs temperature."""
    fig, ax = plt.subplots(figsize=(5.25, 5.25))
    #ax.grid(True, axis="both", which="both", alpha=0.25, linewidth=0.8)
    
    # Sort by temperature
    temp_numeric = np.array([float(t.rstrip('K')) for t in temperatures])
    order = np.argsort(temp_numeric)
    
    temps_sorted = temp_numeric[order]
    xi_sorted = np.array(xi_values)[order]
    
    nT = len(temperatures)
    colors = plt.cm.coolwarm(np.linspace(0, 1, nT))
    
    # Line plot
    ax.plot(temps_sorted, xi_sorted, '-o', lw=1.6, ms=10, color='0.2', zorder=2)
    
    # Colored points
    for idx, (T, xi) in enumerate(zip(temps_sorted, xi_sorted)):
        ax.scatter([T], [xi], s=60, color=colors[idx],
                  edgecolors='white', linewidths=0.6, zorder=3)
    
    ax.set_xlabel(r'$T$ / K')
    ax.set_ylabel(r'$\xi_{\bot}$ / Å')  # perp = \bot and para = \parallel
    
    ax.set_xlim(0, 750)
    ax.set_ylim(0.0, 110.0)
    
    
    # Set number of ticks
    ax.xaxis.set_major_locator(LinearLocator(numticks=4))
    ax.yaxis.set_major_locator(LinearLocator(numticks=6))
    
    output_file = f"{args.output_prefix}_correlation_length.svg"
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
        description='Plot space correlation functions and correlation length.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('-i', '--input-dir', required=True, type=str,
                        help='Input directory containing NPZ files')
    
    parser.add_argument('-o', '--output-prefix', required=True, type=str,
                        help='Output prefix for plot files')
    
    parser.add_argument('-prefix', '--file-prefix', default='space_corr', type=str,
                        help='Prefix of NPZ files to load (default: space_corr)')
    
    parser.add_argument('-temps', '--temperatures', nargs='+', default=None,
                        help='Specific temperatures to plot')
    
    parser.add_argument('--xmin', type=float, default=0,
                        help='X-axis minimum (default: 0)')
    
    parser.add_argument('--xmax', type=float, default=20,
                        help='X-axis maximum (default: 20)')
    
    parser.add_argument('--ylim', nargs=2, type=float, default=None,
                        help='Y-axis limits for correlation plot')
    
    parser.add_argument('--smooth', action='store_true',
                        help='Apply smoothing to correlation curves')
    
    parser.add_argument('--xi-method', default='envelope', 
                        choices=['envelope', 'direct', 'first_zero', 'decay_threshold'],
                        help='Method to extract correlation length (default: envelope)')
    
    parser.add_argument('--xi-threshold', default=0.1, type=float,
                        help='Threshold fraction for decay_threshold method (default: 0.1 = 10%%)')
    
    parser.add_argument('--show', action='store_true',
                        help='Show plots interactively')
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()
    
    print("="*70)
    print("SPACE CORRELATION PLOTTING")
    print("="*70)
    
    # Load data
    temperatures, correlations, distances_list = load_correlation_data(
        args.input_dir, args.file_prefix, args.temperatures
    )
    
    # Extract correlation lengths
    print("\nExtracting correlation lengths...")
    xi_values = []
    
    for temp, corr, dist in zip(temperatures, correlations, distances_list):
        xi = extract_correlation_length(dist, corr, method=args.xi_method, threshold=args.xi_threshold)
        xi_values.append(xi)
        print(f"  {temp}: ξ={xi:.2f} Å")
    
    # Create plots
    print("\nGenerating plots...")
    plot_space_correlation(temperatures, correlations, distances_list, args)
    plot_correlation_length(temperatures, xi_values, args)
    
    print("\n✓ All plots saved!")
    print("="*70)


if __name__ == "__main__":
    main()
