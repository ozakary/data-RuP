import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
import argparse
import sys
from pathlib import Path

# Import the figure_formatting module for consistent plotting style
try:
    import figure_formatting_v2 as ff
    # Set up figure formatting using the function from the module
    ff.set_rcParams(ff.master_formatting)
except ImportError:
    print("Figure formatting module not found. Using default matplotlib settings.")

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Plot structure factor S(q) from individual NPZ data files.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - loads all sq_*.npz files from directory
  python plot_Sq_improved.py -i ./sq_data/ -o sq_plot.pdf
  
  # With custom file prefix
  python plot_Sq_improved.py -i ./sq_data/ -prefix mysim -o plot.pdf
  
  # With custom smoothing and no error bars
  python plot_Sq_improved.py -i ./sq_data/ -o plot.pdf -sigma 2.0 -no-error
  
  # Select specific temperatures
  python plot_Sq_improved.py -i ./sq_data/ -o plot.pdf -temps 100K 200K 300K 400K
  
  # Linear scale instead of log
  python plot_Sq_improved.py -i ./sq_data/ -o plot.pdf -linear
        """
    )
    
    # Required arguments
    parser.add_argument('-i', '--input-dir', required=True, type=str,
                        help='Input directory containing NPZ files from S(q) calculation')
    
    parser.add_argument('-o', '--output', required=True, type=str,
                        help='Output figure file (e.g., sq_plot.pdf, sq_plot.png)')
    
    parser.add_argument('-prefix', '--file-prefix', default='sq', type=str,
                        help='Prefix of NPZ files to load (default: sq, loads sq_*.npz)')
    
    # Optional arguments
    parser.add_argument('-sigma', '--smoothing', default=3.0, type=float,
                        help='Gaussian smoothing sigma (default: 3.0, use 0 for no smoothing)')
    
    parser.add_argument('-no-error', '--no-errorbars', action='store_true',
                        help='Disable error bar shading (default: show error bars)')
    
    parser.add_argument('-temps', '--temperatures', nargs='+', default=None,
                        help='Select specific temperatures to plot (e.g., 100K 200K 300K)')
    
    parser.add_argument('-linear', '--linear-scale', action='store_true',
                        help='Use linear y-scale instead of log scale')
    
    parser.add_argument('-dpi', '--dpi', default=150, type=int,
                        help='Figure DPI (default: 150)')
    
    parser.add_argument('-figsize', '--figure-size', nargs=2, default=[8, 6], type=float,
                        metavar=('WIDTH', 'HEIGHT'),
                        help='Figure size in inches (default: 8 6)')
    
    parser.add_argument('-cmap', '--colormap', default='coolwarm', type=str,
                        help='Matplotlib colormap name (default: coolwarm)')
    
    parser.add_argument('-ylim', '--ylimits', nargs=2, default=None, type=float,
                        metavar=('YMIN', 'YMAX'),
                        help='Y-axis limits (optional)')
    
    return parser.parse_args()


def load_sq_data(input_dir, file_prefix='sq', selected_temps=None):
    """
    Load S(q) data from individual NPZ files in a directory.
    
    Parameters:
    -----------
    input_dir : str or Path
        Directory containing NPZ files
    file_prefix : str
        Prefix of files to load (e.g., 'sq' loads sq_*.npz)
    selected_temps : list or None
        List of specific temperatures to load (e.g., ['100K', '200K'])
    
    Returns:
    --------
    data : dict
        Dictionary containing combined data from all temperature files
    """
    input_dir = Path(input_dir)
    
    if not input_dir.exists():
        print(f"✗ Input directory not found: {input_dir}")
        sys.exit(1)
    
    print(f"Loading data from: {input_dir.absolute()}")
    print(f"Looking for files: {file_prefix}_*.npz")
    
    # Find all matching NPZ files
    pattern = f"{file_prefix}_*.npz"
    npz_files = sorted(input_dir.glob(pattern))
    
    if len(npz_files) == 0:
        print(f"✗ No files matching pattern '{pattern}' found in {input_dir}")
        sys.exit(1)
    
    print(f"Found {len(npz_files)} NPZ files")
    
    # Load data from each file
    temperatures = []
    S_mean_list = []
    S_error_list = []
    qpath_frac = None
    qnorm = None
    source_files = []
    
    for npz_file in npz_files:
        # Extract temperature from filename (e.g., sq_100K.npz -> 100K)
        temp_str = npz_file.stem.replace(f"{file_prefix}_", "")
        
        # Skip if not in selected temperatures
        if selected_temps is not None and temp_str not in selected_temps:
            continue
        
        # Load file
        data = np.load(npz_file, allow_pickle=True)
        
        # Get temperature (prefer stored value, fallback to filename)
        if 'temperature' in data:
            temp = str(data['temperature'])
        else:
            temp = temp_str
        
        temperatures.append(temp)
        S_mean_list.append(data['S_mean'])
        S_error_list.append(data['S_error'])
        
        # Get qpath_frac and qnorm (should be same for all files)
        if qpath_frac is None:
            qpath_frac = data['qpath_frac']
        if qnorm is None and 'qnorm' in data:
            qnorm = data['qnorm']
        
        if 'source_file' in data:
            source_files.append(str(data['source_file']))
        
        print(f"  ✓ Loaded: {npz_file.name} ({temp})")
    
    if len(temperatures) == 0:
        print(f"✗ No temperature data loaded!")
        if selected_temps is not None:
            print(f"  Requested temperatures: {selected_temps}")
            print(f"  Available files: {[f.name for f in npz_files]}")
        sys.exit(1)
    
    # Convert to arrays
    temperatures = np.array(temperatures)
    S_mean = np.array(S_mean_list)
    S_error = np.array(S_error_list)
    
    print(f"\nLoaded {len(temperatures)} temperatures: {list(temperatures)}")
    print(f"Data shape: {S_mean.shape}")
    
    return {
        'qpath_frac': qpath_frac,
        'temperatures': temperatures,
        'qnorm': qnorm,
        'S_mean': S_mean,
        'S_error': S_error,
        'source_files': source_files
    }


def format_q_labels(qpath_frac):
    """
    Format q-point labels from fractional coordinates as [uvw].
    
    Parameters:
    -----------
    qpath_frac : ndarray
        Q-points in fractional coordinates, shape (K, 3)
    
    Returns:
    --------
    labels : list
        List of formatted labels in [uvw] format
    """
    labels = []
    for q in qpath_frac:
        # Format as [uvw] - show integers without decimals
        u, v, w = q[0], q[1], q[2]
        
        # Convert to integers if they are whole numbers
        if np.isclose(u, round(u)):
            u = int(round(u))
        if np.isclose(v, round(v)):
            v = int(round(v))
        if np.isclose(w, round(w)):
            w = int(round(w))
        
        # Format the label
        if isinstance(u, int) and isinstance(v, int) and isinstance(w, int):
            label = f"[{u}{v}{w}]"
        else:
            # If not integers, show with one decimal place
            label = f"[{u:.1f}{v:.1f}{w:.1f}]"
        
        labels.append(label)
    
    return labels


def plot_sq(data, args):
    """
    Create S(q) plot.
    
    Parameters:
    -----------
    data : dict
        Dictionary containing S(q) data
    args : Namespace
        Command-line arguments
    """
    qpath_frac = data['qpath_frac']
    temperatures = data['temperatures']
    S_mean = data['S_mean']
    S_error = data['S_error']
    qnorm = data.get('qnorm', None)
    
    print(f"\nPlotting {len(temperatures)} temperatures")
    
    # Setup tick positions and labels
    K = len(qpath_frac)  # number of high-symmetry vertices
    Nq = S_mean.shape[1]
    x = np.arange(Nq)
    tick_pos = [int(round(k * (Nq - 1) / (K - 1))) for k in range(K)]
    tick_labels = format_q_labels(qpath_frac)
    
    print(f"  Q-path: {' → '.join(tick_labels)}")
    print(f"  Number of q-points: {Nq}")
    
    # Create figure with user's style
    fig, ax = plt.subplots(figsize=(5.75, 6.0))
    
    # Add grid for y-axis to clearly show logarithmic scale
    ax.grid(True, axis="y", which="both", alpha=0.25, linewidth=2.0)
    
    # Vertical guides at symmetry points
    for t in tick_pos:
        ax.axvline(t, color="0.85", linewidth=2.0, zorder=0)
    
    # Helper function for confidence intervals
    def ci95(mean_row, sem):
        """Calculate 95% confidence interval."""
        z = 1.96  # ~95% for large n
        lo = mean_row - z * sem
        hi = mean_row + z * sem
        return lo, hi
    
    # Log-safe small positive clip
    ymin_clip = 1e-10 if not args.linear_scale else 0
    
    # Sort temperatures for legend
    nT = len(temperatures)
    
    # Convert temperatures to numeric for sorting
    temp_numeric = np.array([float(t.rstrip('K')) for t in temperatures])
    order = np.argsort(temp_numeric)
    
    # Get colormap - use coolwarm like user's style
    import matplotlib.cm as cm
    colors = cm.coolwarm(np.linspace(0, 1, nT))
    
    print("\nPlotting S(q) curves...")
    
    # Plot each temperature
    for idx, i in enumerate(order):
        temp = temperatures[i]
        y0 = S_mean[i]
        
        # Apply smoothing if requested
        if args.smoothing > 0:
            y = gaussian_filter(y0, sigma=args.smoothing)
        else:
            y = y0
        
        # Clip for log scale
        if not args.linear_scale:
            y = np.clip(y, ymin_clip, None)
        
        # Calculate confidence intervals
        if not args.no_errorbars:
            lo, hi = ci95(S_mean[i], S_error[i])
            if not args.linear_scale:
                lo = np.clip(lo, ymin_clip, None)
                hi = np.clip(hi, ymin_clip, None)
        
        # Choose color from coolwarm
        color = colors[idx]
        
        # Plot main line with thicker linewidth like user's style
        line, = ax.plot(x, y, linewidth=2.5, color=color, label=temp, zorder=10)
        
        # Add error shading
        if not args.no_errorbars:
            ax.fill_between(x, lo, hi, alpha=0.20, linewidth=0, 
                           color=line.get_color(), zorder=5)
        
        print(f"  {temp}: S(q) range = [{S_mean[i].min():.2e}, {S_mean[i].max():.2e}]")
    
    # Set scale
    if args.linear_scale:
        ax.set_yscale("linear")
        print("\nUsing linear y-scale")
    else:
        ax.set_yscale("log", nonpositive="clip")
        print("\nUsing log y-scale")
    
    # Labels and ticks - match user's style
    ax.set_xlabel(r'$q$ along high-symmetry path')
    ax.set_ylabel(r'$\langle S_{\text{insta}}(q) \rangle$')
    
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels)
    ax.set_xlim(-0.5, Nq - 0.5)
    ax.set_ylim(1e-7, 1e0)
    
    # Set y-limits if specified
    if args.ylimits is not None:
        ax.set_ylim(args.ylimits)
        print(f"Y-limits set to: [{args.ylimits[0]}, {args.ylimits[1]}]")
    
    # Legend below the plot with 4 columns - user's style
#    leg = ax.legend(frameon=False, loc='upper center', 
#                   bbox_to_anchor=(0.5, -0.22), ncol=10)
    
    # Save figure - match user's style
    print(f"\nSaving figure to: {args.output}")
    plt.savefig(args.output, bbox_inches='tight')
    print("✓ Figure saved successfully")
    
    # Show plot
    plt.show()


def main():
    """Main execution function."""
    args = parse_arguments()
    
    print("="*70)
    print("STRUCTURE FACTOR PLOTTING")
    print("="*70)
    
    # Load data from individual temperature files
    data = load_sq_data(
        args.input_dir, 
        file_prefix=args.file_prefix,
        selected_temps=args.temperatures
    )
    
    # Create plot
    plot_sq(data, args)
    
    print("\n✓ All done!")
    print("="*70)


if __name__ == "__main__":
    main()
