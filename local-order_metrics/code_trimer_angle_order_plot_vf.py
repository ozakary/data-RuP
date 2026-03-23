import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import hashlib
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.stats import gaussian_kde
import warnings
import matplotlib.ticker as ticker
warnings.filterwarnings('ignore')

try:
    import figure_formatting_v2 as ff
    ff.set_rcParams(ff.master_formatting)
except ImportError:
    pass

def get_cache_filename(temperatures, n_bins, value_range):
    """Generate unique cache filename based on parameters"""
    params_str = f"{temperatures}_{n_bins}_{value_range}"
    hash_obj = hashlib.md5(params_str.encode())
    return f"cache_angle_{hash_obj.hexdigest()}.npz"

def compute_kde(values, bandwidth='scott', subsample=None):
    """Compute KDE efficiently with optional subsampling"""
    if subsample is not None and len(values) > subsample:
        np.random.seed(42)
        indices = np.random.choice(len(values), subsample, replace=False)
        values_sampled = values[indices]
    else:
        values_sampled = values
    
    kde = gaussian_kde(values_sampled, bw_method=bandwidth)
    return kde

def load_population_statistics(output_dir='.', n_populations=4):
    """Load population statistics from CSV file"""
    stats_file = os.path.join(output_dir, f'trimer_angle_population_statistics_{n_populations}pop.csv')
    
    if not os.path.exists(stats_file):
        print(f"Warning: Population statistics file not found: {stats_file}")
        return None
    
    stats_df = pd.read_csv(stats_file)
    print(f"Loaded population statistics for {len(stats_df)} temperatures ({n_populations} populations)")
    return stats_df

def process_single_temperature(args):
    """Process a single temperature file - for parallel execution"""
    temp, output_dir, bin_edges, kde_subsample, kde_bandwidth = args
    
    csv_filename = os.path.join(output_dir, f'trimer_angle_{temp}K.csv')
    
    # Load data efficiently
    values = []
    chunksize = 1000000
    for chunk in pd.read_csv(csv_filename, chunksize=chunksize, usecols=['trimer_angle']):
        values.append(chunk['trimer_angle'].values)
    
    values = np.concatenate(values)
    
    # Compute histogram
    counts, _ = np.histogram(values, bins=bin_edges)
    
    # Compute KDE
    kde = compute_kde(values, bandwidth=kde_bandwidth, subsample=kde_subsample)
    
    stats = {
        'mean': values.mean(),
        'std': values.std(),
        'n_measurements': len(values)
    }
    
    return temp, counts, kde, stats

def load_or_compute_histograms(temperatures, output_dir='.', n_bins=100, 
                               value_range=None, force_recompute=False,
                               kde_subsample=50000, kde_bandwidth='scott'):
    """Load cached data or compute histograms and KDEs in parallel"""
    
    cache_file = get_cache_filename(temperatures, n_bins, value_range)
    cache_path = os.path.join(output_dir, cache_file)
    
    if os.path.exists(cache_path) and not force_recompute:
        print(f"Loading cached data from {cache_file}")
        data = np.load(cache_path, allow_pickle=True)
        return (data['bin_edges'], data['bin_centers'], 
                dict(zip(data['temperatures'], data['histograms'])),
                data['kdes'].item(),
                data['stats'].item())
    
    print("Computing histograms and KDEs (this will be cached for future use)...")
    
    # Define bins
    if value_range is None:
        # Auto-detect range from first temperature
        first_temp = temperatures[0]
        csv_file = os.path.join(output_dir, f'trimer_angle_{first_temp}K.csv')
        df_sample = pd.read_csv(csv_file, nrows=10000)
        value_range = (df_sample['trimer_angle'].min() - 5, 
                      df_sample['trimer_angle'].max() + 5)
        print(f"Auto-detected value range: {value_range}")
    
    bin_edges = np.linspace(value_range[0], value_range[1], n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Prepare arguments for parallel processing
    args_list = [(temp, output_dir, bin_edges, kde_subsample, kde_bandwidth) 
                 for temp in temperatures]
    
    # Process in parallel
    histograms = {}
    kdes = {}
    stats = {}
    
    n_processes = min(len(temperatures), os.cpu_count())
    print(f"Processing {len(temperatures)} temperatures using {n_processes} processes...")
    print(f"KDE settings: subsample={kde_subsample}, bandwidth='{kde_bandwidth}'")
    
    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        futures = {executor.submit(process_single_temperature, args): args[0] 
                  for args in args_list}
        
        with tqdm(total=len(futures), desc="Processing") as pbar:
            for future in as_completed(futures):
                temp, counts, kde, temp_stats = future.result()
                histograms[temp] = counts
                kdes[temp] = kde
                stats[temp] = temp_stats
                print(f"  {temp}K: {temp_stats['n_measurements']:,} measurements, "
                      f"mean={temp_stats['mean']:.2f}°±{temp_stats['std']:.2f}°")
                pbar.update(1)
    
    # Save to cache
    print(f"Saving cache to {cache_file}")
    np.savez_compressed(cache_path,
                       bin_edges=bin_edges,
                       bin_centers=bin_centers,
                       temperatures=temperatures,
                       histograms=[histograms[t] for t in temperatures],
                       kdes=kdes,
                       stats=stats)
    
    return bin_edges, bin_centers, histograms, kdes, stats

def plot_histogram_with_kde(temp, values, kde, bin_edges, output_dir='.', 
                            value_range=None, figsize=(6, 4)):
    """
    Plot histogram with KDE overlay for a single temperature
    
    Parameters:
    -----------
    temp : int
        Temperature in K
    values : array
        Raw trimer angle data
    kde : gaussian_kde object
        KDE object
    bin_edges : array
        Bin edges for histogram
    output_dir : str
        Directory to save figure
    value_range : tuple or None
        (min, max) value range, if None uses bin_edges range
    figsize : tuple
        Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use bin_edges range if value_range not specified
    if value_range is None:
        value_range = (bin_edges[0], bin_edges[-1])
    
    # Plot histogram
    counts, _, patches = ax.hist(values, bins=bin_edges, density=True, 
                                 alpha=0.6, color='#9B59B6', 
                                 edgecolor='white', linewidth=0.25, label='Data')
    
    # Plot KDE
    x_smooth = np.linspace(value_range[0], value_range[1], 1000)
    kde_values = kde(x_smooth)
    ax.plot(x_smooth, kde_values, '-', color='#2ECC71', linewidth=2, label='KDE')
    
    # Labels and title
    ax.set_xlabel('Trimer Angle / °')
    ax.set_ylabel('Density')
#    ax.set_title(f'Trimer Angle Distribution at {temp} K')
    ax.legend(frameon=False)
    
    # Set x-axis limits
    ax.set_xlim(value_range[0], value_range[1])
    ax.set_ylim(0, 0.15)
    
    plt.tight_layout()
    
    # Save figure
    fig_filename = os.path.join(output_dir, f'trimer_angle_histogram_{temp}K.svg')
    plt.savefig(fig_filename, dpi=300, bbox_inches='tight')
    print(f"Saved histogram plot to {fig_filename}")
    
    plt.close(fig)
    
    return fig, ax

def plot_2d_heatmap(temperatures, output_dir='.', n_bins=100, 
                    value_range=None, force_recompute=False,
                    colormap='RdPu', figsize=(6, 4),
                    kde_subsample=50000, kde_bandwidth='scott',
                    temp_resolution=100, value_resolution=500,
                    interpolation='cubic',
                    n_populations=4,
                    show_population_curve=True,
                    curve_colors=None,
                    curve_linewidth=2.0,
                    curve_alpha=0.8,
                    show_error_band=False,
                    error_band_alpha=0.2,
                    plot_individual_histograms=True,
                    histogram_temps=[100, 600],
                    horizontal_line_temps=[100, 600],
                    histogram_figsize=(6, 4)):
    """
    Create 2D heatmap with trimer angle on x-axis, temperature on y-axis.
    Overlays n_populations curves (default 4 for symmetric structure).
    
    Parameters:
    -----------
    temp_resolution : int
        Number of points in temperature interpolation
    value_resolution : int
        Number of points in value interpolation
    interpolation : str
        Interpolation method: 'cubic', 'linear', or 'nearest'
    n_populations : int
        Number of population curves to plot (default 4)
    show_population_curve : bool
        Whether to overlay dashed curves for population averages
    curve_colors : list or None
        Colors for the population curves
    curve_linewidth : float
        Line width for population curves
    curve_alpha : float
        Alpha (transparency) for population curves
    show_error_band : bool
        Whether to show ±1 std deviation bands around curves
    error_band_alpha : float
        Alpha for error bands
    plot_individual_histograms : bool
        Whether to plot individual histogram plots
    histogram_temps : list
        List of temperatures for individual histogram plots (must exist in data)
    horizontal_line_temps : list
        List of temperatures for horizontal dashed lines (can be interpolated values)
    histogram_figsize : tuple
        Figure size for individual histogram plots
    """
    
    # Load or compute histogram and KDE data
    bin_edges, bin_centers, histograms, kdes, stats = load_or_compute_histograms(
        temperatures, output_dir, n_bins, value_range, force_recompute,
        kde_subsample, kde_bandwidth)
    
    print("\nCreating 2D heatmap...")
    
    # Create high-resolution value grid
    x_values = np.linspace(bin_edges[0], bin_edges[-1], value_resolution)
    
    # Build 2D array
    Z = np.zeros((len(temperatures), value_resolution))
    
    print("Evaluating KDEs on high-resolution grid...")
    for i, temp in enumerate(temperatures):
        kde = kdes[temp]
        Z[i, :] = kde(x_values)
        Z[i, :] *= stats[temp]['n_measurements']
    
    # Create temperature array
    temp_array = np.array(temperatures)
    
    # Create smooth temperature grid
    temp_smooth = np.linspace(temp_array.min(), temp_array.max(), temp_resolution)
    
    # Interpolate in temperature direction
    print(f"Interpolating in temperature direction using {interpolation} interpolation...")
    Z_smooth = np.zeros((temp_resolution, value_resolution))
    
    if interpolation == 'cubic':
        from scipy.interpolate import CubicSpline
        for j in range(value_resolution):
            cs = CubicSpline(temp_array, Z[:, j])
            Z_smooth[:, j] = cs(temp_smooth)
    elif interpolation == 'linear':
        for j in range(value_resolution):
            Z_smooth[:, j] = np.interp(temp_smooth, temp_array, Z[:, j])
    else:
        for j in range(value_resolution):
            indices = np.searchsorted(temp_array, temp_smooth)
            indices = np.clip(indices, 0, len(temp_array) - 1)
            Z_smooth[:, j] = Z[indices, j]
    
    Z_smooth = np.maximum(Z_smooth, 0)
    
    # Load population statistics
    pop_stats = None
    if show_population_curve:
        pop_stats = load_population_statistics(output_dir, n_populations=n_populations)
        if pop_stats is None:
            print("Warning: Cannot plot population curve - statistics file not found")
            show_population_curve = False
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    X, Y = np.meshgrid(x_values, temp_smooth)
    im = ax.pcolormesh(X, Y, Z_smooth, cmap=colormap, shading='auto', alpha=0.9)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Count')
    cbar.ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    cbar.ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    # Overlay population curves
    if show_population_curve and pop_stats is not None:
        print(f"\nAdding {n_populations} population mode curves...")
        
        temps_pop = pop_stats['temperature'].values
        
        # Default colors for populations
        if curve_colors is None:
            if n_populations == 4:
                curve_colors = ['k', 'k', 'k', 'k']#'white', 'cyan', 'yellow', 'magenta']
            elif n_populations == 3:
                curve_colors = ['k', 'k', 'k']
            else:
                # Generate colors if different number
                from matplotlib import cm
                cmap = cm.get_cmap('Set1')
                curve_colors = [cmap(i/n_populations) for i in range(n_populations)]
        
        # Plot curves for each population
        for pop_idx in range(1, n_populations + 1):
            pop_mode = pop_stats[f'pop{pop_idx}_mode'].values
            pop_std = pop_stats[f'pop{pop_idx}_std'].values
            
            color = curve_colors[pop_idx - 1]
            label = f'Pop {pop_idx}'
            
            ax.plot(pop_mode, temps_pop, '--', color=color, 
                    linewidth=curve_linewidth, alpha=curve_alpha,
                    label=label, zorder=10)
            
            # Optionally add error bands
            if show_error_band:
                ax.fill_betweenx(temps_pop, pop_mode - pop_std, pop_mode + pop_std,
                                color=color, alpha=error_band_alpha, zorder=5)
            
            print(f"  Pop{pop_idx} at {temps_pop[0]:.0f}K: {pop_mode[0]:.2f}°, "
                  f"at {temps_pop[-1]:.0f}K: {pop_mode[-1]:.2f}°")
        
#        ax.legend(loc='upper left', framealpha=0.7, fontsize=9)
    
    # Add horizontal dashed lines at specified temperatures
    for line_temp in horizontal_line_temps:
        ax.axhline(y=line_temp, color='white', linestyle='--', linewidth=2.0, alpha=0.8)
    
    # Labels
    ax.set_xlabel('Trimer Angle / °')
    ax.set_ylabel('$T$ / K')
    
    # Set limits
    ax.set_xlim(bin_edges[0], bin_edges[-1])
    ax.set_ylim(temp_array.min(), temp_array.max())
    
    plt.tight_layout()
    
    # Save figure
    fig_filename = os.path.join(output_dir, 'trimer_angle_2d_heatmap.png')
    plt.savefig(fig_filename, dpi=300, bbox_inches='tight')
    print(f"\nSaved plot to {fig_filename}")
    
    plt.show()
    
    # Plot individual histograms if requested
    if plot_individual_histograms:
        print("\nCreating individual histogram plots...")
        for hist_temp in histogram_temps:
            if hist_temp in temperatures:
                # Load the raw data for this temperature
                csv_filename = os.path.join(output_dir, f'trimer_angle_{hist_temp}K.csv')
                print(f"\nLoading data for {hist_temp}K histogram...")
                
                values = []
                chunksize = 1000000
                for chunk in pd.read_csv(csv_filename, chunksize=chunksize, usecols=['trimer_angle']):
                    values.append(chunk['trimer_angle'].values)
                
                values = np.concatenate(values)
                
                # Get the KDE for this temperature
                kde = kdes[hist_temp]
                
                # Plot histogram with KDE
                plot_histogram_with_kde(hist_temp, values, kde, bin_edges, 
                                       output_dir, value_range=(bin_edges[0], bin_edges[-1]), 
                                       figsize=histogram_figsize)
            else:
                print(f"Warning: Temperature {hist_temp}K not found in processed data")
    
    return fig, ax, Z_smooth, x_values, temp_smooth

def main():
    """Main function for creating 2D heatmap"""
    
    # Configuration
    temperatures = [50, 100, 110, 120, 130, 140, 150, 160, 170, 180, 
                    190, 200, 250, 260, 270, 280, 290, 300, 310, 320, 
                    330, 340, 350, 400, 450, 500, 550, 600, 650, 700]
    output_dir = '.'
    n_bins = 200
    value_range = None  # Auto-detect
    
    # KDE settings
    kde_subsample = 50000
    kde_bandwidth = 'scott'
    
    # 2D heatmap settings
    temp_resolution = 500
    value_resolution = 500
    interpolation = 'linear'
    
    # Population curve settings (4 populations for symmetric structure)
    n_populations = 4
    show_population_curve = True
    curve_colors = None  # Will auto-select: white, cyan, yellow, magenta
    curve_linewidth = 2.5
    curve_alpha = 1.0
    show_error_band = True
    error_band_alpha = 0.08
    
    # Individual histogram settings
    plot_individual_histograms = True  # Set to False to skip histogram plots
    histogram_temps = [100, 600]  # Temperatures for individual histograms (must exist in data)
    horizontal_line_temps = [100, 600]  # Temperatures for horizontal lines (can be interpolated)
    histogram_figsize = (6, 4)  # Figure size for histogram plots (independent of 2D heatmap)
    
    # Check which files exist
    existing_temps = []
    for temp in temperatures:
        csv_filename = os.path.join(output_dir, f'trimer_angle_{temp}K.csv')
        if os.path.exists(csv_filename):
            existing_temps.append(temp)
    
    if len(existing_temps) == 0:
        print("No data files found!")
        return
    
    print(f"Found data for temperatures: {existing_temps}")
    
    # Create 2D heatmap
    fig, ax, Z, x_values, temp_smooth = plot_2d_heatmap(
        existing_temps, 
        output_dir=output_dir,
        n_bins=n_bins,
        value_range=value_range,
        force_recompute=False,
        colormap='RdBu_r',  # Try: 'Spectral_r', 'RdBu_r', 'RdYlBu_r', 'viridis', 'plasma', 'inferno', 'hot', 'coolwarm'
        figsize=(6, 6),
        kde_subsample=kde_subsample,
        kde_bandwidth=kde_bandwidth,
        temp_resolution=temp_resolution,
        value_resolution=value_resolution,
        interpolation=interpolation,
        n_populations=n_populations,
        show_population_curve=show_population_curve,
        curve_colors=curve_colors,
        curve_linewidth=curve_linewidth,
        curve_alpha=curve_alpha,
        show_error_band=show_error_band,
        error_band_alpha=error_band_alpha,
        plot_individual_histograms=plot_individual_histograms,
        histogram_temps=histogram_temps,
        horizontal_line_temps=horizontal_line_temps,
        histogram_figsize=histogram_figsize
    )
    
    print("\nPlotting complete!")
    print("\nSettings used:")
    print(f"  - KDE subsample size: {kde_subsample}")
    print(f"  - KDE bandwidth method: {kde_bandwidth}")
    print(f"  - Temperature resolution: {temp_resolution}")
    print(f"  - Value resolution: {value_resolution}")
    print(f"  - Interpolation method: {interpolation}")
    print(f"  - Population curve: {show_population_curve}")
    print(f"  - Number of populations: {n_populations}")
    print(f"  - Individual histograms: {plot_individual_histograms}")
    if plot_individual_histograms:
        print(f"  - Histogram temperatures: {histogram_temps} K")
    print(f"  - Horizontal line temperatures: {horizontal_line_temps} K")
    print(f"  - Histogram figure size: {histogram_figsize}")
    print("\nGenerated files:")
    print("  1. trimer_angle_2d_heatmap.png (2D heatmap with horizontal lines)")
    if plot_individual_histograms:
        for hist_temp in histogram_temps:
            print(f"  {histogram_temps.index(hist_temp)+2}. trimer_angle_histogram_{hist_temp}K.png")
    print("\nTo change plot appearance without reprocessing data:")
    print("  - Modify colormap, figsize, temp_resolution, value_resolution, or interpolation")
    print("  - Modify curve appearance: curve_colors, curve_linewidth, curve_alpha")
    print("  - Modify histogram_temps to plot different temperatures (must exist in data)")
    print("  - Modify histogram_figsize to change histogram plot size independently")
    print("  - Modify horizontal_line_temps to add horizontal lines at any temperature")
    print("  - The processed KDE data is cached and will be reused")
    print("\nTo force data reprocessing:")
    print("  - Set force_recompute=True in plot_2d_heatmap()")

if __name__ == "__main__":
    main()
