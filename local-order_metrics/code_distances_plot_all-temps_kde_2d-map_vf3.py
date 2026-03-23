import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import hashlib
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.stats import gaussian_kde
from scipy.interpolate import interp2d, RectBivariateSpline
import warnings
import matplotlib.ticker as ticker
warnings.filterwarnings('ignore')

# Optional: Set up figure formatting
try:
    import figure_formatting_v2 as ff
    ff.set_rcParams(ff.master_formatting)
except ImportError:
    pass

def get_cache_filename(temperatures, n_bins, distance_range):
    """Generate unique cache filename based on parameters"""
    params_str = f"{temperatures}_{n_bins}_{distance_range}"
    hash_obj = hashlib.md5(params_str.encode())
    return f"cache_ru110_{hash_obj.hexdigest()}.npz"

def compute_kde(distances, bandwidth='scott', subsample=None):
    """
    Compute KDE efficiently with optional subsampling
    
    Parameters:
    -----------
    distances : array
        Distance data
    bandwidth : float or str
        Bandwidth for KDE. 'scott' or 'silverman' for automatic, or float value
    subsample : int or None
        If provided, randomly subsample this many points for KDE computation
    """
    if subsample is not None and len(distances) > subsample:
        # Randomly subsample for efficiency
        np.random.seed(42)  # For reproducibility
        indices = np.random.choice(len(distances), subsample, replace=False)
        distances_sampled = distances[indices]
    else:
        distances_sampled = distances
    
    # Compute KDE
    kde = gaussian_kde(distances_sampled, bw_method=bandwidth)
    
    return kde

def load_population_statistics(output_dir='.'):
    """
    Load population statistics from CSV file
    
    Returns:
    --------
    stats_df : DataFrame
        DataFrame with columns: temperature, pop1_mean, pop1_std, pop2_mean, pop2_std, etc.
    """
    stats_file = os.path.join(output_dir, 'population_statistics.csv')
    
    if not os.path.exists(stats_file):
        print(f"Warning: Population statistics file not found: {stats_file}")
        return None
    
    stats_df = pd.read_csv(stats_file)
    print(f"Loaded population statistics for {len(stats_df)} temperatures")
    return stats_df

def process_single_temperature(args):
    """Process a single temperature file - for parallel execution"""
    temp, output_dir, bin_edges, kde_subsample, kde_bandwidth = args
    
    csv_filename = os.path.join(output_dir, f'ru_110_distances_{temp}K.csv')
    
    # Load data efficiently
    distances = []
    chunksize = 1000000
    for chunk in pd.read_csv(csv_filename, chunksize=chunksize, usecols=['distance_angstrom']):
        distances.append(chunk['distance_angstrom'].values)
    
    distances = np.concatenate(distances)
    
    # Compute histogram
    counts, _ = np.histogram(distances, bins=bin_edges)
    
    # Compute KDE
    kde = compute_kde(distances, bandwidth=kde_bandwidth, subsample=kde_subsample)
    
    stats = {
        'mean': distances.mean(),
        'std': distances.std(),
        'n_measurements': len(distances)
    }
    
    return temp, counts, kde, stats

def load_or_compute_histograms(temperatures, output_dir='.', n_bins=100, 
                               distance_range=(2.6, 4.0), force_recompute=False,
                               kde_subsample=50000, kde_bandwidth='scott'):
    """Load cached data or compute histograms and KDEs in parallel"""
    
    cache_file = get_cache_filename(temperatures, n_bins, distance_range)
    cache_path = os.path.join(output_dir, cache_file)
    
    # Check if cache exists and not forcing recompute
    if os.path.exists(cache_path) and not force_recompute:
        print(f"Loading cached data from {cache_file}")
        data = np.load(cache_path, allow_pickle=True)
        return (data['bin_edges'], data['bin_centers'], 
                dict(zip(data['temperatures'], data['histograms'])),
                data['kdes'].item(),
                data['stats'].item())
    
    print("Computing histograms and KDEs (this will be cached for future use)...")
    
    # Define bins
    bin_edges = np.linspace(distance_range[0], distance_range[1], n_bins + 1)
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
                      f"mean={temp_stats['mean']:.3f}±{temp_stats['std']:.3f} Å")
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

def plot_histogram_with_kde(temp, distances, kde, bin_edges, output_dir='.', 
                            distance_range=(2.5, 4.0), figsize=(4.0, 4.0)):
    """
    Plot histogram with KDE overlay for a single temperature
    
    Parameters:
    -----------
    temp : int
        Temperature in K
    distances : array
        Raw distance data
    kde : gaussian_kde object
        KDE object
    bin_edges : array
        Bin edges for histogram
    output_dir : str
        Directory to save figure
    distance_range : tuple
        (min, max) distance range
    figsize : tuple
        Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot histogram
    counts, _, patches = ax.hist(distances, bins=bin_edges, density=True, 
                                 alpha=0.6, color='#9B59B6', 
                                 edgecolor='white', linewidth=0.25, label='Data')
    
    # Plot KDE
    x_smooth = np.linspace(distance_range[0], distance_range[1], 1000)
    kde_values = kde(x_smooth)
    ax.plot(x_smooth, kde_values, '-', color='#2ECC71', linewidth=2, label='KDE')
    
    # Labels and title
    ax.set_xlabel('$r$(Ru$-$Ru) / Å')
    ax.set_ylabel('Density')
    ax.legend(frameon=False, loc='upper left')
    
    # Set x-axis limits
    ax.set_xlim(distance_range[0], distance_range[1])
    ax.set_ylim(0, 8)
    
    plt.tight_layout()
    
    # Save figure
    fig_filename = os.path.join(output_dir, f'ru_110_histogram_{temp}K.svg')
    plt.savefig(fig_filename, dpi=300, bbox_inches='tight')
    print(f"Saved histogram plot to {fig_filename}")
    
    plt.close(fig)
    
    return fig, ax

def plot_2d_heatmap(temperatures, output_dir='.', n_bins=100, 
                    distance_range=(2.6, 4.0), force_recompute=False,
                    colormap='RdPu', figsize=(6, 4),
                    kde_subsample=50000, kde_bandwidth='scott',
                    temp_resolution=100, dist_resolution=500,
                    interpolation='cubic',
                    show_population_curves=True,
                    curve_color='white',
                    curve_linewidth=2.0,
                    curve_alpha=0.8,
                    show_error_bands=False,
                    error_band_alpha=0.2,
                    plot_individual_histograms=True,
                    histogram_temps=[100, 600],
                    horizontal_line_temps=[100, 600]):
    """
    Create 2D heatmap with distance on x-axis, temperature on y-axis, 
    and frequency/count as color. Optionally overlay population average curves.
    
    Parameters:
    -----------
    temp_resolution : int
        Number of points in temperature interpolation
    dist_resolution : int
        Number of points in distance interpolation
    interpolation : str
        Interpolation method: 'cubic', 'linear', or 'nearest'
    show_population_curves : bool
        Whether to overlay dashed curves for population averages
    curve_color : str
        Color for the population curves
    curve_linewidth : float
        Line width for population curves
    curve_alpha : float
        Alpha (transparency) for population curves
    show_error_bands : bool
        Whether to show ±1 std deviation bands around curves
    error_band_alpha : float
        Alpha for error bands
    plot_individual_histograms : bool
        Whether to plot individual histogram plots
    histogram_temps : list
        List of temperatures for individual histogram plots (must exist in data)
    horizontal_line_temps : list
        List of temperatures for horizontal dashed lines (can be interpolated values)
    """
    
    # Load or compute histogram and KDE data
    bin_edges, bin_centers, histograms, kdes, stats = load_or_compute_histograms(
        temperatures, output_dir, n_bins, distance_range, force_recompute,
        kde_subsample, kde_bandwidth)
    
    print("\nCreating 2D heatmap...")
    
    # Create high-resolution distance grid
    x_dist = np.linspace(distance_range[0], distance_range[1], dist_resolution)
    
    # Build 2D array: rows are temperatures, columns are distances
    Z = np.zeros((len(temperatures), dist_resolution))
    
    print("Evaluating KDEs on high-resolution grid...")
    for i, temp in enumerate(temperatures):
        kde = kdes[temp]
        # Evaluate KDE on high-resolution distance grid
        Z[i, :] = kde(x_dist)
        # Normalize by the number of measurements to get proper density
        Z[i, :] *= stats[temp]['n_measurements']
    
    # Create temperature array
    temp_array = np.array(temperatures)
    
    # Create smooth temperature grid
    temp_smooth = np.linspace(temp_array.min(), temp_array.max(), temp_resolution)
    
    # Interpolate in temperature direction to create smooth 2D map
    print(f"Interpolating in temperature direction using {interpolation} interpolation...")
    Z_smooth = np.zeros((temp_resolution, dist_resolution))
    
    if interpolation == 'cubic':
        # Use cubic spline interpolation
        from scipy.interpolate import CubicSpline
        for j in range(dist_resolution):
            cs = CubicSpline(temp_array, Z[:, j])
            Z_smooth[:, j] = cs(temp_smooth)
    elif interpolation == 'linear':
        # Use linear interpolation
        for j in range(dist_resolution):
            Z_smooth[:, j] = np.interp(temp_smooth, temp_array, Z[:, j])
    else:  # nearest
        # Use nearest neighbor
        for j in range(dist_resolution):
            indices = np.searchsorted(temp_array, temp_smooth)
            indices = np.clip(indices, 0, len(temp_array) - 1)
            Z_smooth[:, j] = Z[indices, j]
    
    # Ensure non-negative values
    Z_smooth = np.maximum(Z_smooth, 0)
    
    # Load population statistics if available
    pop_stats = None
    if show_population_curves:
        pop_stats = load_population_statistics(output_dir)
        if pop_stats is None:
            print("Warning: Cannot plot population curves - statistics file not found")
            show_population_curves = False
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot using pcolormesh for smooth appearance
    X, Y = np.meshgrid(x_dist, temp_smooth)
    
    im = ax.pcolormesh(X, Y, Z_smooth, cmap=colormap, shading='auto', alpha=0.9)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Count')
    
    # Format colorbar with scientific notation
    cbar.ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    cbar.ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    # Overlay population curves if requested
    if show_population_curves and pop_stats is not None:
        print("\nAdding population average curves...")
        
        temps_pop = pop_stats['temperature'].values
        pop1_mean = pop_stats['pop1_mean'].values
        pop1_std = pop_stats['pop1_std'].values
        pop2_mean = pop_stats['pop2_mean'].values
        pop2_std = pop_stats['pop2_std'].values
        
        # Plot population 1 (shorter distances)
        ax.plot(pop1_mean, temps_pop, '--', color=curve_color, 
                linewidth=curve_linewidth, alpha=curve_alpha,
                label='Pop. 1 (short)')
        
        # Plot population 2 (longer distances)
        ax.plot(pop2_mean, temps_pop, '--', color=curve_color,
                linewidth=curve_linewidth, alpha=curve_alpha,
                label='Pop. 2 (long)')
        
        # Optionally add error bands (±1 std)
        if show_error_bands:
            ax.fill_betweenx(temps_pop, pop1_mean - pop1_std, pop1_mean + pop1_std,
                            color=curve_color, alpha=error_band_alpha)
            ax.fill_betweenx(temps_pop, pop2_mean - pop2_std, pop2_mean + pop2_std,
                            color=curve_color, alpha=error_band_alpha)
        
        # Add legend
#        ax.legend(loc='upper left', framealpha=0.7, fontsize=9)
        
        # Print some statistics
        print(f"  Population 1 at {temps_pop[0]:.0f}K: {pop1_mean[0]:.3f} Å")
        print(f"  Population 2 at {temps_pop[0]:.0f}K: {pop2_mean[0]:.3f} Å")
        print(f"  Population 1 at {temps_pop[-1]:.0f}K: {pop1_mean[-1]:.3f} Å")
        print(f"  Population 2 at {temps_pop[-1]:.0f}K: {pop2_mean[-1]:.3f} Å")
        
        # Identify merging point (where curves get closest)
        distances_between = np.abs(pop2_mean - pop1_mean)
        min_dist_idx = np.argmin(distances_between)
        merge_temp = temps_pop[min_dist_idx]
        merge_dist = distances_between[min_dist_idx]
        print(f"  Closest approach at {merge_temp:.0f}K: Δr = {merge_dist:.3f} Å")
    
    # Add horizontal dashed lines at specified temperatures
    for line_temp in horizontal_line_temps:
        ax.axhline(y=line_temp, color='white', linestyle='--', linewidth=2.0, alpha=0.8)
    
    # Labels
    ax.set_xlabel('$r$(Ru$-$Ru) / Å')
    ax.set_ylabel('$T$ / K')
    
    # Set limits
    ax.set_xlim(distance_range[0], distance_range[1])
    ax.set_ylim(temp_array.min(), temp_array.max())
    
    plt.tight_layout()
    
    # Save figure
    fig_filename = os.path.join(output_dir, 'ru_110_2d_heatmap_with_curves.png')
    plt.savefig(fig_filename, dpi=300, bbox_inches='tight')
    print(f"\nSaved plot to {fig_filename}")
    
    plt.show()
    
    # Plot individual histograms if requested
    if plot_individual_histograms:
        print("\nCreating individual histogram plots...")
        for hist_temp in histogram_temps:
            if hist_temp in temperatures:
                # Load the raw distance data for this temperature
                csv_filename = os.path.join(output_dir, f'ru_110_distances_{hist_temp}K.csv')
                print(f"\nLoading data for {hist_temp}K histogram...")
                
                distances = []
                chunksize = 1000000
                for chunk in pd.read_csv(csv_filename, chunksize=chunksize, usecols=['distance_angstrom']):
                    distances.append(chunk['distance_angstrom'].values)
                
                distances = np.concatenate(distances)
                
                # Get the KDE for this temperature
                kde = kdes[hist_temp]
                
                # Plot histogram with KDE
                plot_histogram_with_kde(hist_temp, distances, kde, 100, 
                                       output_dir, distance_range)
            else:
                print(f"Warning: Temperature {hist_temp}K not found in processed data")
    
    return fig, ax, Z_smooth, x_dist, temp_smooth

def main():
    """Main function for creating 2D heatmap"""
    
    # Configuration
    # Make sure to include all temperatures you have computed
    temperatures = [50, 100, 110, 120, 130, 140, 150, 160, 170, 180, 
                    190, 200, 250, 260, 270, 280, 290, 300, 310, 320, 
                    330, 340, 350, 400, 450, 500, 550, 600, 650, 700]
    output_dir = '.'
    n_bins = 200
    distance_range = (2.5, 4.0)
    
    # KDE settings for distance dimension
    kde_subsample = 50000  # Use 50k points for KDE
    kde_bandwidth = 'scott'  # 'scott', 'silverman', or a float value
    
    # 2D heatmap settings
    temp_resolution = 500  # Number of interpolated temperature points
    dist_resolution = 500  # Number of distance points
    interpolation = 'linear'  # 'cubic', 'linear', or 'nearest'
    
    # Population curve settings
    show_population_curves = True  # Set to False to hide curves
    curve_color = 'k'  # Color of the dashed curves
    curve_linewidth = 2.2  # Line width
    curve_alpha = 1.0  # Transparency (0-1)
    show_error_bands = True  # Show ±1 std deviation bands
    error_band_alpha = 0.08  # Transparency of error bands
    
    # Individual histogram settings
    plot_individual_histograms = True  # Set to False to skip histogram plots
    histogram_temps = [100, 600]  # Temperatures for individual histograms (must exist in data)
    horizontal_line_temps = [100, 600]  # Temperatures for horizontal lines (can be interpolated)
    
    # Check which files exist
    existing_temps = []
    for temp in temperatures:
        csv_filename = os.path.join(output_dir, f'ru_110_distances_{temp}K.csv')
        if os.path.exists(csv_filename):
            existing_temps.append(temp)
    
    if len(existing_temps) == 0:
        print("No data files found!")
        return
    
    print(f"Found data for temperatures: {existing_temps}")
    
    # Create 2D heatmap (will use cache if available)
    fig, ax, Z, x_dist, temp_smooth = plot_2d_heatmap(
        existing_temps, 
        output_dir=output_dir,
        n_bins=n_bins,
        distance_range=distance_range,
        force_recompute=False,  # Set to True to force recomputation
        colormap='RdBu_r',  # Try: 'Spectral_r', 'RdBu_r', 'RdYlBu_r', 'viridis', 'plasma', 'inferno', 'hot', 'coolwarm'
        figsize=(6, 6),  # Adjust figure size here
        kde_subsample=kde_subsample,
        kde_bandwidth=kde_bandwidth,
        temp_resolution=temp_resolution,
        dist_resolution=dist_resolution,
        interpolation=interpolation,
        show_population_curves=show_population_curves,
        curve_color=curve_color,
        curve_linewidth=curve_linewidth,
        curve_alpha=curve_alpha,
        show_error_bands=show_error_bands,
        error_band_alpha=error_band_alpha,
        plot_individual_histograms=plot_individual_histograms,
        histogram_temps=histogram_temps,
        horizontal_line_temps=horizontal_line_temps
    )
    
    print("\nPlotting complete!")
    print("\nSettings used:")
    print(f"  - KDE subsample size: {kde_subsample}")
    print(f"  - KDE bandwidth method: {kde_bandwidth}")
    print(f"  - Temperature resolution: {temp_resolution}")
    print(f"  - Distance resolution: {dist_resolution}")
    print(f"  - Interpolation method: {interpolation}")
    print(f"  - Population curves: {show_population_curves}")
    if show_population_curves:
        print(f"  - Curve color: {curve_color}")
        print(f"  - Curve linewidth: {curve_linewidth}")
        print(f"  - Curve alpha: {curve_alpha}")
        print(f"  - Error bands: {show_error_bands}")
    print(f"  - Individual histograms: {plot_individual_histograms}")
    if plot_individual_histograms:
        print(f"  - Histogram temperatures: {histogram_temps} K")
    print(f"  - Horizontal line temperatures: {horizontal_line_temps} K")
    print("\nGenerated files:")
    print("  1. ru_110_2d_heatmap_with_curves.png (2D heatmap with horizontal lines)")
    if plot_individual_histograms:
        for hist_temp in histogram_temps:
            print(f"  {histogram_temps.index(hist_temp)+2}. ru_110_histogram_{hist_temp}K.png")
    print("\nTo change plot appearance without reprocessing data:")
    print("  - Modify colormap, figsize, temp_resolution, dist_resolution, or interpolation")
    print("  - Modify curve appearance: curve_color, curve_linewidth, curve_alpha")
    print("  - Modify histogram_temps to plot different temperatures (must exist in data)")
    print("  - Modify horizontal_line_temps to add horizontal lines at any temperature")
    print("  - The processed KDE data is cached and will be reused")
    print("\nTo force data reprocessing:")
    print("  - Set force_recompute=True in plot_2d_heatmap()")

if __name__ == "__main__":
    main()
