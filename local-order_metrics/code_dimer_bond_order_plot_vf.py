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

# Optional: Set up figure formatting
try:
    import figure_formatting_v2 as ff
    ff.set_rcParams(ff.master_formatting)
except ImportError:
    pass

def get_cache_filename(temperatures, n_bins, value_range):
    """Generate unique cache filename based on parameters"""
    params_str = f"{temperatures}_{n_bins}_{value_range}"
    hash_obj = hashlib.md5(params_str.encode())
    return f"cache_dimer_{hash_obj.hexdigest()}.npz"

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

def load_population_statistics(output_dir='.'):
    """Load population statistics from CSV file"""
    stats_file = os.path.join(output_dir, 'dimer_population_statistics.csv')
    
    if not os.path.exists(stats_file):
        print(f"Warning: Population statistics file not found: {stats_file}")
        return None
    
    stats_df = pd.read_csv(stats_file)
    print(f"Loaded population statistics for {len(stats_df)} temperatures")
    return stats_df

def process_single_temperature(args):
    """Process a single temperature file - for parallel execution"""
    temp, output_dir, bin_edges, kde_subsample, kde_bandwidth = args
    
    csv_filename = os.path.join(output_dir, f'dimer_order_{temp}K.csv')
    
    # Load data efficiently
    values = []
    chunksize = 1000000
    for chunk in pd.read_csv(csv_filename, chunksize=chunksize, usecols=['dimer_order']):
        values.append(chunk['dimer_order'].values)
    
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
        csv_file = os.path.join(output_dir, f'dimer_order_{first_temp}K.csv')
        df_sample = pd.read_csv(csv_file, nrows=10000)
        value_range = (df_sample['dimer_order'].min() - 0.1, 
                      df_sample['dimer_order'].max() + 0.1)
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
                      f"mean={temp_stats['mean']:.4f}±{temp_stats['std']:.4f}")
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
                            value_range=(-1, 1), figsize=(6.0, 4.0)):
    """
    Plot histogram with KDE overlay for a single temperature
    
    Parameters:
    -----------
    temp : int
        Temperature in K
    values : array
        Raw dimer order data
    kde : gaussian_kde object
        KDE object
    bin_edges : array
        Bin edges for histogram
    output_dir : str
        Directory to save figure
    value_range : tuple
        (min, max) value range
    figsize : tuple
        Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot histogram
    counts, _, patches = ax.hist(values, bins=bin_edges, density=True, 
                                 alpha=0.6, color='#9B59B6', 
                                 edgecolor='white', linewidth=0.25, label='Data')
    
    # Plot KDE
    x_smooth = np.linspace(value_range[0], value_range[1], 1000)
    kde_values = kde(x_smooth)
    ax.plot(x_smooth, kde_values, '-', color='#2ECC71', linewidth=2, label='KDE')
    
    # Labels and title
    ax.set_xlabel('Dimer Bond Order')
    ax.set_ylabel('Density')
#    ax.set_title(f'Dimer Bond Order Distribution at {temp} K')
    ax.legend(frameon=False)
    
    # Set x-axis limits
    ax.set_xlim(value_range[0], value_range[1])
    ax.set_ylim(0, 3)
    
    plt.tight_layout()
    
    # Save figure
    fig_filename = os.path.join(output_dir, f'dimer_order_histogram_{temp}K.svg')
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
                    show_population_curves=True,
                    curve_colors=['white', 'cyan', 'yellow'],
                    curve_linewidth=2.0,
                    curve_alpha=0.8,
                    show_error_bands=False,
                    error_band_alpha=0.2,
                    plot_individual_histograms=True,
                    histogram_temps=[50, 600],
                    horizontal_line_temps=[50, 600],
                    histogram_figsize=(6, 4)):
    """
    Create 2D heatmap with dimer order on x-axis, temperature on y-axis.
    Overlays 3 population curves.
    
    Parameters:
    -----------
    temp_resolution : int
        Number of points in temperature interpolation
    value_resolution : int
        Number of points in value interpolation
    interpolation : str
        Interpolation method: 'cubic', 'linear', or 'nearest'
    show_population_curves : bool
        Whether to overlay dashed curves for population averages
    curve_colors : list
        Colors for the 3 population curves
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
    histogram_figsize : tuple
        Figure size for individual histogram plots
    """
    
    # Load or compute histogram and KDE data
    bin_edges, bin_centers, histograms, kdes, stats = load_or_compute_histograms(
        temperatures, output_dir, n_bins, value_range, force_recompute,
        kde_subsample, kde_bandwidth)
    
    print("\nCreating 2D heatmap...")
    
    # Create high-resolution value grid
    x_values = np.linspace(-1, 1, value_resolution)#bin_edges[0], bin_edges[-1], value_resolution)
    
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
    if show_population_curves:
        pop_stats = load_population_statistics(output_dir)
        if pop_stats is None:
            print("Warning: Cannot plot population curves - statistics file not found")
            show_population_curves = False
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    X, Y = np.meshgrid(x_values, temp_smooth)
    im = ax.pcolormesh(X, Y, Z_smooth, cmap=colormap, shading='auto', alpha=0.9)#, vmin=0, vmax=2e7)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Count')
    cbar.ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    cbar.ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    # Overlay population curves
    if show_population_curves and pop_stats is not None:
        print("\nAdding population average curves...")
        
        temps_pop = pop_stats['temperature'].values
        pop1_mean = pop_stats['pop1_mean'].values
        pop1_std = pop_stats['pop1_std'].values
        pop2_mean = pop_stats['pop2_mean'].values
        pop2_std = pop_stats['pop2_std'].values
        pop3_mean = pop_stats['pop3_mean'].values
        pop3_std = pop_stats['pop3_std'].values
        
        # Plot population 1
        ax.plot(pop1_mean, temps_pop, '--', color=curve_colors[0], 
                linewidth=curve_linewidth, alpha=curve_alpha,
                label='Pop. 1')
        
        # Plot population 2
        ax.plot(pop2_mean, temps_pop, '--', color=curve_colors[1],
                linewidth=curve_linewidth, alpha=curve_alpha,
                label='Pop. 2')
        
        # Plot population 3
        ax.plot(pop3_mean, temps_pop, '--', color=curve_colors[2],
                linewidth=curve_linewidth, alpha=curve_alpha,
                label='Pop. 3')
        
        # Optionally add error bands
        if show_error_bands:
            ax.fill_betweenx(temps_pop, pop1_mean - pop1_std, pop1_mean + pop1_std,
                            color=curve_colors[0], alpha=error_band_alpha)
            ax.fill_betweenx(temps_pop, pop2_mean - pop2_std, pop2_mean + pop2_std,
                            color=curve_colors[1], alpha=error_band_alpha)
            ax.fill_betweenx(temps_pop, pop3_mean - pop3_std, pop3_mean + pop3_std,
                            color=curve_colors[2], alpha=error_band_alpha)
        
#        ax.legend(loc='upper left', framealpha=0.7, fontsize=9)
        
        print(f"  Population 1 at {temps_pop[0]:.0f}K: {pop1_mean[0]:.4f}")
        print(f"  Population 2 at {temps_pop[0]:.0f}K: {pop2_mean[0]:.4f}")
        print(f"  Population 3 at {temps_pop[0]:.0f}K: {pop3_mean[0]:.4f}")
        print(f"  Population 1 at {temps_pop[-1]:.0f}K: {pop1_mean[-1]:.4f}")
        print(f"  Population 2 at {temps_pop[-1]:.0f}K: {pop2_mean[-1]:.4f}")
        print(f"  Population 3 at {temps_pop[-1]:.0f}K: {pop3_mean[-1]:.4f}")
    
    # Add horizontal dashed lines at specified temperatures
    for line_temp in horizontal_line_temps:
        ax.axhline(y=line_temp, color='white', linestyle='--', linewidth=2.0, alpha=0.8)
    
    # Labels
    ax.set_xlabel('Dimer Bond Order')
    ax.set_ylabel('$T$ / K')
    
    # Set limits
    ax.set_xlim(-1, 1)#bin_edges[0], bin_edges[-1])
    ax.set_ylim(temp_array.min(), temp_array.max())
    
    plt.tight_layout()
    
    # Save figure
    fig_filename = os.path.join(output_dir, 'dimer_order_2d_heatmap.png')
    plt.savefig(fig_filename, dpi=300, bbox_inches='tight')
    print(f"\nSaved plot to {fig_filename}")
    
    plt.show()
    
    # Plot individual histograms if requested
    if plot_individual_histograms:
        print("\nCreating individual histogram plots...")
        for hist_temp in histogram_temps:
            if hist_temp in temperatures:
                # Load the raw data for this temperature
                csv_filename = os.path.join(output_dir, f'dimer_order_{hist_temp}K.csv')
                print(f"\nLoading data for {hist_temp}K histogram...")
                
                values = []
                chunksize = 1000000
                for chunk in pd.read_csv(csv_filename, chunksize=chunksize, usecols=['dimer_order']):
                    values.append(chunk['dimer_order'].values)
                
                values = np.concatenate(values)
                
                # Get the KDE for this temperature
                kde = kdes[hist_temp]
                
                # Plot histogram with KDE
                plot_histogram_with_kde(hist_temp, values, kde, bin_edges, 
                                       output_dir, value_range=(-1, 1), 
                                       figsize=histogram_figsize)
            else:
                print(f"Warning: Temperature {hist_temp}K not found in processed data")
    
    return fig, ax, Z_smooth, x_values, temp_smooth

def main():
    """Main function for creating 2D heatmap"""
    
    # Configuration
    temperatures = [10, 50, 100, 150, 200, 250, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 450, 500, 550, 600, 650, 700]
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
    
    # Population curve settings (3 curves for dimer)
    show_population_curves = True
    curve_colors = ['k', 'k', 'k']  # Colors for 3 populations
    curve_linewidth = 2.2
    curve_alpha = 1.0
    show_error_bands = True
    error_band_alpha = 0.08
    
    # Individual histogram settings
    plot_individual_histograms = True  # Set to False to skip histogram plots
    histogram_temps = [50, 600]  # Temperatures for individual histograms (must exist in data)
    horizontal_line_temps = [50, 600]  # Temperatures for horizontal lines (can be interpolated)
    histogram_figsize = (6, 4)  # Figure size for histogram plots (independent of 2D heatmap)
    
    # Check which files exist
    existing_temps = []
    for temp in temperatures:
        csv_filename = os.path.join(output_dir, f'dimer_order_{temp}K.csv')
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
        show_population_curves=show_population_curves,
        curve_colors=curve_colors,
        curve_linewidth=curve_linewidth,
        curve_alpha=curve_alpha,
        show_error_bands=show_error_bands,
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
    print(f"  - Population curves: {show_population_curves}")
    print(f"  - Number of populations: 3")
    print(f"  - Individual histograms: {plot_individual_histograms}")
    if plot_individual_histograms:
        print(f"  - Histogram temperatures: {histogram_temps} K")
    print(f"  - Horizontal line temperatures: {horizontal_line_temps} K")
    print(f"  - Histogram figure size: {histogram_figsize}")
    print("\nGenerated files:")
    print("  1. dimer_order_2d_heatmap.png (2D heatmap with horizontal lines)")
    if plot_individual_histograms:
        for hist_temp in histogram_temps:
            print(f"  {histogram_temps.index(hist_temp)+2}. dimer_order_histogram_{hist_temp}K.png")
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
