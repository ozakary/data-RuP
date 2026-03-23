import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import RectBivariateSpline
from glob import glob
import os
from tqdm import tqdm
import seaborn as sns
import hashlib
import warnings
import matplotlib.ticker as ticker
from matplotlib.ticker import LinearLocator, FormatStrFormatter
warnings.filterwarnings('ignore')

# Import the figure_formatting module for consistent plotting style
try:
    import figure_formatting_v2 as ff
    # Set up figure formatting using the function from the module
    ff.set_rcParams(ff.master_formatting)
except ImportError:
    print("Figure formatting module not found. Using default matplotlib settings.")

def read_rdf_file(filename):
    """Read RDF data from a file and extract the distance and g(r) values."""
    distances = []
    g_r_values = []
    
    with open(filename, 'r') as f:
        # Skip the header lines
        for i, line in enumerate(f):
            if i < 2:  # Skip the first two lines (headers)
                continue
            
            parts = line.strip().split()
            if len(parts) >= 2:
                distances.append(float(parts[0]))
                g_r_values.append(float(parts[1]))
    
    return np.array(distances), np.array(g_r_values)

def smooth_rdf(distances, rdf_values, method='gaussian', sigma=3.0):
    """
    Apply smoothing to RDF data with Gaussian filter.
    
    Parameters:
    -----------
    distances : array
        Distance values
    rdf_values : array
        RDF values
    method : str
        'gaussian' for Gaussian smoothing
    sigma : float
        Standard deviation for Gaussian kernel
        Larger values = more smoothing. Typical range: 2.0-8.0
    """
    if method == 'gaussian':
        # Gaussian smoothing
        smoothed_rdf = gaussian_filter1d(rdf_values, sigma=sigma)
    else:
        smoothed_rdf = rdf_values
    
    # Ensure no negative values in the smoothed RDF
    smoothed_rdf = np.maximum(smoothed_rdf, 0)
    
    return smoothed_rdf

def get_cache_filename(temperatures, distance_range):
    """Generate unique cache filename based on parameters"""
    params_str = f"{temperatures}_{distance_range}"
    hash_obj = hashlib.md5(params_str.encode())
    return f"cache_rdf_2d_{hash_obj.hexdigest()}.npz"

def load_or_compute_rdfs(temperature_patterns, distance_range=(2.5, 4.0), 
                         sigma=6.5, force_recompute=False, output_dir='.'):
    """Load cached RDF data or compute RDFs with smoothing"""
    
    cache_file = get_cache_filename(list(temperature_patterns.keys()), distance_range)
    cache_path = os.path.join(output_dir, cache_file)
    
    # Check if cache exists and not forcing recompute
    if os.path.exists(cache_path) and not force_recompute:
        print(f"Loading cached RDF data from {cache_file}")
        data = np.load(cache_path, allow_pickle=True)
        return (data['distances'], 
                data['temperatures'],
                data['rdfs'].item())
    
    print("Computing RDFs with smoothing (this will be cached for future use)...")
    
    rdfs_dict = {}
    distances = None
    temps_list = []
    
    # Process each temperature
    for temp, pattern in tqdm(temperature_patterns.items(), desc="Processing temperatures"):
        # Check if the pattern matches any files
        if not glob(pattern):
            print(f"Warning: No files found for pattern {pattern}. Skipping {temp}.")
            continue
        
        # Read RDF file
        dist, avg_rdf = read_rdf_file(pattern)
        
        # Store distances from first file (should be same for all)
        if distances is None:
            distances = dist
        
        # Smooth the RDF
        smoothed_rdf = smooth_rdf(dist, avg_rdf, method='gaussian', sigma=sigma)
        
        # Extract numeric temperature value
        temp_numeric = float(temp.replace(' K', ''))
        temps_list.append(temp_numeric)
        rdfs_dict[temp_numeric] = smoothed_rdf
    
    # Convert to array
    temperatures = np.array(sorted(temps_list))
    
    # Filter distances to the specified range
    mask = (distances >= distance_range[0]) & (distances <= distance_range[1])
    distances_filtered = distances[mask]
    
    # Filter RDFs to match distance range
    for temp in temperatures:
        rdfs_dict[temp] = rdfs_dict[temp][mask]
    
    # Save to cache
    print(f"Saving cache to {cache_file}")
    np.savez_compressed(cache_path,
                       distances=distances_filtered,
                       temperatures=temperatures,
                       rdfs=rdfs_dict)
    
    return distances_filtered, temperatures, rdfs_dict

def plot_2d_heatmap(temperature_patterns, output_dir='.', distance_range=(2.65, 3.75),
                   force_recompute=False, colormap='RdBu_r', figsize=(6, 6),
                   sigma=6.5, temp_resolution=500, dist_resolution=500,
                   interpolation='linear', horizontal_line_temps=[]):
    """
    Create 2D heatmap of RDF data across temperatures
    
    Parameters:
    -----------
    temperature_patterns : dict
        Dictionary mapping temperature labels to file patterns
    output_dir : str
        Output directory
    distance_range : tuple
        (min, max) distance range in Angstroms
    force_recompute : bool
        Force recomputation even if cache exists
    colormap : str
        Matplotlib colormap name
    figsize : tuple
        Figure size (width, height)
    sigma : float
        Gaussian smoothing parameter for RDF
    temp_resolution : int
        Number of interpolated temperature points
    dist_resolution : int
        Number of distance points for interpolation
    interpolation : str
        Interpolation method ('linear', 'cubic', 'nearest')
    horizontal_line_temps : list
        Temperatures at which to draw horizontal lines
    """
    
    # Load or compute RDF data
    distances, temperatures, rdfs_dict = load_or_compute_rdfs(
        temperature_patterns, distance_range=distance_range,
        sigma=sigma, force_recompute=force_recompute, output_dir=output_dir
    )
    
    print(f"\nCreating 2D heatmap with {len(temperatures)} temperatures...")
    print(f"Distance range: {distance_range[0]:.2f} - {distance_range[1]:.2f} Å")
    print(f"Temperature range: {temperatures.min():.0f} - {temperatures.max():.0f} K")
    
    # Create 2D array from RDF data
    rdf_matrix = np.zeros((len(temperatures), len(distances)))
    for i, temp in enumerate(temperatures):
        rdf_matrix[i, :] = rdfs_dict[temp]
    
    # Create interpolation grid for smooth visualization
    print(f"Interpolating to {temp_resolution}x{dist_resolution} grid...")
    
    # Create smooth temperature and distance grids
    temp_smooth = np.linspace(temperatures.min(), temperatures.max(), temp_resolution)
    x_dist = np.linspace(distances.min(), distances.max(), dist_resolution)
    
    # Interpolate the RDF matrix
    if interpolation == 'cubic':
        interp_func = RectBivariateSpline(temperatures, distances, rdf_matrix, 
                                         kx=3, ky=3, s=0)
    elif interpolation == 'linear':
        interp_func = RectBivariateSpline(temperatures, distances, rdf_matrix, 
                                         kx=1, ky=1, s=0)
    else:  # nearest
        interp_func = RectBivariateSpline(temperatures, distances, rdf_matrix, 
                                         kx=1, ky=1, s=0)
    
    Z_smooth = interp_func(temp_smooth, x_dist)
    
    # Create the plot
    print("Creating heatmap...")
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create meshgrid for pcolormesh
    X, Y = np.meshgrid(x_dist, temp_smooth)
    
    # Plot the heatmap
    im = ax.pcolormesh(X, Y, Z_smooth, cmap=colormap, shading='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label=r'$G_{\text{avg}}(r)$')
    cbar.ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))    
    cbar.ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    # Set number of ticks
    ax.xaxis.set_major_locator(LinearLocator(numticks=3))
    ax.yaxis.set_major_locator(LinearLocator(numticks=6))
    
    # Adding text    
    ax.text(x=3.15, y=165, s='180 K', color='white', va='top', ha='left', rotation=0)
    ax.text(x=3.40, y=380, s='330 K', color='white', va='top', ha='left', rotation=0)    
#    ax.text(0.98, 0.98, 'Heating', transform=ax.transAxes, ha='right', va='top', color='white')        
    
    # Add horizontal dashed lines at specified temperatures
    for line_temp in horizontal_line_temps:
        ax.axhline(y=line_temp, color='white', linestyle='--', linewidth=2.0, alpha=0.8)
    
    # Labels
    ax.set_xlabel(r'$r$(Ru$-$Ru) / Å')
    ax.set_ylabel('$T$ / K')
    
    # Set limits
    ax.set_xlim(distance_range[0], distance_range[1])
    ax.set_ylim(temperatures.min(), temperatures.max())
    
    plt.tight_layout()
    
    # Save figure
    fig_filename = os.path.join(output_dir, 'rdf_2d_heatmap.png')
    plt.savefig(fig_filename, dpi=300, bbox_inches='tight')
    print(f"\nSaved plot to {fig_filename}")
    
    plt.show()
    
    return fig, ax, Z_smooth, x_dist, temp_smooth

def main():
    """Main function for creating RDF 2D heatmap"""
    
    # Configuration - Define the path patterns to the RDF files for different temperatures
    temperature_patterns = {
        '50 K': '../../50K/lammps_out/avg_rdf_analysis/partial_rdfs.txt',
        '100 K': '../../100K/lammps_out/avg_rdf_analysis/partial_rdfs.txt',
        '110 K': '../../110K/lammps_out/avg_rdf_analysis/partial_rdfs.txt',
        '120 K': '../../120K/lammps_out/avg_rdf_analysis/partial_rdfs.txt',
        '130 K': '../../130K/lammps_out/avg_rdf_analysis/partial_rdfs.txt',
        '140 K': '../../140K/lammps_out/avg_rdf_analysis/partial_rdfs.txt',                        
        '150 K': '../../150K/lammps_out/avg_rdf_analysis/partial_rdfs.txt',
        '160 K': '../../160K/lammps_out/avg_rdf_analysis/partial_rdfs.txt',
        '170 K': '../../170K/lammps_out/avg_rdf_analysis/partial_rdfs.txt',
        '180 K': '../../180K/lammps_out/avg_rdf_analysis/partial_rdfs.txt',
        '190 K': '../../190K/lammps_out/avg_rdf_analysis/partial_rdfs.txt',                                
        '200 K': '../../200K/lammps_out/avg_rdf_analysis/partial_rdfs.txt',
        '250 K': '../../250K/lammps_out/avg_rdf_analysis/partial_rdfs.txt',
        '260 K': '../../260K/lammps_out/avg_rdf_analysis/partial_rdfs.txt',
        '270 K': '../../270K/lammps_out/avg_rdf_analysis/partial_rdfs.txt',
        '280 K': '../../280K/lammps_out/avg_rdf_analysis/partial_rdfs.txt',
        '290 K': '../../290K/lammps_out/avg_rdf_analysis/partial_rdfs.txt',        
        '300 K': '../../300K/lammps_out/avg_rdf_analysis/partial_rdfs.txt',
        '310 K': '../../310K/lammps_out/avg_rdf_analysis/partial_rdfs.txt',
        '320 K': '../../320K/lammps_out/avg_rdf_analysis/partial_rdfs.txt',       
        '330 K': '../../330K/lammps_out/avg_rdf_analysis/partial_rdfs.txt',
        '340 K': '../../340K/lammps_out/avg_rdf_analysis/partial_rdfs.txt',                
        '350 K': '../../350K/lammps_out/avg_rdf_analysis/partial_rdfs.txt',
        '400 K': '../../400K/lammps_out/avg_rdf_analysis/partial_rdfs.txt',
        '450 K': '../../450K/lammps_out/avg_rdf_analysis/partial_rdfs.txt',
        '500 K': '../../500K/lammps_out/avg_rdf_analysis/partial_rdfs.txt',
        '550 K': '../../550K/lammps_out/avg_rdf_analysis/partial_rdfs.txt',
        '600 K': '../../600K/lammps_out/avg_rdf_analysis/partial_rdfs.txt',
        '650 K': '../../650K/lammps_out/avg_rdf_analysis/partial_rdfs.txt',
        '700 K': '../../700K/lammps_out/avg_rdf_analysis/partial_rdfs.txt',
    }
    
    output_dir = '.'
    distance_range = (2.65, 3.75)  # Angstroms
    
    # Smoothing parameter for RDF
    sigma = 6.5
    
    # 2D heatmap settings (matching the Ru-Ru distances code)
    temp_resolution = 500  # Number of interpolated temperature points
    dist_resolution = 500  # Number of distance points
    interpolation = 'linear'  # 'cubic', 'linear', or 'nearest'
    
    # Visualization settings
    colormap = 'RdBu_r'  # Try: 'Spectral_r', 'RdBu_r', 'RdYlBu_r', 'viridis', 'plasma', 'inferno', 'hot', 'coolwarm'
    figsize = (6, 6)
    horizontal_line_temps = [180, 330]  # Temperatures for horizontal lines
    
    # Create 2D heatmap (will use cache if available)
    fig, ax, Z, x_dist, temp_smooth = plot_2d_heatmap(
        temperature_patterns,
        output_dir=output_dir,
        distance_range=distance_range,
        force_recompute=False,  # Set to True to force recomputation
        colormap=colormap,
        figsize=figsize,
        sigma=sigma,
        temp_resolution=temp_resolution,
        dist_resolution=dist_resolution,
        interpolation=interpolation,
        horizontal_line_temps=horizontal_line_temps
    )
    
    print("\nPlotting complete!")
    print("\nSettings used:")
    print(f"  - Gaussian smoothing sigma: {sigma}")
    print(f"  - Temperature resolution: {temp_resolution}")
    print(f"  - Distance resolution: {dist_resolution}")
    print(f"  - Interpolation method: {interpolation}")
    print(f"  - Colormap: {colormap}")
    print(f"  - Horizontal line temperatures: {horizontal_line_temps} K")
    print("\nGenerated files:")
    print("  1. rdf_2d_heatmap.png (2D heatmap with horizontal lines)")
    print("\nTo change plot appearance without reprocessing data:")
    print("  - Modify colormap, figsize, temp_resolution, dist_resolution, or interpolation")
    print("  - Modify horizontal_line_temps to add horizontal lines at any temperature")
    print("  - The processed RDF data is cached and will be reused")
    print("\nTo force data reprocessing:")
    print("  - Set force_recompute=True in plot_2d_heatmap()")

if __name__ == "__main__":
    main()
