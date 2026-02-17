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

def process_rdf_files(file_pattern, max_files=None, chunk_size=100):
    """
    Process multiple RDF files and average their values using chunking.
    
    Args:
        file_pattern: Glob pattern to match RDF files
        max_files: Maximum number of files to process (None = process all)
        chunk_size: Number of files to process in each chunk
    """
    all_files = sorted(glob(file_pattern))
    if max_files is not None:
        all_files = all_files[:max_files]
    
    if not all_files:
        raise FileNotFoundError(f"No files found matching pattern: {file_pattern}")
    
    print(f"Found {len(all_files)} files matching pattern")
    
    # Read the first file to get distances and initialize arrays
    distances, _ = read_rdf_file(all_files[0])
    sum_rdf = np.zeros_like(distances, dtype=np.float64)
    file_count = 0
    
    # Process files in chunks to save memory
    for i in range(0, len(all_files), chunk_size):
        chunk_files = all_files[i:i+chunk_size]
        chunk_sum = np.zeros_like(distances, dtype=np.float64)
        
        for filename in tqdm(chunk_files, desc=f"Processing chunk {i//chunk_size + 1}/{(len(all_files)-1)//chunk_size + 1}"):
            try:
                _, rdf_values = read_rdf_file(filename)
                chunk_sum += rdf_values
                file_count += 1
            except Exception as e:
                print(f"Error processing {filename}: {e}")
        
        # Add chunk sum to total sum
        sum_rdf += chunk_sum
    
    # Calculate the average
    avg_rdf = sum_rdf / file_count if file_count > 0 else sum_rdf
    
    return distances, avg_rdf

def smooth_rdf(distances, rdf_values, window_size=11, poly_order=1):
    """Apply Savitzky-Golay filter to smooth the RDF data."""
    # Make sure window_size is odd and smaller than the data length
    if window_size >= len(rdf_values):
        window_size = min(len(rdf_values) - 1, 3)
    if window_size % 2 == 0:
        window_size -= 1
    
    # Apply the Savitzky-Golay filter for smoothing
    smoothed_rdf = savgol_filter(rdf_values, window_size, poly_order)
    
    # Ensure no negative values in the smoothed RDF
    smoothed_rdf = np.maximum(smoothed_rdf, 0)
    
    return smoothed_rdf

def get_cache_filename(temperatures, distance_range, max_files):
    """Generate unique cache filename based on parameters"""
    params_str = f"{temperatures}_{distance_range}_{max_files}"
    hash_obj = hashlib.md5(params_str.encode())
    return f"cache_rdf_instantaneous_2d_{hash_obj.hexdigest()}.npz"

def load_or_compute_rdfs(temperature_patterns, distance_range=(2.65, 3.75), 
                         max_files_per_temp=100001, window_size=11, poly_order=1,
                         force_recompute=False, output_dir='.'):
    """Load cached RDF data or compute RDFs with smoothing from instantaneous files"""
    
    cache_file = get_cache_filename(list(temperature_patterns.keys()), distance_range, max_files_per_temp)
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
    for temp, pattern in temperature_patterns.items():
        print(f"\nProcessing {temp} data...")
        
        # Check if the pattern matches any files
        if not glob(pattern):
            print(f"Warning: No files found for pattern {pattern}. Skipping {temp}.")
            continue
        
        # Process RDF files (averaging multiple files)
        dist, avg_rdf = process_rdf_files(pattern, max_files=max_files_per_temp)
        
        # Store distances from first temperature (should be same for all)
        if distances is None:
            distances = dist
        
        # Smooth the RDF
        print("Smoothing RDF data...")
        smoothed_rdf = smooth_rdf(dist, avg_rdf, window_size=window_size, poly_order=poly_order)
        
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
    print(f"\nSaving cache to {cache_file}")
    np.savez_compressed(cache_path,
                       distances=distances_filtered,
                       temperatures=temperatures,
                       rdfs=rdfs_dict)
    
    return distances_filtered, temperatures, rdfs_dict

def plot_2d_heatmap(temperature_patterns, output_dir='.', distance_range=(2.65, 3.75),
                   force_recompute=False, colormap='RdBu_r', figsize=(6, 6),
                   max_files_per_temp=100001, window_size=11, poly_order=1,
                   temp_resolution=500, dist_resolution=500,
                   interpolation='linear', horizontal_line_temps=[]):
    """
    Create 2D heatmap of instantaneous RDF data across temperatures
    
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
    max_files_per_temp : int
        Maximum number of instantaneous RDF files to process per temperature
    window_size : int
        Window size for Savitzky-Golay smoothing
    poly_order : int
        Polynomial order for Savitzky-Golay smoothing
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
        max_files_per_temp=max_files_per_temp, window_size=window_size,
        poly_order=poly_order, force_recompute=force_recompute, output_dir=output_dir
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
    cbar = plt.colorbar(im, ax=ax, label=r'$\langle G_{\text{insta}}(r) \rangle$')
    cbar.ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    cbar.ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
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
    fig_filename = os.path.join(output_dir, 'rdf_instantaneous_2d_heatmap.png')
    plt.savefig(fig_filename, dpi=300, bbox_inches='tight')
    print(f"\nSaved plot to {fig_filename}")
    
    plt.show()
    
    return fig, ax, Z_smooth, x_dist, temp_smooth

def main():
    """Main function for creating instantaneous RDF 2D heatmap"""
    
    # Configuration - Define the path patterns to the instantaneous RDF files for different temperatures
    temperature_patterns = {
        '10 K': '../../10K/lammps_out/rdf_analysis/partial_rdfs.*.txt',
        '50 K': '../../50K/lammps_out/rdf_analysis/partial_rdfs.*.txt',
        '100 K': '../../100K/lammps_out/rdf_analysis/partial_rdfs.*.txt',
        '150 K': '../../150K/lammps_out/rdf_analysis/partial_rdfs.*.txt',
        '200 K': '../../200K/lammps_out/rdf_analysis/partial_rdfs.*.txt',
        '250 K': '../../250K/lammps_out/rdf_analysis/partial_rdfs.*.txt',
        '300 K': '../../300K/lammps_out/rdf_analysis/partial_rdfs.*.txt',
        '310 K': '../../310K/lammps_out/rdf_analysis/partial_rdfs.*.txt',
        '320 K': '../../320K/lammps_out/rdf_analysis/partial_rdfs.*.txt',
        '330 K': '../../330K/lammps_out/rdf_analysis/partial_rdfs.*.txt',
        '340 K': '../../340K/lammps_out/rdf_analysis/partial_rdfs.*.txt',
        '350 K': '../../350K/lammps_out/rdf_analysis/partial_rdfs.*.txt',
        '360 K': '../../360K/lammps_out/rdf_analysis/partial_rdfs.*.txt',
        '370 K': '../../370K/lammps_out/rdf_analysis/partial_rdfs.*.txt',
        '380 K': '../../380K/lammps_out/rdf_analysis/partial_rdfs.*.txt',
        '390 K': '../../390K/lammps_out/rdf_analysis/partial_rdfs.*.txt',                
        '400 K': '../../400K/lammps_out/rdf_analysis/partial_rdfs.*.txt',
        '450 K': '../../450K/lammps_out/rdf_analysis/partial_rdfs.*.txt',
        '500 K': '../../500K/lammps_out/rdf_analysis/partial_rdfs.*.txt',
        '550 K': '../../550K/lammps_out/rdf_analysis/partial_rdfs.*.txt',
        '600 K': '../../600K/lammps_out/rdf_analysis/partial_rdfs.*.txt',
        '650 K': '../../650K/lammps_out/rdf_analysis/partial_rdfs.*.txt',
        '700 K': '../../700K/lammps_out/rdf_analysis/partial_rdfs.*.txt',
    }
    
    output_dir = '.'
    distance_range = (2.65, 3.75)  # Angstroms
    
    # Processing parameters
    max_files_per_temp = 100001  # Maximum number of instantaneous RDF files to process
    window_size = 11  # Savitzky-Golay smoothing window
    poly_order = 1  # Savitzky-Golay polynomial order
    
    # 2D heatmap settings (matching the Ru-Ru distances code)
    temp_resolution = 500  # Number of interpolated temperature points
    dist_resolution = 500  # Number of distance points
    interpolation = 'linear'  # 'cubic', 'linear', or 'nearest'
    
    # Visualization settings
    colormap = 'RdBu_r'  # Try: 'Spectral_r', 'RdBu_r', 'RdYlBu_r', 'viridis', 'plasma', 'inferno', 'hot', 'coolwarm'
    figsize = (6, 6)
    horizontal_line_temps = [320, 360]  # Temperatures for horizontal lines
    
    # Create 2D heatmap (will use cache if available)
    fig, ax, Z, x_dist, temp_smooth = plot_2d_heatmap(
        temperature_patterns,
        output_dir=output_dir,
        distance_range=distance_range,
        force_recompute=False,  # Set to True to force recomputation
        colormap=colormap,
        figsize=figsize,
        max_files_per_temp=max_files_per_temp,
        window_size=window_size,
        poly_order=poly_order,
        temp_resolution=temp_resolution,
        dist_resolution=dist_resolution,
        interpolation=interpolation,
        horizontal_line_temps=horizontal_line_temps
    )
    
    print("\nPlotting complete!")
    print("\nSettings used:")
    print(f"  - Max files per temperature: {max_files_per_temp}")
    print(f"  - Savitzky-Golay window size: {window_size}")
    print(f"  - Savitzky-Golay polynomial order: {poly_order}")
    print(f"  - Temperature resolution: {temp_resolution}")
    print(f"  - Distance resolution: {dist_resolution}")
    print(f"  - Interpolation method: {interpolation}")
    print(f"  - Colormap: {colormap}")
    print(f"  - Horizontal line temperatures: {horizontal_line_temps} K")
    print("\nGenerated files:")
    print("  1. rdf_instantaneous_2d_heatmap.png (2D heatmap with horizontal lines)")
    print("\nTo change plot appearance without reprocessing data:")
    print("  - Modify colormap, figsize, temp_resolution, dist_resolution, or interpolation")
    print("  - Modify horizontal_line_temps to add horizontal lines at any temperature")
    print("  - The processed RDF data is cached and will be reused")
    print("\nTo force data reprocessing:")
    print("  - Set force_recompute=True in plot_2d_heatmap()")

if __name__ == "__main__":
    # Set numpy to use less memory
    np.seterr(all='ignore')  # Ignore numpy warnings
    main()
