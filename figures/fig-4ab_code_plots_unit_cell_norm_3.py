import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy import stats
from scipy.stats import linregress
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# Optional: Set up figure formatting (uncomment if you have the module)
import figure_formatting as ff
ff.set_rcParams(ff.master_formatting)


def calculate_lattice_properties(lx, ly, lz, xy, xz, yz, supercell=(4, 5, 2)):
    """
    Calculate lattice parameters and angles from LAMMPS triclinic representation
    and convert from supercell to unit cell parameters
    
    Parameters:
    -----------
    lx, ly, lz, xy, xz, yz : float
        LAMMPS triclinic box parameters
    supercell : tuple
        Supercell dimensions (na, nb, nc)
    """
    na, nb, nc = supercell
    
    # Convert from LAMMPS triclinic to standard lattice parameters (supercell)
    a_super = lx
    b_super = np.sqrt(ly**2 + xy**2)
    c_super = np.sqrt(lz**2 + xz**2 + yz**2)
    
    # Convert to unit cell parameters by dividing by supercell dimensions
    a = a_super / na
    b = b_super / nb
    c = c_super / nc
    
    # Calculate angles (in degrees) - angles remain the same for supercell and unit cell
    alpha = np.arccos((xy*xz + ly*yz)/(b_super*c_super)) * 180/np.pi
    beta = np.arccos(xz/c_super) * 180/np.pi
    gamma = np.arccos(xy/b_super) * 180/np.pi
    
    return a, b, c, alpha, beta, gamma

def analyze_temperature_data(supercell=(4, 5, 2)):
    """
    Analyze lattice parameters for all temperatures
    
    Parameters:
    -----------
    supercell : tuple
        Supercell dimensions (na, nb, nc)
    """
    temperatures = [50, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 400, 450, 500, 550, 600, 650, 700]
    results = []
    
    print(f"Converting from {supercell[0]}×{supercell[1]}×{supercell[2]} supercell to unit cell")
    print("="*60)
    
    for temp in temperatures:
        file_path = f"../../{temp}K/lammps_out/lattice_params.dat"
        
        if os.path.exists(file_path):
            print(f"Processing {temp}K...")
            
            # Read the data file (skip comment lines)
            data = []
            with open(file_path, 'r') as f:
                for line in f:
                    if not line.startswith('#'):
                        values = line.strip().split()
                        if len(values) >= 10:  # Ensure we have all columns
                            data.append([float(x) for x in values])
            
            if len(data) == 0:
                print(f"Warning: No data found in {file_path}")
                continue
                
            data = np.array(data)
            
            # Extract LAMMPS triclinic parameters (last 6 columns)
            lx, ly, lz = data[:, 4], data[:, 5], data[:, 6]
            xy, xz, yz = data[:, 7], data[:, 8], data[:, 9]
            
            # Calculate lattice properties for each frame
            lattice_props = []
            valid_frames = []
            
            for i in range(len(data)):
                try:
                    a, b, c, alpha, beta, gamma = calculate_lattice_properties(
                        lx[i], ly[i], lz[i], xy[i], xz[i], yz[i], supercell=supercell)
                    
                    # Check for valid angles (sometimes numerical issues can occur)
                    if not (np.isnan(alpha) or np.isnan(beta) or np.isnan(gamma)):
                        lattice_props.append([a, b, c, alpha, beta, gamma])
                        valid_frames.append(i)
                except:
                    # Skip frames with numerical issues
                    continue
            
            if len(lattice_props) == 0:
                print(f"Warning: No valid lattice parameters calculated for {temp}K")
                continue
                
            lattice_props = np.array(lattice_props)
            
            # Use last 20% of simulation for analysis (equilibrated region)
            n_total = len(lattice_props)
            n_equilibrated = int(n_total * 0.0)
            equilibrated_props = lattice_props[n_equilibrated:]
            
            print(f"  Using {len(equilibrated_props)} equilibrated frames out of {n_total} total")
            print(f"  Unit cell: a={equilibrated_props[-1,0]:.3f} Å, b={equilibrated_props[-1,1]:.3f} Å, c={equilibrated_props[-1,2]:.3f} Å")
            
            # Calculate mean and SEM for each parameter
            n_frames = len(equilibrated_props)
            means = np.mean(equilibrated_props, axis=0)
            stds = np.std(equilibrated_props, axis=0, ddof=1)
            sems = stds / np.sqrt(n_frames)
            
            result = {
                'temperature': temp,
                'a_mean': means[0], 'a_sem': sems[0],
                'b_mean': means[1], 'b_sem': sems[1],
                'c_mean': means[2], 'c_sem': sems[2],
                'alpha_mean': means[3], 'alpha_sem': sems[3],
                'beta_mean': means[4], 'beta_sem': sems[4],
                'gamma_mean': means[5], 'gamma_sem': sems[5],
                'n_frames': n_frames
            }
            results.append(result)
        else:
            print(f"Warning: File {file_path} not found")
    
    return pd.DataFrame(results)

def plot_combined_lattice_parameters(df, reference_temp=50, filename='lattice_abc_absolute.svg', 
                                     shade_region=(100, 150)):
    """
    Plot a, b, and c lattice parameters together as absolute change relative to reference temperature
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame with temperature and lattice parameter data
    reference_temp : float
        Reference temperature for normalization (default: 50 K)
    filename : str
        Output filename
    shade_region : tuple
        Temperature range to shade (T_min, T_max), or None for no shading
    """
    if df.empty:
        print("No data to plot")
        return
    
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    
    # Get reference values at reference temperature
    ref_row = df[df['temperature'] == reference_temp]
    if ref_row.empty:
        print(f"Warning: Reference temperature {reference_temp} K not found in data")
        print(f"Available temperatures: {df['temperature'].values}")
        return
    
    a_ref = ref_row['a_mean'].values[0]
    b_ref = ref_row['b_mean'].values[0]
    c_ref = ref_row['c_mean'].values[0]
    
    print(f"\nReference values at {reference_temp} K:")
    print(f"  a₀ = {a_ref:.4f} Å")
    print(f"  b₀ = {b_ref:.4f} Å")
    print(f"  c₀ = {c_ref:.4f} Å")
    
    # Extract temperatures
    temps = df['temperature'].values
    
    # Calculate absolute changes for each parameter
    params = ['a', 'b', 'c']
    colors = ['#E74C3C', '#3498DB', '#2ECC71']  # Red, Blue, Green
    labels = [r'$\Delta a$', r'$\Delta b$', r'$\Delta c$']
    fmts = ['o-', 'v-', 's-']
    ref_values = [a_ref, b_ref, c_ref]
    
    for param, color, label, fmt, ref_val in zip(params, colors, labels, fmts, ref_values):
        means = df[f'{param}_mean'].values
        sems = df[f'{param}_sem'].values
        
        # Calculate absolute change: ℓ(T) - ℓ₀ (in Angstroms)
        absolute_change = means - ref_val
        
        # Error is just the SEM
        absolute_error = sems
        
        # Plot with error bars
        ax.errorbar(temps, absolute_change, yerr=absolute_error, 
                   color=color, fmt=fmt, capsize=4, capthick=1.5, 
                   linewidth=2, markersize=6, label=label, alpha=0.8)
    
   
    # Adding vertical dashed-lines
    ax.axvline(x=180, linestyle='--', color='gray', linewidth=2.0)
    ax.axvline(x=330, linestyle='--', color='gray', linewidth=2.0)    

    # Adding text    
    ax.text(x=185, y=0.30, s='180 K', color='gray', va='top', ha='left', rotation=0)
    ax.text(x=335, y=0.15, s='330 K', color='gray', va='top', ha='left', rotation=0)    
#    ax.text(0.96, 0.96, 'Heating', transform=ax.transAxes, ha='right', va='top', color='black')
    
    # Formatting
    ax.set_xlim(50, 700)
    ax.set_ylim(-0.2, 0.4)
    ax.set_xlabel(r'$T$ / K')
    ax.set_ylabel(r'$\Delta \ell$ / Å')
    
    # Set number of ticks
    ax.xaxis.set_major_locator(LinearLocator(numticks=6))
    ax.yaxis.set_major_locator(LinearLocator(numticks=4))
    
    # Format y-axis with 2 decimals
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    # Legend
    ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.4, -0.35), ncol=3)
    
    # Grid for easier reading (optional)
#    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=400, bbox_inches='tight')
    plt.show()
    
    print(f"\nSaved {filename}")
    
    # Print summary statistics
    print("\nAbsolute lattice parameter changes (50 K → 700 K):")
    print("-" * 50)
    for param, label in zip(params, labels):
        initial = df[df['temperature'] == 50][f'{param}_mean'].values[0]
        final = df[df['temperature'] == 700][f'{param}_mean'].values[0]
        absolute_change = final - initial
        percent_change = (final - initial) / initial * 100
        print(f"  {label}: {absolute_change:+.4f} Å ({initial:.4f} → {final:.4f} Å, {percent_change:+.2f}%)")


def plot_combined_angles(df, reference_temp=50, filename='angles_alphabetagamma_absolute.svg', 
                         shade_region=(100, 150)):
    """
    Plot alpha, beta, and gamma angles together as absolute change relative to reference temperature
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame with temperature and angle data
    reference_temp : float
        Reference temperature for normalization (default: 50 K)
    filename : str
        Output filename
    shade_region : tuple
        Temperature range to shade (T_min, T_max), or None for no shading
    """
    if df.empty:
        print("No data to plot")
        return
    
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    
    # Get reference values at reference temperature
    ref_row = df[df['temperature'] == reference_temp]
    if ref_row.empty:
        print(f"Warning: Reference temperature {reference_temp} K not found in data")
        print(f"Available temperatures: {df['temperature'].values}")
        return
    
    alpha_ref = ref_row['alpha_mean'].values[0]
    beta_ref = ref_row['beta_mean'].values[0]
    gamma_ref = ref_row['gamma_mean'].values[0]
    
    print(f"\nReference angle values at {reference_temp} K:")
    print(f"  α₀ = {alpha_ref:.4f}°")
    print(f"  β₀ = {beta_ref:.4f}°")
    print(f"  γ₀ = {gamma_ref:.4f}°")
    
    # Extract temperatures
    temps = df['temperature'].values
    
    # Calculate changes for each angle (in degrees, not percentage)
    params = ['alpha', 'beta', 'gamma']
    colors = ['#9B59B6', '#E67E22', '#1ABC9C']  # Purple, Orange, Teal
    labels = [r'$\Delta \alpha$', r'$\Delta \beta$', r'$\Delta \gamma$']
    fmts = ['o-', 'v-', 's-']
    ref_values = [alpha_ref, beta_ref, gamma_ref]
    
    for param, color, label, fmt, ref_val in zip(params, colors, labels, fmts, ref_values):
        means = df[f'{param}_mean'].values
        sems = df[f'{param}_sem'].values
        
        # Calculate absolute change: angle(T) - angle₀ (in degrees)
        angle_change = means - ref_val
        
        # Error is just the SEM
        angle_error = sems
        
        # Plot with error bars
        ax.errorbar(temps, angle_change, yerr=angle_error, 
                   color=color, fmt=fmt, capsize=4, capthick=1.5, 
                   linewidth=2, markersize=6, label=label, alpha=0.8)
    

    # Adding vertical dashed-lines
    ax.axvline(x=180, linestyle='--', color='gray', linewidth=2.0)
    ax.axvline(x=330, linestyle='--', color='gray', linewidth=2.0)
    
    # Adding text    
    ax.text(x=185, y=-0.05, s='180 K', color='gray', va='top', ha='left', rotation=0)
    ax.text(x=335, y=-0.15, s='330 K', color='gray', va='top', ha='left', rotation=0)    
#    ax.text(0.96, 0.96, 'Heating', transform=ax.transAxes, ha='right', va='top', color='black')
   
    # Formatting
    ax.set_xlim(50, 700)
    ax.set_ylim(-0.2, 0.4)
    ax.set_xlabel(r'$T$ / K')
    ax.set_ylabel(r'$\Delta \theta$ / °')
    
    # Set number of ticks
    ax.xaxis.set_major_locator(LinearLocator(numticks=6))
    ax.yaxis.set_major_locator(LinearLocator(numticks=4))
    
    # Format y-axis with 2 decimals
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    # Legend
    ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.4, -0.35), ncol=3)
    
    # Grid for easier reading (optional)
#    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=400, bbox_inches='tight')
    plt.show()
    
    print(f"\nSaved {filename}")
    
    # Print summary statistics
    print("\nAngle change summary (50 K → 700 K):")
    print("-" * 50)
    for param, label in zip(params, labels):
        initial = df[df['temperature'] == 50][f'{param}_mean'].values[0]
        final = df[df['temperature'] == 700][f'{param}_mean'].values[0]
        angle_change = final - initial
        print(f"  {label}: {angle_change:+.4f}° ({initial:.4f}° → {final:.4f}°)")


def main():
    """
    Main analysis function
    """
    print("Starting lattice parameter analysis (UNIT CELL)...")
    print("="*60)
    
    # Define supercell dimensions
    supercell = (4, 5, 2)  # (na, nb, nc)
    
    # Analyze all temperature data
    df = analyze_temperature_data(supercell=supercell)
    
    if df.empty:
        print("Error: No data could be processed!")
        return
    
    print(f"\nSuccessfully processed {len(df)} temperature points")
    print("\nSummary of UNIT CELL results:")
    print(df[['temperature', 'a_mean', 'b_mean', 'c_mean', 'alpha_mean', 'beta_mean', 'gamma_mean']].round(3))
    
    # Save results to CSV
    df.to_csv('lattice_parameters_unitcell_summary.csv', index=False)
    print("\nSaved summary to lattice_parameters_unitcell_summary.csv")
    
    print("\nGenerating plots...")
    print("="*60)
    
    # Plot 1: Combined a, b, c parameters (absolute change in Angstroms)
    print("\n--- Combined lattice parameters (a, b, c) ---")
    plot_combined_lattice_parameters(df, reference_temp=50, 
                                    filename='lattice_abc_absolute.svg',
                                    shade_region=(100, 150))
    
    # Plot 2: Combined alpha, beta, gamma angles (absolute change in degrees)
    print("\n--- Combined angles (α, β, γ) ---")
    plot_combined_angles(df, reference_temp=50,
                        filename='angles_alphabetagamma_absolute.svg',
                        shade_region=(100, 150))
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print(f"Generated 2 combined SVG plots:")
    print(f"  1. lattice_abc_absolute.svg (Δa, Δb, Δc in Å)")
    print(f"  2. angles_alphabetagamma_absolute.svg (Δα, Δβ, Δγ in °)")
    
    # Print key findings
    print("\n" + "="*60)
    print("Key findings:")
    print("-" * 60)
    
    # Lattice parameter changes
    print(f"Lattice parameter changes (50 K → 700 K):")
    for param in ['a', 'b', 'c']:
        initial = df.iloc[0][f'{param}_mean']
        final = df.iloc[-1][f'{param}_mean']
        abs_change = final - initial
        pct_change = (final - initial) / initial * 100
        print(f"  {param}: {abs_change:+.4f} Å ({pct_change:+.2f}%)")
    
    # Angle changes
    print(f"\nAngle changes (50 K → 700 K):")
    for param, symbol in [('alpha', 'α'), ('beta', 'β'), ('gamma', 'γ')]:
        initial = df.iloc[0][f'{param}_mean']
        final = df.iloc[-1][f'{param}_mean']
        abs_change = final - initial
        print(f"  {symbol}: {abs_change:+.4f}°")
    
    # Interpretation
    gamma_range = df['gamma_mean'].max() - df['gamma_mean'].min()
    print(f"\nγ angle total variation: {gamma_range:.4f}°")
    
    if gamma_range > 5.0:
        print("  → Significant γ angle change detected - possible phase transition!")
    elif gamma_range > 1.0:
        print("  → Moderate γ angle change observed")
    else:
        print("  → Small angle changes - stable structure")

if __name__ == "__main__":
    main()
