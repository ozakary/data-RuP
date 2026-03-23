import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.ticker import StrMethodFormatter
from ase.io import read
# Import the figure_formatting module
import figure_formatting_v2 as ff
# Set up figure formatting using the function from the module
ff.set_rcParams(ff.master_formatting)

def create_joint_plot(x_data, y_data, xlabel, ylabel, filename, scale_factor=1.0, 
                     tick_format='{x:.3f}', stats_format=None, stats_position='top-left'):
    """Create and save a joint plot with marginal distributions.
    
    Args:
        x_data: X-axis data
        y_data: Y-axis data
        xlabel: X-axis label
        ylabel: Y-axis label
        filename: Output filename
        scale_factor: Factor by which data has been scaled for display (default=1.0)
        tick_format: Format string for tick labels (default='{x:.3f}')
        stats_format: Dictionary with format strings for R2, RMSE, and MAE 
                     (default={'r2': 6, 'rmse': 6, 'mae': 6})
        stats_position: Position of statistics text, either 'top-left' or 'bottom-right' (default='top-left')
    """
    # Set default stats formatting if not provided
    if stats_format is None:
        stats_format = {'r2': 6, 'rmse': 6, 'mae': 6}
    
    plt.figure(figsize=(6, 6))
    
    # Create the joint plot
    g = sns.JointGrid(data=None, x=x_data, y=y_data, height=6)
    
    # Add the scatter plot with specific color and size
    g.plot_joint(plt.scatter, alpha=0.5, color='#E9C46A', s=100, edgecolor='#E9C46A')
    
    # Add marginal distributions with matching color
    n_bins = 50
    g.plot_marginals(sns.histplot, color='#E9C46A', kde=False, bins=n_bins)
    
    # Update KDE line color in marginal plots
    for ax in [g.ax_marg_x, g.ax_marg_y]:
        for line in ax.lines:
            line.set_color('#E9C46A')
            
    # Add labels and title
    g.ax_joint.set_xlabel(xlabel)
    g.ax_joint.set_ylabel(ylabel)
    # Control number of ticks
    x_min, x_max = g.ax_joint.get_xlim()
    y_min, y_max = g.ax_joint.get_ylim()
    
    overall_min = min(x_min, y_min)
    overall_max = max(x_max, y_max)
    
    # Create 4 evenly spaced ticks
    x_ticks = np.linspace(overall_min, overall_max, 4)
    y_ticks = np.linspace(overall_min, overall_max, 4)
    
    g.ax_joint.set_xticks(x_ticks)
    g.ax_joint.set_yticks(y_ticks)
 
    # Format tick labels using the custom format
    g.ax_joint.xaxis.set_major_formatter(StrMethodFormatter(tick_format))
    g.ax_joint.yaxis.set_major_formatter(StrMethodFormatter(tick_format))
    # Rotate x-axis tick labels by 45 degrees
    plt.setp(g.ax_joint.get_xticklabels(), rotation=45, ha='right')
    plt.setp(g.ax_joint.get_yticklabels(), rotation=45, ha='right')
    # Calculate statistics on the raw (unscaled) data
    # Rescale the data if necessary
    x_raw = x_data / scale_factor
    y_raw = y_data / scale_factor
    
    # Add regression statistics on raw data
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_raw, y_raw)
    rmse = np.sqrt(np.mean((x_raw - y_raw)**2))
    mae = np.mean(np.abs(x_raw - y_raw))
    
    # Display the statistics (raw data) with custom formatting
    stats_text = f'R² = {r_value**2:.{stats_format["r2"]}f}\nRMSE = {rmse:.{stats_format["rmse"]}f}\nMAE = {mae:.{stats_format["mae"]}f}'
    
    # Set position based on stats_position parameter
    if stats_position == 'bottom-right':
        g.ax_joint.text(0.95, 0.05, stats_text,
                       transform=g.ax_joint.transAxes,
                       verticalalignment='bottom',
                       horizontalalignment='right')
    else:  # default to top-left
        g.ax_joint.text(0.05, 0.75, stats_text,
                       transform=g.ax_joint.transAxes)
 
    # Add regression line (for display data)
    slope_display, intercept_display, _, _, _ = stats.linregress(x_data, y_data)
    line_x = np.array([min(x_data), max(x_data)])
    line_y = slope_display * line_x + intercept_display
    g.ax_joint.plot(line_x, line_y, alpha=1, label=f'y = {slope_display:.4f}x + {intercept_display:.4f}', color='#264653')
    # Plot identity line (x=y) for reference
    min_val = min(g.ax_joint.get_xlim()[0], g.ax_joint.get_ylim()[0])
    max_val = max(g.ax_joint.get_xlim()[1], g.ax_joint.get_ylim()[1])
    g.ax_joint.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3)
    # Save the plot
    plt.savefig(filename, dpi=400, bbox_inches='tight')
    plt.close()

def export_to_csv(data, filename):
    """Export data to a CSV file."""
    with open(filename, 'w') as f:
        # Write header
        header = ','.join(str(key) for key in data.keys())
        f.write(header + '\n')
        
        # Write data rows
        for i in range(len(next(iter(data.values())))):
            row = ','.join(str(data[key][i]) if i < len(data[key]) else '' for key in data.keys())
            f.write(row + '\n')
    
    print(f"Data exported to {filename}")

def main():
    # Define input filename
    input_file = 'rup_dataset_test_predictions.xyz'
    
    print(f"Reading structures from {input_file}...")
    # Use ASE to read all structures
    structures = read(input_file, index=':', format='extxyz')
    print(f"Loaded {len(structures)} structures")
    
    # Extract energies and separate by phase
    print("\nExtracting energies and separating by phase...")
    dft_energies_mono = []  # 144 atoms
    ml_energies_mono = []
    dft_energies_ortho = []  # 216 atoms
    ml_energies_ortho = []
    
    for atoms in structures:
        n_atoms = len(atoms)
        if 'REF_energy' in atoms.info and 'MACE_energy' in atoms.info:
            if n_atoms == 144:
                dft_energies_mono.append(atoms.info['REF_energy'])
                ml_energies_mono.append(atoms.info['MACE_energy'])
            elif n_atoms == 216:
                dft_energies_ortho.append(atoms.info['REF_energy'])
                ml_energies_ortho.append(atoms.info['MACE_energy'])
    
    dft_energies_mono = np.array(dft_energies_mono)
    ml_energies_mono = np.array(ml_energies_mono)
    dft_energies_ortho = np.array(dft_energies_ortho)
    ml_energies_ortho = np.array(ml_energies_ortho)
    
    print(f"  Found {len(dft_energies_mono)} monoclinic (144 atoms) energy pairs")
    print(f"  Found {len(dft_energies_ortho)} orthorhombic (216 atoms) energy pairs")
    
    # Also keep combined data for other analyses
    dft_energies = np.concatenate([dft_energies_mono, dft_energies_ortho])
    ml_energies = np.concatenate([ml_energies_mono, ml_energies_ortho])
    
    # Extract forces
    print("\nExtracting forces...")
    dft_forces_list = []
    ml_forces_list = []
    for atoms in structures:
        if 'REF_forces' in atoms.arrays and 'MACE_forces' in atoms.arrays:
            dft_forces_list.append(atoms.arrays['REF_forces'])
            ml_forces_list.append(atoms.arrays['MACE_forces'])
    
    dft_forces = np.vstack(dft_forces_list)
    ml_forces = np.vstack(ml_forces_list)
    print(f"  Found {len(dft_forces)} force vectors")
    
    # Extract stress
    print("\nExtracting stress...")
    dft_stress_list = []
    ml_stress_list = []
    for atoms in structures:
        if 'REF_stress' in atoms.info and 'MACE_stress' in atoms.info:
            dft_stress_list.append(atoms.info['REF_stress'])
            ml_stress_list.append(atoms.info['MACE_stress'])
    
    if len(dft_stress_list) > 0:
        dft_stress = np.array(dft_stress_list)
        ml_stress = np.array(ml_stress_list)
        print(f"  Found {len(dft_stress)} stress tensors")
    else:
        print("  No stress data found")
        dft_stress = None
        ml_stress = None
    
    # Print some verification
    print("\n" + "="*60)
    print("Data Verification:")
    print(f"\nMonoclinic phase (144 atoms):")
    print(f"  Energy range - DFT: [{dft_energies_mono.min():.2f}, {dft_energies_mono.max():.2f}] eV")
    print(f"  Energy range - ML:  [{ml_energies_mono.min():.2f}, {ml_energies_mono.max():.2f}] eV")
    print(f"\nOrthorhombic phase (216 atoms):")
    print(f"  Energy range - DFT: [{dft_energies_ortho.min():.2f}, {dft_energies_ortho.max():.2f}] eV")
    print(f"  Energy range - ML:  [{ml_energies_ortho.min():.2f}, {ml_energies_ortho.max():.2f}] eV")
    print(f"\nCombined:")
    print(f"  Energy range - DFT: [{dft_energies.min():.2f}, {dft_energies.max():.2f}] eV")
    print(f"  Energy range - ML:  [{ml_energies.min():.2f}, {ml_energies.max():.2f}] eV")
    print(f"  Force range - DFT: [{dft_forces.min():.4f}, {dft_forces.max():.4f}] eV/Å")
    print(f"  Force range - ML:  [{ml_forces.min():.4f}, {ml_forces.max():.4f}] eV/Å")
    if dft_stress is not None:
        print(f"  Stress range - DFT: [{dft_stress.min():.6f}, {dft_stress.max():.6f}] eV/Å³")
        print(f"  Stress range - ML:  [{ml_stress.min():.6f}, {ml_stress.max():.6f}] eV/Å³")
    print("="*60 + "\n")
    
    # Define custom formatters
    energy_format = '{x:.3f}'
    force_format = '{x:.2f}'
    stress_format = '{x:.0f}'
    
    # Define custom statistics formatting
    energy_stats_format = {'r2': 4, 'rmse': 3, 'mae': 3}
    force_stats_format = {'r2': 4, 'rmse': 4, 'mae': 4}
    stress_stats_format = {'r2': 4, 'rmse': 6, 'mae': 6}
    
    # ========== ENERGY PLOTS ==========
    print("Creating energy comparison plots...")
    
    # Export energy data with phase information
    phase_labels = ['monoclinic'] * len(dft_energies_mono) + ['orthorhombic'] * len(dft_energies_ortho)
    energy_data = {
        'structure_index': np.arange(len(dft_energies)),
        'phase': phase_labels,
        'dft_energy': dft_energies,
        'ml_energy': ml_energies
    }
    export_to_csv(energy_data, 'energy_comparison.csv')
    
# Energy plot with scaling - MONOCLINIC PHASE
    display_scale = 1e-3
    if len(dft_energies_mono) > 0:
        print("  Creating monoclinic phase energy plot...")
        create_joint_plot(
            dft_energies_mono*display_scale, ml_energies_mono*display_scale,
            r'$E_{\text{DFT}}^{\text{mono}}$ $\times10^{-3}$ / eV', r'$E_{\text{ML}}^{\text{mono}}$ $\times10^{-3}$ / eV',
            'energy_comparison_monoclinic.svg',
            scale_factor=display_scale,
            tick_format=energy_format,
            stats_format=energy_stats_format,
            stats_position='bottom-right'
        )
    
    # Energy plot with scaling - ORTHORHOMBIC PHASE
    if len(dft_energies_ortho) > 0:
        print("  Creating orthorhombic phase energy plot...")
        create_joint_plot(
            dft_energies_ortho*display_scale, ml_energies_ortho*display_scale,
            r'$E_{\text{DFT}}^{\text{ortho}}$ $\times10^{-3}$ / eV', r'$E_{\text{ML}}^{\text{ortho}}$ $\times10^{-3}$ / eV',
            'energy_comparison_orthorhombic.svg',
            scale_factor=display_scale,
            tick_format=energy_format,
            stats_format=energy_stats_format,
            stats_position='bottom-right'
        )
    
    # ========== FORCE PLOTS ==========
    print("Creating force comparison plots...")
    
    # Export force data
    force_data = {
        'atom_index': np.arange(len(dft_forces)),
        'dft_force_x': dft_forces[:,0],
        'ml_force_x': ml_forces[:,0],
        'dft_force_y': dft_forces[:,1],
        'ml_force_y': ml_forces[:,1],
        'dft_force_z': dft_forces[:,2],
        'ml_force_z': ml_forces[:,2]
    }
    export_to_csv(force_data, 'forces_comparison.csv')
    
    # Total forces
    create_joint_plot(
        dft_forces.flatten(), ml_forces.flatten(),
        r'$\vec{f}_{\text{DFT}}$ / eV·Å$^{-1}$', r'$\vec{f}_{\text{ML}}$ / eV·Å$^{-1}$',
        'forces_total_comparison.png',
        scale_factor=1.0,
        tick_format=force_format,
        stats_format=force_stats_format
    )
    
    # Individual force components
    components = [r'$f_{x}$', r'$f_{y}$', r'$f_{z}$']
    components_2 = ['fx', 'fy', 'fz']
    for i in range(3):
        create_joint_plot(
            dft_forces[:,i], ml_forces[:,i],
            components[i] + r'$^{{\text{DFT}}}$ / eV·Å$^{{-1}}$',
            components[i] + r'$^{{\text{ML}}}$ / eV·Å$^{{-1}}$',
            f'force_{components_2[i]}_comparison.png',
            scale_factor=1.0,
            tick_format=force_format,
            stats_format=force_stats_format
        )
    
    # ========== STRESS PLOTS ==========
    if dft_stress is not None:
        print("Creating stress comparison plots...")
        
        # Export stress data
        stress_data = {'structure_index': np.arange(len(dft_stress))}
        stress_components = [
            ('xx', 0), ('yy', 1), ('zz', 2),
            ('xy', 3), ('xz', 4), ('yz', 5),
            ('yx', 6), ('zx', 7), ('zy', 8)
        ]
        
        for label, idx in stress_components:
            if idx < dft_stress.shape[1]:
                stress_data[f'dft_stress_{label}'] = dft_stress[:,idx]
                stress_data[f'ml_stress_{label}'] = ml_stress[:,idx]
        
        export_to_csv(stress_data, 'stress_comparison.csv')
        
        # Scale for display
        display_scale = 1e3
        dft_stress_scaled = dft_stress * display_scale
        ml_stress_scaled = ml_stress * display_scale
        
        # Total stress
        create_joint_plot(
            dft_stress_scaled.flatten(), ml_stress_scaled.flatten(),
            r'$\boldsymbol{s}_{\text{DFT}}$ $\times10^{3}$ / eV·Å$^{-3}$', 
            r'$\boldsymbol{s}_{\text{ML}}$ $\times10^{3}$ / eV·Å$^{-3}$',
            'stress_total_comparison.svg',
            scale_factor=display_scale,
            tick_format=stress_format,
            stats_format=stress_stats_format
        )
        
        # Individual stress components
        dft_stress_template = r'$s_{{{}}}^{{\text{{DFT}}}} \times10^{{3}}$ / eV·Å$^{{-3}}$'
        ml_stress_template = r'$s_{{{}}}^{{\text{{ML}}}} \times10^{{3}}$ / eV·Å$^{{-3}}$'
        
        for label, idx in stress_components:
            if idx < dft_stress.shape[1]:
                create_joint_plot(
                    dft_stress_scaled[:,idx], ml_stress_scaled[:,idx],
                    dft_stress_template.format(label),
                    ml_stress_template.format(label),
                    f'stress_{label}_comparison.svg',
                    scale_factor=display_scale,
                    tick_format=stress_format,
                    stats_format=stress_stats_format
                )
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("  - Plots saved as SVG files")
    print("  - Separate energy plots created for monoclinic and orthorhombic phases")
    print("  - Data exported to CSV files")
    print("="*60)

if __name__ == "__main__":
    main()
