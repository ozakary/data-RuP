import numpy as np
from ase.io import read
from dynasor.core.reciprocal import calc_rho_q
from pathlib import Path
import sys
import glob
from tqdm import tqdm
import time
import argparse

# ---- Build a 1D q-path (polyline) in CARTESIAN units (rad/Å) for ONE frame ----
def qpath_cart_from_cell(cell, qpts_frac, n_per_seg):
    """
    Convert fractional q-points to cartesian coordinates.
    
    Parameters:
    -----------
    cell : array_like
        Unit cell matrix
    qpts_frac : array_like
        Q-points in fractional coordinates, shape (K, 3)
    n_per_seg : int
        Number of points per segment
    
    Returns:
    --------
    q_cart : ndarray
        Q-points in cartesian coordinates (rad/Å)
    tick_idx : ndarray
        Indices for tick marks
    """
    A = np.asarray(cell, dtype=np.float64)
    B = 2*np.pi * np.linalg.inv(A).T  # reciprocal basis, rows = b1,b2,b3 [1/Å]
    qs = []
    tick_idx = [0]
    
    for a, b in zip(qpts_frac[:-1], qpts_frac[1:]):
        ts = np.linspace(0.0, 1.0, n_per_seg, endpoint=False)
        seg_frac = a + (b - a)[None, :] * ts[:, None]
        qs.append(seg_frac @ B)  # (n_per_seg,3) in rad/Å
        tick_idx.append(tick_idx[-1] + len(ts))
    
    qs.append(qpts_frac[-1][None, :] @ B)  # final endpoint
    q_cart = np.vstack(qs)  # (Nq,3)
    
    return q_cart, np.array(tick_idx)


# ---- S(q) for a single frame (total) using dynasor's kernel (fast) ----
def Sq_frame_total(positions, q_cart):
    """
    Calculate structure factor for a single frame.
    
    Parameters:
    -----------
    positions : ndarray
        Atomic positions, shape (N_atoms, 3)
    q_cart : ndarray
        Q-points in cartesian coordinates
    
    Returns:
    --------
    rho : ndarray
        Density in reciprocal space (normalized)
    """
    rho = calc_rho_q(positions, q_cart)  # complex array (Nq,)
    return rho / (positions.shape[0]**0.5)


# ---- Main computation: Read XYZ and compute S(q) ----
def compute_Sq_from_xyz(xyz_file, q_cart, start_frame=0, step=1, max_frames=None):
    """
    Compute S(q) from XYZ trajectory file.
    
    Parameters:
    -----------
    xyz_file : str or Path
        Path to XYZ trajectory file
    q_cart : ndarray
        Q-points in cartesian coordinates
    start_frame : int
        Starting frame index (default: 0)
    step : int
        Step between frames (default: 1, use all frames)
    max_frames : int or None
        Maximum number of frames to process (None = all)
    
    Returns:
    --------
    qnorm_avg : ndarray
        Average q-point norms (for x-axis)
    S_mean : ndarray
        Mean structure factor
    S_error : ndarray
        Standard error of the mean
    """
    xyz_file = Path(xyz_file)
    
    if not xyz_file.exists():
        raise FileNotFoundError(f"XYZ file not found: {xyz_file}")
    
    print(f"  Reading trajectory file: {xyz_file.name}")
    
    # Read all frames starting from start_frame
    index_str = f"{start_frame}:"
    frames = read(str(xyz_file), index=index_str)
    
    total_frames = len(frames)
    print(f"  Total frames available: {total_frames}")
    
    # Determine which frames to use
    if max_frames is not None:
        frames_to_use = min(max_frames, total_frames)
    else:
        frames_to_use = total_frames
    
    # Account for step
    frames_to_use = len(range(0, frames_to_use, step))
    print(f"  Processing {frames_to_use} frames (step={step})")
    
    S_stack = []
    qnorm_stack = []
    
    # Process frames with progress bar
    frame_count = 0
    for i, at in enumerate(tqdm(frames, desc="  Computing S(q)", unit="frame")):
        if i % step == 0:
            Sq = Sq_frame_total(at.get_positions(), q_cart)
            S_stack.append(Sq)
            qnorm_stack.append(np.linalg.norm(q_cart, axis=1))
            frame_count += 1
            
            if max_frames is not None and frame_count >= max_frames:
                break
    
    # Stack and compute statistics
    S_stack = np.vstack(S_stack)  # shape (n_frames, Nq)
    
    # Compute S(q) = <|rho(q)|^2>
    S_mean = np.mean(S_stack * S_stack.conj(), axis=0).real
    S_std = np.std(S_stack * S_stack.conj(), axis=0).real
    S_error = S_std / np.sqrt(len(S_stack))
    
    qnorm_avg = np.mean(np.vstack(qnorm_stack), axis=0)
    
    print(f"  ✓ Processed {len(S_stack)} frames successfully")
    
    return qnorm_avg, S_mean, S_error


# ---- Process multiple temperatures ----
def process_all_temperatures(base_dir, temperatures, q_cart, file_pattern, 
                            start_frame=0, step=1, max_frames=None,
                            subdirectory="lammps_out"):
    """
    Process XYZ files for all temperatures.
    
    Parameters:
    -----------
    base_dir : str or Path
        Base directory containing temperature folders
    temperatures : list
        List of temperature strings (e.g., ['10K', '50K', ...])
    q_cart : ndarray
        Q-points in cartesian coordinates
    file_pattern : str
        Pattern to match XYZ files
    start_frame : int
        Starting frame index
    step : int
        Step between frames
    max_frames : int or None
        Maximum frames to process per temperature
    subdirectory : str
        Subdirectory within each temperature folder containing XYZ files
    
    Returns:
    --------
    results : dict
        Dictionary with temperature as key and (qnorm, S_mean, S_error) as value
    """
    base_dir = Path(base_dir)
    
    print("="*70)
    print("STRUCTURE FACTOR CALCULATION")
    print("="*70)
    print(f"\nBase directory: {base_dir.absolute()}")
    print(f"File pattern: *{file_pattern}*.xyz")
    print(f"Subdirectory: {subdirectory}")
    print(f"Number of temperatures: {len(temperatures)}")
    print(f"Temperatures: {', '.join(temperatures)}")
    print(f"\nFrame settings:")
    print(f"  Start frame: {start_frame}")
    print(f"  Step: {step}")
    print(f"  Max frames per temperature: {'All' if max_frames is None else max_frames}")
    print(f"\nQ-path settings:")
    print(f"  Number of q-points: {len(q_cart)}")
    print(f"  Q-range: {np.linalg.norm(q_cart[0]):.4f} - {np.linalg.norm(q_cart[-1]):.4f} rad/Å")
    print("="*70)
    
    results = {}
    
    for temp in temperatures:
        print(f"\n{'='*70}")
        print(f"Processing temperature: {temp}")
        print(f"{'='*70}")
        
        # Construct path to XYZ file
        temp_dir = base_dir / temp / subdirectory
        
        if not temp_dir.exists():
            print(f"  ✗ Directory not found: {temp_dir}")
            print(f"  Skipping {temp}...")
            continue
        
        # Find matching XYZ file
        pattern = f"*{file_pattern}*.xyz"
        matching_files = list(temp_dir.glob(pattern))
        
        if len(matching_files) == 0:
            print(f"  ✗ No files matching pattern '{pattern}' found in {temp_dir}")
            print(f"  Skipping {temp}...")
            continue
        elif len(matching_files) > 1:
            print(f"  ⚠ Warning: Multiple files found:")
            for f in matching_files:
                print(f"    - {f.name}")
            print(f"  Using first file: {matching_files[0].name}")
        
        xyz_file = matching_files[0]
        
        try:
            # Compute S(q)
            start_time = time.time()
            qnorm, S_mean, S_error = compute_Sq_from_xyz(
                xyz_file, q_cart, 
                start_frame=start_frame, 
                step=step, 
                max_frames=max_frames
            )
            elapsed = time.time() - start_time
            
            results[temp] = {
                'qnorm': qnorm,
                'S_mean': S_mean,
                'S_error': S_error,
                'file': str(xyz_file)
            }
            
            print(f"  ✓ Completed in {elapsed:.1f} seconds")
            print(f"  S(q) range: {S_mean.min():.4e} - {S_mean.max():.4e}")
            
        except Exception as e:
            print(f"  ✗ Error processing {temp}: {str(e)}")
            print(f"  Skipping {temp}...")
            continue
    
    return results


# ---- Save results ----
def save_results(results, qpath_frac, output_dir, prefix="sq"):
    """
    Save S(q) results to individual NPZ files per temperature.
    
    Parameters:
    -----------
    results : dict
        Results dictionary from process_all_temperatures
    qpath_frac : ndarray
        Q-path in fractional coordinates
    output_dir : str or Path
        Output directory for NPZ files
    prefix : str
        Prefix for output files (default: "sq")
    """
    if len(results) == 0:
        print("\n✗ No results to save!")
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print(f"{'='*70}")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"File prefix: {prefix}")
    
    saved_files = []
    
    for temp in sorted(results.keys(), key=lambda x: float(x.rstrip('K'))):
        output_file = output_dir / f"{prefix}_{temp}.npz"
        
        # Save individual temperature data
        np.savez(
            output_file,
            temperature=temp,
            qpath_frac=qpath_frac,
            qnorm=results[temp]['qnorm'],
            S_mean=results[temp]['S_mean'],
            S_error=results[temp]['S_error'],
            source_file=results[temp]['file']
        )
        
        saved_files.append(output_file.name)
        print(f"  ✓ Saved: {output_file.name}")
    
    print(f"\nTotal files saved: {len(saved_files)}")
    print(f"{'='*70}\n")
    
    return saved_files


# ---- Parse command-line arguments ----
def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Calculate structure factor S(q) from MD trajectory XYZ files at multiple temperatures.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - saves individual files per temperature
  python findSq_improved.py -parent ../../../ -i rup_traj_sampled-50_100ps -o ./sq_data/
  
  # With custom prefix and subdirectory
  python findSq_improved.py -parent ../../../ -i traj -o ./results/ -prefix mysim -subdir data
  
  # Process every 10th frame starting from frame 100
  python findSq_improved.py -parent ../../../ -i traj -o ./sq_data/ -start 100 -step 10
  
  # Specify custom temperature list
  python findSq_improved.py -parent ../../../ -i traj -o ./sq_data/ -temps 100K 200K 300K 400K
  
  # Add a single new temperature without recomputing all
  python findSq_improved.py -parent ../../../ -i traj -o ./sq_data/ -temps 450K
        """
    )
    
    # Required arguments
    parser.add_argument('-parent', '--parent-dir', required=True, type=str,
                        help='Parent directory containing temperature folders (e.g., ../../../)')
    
    parser.add_argument('-i', '--input-pattern', required=True, type=str,
                        help='Pattern to match XYZ files (e.g., rup_traj_sampled-50_100ps)')
    
    parser.add_argument('-o', '--output-dir', required=True, type=str,
                        help='Output directory for NPZ files (e.g., ./sq_data/)')
    
    parser.add_argument('-prefix', '--file-prefix', default='sq', type=str,
                        help='Prefix for output files (default: sq, creates sq_100K.npz, sq_200K.npz, etc.)')
    
    # Optional arguments
    parser.add_argument('-subdir', '--subdirectory', default='lammps_out', type=str,
                        help='Subdirectory within temperature folders (default: lammps_out)')
    
    parser.add_argument('-ref', '--reference', default='monoclinic_fully_relaxed.vasp', type=str,
                        help='Reference structure file for q-path (default: monoclinic_fully_relaxed.vasp)')
    
    parser.add_argument('-start', '--start-frame', default=0, type=int,
                        help='Starting frame index (default: 0)')
    
    parser.add_argument('-step', '--frame-step', default=1, type=int,
                        help='Step between frames (default: 1, use all frames)')
    
    parser.add_argument('-max', '--max-frames', default=None, type=int,
                        help='Maximum frames to process (default: None, process all)')
    
    parser.add_argument('-nq', '--nq-per-segment', default=50, type=int,
                        help='Number of q-points per segment (default: 50)')
    
    parser.add_argument('-temps', '--temperatures', nargs='+', default=None,
                        help='List of temperature folders (e.g., 100K 200K 300K). If not specified, uses default list.')
    
    return parser.parse_args()


# ---- Main execution ----
if __name__ == "__main__":
    
    # Parse command-line arguments
    args = parse_arguments()
    
    # Define temperatures
    if args.temperatures is not None:
        temperatures = args.temperatures
    else:
        temperatures = [
            "50K", "100K", "110K", "120K", "130K", "140K", "150K", "160K", 
            "170K", "180K", "190K", "200K", "250K", "260K", "270K", "280K",
            "290K", "300K", "310K", "320K", "330K", "340K", "350K", "400K",
            "450K", "500K", "550K", "600K", "650K", "700K"
        ]
    
    # Define q-path in fractional coordinates
    # This path shows clear distinction between phases
    qpath_frac = np.array([
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
        [0, 0, 1],
    ])
    
    # Read reference structure for q-path conversion
    if not Path(args.reference).exists():
        print(f"✗ Reference structure not found: {args.reference}")
        print("  Please make sure this file is in the current directory.")
        sys.exit(1)
    
    print(f"Reading reference structure: {args.reference}")
    mono_pure_cell = read(args.reference)
    
    # Convert q-path to cartesian coordinates
    q_cart, tick_idx = qpath_cart_from_cell(
        mono_pure_cell.cell.array, 
        qpath_frac, 
        args.nq_per_segment
    )
    
    # Process all temperatures
    results = process_all_temperatures(
        base_dir=args.parent_dir,
        temperatures=temperatures,
        q_cart=q_cart,
        file_pattern=args.input_pattern,
        start_frame=args.start_frame,
        step=args.frame_step,
        max_frames=args.max_frames,
        subdirectory=args.subdirectory
    )
    
    # Save results (individual files per temperature)
    save_results(results, qpath_frac, args.output_dir, prefix=args.file_prefix)
    
    print("✓ All done!")
