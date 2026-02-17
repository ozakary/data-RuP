#!/usr/bin/env python3
"""
Compute time correlation function from pre-computed trimer/dimer order CSV files.
Much faster than computing order parameters on-the-fly.
Saves individual NPZ files per temperature.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import time
import argparse
from tqdm import tqdm


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Compute time correlation function from pre-computed order parameter CSV files.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python compute_time_correlation.py -i ./trimer_order_data/ -o ./time_corr_data/
  
  # Custom settings
  python compute_time_correlation.py -i ./data/ -o ./results/ -pattern "trimer_order_*K.csv"
  
  # Specific temperatures only
  python compute_time_correlation.py -i ./data/ -o ./results/ -temps 100K 200K 300K
        """
    )
    
    parser.add_argument('-i', '--input-dir', required=True, type=str,
                        help='Input directory containing CSV files with order parameters')
    
    parser.add_argument('-o', '--output-dir', required=True, type=str,
                        help='Output directory for NPZ files')
    
    parser.add_argument('-prefix', '--file-prefix', default='time_corr', type=str,
                        help='Prefix for output files (default: time_corr)')
    
    parser.add_argument('-pattern', '--csv-pattern', default='trimer_order_*K.csv', type=str,
                        help='Pattern to match CSV files (default: trimer_order_*K.csv)')
    
    parser.add_argument('-maxtime', '--max-time', default=400, type=int,
                        help='Maximum time difference to record (default: 400 frames)')
    
    parser.add_argument('-temps', '--temperatures', nargs='+', default=None,
                        help='Specific temperatures to compute (default: all)')
    
    parser.add_argument('--timestep-ps', default=0.05, type=float,
                        help='Timestep in ps (default: 0.05)')
    
    return parser.parse_args()


def load_order_parameters_from_csv(csv_file):
    """
    Load order parameters from CSV file.
    
    Returns:
    --------
    order_by_frame : dict
        Dictionary mapping timestep to array of order parameters
    """
    print(f"  Loading: {csv_file.name}")
    
    df = pd.read_csv(csv_file)
    
    # Determine order parameter column name (trimer_order or dimer_order)
    if 'trimer_order' in df.columns:
        order_col = 'trimer_order'
    elif 'dimer_order' in df.columns:
        order_col = 'dimer_order'
    else:
        raise ValueError(f"Could not find 'trimer_order' or 'dimer_order' column in {csv_file.name}")
    
    print(f"  Using column: {order_col}")
    
    # Group by timestep
    grouped = df.groupby('timestep')
    
    order_by_frame = {}
    for timestep, group in grouped:
        order_params = group[order_col].values
        order_by_frame[timestep] = order_params
    
    print(f"  Loaded {len(order_by_frame)} timesteps")
    print(f"  Order parameters per frame: ~{len(next(iter(order_by_frame.values())))}")
    
    return order_by_frame


def compute_time_correlation(order_by_frame, max_time=400):
    """
    Compute time correlation function.
    
    Parameters:
    -----------
    order_by_frame : dict
        Dictionary mapping timestep to order parameter array
    max_time : int
        Maximum time difference
    
    Returns:
    --------
    corr : ndarray
        Time correlation function
    """
    # Convert to sorted arrays
    timesteps = sorted(order_by_frame.keys())
    
    # Find minimum length
    min_len = min(len(order_by_frame[t]) for t in timesteps)
    
    print(f"  Trimming to {min_len} atoms per frame")
    
    # Create array of order parameters
    order_array = np.array([order_by_frame[t][:min_len] for t in timesteps])
    
    total_timesteps = len(timesteps)
    max_time = min(max_time, total_timesteps - 1)
    
    print(f"  Computing correlation (max_time={max_time})...")
    
    corr = np.zeros(max_time)
    count = np.zeros(max_time)
    
    # Compute correlation
    for t1 in tqdm(range(total_timesteps), desc="  Computing", unit="step"):
        for t2 in range(t1, total_timesteps):
            del_time = t2 - t1
            if del_time < max_time:
                corr[del_time] += np.dot(order_array[t1], order_array[t2]) / len(order_array[0])
                count[del_time] += 1
    
    corr = corr / count
    
    return corr


def main():
    """Main execution function."""
    args = parse_arguments()
    
    print("="*70)
    print("TIME CORRELATION COMPUTATION (FROM CSV)")
    print("="*70)
    print(f"\nInput directory: {args.input_dir}")
    print(f"CSV pattern: {args.csv_pattern}")
    print(f"Output directory: {args.output_dir}")
    
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"\n✗ Input directory not found: {input_dir}")
        sys.exit(1)
    
    # Find CSV files
    csv_files = sorted(input_dir.glob(args.csv_pattern))
    
    if len(csv_files) == 0:
        print(f"\n✗ No CSV files found matching pattern: {args.csv_pattern}")
        sys.exit(1)
    
    print(f"Found {len(csv_files)} CSV files")
    
    # Filter by temperature if specified
    if args.temperatures is not None:
        temp_set = set(args.temperatures)
        csv_files = [f for f in csv_files if any(t in f.stem for t in temp_set)]
        print(f"Filtered to {len(csv_files)} files for specified temperatures")
    
    print(f"{'='*70}\n")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each CSV file
    results = {}
    
    for csv_file in csv_files:
        # Extract temperature from filename
        import re
        match = re.search(r'(\d+)K', csv_file.stem)
        if not match:
            print(f"⚠ Could not extract temperature from {csv_file.name}, skipping")
            continue
        
        temp = f"{match.group(1)}K"
        
        print(f"{'='*70}")
        print(f"Processing: {temp}")
        print(f"{'='*70}")
        
        start_time = time.time()
        
        try:
            # Load order parameters
            order_by_frame = load_order_parameters_from_csv(csv_file)
            
            # Compute correlation
            corr = compute_time_correlation(order_by_frame, max_time=args.max_time)
            
            # Save result
            output_file = output_dir / f"{args.file_prefix}_{temp}.npz"
            np.savez(
                output_file,
                temperature=temp,
                correlation=corr,
                max_time=args.max_time,
                timestep_ps=args.timestep_ps
            )
            
            elapsed = time.time() - start_time
            print(f"  ✓ Completed in {elapsed:.1f} seconds")
            print(f"  ✓ Saved: {output_file.name}")
            print(f"  Correlation range: [{corr.min():.4f}, {corr.max():.4f}]")
            
            results[temp] = output_file
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
        
        print()
    
    print(f"{'='*70}")
    print(f"COMPLETED: {len(results)} temperatures processed")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
