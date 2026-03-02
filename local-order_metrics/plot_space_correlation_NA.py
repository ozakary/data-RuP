#!/usr/bin/env python4
"""
Plot space correlation functions and extract correlation length.
Creates 3 separate plots: correlation curves and correlation length vs temperature.
update: also performs FFT of the C(r) curve, fits the prominient peak with Lorentzian to extract correlation length: this fit is noisy but trend is similar to the \xi value from integration.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from scipy.ndimage import gaussian_filter1d

from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy.optimize import curve_fit

# Import figure formatting if available
try:
    import figure_formatting_v2 as ff
    ff.set_rcParams(ff.master_formatting)
except ImportError:
    print("Figure formatting module not found. Using default matplotlib settings.")


# ============================================================================
# Analysis Functions
# ============================================================================


def extract_correlation_length(distances, correlation, rmax=None, window_frac=0.05):
    """
    Extract correlation length via the integral method:

      xi = integral(r * |C(r)|, dr) / integral(|C(r)|, dr)

    No fitting or peak detection. Works for both oscillatory and monotonic C(r).

    Parameters
    ----------
    distances   : ndarray  — distance array
    correlation : ndarray  — C(r) array
    rmax        : float    — max distance for integration (exclude flat zero tail)
    window_frac : float    — Savgol smoothing window fraction (default: 0.05)

    Returns
    -------
    xi : float — correlation length in Angstrom
    """
    # Skip r=0 (self-correlation) and apply rmax
    mask = distances >= 0
    if rmax is not None:
        mask = mask & (distances <= rmax)

    r     = distances[mask]
    c_abs = np.abs(correlation[mask])

    denominator = np.trapezoid(c_abs, r)
    if denominator < 1e-12:
        return np.nan

    #xi = np.trapz(r * c_abs, r) / denominator
    #second moment correlation length [this is a standard definition for non-monotonic correlation function]
    xi = np.trapezoid(r*r * c_abs, r) / denominator
    xi=np.sqrt(xi)
    return xi if np.isfinite(xi) and xi > 0 else np.nan


def period_from_fft(distances, correlation,rmax=None, smooth_sigma=0.5):
    """
    Dominant period from (optionally smoothed) FFT power spectrum.
    Returns NaN if no zero crossings exist (C(t) never crosses baseline).
    """
    #if not zero_crossings(z):
    #    return np.nan

    mask = distances >= 0
    #"""
    if rmax is not None:
        mask = mask & (distances <= rmax)
    #"""

    r     = distances[mask]
    c = correlation[mask]
    dr = r[1]-r[0]
    

    freqs = np.array(np.fft.rfftfreq(len(c), d=dr)) ##[1:])  #i believe 2pi factor is not accounted for
    freqs=2*np.pi * freqs

    #corr_smooth = gaussian_filter1d(c, sigma=2.0)
    #power = np.abs(np.fft.rfft(corr_smooth-np.average(corr_smooth)))**2

    power = np.abs(np.fft.rfft(c-np.average(c)))**2
    #print(power)
    #np.abs(np.fft.rfft(c))[1:] #** 2
    #power = np.abs(np.fft.rfft(corr_smooth))[1:] ** 2

    #freqs = np.fft.rfftfreq(len(c), d=dr)
    #power = (np.abs(np.fft.rfft(c)))**2 #np.abs(np.fft.rfft(c)) ** 2
    #freqs = np.fft.fftfreq(len(c), d=dr)
    #power = np.abs(np.fft.fft(c)) ** 2

    

    if smooth_sigma > 0:
        power = gaussian_filter1d(power.astype(float), sigma=smooth_sigma)

    #print(freqs, power)

    return freqs, power
    #f_peak = freqs[np.argmax(power)]
    #return 1.0 / f_peak if f_peak > 0 else np.nan



def lorentzian(q, A,Q,B,xi):
    return A / (1.0 + ((q - Q) * xi)**2) + B

def fit_lorentzian(qvals,signal):

    # choose a peak center guess Q0 and fitting window +/- dq_win
    Q0 = qvals[np.argmax(signal[:-20])]          # or choose one of the two peaks manually
    dq_win = 4*(qvals[1]-qvals[0])              # adjust to include peak shape but not neighbors
    #print(f"dq_win {dq_win}")
    mask = (qvals > Q0 - dq_win) & (qvals < Q0 + dq_win)

    A=signal[mask].max()-signal[mask].min()
    B= signal[mask].min()  
    """
    xi=[Q0,10, A] # initial guess
    ##popt, pcov = curve_fit(lorentzian, qvals[mask], signal[mask], p0=xi, other_args=[A, Q0,B])
    popt, pcov = curve_fit(
        lambda q,Q, xi,A: A / (1.0 + ((q - Q) * xi)**2) + B,
        qvals[mask], signal[mask],
        p0=xi, 
        bounds=([0,0,-20],[10,100,20])
    )
    """
    #fix A??
    xi=[Q0,10] # initial guess
    ##popt, pcov = curve_fit(lorentzian, qvals[mask], signal[mask], p0=xi, other_args=[A, Q0,B])
    popt, pcov = curve_fit(
        lambda q,Q, xi: A / (1.0 + ((q - Q) * xi)**2) + B,
        qvals[mask], signal[mask],
        p0=xi, 
        bounds=([0,0],[10,100])
    )


    #print(popt, pcov)
    #the diagonal of pcov gives variance; take sqrt to get error
    Q_fit=popt[0]
    Q_fit_err=np.sqrt( pcov[0,0])
    xi_fit = popt[1]
    #A_fit=popt[2]
    #print(pcov[0,0])
    xi_err = np.sqrt(pcov[1,1])
    #print(f"Q = {Q_fit:3f}1/Å +/- {Q_fit_err:3f} ξ= {xi_fit:.3f} +/-  {xi_err:3f}Å")

    qvals_dense=np.linspace(np.min(qvals[mask]), np.max(qvals[mask]),50)
    #return xi_fit, xi_err, Q_fit, Q_fit_err, qvals_dense,lorentzian(qvals_dense,A_fit, Q_fit,B,xi_fit)
    return xi_fit, xi_err, Q_fit, Q_fit_err, qvals_dense,lorentzian(qvals_dense,A, Q_fit,B,xi_fit)
    

"""
#fitting with lorentzian is not good because I only have about 4 or 5 points around peak; instead find the width of the peak and take inverse

def find_HWHM()
    # choose a peak center guess Q0 and fitting window +/- dq_win
    Q0 = qvals[np.argmax(signal)]          # or choose one of the two peaks manually
    dq_win = 5*(qvals[1]-qvals[0])              # adjust to include peak shape but not neighbors
    print(f"dq_win {dq_win}")
    mask = (qvals > Q0 - dq_win) & (qvals < Q0 + dq_win)

    max_val=signal[mask].max()-signal[mask].min()

    p0 = [signal[mask].max()-signal[mask].min(), Q0, 10, signal[mask].min()]  # initial guess

"""

# ============================================================================
# Data Loading
# ============================================================================

def load_correlation_data(input_dir, file_prefix='space_corr', selected_temps=None):
    """Load space correlation data from individual NPZ files."""
    input_dir = Path(input_dir)
    
    if not input_dir.exists():
        print(f"✗ Input directory not found: {input_dir}")
        sys.exit(1)
    
    print(f"Loading data from: {input_dir.absolute()}")
    print(f"Looking for files: {file_prefix}_*.npz")
    
    pattern = f"{file_prefix}_*.npz"
    npz_files = sorted(input_dir.glob(pattern))
    
    if len(npz_files) == 0:
        print(f"✗ No files matching pattern '{pattern}' found")
        sys.exit(1)
    
    print(f"Found {len(npz_files)} NPZ files")
    
    temperatures = []
    correlations = []
    distances_list = []
    
    for npz_file in npz_files:
        # Extract temperature from filename
        temp_str = npz_file.stem.replace(f"{file_prefix}_", "")
        
        if selected_temps is not None and temp_str not in selected_temps:
            continue
        
        data = np.load(npz_file, allow_pickle=True)
        
        temp = str(data['temperature'])
        corr = data['correlation']
        distances = data['distances']
        
        temperatures.append(temp)
        correlations.append(corr)
        distances_list.append(distances)
        
        print(f"  ✓ Loaded: {npz_file.name} ({temp})")
    
    if len(temperatures) == 0:
        print(f"✗ No temperature data loaded!")
        sys.exit(1)
    
    print(f"\nLoaded {len(temperatures)} temperatures")
    
    return temperatures, correlations, distances_list


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_space_correlation(temperatures, correlations, distances_list, args):
    """Plot space correlation curves."""
    fig, ax = plt.subplots(figsize=(5.25, 5.25))
    ax.grid(True, axis="y", which="both", alpha=0.5, linewidth=2.0)
    
    # Sort by temperature
    temp_numeric = np.array([float(t.rstrip('K')) for t in temperatures])
    order = np.argsort(temp_numeric)
    
    nT = len(temperatures)
    colors = plt.cm.coolwarm(np.linspace(0, 1, nT))
    
    for idx, i in enumerate(order):
        temp = temperatures[i]
        y = correlations[i]
        sigma = 1.0 #0.5 #len(y) * window_frac / 4
        y_smooth = gaussian_filter1d(y, sigma=sigma)

        r = distances_list[i]
        
        color = colors[idx]
        ax.plot(r, y_smooth, linewidth=2.5, color=color, label=temp)
    
    ax.set_xlabel(r'$r$ / Å')
    ax.set_ylabel(r'$C(r)$')
    ax.set_xlim(args.xmin, args.xmax)
    #ax.set_ylim(-0.15, 0.2)
    
    ax.axhline(y=0, color='grey', linestyle='--', linewidth=2.5)    
    
    # Set number of ticks
    ax.xaxis.set_major_locator(LinearLocator(numticks=5))
    ax.yaxis.set_major_locator(LinearLocator(numticks=8))
    
    if args.ylim is not None:
        ax.set_ylim(args.ylim)
    
#    ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=4)
    
    #output_file = f"{args.output_prefix}_correlation.svg"
    output_file = f"{args.output_prefix}_correlation.png"
    plt.savefig(output_file, bbox_inches='tight')
    print(f"\n✓ Saved: {output_file}")
    
    if args.show:
        plt.show()
    plt.close()


def plot_correlation_length(temperatures, xi_values, args):
    """Plot correlation length vs temperature."""
    fig, ax = plt.subplots(figsize=(5.25, 5.25))
    #ax.grid(True, axis="both", which="both", alpha=0.25, linewidth=0.8)
    
    # Sort by temperature
    temp_numeric = np.array([float(t.rstrip('K')) for t in temperatures])
    order = np.argsort(temp_numeric)
    
    temps_sorted = temp_numeric[order]
    xi_sorted = np.array(xi_values)[order]
    
    nT = len(temperatures)
    colors = plt.cm.coolwarm(np.linspace(0, 1, nT))
    
    # Line plot
    ax.plot(temps_sorted, xi_sorted, '-o', lw=1.6, ms=10, color='0.2', zorder=2)
    
    # Colored points
    for idx, (T, xi) in enumerate(zip(temps_sorted, xi_sorted)):
        ax.scatter([T], [xi], s=60, color=colors[idx],
                  edgecolors='white', linewidths=0.6, zorder=3)
                  
    
    ax.set_xlabel(r'$T$ / K')
    ax.set_ylabel(r'$\xi$ / Å')
    
    ax.set_xlim(0, 750)
    ax.set_ylim(0.0, 15.0)
    
    
    # Set number of ticks
    ax.xaxis.set_major_locator(LinearLocator(numticks=4))
    ax.yaxis.set_major_locator(LinearLocator(numticks=7))
    
    #output_file = f"{args.output_prefix}_correlation_length.svg"
    output_file = f"{args.output_prefix}_correlation_length.png"
    plt.savefig(output_file, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    
    if args.show:
        plt.show()
    plt.close()


def plot_correlation_length_from_fft(temperatures, xi_values, xi_values_error, args):
    """Plot correlation length vs temperature."""
    fig, ax = plt.subplots(figsize=(5.25, 5.25))
    #ax.grid(True, axis="both", which="both", alpha=0.25, linewidth=0.8)
    
    # Sort by temperature
    temp_numeric = np.array([float(t.rstrip('K')) for t in temperatures])
    order = np.argsort(temp_numeric)
    
    temps_sorted = temp_numeric[order]
    xi_sorted = np.array(xi_values)[order]
    xi_err_sorted = np.array(xi_values_error)[order]
    
    nT = len(temperatures)
    colors = plt.cm.coolwarm(np.linspace(0, 1, nT))
    
    # Line plot
    ax.plot(temps_sorted, xi_sorted, '-', lw=1.6, ms=10, color='0.2', zorder=2)
    ##ax.errorbar(temps_sorted, abs(xi_sorted),  yerr=xi_err_sorted, lw=1.6, ms=10, color='0.2', zorder=2)
    ax.errorbar(temps_sorted, abs(xi_sorted),  yerr=xi_err_sorted,
                fmt='none', ecolor='0.4', elinewidth=1.2, capsize=3, zorder=2)
    
    # Colored points
    for idx, (T, xi) in enumerate(zip(temps_sorted, xi_sorted)):
        ax.scatter([T], [xi], s=60, color=colors[idx],
                  edgecolors='white', linewidths=0.6, zorder=3)
                  
    
    ax.set_xlabel(r'$T$ / K')
    ax.set_ylabel(r'$\xi$ / Å')
    
    ax.set_xlim(0, 750)
    ax.set_ylim(0.0, 15.0)
    
    
    # Set number of ticks
    ax.xaxis.set_major_locator(LinearLocator(numticks=4))
    ax.yaxis.set_major_locator(LinearLocator(numticks=7))
    
    #output_file = f"{args.output_prefix}_correlation_length.svg"
    output_file = f"{args.output_prefix}_correlation_length_fft.png"
    plt.savefig(output_file, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    
    if args.show:
        plt.show()
    plt.close()

def plot_fft(temperatures, fft,qvec,lorentzian,qvec2,args):

    """Plot correlation length vs temperature."""


    fig, ax = plt.subplots(figsize=(5.25, 5.25))
    ax.grid(True, axis="y", which="both", alpha=0.5, linewidth=2.0)
    
    # Sort by temperature
    temp_numeric = np.array([float(t.rstrip('K')) for t in temperatures])
    order = np.argsort(temp_numeric)
    
    nT = len(temperatures)
    colors = plt.cm.coolwarm(np.linspace(0, 1, nT))
    
    for idx, i in enumerate(order):
        temp = temperatures[i]
        y = fft[i]
        q = qvec[i]
        #y_smooth = gaussian_filter1d(y, sigma=0.5)

        yy=lorentzian[i]
        qq=qvec2[i]
        
        color = colors[idx]
        #ax.plot(q, y_smooth, linewidth=2.5, color=color, label=temp)
        if (i==0):
            ax.plot(q, y, linewidth=2.5, color=color, label="data")
            ax.plot(qq, yy,  "--", color=color, label="fit")
        else:
            ax.plot(q, y, linewidth=2.5, color=color)
            ax.plot(qq, yy,  "--", color=color)

    #ax.legend()    
    ax.set_xlabel(r'$q$ / Å$^{-1}$')
    ax.set_ylabel(r'$FFT$')
    #ax.set_xlim(args.xmin, args.xmax)
    ax.set_xlim(np.min(qvec2),np.max(qvec2)) 
    #ax.set_ylim(-0.15, 0.2)
    
    ax.axhline(y=0, color='grey', linestyle='--', linewidth=2.5)    
    
    #"""
    # Set number of ticks
    ax.xaxis.set_major_locator(LinearLocator(numticks=5))
    ax.yaxis.set_major_locator(LinearLocator(numticks=8))
    
    if args.ylim is not None:
        ax.set_ylim(args.ylim)
    #"""
    
    ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=4)
    
    #output_file = f"{args.output_prefix}_correlation.svg"
    output_file = f"{args.output_prefix}_fft.png"
    plt.savefig(output_file, bbox_inches='tight')
    print(f"\n✓ Saved: {output_file}")
    
    if args.show:
        plt.show()
    plt.close()




# ============================================================================
# Main
# ============================================================================

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Plot space correlation functions and correlation length.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('-i', '--input-dir', required=True, type=str,
                        help='Input directory containing NPZ files')
    
    parser.add_argument('-o', '--output-prefix', required=True, type=str,
                        help='Output prefix for plot files')
    
    parser.add_argument('-prefix', '--file-prefix', default='space_corr', type=str,
                        help='Prefix of NPZ files to load (default: space_corr)')
    
    parser.add_argument('-temps', '--temperatures', nargs='+', default=None,
                        help='Specific temperatures to plot')
    
    parser.add_argument('--xmin', type=float, default=0,
                        help='X-axis minimum (default: 0)')
    
    parser.add_argument('--xmax', type=float, default=20,
                        help='X-axis maximum (default: 20)')
    
    parser.add_argument('--ylim', nargs=2, type=float, default=None,
                        help='Y-axis limits for correlation plot')
    
    parser.add_argument('--rmax', type=float, default=None,
                        help='Maximum distance for xi integration in A (default: full range). '
                             'Use to exclude flat zero tail, e.g. --rmax 18')
    
    parser.add_argument('--show', action='store_true',
                        help='Show plots interactively')
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()
    
    print("="*70)
    print("SPACE CORRELATION PLOTTING")
    print("="*70)
    
    # Load data
    temperatures, correlations, distances_list = load_correlation_data(
        args.input_dir, args.file_prefix, args.temperatures
    )
    #print(distances_list)
    
    # Extract correlation lengths
    print("\nExtracting correlation lengths...")
    xi_values = []
    fft=[]
    lorentzian=[]
    qvector=[]
    qvector2=[]

    xi_values_fft=[]
    xi_values_fft_error=[]
    
    for temp, corr, dist in zip(temperatures, correlations, distances_list):
        xi = extract_correlation_length(dist, corr, rmax=args.rmax)
        xi_values.append(xi)

        qvals,amp=period_from_fft(dist, corr) #,rmax=None) #, smooth_sigma=2.0)
        fft.append(amp)
        qvector.append(qvals)
        #print(qvals)
        #print(f"Fitting results from Lorentian fit of the FFT for temp {temp}\n")
        xi_fft, xi_fft_error, q_fit, q_fit_err, qqvals,function=fit_lorentzian(qvals,amp)
        lorentzian.append(function)
        qvector2.append(qqvals)

        xi_values_fft.append(xi_fft)
        xi_values_fft_error.append(xi_fft_error)
        
        print(f"  {temp}: ξ_int={xi:.3f} Å, ξ_fft={xi_fft:.3f} +/- {xi_fft_error:.3f}, Q0_fft={q_fit:.3f} +/- {q_fit_err:.3f}")
        #print(f"  {temp}: ξ={xi:.2f} Å")
    
    # Create plots
    print("\nGenerating plots...")
    plot_space_correlation(temperatures, correlations, distances_list, args)
    plot_correlation_length(temperatures, xi_values, args)

    plot_fft(temperatures,fft,qvector,lorentzian,qvector2,args) 
    plot_correlation_length_from_fft(temperatures, xi_values_fft, xi_values_fft_error, args)
    
    print("\n✓ All plots saved!")
    print("="*70)


if __name__ == "__main__":
    main()
