#!/usr/bin/env python3
r"""
Plot space correlation functions and extract correlation length.
Outputs: C(r) curves, correlation length vs T (integral method),
         FFT power spectrum with Lorentzian fit, and xi from FFT vs T.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from scipy.ndimage import gaussian_filter1d
from matplotlib.ticker import LinearLocator
from scipy.optimize import curve_fit

try:
    import figure_formatting_v2 as ff
    ff.set_rcParams(ff.master_formatting)
except ImportError:
    print("Figure formatting module not found. Using default matplotlib settings.")


# ============================================================================
# Analysis
# ============================================================================

def extract_correlation_length(distances, correlation, rmax=None):
    """
    Extract correlation length via second-moment integral:
      xi = sqrt( integral(r^2 * |C(r)|, dr) / integral(|C(r)|, dr) )
    """
    mask = distances >= 0
    if rmax is not None:
        mask = mask & (distances <= rmax)

    r     = distances[mask]
    c_abs = np.abs(correlation[mask])

    denominator = np.trapz(c_abs, r)
    if denominator < 1e-12:
        return np.nan

    xi = np.sqrt(np.trapz(r * r * c_abs, r) / denominator)
    return xi if np.isfinite(xi) and xi > 0 else np.nan


def period_from_fft(distances, correlation, rmax=None, smooth_sigma=0.5):
    """FFT power spectrum of C(r), frequencies in rad/Å."""
    mask = distances >= 0
    if rmax is not None:
        mask = mask & (distances <= rmax)

    r  = distances[mask]
    c  = correlation[mask]
    dr = r[1] - r[0]

    freqs = 2 * np.pi * np.fft.rfftfreq(len(c), d=dr)
    power = np.abs(np.fft.rfft(c - np.mean(c))) ** 2

    if smooth_sigma > 0:
        power = gaussian_filter1d(power.astype(float), sigma=smooth_sigma)

    return freqs, power


def lorentzian(q, A, Q, B, xi):
    return A / (1.0 + ((q - Q) * xi) ** 2) + B


def fit_lorentzian(qvals, signal):
    """Fit a Lorentzian to the dominant FFT peak.
    Returns xi, xi_err, Q, Q_err, q_dense, fit_curve.
    """
    Q0 = qvals[np.argmax(signal[:-20])]
    dq   = 4 * (qvals[1] - qvals[0])
    mask = (qvals > Q0 - dq) & (qvals < Q0 + dq)

    A = signal[mask].max() - signal[mask].min()
    B = signal[mask].min()

    popt, pcov = curve_fit(
        lambda q, Q, xi: A / (1.0 + ((q - Q) * xi) ** 2) + B,
        qvals[mask], signal[mask],
        p0=[Q0, 10],
        bounds=([0, 0], [10, 100])
    )

    Q_fit, xi_fit = popt
    Q_err, xi_err = np.sqrt(np.diag(pcov))
    q_dense       = np.linspace(qvals[mask].min(), qvals[mask].max(), 50)

    return xi_fit, xi_err, Q_fit, Q_err, q_dense, lorentzian(q_dense, A, Q_fit, B, xi_fit)


# ============================================================================
# Data Loading
# ============================================================================

def load_correlation_data(input_dir, file_prefix='space_corr', selected_temps=None):
    """Load space correlation data from NPZ files."""
    input_dir = Path(input_dir)
    if not input_dir.exists():
        print(f"✗ Input directory not found: {input_dir}")
        sys.exit(1)

    print(f"Loading data from: {input_dir.absolute()}")
    npz_files = sorted(input_dir.glob(f"{file_prefix}_*.npz"))

    if len(npz_files) == 0:
        print(f"✗ No files matching '{file_prefix}_*.npz' found")
        sys.exit(1)

    print(f"Found {len(npz_files)} NPZ files")
    temperatures, correlations, distances_list = [], [], []

    for npz_file in npz_files:
        temp_str = npz_file.stem.replace(f"{file_prefix}_", "")
        if selected_temps is not None and temp_str not in selected_temps:
            continue

        data = np.load(npz_file, allow_pickle=True)
        temperatures.append(str(data['temperature']))
        correlations.append(data['correlation'])
        distances_list.append(data['distances'])
        print(f"  ✓ Loaded: {npz_file.name} ({temp_str})")

    if len(temperatures) == 0:
        print("✗ No temperature data loaded!")
        sys.exit(1)

    print(f"\nLoaded {len(temperatures)} temperatures")
    return temperatures, correlations, distances_list


# ============================================================================
# Plotting
# ============================================================================

def plot_space_correlation(temperatures, correlations, distances_list, args):
    """Plot C(r) curves for all temperatures."""
    fig, ax = plt.subplots(figsize=(5.25, 5.25))
    ax.grid(True, axis="y", which="both", alpha=0.5, linewidth=2.0)

    temp_numeric = np.array([float(t.rstrip('K')) for t in temperatures])
    order  = np.argsort(temp_numeric)
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(temperatures)))

    for idx, i in enumerate(order):
        y_smooth = gaussian_filter1d(correlations[i], sigma=1.0)
        ax.plot(distances_list[i], y_smooth, linewidth=2.5,
                color=colors[idx], label=temperatures[i])

    ax.set_xlabel(r'$r$ / Å')
    ax.set_ylabel(r'$C(r)$')
    ax.set_xlim(args.xmin, args.xmax)
    ax.set_ylim(-0.25, 0.25)
    ax.axhline(y=0, color='grey', linestyle='--', linewidth=2.5)
    ax.xaxis.set_major_locator(LinearLocator(numticks=5))
    ax.yaxis.set_major_locator(LinearLocator(numticks=7))
    if args.ylim is not None:
        ax.set_ylim(args.ylim)

    output_file = f"{args.output_prefix}_correlation.svg"
    plt.savefig(output_file, bbox_inches='tight')
    print(f"\n✓ Saved: {output_file}")
    if args.show:
        plt.show()
    plt.close()


def plot_correlation_length(temperatures, xi_values, args):
    """Plot correlation length (integral method) vs temperature."""
    fig, ax = plt.subplots(figsize=(5.25, 5.25))

    temp_numeric = np.array([float(t.rstrip('K')) for t in temperatures])
    order        = np.argsort(temp_numeric)
    temps_sorted = temp_numeric[order]
    xi_sorted    = np.array(xi_values)[order]
    colors       = plt.cm.coolwarm(np.linspace(0, 1, len(temperatures)))

    ax.plot(temps_sorted, xi_sorted, '-', lw=1.6, color='0.2', zorder=2)
    for idx, (T, xi) in enumerate(zip(temps_sorted, xi_sorted)):
        ax.scatter([T], [xi], s=60, color=colors[idx], edgecolors='white', linewidths=0.6, zorder=3)

    ax.set_xlabel(r'$T$ / K')
    ax.set_ylabel(r'$\xi_{\text{sm}}$ / Å')
    ax.set_xlim(0, 750)
    ax.set_ylim(0.0, 20.0)
    ax.xaxis.set_major_locator(LinearLocator(numticks=4))
    ax.yaxis.set_major_locator(LinearLocator(numticks=6))

    output_file = f"{args.output_prefix}_correlation_length.svg"
    plt.savefig(output_file, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    if args.show:
        plt.show()
    plt.close()


def plot_correlation_length_from_fft(temperatures, xi_values, xi_errors, args):
    """Plot correlation length (FFT/Lorentzian method) vs temperature."""
    fig, ax = plt.subplots(figsize=(5.25, 5.25))

    temp_numeric  = np.array([float(t.rstrip('K')) for t in temperatures])
    order         = np.argsort(temp_numeric)
    temps_sorted  = temp_numeric[order]
    xi_sorted     = np.array(xi_values)[order]
    xi_err_sorted = np.array(xi_errors)[order]
    colors        = plt.cm.coolwarm(np.linspace(0, 1, len(temperatures)))

    ax.plot(temps_sorted, xi_sorted, '-', lw=1.6, color='0.2', zorder=2)
    ax.errorbar(temps_sorted, np.abs(xi_sorted), yerr=xi_err_sorted,
                fmt='none', ecolor='0.4', elinewidth=1.2, capsize=3, zorder=2)
    for idx, (T, xi) in enumerate(zip(temps_sorted, xi_sorted)):
        ax.scatter([T], [xi], s=60, color=colors[idx], edgecolors='white', linewidths=0.6, zorder=3)

    ax.set_xlabel(r'$T$ / K')
    ax.set_ylabel(r'$\xi_{\text{oz}}$ / Å')
    ax.set_xlim(0, 750)
    ax.set_ylim(0.0, 15.0)
    ax.xaxis.set_major_locator(LinearLocator(numticks=4))
    ax.yaxis.set_major_locator(LinearLocator(numticks=7))

    output_file = f"{args.output_prefix}_correlation_length_fft.svg"
    plt.savefig(output_file, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    if args.show:
        plt.show()
    plt.close()


def plot_fft(temperatures, fft_list, qvec_list, lorentzian_list, qvec2_list, args):
    """Plot FFT power spectra with Lorentzian fits for all temperatures."""
    fig, ax = plt.subplots(figsize=(5.25, 5.25))

    temp_numeric = np.array([float(t.rstrip('K')) for t in temperatures])
    order        = np.argsort(temp_numeric)
    colors       = plt.cm.coolwarm(np.linspace(0, 1, len(temperatures)))

    for idx, i in enumerate(order):
        color      = colors[idx]
        label_data = "Data" if idx == 0 else None
        label_fit  = "Fit"  if idx == 0 else None
        ax.plot(qvec_list[i],  fft_list[i],        linewidth=2.5, color=color, label=label_data)
        ax.plot(qvec2_list[i], lorentzian_list[i],  linestyle='--', color=color, label=label_fit)

    ax.set_xlabel(r'$q$ / Å$^{-1}$')
    ax.set_ylabel(r'FFT power')
    ax.set_xlim(np.min(qvec2_list), np.max(qvec2_list))
    ax.set_ylim(0.0, 5.5)
    ax.axhline(y=0, color='grey', linestyle='--', linewidth=2.5)
    ax.xaxis.set_major_locator(LinearLocator(numticks=5))
    ax.yaxis.set_major_locator(LinearLocator(numticks=8))
    if args.ylim is not None:
        ax.set_ylim(args.ylim)
    ax.legend(frameon=False, loc='upper left')

    output_file = f"{args.output_prefix}_fft.svg"
    plt.savefig(output_file, bbox_inches='tight')
    print(f"\n✓ Saved: {output_file}")
    if args.show:
        plt.show()
    plt.close()


# ============================================================================
# Main
# ============================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Plot space correlation functions and correlation length.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('-i',      '--input-dir',     required=True, type=str)
    parser.add_argument('-o',      '--output-prefix', required=True, type=str)
    parser.add_argument('-prefix', '--file-prefix',   default='space_corr', type=str)
    parser.add_argument('-temps',  '--temperatures',  nargs='+', default=None)
    parser.add_argument('--xmin',  type=float, default=0)
    parser.add_argument('--xmax',  type=float, default=20)
    parser.add_argument('--ylim',  nargs=2, type=float, default=None)
    parser.add_argument('--rmax',  type=float, default=None,
                        help='Max distance for xi integration (Å), e.g. --rmax 18')
    parser.add_argument('--show',  action='store_true')
    return parser.parse_args()


def main():
    args = parse_arguments()
    Path(args.output_prefix).parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SPACE CORRELATION PLOTTING")
    print("=" * 70)

    temperatures, correlations, distances_list = load_correlation_data(
        args.input_dir, args.file_prefix, args.temperatures
    )

    print("\nExtracting correlation lengths...")
    xi_values, xi_fft_values, xi_fft_errors = [], [], []
    fft_list, lorentzian_list, qvec_list, qvec2_list = [], [], [], []

    for temp, corr, dist in zip(temperatures, correlations, distances_list):
        xi = extract_correlation_length(dist, corr, rmax=args.rmax)
        xi_values.append(xi)

        qvals, amp = period_from_fft(dist, corr)
        fft_list.append(amp)
        qvec_list.append(qvals)

        xi_fft, xi_fft_err, q_fit, q_fit_err, qqvals, fit_curve = fit_lorentzian(qvals, amp)
        lorentzian_list.append(fit_curve)
        qvec2_list.append(qqvals)
        xi_fft_values.append(xi_fft)
        xi_fft_errors.append(xi_fft_err)

        print(f"  {temp}: ξ_int={xi:.3f} Å  ξ_fft={xi_fft:.3f} ± {xi_fft_err:.3f}  Q0={q_fit:.3f} ± {q_fit_err:.3f}")

    print("\nGenerating plots...")
    plot_space_correlation(temperatures, correlations, distances_list, args)
    plot_correlation_length(temperatures, xi_values, args)
    plot_fft(temperatures, fft_list, qvec_list, lorentzian_list, qvec2_list, args)
    plot_correlation_length_from_fft(temperatures, xi_fft_values, xi_fft_errors, args)

    print("\n✓ All plots saved!")
    print("=" * 70)


if __name__ == "__main__":
    main()
