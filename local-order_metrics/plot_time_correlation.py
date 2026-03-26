#!/usr/bin/env python3
"""
Plot time correlation functions and extract characteristic times.
Creates 3 separate plots: correlation curves, relaxation times, C_inf plateau.
Also gives diagnostics_*.svg for each temperature showing fit vs data and residual.

Fits C(t) = A * exp(-t/tau) * cos(2*pi*freq*t + phi) + C_inf
freq is estimated from FFT, C_inf from the tail; only A, tau, phi are free.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import sys
from functools import partial
from matplotlib.ticker import LinearLocator
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


try:
    import figure_formatting_v2 as ff
    ff.set_rcParams(ff.master_formatting)
except ImportError:
    print("Figure formatting module not found. Using default matplotlib settings.")


# ============================================================================
# FFT and model
# ============================================================================

def period_from_fft(z, dt, smooth_sigma=1.0):
    """Dominant frequency from FFT power spectrum of baseline-subtracted signal."""
    freqs = np.fft.rfftfreq(len(z), d=dt)
    power = np.abs(np.fft.rfft(z - np.mean(z))) ** 2
    if smooth_sigma > 0:
        power = gaussian_filter1d(power.astype(float), sigma=smooth_sigma)
    power[0] = 0
    f_peak = freqs[np.argmax(power)]
    return f_peak if f_peak > 0 else np.nan

def model_oscillating_decay(t, A, tau, phi, B, tau2, C_inf, freq):
    """C(t) = A*exp(-(t/tau)^2)*cos(2*pi*freq*t + phi) + B*exp(-t/tau2) + C_inf"""
    return (A * np.exp(-(t / tau) ** 2) * np.cos(2 * np.pi * freq * t + phi)
            + B * np.exp(-t / tau2) + C_inf)


def fit_relaxation(t, y, freq_guess=1, tmax=None):
    """
    Fit C(t) = A*exp(-t/tau)*cos(2*pi*freq*t + phi) + B*exp(-t/tau2) + C_inf.
    freq and C_inf are fixed; free parameters are A, tau, phi, B, tau2.
    Returns A, tau, phi, B, tau2, C_inf, perr, fit_ok.
    """
    if tmax is not None:
        mask = t <= tmax
        t, y = t[mask], y[mask]

    freq_guess = np.clip(freq_guess, 0.1, 20.0) if np.isfinite(freq_guess) else 1.0
    
    tail   = y[-max(5, int(0.05 * len(y))):]
    Cinf0  = max(0.0, float(np.mean(tail)))
    A0     = max(y[0] - Cinf0, 1e-6)
    tau0   = t[-1] / 4
    phi0   = 0.0
    B0     = A0 * 0.3
    tau2_0 = t[-1] / 10

    bounds_lo = [0,      1e-4,   -np.pi, 0,      1e-4,   0.1 ]
    bounds_hi = [np.inf, np.inf,  np.pi, np.inf, np.inf, 20.0]
    best_cost, best_popt, best_pcov = np.inf, None, None
    rng = np.random.default_rng(42)

    # try original guess first, then random restarts
    p0_candidates = [[A0, tau0, phi0, B0, tau2_0, freq_guess]]
    for _ in range(20):
        p0_candidates.append([
            rng.uniform(0,       A0 * 2),
            rng.uniform(1e-4,    t[-1]),
            rng.uniform(-np.pi,  np.pi),
            rng.uniform(0,       A0),
            rng.uniform(1e-4,    t[-1] / 2),
            rng.uniform(0.5,     10.0),
        ])

    for p0 in p0_candidates:
        try:
            popt, pcov = curve_fit(
                lambda t, A, tau, phi, B, tau2, freq: model_oscillating_decay(
                    t, A, tau, phi, B, tau2, Cinf0, freq),
                t, y,
                p0=p0,
                bounds=(bounds_lo, bounds_hi),
                maxfev=100000,
                method='trf'
            )
            residuals = y - model_oscillating_decay(t, *popt[:5], Cinf0, popt[5])
            cost = np.sum(residuals ** 2)
            if cost < best_cost:
                best_cost, best_popt, best_pcov = cost, popt, pcov
        except RuntimeError:
            continue

    if best_popt is None:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.array([np.nan] * 6), False

    perr = np.sqrt(np.diag(best_pcov))
    A_fit, tau_fit, phi_fit, B_fit, tau2_fit, freq_fit = best_popt
    return A_fit, tau_fit, phi_fit, B_fit, tau2_fit, freq_fit, Cinf0, perr, True


# ============================================================================
# Data loading
# ============================================================================

def load_correlation_data(input_dir, file_prefix='time_corr', selected_temps=None):
    """Load time correlation data from NPZ files."""
    input_dir = Path(input_dir)
    if not input_dir.exists():
        print(f"✗ Input directory not found: {input_dir}")
        sys.exit(1)

    npz_files = sorted(input_dir.glob(f"{file_prefix}_*.npz"))
    if len(npz_files) == 0:
        print(f"✗ No files matching '{file_prefix}_*.npz' found")
        sys.exit(1)

    print(f"Found {len(npz_files)} NPZ files")
    temperatures, correlations, timesteps = [], [], []

    for npz_file in npz_files:
        temp_str = npz_file.stem.replace(f"{file_prefix}_", "")
        if selected_temps is not None and temp_str not in selected_temps:
            continue
        data = np.load(npz_file, allow_pickle=True)
        temperatures.append(str(data['temperature']))
        correlations.append(data['correlation'])
        timesteps.append(float(data['timestep_ps']))
        print(f"  ✓ {npz_file.name}")

    if len(temperatures) == 0:
        print("✗ No data loaded!")
        sys.exit(1)

    print(f"\nLoaded {len(temperatures)} temperatures")
    return temperatures, correlations, timesteps


# ============================================================================
# Plotting
# ============================================================================

def plot_time_correlation(temperatures, correlations, timesteps, args):
    """Plot C(t) curves for all temperatures."""
    fig, ax = plt.subplots(figsize=(5.25, 5.25))
    ax.grid(True, axis="y", which="both", alpha=0.5, linewidth=2.0)

    temp_numeric = np.array([float(t.rstrip('K')) for t in temperatures])
    order  = np.argsort(temp_numeric)
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(temperatures)))

    for idx, i in enumerate(order):
        t = np.arange(len(correlations[i])) * timesteps[i]
        ax.plot(t, correlations[i], linewidth=2.5, color=colors[idx], label=temperatures[i])

    # --- Inset: zoom into first 5 ps ---
    axins = inset_axes(ax, width="50%", height="40%", loc='upper right',
                       bbox_to_anchor=(0.02, 0.05, 0.9, 0.92),
                       bbox_transform=ax.transAxes)
    for idx, i in enumerate(order):
        t = np.arange(len(correlations[i])) * timesteps[i]
        zoom_mask = t <= 5.0
        axins.plot(t[zoom_mask], correlations[i][zoom_mask],
                   linewidth=1.2, color=colors[idx])
    axins.axhline(0, color='grey', linestyle='--', linewidth=1.5)
    axins.set_xlim(0.0, 1.0)
    axins.set_ylim(-0.05, 0.3)
    axins.xaxis.set_major_locator(LinearLocator(numticks=3))
    axins.yaxis.set_major_locator(LinearLocator(numticks=3))


    ax.set_xlabel(f't / {args.unit}')
    ax.set_ylabel(r'$C(t)$')
    ax.set_xlim(0.0, 20.0)
    ax.set_ylim(-0.05, 0.3)
    ax.xaxis.set_major_locator(LinearLocator(numticks=5))
    ax.yaxis.set_major_locator(LinearLocator(numticks=8))
    ax.axhline(y=0, color='grey', linestyle='--', linewidth=2.5)
    if args.ylim is not None:
        ax.set_ylim(args.ylim)
#    ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=10)

    output_file = f"{args.output_prefix}_correlation.svg"
    plt.savefig(output_file, bbox_inches='tight')
    print(f"\n✓ Saved: {output_file}")
    if args.show:
        plt.show()
    plt.close()


def plot_relaxation_time(temperatures, tau_values, tau_errors, args):
    """Plot relaxation time vs temperature with error bars."""
    fig, ax = plt.subplots(figsize=(5.25, 5.25))

    temp_numeric = np.array([float(t.rstrip('K')) for t in temperatures])
    order        = np.argsort(temp_numeric)
    temps_sorted = temp_numeric[order]
    tau_sorted   = np.array(tau_values)[order]
    err_sorted   = np.array(tau_errors)[order]
    colors       = plt.cm.coolwarm(np.linspace(0, 1, len(temperatures)))

    ax.plot(temps_sorted, tau_sorted, '-', lw=1.6, color='0.2', zorder=2)
    ax.errorbar(temps_sorted, tau_sorted, yerr=err_sorted,
                fmt='none', ecolor='0.4', elinewidth=1.2, capsize=3, zorder=2)
    for idx, (T, tau) in enumerate(zip(temps_sorted, tau_sorted)):
        ax.scatter([T], [tau], s=60, color=colors[idx], edgecolors='white', linewidths=0.6, zorder=3)

    ax.set_xlabel(r'$T$ / K')
    ax.set_ylabel(r'$\tau_{2}$ / ' + args.unit)
    ax.set_xlim(0, 750)
    ax.set_ylim(0.0, 5.0)
    ax.xaxis.set_major_locator(LinearLocator(numticks=4))
    ax.yaxis.set_major_locator(LinearLocator(numticks=6))
#    ax.axhline(y=0, color='grey', linestyle='--', linewidth=2.5)

    output_file = f"{args.output_prefix}_relaxation_time.svg"
    plt.savefig(output_file, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    if args.show:
        plt.show()
    plt.close()


def plot_freq_osc(temperatures, freq_osc_values, freq_osc_errors, args):
    """Plot oscillation frequency vs temperature."""
    fig, ax = plt.subplots(figsize=(5.25, 5.25))

    temp_numeric  = np.array([float(t.rstrip('K')) for t in temperatures])
    order         = np.argsort(temp_numeric)
    temps_sorted  = temp_numeric[order]
    freq_osc_sorted  = np.array(freq_osc_values)[order]
    colors        = plt.cm.coolwarm(np.linspace(0, 1, len(temperatures)))
    valid         = np.isfinite(freq_osc_sorted)

    invalid_temps = temps_sorted[~valid]
    if len(invalid_temps) > 0:
        ax.axvspan(invalid_temps.min() - 5, invalid_temps.max() + 5,
                   alpha=0.12, color='gray', zorder=0)
        ax.text(460, 0.2, 'no oscillations', ha='center', va='center',
                fontsize=20, color='dimgray', style='italic', rotation=90, transform=ax.transData)

    err_sorted = np.array(freq_osc_errors)[order]
    ax.plot(temps_sorted[valid], freq_osc_sorted[valid], '-', lw=1.6, color='0.2', zorder=2)
    ax.errorbar(temps_sorted[valid], freq_osc_sorted[valid], yerr=err_sorted[valid],
                fmt='none', ecolor='0.4', elinewidth=1.2, capsize=3, zorder=2)
    for idx, (T, freq_osc) in enumerate(zip(temps_sorted, freq_osc_sorted)):
        if np.isfinite(freq_osc):
            ax.scatter([T], [freq_osc], s=60, color=colors[idx], edgecolors='white', linewidths=0.6, zorder=3)

    ax.set_xlabel(r'$T$ / K')
    ax.set_ylabel(r'$f$ / ' + args.freq_osc_unit)
    ax.set_xlim(0, 750)
    ax.set_ylim(0, 5)
    ax.xaxis.set_major_locator(LinearLocator(numticks=4))
    ax.yaxis.set_major_locator(LinearLocator(numticks=6))

    output_file = f"{args.output_prefix}_freq_osc-osc.svg"
    plt.savefig(output_file, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    if args.show:
        plt.show()
    plt.close()


def plot_cinf(temperatures, cinf_values, args):
    """Plot C_inf plateau vs temperature."""
    fig, ax = plt.subplots(figsize=(5.25, 5.25))

    temp_numeric = np.array([float(t.rstrip('K')) for t in temperatures])
    order        = np.argsort(temp_numeric)
    temps_sorted = temp_numeric[order]
    cinf_sorted  = np.array(cinf_values)[order]
    colors       = plt.cm.coolwarm(np.linspace(0, 1, len(temperatures)))

    ax.plot(temps_sorted, cinf_sorted, '-', lw=1.6, color='0.2', zorder=2)
    for idx, (T, cinf) in enumerate(zip(temps_sorted, cinf_sorted)):
        ax.scatter([T], [cinf], s=60, color=colors[idx], edgecolors='white', linewidths=0.6, zorder=3)

    ax.set_xlabel(r'$T$ / K')
    ax.set_ylabel(r'$C(t\rightarrow \infty)$')
    ax.set_xlim(0, 750)
    ax.set_ylim(-0.02, 0.32)
    ax.xaxis.set_major_locator(LinearLocator(numticks=4))
    ax.yaxis.set_major_locator(LinearLocator(numticks=6))
    ax.axhline(y=0, color='grey', linestyle='--', linewidth=2.5)

    output_file = f"{args.output_prefix}_cinf.svg"
    plt.savefig(output_file, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    if args.show:
        plt.show()
    plt.close()


def plot_diagnostics(t, y, prediction, temp, Rf, args):
    """Two-panel diagnostic plot: fit vs data (top) and residual (bottom), with zoom inset."""
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    residual = y - prediction

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7),
                                   gridspec_kw={'height_ratios': [3, 1]},
                                   sharex=True)

    # --- Top panel: data vs fit ---
    ax1.plot(t, y, color='steelblue', linewidth=2.0, label='Data')
    ax1.plot(t, prediction, color='tomato', linewidth=2.0, linestyle='--', label='Fit')
    ax1.axhline(0, color='grey', linestyle='--', linewidth=2.0)
    ax1.set_ylabel('C(t)')
    ax1.set_title(f'T = {temp}', y=0.95)
    leg = ax1.legend(frameon=True, loc = "lower right")
    leg.get_frame().set_edgecolor('gray')
    leg.get_frame().set_alpha(0.25)

    y_min  = min(np.min(y), np.min(prediction))
    y_max  = max(np.max(y), np.max(prediction))
    margin = 0.05 * (y_max - y_min)
    ax1.set_ylim(y_min - margin, y_max + margin)
    ax1.set_xlim(0.0, 10.0)

    # --- Inset: zoom into first 20% of time range ---
    t_zoom_max = t[-1] * 0.20
    zoom_mask  = t <= t_zoom_max

    axins = inset_axes(ax1, width="50%", height="40%", loc='upper right',
                       bbox_to_anchor=(0.02, 0.05, 0.9, 0.92),
                       bbox_transform=ax1.transAxes)

    axins.plot(t[zoom_mask], y[zoom_mask], color='steelblue', linewidth=1.2)
    axins.plot(t[zoom_mask], prediction[zoom_mask], color='tomato',
               linewidth=2.0, linestyle='--')
    axins.axhline(0, color='grey', linestyle='--', linewidth=2.0)
    axins.set_xlim(0.0, 0.5)

    y_zoom_min = min(np.min(y[zoom_mask]), np.min(prediction[zoom_mask]))
    y_zoom_max = max(np.max(y[zoom_mask]), np.max(prediction[zoom_mask]))
    zm = 0.05 * (y_zoom_max - y_zoom_min)
    axins.set_ylim(y_zoom_min - zm, y_zoom_max + zm)

    # --- Bottom panel: residual ---
    ax2.plot(t, residual, color='#666666', linewidth=2.0)
    ax2.axhline(0, color='black', linestyle='--', linewidth=2.0)
    ax2.grid(axis='y', linestyle=':', alpha=0.7)
    ax2.set_xlabel(f't / {args.unit}')
    ax2.set_ylabel(r'$\Delta C(t)$')
    ax2.text(0.98, 0.85, f'$G$ = {Rf:.3f}%', transform=ax2.transAxes,
             ha='right', va='top')

    r_abs = max(np.abs(residual).max(), 1e-10)
    ax2.set_ylim(-r_abs * 1.2, r_abs * 1.2)
    ax2.set_xlim(0.0, 10.0)

    plt.tight_layout()
    fig.subplots_adjust(hspace=0.08)

    outfile = f"{args.output_prefix}_diagnostics_{temp}.svg"
    plt.savefig(outfile, bbox_inches='tight')
    plt.close()

def plot_fft_spectrum(temperatures, correlations, timesteps, args, smooth_sigma=2.0):
    """Plot FFT power spectrum as offset curves (ridge plot) with temperature axis."""
    fig, ax = plt.subplots(figsize=(5.25, 5.25))

    temp_numeric = np.array([float(t.rstrip('K')) for t in temperatures])
    order  = np.argsort(temp_numeric)
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(temperatures)))

    y_offset = 0.3  # adjust this to control spacing between curves

    for idx in range(len(order) - 1, -1, -1):
        i = order[idx]
        offset = idx * y_offset
        color  = colors[idx]
        y  = correlations[i]
        dt = timesteps[i]

        base  = float(np.mean(y[-max(5, int(0.10 * len(y))):]))
        z     = y - base
        freqs = np.fft.rfftfreq(len(z), d=dt)[1:]
        power = np.abs(np.fft.rfft(z))[1:] ** 2
        if smooth_sigma > 0:
            power = gaussian_filter1d(power.astype(float), sigma=smooth_sigma)
        power = power / power.max() if power.max() > 0 else power

        ax.plot(freqs, power*2 + offset, linewidth=2.0, color=colors[idx],
                label=temperatures[i])
        ax.fill_between(freqs, offset, power*2 + offset,
                        alpha=0.05, color=colors[idx])
        # Baseline for each curve
        ax.axhline(offset, color=colors[idx], linewidth=0.5, linestyle='--', alpha=0.4)

    ax.set_xlabel(r'$f$ / THz')
    ax.set_ylabel(r'$T$ / K')

    # Replace y ticks with temperature labels
    ax.set_yticks([idx * y_offset for idx in range(len(order))])
    ax.set_yticks([idx * y_offset + 0.5 * y_offset for idx in range(len(order))], minor=True)
    ax.set_yticklabels([temperatures[i] for i in order])

    ax.set_xlim(0, 15)
    ax.grid(False)

    ax.set_ylabel('')
    ax.yaxis.set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    output_file = f"{args.output_prefix}_fft_spectrum.svg"
    plt.savefig(output_file, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    if args.show:
        plt.show()
    plt.close()

# ============================================================================
# Main
# ============================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Plot time correlation functions and characteristic times.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('-i', '--input-dir', required=True, type=str)
    parser.add_argument('-o', '--output-prefix', required=True, type=str)
    parser.add_argument('-prefix', '--file-prefix', default='time_corr', type=str)
    parser.add_argument('-temps', '--temperatures', nargs='+', default=None)
    parser.add_argument('--unit', default='ps', type=str)
    parser.add_argument('--freq_osc-unit', default='THz', type=str)
    parser.add_argument('--xmin', type=float, default=0)
    parser.add_argument('--xmax', type=float, default=None)
    parser.add_argument('--ylim', nargs=2, type=float, default=None)
    parser.add_argument('--tmax', type=float, default=None)
    parser.add_argument('--smooth-sigma', type=float, default=2.0)
    parser.add_argument('--show', action='store_true')
    return parser.parse_args()


def main():
    args = parse_arguments()

    print("=" * 70)
    print("TIME CORRELATION PLOTTING")
    print("=" * 70)
    if args.tmax:
        print(f"Fitting up to tmax = {args.tmax} ps")

    temperatures, correlations, timesteps = load_correlation_data(
        args.input_dir, args.file_prefix, args.temperatures
    )

    print("\nFitting C(t) = A·exp(-t/τ)·cos(2π·freq·t + φ) + C_inf ...")
    print(f"  {'Temp':<10} {'tau2':>10} {'±':>8} {'A':>10} {'±':>8} {'phi':>8} {'±':>8} {'tau1':>8} {'±':>8} {'B':>8} {'±':>8} {'cinf':>8} {'freq':>8} {'±':>8} {'OK':>4}")
    print(f"  {'-'*10} {'-'*10} {'-'*8} {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*4}")

    tau_values, tau_errors, cinf_values, freq_values, freq_errors = [], [], [], [], []
    A_values, A_errors, phi_values, phi_errors = [], [], [], []
    tau1_values, tau1_errors, B_values, B_errors = [], [], [], []

    for temp, y, dt in zip(temperatures, correlations, timesteps):
        t = np.arange(len(y)) * dt

        base     = float(np.mean(y[-max(5, int(0.10 * len(y))):]))
        freq_fft = period_from_fft(y - base, dt, smooth_sigma=args.smooth_sigma)

        A, tau, phi_fit, B, tau2, freq_fit, cinf, perr, ok = fit_relaxation(t, y, freq_guess=freq_fft, tmax=args.tmax)
        prediction = model_oscillating_decay(t, A, tau, phi_fit, B, tau2, cinf, freq_fit)

        if args.tmax is not None:
            mask = t <= args.tmax
            Rf = (1 - np.sum(np.abs(prediction[mask] - y[mask])) / max(np.sum(np.abs(y[mask])), 1e-10)) * 100
        else:
            Rf = (1 - np.sum(np.abs(prediction - y)) / max(np.sum(np.abs(y)), 1e-10)) * 100

        plot_diagnostics(t, y, prediction, temp, Rf, args)

        tau_values.append(tau2)
        tau_errors.append(perr[4])
        cinf_values.append(cinf)
        freq_values.append(freq_fit)
        freq_errors.append(perr[5])
        A_values.append(A)
        A_errors.append(perr[0])
        tau1_values.append(tau)
        tau1_errors.append(perr[1])
        phi_values.append(phi_fit)
        phi_errors.append(perr[2])
        B_values.append(B)
        B_errors.append(perr[3])

        status   = '✓' if ok else '✗'
        freq_str = f'{freq_fit:.4f}'
        print(f"  {temp:<10} {tau2:>10.4f} {perr[4]:>8.4f} {A:>10.4f} {perr[0]:>8.4f} {phi_fit:>8.4f} {perr[2]:>8.4f} {tau:>8.4f} {perr[1]:>8.4f} {B:>8.4f} {perr[3]:>8.4f} {cinf:>8.4f} {freq_str:>8} {perr[5]:>8.4f} {status:>4}")

    results = []
    for i, temp in enumerate(temperatures):
        results.append({
            'Temp':     temp,
            'tau2':     tau_values[i],
            'tau2_err': tau_errors[i],
            'A':        A_values[i],
            'A_err':    A_errors[i],
            'phi':      phi_values[i],
            'phi_err':  phi_errors[i],
            'tau1':     tau1_values[i],
            'tau1_err': tau1_errors[i],
            'B':        B_values[i],
            'B_err':    B_errors[i],
            'cinf':     cinf_values[i],
            'freq':     freq_values[i],
            'freq_err': freq_errors[i],
        })
    df = pd.DataFrame(results)
    csv_file = f"{args.output_prefix}_fit_results.csv"
    df.to_csv(csv_file, index=False)
    print(f"\n✓ Saved fit results to: {csv_file}")

    print("\nGenerating summary plots...")
    plot_time_correlation(temperatures, correlations, timesteps, args)
    plot_relaxation_time(temperatures, tau_values, tau_errors, args)
    plot_cinf(temperatures, cinf_values, args)
    plot_freq_osc(temperatures, freq_values, freq_errors, args)
    plot_fft_spectrum(temperatures, correlations, timesteps, args, smooth_sigma=args.smooth_sigma)

    print("\n✓ All plots saved!")
    print("=" * 70)


if __name__ == "__main__":
    main()
