#!/usr/bin/env python3
"""
Plot time correlation functions and extract characteristic times.
Creates 3 separate plots: correlation curves, relaxation times, C_inf plateau.
Also gives diagonistics_*.png file for each temperature showing fitted vs original correlation function and the residual plot.

Fits C(t) = A * exp(-t/tau) cos(\omega t + phi) + C_inf to extract tau; \omega is found from FFT to make the fitting more robust,  C_inf is extracted from the curve
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

from matplotlib.ticker import LinearLocator
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d

from functools import partial

try:
    import figure_formatting_v2 as ff
    ff.set_rcParams(ff.master_formatting)
except ImportError:
    print("Figure formatting module not found. Using default matplotlib settings.")


# ============================================================================
# FFT — T_osc
# ============================================================================

def zero_crossings(z, threshold_frac=0.05):
    """
    Return True if z has meaningful zero crossings.
    Ignores sign changes smaller than threshold_frac * max(|z|) to avoid noise triggers.
    """
    amp = np.max(np.abs(z))
    if amp == 0:
        return False
    threshold = threshold_frac * amp
    # Only consider points with |z| > threshold when checking sign changes
    s = np.where(np.abs(z) > threshold, np.sign(z), 0)
    for k in range(1, len(s)):
        if s[k] != 0 and s[k-1] != 0 and s[k] != s[k-1]:
            return True
    return False


def period_from_fft(z, dt, smooth_sigma=1.0):
    """
    Dominant period from (optionally smoothed) FFT power spectrum.
    Returns NaN if no zero crossings exist (C(t) never crosses baseline).
    """
    """
    if not zero_crossings(z):
        return np.nan
    """

    #freqs = np.fft.rfftfreq(len(z), d=dt)[1:]
    #power = np.abs(np.fft.rfft(z))[1:] ** 2

    #subtract the average value which will remove the first peak(??)
    
    freqs = np.fft.rfftfreq(len(z), d=dt)  #2*np.pi # i think we do not need 2pi factor here because we are after frequency/not omega

    #print(f"frequencies{freqs}")
    #print(f"frequency grid {freqs}")
    power = np.abs(np.fft.rfft(z-np.average(z))) ** 2

    if smooth_sigma > 0:
        power = gaussian_filter1d(power.astype(float), sigma=smooth_sigma)

    f_peak = freqs[np.argmax(power)]
    #print(f_peak)
    #plt.plot(freqs,power)
    #plt.show()
    #return 1.0 / f_peak if f_peak > 0 else np.nan
    return f_peak

def model_decay(t, A, tau, C_inf):
    """C(t) = A * exp(-t/tau) + C_inf"""
    return A * np.exp(-t / tau) + C_inf


def fit_relaxation(t, y, tmax=None):
    """
    Fit C(t) = A * exp(-t/tau) + C_inf.

    Returns tau, C_inf, A, perr (uncertainties), fit_ok
    """
    if tmax is not None:
        mask = t <= tmax
        t_fit, y_fit = t[mask], y[mask]
    else:
        t_fit, y_fit = t, y

    A0    = y_fit[0] - y_fit[-1]
    tau0  = t_fit[-1] / 4
    Cinf0 = max(0.0, y_fit[-1])

    try:
        popt, pcov = curve_fit(
            model_decay, t_fit, y_fit,
            p0=[A0, tau0, Cinf0],
            bounds=([0, 1e-6, 0], [np.inf, np.inf, np.inf]),
            maxfev=10000
        )
        perr = np.sqrt(np.diag(pcov))
        return popt[1], popt[2], popt[0], perr, True
    except RuntimeError:
        return np.nan, np.nan, np.nan, np.array([np.nan]*3), False


def model_oscillating_decay(t, A, tau, phi, C_inf, freq):
    """C(t) = A * exp(-t/tau) + C_inf"""
    if (freq<=1.0):
        #C_inf=0
        return A * np.exp((-t / tau)) + C_inf
    else:
        return A * np.exp(-t / tau)*np.cos(2*np.pi*freq*t + phi) + C_inf


def fit_relaxation2(t, y, freq_guess=1, tmax=None):
    """
    Fit C(t) = A * exp(-t/tau) + C_inf.

    Returns tau, C_inf, A, perr (uncertainties), fit_ok
    """
    if tmax is not None:
        mask = t <= tmax
        t_fit, y_fit = t[mask], y[mask]
    else:
        t_fit, y_fit = t, y

    A0    = y_fit[0] - y_fit[-1]
    tau0  = t_fit[-1] / 4
    Cinf0 = np.average(y_fit[:-5]) #max(0.0, y_fit[-1])
    if (Cinf0 <0.02):
        Cinf0=0
    #omega=3
    phi=2

    #if (freq_guess<1.0):
    #    freq_guess=0.0

    try:
        """
        popt, pcov = curve_fit(
            model_oscillating_decay, t_fit, y_fit,
            p0=[A0, tau0, Cinf0,phi],
            bounds=([0, 0, 0,0,-np.inf], [np.inf, np.inf, 1,10, np.inf]),
            maxfev=10000
        )
        """
        #make omega constant
        popt, pcov = curve_fit(partial(model_oscillating_decay, C_inf = Cinf0, freq=freq_guess),
            t_fit, y_fit,
            p0=[A0, tau0,phi],
            bounds=([0, 0,-np.pi], [1, 10, np.pi]),
            maxfev=100000
        )

        perr = np.sqrt(np.diag(pcov))
        #print(f"omega and phi vals {popt[3], popt[4]}")

        #plot diagnostics and return the plots
        #A_fit,tau_fit,cinf_fit,freq_fit,phi_fit=popt
        #return tau_fit, cinf_fit,A_fit,freq_fit, phi_fit, perr, True
        A_fit,tau_fit,phi_fit=popt
        

        #return popt[1], popt[2], popt[0], popt[3], perr, True
        return A_fit, tau_fit, phi_fit, Cinf0, perr, True
    except RuntimeError:
        return np.nan, np.nan, np.nan, np.array([np.nan]*3), False


def model_two_oscillating_decay(t, A1, A2, tau1,tau2, C_inf, freq1, freq2,  phi1, phi2):
    """C(t) = A * exp(-t/tau) + C_inf"""
    return A1*np.exp(-t / tau1)*np.cos(2*np.pi*freq1*t + phi1) + A2*np.exp(-t / tau2)*np.cos(2*np.pi*freq2*t + phi2)  + C_inf


def fit_relaxation3(t, y, freq_guess=1, tmax=None):
    """
    Fit C(t) = A * exp(-t/tau) + C_inf.

    Returns tau, C_inf, A, perr (uncertainties), fit_ok
    """
    if tmax is not None:
        mask = t <= tmax
        t_fit, y_fit = t[mask], y[mask]
    else:
        t_fit, y_fit = t, y

    A10    = y_fit[0] - y_fit[-1]
    A20    = (y_fit[0] - y_fit[-1])*0.5
    tau10  = t_fit[-1] / 4
    tau20  = t_fit[-1] / 10
    Cinf0 = max(0.0, y_fit[-1])
    #omega=3
    phi10=np.random.rand()
    phi20=np.random.rand()
    freq10 = freq_guess*0.9
    freq20 = freq_guess*1.1

    #if (freq_guess<1.0):
    #    freq_guess=0.0

    try:
        popt, pcov = curve_fit(
            model_two_oscillating_decay, t_fit, y_fit,
            p0=[A10, A20, tau10, tau20, Cinf0,freq10, freq20,phi10, phi20],
            bounds=([0, 0, 0, 0, 0, 0,0,-np.pi, -np.pi], [np.inf, np.inf, np.inf, np.inf, 1, 10,10, np.pi, np.pi]), 
            maxfev=10000
        )
        perr = np.sqrt(np.diag(pcov))
        #print(f"omega and phi vals {popt[3], popt[4]}")

        #plot diagnostics and return the plots
        A1_fit,A2_fit, tau1_fit, tau2_fit, cinf_fit,freq1_fit,freq2_fit, phi1_fit, phi2_fit=popt
        

        #return popt[1], popt[2], popt[0], popt[3], perr, True
        return tau1_fit, tau2_fit, cinf_fit,A1_fit, A2_fit, freq1_fit, freq2_fit, phi1_fit, phi2_fit, perr, True
    except RuntimeError:
        return np.nan, np.nan, np.nan, np.array([np.nan]*3), False




# ============================================================================
# Data Loading
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
        ax.plot(t, correlations[i], linewidth=2.5, color=colors[idx],
                label=temperatures[i])

    ax.set_xlabel(f't / {args.unit}')
    ax.set_ylabel(r'$C(t)$')
    ax.set_xlim(0, 5)
    ax.set_ylim(-0.05, 0.3)
    ax.xaxis.set_major_locator(LinearLocator(numticks=5))
    ax.yaxis.set_major_locator(LinearLocator(numticks=8))
    
    ax.axhline(y=0, color='grey', linestyle='--', linewidth=2.5)    

    if args.ylim is not None:
        ax.set_ylim(args.ylim)

    ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=8)

    #output_file = f"{args.output_prefix}_correlation_legend.svg"
    output_file = f"{args.output_prefix}_correlation_legend.png"
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
        ax.scatter([T], [tau], s=60, color=colors[idx],
                   edgecolors='white', linewidths=0.6, zorder=3)

    ax.set_xlabel(r'$T$ / K')
    ax.set_ylabel(r'$\tau$ / ' + args.unit)
    ax.set_xlim(0, 750)
    ax.set_ylim(0.05, 0.4)
    ax.xaxis.set_major_locator(LinearLocator(numticks=4))
    ax.yaxis.set_major_locator(LinearLocator(numticks=6))

    ax.axhline(y=0, color='grey', linestyle='--', linewidth=2.5)

    #output_file = f"{args.output_prefix}_relaxation_time.svg"
    output_file = f"{args.output_prefix}_relaxation_time.png"
    plt.savefig(output_file, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")

    if args.show:
        plt.show()
    plt.close()


def plot_tosc(temperatures, tosc_values, args):
    """Plot T_osc vs temperature. NaN values (no zero crossings) are simply absent."""
    fig, ax = plt.subplots(figsize=(5.25, 5.25))

    temp_numeric = np.array([float(t.rstrip('K')) for t in temperatures])
    order        = np.argsort(temp_numeric)
    temps_sorted = temp_numeric[order]
    tosc_sorted  = np.array(tosc_values)[order]
    colors       = plt.cm.coolwarm(np.linspace(0, 1, len(temperatures)))

    # Only plot where T_osc is defined
    valid = np.isfinite(tosc_sorted)

    # Shade the region where C(t) has no zero crossings
    invalid_temps = temps_sorted[~valid]
    if len(invalid_temps) > 0:
        T_shade_min = invalid_temps.min() - 5
        T_shade_max = invalid_temps.max() + 5
        ax.axvspan(T_shade_min, T_shade_max, alpha=0.12, color='gray', zorder=0)
        ax.text(460, 0.2, 'no oscillations',
                ha='center', va='center',
                fontsize=20, color='dimgray', style='italic', rotation = 90, transform=ax.transData)

    ax.plot(temps_sorted[valid], tosc_sorted[valid], '-', lw=1.6, color='0.2', zorder=2)

    for idx, (T, tosc) in enumerate(zip(temps_sorted, tosc_sorted)):
        if np.isfinite(tosc):
            ax.scatter([T], [tosc], s=60, color=colors[idx],
                       edgecolors='white', linewidths=0.6, zorder=3)

    ax.set_xlabel(r'$T$ / K')
    ax.set_ylabel(r'$T_{\mathrm{osc}}$ / ' + args.unit)
    ax.set_xlim(0, 750)
    ax.set_ylim(0, 0.6)
    ax.xaxis.set_major_locator(LinearLocator(numticks=4))
    ax.yaxis.set_major_locator(LinearLocator(numticks=6))

    #output_file = f"{args.output_prefix}_tosc.svg"
    output_file = f"{args.output_prefix}_tosc.png"
    plt.savefig(output_file, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")

    if args.show:
        plt.show()
    plt.close()

def plot_omega(temperatures, omega_values, args):
    """Plot omega vs temperature. """
    fig, ax = plt.subplots(figsize=(5.25, 5.25))

    temp_numeric = np.array([float(t.rstrip('K')) for t in temperatures])
    order        = np.argsort(temp_numeric)
    temps_sorted = temp_numeric[order]
    omega_sorted  = np.array(omega_values)[order]
    colors       = plt.cm.coolwarm(np.linspace(0, 1, len(temperatures)))

    # Only plot where T_osc is defined
    valid = np.isfinite(omega_sorted)

    # Shade the region where C(t) has no zero crossings
    invalid_temps = temps_sorted[~valid]
    if len(invalid_temps) > 0:
        T_shade_min = invalid_temps.min() - 5
        T_shade_max = invalid_temps.max() + 5
        ax.axvspan(T_shade_min, T_shade_max, alpha=0.12, color='gray', zorder=0)
        ax.text(460, 0.2, 'no oscillations',
                ha='center', va='center',
                fontsize=20, color='dimgray', style='italic', rotation = 90, transform=ax.transData)

    ax.plot(temps_sorted[valid], omega_sorted[valid], '-', lw=1.6, color='0.2', zorder=2)

    for idx, (T, omega) in enumerate(zip(temps_sorted, omega_sorted)):
        if np.isfinite(omega):
            ax.scatter([T], [omega], s=60, color=colors[idx],
                       edgecolors='white', linewidths=0.6, zorder=3)

    ax.set_xlabel(r'$T$ / K')
    ax.set_ylabel(r'$\omega$ / ' + args.omega_unit)
    ax.set_xlim(0, 750)
    ax.set_ylim(0, 30)
    ax.xaxis.set_major_locator(LinearLocator(numticks=4))
    ax.yaxis.set_major_locator(LinearLocator(numticks=6))

    #output_file = f"{args.output_prefix}_tosc.svg"
    output_file = f"{args.output_prefix}_omega.png"
    plt.savefig(output_file, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")

    if args.show:
        plt.show()
    plt.close()


def plot_cinf(temperatures, cinf_values, cinf_errors, args):
    """Plot C_inf plateau vs temperature with error bars."""
    fig, ax = plt.subplots(figsize=(5.25, 5.25))

    temp_numeric  = np.array([float(t.rstrip('K')) for t in temperatures])
    order         = np.argsort(temp_numeric)
    temps_sorted  = temp_numeric[order]
    cinf_sorted   = np.array(cinf_values)[order]
    #err_sorted    = np.array(cinf_errors)[order]
    colors        = plt.cm.coolwarm(np.linspace(0, 1, len(temperatures)))

    ax.plot(temps_sorted, cinf_sorted, '-', lw=1.6, color='0.2', zorder=2)
    #ax.errorbar(temps_sorted, cinf_sorted, yerr=err_sorted,
                #fmt='none', ecolor='0.4', elinewidth=1.2, capsize=3, zorder=2)

    for idx, (T, cinf) in enumerate(zip(temps_sorted, cinf_sorted)):
        ax.scatter([T], [cinf], s=60, color=colors[idx],
                   edgecolors='white', linewidths=0.6, zorder=3)

    ax.set_xlabel(r'$T$ / K')
    ax.set_ylabel(r'$C(t\rightarrow \infty)$')
    ax.set_xlim(0, 750)
    ax.set_ylim(-0.02, 0.32)
    ax.xaxis.set_major_locator(LinearLocator(numticks=4))
    ax.yaxis.set_major_locator(LinearLocator(numticks=6))

    ax.axhline(y=0, color='grey', linestyle='--', linewidth=2.5)

    #output_file = f"{args.output_prefix}_cinf.svg"
    output_file = f"{args.output_prefix}_cinf.png"
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
    parser.add_argument('-i', '--input-dir', required=True, type=str,
                        help='Input directory containing NPZ files')
    parser.add_argument('-o', '--output-prefix', required=True, type=str,
                        help='Output prefix for plot files')
    parser.add_argument('-prefix', '--file-prefix', default='time_corr', type=str,
                        help='Prefix of NPZ files to load (default: time_corr)')
    parser.add_argument('-temps', '--temperatures', nargs='+', default=None,
                        help='Specific temperatures to plot')
    parser.add_argument('--unit', default='ps', type=str,
                        help='Time unit label (default: ps)')
    parser.add_argument('--omega-unit', default='THz', type=str,
                        help='Frequency unit label (default: ps)')
    parser.add_argument('--xmin', type=float, default=0,
                        help='X-axis minimum for C(t) plot (default: 0)')
    parser.add_argument('--xmax', type=float, default=None,
                        help='X-axis maximum for C(t) plot (default: auto)')
    parser.add_argument('--ylim', nargs=2, type=float, default=None,
                        help='Y-axis limits for C(t) plot')
    parser.add_argument('--tmax', type=float, default=None,
                        help='Fit C(t) only up to this time in ps (default: full range)')
    parser.add_argument('--smooth-sigma', type=float, default=2.0,
                        help='Gaussian smoothing sigma for FFT T_osc extraction in bins (default: 2.0, set 0 to disable)')
    parser.add_argument('--show', action='store_true',
                        help='Show plots interactively')
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

    print("\n First finding the oscillation frequency from FFT")
    print("\nFitting C(t) = A·exp(-t/τ)cos(\omega t) + C_inf keeping \omega, C_inf fixed and extracting T_osc ...")
    #print(f"  {'Temp':<10} {'tau (ps)':>10} {'±':>8} {'C_inf':>10} {'±':>8} {'T_osc':>8} {'OK':>4}")

    print(f"  {'Temp' :<10} {'tau (ps)' :>10} {'±':>8} {'A' :>10} {'±':>8} {'phi' :>8} {'±':>8} {'cinf' :>8}  {'freq' :>8} {'OK' :>4}")
    print(f"  {'-'*10} {'-'*10} {'-'*8} {'-'*10} {'-'*8} {'-'*8} {'-'*4}")

    tau_values, tau_errors = [], []
    cinf_values, cinf_errors = [], []
    #tosc_values = []
    freq_values_from_fft = []
    freq_values_from_fit = []

    for temp, y, dt in zip(temperatures, correlations, timesteps):
        t = np.arange(len(y)) * dt
        #print(f"Time steps {t}")


        # T_osc from smoothed FFT (NaN if no zero crossings)
        base = float(np.mean(y[-max(5, int(0.10*len(y))):]))
        z    = y - base
        freq_fft = period_from_fft(z, dt, smooth_sigma=args.smooth_sigma)
        freq_values_from_fft.append(freq_fft)

        # Exponential fit
        #use the omega value from fft to fit oscillatory exponential
        #tau, cinf, A, perr, ok = fit_relaxation(t, y, tmax=args.tmax)
        #tau, cinf, A, freq_fit, phi_fit, perr, ok  = fit_relaxation2(t, y, freq_guess= freq_fft,tmax=args.tmax)
        #prediction=model_oscillating_decay(t_dense, A,tau, cinf, freq_fit, phi_fit)

        A, tau, phi_fit, cinf, perr, ok  = fit_relaxation2(t, y, freq_guess= freq_fft,tmax=args.tmax)
        t_dense=t #np.linspace(0,np.max(t),100)
        prediction=model_oscillating_decay(t_dense, A,tau, phi_fit, cinf, freq_fft) ##, phi_fit)
        #print(f"Temp {temp}, {tau}, {A}, {phi_fit}, {perr}")

        #tau1, tau2, cinf,A1, A2, freq1, freq2, phi1, phi2, perr, ok=fit_relaxation3(t, y, freq_guess= freq_fft,tmax=args.tmax)
        #prediction=model_two_oscillating_decay(t_dense, A1, A2, tau1, tau2, cinf, freq1, freq2, phi1, phi2)

        #residual:
        MAE=np.average(np.abs(prediction-y))
        #print(f"MAE = {MAE}")

        #print(tau1, tau2, cinf,A1, A2, freq1, freq2,  phi1, phi2, perr)
        ##diagontic plot:
        #first prediction

        fig, axs = plt.subplots(1, 2) #, sharey=True) #
        axs[0].plot(t,y, label="truth")
        axs[0].plot(t_dense,prediction, label="prediction")
        axs[0].legend()
        
        axs[1].plot(t, y-prediction,label="residual")
        axs[1].text(0.80, 0.85, f"MAE: {MAE:6.4f}", transform=axs[1].transAxes) 
                    #horizontalalignment='right', verticalalignment='top',  fontsize=14, color='blue')

        axs[1].legend()
        #plt.legend()
        #fig.tight_layout()
        outfile="diagnostics_"+temp+".png"
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()

        #"""
        tau_values.append(tau)
        tau_errors.append(perr[1])
        cinf_values.append(cinf)
        #cinf_errors.append(perr[2])
        #freq_values_from_fit.append(freq_fit)

        status = '✓' if ok else '✗'
        #tosc_str = f'{tosc:.4f}' if np.isfinite(tosc) else '  N/A  '
        freq_str = f'{freq_fft:.4f} ' ### if np.isfinite(omega) else '  N/A  '
        print(f"  {temp:<10} {tau:>10.4f} {perr[1]:>8.4f} {A:>10.4f} {perr[0]:8.4f} {phi_fit:10.4f} {perr[2]:8.4f} {cinf:>10.4f}  {freq_str:>8} {status:>4}")
        #"""
    print("\nGenerating plots...")
    plot_time_correlation(temperatures, correlations, timesteps, args)
    plot_relaxation_time(temperatures, tau_values, tau_errors, args)
    plot_cinf(temperatures, cinf_values, cinf_errors, args)
    #plot_tosc(temperatures, tosc_values, args)
    #to plot omega, need to multiply frequency by 2pi
    plot_omega(temperatures, 2*np.pi*np.array(freq_values_from_fft), args)

    print("\n✓ All plots saved!")
    print("=" * 70)


if __name__ == "__main__":
    main()
