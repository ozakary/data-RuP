
#!/usr/bin/env python3

"""
plots dos for different curves, first  performs gaussian smearing if sigma is passed
outputs dos_comparison.pdf file
"""


import argparse
import matplotlib as mpl
import numpy as np
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy.ndimage import gaussian_filter1d


#"""
# Apply your global style FIRST (before importing pyplot)
try:
    import figure_formatting_v2 as ff
    ff.set_rcParams(ff.master_formatting)
except ImportError:
    print("Figure formatting module not found. Using default matplotlib settings.")
#"""
import matplotlib.pyplot as plt  # import AFTER style is applied


# -----------------------
# Helpers
# -----------------------
def load_two_col(fname: str):
    """
    Load a whitespace-delimited 2-column file: E DOS
    Skips blank lines and comments (#, !, @).
    """
    data = np.loadtxt(fname, comments=('#', '!', '@'))
    if data.ndim == 1:
        # single line case
        data = data[None, :]
    if data.shape[1] < 2:
        raise ValueError(f"{fname} has <2 columns. Got shape {data.shape}.")
    return data[:, 0], data[:, 1]



def transform(E, DOS, xshift, yscale, yoffset, sigma=0.0):
    # gnuplot used ($1 - xshift):($2*yscale + yoffset)
    if sigma==0.0:
        return (E - xshift), (DOS * yscale + yoffset)
    else:
        y_smooth = gaussian_filter1d(DOS * yscale + yoffset, sigma=args.sigma, mode=args.mode)
        return (E - xshift), y_smooth


if __name__ == "__main__":



    ap = argparse.ArgumentParser()
    #ap.add_argument("--input", type=str, default=None, help="Path to 2-column file: x y")
    ap.add_argument("--sigma", type=float, default=10.0, help="Gaussian sigma in in units of grid length")
    ap.add_argument(
        "--mode",
        type=str,
        default="nearest",
        help="Boundary handling: reflect, nearest, mirror, wrap, constant",
    )
    #ap.add_argument("--out", type=str, default="smoothed.txt",
    #                help="Output file for smoothed curve (2 columns: x y_smooth)")
    #ap.add_argument("--delimiter", type=str, default="\t",
    #                help=r"Delimiter for output file (default: tab; use ',' for CSV)")
    #ap.add_argument("--header", action="store_true",
    #                help="Write a header line to the output file")

    ap.add_argument("--xmin", type=float, default=-2.0, help=" min x value wrt the Fermi level for plotting")
    ap.add_argument("--xmax", type=float, default=2.0, help=" max x value wrt the Fermi level for plotting")

    ap.add_argument("--ymax", type=float, default=3.5, help=" max y value for plotting")
    #ap.add_argument("--ymin", type=float, default=2.0, help=" max x value wrt the Fermi level for plotting")
    ap.add_argument("--figout", type=str, default="dos_comparison.pdf",
                    help="Save figure to file (e.g. out.png); otherwise show interactively")
    args = ap.parse_args()


    # -----------------------
    # Fermi energies from DFT calculation 
    # -----------------------
    ef_300 = 11.41
    ef_350 = 11.4436
    ef_370 = 11.4112
    shift  = 0.05

    # Main panel ranges 
    x_main = (args.xmin,args.xmax)
    y_main = (0.0, args.ymax)

    # Inset
    x_inset = (-0.25, 0.25)
    y_inset = (0.0, 1.5)

    # -----------------------
    # Plot definitions
    # -----------------------
    # Each entry: (label, filename, xshift, yscale, yoffset, color)

    curves = [
        ("orthorhombic", "./DOS_total_ortho_all_relaxed",            0.1,   1/4,     0.0,     "black"),
        ("monoclinic",   "./DOS_total_mono_370K_all_relaxed",      0.025, 1/(4*9), 0.0,     "red"),
        ("T=300K",       "dos_300",                              ef_300,1/(4*9), -shift,  "blue"),
        ("T=350K",       "dos_350",                              ef_350,1/(4*9), -shift,  "magenta"),
        ("T=370K",       "dos_370",                              ef_370,1/(4*9), 0.0,     "green"),
    ]


    fig, ax = plt.subplots(figsize=(10.5, 8.0), constrained_layout=True)

    # Main plot
    energy=[]
    dos=[]
    for label, fname, xshift, yscale, yoffset, color in curves:
        E, D = load_two_col(fname)
        Ex, Dy = transform(E, D, xshift, yscale, yoffset,sigma=args.sigma)
        energy.append(Ex)
        dos.append(Dy)



    # Vertical line at x=0 (your "vert" + "0.0 w l")
    ax.axvline(0.0, lw=2.2, color="0.35",ls="dashed")
    # Horizontal y=0 reference line
    ax.axhline(0.0, lw=1.6, color="0.6")

    ax.set_xlim(*x_main)
    ax.set_ylim(*y_main)
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("DOS (States/eV/fu)")


    # Nice ticks
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))

    ax.tick_params(which="both", direction="in", top=True, right=True, length=6)
    ax.tick_params(which="minor", length=3)

    # Optional subtle grid (comment out if you prefer none)
    #ax.grid(which="major", alpha=0.18)
    #ax.grid(which="minor", alpha=0.08)

    # -----------------------
    # Inset zoom
    # -----------------------
    # Position roughly like gnuplot origin 0.53,0.52 size 0.40,0.42
    # Using axes fraction coordinates
    axins = inset_axes(
        ax, width="36%", height="38%", loc="upper right",
        bbox_to_anchor=(0.0, 0.0, 1.0, 1.0),
        bbox_transform=ax.transAxes,
        borderpad=1.5
    )



    i=0
    for label, fname, xshift, yscale, yoffset, color in curves:
       # E, D = load_two_col(fname)
       # Ex, Dy = transform(E, D, xshift, yscale, yoffset)
    
        ax.plot(energy[i], dos[i], lw=2.8, color=color, label=label)
        axins.plot(energy[i], dos[i], lw=2.5, color=color)
        i+=1


    # Legend placement similar to gnuplot: "bottom left" / "key left"
    leg = ax.legend(loc="lower left", frameon=True, framealpha=0.9, edgecolor="0.8")
    leg.get_frame().set_linewidth(0.8)

    axins.axvline(0.0, lw=2.0, color="0.35",ls="dashed")
    axins.axhline(0.0, lw=1.4, color="0.6")

    axins.set_xlim(*x_inset)
    axins.set_ylim(*y_inset)

    # Inset ticks like gnuplot
    axins.xaxis.set_major_locator(MultipleLocator(0.1))
    axins.yaxis.set_major_locator(MultipleLocator(0.5))
    axins.xaxis.set_minor_locator(AutoMinorLocator(5))
    axins.yaxis.set_minor_locator(AutoMinorLocator(5))

    axins.tick_params(which="both", direction="in", top=True, right=True, length=5)
    axins.tick_params(which="minor", length=2.5)

    # No labels inside inset (as in your gnuplot: set xlabel ""; set ylabel "")
    axins.set_xlabel("")
    axins.set_ylabel("")
    axins.set_title("")

    # Give inset a slightly thicker border so it reads clearly
    for spine in axins.spines.values():
        spine.set_linewidth(1.2)

    # -----------------------
    # Save
    if args.figout:
        plt.savefig(args.figout, dpi=300)
        plt.show()
    else:
        plt.show()
