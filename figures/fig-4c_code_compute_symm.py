import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import hashlib
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
warnings.filterwarnings('ignore')

# Optional: Set up figure formatting
try:
    import figure_formatting_v2 as ff
    ff.set_rcParams(ff.master_formatting)
except ImportError:
    pass

try:
    import spglib
except ImportError:
    raise ImportError("spglib is required: pip install spglib")

from ase.io import read
from tqdm import tqdm

# ---- Temperature paths ----
temperature_paths = {
    50:  "../../50K/lammps_out/rup_traj_sampled-10_100ps_average_structure_50K.xyz",
    100: "../../100K/lammps_out/rup_traj_sampled-10_100ps_average_structure_100K.xyz",
    110: "../../110K/lammps_out/rup_traj_sampled-10_100ps_average_structure_110K.xyz",
    120: "../../120K/lammps_out/rup_traj_sampled-10_100ps_average_structure_120K.xyz",
    130: "../../130K/lammps_out/rup_traj_sampled-10_100ps_average_structure_130K.xyz",
    140: "../../140K/lammps_out/rup_traj_sampled-10_100ps_average_structure_140K.xyz",
    150: "../../150K/lammps_out/rup_traj_sampled-10_100ps_average_structure_150K.xyz",
    160: "../../160K/lammps_out/rup_traj_sampled-10_100ps_average_structure_160K.xyz",
    170: "../../170K/lammps_out/rup_traj_sampled-10_100ps_average_structure_170K.xyz",
    180: "../../180K/lammps_out/rup_traj_sampled-10_100ps_average_structure_180K.xyz",
    190: "../../190K/lammps_out/rup_traj_sampled-10_100ps_average_structure_190K.xyz",
    200: "../../200K/lammps_out/rup_traj_sampled-10_100ps_average_structure_200K.xyz",
    250: "../../250K/lammps_out/rup_traj_sampled-10_100ps_average_structure_250K.xyz",
    260: "../../260K/lammps_out/rup_traj_sampled-10_100ps_average_structure_260K.xyz",
    270: "../../270K/lammps_out/rup_traj_sampled-10_100ps_average_structure_270K.xyz",
    280: "../../280K/lammps_out/rup_traj_sampled-10_100ps_average_structure_280K.xyz",
    290: "../../290K/lammps_out/rup_traj_sampled-10_100ps_average_structure_290K.xyz",
    300: "../../300K/lammps_out/rup_traj_sampled-10_100ps_average_structure_300K.xyz",
    310: "../../310K/lammps_out/rup_traj_sampled-10_100ps_average_structure_310K.xyz",
    320: "../../320K/lammps_out/rup_traj_sampled-10_100ps_average_structure_320K.xyz",
    330: "../../330K/lammps_out/rup_traj_sampled-10_100ps_average_structure_330K.xyz",
    340: "../../340K/lammps_out/rup_traj_sampled-10_100ps_average_structure_340K.xyz",
    350: "../../350K/lammps_out/rup_traj_sampled-10_100ps_average_structure_350K.xyz",
    400: "../../400K/lammps_out/rup_traj_sampled-10_100ps_average_structure_400K.xyz",
    450: "../../450K/lammps_out/rup_traj_sampled-10_100ps_average_structure_450K.xyz",
    500: "../../500K/lammps_out/rup_traj_sampled-10_100ps_average_structure_500K.xyz",
    550: "../../550K/lammps_out/rup_traj_sampled-10_100ps_average_structure_550K.xyz",
    600: "../../600K/lammps_out/rup_traj_sampled-10_100ps_average_structure_600K.xyz",
    650: "../../650K/lammps_out/rup_traj_sampled-10_100ps_average_structure_650K.xyz",
    700: "../../700K/lammps_out/rup_traj_sampled-10_100ps_average_structure_700K.xyz",
}


# ---- Unique cache filename ----
def get_cache_filename(temperature_paths, tolerances):
    """
    Generate a unique cache filename based on the temperatures, their file
    paths, and the tolerance array.  Different runs with different inputs
    always get different cache files and never overwrite each other.
    """
    key = (
        str(sorted(temperature_paths.items()))
        + str(list(tolerances))
    )
    uid = hashlib.md5(key.encode()).hexdigest()[:10]
    return f"symmetry_cache_{uid}.npz"


# ---- Space group classification ----
def classify_spacegroup(sg_string):
    if sg_string is None:
        return "$P1$ (1)"
    if "14" in sg_string:
        return "$P2_1/c$ (14)"
    if "62" in sg_string:
        return "$Pnma$ (62)"
    if " (4)" in sg_string:
        return "$P2_1$ (4)"
    if sg_string.startswith("P1"):
        return "$P1$ (1)"
    return sg_string


# ---- Colour map for space groups ----
SPACEGROUP_COLORS = {
    "$P2_1/c$ (14)": "#2A9D8F",   # teal   — monoclinic
    "$Pnma$ (62)":   "#E9C46A",   # gold    — orthorhombic
    "$P2_1$ (4)":    "#E76F51",   # orange  — intermediate
    "$P1$ (1)":      "#9E9E9E",   # grey   — no symmetry
}
DEFAULT_COLOR = "#FF9800"         # orange — unexpected SG


def get_sg_color(sg_label):
    return SPACEGROUP_COLORS.get(sg_label, DEFAULT_COLOR)


# ---- Cache helpers ----
def save_cache(results, existing_temps, tolerances, cache_path):
    sg_labels = np.array([
        [results[t][tol] for tol in tolerances]
        for t in existing_temps
    ])
    np.savez(
        cache_path,
        temperatures=np.array(existing_temps),
        tolerances=np.array(tolerances),
        sg_labels=sg_labels,
    )
    print(f"  Cache saved -> {cache_path}")


def load_cache(cache_path, tolerances):
    if not os.path.exists(cache_path):
        return None

    print(f"Found cache file: {cache_path}")
    data = np.load(cache_path, allow_pickle=False)

    cached_tols = data["tolerances"].tolist()

    # Guard against shape mismatch before value comparison
    if len(cached_tols) != len(tolerances) or not np.allclose(cached_tols, tolerances):
        print("  Tolerances changed — ignoring cache and recomputing.")
        return None

    existing_temps = data["temperatures"].tolist()
    sg_labels      = data["sg_labels"]

    results = {
        int(temp): {
            float(tol): str(sg_labels[i, j])
            for j, tol in enumerate(cached_tols)
        }
        for i, temp in enumerate(existing_temps)
    }

    print(f"  Loaded {len(existing_temps)} temperatures x "
          f"{len(cached_tols)} tolerances from cache.\n")
    return results, [int(t) for t in existing_temps]


# ---- Parallel worker (one process per temperature) ----
def _worker(args):
    """Read one XYZ file and compute symmetry for all tolerances."""
    temp, path, tolerances = args
    atoms = read(path)
    cell  = (
        atoms.get_cell().array,
        atoms.get_scaled_positions(),
        atoms.get_atomic_numbers(),
    )
    sg_per_tol = {}
    for tol in tolerances:
        sg_string       = spglib.get_spacegroup(cell, symprec=tol)
        sg_per_tol[tol] = classify_spacegroup(sg_string)
    return temp, sg_per_tol


# ---- Main analysis ----
def run_analysis(temperature_paths, tolerances,
                 cache_path=None, force_recompute=False,
                 n_workers=8):
    """
    Compute space group for every (temperature, tolerance) pair.
    Parallelised over temperatures; results are cached automatically.

    Parameters
    ----------
    n_workers : int
        Number of parallel processes. Set to os.cpu_count() for max throughput.
    """
    tolerances = list(tolerances)

    # Auto-generate unique cache filename if not provided
    if cache_path is None:
        cache_path = get_cache_filename(temperature_paths, tolerances)

    if not force_recompute:
        cached = load_cache(cache_path, tolerances)
        if cached is not None:
            return cached

    # Filter to files that actually exist
    existing_temps = []
    for temp, path in sorted(temperature_paths.items()):
        if os.path.exists(path):
            existing_temps.append(temp)
        else:
            print(f"  Warning: file not found for {temp} K — {path}")

    if not existing_temps:
        raise FileNotFoundError("No structure files found. Check your paths.")

    n_workers = min(n_workers, len(existing_temps))
    print(f"Found {len(existing_temps)} temperature files.")
    print(f"Tolerances: {len(tolerances)} values "
          f"({tolerances[0]:.3f} -> {tolerances[-1]:.3f})")
    print(f"Parallelising over {n_workers} workers...\n")

    args_list = [
        (temp, temperature_paths[temp], tolerances)
        for temp in existing_temps
    ]

    results = {}
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_worker, args): args[0]
                   for args in args_list}
        with tqdm(total=len(futures), desc="Computing symmetry") as pbar:
            for future in as_completed(futures):
                temp, sg_per_tol = future.result()
                results[temp]    = sg_per_tol
                pbar.update(1)

    save_cache(results, existing_temps, tolerances, cache_path)

    return results, existing_temps


# ---- Plotting ----
def plot_symmetry_map(results, existing_temps, tolerances, output_dir="."):
    """
    Grid of equal-height touching horizontal bars.
    x = symprec tolerance  |  y = temperature row (equal height, not T-scaled)
    Dashed black lines separate rows; each row is labelled with its temperature.
    """
    temps     = sorted(existing_temps)
    n_temps   = len(temps)
    tol_width = tolerances[1] - tolerances[0]

    # Row height — decrease to make bars thinner
    row_height = 0.5

    fig, ax    = plt.subplots(figsize=(8.0, 8.0))

    # Collect all unique SG labels present in results
    all_sg_labels = set()
    for temp in temps:
        for tol in tolerances:
            all_sg_labels.add(results[temp][tol])

    # Draw one filled rectangle per (temp_row, tolerance)
    for i, temp in enumerate(temps):
        y_center = i * row_height + row_height / 2
        for tol in tolerances:
            sg    = results[temp][tol]
            color = get_sg_color(sg)
            ax.barh(
                y=y_center,
                width=tol_width,
                left=tol - tol_width / 2,
                height=row_height,
                color=color,
                alpha=0.65,
                linewidth=0,
                align='center',
            )

        # Dashed separator between rows
        if i < n_temps - 1:
            y_sep = (i + 1) * row_height
            ax.axhline(y=y_sep, color='black', linestyle='--',
                       linewidth=1.5, zorder=5)
                       
            # Small tick marks at each tolerance position
            tick_height = row_height * 0.15
            for tol in tolerances:
                ax.plot([tol, tol], [y_sep - tick_height, y_sep + tick_height],
                        color='black', linewidth=1.5, zorder=6)

        # Temperature label to the right
        ax.text(
            tolerances[0] - tol_width * 0.2,
            y_center,
            f"{temp} K",
            va='center', ha='right', fontsize=14
        )

    # x-axis ticks at every tolerance value
    ax.set_xticks(tolerances)
    ax.set_xticklabels([f"{t:.2f}" for t in tolerances])
    ax.tick_params(axis='x', rotation=0)
    ax.set_xlabel(r"Atomic position tolerance / Å")

    # Hide y-axis (temperature labels are drawn as text)
    ax.set_yticks([])
    ax.set_ylabel("")

    ax.set_xlim(0.01, 0.10)
    ax.set_ylim(0, n_temps * row_height)

    # Legend at the bottom
    preferred_order = ["$P1$ (1)", "$P2_1$ (4)", "$P2_1/c$ (14)", "$Pnma$ (62)"]
    ordered_labels  = [sg for sg in preferred_order if sg in all_sg_labels]
    ordered_labels += [sg for sg in all_sg_labels   if sg not in preferred_order]

    legend_handles = [
        mpatches.Patch(color=get_sg_color(sg), label=sg, alpha=0.65)
        for sg in ordered_labels
    ]
    ax.legend(handles=legend_handles, frameon=False,
              loc='upper center', bbox_to_anchor=(0.5, -0.15),
              ncol=len(ordered_labels))

    plt.tight_layout()

    out_path = os.path.join(output_dir, "symmetry_vs_temperature.svg")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved plot -> {out_path}")
    plt.show()

    return fig, ax


# ---- Summary table ----
def print_summary(results, existing_temps, tolerances):
    tol_strs = [f"{t:.2f}" for t in tolerances]
    header   = f"{'Temp (K)':>10} | " + " | ".join(f"{'tol='+s:>14}" for s in tol_strs)
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for temp in existing_temps:
        row  = f"{temp:>10} | "
        row += " | ".join(f"{results[temp][tol]:>14}" for tol in tolerances)
        print(row)
    print("=" * len(header))


# ---- Main ----
def main():

    tolerances      = np.arange(0.01, 0.11, 0.01)
    output_dir      = "."
    force_recompute = False   # <- set True to ignore cache and rerun everything
    n_workers       = 12       # <- adjust to your machine/node core count

    print("=" * 60)
    print("SYMMETRY ANALYSIS vs TEMPERATURE AND TOLERANCE")
    print("=" * 60)

    # cache_path is auto-generated from a hash of the inputs —
    # different tolerance ranges or temperature sets get different files
    results, existing_temps = run_analysis(
        temperature_paths,
        tolerances,
        cache_path=None,            # None = auto-generate unique filename
        force_recompute=force_recompute,
        n_workers=n_workers,
    )

    print_summary(results, existing_temps, tolerances)

    plot_symmetry_map(results, existing_temps, tolerances, output_dir=output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
