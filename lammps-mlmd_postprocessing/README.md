# Postprocessing of LAMMPS MLMD Trajectory Files
---
📄 Author: **Ouail Zakary**
- 📧 Email: [Ouail.Zakary@oulu.fi](mailto:Ouail.Zakary@oulu.fi)
- 🔗 ORCID: [0000-0002-7793-3306](https://orcid.org/0000-0002-7793-3306)
- 🌐 Website: [Personal Webpage](https://cc.oulu.fi/~nmrwww/members/Ouail_Zakary.html)
- 📁 Portfolio: [Academic Portfolio](https://ozakary.github.io/)
---
This directory contains the Python scripts used to postprocess the raw LAMMPS MLMD trajectory files (`.dump` format) produced for the monoclinic phase of RuP at all 23 simulation temperatures. The postprocessing consists of two steps: (1) converting the LAMMPS dump trajectories to XYZ format, and (2) generating time-averaged structures with NPT-correct unwrapping.

---

## Step 1 — Trajectory Conversion: LAMMPS Dump to XYZ

The raw LAMMPS `trajectory.dump` files are converted to the extended XYZ format using **TrajSlicer**, a lightweight trajectory processing tool. For each temperature, the following command is run inside the corresponding simulation directory:

```bash
python3 trajslicer_src.py trajectory.dump rup_traj_sampled-50_100ps.xyz \
    --start 0 --end 2000 \
    --labels 1:Ru 2:P
```

| Argument | Value | Description |
|----------|-------|-------------|
| `trajectory.dump` | — | Input LAMMPS dump file |
| `rup_traj_sampled-50_100ps.xyz` | — | Output XYZ trajectory file |
| `--start` | 0 | First frame to extract |
| `--end` | 2000 | Last frame to extract |
| `--labels` | `1:Ru 2:P` | Map LAMMPS atom types to element symbols |

This extracts all **2000 frames** of the 100 ps trajectory (written every 25 steps × 50,000 steps). The `sampled-50` in the filename reflects the 25-step dump interval relative to the 2 fs timestep (i.e., one frame every 50 fs).

> **TrajSlicer** source code and documentation: [https://github.com/ozakary/TrajSlicer](https://github.com/ozakary/TrajSlicer)

---

## Step 2 — Average Structure Generation

Time-averaged structures are computed for each temperature using a method corrected for NpT simulations, which accounts for continuous cell fluctuations and periodic boundary crossings. The script is run from the postprocessing directory:

```bash
python3 generate_average_strs_corrected.py
```

This processes all 23 temperatures in sequence and writes one average structure file per temperature.

### Algorithm

The average structure is computed in four stages:

**1. Read trajectory and collect fractional coordinates**

All frames are loaded and atomic positions are converted to fractional (scaled) coordinates relative to each frame's instantaneous simulation cell. This is essential for NpT simulations where the cell shape and volume fluctuate over time.

**2. Unwrap trajectories in fractional space**

To avoid artificial averaging across periodic boundary discontinuities, atomic trajectories are unwrapped frame by frame. For each atom, the fractional displacement between consecutive frames is computed, and any displacement larger than ±0.5 (indicating a periodic boundary crossing) is corrected by ±1.0:

```python
delta = all_scaled[i] - all_scaled[i-1]
jumps = np.round(delta)
all_scaled[i] = all_scaled[i-1] + (delta - jumps)
```

**3. Average fractional coordinates and cell**

The mean fractional position is computed over all frames and wrapped back into the [0, 1) range. The average simulation cell is computed as the arithmetic mean of all instantaneous cell matrices.

**4. Construct and validate the average structure**

An ASE `Atoms` object is constructed from the averaged fractional positions and averaged cell. A minimum interatomic distance check (threshold: 1.7 Å) is performed to validate the resulting structure.

### Output Files

For each of the 23 temperatures, the script writes:

```
rup_traj_sampled-50_100ps_average_structure_{temp}K.xyz
```

These files are saved into the corresponding `lammps_out/` subdirectory of each temperature simulation folder.

### Temperatures Processed

10 K, 50 K, 100 K, 150 K, 200 K, 250 K, 300 K, 310 K, 320 K, 330 K, 340 K, 350 K, 360 K, 370 K, 380 K, 390 K, 400 K, 450 K, 500 K, 550 K, 600 K, 650 K, 700 K

---

## Directory Structure

```
./lammps-mlmd_postprocessing/
├── trajslicer_src.py
└── generate_average_strs_corrected.py
```

The scripts read from and write to the individual temperature subdirectories of `../lammps-mlmd_simulations/`, following this relative path convention:

```
../lammps-mlmd_simulations/
├── 10K/
│   └── lammps_out/
│       ├── trajectory.dump                                    # input
│       ├── rup_traj_sampled-50_100ps.xyz                      # Step 1 output
│       └── rup_traj_sampled-50_100ps_average_structure_10K.xyz # Step 2 output
├── 50K/
│   └── lammps_out/
│       ├── ...
...
└── 700K/
    └── lammps_out/
        ├── ...
```

---

## Input Files

- [`trajslicer_src.py`](https://github.com/ozakary/TrajSlicer/trajslicer_src.py): Trajectory conversion tool — extracts and relabels frames from a LAMMPS dump file into XYZ format. Source: [github.com/ozakary/TrajSlicer](https://github.com/ozakary/TrajSlicer)
- [`generate_average_strs_corrected.py`](./generate_average_strs_corrected.py): Computes NpT-corrected time-averaged structures from XYZ trajectories for all 23 temperatures

## Output Files

Per temperature (written to `../lammps-mlmd_simulations/{T}K/lammps_out/`):

- `rup_traj_sampled-50_100ps.xyz`: Full 2,000-frame XYZ trajectory (converted from LAMMPS dump)
- `rup_traj_sampled-50_100ps_average_structure_{T}K.xyz`: Time-averaged structure in XYZ format

---

## Requirements

- **Python**: ≥ 3.8
- **ASE**: Atomic Simulation Environment (`ase`) — for reading/writing XYZ files and handling periodic structures
- **NumPy**: Array operations and averaging
- **SciPy**: Minimum distance validation (`scipy.spatial.distance.pdist`)
- **tqdm**: Progress bar display during trajectory processing

---
For further details, please refer to the respective folders or contact the author via the provided email.
