# Python Scripts for Machine Learning-Accelerated Phonon Dispersion Analysis
---
📄 Author: **Ouail Zakary**
- 📧 Email: [Ouail.Zakary@oulu.fi](mailto:Ouail.Zakary@oulu.fi)
- 🔗 ORCID: [0000-0002-7793-3306](https://orcid.org/0000-0002-7793-3306)
- 🌐 Website: [Personal Webpage](https://cc.oulu.fi/~nmrwww/members/Ouail_Zakary.html)
- 📁 Portfolio: [Academic Portfolio](https://ozakary.github.io/)
---
This directory contains the Python scripts and SLURM job scripts for the machine learning-accelerated phonon dispersion of RuP at all 23 temperatures using the finite displacement method. Forces are evaluated using the fine-tuned MACE potential on the time-averaged structures. The workflow consists of five sequential steps: (1) generate displaced structures, (2) predict MACE forces, (3) compute force constants, (4) calculate the phonon band structure, and (5) plot the band structure and phonon density of states.

All calculations were performed on the **LUMI supercomputer** at CSC — IT Center for Science (Finland). More details: [docs.lumi-supercomputer.eu](https://docs.lumi-supercomputer.eu/)

---

## Directory Structure

```
<root>/phonon_dispersion/
├── 1-generate_displacements_improved.py
├── 2-batch_mace_forces_improved.py
├── 3-computing_force_constants.py
├── 4-calculate_band_structure_{T}K_oom-fix.py   # one per temperature
├── 5-plot_band_structure_improved.py
├── fine-tuned_mace-mp-0b3-medium_compiled.model
├── job_script_1-generate_displacements.sh
├── job_script_2-mace_forces_predict_{T}K.sh     # one per temperature
├── job_script_3-compute_force_constants.sh
├── job_script_4-calculate_band_structure_{T}K_oom-fix.sh  # one per temperature
├── 10K/
│   ├── rup_traj_sampled-50_100ps_average_structure_10K.xyz
│   ├── phonopy_10K.yaml
│   ├── displaced_structures/
│   │   ├── all_displaced_structures.xyz
│   │   └── all_displaced_structures_out.xyz
│   ├── FORCE_CONSTANTS
│   ├── phonopy_10K_with_forces.yaml
│   ├── band_data_10K.npz
│   ├── phonon_band_10K.png
│   └── phonon_band_Ru_projection_10K.png
├── 50K/
│   └── ...
...
└── 700K/
    └── ...
```

The time-averaged structures (`rup_traj_sampled-50_100ps_average_structure_{T}K.xyz`) are copied from the MLMD postprocessing step into each temperature subdirectory before running the workflow.

---

## Workflow Overview

```
Time-averaged structure
        │
        ▼
Step 1: Generate displaced structures (Phonopy finite displacement)
        │
        ▼
Step 2: MACE force prediction on all displaced structures (GPU)
        │
        ▼
Step 3: Compute force constants (Phonopy)
        │
        ▼
Step 4: Calculate phonon band structure + Ru projection
        │
        ▼
Step 5: Plot band structure + DOS (all temperatures)
```

---

## Step 1 — Generate Displaced Structures

Run from `<root>/phonon_dispersion/`:

```bash
sbatch job_script_1-generate_displacements.sh
```

which executes:

```bash
python3 1-generate_displacements_improved.py
```

For each temperature, this script reads the time-averaged structure, generates symmetry-inequivalent finite displacements using **Phonopy** (finite displacement method), and writes all displaced structures into a single multi-frame XYZ file.

### Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Displacement distance** | 0.02 Å | Atomic displacement amplitude |
| **Supercell matrix** | 1×1×1 | No supercell expansion (structure is already large) |
| **Input** (per temperature) | `./{T}K/rup_traj_sampled-50_100ps_average_structure_{T}K.xyz` | Time-averaged structure |

### Output (per temperature)

| File | Description |
|------|-------------|
| `./{T}K/displaced_structures/all_displaced_structures.xyz` | Multi-frame XYZ with all displaced configurations |
| `./{T}K/phonopy_{T}K.yaml` | Phonopy object saved for reuse in subsequent steps |

### SLURM Configuration

| Parameter | Value |
|-----------|-------|
| **Partition** | `standard` |
| **MPI tasks** | 40 |
| **Wall time** | 2 hours |
| **Module** | `cray-python` |

---

## Step 2 — MACE Force Prediction

Force prediction is submitted **independently per temperature** to allow parallel execution across all 23 temperatures simultaneously. A dedicated job script and a temperature-specific input path are used for each:

```bash
sbatch job_script_2-mace_forces_predict_{T}K.sh
```

which executes (example for 10 K):

```bash
python3 2-batch_mace_forces_improved.py \
    -m ./fine-tuned_mace-mp-0b3-medium_compiled.model \
    -i 10K/displaced_structures/all_displaced_structures.xyz \
    -o 10K/displaced_structures/all_displaced_structures_out.xyz \
    -dtype float64 \
    --save-frequency 1
```

This script loads all displaced structures from the multi-frame XYZ, runs the fine-tuned MACE model on each configuration, and writes the resulting energies, forces, and stresses back into an output multi-frame XYZ file. Calculations support GPU acceleration and can be resumed with `--resume` if interrupted.

### Key Parameters

| Argument | Value | Description |
|----------|-------|-------------|
| `-m` | `fine-tuned_mace-mp-0b3-medium_compiled.model` | Fine-tuned MACE potential |
| `-dtype` | `float64` | Floating point precision |
| `--save-frequency` | 1 | Save output after every structure |
| `--resume` | — | Resume from existing partial output if interrupted |

### Output (per temperature)

| File | Description |
|------|-------------|
| `./{T}K/displaced_structures/all_displaced_structures_out.xyz` | Multi-frame XYZ with MACE-predicted energies (`MACE_energy`), forces (`MACE_forces`), and stresses (`MACE_stress`) |

### SLURM Configuration

| Parameter | Value |
|-----------|-------|
| **Partition** | `standard-g` (GPU partition) |
| **Nodes** | 1 |
| **GPUs** | 1 (per temperature job) |
| **Module** | `pytorch/2.7` |

---

## Step 3 — Compute Force Constants

Run from `<root>/phonon_dispersion/`:

```bash
sbatch job_script_3-compute_force_constants.sh
```

which executes:

```bash
python3 3-computing_force_constants.py
```

This script reads the MACE-predicted forces from the output multi-frame XYZ for each temperature, collects them into a forces array, and uses Phonopy to compute the interatomic force constants (IFCs) via the finite displacement method.

### Output (per temperature)

| File | Description |
|------|-------------|
| `./{T}K/FORCE_CONSTANTS` | Interatomic force constants in Phonopy format |
| `./{T}K/phonopy_{T}K_with_forces.yaml` | Phonopy object including force constants |

### SLURM Configuration

| Parameter | Value |
|-----------|-------|
| **Partition** | `standard` |
| **MPI tasks** | 40 |
| **Module** | `cray-python` |

---

## Step 4 — Calculate Phonon Band Structure

Band structure calculation is submitted **independently per temperature** (analogous to Step 2). A dedicated script and job file exist for each temperature:

```bash
sbatch job_script_4-calculate_band_structure_{T}K_oom-fix.sh
```

which executes:

```bash
python3 4-calculate_band_structure_{T}K_oom-fix.py
```

This script loads the force constants, defines a high-symmetry q-path through the monoclinic Brillouin zone, and computes the phonon frequencies and Ru-projected eigenvector weights at each q-point. The `oom-fix` suffix reflects an out-of-memory fix applied to handle the large number of bands in the time-averaged supercell structures.

### High-Symmetry Path

The phonon band structure is computed along the following continuous path through the monoclinic Brillouin zone (fractional coordinates):

```
Γ --> X --> Y --> Γ --> Z --> R --> Γ
[0 0 0] --> [0.5 0 0] --> [0.5 0.5 0] --> [0 0 0] --> [0 0 0.5] --> [0.5 0.5 0.5] --> [0 0 0]
```

26 q-points are used per segment.

### Output (per temperature)

| File | Description |
|------|-------------|
| `./{T}K/band_data_{T}K.npz` | Compressed NumPy archive: distances, frequencies, Ru weights, special points and labels |
| `./{T}K/phonon_band_{T}K.png` | Simple phonon band structure (all bands in blue) |
| `./{T}K/phonon_band_Ru_projection_{T}K.png` | Band structure coloured by Ru atomic character |

### SLURM Configuration

| Parameter | Value |
|-----------|-------|
| **Partition** | `standard` |
| **MPI tasks** | 1 |
| **Module** | `cray-python` |

---

## Step 5 — Plot Band Structure and DOS

Run from `<root>/phonon_dispersion/` (locally, after downloading `band_data_{T}K.npz` files):

```bash
python3 5-plot_band_structure_improved.py
```

This script reads the pre-computed `band_data_{T}K.npz` files for all temperatures, calculates the phonon density of states (DOS) using Gaussian broadening, and produces combined band structure + DOS figures. DOS data is cached on first run for fast re-plotting.

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `y_limit` | (−4.0, 16.0) THz | Frequency axis range |
| `dos_sigma` | 0.05 THz | Gaussian broadening for DOS |
| `recalculate_dos` | `False` | Force DOS recalculation (ignore cache) |
| Colormap | `coolwarm` | Temperature colour mapping (blue = cold, red = hot) |

### Output

| File | Description |
|------|-------------|
| `./{T}K/phonon_band_dos_{T}K.png` | Band structure + total DOS panel (per temperature) |
| `./phonon_dos_temperature_comparison.png` | Overlay of total DOS across all temperatures |
| `./{T}K/dos_cache_{T}K_sigma{σ}.npz` | Cached DOS data for fast re-plotting |

---

## Input Files

| File | Description |
|------|-------------|
| [`1-generate_displacements_improved.py`](./1-generate_displacements_improved.py) | Generates Phonopy finite-displacement structures for all temperatures |
| [`2-batch_mace_forces_improved.py`](./2-batch_mace_forces_improved.py) | Predicts MACE forces on all displaced structures (GPU) |
| [`3-computing_force_constants.py`](./3-computing_force_constants.py) | Collects MACE forces and computes Phonopy force constants |
| [`4-calculate_band_structure_{T}K_oom-fix.py`](./4-calculate_band_structure_10K_oom-fix.py) | Computes phonon band structure + Ru projection for temperature T |
| [`5-plot_band_structure_improved.py`](./5-plot_band_structure_improved.py) | Plots band structure + DOS for all temperatures |
| [`job_script_1-generate_displacements.sh`](./job_script_1-generate_displacements.sh) | SLURM job for Step 1 |
| [`job_script_2-mace_forces_predict_{T}K.sh`](./job_script_2-mace_forces_predict_10K.sh) | SLURM job for Step 2 (one per temperature) |
| [`job_script_3-compute_force_constants.sh`](./job_script_3-compute_force_constants.sh) | SLURM job for Step 3 |
| [`job_script_4-calculate_band_structure_{T}K_oom-fix.sh`](./job_script_4-calculate_band_structure_10K_oom-fix.sh) | SLURM job for Step 4 (one per temperature) |
| `fine-tuned_mace-mp-0b3-medium_compiled.model` | Compiled fine-tuned MACE potential used for force prediction |

---

## Requirements

- **Supercomputer**: LUMI supercomputer at CSC — IT Center for Science (Finland). More details: [docs.lumi-supercomputer.eu](https://docs.lumi-supercomputer.eu/)
- **GPU**: AMD MI250X (partition `standard-g`) for Step 2
- **CPU**: Standard partition (40 tasks) for Steps 1, 3, 4
- **Python**: `cray-python` module (Steps 1, 3, 4, 5); `pytorch/2.7` module (Step 2)
- **Phonopy**: Finite displacement method and force constant computation
- **MACE**: `mace.calculators.MACECalculator` for force prediction
- **ASE**: Reading/writing XYZ and structure files
- **NumPy**: Array operations and data storage
- **Matplotlib**: Band structure and DOS plotting
- **tqdm**: Progress bar display

---
For further details, please refer to the respective folders or contact the author via the provided email.
