# Python Scripts for Local-Order Metrics
---
📄 Author: **Ouail Zakary**
- 📧 Email: [Ouail.Zakary@oulu.fi](mailto:Ouail.Zakary@oulu.fi)
- 🔗 ORCID: [0000-0002-7793-3306](https://orcid.org/0000-0002-7793-3306)
- 🌐 Website: [Personal Webpage](https://cc.oulu.fi/~nmrwww/members/Ouail_Zakary.html)
- 📁 Portfolio: [Academic Portfolio](https://ozakary.github.io/)
---
This directory contains the Python scripts used to compute and visualize local structural order metrics from the RuP MLMD trajectories. Five categories of metrics are covered: (1) Ru–Ru distances along the [110] direction, (2) dimer bond order, (3) trimer angle order, (4) trimer bond order, and (5) time auto-correlation and space correlation functions.

---

## Directory Structure

```
<root>/data_analysis/
├── ru-ru_along-110/
│   ├── code_distances_compute_vf2.py
│   └── code_distances_plot_all-temps_kde_2d-map_vf3.py
├── dimer_bond_order/
│   ├── code_dimer_bond_order_compute.py
│   └── code_dimer_bond_order_plot_vf.py
├── trimer_angle_order/
│   ├── code_trimer_angle_order_compute_v2.py
│   └── code_trimer_angle_order_plot_vf.py
├── trimer_bond_order/
│   ├── code_trimer_bond_order_plot_vf.py
│   └── code_trimer_bond_order_plot_vf.py
└── correlation_functions/
    ├── time_correlation/
    │   ├── compute_time_correlation.py
    │   └── plot_time_correlation.py
    └── space_correlation/
        ├── compute_space_correlation_para-perp.py
        └── plot_space_correlation.py
```

---

## 1. Ru–Ru Distances Along the [110] Direction

### Step 1 — Compute

Run from `<root>/data_analysis/ru-ru_along-110/`:

```bash
python3 code_distances_compute_vf2.py
```

Computes the Ru–Ru distances along the [110] crystallographic direction as a function of temperatures, reading from the XYZ trajectories in each temperature's `lammps_out/` directory.

### Step 2 — Plot

Run from `<root>/data_analysis/ru-ru_along-110/`:

```bash
python3 code_distances_plot_all-temps_kde_2d-map_vf3.py
```

Produces KDE (kernel density estimate) distributions and 2D time–distance maps of the Ru–Ru distances along [110] across all temperatures.

### Input / Output

| | Path |
|---|---|
| **Input** (per temperature) | `../../{T}K/lammps_out/rup_traj_sampled-10_100ps.xyz` |
| **Output** | Ru-Ru distance data and plots |

---

## 2. Dimer Bond Order

### Step 1 — Compute

Run from `<root>/data_analysis/dimer_bond_order/`:

```bash
python3 code_dimer_bond_order_compute.py
```

Computes the dimer bond order parameter from the MLMD trajectories for all temperatures.

### Step 2 — Plot

Run from `<root>/data_analysis/dimer_bond_order/`:

```bash
python3 code_dimer_bond_order_plot_vf.py
```

Visualizes the dimer bond order parameter as a function of temperature.

### Input / Output

| | Path |
|---|---|
| **Input** (per temperature) | `../../{T}K/lammps_out/rup_traj_sampled-10_100ps.xyz` |
| **Output** | Dimer bond order data and plots |

---

## 3. Trimer Angle Order

### Step 1 — Compute

Run from `<root>/data_analysis/trimer_angle_order/`:

```bash
python3 code_trimer_angle_order_compute_v2.py
python3 fix_population_tracking.py
```

Computes the trimer angle order parameter from the MLMD trajectories for all temperatures.

### Step 2 — Plot

Run from `<root>/data_analysis/trimer_angle_order/`:

```bash
python3 code_trimer_angle_order_plot_vf.py
```

Produces plots of the trimer angle order parameter as a function of temperature.

### Input / Output

| | Path |
|---|---|
| **Input** (per temperature) | `../../{T}K/lammps_out/rup_traj_sampled-10_100ps.xyz` |
| **Output** | Trimer angle order data and plots |

---

## 4. Trimer Bond Order

### Step 1 — Compute / Plot

Run from `<root>/data_analysis/trimer_bond_order/`:

```bash
python3 code_trimer_bond_order_plot_vf.py
```

### Step 2 — Final Plot

Run from `<root>/data_analysis/trimer_bond_order/`:

```bash
python3 code_trimer_bond_order_plot_vf.py
```

Computes and visualizes the trimer bond order parameter across all temperatures.

### Input / Output

| | Path |
|---|---|
| **Input** (per temperature) | `../../{T}K/lammps_out/rup_traj_sampled-10_100ps.xyz` |
| **Output** | Trimer bond order data and plots |

---

## 5. Correlation Functions

### 5.1 Time Auto-Correlation Function

Run from `<root>/data_analysis/correlation_functions/time_correlation/`:

#### Step 1 — Compute

```bash
python3 compute_time_correlation.py \
  -i ../../dimer_bond_order/ \
  -o ./time_corr_data/ \
  -pattern "dimer_order_*K.csv" \
  -maxtime 500 \
  --timestep-ps 0.01
```

Computes the time auto-correlation function of the dimer bond order parameter from the data produced in Section 2.

Key arguments:

| Argument | Value | Description |
|----------|-------|-------------|
| `-i` | `../../dimer_bond_order/` | Directory containing `dimer_order_{T}K.csv` files |
| `-o` | `./time_corr_data/` | Output directory for correlation data |
| `-pattern` | `dimer_order_*K.csv` | File pattern to match input CSV files |
| `-maxtime` | 101 | Maximum lag time in frames |
| `--timestep-ps` | 0.01 | The timestep in the XYZ trajectory (1 fs * 10 sampling steps) |

#### Step 2 — Plot

```bash
python3 plot_time_correlation.py \
  -i ./time_corr_data/ \
  -o ./figures/dimer \
  --tmax 2.0 \
  --smooth-sigma 2.0 \
  --show
```

Key arguments:

| Argument | Value | Description |
|----------|-------|-------------|
| `-i` | `./time_corr_data/` | Input directory with computed correlation data |
| `-o` | `./figures/dimer` | Output figure filename (prefix) |
| `--tmax` | 2.0 | Fit C(t) only up to this time in ps (default: full range) |
| `--smooth-sigma` | 2.0 | Gaussian smoothing sigma for FFT T_osc extraction in bins (default: 2.0, set 0 to disable) |
| `--show` | True | To show the plots |

#### Output

- Time auto-correlation function C(t) of the dimer bond order parameter for all temperatures
- **Auto-correlation time** τ extracted from the decay of C(t)
- **Oscillation frequency** extracted from C(t)
- **Long-time plateau** extracted from C(t)

### Input / Output

| | Path |
|---|---|
| **Input** | `../../dimer_bond_order/dimer_order_{T}K.csv` |
| **Intermediate** | `./time_corr_data/` |
| **Output** | C(t) data and plots |

---

### 5.2 Space Correlation Function

Run from `<root>/data_analysis/correlation_functions/space_correlation/`:

**Step 1 — Compute:**

```bash
python3 compute_space_correlation_para-perp.py \
  -parent ../../../ \
  -i rup_traj_sampled-10_100ps \
  -o ./space_corr_parallel_step-10/ \
  -delta 0.5 \
  -start 0 \
  -step 10 \
  --max-distance 30 \
  --parallel
```

**Step 2 — Plot:**

```bash
python3 plot_space_correlation.py \
  -i ./space_corr_parallel_step-10/ \
  -o ./figures/space_corr_parallel_plot_step-10 \
  --xmax 30 \
  --rmax 25 \
  --show
```

#### Key Arguments (compute)

| Argument | Value | Description |
|----------|-------|-------------|
| `-parent` | `../../../` | Root directory containing temperature subfolders |
| `-i` | `rup_traj_sampled-10_100ps` | XYZ file pattern to match |
| `-o` | `./space_corr_parallel_step-10/` | Output directory |
| `-delta` | 0.5 Å | Spatial binning resolution |
| `-start` | 0 | First frame to process |
| `-step` | 10 | Frame stride (every 10th frame used) |
| `--max-distance` | 30 Å | Maximum correlation distance |
| `--parallel` | — | Compute along-chain (parallel) correlation instead of inter-chain |

#### Key Arguments (plot)

| Argument | Value | Description |
|----------|-------|-------------|
| `-i` | `./space_corr_parallel_step-10/` | Input directory |
| `-o` | `figures/space_corr_parallel_plot_step-10` | Output figure prefix |
| `--xmax` | 30 Å | Maximum distance shown on x-axis |
| `--rmax` | 25 Å | Maximum distance for xi integration in A (default: full range). Use to exclude flat zero tail |
| `--show` | True | To show the plots |

#### Output

- **Space correlation function** C(r) and its corresponding **correlation lengths** ξ<sub>sm</sub> and ξ<sub>oz</sub> for all temperatures

### Input / Output

| | Path |
|---|---|
| **Input** (per temperature) | `../../../{T}K/lammps_out/rup_traj_sampled-10_100ps.xyz` |
| **Intermediate** | `./space_corr_parallel/` |
| **Output** | Space correlation function data and plots |

---

## Input Files

| File | Location | Description |
|------|----------|-------------|
| [`code_distances_compute_vf2.py`](./code_distances_compute_vf2.py) | `ru-ru_along-110-direction_vs_time_and_distr/` | Computes Ru–Ru distances along [110] |
| [`code_distances_plot_all-temps_kde_2d-map_vf3.py`](./code_distances_plot_all-temps_kde_2d-map_vf3.py) | `ru-ru_along-110-direction_vs_time_and_distr/` | Plots KDE distributions and 2D maps |
| [`code_dimer_bond_order_compute.py`](./code_dimer_bond_order_compute.py) | `dimer_bond_order/` | Computes dimer bond order parameter |
| [`code_dimer_bond_order_plot_vf.py`](./code_dimer_bond_order_plot_vf.py) | `dimer_bond_order/` | Plots dimer bond order |
| [`code_trimer_angle_order_compute_v2.py`](./code_trimer_angle_order_compute_v2.py) | `trimer_angle_order/` | Computes trimer angle order parameter |
| [`fix_population_tracking.py`](./fix_population_tracking.py) | `trimer_angle_order/` | To fix the population assignements |
| [`code_trimer_angle_order_plot_vf.py`](./code_trimer_angle_order_plot_vf.py) | `trimer_angle_order/` | Plots trimer angle order |
| [`code_trimer_bond_order_plot_vf.py`](./code_trimer_bond_order_plot_vf.py) | `trimer_bond_order/` | Computes and plots trimer bond order |
| [`code_trimer_bond_order_plot_vf.py`](./code_trimer_bond_order_plot_vf.py) | `trimer_bond_order/` | Final-version plot of trimer bond order |
| [`compute_time_correlation.py`](./compute_time_correlation.py) | `correlation_functions/time_correlation/` | Computes time auto-correlation of dimer bond order |
| [`plot_time_correlation.py`](./plot_time_correlation.py) | `correlation_functions/time_correlation/` | Plots time auto-correlation, relaxation time, oscillation period, and long-time plateau |
| [`compute_space_correlation_para-perp.py`](./compute_space_correlation_para-perp.py) | `correlation_functions/space_correlation/` | Computes perpendicular and parallel space correlation functions |
| [`plot_space_correlation.py`](./plot_space_correlation.py) | `correlation_functions/space_correlation/` | Plots space correlation functions and extracts correlation lengths |

---

## Requirements

- **Python**: ≥ 3.8
- **NumPy**: Array operations
- **Matplotlib**: Plotting and figure export
- **SciPy**: Smoothing, envelope fitting for correlation length extraction
- **ASE**: Reading XYZ trajectory files
- **pandas**: Reading and writing CSV files
- **tqdm**: Progress bar display

---
For further details, please refer to the respective folders or contact the author via the provided email.
