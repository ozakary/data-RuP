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
<root>/data_analysis_dense_sampling/
├── ru-ru_along-110-direction_vs_time_and_distr/
│   ├── code_distances_compute_vf.py
│   └── code_distances_plot_all-temps_kde_2d-map_vf3.py
├── dimer_bond_order/
│   ├── code_dimer_bond_order_compute.py
│   └── code_dimer_bond_order_plot_vf.py
├── trimer_angle_order_dashed-curves/
│   ├── code_trimer_angle_order_compute_v2.py
│   └── code_trimer_angle_order_plot_vf.py
├── trimer_bond_order/
│   ├── code_trimer_bond_order_plot.py
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

Run from `<root>/data_analysis_dense_sampling/ru-ru_along-110-direction_vs_time_and_distr/`:

```bash
python3 code_distances_compute_vf.py
```

Computes the Ru–Ru distances along the [110] crystallographic direction as a function of temperatures, reading from the XYZ trajectories in each temperature's `lammps_out/` directory.

### Step 2 — Plot

Run from `<root>/data_analysis_dense_sampling/ru-ru_along-110-direction_vs_time_and_distr/`:

```bash
python3 code_distances_plot_all-temps_kde_2d-map_vf3.py
```

Produces KDE (kernel density estimate) distributions and 2D time–distance maps of the Ru–Ru distances along [110] across all temperatures.

### Input / Output

| | Path |
|---|---|
| **Input** (per temperature) | `../../{T}K/lammps_out/rup_traj_sampled-50_100ps.xyz` |
| **Output** | Ru-Ru distance data and plots |

---

## 2. Dimer Bond Order

### Step 1 — Compute

Run from `<root>/data_analysis_dense_sampling/dimer_bond_order/`:

```bash
python3 code_dimer_bond_order_compute.py
```

Computes the dimer bond order parameter from the MLMD trajectories for all temperatures.

### Step 2 — Plot

Run from `<root>/data_analysis_dense_sampling/dimer_bond_order/`:

```bash
python3 code_dimer_bond_order_plot_vf.py
```

Visualizes the dimer bond order parameter as a function of temperature.

### Input / Output

| | Path |
|---|---|
| **Input** (per temperature) | `../../{T}K/lammps_out/rup_traj_sampled-50_100ps.xyz` |
| **Output** | Dimer bond order data and plots |

---

## 3. Trimer Angle Order

### Step 1 — Compute

Run from `<root>/data_analysis_dense_sampling/trimer_angle_order_dashed-curves/`:

```bash
python3 code_trimer_angle_order_compute_v2.py
```

Computes the trimer angle order parameter from the MLMD trajectories for all temperatures.

### Step 2 — Plot

Run from `<root>/data_analysis_dense_sampling/trimer_angle_order_dashed-curves/`:

```bash
python3 code_trimer_angle_order_plot_vf.py
```

Produces plots of the trimer angle order parameter as a function of temperature.

### Input / Output

| | Path |
|---|---|
| **Input** (per temperature) | `../../{T}K/lammps_out/rup_traj_sampled-50_100ps.xyz` |
| **Output** | Trimer angle order data and plots |

---

## 4. Trimer Bond Order

### Step 1 — Compute / Plot

Run from `<root>/data_analysis_dense_sampling/trimer_bond_order/`:

```bash
python3 code_trimer_bond_order_plot.py
```

### Step 2 — Final Plot

Run from `<root>/data_analysis_dense_sampling/trimer_bond_order/`:

```bash
python3 code_trimer_bond_order_plot_vf.py
```

Computes and visualizes the trimer bond order parameter across all temperatures.

### Input / Output

| | Path |
|---|---|
| **Input** (per temperature) | `../../{T}K/lammps_out/rup_traj_sampled-50_100ps.xyz` |
| **Output** | Trimer bond order data and plots |

---

## 5. Correlation Functions

### 5.1 Time Auto-Correlation Function

Run from `<root>/data_analysis_dense_sampling/correlation_functions/time_correlation/`:

#### Step 1 — Compute

```bash
python3 compute_time_correlation.py \
    -i ../../dimer_bond_order/ \
    -o ./time_corr_data/ \
    -pattern "dimer_order_*K.csv" \
    -maxtime 400
```

Computes the time auto-correlation function of the dimer bond order parameter from the data produced in Section 2.

Key arguments:

| Argument | Value | Description |
|----------|-------|-------------|
| `-i` | `../../dimer_bond_order/` | Directory containing `dimer_order_{T}K.csv` files |
| `-o` | `./time_corr_data/` | Output directory for correlation data |
| `-pattern` | `dimer_order_*K.csv` | File pattern to match input CSV files |
| `-maxtime` | 400 | Maximum lag time in frames |

#### Step 2 — Plot

```bash
python3 plot_time_correlation.py \
    -i ./time_corr_data/ \
    -o time_corr_plot \
    --dt 0.05 \
    --unit ps \
    --xmax 5.0 \
    --no-baseline
```

Key arguments:

| Argument | Value | Description |
|----------|-------|-------------|
| `-i` | `./time_corr_data/` | Input directory with computed correlation data |
| `-o` | `time_corr_plot` | Output figure filename (prefix) |
| `--dt` | 0.05 | Time step between frames in ps (50 fs) |
| `--unit` | `ps` | Time axis unit |
| `--xmax` | 5.0 | Maximum time shown on x-axis (ps) |
| `--no-baseline` | — | Do not subtract baseline from correlation curves |

#### Output

- Time auto-correlation function C(t) of the dimer bond order parameter for all temperatures
- **Relaxation time** τ extracted from the decay of C(t)
- **Oscillation period** extracted from C(t)

### Input / Output

| | Path |
|---|---|
| **Input** | `../../dimer_bond_order/dimer_order_{T}K.csv` |
| **Intermediate** | `./time_corr_data/` |
| **Output** | C(t) data and plots |

---

### 5.2 Space Correlation Functions (Perpendicular and Parallel)

Run from `<root>/data_analysis_dense_sampling/correlation_functions/space_correlation/`:

Two directional space correlation functions are computed: **perpendicular** (inter-chain) and **parallel** (along [110] Ru–Ru chains).

#### Perpendicular (Inter-Chain) Correlation

**Step 1 — Compute:**

```bash
python3 compute_space_correlation_para-perp.py \
    -parent ../../../ \
    -i rup_traj_sampled-50_100ps \
    -o ./space_corr_perp/ \
    -delta 1.0 \
    -start 0 \
    -step 10 \
    --max-distance 40
```

**Step 2 — Plot:**

```bash
python3 plot_space_correlation.py \
    -i ./space_corr_perp/ \
    -o space_corr_perp_plot \
    --xmax 40 \
    --smooth \
    --xi-method envelope
```

#### Parallel (Along [110] Chains) Correlation

**Step 1 — Compute:**

```bash
python3 compute_space_correlation_para-perp.py \
    -parent ../../../ \
    -i rup_traj_sampled-50_100ps \
    -o ./space_corr_parallel/ \
    -delta 1.0 \
    -start 0 \
    -step 10 \
    --max-distance 40 \
    --parallel
```

**Step 2 — Plot:**

```bash
python3 plot_space_correlation.py \
    -i ./space_corr_parallel/ \
    -o space_corr_parallel_plot \
    --xmax 40 \
    --smooth \
    --xi-method envelope
```

#### Key Arguments (compute)

| Argument | Value | Description |
|----------|-------|-------------|
| `-parent` | `../../../` | Root directory containing temperature subfolders |
| `-i` | `rup_traj_sampled-50_100ps` | XYZ file pattern to match |
| `-o` | `./space_corr_perp/` or `./space_corr_parallel/` | Output directory |
| `-delta` | 1.0 Å | Spatial binning resolution |
| `-start` | 0 | First frame to process |
| `-step` | 10 | Frame stride (every 10th frame used) |
| `--max-distance` | 40 Å | Maximum correlation distance |
| `--parallel` | — | Compute along-chain (parallel) correlation instead of inter-chain |

#### Key Arguments (plot)

| Argument | Value | Description |
|----------|-------|-------------|
| `-i` | `./space_corr_perp/` or `./space_corr_parallel/` | Input directory |
| `-o` | `space_corr_perp_plot` or `space_corr_parallel_plot` | Output figure prefix |
| `--xmax` | 40 Å | Maximum distance shown on x-axis |
| `--smooth` | — | Apply smoothing to correlation curves |
| `--xi-method` | `envelope` | Method for extracting correlation length ξ (envelope fit) |

#### Output

- **Perpendicular space correlation function** C⊥(r) and its corresponding **correlation length** ξ⊥ for all temperatures
- **Parallel space correlation function** C∥(r) and its corresponding **correlation length** ξ∥ for all temperatures

### Input / Output

| | Path |
|---|---|
| **Input** (per temperature) | `../../../{T}K/lammps_out/rup_traj_sampled-50_100ps.xyz` |
| **Intermediate** | `./space_corr_perp/` and `./space_corr_parallel/` |
| **Output** | Space correlation function data and plots |

---

## Input Files

| File | Location | Description |
|------|----------|-------------|
| [`code_distances_compute_vf.py`](./code_distances_compute_vf.py) | `ru-ru_along-110-direction_vs_time_and_distr/` | Computes Ru–Ru distances along [110] |
| [`code_distances_plot_all-temps_kde_2d-map_vf3.py`](./code_distances_plot_all-temps_kde_2d-map_vf3.py) | `ru-ru_along-110-direction_vs_time_and_distr/` | Plots KDE distributions and 2D maps |
| [`code_dimer_bond_order_compute.py`](./code_dimer_bond_order_compute.py) | `dimer_bond_order/` | Computes dimer bond order parameter |
| [`code_dimer_bond_order_plot_vf.py`](./code_dimer_bond_order_plot_vf.py) | `dimer_bond_order/` | Plots dimer bond order |
| [`code_trimer_angle_order_compute_v2.py`](./code_trimer_angle_order_compute_v2.py) | `trimer_angle_order_dashed-curves/` | Computes trimer angle order parameter |
| [`code_trimer_angle_order_plot_vf.py`](./code_trimer_angle_order_plot_vf.py) | `trimer_angle_order_dashed-curves/` | Plots trimer angle order |
| [`code_trimer_bond_order_plot.py`](./code_trimer_bond_order_plot.py) | `trimer_bond_order/` | Computes and plots trimer bond order |
| [`code_trimer_bond_order_plot_vf.py`](./code_trimer_bond_order_plot_vf.py) | `trimer_bond_order/` | Final-version plot of trimer bond order |
| [`compute_time_correlation.py`](./compute_time_correlation.py) | `correlation_functions/time_correlation/` | Computes time auto-correlation of dimer bond order |
| [`plot_time_correlation.py`](./plot_time_correlation.py) | `correlation_functions/time_correlation/` | Plots time auto-correlation, relaxation time, oscillation period |
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
