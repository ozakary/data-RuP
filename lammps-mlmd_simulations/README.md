# LAMMPS Input Files for Machine Learning Molecular Dynamics (MLMD) Simulations
---
📄 Author: **Ouail Zakary**
- 📧 Email: [Ouail.Zakary@oulu.fi](mailto:Ouail.Zakary@oulu.fi)
- 🔗 ORCID: [0000-0002-7793-3306](https://orcid.org/0000-0002-7793-3306)
- 🌐 Website: [Personal Webpage](https://cc.oulu.fi/~nmrwww/members/Ouail_Zakary.html)
- 📁 Portfolio: [Academic Portfolio](https://ozakary.github.io/)
---
This directory contains the `LAMMPS` input files and SLURM job scripts for running machine learning-accelerated molecular dynamics (MLMD) simulations of RuP using the fine-tuned `MACE-MP-0b3` interatomic potential. Simulations are performed for the **monoclinic phase** across a wide temperature range, using GPU-accelerated `LAMMPS` with Kokkos and the MACE pair style.

---

## Simulation Overview

The MLMD simulations use the fine-tuned MACE potential (`fine-tuned_mace-mp-0b3-medium_vf_compiled.model-lammps.pt`) to perform long NpT molecular dynamics runs far beyond what is accessible with direct AIMD. Each simulation runs for **110 ps** (110000 steps at 1 fs timestep, 10 ps equilibration and 100  ps production) with a fully flexible simulation cell (triclinic NPT ensemble).

---

## Simulation Parameters (LAMMPS Input)

### System Setup

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Units** | `metal` | LAMMPS metal units (eV, Å, ps) |
| **Atom style** | `atomic` | Standard atomic representation |
| **Boundary conditions** | `p p p` | Periodic in all three directions |
| **Dimensions** | 3 | Three-dimensional simulation |
| **Newton** | `on` | Newton's 3rd law pairs enabled |
| **Supercell replication** | 4×5×2 | Replicated from single-crystal input |
| **Box geometry** | Triclinic | Full cell flexibility (`change_box all triclinic`) |

### Interatomic Potential

| Parameter | Value |
|-----------|-------|
| **Pair style** | `mace no_domain_decomposition` |
| **Model file** | `fine-tuned_mace-mp-0b3-medium_vf_compiled.model-lammps.pt` |
| **Element mapping** | Type 1: Ru, Type 2: P |
| **Mass (Ru)** | 101.07 g/mol |
| **Mass (P)** | 30.973761 g/mol |

### MD Integration

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Timestep** | 1 fs (0.001 ps) | Integration time step |
| **Total steps** | 110,000 | Total simulation duration: 10 ps equilibration + 100 ps production |
| **Ensemble** | NpT — triclinic (`tri`) | Fully flexible cell (all 6 cell parameters) |
| **Pressure** | 0.0 bar (isotropic) | Target pressure |
| **Thermostat damping** | 100 × dt = 0.2 ps | Nosé-Hoover thermostat time constant |
| **Barostat damping** | 1000 × dt = 2.0 ps | Nosé-Hoover barostat time constant |
| **Velocity initialization** | Random, seed 123456 | Initialized at target temperature |

### Neighbor List

| Parameter | Value |
|-----------|-------|
| **Skin distance** | 1.0 Å |
| **Algorithm** | `bin` |
| **Update delay** | 5 steps |
| **Update frequency** | Every 1 step |

### Output Settings

| Output | Frequency | File |
|--------|-----------|------|
| Thermodynamic data | Every 10 steps | Screen / log |
| Lattice parameters | Every 10 steps | `lattice_params.dat` |
| Atomic trajectory | Every 10 steps | `trajectory.dump` |
| Restart files | Every 1000 steps | `tmp.restart` |
| Final configuration | End of run | `final_config.data` |

**Thermo quantities logged**: `step`, `temp`, `press`, `pe`, `ke`, `etotal`, `vol`, `lx`, `ly`, `lz`, `xy`, `xz`, `yz`

---

## Simulation Conditions

Simulations are performed using the experimentally determined structure of the **monoclinic phase** of RuP at 11 K as an initial configuration for the MLMD at the following 30 temperatures:

50 K, 100 K, 110 K, 120 K, 130 K, 140 K, 150 K, 160 K, 170 K, 180 K, 190 K, 200 K, 250 K, 260 K, 270 K, 280 K, 290 K, 300 K, 310 K, 320 K, 330 K, 340 K, 350 K, 400 K, 450 K, 500 K, 550 K, 600 K, 650 K, 700 K.

The temperature is controlled via the variable `TEMP` in the input file, adjusted for each individual simulation:

```lammps
variable TEMP equal 50.0   # e.g., 50 K → modify for each run
```

---

## Directory Structure

```
./lammps-mlmd_simulations/
├── monoclinic_pdf_refinement_single_crystal_11K.data
├── fine-tuned_mace-mp-0b3-medium_vf_compiled.model-lammps.pt
├── 50K/
│   ├── rup_mono_lammps.in
│   └── lammps-gpu.sh
├── 100K/
├── 110K/
├── 120K/
├── 130K/
├── 140K/
├── 150K/
├── 160K/
├── 170K/
├── 180K/
├── 190K/
├── 200K/
├── 250K/
├── 260K/
├── 270K/
├── 280K/
├── 290K/
├── 300K/
├── 310K/
├── 320K/
├── 330K/
├── 340K/
├── 350K/
├── 400K/
├── 450K/
├── 500K/
├── 550K/
├── 600K/
├── 650K/
└── 700K/
```

Each temperature subdirectory contains the same set of input files, with only the `TEMP` variable in the LAMMPS input script adjusted accordingly.

---

## Input Files

- [`mono_lammps.in`](./mono_lammps.in): LAMMPS input script defining the simulation setup, potential, ensemble, and output
- [`monoclinic_pdf_refinement_single_crystal_11K.data`](./monoclinic_pdf_refinement_single_crystal_11K.data): Initial atomic structure for the monoclinic phase (from PDF refinement), replicated 3×4×3 at runtime
- [`fine-tuned_mace-mp-0b3-medium_vf_compiled.model-lammps.pt`](https://doi.org/10.5281/zenodo.18709769): TorchScript-compiled fine-tuned MACE potential for use with LAMMPS
- [`lammps-gpu.sh`](./lammps-gpu.sh): SLURM batch job script for GPU-accelerated LAMMPS on LUMI

## Output Files

Each simulation produces the following files:

- `log.lammps`: Full LAMMPS log with thermodynamic output every 25 steps
- `lattice_params.dat`: Time series of all six cell parameters (`lx`, `ly`, `lz`, `xy`, `xz`, `yz`), volume, temperature, and pressure — written every 25 steps
- `trajectory.dump`: Full atomic trajectory in LAMMPS dump format (positions and atom types), written every 25 steps
- `tmp.restart` / `tmp.restart.{step}`: Binary restart files written every 2,500 steps for job continuation
- `final_config.data`: LAMMPS data file of the final atomic configuration

---

## SLURM Job Configuration

| Parameter | Value |
|-----------|-------|
| **GPU** | 1× AMD MI250x |
| **OMP threads** | 1 (`OMP_NUM_THREADS=1`) |

LAMMPS is invoked with the Kokkos GPU backend and the MACE pair style:

```bash
srun lmp -in rup_mono_lammps.in -k on g 1 -sf kk -pk kokkos newton on neigh half
```

- `-sf kk`: Use Kokkos suffix for all compatible styles
- `-k on g 1`: Enable Kokkos on 1 GPUs
- `-pk kokkos`: Activate Kokkos package

## Requirements

- **Supercomputer**: LUMI supercomputer. More details: [docs.lumi-supercomputer.eu/](https://docs.lumi-supercomputer.eu/)
- **GPU**: AMD MI250x
- **LAMMPS**: Custom build with MACE pair style, Kokkos, and LibTorch support
- **MACE-LAMMPS**: Compiled TorchScript model (`.model-lammps.pt`) generated from the fine-tuned MACE checkpoint (see [MACE GitHub](https://github.com/ACEsuit/mace))

---
For further details, please refer to the respective folders or contact the author via the provided email.
