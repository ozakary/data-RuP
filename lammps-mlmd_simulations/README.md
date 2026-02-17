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

The MLMD simulations use the fine-tuned MACE potential (`fine-tuned_mace-mp-0b3-medium_compiled.model-lammps.pt`) to perform long NpT molecular dynamics runs far beyond what is accessible with direct AIMD. Each simulation runs for **100 ps** (50000 steps at 2 fs timestep) with a fully flexible simulation cell (triclinic NPT ensemble).

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
| **Supercell replication** | 3×4×3 | Replicated from single-crystal input |
| **Box geometry** | Triclinic | Full cell flexibility (`change_box all triclinic`) |

### Interatomic Potential

| Parameter | Value |
|-----------|-------|
| **Pair style** | `mace no_domain_decomposition` |
| **Model file** | `fine-tuned_mace-mp-0b3-medium_compiled.model-lammps.pt` |
| **Element mapping** | Type 1: Ru, Type 2: P |
| **Mass (Ru)** | 101.07 g/mol |
| **Mass (P)** | 30.973761 g/mol |

### MD Integration

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Timestep** | 2 fs (0.002 ps) | Integration time step |
| **Total steps** | 50,000 | Total simulation duration: 100 ps |
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
| Thermodynamic data | Every 25 steps | Screen / log |
| Lattice parameters | Every 25 steps | `lattice_params.dat` |
| Atomic trajectory | Every 25 steps | `trajectory.dump` |
| Restart files | Every 2,500 steps | `tmp.restart` |
| Final configuration | End of run | `final_config.data` |

**Thermo quantities logged**: `step`, `temp`, `press`, `pe`, `ke`, `etotal`, `vol`, `lx`, `ly`, `lz`, `xy`, `xz`, `yz`

---

## Simulation Conditions

Simulations are performed for the **monoclinic phase** of RuP at the following 23 temperatures:

10 K, 50 K, 100 K, 150 K, 200 K, 250 K, 300 K, 310 K, 320 K, 330 K, 340 K, 350 K, 360 K, 370 K, 380 K, 390 K, 400 K, 450 K, 500 K, 550 K, 600 K, 650 K, 700 K

The temperature is controlled via the variable `TEMP` in the input file, adjusted for each individual simulation:

```lammps
variable TEMP equal 10.0   # e.g., 10 K → modify for each run
```

---

## Directory Structure

```
./lammps-mlmd_simulations/
├── 10K/
│   ├── mono_lammps.in
│   ├── monoclinic_pdf_refinement_single_crystal_11K.data
│   ├── fine-tuned_mace-mp-0b3-medium_compiled.model-lammps.pt
│   └── lammps-gpu.sh
├── 50K/
├── 100K/
├── 150K/
├── 200K/
├── 250K/
├── 300K/
├── 310K/
├── 320K/
├── 330K/
├── 340K/
├── 350K/
├── 360K/
├── 370K/
├── 380K/
├── 390K/
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
- `fine-tuned_mace-mp-0b3-medium_compiled.model-lammps.pt`: TorchScript-compiled fine-tuned MACE potential for use with LAMMPS
- [`lammps-gpu.sh`](./lammps-gpu.sh): SLURM batch job script for GPU-accelerated LAMMPS on Mahti

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
| **GPU** | 1× NVIDIA A100 |
| **OMP threads** | 1 (`OMP_NUM_THREADS=1`) |

### Software Environment

| Component | Version / Path |
|-----------|---------------|
| GCC | 11.2.0 |
| OpenMPI | 4.1.2 |
| FFTW | 3.3.10-mpi |
| CUDA | 11.5.0 |
| cuDNN | 8.3.3.40-11.5 |
| Intel MKL | 2021.4.0 |
| LAMMPS | Custom build with MACE + Kokkos (`lammps-mace`) |
| LibTorch | Installed at `$LIBTORCH_DIR` |

LAMMPS is invoked with the Kokkos GPU backend and the MACE pair style:

```bash
srun -n 1 lmp -sf kk -k on g 4 -pk kokkos -in mono_lammps.in
```

- `-sf kk`: Use Kokkos suffix for all compatible styles
- `-k on g 4`: Enable Kokkos on 4 GPUs
- `-pk kokkos`: Activate Kokkos package

## Requirements

- **Supercomputer**: Mahti supercomputer at CSC — IT Center for Science (Finland). More details: [www.mahti.csc.fi](https://www.mahti.csc.fi)
- **GPU**: NVIDIA A100
- **LAMMPS**: Custom build with MACE pair style, Kokkos, and LibTorch support
- **MACE-LAMMPS**: Compiled TorchScript model (`.model-lammps.pt`) generated from the fine-tuned MACE checkpoint (see [MACE GitHub](https://github.com/ACEsuit/mace))

---
For further details, please refer to the respective folders or contact the author via the provided email.
