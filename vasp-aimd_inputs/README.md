# VASP Input Files for Ab Initio Molecular Dynamics (AIMD) Simulations
---
📄 Author: **Ouail Zakary**
- 📧 Email: [Ouail.Zakary@oulu.fi](mailto:Ouail.Zakary@oulu.fi)
- 🔗 ORCID: [0000-0002-7793-3306](https://orcid.org/0000-0002-7793-3306)
- 🌐 Website: [Personal Webpage](https://cc.oulu.fi/~nmrwww/members/Ouail_Zakary.html)
- 📁 Portfolio: [Academic Portfolio](https://ozakary.github.io/)
---
This directory contains the input files and job scripts for performing ab initio molecular dynamics (AIMD) simulations using VASP. Simulations were carried out for two phases of RuP — the **monoclinic** and **orthorhombic** phases — across multiple temperatures to study phase transition behavior and short-range order.

## Computational Parameters (INCAR)

- **System**: RuP monoclinic / orthorhombic phase
- **Functional**: PBE with DFT-D4 dispersion correction (`IVDW=13`)
- **Plane-wave cutoff**: 650 eV
- **Precision**: Normal
- **Electronic convergence**: 1×10<sup>-6</sup> eV
- **Smearing**: Gaussian, 0.01 eV (`ISMEAR=0`, `SIGMA=0.01`)
- **Real-space projection**: Automatic (`LREAL=A`)
- **Parallelization**: 64 bands per processor (`NPAR=64`)
- **Wave functions**: Not written (`LWAVE=FALSE`)
- **Charge density**: Not written (`LCHARG=FALSE`)
- **MD Algorithm**: Langevin thermostat (`MDALGO=3`)
- **Ionic steps per run**: 500 (`NSW=500`)
- **Time step**: 1 fs (`POTIM=1`)
- **Symmetry**: Disabled (`ISYM=0`)
- **Stress tensor**: Computed (`ISIF=3`)
- **Langevin friction (ionic)**: 10.0 ps<sup>-1</sup> (`LANGEVIN_GAMMA=10.0 10.0`)
- **Langevin friction (lattice)**: 10.0 ps<sup>-1</sup> (`LANGEVIN_GAMMA_L=10.0`)
- **Ionic mass**: 1000 (`PMASS=1000`)
- **Start from**: Existing wave functions (`ISTART=2`)

## k-Point Sampling (KPOINTS)

- **Scheme**: Γ-point only
- **Grid**: 1×1×1 (Monkhorst-Pack / Gamma-centered)

## Pseudopotentials (POTCAR)

| Element | Pseudopotential | Core Configuration | Valence Electrons |
|---------|----------------|-------------------|-------------------|
| **Ru** | PAW_PBE Ru_sv_GW | [Ar] 3d<sup>10</sup> | 4s<sup>2</sup>4p<sup>6</sup>4d<sup>8</sup> (16 electrons) |
| **P** | PAW_PBE P_GW | [Ne] | 3s<sup>2</sup>3p<sup>3</sup> (5 electrons) |

## Simulation Conditions

AIMD simulations were performed independently for each phase at the following temperatures:

| Phase | Temperatures |
|-------|-------------|
| **Monoclinic** | 11 K, 50 K, 100 K, 150 K, 200 K, 250 K, 270 K, 300 K |
| **Orthorhombic** | 200 K, 250 K, 300 K, 330 K, 350 K, 400 K, 450 K, 500 K |

Each simulation run consists of 500 ionic steps (1 fs timestep), with multiple consecutive runs chained together via the job script (i.e., the `CONTCAR` from one run is used as the `POSCAR` for the next). The trajectory data accumulates across runs via incrementally backed-up `XDATCAR` files.

## Directory Structure

```
./vasp-aimd_inputs/
├── monoclinic/
│   ├── 11K/
│   │   ├── INCAR
│   │   ├── KPOINTS
│   │   ├── POSCAR
│   │   ├── POTCAR
│   │   └── aimd.job
│   ├── 50K/
│   ├── 100K/
│   ├── 150K/
│   ├── 200K/
│   ├── 250K/
│   ├── 270K/
│   └── 300K/
├── orthorhombic/
│   ├── 200K/
│   │   ├── INCAR
│   │   ├── KPOINTS
│   │   ├── POSCAR
│   │   ├── POTCAR
│   │   └── aimd.job
│   ├── 250K/
│   ├── 300K/
│   ├── 330K/
│   ├── 350K/
│   ├── 400K/
│   ├── 450K/
│   └── 500K/
```

## Example Input Files (Monoclinic Phase at 11 K)

- [`INCAR`](./INCAR): AIMD calculation parameters (functional, MD settings, thermostat)
- [`KPOINTS`](./KPOINTS): k-point sampling specification (Γ-point only)
- [`POTCAR`](./POTCAR): Pseudopotential data for Ru and P
- [`POSCAR`](./POSCAR): Initial atomic positions for each phase
- [`aimd.job`](./aimd.job): SLURM batch job script for Mahti

## Output Files

Each simulation run generates the following VASP output files, which are incrementally backed up across successive runs:

- `XDATCAR.{i}`: Full ionic trajectory for run `i`
- `OUTCAR.{i}`: Detailed results (energies, forces, stress tensor) for run `i`
- `CONTCAR.{i}`: Final atomic positions of run `i` (used as input to run `i+1`)
- `OSZICAR.{i}`: SCF convergence information for run `i`
- `REPORT.{i}`: MD-specific output (temperature, kinetic energy) for run `i`
- `vasprun.xml.{i}`: Complete calculation data in XML format for run `i`
- `vaspout.h5.{i}`: HDF5 output for run `i`
- `PCDAT.{i}`: Pair correlation data for run `i`
- `DOSCAR.{i}`: Density of states for run `i`

## Job Script Details

The SLURM job script chains multiple consecutive AIMD runs automatically:
- **MPI tasks**: 1024
- **VASP executable**: `vasp_gam_dftd4` (Γ-point version with DFT-D4 support)
- **Run chaining**: `CONTCAR` from run `i` is automatically copied to `POSCAR` for run `i+1`

## Requirements

- **Supercomputer**: Mahti supercomputer at CSC — IT Center for Science (Finland). More details: [www.mahti.csc.fi](https://www.mahti.csc.fi)
- **VASP**: Version 6.3.2 with DFT-D4 support
---
For further details, please refer to the respective folders or contact the author via the provided email.
