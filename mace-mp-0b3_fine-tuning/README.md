# Dataset Preparation and Fine-Tuning of the MACE-MP-0b3 Foundation Model
---
📄 Author: **Ouail Zakary**
- 📧 Email: [Ouail.Zakary@oulu.fi](mailto:Ouail.Zakary@oulu.fi)
- 🔗 ORCID: [0000-0002-7793-3306](https://orcid.org/0000-0002-7793-3306)
- 🌐 Website: [Personal Webpage](https://cc.oulu.fi/~nmrwww/members/Ouail_Zakary.html)
- 📁 Portfolio: [Academic Portfolio](https://ozakary.github.io/)
---
This directory contains the dataset preparation scripts, the `.config` file, and the job script for fine-tuning the `MACE-MP-0b3` atomistic foundation model on AIMD-derived RuP data. The procedure is divided into three stages: (1) trajectory conversion and concatenation, (2) dataset sampling and splitting, and (3) model fine-tuning.

---

## Stage 1 — Trajectory Conversion and Concatenation

### 1.1 Convert VASP OUTCAR Files to XYZ Format

For each AIMD temperature folder (for both monoclinic and orthorhombic phases), the individual `OUTCAR.${i}` files produced by the chained VASP runs are converted to the extended XYZ format using the Atomic Simulation Environment (ASE):

```bash
for i in $(seq 1 1 10); do
    ase convert OUTCAR.${i} traj_${i}.xyz
done
```

This produces 10 XYZ trajectory files per temperature, each containing 500 ionic configurations (1 fs timestep × 500 steps per run).

### 1.2 Concatenate Trajectories per Temperature

The 10 trajectory files are concatenated into a single XYZ file per temperature:

```bash
cat traj_1.xyz traj_2.xyz ... traj_10.xyz \
    > mono_NpT_PBE-D4_traj-5ps_11K.xyz
```

This results in a full trajectory of **5000 configurations (5 ps)** per temperature. The same procedure is repeated for each of the 16 AIMD temperatures:

| Phase | Temperature | Full Dataset File |
|-------|-------------|-------------------|
| Monoclinic | 11 K | `mono_NpT_PBE-D4_traj-5ps_11K.xyz` |
| Monoclinic | 50 K | `mono_NpT_PBE-D4_traj-5ps_50K.xyz` |
| Monoclinic | 100 K | `mono_NpT_PBE-D4_traj-5ps_100K.xyz` |
| Monoclinic | 150 K | `mono_NpT_PBE-D4_traj-5ps_150K.xyz` |
| Monoclinic | 200 K | `mono_NpT_PBE-D4_traj-5ps_200K.xyz` |
| Monoclinic | 250 K | `mono_NpT_PBE-D4_traj-5ps_250K.xyz` |
| Monoclinic | 270 K | `mono_NpT_PBE-D4_traj-5ps_270K.xyz` |
| Monoclinic | 300 K | `mono_NpT_PBE-D4_traj-5ps_300K.xyz` |
| Orthorhombic | 200 K | `ortho_NpT_PBE-D4_traj-5ps_200K.xyz` |
| Orthorhombic | 250 K | `ortho_NpT_PBE-D4_traj-5ps_250K.xyz` |
| Orthorhombic | 300 K | `ortho_NpT_PBE-D4_traj-5ps_300K.xyz` |
| Orthorhombic | 330 K | `ortho_NpT_PBE-D4_traj-5ps_330K.xyz` |
| Orthorhombic | 350 K | `ortho_NpT_PBE-D4_traj-5ps_350K.xyz` |
| Orthorhombic | 400 K | `ortho_NpT_PBE-D4_traj-5ps_400K.xyz` |
| Orthorhombic | 450 K | `ortho_NpT_PBE-D4_traj-5ps_450K.xyz` |
| Orthorhombic | 500 K | `ortho_NpT_PBE-D4_traj-5ps_500K.xyz` |

---

## Stage 2 — Dataset Sampling and Splitting

### 2.1 Concatenate All Datasets

All 16 trajectories (monoclinic + orthorhombic, all temperatures) are merged into a single master dataset file:

```bash
cat mono_NpT_PBE-D4_traj-5ps_11K.xyz \
    mono_NpT_PBE-D4_traj-5ps_50K.xyz \
    mono_NpT_PBE-D4_traj-5ps_100K.xyz \
    mono_NpT_PBE-D4_traj-5ps_150K.xyz \
    mono_NpT_PBE-D4_traj-5ps_200K.xyz \
    mono_NpT_PBE-D4_traj-5ps_250K.xyz \
    mono_NpT_PBE-D4_traj-5ps_270K.xyz \
    mono_NpT_PBE-D4_traj-5ps_300K.xyz \
    ortho_NpT_PBE-D4_traj-5ps_200K.xyz \
    ortho_NpT_PBE-D4_traj-5ps_250K.xyz \
    ortho_NpT_PBE-D4_traj-5ps_300K.xyz \
    ortho_NpT_PBE-D4_traj-5ps_330K.xyz \
    ortho_NpT_PBE-D4_traj-5ps_350K.xyz \
    ortho_NpT_PBE-D4_traj-5ps_400K.xyz \
    ortho_NpT_PBE-D4_traj-5ps_450K.xyz \
    ortho_NpT_PBE-D4_traj-5ps_500K.xyz \
    > mono_and_ortho_all-temps_NpT_PBE-D4_full-traj.xyz
```

The master dataset contains **80000 configurations** spanning both phases and all simulation temperatures.

### 2.2 Sample Trajectories (Every 10 Steps)

To reduce redundancy and decorrelate configurations, the master dataset file is sampled by retaining every 10th ionic step, reducing 80000 frames to **8000 frames**:

For this task, we used the code: [`trajslicer_src.py`](https://github.com/ozakary/TrajSlicer/blob/main/trajslicer_src.py)

```bash
python3 trajslicer_src.py mono_and_ortho_all-temps_NpT_PBE-D4_full-traj.xyz mono_and_ortho_all-temps_NpT_PBE-D4_sampled-10fs-traj_ML-dataset.xyz --sample 10
```

The master sampled dataset file `mono_and_ortho_all-temps_NpT_PBE-D4_sampled-10fs-traj_ML-dataset.xyz` contains **8000 configurations**.

### 2.3 Split into Train / Validation / Test Sets

The master sampled dataset is split into training, validation, and test subsets using [`xyz_splitter.py`](./dataset/xyz_splitter.py). The split is performed randomly with a fixed random seed for reproducibility:

```bash
python3 xyz_splitter.py mono_and_ortho_all-temps_NpT_PBE-D4_sampled-10fs-traj_ML-dataset.xyz \
    --train_size 6400 \
    --valid_size 800 \
    --test_size 800 \
    --rand_split true \
    --output_prefix rup_dataset \
    --seed 123
```

| Split | Size | Output File |
|-------|------|-------------|
| Training | 6,400 configurations | `rup_dataset_train.xyz` |
| Validation | 800 configurations | `rup_dataset_valid.xyz` |
| Test | 800 configurations | `rup_dataset_test.xyz` |

---

## Stage 3 — Fine-Tuning of MACE-MP-0b3

### Directory Structure

```
./mace-mp-0b3_fine-tuning/
├── dataset/
│   ├── sampling_data.py
│   ├── xyz_splitter.py
│   ├── mono_and_ortho_all-temps_NpT_PBE-D4_sampled-10fs-traj_ML-dataset.xyz
│   ├── rup_dataset_train.xyz
│   ├── rup_dataset_valid.xyz
│   └── rup_dataset_test.xyz
├── mace-mp-0b3-medium.model
└── script_mace_fm_fine-tuning.job
```

### 3.1 Foundation Model

The pre-trained foundation model `mace-mp-0b3-medium.model` is placed in the root of the fine-tuning directory before launching the training job. It can be downloaded from the official MACE-MP repository.

### 3.2 Fine-Tuning Parameters

The fine-tuning is launched via:

```bash
sbatch script_mace_fm_fine-tuning.job
```

Key hyperparameters and architectural settings used in `mace_run_train`:

#### Model Architecture

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--foundation_model` | `mace-mp-0b3-medium.model` | Pre-trained foundation model |
| `--multiheads_finetuning` | `False` | Single-head fine-tuning |
| `--num_interactions` | 2 | Number of message-passing layers |
| `--correlation` | 3 | Body-order of the ACE basis |
| `--max_ell` | 3 | Maximum spherical harmonic degree |
| `--max_L` | 2 | Maximum irrep order in hidden layers |
| `--r_max` | 6.0 Å | Cutoff radius |
| `--num_channels` | 128 | Number of hidden channels |
| `--num_radial_basis` | 10 | Number of radial basis functions |
| `--MLP_irreps` | `16x0e` | MLP irreducible representations |
| `--interaction_first` | `RealAgnosticResidualInteractionBlock` | First interaction block type |
| `--interaction` | `RealAgnosticResidualInteractionBlock` | Subsequent interaction block type |

#### Loss Function and Weights

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--loss` | `universal` | Universal loss function |
| `--energy_weight` | 1 | Weight for energy loss term |
| `--forces_weight` | 10 | Weight for forces loss term |
| `--stress_weight` | 100 | Weight for stress loss term |
| `--stress_key` | `REF_stress` | Key for reference stress in XYZ files |
| `--error_table` | `PerAtomMAE` | Error metric reported during training |

#### Training Settings

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--max_num_epochs` | 1000 | Maximum number of training epochs |
| `--patience` | 30 | Early stopping patience (epochs) |
| `--batch_size` | 2 | Training batch size |
| `--valid_batch_size` | 4 | Validation batch size |
| `--eval_interval` | 1 | Validation frequency (every epoch) |
| `--seed` | 3 | Random seed for reproducibility |
| `--default_dtype` | `float64` | Floating point precision |
| `--device` | `cuda` | GPU-accelerated training |

#### Optimizer Settings

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--lr` | 0.005 | Initial learning rate |
| `--weight_decay` | 1×10<sup>-8</sup> | L2 regularization |
| `--ema` | — | Exponential moving average of weights |
| `--ema_decay` | 0.995 | EMA decay rate |
| `--amsgrad` | — | Use AMSGrad variant of Adam |
| `--scheduler_patience` | 5 | LR scheduler patience (epochs) |
| `--clip_grad` | 100 | Gradient clipping threshold |
| `--scaling` | `rms_forces_scaling` | Input scaling strategy |

#### Other Settings

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--atomic_numbers` | `"[15, 44]"` | Atomic numbers for P and Ru, respectively |
| `--E0s` | `"{15: -0.01998124, 44: -0.13979238}"` | Atomic energy references |
| `--enable_cueq` | `True` | Enable CuEq for GPU-accelerated equivariant ops |
| `--restart_latest` | — | Resume training from the latest checkpoint |

### 3.3 SLURM Job Configuration

| Parameter | Value |
|-----------|-------|
| **Nodes** | 1 |
| **MPI tasks** | 1 |
| **GPU** | 1× NVIDIA A100 |
| **Wall time** | 36 hours |
| **PyTorch module** | `pytorch/2.7` |
| **Virtual environment** | `mace_env_2` |

### 3.4 Output Files

Upon completion, `mace_run_train` produces the following files in the fine-tuning directory:

- `fine-tuned_mace-mp-0b3-medium_vf.model`: Final fine-tuned model (float64 precision)
- `fine-tuned_mace-mp-0b3-medium_vf_compiled.model`: Torch-compiled model for deployment
- `fine-tuned_mace-mp-0b3-medium_vf_run-{n}.log`: Training log with per-epoch metrics
- `checkpoints/`: Directory containing model checkpoints saved during training

## Input Files

- [`trajslicer_src.py`](https://github.com/ozakary/TrajSlicer/blob/main/trajslicer_src.py): Script to sample AIMD trajectories (every 10th step)
- [`xyz_splitter.py`](./dataset/xyz_splitter.py): Script to randomly split the master XYZ dataset into train/validation/test sets
- [`mono_and_ortho_all-temps_NpT_PBE-D4_sampled-10fs-traj_ML-dataset.xyz`](https://doi.org/10.5281/zenodo.18709769): XYZ dataset (all phases and temperatures, subsampled)
- [`mace-mp-0b3-medium.model`](./https://github.com/ACEsuit/mace-foundations/releases/tag/mace_mp_0b3): Pre-trained MACE-MP-0b3 foundation model (medium size)
- [`script_mace_fm_fine-tuning.job`](./script_mace_fm_fine-tuning.job): SLURM batch job script for fine-tuning on Mahti

## Requirements

- **Supercomputer**: Mahti supercomputer at CSC — IT Center for Science (Finland). More details: [www.mahti.csc.fi](https://www.mahti.csc.fi)
- **GPU**: NVIDIA A100
- **Python**: PyTorch module `pytorch/2.7`
- **MACE**: Installed in virtual environment `mace_env_2` (see [MACE GitHub](https://github.com/ACEsuit/mace))
- **ASE**: Atomic Simulation Environment, for trajectory conversion (`ase convert`)

---
For further details, please refer to the respective folders or contact the author via the provided email.
