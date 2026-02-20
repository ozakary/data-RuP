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
    ase convert OUTCAR.${i} rup_traj_${i}.xyz
done
```

This produces 10 XYZ trajectory files per temperature, each containing 500 ionic configurations (1 fs timestep × 500 steps per run).

### 1.2 Concatenate Trajectories per Temperature

The 10 trajectory files are concatenated into a single XYZ file per temperature:

```bash
cat rup_traj_1.xyz rup_traj_2.xyz ... rup_traj_10.xyz \
    > monoclinic_11K_NpT_DFTD4_ML-dataset.xyz
```

This results in a full trajectory of **5000 configurations (5 ps)** per temperature. The same procedure is repeated for each of the 10 AIMD temperatures:

| Phase | Temperature | Full Dataset File |
|-------|-------------|-------------------|
| Monoclinic | 11 K | `monoclinic_11K_NpT_DFTD4_ML-dataset.xyz` |
| Monoclinic | 100 K | `monoclinic_100K_NpT_DFTD4_ML-dataset.xyz` |
| Monoclinic | 200 K | `monoclinic_200K_NpT_DFTD4_ML-dataset.xyz` |
| Monoclinic | 250 K | `monoclinic_250K_NpT_DFTD4_ML-dataset.xyz` |
| Monoclinic | 270 K | `monoclinic_270K_NpT_DFTD4_ML-dataset.xyz` |
| Monoclinic | 300 K | `monoclinic_300K_NpT_DFTD4_ML-dataset.xyz` |
| Orthorhombic | 300 K | `orthorhombic_300K_NpT_DFTD4_ML-dataset.xyz` |
| Orthorhombic | 370 K | `orthorhombic_370K_NpT_DFTD4_ML-dataset.xyz` |
| Orthorhombic | 400 K | `orthorhombic_400K_NpT_DFTD4_ML-dataset.xyz` |
| Orthorhombic | 450 K | `orthorhombic_450K_NpT_DFTD4_ML-dataset.xyz` |

---

## Stage 2 — Dataset Sampling and Splitting

### 2.1 Subsample Trajectories (Every 10 Steps)

To reduce redundancy and decorrelate configurations, each per-temperature dataset is subsampled by retaining every 10th ionic step, reducing 5000 frames to **500 frames** per temperature:

```bash
python3 sampling_data.py
```

The script [`sampling_data.py`](./sampling_data.py) reads the full XYZ trajectory and writes the subsampled output with the prefix `sampled-2_`:

| Phase | Temperature | Sampled Dataset File |
|-------|-------------|----------------------|
| Monoclinic | 11 K | `sampled-2_monoclinic_11K_NpT_DFTD4_ML-dataset.xyz` |
| Monoclinic | 100 K | `sampled-2_monoclinic_100K_NpT_DFTD4_ML-dataset.xyz` |
| Monoclinic | 200 K | `sampled-2_monoclinic_200K_NpT_DFTD4_ML-dataset.xyz` |
| Monoclinic | 250 K | `sampled-2_monoclinic_250K_NpT_DFTD4_ML-dataset.xyz` |
| Monoclinic | 270 K | `sampled-2_monoclinic_270K_NpT_DFTD4_ML-dataset.xyz` |
| Monoclinic | 300 K | `sampled-2_monoclinic_300K_NpT_DFTD4_ML-dataset.xyz` |
| Orthorhombic | 300 K | `sampled-2_orthorhombic_300K_NpT_DFTD4_ML-dataset.xyz` |
| Orthorhombic | 370 K | `sampled-2_orthorhombic_370K_NpT_DFTD4_ML-dataset.xyz` |
| Orthorhombic | 400 K | `sampled-2_orthorhombic_400K_NpT_DFTD4_ML-dataset.xyz` |
| Orthorhombic | 450 K | `sampled-2_orthorhombic_450K_NpT_DFTD4_ML-dataset.xyz` |

### 2.2 Concatenate All Sampled Datasets

All 10 subsampled trajectories (monoclinic + orthorhombic, all temperatures) are merged into a single master dataset file:

```bash
cat sampled-2_monoclinic_11K_NpT_DFTD4_ML-dataset.xyz \
    sampled-2_monoclinic_100K_NpT_DFTD4_ML-dataset.xyz \
    sampled-2_monoclinic_200K_NpT_DFTD4_ML-dataset.xyz \
    sampled-2_monoclinic_250K_NpT_DFTD4_ML-dataset.xyz \
    sampled-2_monoclinic_270K_NpT_DFTD4_ML-dataset.xyz \
    sampled-2_monoclinic_300K_NpT_DFTD4_ML-dataset.xyz \
    sampled-2_orthorhombic_300K_NpT_DFTD4_ML-dataset.xyz \
    sampled-2_orthorhombic_370K_NpT_DFTD4_ML-dataset.xyz \
    sampled-2_orthorhombic_400K_NpT_DFTD4_ML-dataset.xyz \
    sampled-2_orthorhombic_450K_NpT_DFTD4_ML-dataset.xyz \
    > sampled-2_mono_and_ortho_all_NpT_DFTD4_ML-dataset.xyz
```

The master dataset contains **5000 configurations** spanning both phases and all simulation temperatures.

### 2.3 Split into Train / Validation / Test Sets

The master dataset is split into training, validation, and test subsets using [`xyz_splitter.py`](./dataset/xyz_splitter.py). The split is performed randomly with a fixed random seed for reproducibility:

```bash
python3 xyz_splitter.py sampled-2_mono_and_ortho_all_NpT_DFTD4_ML-dataset.xyz \
    --train_size 1600 \
    --valid_size 200 \
    --test_size 200 \
    --rand_split true \
    --output_prefix rup_dataset \
    --seed 123
```

| Split | Size | Output File |
|-------|------|-------------|
| Training | 1,600 configurations | `rup_dataset_train.xyz` |
| Validation | 200 configurations | `rup_dataset_valid.xyz` |
| Test | 200 configurations | `rup_dataset_test.xyz` |

---

## Stage 3 — Fine-Tuning of MACE-MP-0b3

### Directory Structure

```
./mace-mp-0b3_fine-tuning/
├── dataset/
│   ├── sampling_data.py
│   ├── xyz_splitter.py
│   ├── sampled-2_mono_and_ortho_all_NpT_DFTD4_ML-dataset.xyz
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
| `--E0s` | `average` | Atomic energy references (averaged) |
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

- `fine-tuned_mace-mp-0b3-medium.model`: Final fine-tuned model (float64 precision)
- `fine-tuned_mace-mp-0b3-medium_compiled.model`: Torch-compiled model for deployment
- `fine-tuned_mace-mp-0b3-medium_run-{n}.log`: Training log with per-epoch metrics
- `checkpoints/`: Directory containing model checkpoints saved during training

## Input Files

- [`sampling_data.py`](./dataset/sampling_data.py): Script to subsample AIMD trajectories (every 10th step)
- [`xyz_splitter.py`](./dataset/xyz_splitter.py): Script to randomly split the master XYZ dataset into train/validation/test sets
- [`sampled-2_mono_and_ortho_all_NpT_DFTD4_ML-dataset.xyz`](https://doi.org/10.5281/zenodo.18709769): XYZ dataset (all phases and temperatures, subsampled)
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
