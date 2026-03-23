#!/bin/bash -l
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --time=48:00:00
#SBATCH --account=<project_ID>
#SBATCH --output=test_lammps-gpu_%j.out
#SBATCH --error=test_lammps-gpu_%j.err

ml LUMI
ml partition/G
ml PrgEnv-amd
ml rocm
ml cray-fftw

export MPICH_GPU_SUPPORT_ENABLED=1
export OMP_NUM_THREADS=1
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export LD_LIBRARY_PATH=/path/to/torch/lib/:$LD_LIBRARY_PATH

srun /path/to/lmp \
  -in rup_mono_lammps.in -k on g 1 -sf kk -pk kokkos newton on neigh half
