#!/bin/bash -l

#--------------------------------------------------------------------------------------------------------------------------------
#SBATCH --partition=standard-g
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --account=<project_ID>
#SBATCH --output=calc_band_str_50K_%j.out
#SBATCH --error=calc_band_str_50K_%j.err
#SBATCH --get-user-env
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#--------------------------------------------------------------------------------------------------------------------------------

module load cray-python

source /path/to/data_analysis_env/bin/activate

# Set unlimited stack size
ulimit -s unlimited

python3 -u 4-calculate_band_structure_50K_oom-fix.py

exit 0
