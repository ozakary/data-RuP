#!/bin/bash -l

#--------------------------------------------------------------------------------------------------------------------------------
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --time=16:00:00
#SBATCH --account=project_462001159
#SBATCH --output=calc_band_str_10K_%j.out
#SBATCH --error=calc_band_str_10K_%j.err
#SBATCH --get-user-env
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#--------------------------------------------------------------------------------------------------------------------------------

module load cray-python

source /scratch/project_462001159/zakaryou/packages/data_analysis_packages/data_analysis_env/bin/activate

# Set unlimited stack size
ulimit -s unlimited

python3 -u 4-calculate_band_structure_10K_oom-fix.py

exit 0
