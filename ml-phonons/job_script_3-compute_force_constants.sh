#!/bin/bash -l

#--------------------------------------------------------------------------------------------------------------------------------
#SBATCH --partition=standard
#SBATCH --ntasks=40
#SBATCH --time=08:00:00
#SBATCH --account=project_462001159
#SBATCH --output=compt_forces_%j.out
#SBATCH --error=compt_forces_%j.err
#SBATCH --get-user-env
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#--------------------------------------------------------------------------------------------------------------------------------

module load cray-python

source /scratch/project_462001159/zakaryou/packages/data_analysis_packages/data_analysis_env/bin/activate

# Set unlimited stack size
ulimit -s unlimited

python3 3-computing_force_constants.py

exit 0
