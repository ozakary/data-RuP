#!/bin/bash -l

#--------------------------------------------------------------------------------------------------------------------------------
#SBATCH --partition=standard-g
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --gpus-per-node=1
#SBATCH --time=16:00:00
#SBATCH --account=<project_ID>
#SBATCH --output=compt_forces_%j.out
#SBATCH --error=compt_forces_%j.err
#SBATCH --get-user-env
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#--------------------------------------------------------------------------------------------------------------------------------

module load cray-python

source /path/to/data_analysis_env/bin/activate

# Set unlimited stack size
ulimit -s unlimited

python3 3-computing_force_constants.py

exit 0
