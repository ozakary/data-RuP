#!/bin/bash -l

#--------------------------------------------------------------------------------------------------------------------------------
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --gpus-per-node=1
#SBATCH --time=03:00:00
#SBATCH --account=<project_ID>
#SBATCH --output=gen_disp_%j.out
#SBATCH --error=gen_disp_%j.err
#SBATCH --get-user-env
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#--------------------------------------------------------------------------------------------------------------------------------

module load cray-python

source /path/to/data_analysis_env/bin/activate

# Set unlimited stack size
ulimit -s unlimited

python3 1-generate_displacements_improved.py

exit 0
