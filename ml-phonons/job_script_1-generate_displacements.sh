#!/bin/bash -l

#--------------------------------------------------------------------------------------------------------------------------------
#SBATCH --partition=standard
#SBATCH --ntasks=40
#SBATCH --time=02:00:00
#SBATCH --account=project_462001159
#SBATCH --output=gen_disp_%j.out
#SBATCH --error=gen_disp_%j.err
#SBATCH --get-user-env
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#--------------------------------------------------------------------------------------------------------------------------------

module load cray-python

source /scratch/project_462001159/zakaryou/packages/data_analysis_packages/data_analysis_env/bin/activate

# Set unlimited stack size
ulimit -s unlimited

python3 1-generate_displacements_improved.py

exit 0
