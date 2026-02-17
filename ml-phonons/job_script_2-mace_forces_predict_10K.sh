#!/bin/bash
#--------------------------------------------------------------------------------------------------------------------------------
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=18:00:00
#SBATCH --account=project_462001159
#SBATCH --output=mlp-predict_%j.out
#SBATCH --error=mlp-predict_%j.err
#SBATCH --get-user-env
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#--------------------------------------------------------------------------------------------------------------------------------

module use /appl/local/csc/modulefiles/
module load pytorch/2.7

# Using the 'ulimit' command to control the user-level resource limits for processes. the option '-s unlimited' specifies that 
# there should be no limit on the stack size for the processes launched by the job. The stack size is the amount of memory 
# allocated for the call stack of a program.
ulimit -s unlimited

# Start of training the MACE architecture :

python3 2-batch_mace_forces_improved.py \
    -m ./fine-tuned_mace-mp-0b3-medium_compiled.model \
    -i 10K/displaced_structures/all_displaced_structures.xyz \
    -o 10K/displaced_structures/all_displaced_structures_out.xyz \
    -dtype float64 \
    --save-frequency 1

exit 0
