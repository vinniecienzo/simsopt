#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20000
#SBATCH --account=torch_pr_292_courant
#SBATCH --array=0-15
#SBATCH --output=slurm_outputs/slurm-%A_%a.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=egl5916@nyu.edu
#SBATCH --account=torch_pr_292_courant
cd $SLURM_SUBMIT_DIR                                                                                                    

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# Call the wrapper, letting it launch MPI
../../../../run-simsopt.bash python stage_two_optimization.py
