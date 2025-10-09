#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --mem=20000
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --array=0-7
#SBATCH --output=/scratch/projects/kaptanoglulab/EL/simsopt3/simsopt/examples/2_Intermediate/slurm_outputs/slurm-%A_%a.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=egl5916@nyu.edu   

cd $SLURM_SUBMIT_DIR
# Submit an array of N jobs. 
export OMP_NUM_THREADS=1  # number of threads for OpenMP
export MKL_NUM_THREADS=1  # number of threads for Intel MKL
# IMPORTANT: Activate your conda environment 
# (e.g., 'conda activate simsopt_env') or any other required environment before running this script.

# Make Python flush prints immediately
export PYTHONUNBUFFERED=1

#run file with activated environment first
/scratch/projects/kaptanoglulab/EL/run-simsopt.bash python stage_two_optimization_all.py
