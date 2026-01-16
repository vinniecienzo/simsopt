#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --array=0
#SBATCH --output=/scratch/projects/kaptanoglulab/VG/simsopt/examples/2_Intermediate/Temp_Storage/slurm-%A_%a.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=vmg6966@nyu.edu   

cd $SLURM_SUBMIT_DIR
# Submit an array of N jobs. 
export OMP_NUM_THREADS=1  # number of threads for OpenMP
export MKL_NUM_THREADS=1  # number of threads for Intel MKL
# IMPORTANT: Activate your conda environment 
# (e.g., 'conda activate simsopt_env') or any other required environment before running this script.

# Make Python flush prints immediately
export PYTHONUNBUFFERED=1

#run file with activated environment first
/scratch/projects/kaptanoglulab/VG/run-simsopt.bash python stage_two_aug_lag.py