#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --array=0
#SBATCH --output=/scratch/projects/kaptanoglulab/VG/simsopt/examples/2_Intermediate/Temp_Storage/slurm-%A_%a.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=vmg6966@nyu.edu    

cd $SLURM_SUBMIT_DIR                                                                                                                        

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# Call the wrapper, letting it launch MPI
/scratch/projects/kaptanoglulab/VG/run-simsopt.bash mpiexec --oversubscribe -n 4 python stage_two_optimization_stochastic_aug_lag.py

