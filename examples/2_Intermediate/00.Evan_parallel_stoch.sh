#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=20000
#SBATCH --array=0-7
#SBATCH --output=/scratch/projects/kaptanoglulab/EL/simsopt3/simsopt/examples/2_Intermediate/slurm_outputs/slurm-%A_%a.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=egl5916@nyu.edu   

cd $SLURM_SUBMIT_DIR                                                                                                                        

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# Call the wrapper, letting it launch MPI
/scratch/projects/kaptanoglulab/VG/run-simsopt.bash mpiexec --oversubscribe -n 16 python stage_two_optimization_stochastic_all.py