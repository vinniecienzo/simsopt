#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --nodes=1
<<<<<<< HEAD
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --array=0-7
#SBATCH --output=/scratch/projects/kaptanoglulab/VG/simsopt/examples/2_Intermediate/Temp_Storage/slurm-%A_%a.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=vmg6966@nyu.edu    
=======
#SBATCH --cpus-per-task=16
<<<<<<< HEAD
#SBATCH --mem=20000
#SBATCH --array=0-7
#SBATCH --output=/scratch/projects/kaptanoglulab/EL/simsopt3/simsopt/examples/2_Intermediate/slurm_outputs/slurm-%A_%a.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=vmg6966@nyu.edu  
>>>>>>> EL_1

=======
#SBATCH --mem=100G
#SBATCH --account=torch_pr_292_courant
#SBATCH --array=0-9
#SBATCH --output=slurm_outputs/slurm-%A_%a.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=vmg6966@nyu.edu   
#SBATCH --account=torch_pr_292_courant
>>>>>>> VG_St_Aug_Lag
cd $SLURM_SUBMIT_DIR                                                                                                                        

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# Call the wrapper, letting it launch MPI
<<<<<<< HEAD
<<<<<<< HEAD
/scratch/projects/kaptanoglulab/VG/run-simsopt.bash mpiexec --oversubscribe -n 4 python stage_two_optimization_stochastic.py

=======
/scratch/projects/kaptanoglulab/VG/run-simsopt.bash mpiexec --oversubscribe -n 16 python stage_two_optimization_stochastic_all.py
>>>>>>> EL_1
=======
/scratch/vmg6966/run-simsopt.bash mpiexec --oversubscribe -n 16 python stage_two_optimization_stochastic_sigma_mesh.py
>>>>>>> VG_St_Aug_Lag
