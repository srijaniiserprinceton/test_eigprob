#!/bin/bash
#SBATCH --job-name=MCMC_MPI_eigprob      # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=10              # number of tasks per node
# SBATCH --cpus-per-task=40       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=60G
# SBATCH --mem-per-cpu=18G        # memory per cpu-core (4G is default)
#SBATCH --time=00:10:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=sbdas@princeton.edu

module purge
module load anaconda3
module load openmpi/gcc
conda activate helio
mpiexec -n 4 python src/mcmc.py
