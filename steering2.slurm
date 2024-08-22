#!/bin/bash

#SBATCH --job-name=steering_job_2    # Job name
#SBATCH --nodes=10                 # Request 10 nodes
#SBATCH --ntasks=10                # One task per node
#SBATCH --cpus-per-task=48         # Allocate 48 CPUs per task
#SBATCH --partition=par480-i       # Use the appropriate partition
#SBATCH -o mpi_output.out          # MPI-related output file
#SBATCH -e mpi_errors.err          # MPI-related error file

# Change to the directory where the job was submitted
cd $SLURM_SUBMIT_DIR

#### Load necessary modules ####
module load gcc-runtime    
module load hwloc
module load gnu12
module load openmpi4 

#### Set relevant environment variables ####
export OMP_NUM_THREADS=48  # Use all available threads per task

export MOSEKLM_LICENSE_FILE=/home/users/arthurdo/Julia/mosek.lic

#### Execute the job using mpirun ####
mpirun -np $SLURM_NTASKS julia generate_dataset.jl > code_output.out 
