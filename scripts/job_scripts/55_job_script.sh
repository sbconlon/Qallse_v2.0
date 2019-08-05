#!/bin/bash
#SBATCH -N 1
#SBATCH -C haswell
#SBATCH -q debug
#SBATCH -J 55_Numpy_Test
#SBATCH --mail-user=conlon@berkeley.edu
#SBATCH --mail-type=ALL
#SBATCH -t 00:30:00

srun -n 1 -c 64 --cpu_bind=cores 4-0_55_test_doublet_making.py
