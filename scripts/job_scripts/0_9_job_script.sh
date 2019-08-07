#!/bin/bash
#SBATCH -C haswell
#SBATCH -q debug
#SBATCH -J 0_9_Parallel_Test
#SBATCH --mail-user=conlon@berkeley.edu
#SBATCH --mail-type=ALL
#SBATCH -t 00:30:00

#run the application:
srun -n 1 -c 64 4-0_9_test_doublet_making.py
