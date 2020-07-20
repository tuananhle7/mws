#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH -t 1-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tuananh@mit.edu
#SBATCH --job-name=gmm
#SBATCH -o ./slurm/%j.out
#SBATCH -e ./slurm/%j.err

cd /om2/user/tuananh/git/insperatum/rrws/gmm-2/

SEED=$1
NUM_PARTICLES=$2
ALGORITHM=$3

source ~/.bashrc
which python
python --version
python run.py --seed $SEED --num-particles $NUM_PARTICLES --memory-size $NUM_PARTICLES --algorithm $ALGORITHM --num-iterations 50000  | tee ./slurm/${SLURM_JOB_ID}_temp.out_err