#!/bin/bash
#
#SBATCH -J N1V512
#SBATCH -o N1V512.out
#SBATCH -e N1V512.err
#
#SBATCH --mail-user davide.locatelli@upc.edu
#SBATCH --mail-type FAIL
#
#SBATCH --partition interact
#SBATCH --nodelist node802
#
#SBATCH --mem 16G
#SBATCH --cpus-per-task 2
#SBATCH --gpus 1

source /home/usuaris/locatelli/.bashrc
conda activate egg-env
python3 train.py --checkpoint_dir="N1V512" --checkpoint_freq=5 --batch_size=64 --vocab_size=512 --com_len=1 --lr=0.0001 --n_epochs=10
conda deactivate
