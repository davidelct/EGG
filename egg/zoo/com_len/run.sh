#!/bin/bash
#
#SBATCH -J N2V25
#SBATCH -o N2V25.out
#SBATCH -e N2V25.err
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
python3 train.py --checkpoint_dir="N2V25" --checkpoint_freq=5 --batch_size=64 --vocab_size=25 --com_len=2 --lr=0.0001 --n_epochs=10
conda deactivate
