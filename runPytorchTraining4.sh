#!/bin/bash
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --gres gpu:rtx6000:1
#SBATCH --job-name PyTorchDnCNN4
#SBATCH --time 20-00
#SBATCH --mem 80G
#SBATCH -o PytorchDnCNN4.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mano.brabant.etu@univ-lemans.fr

python3 main_holo.py --batch_size 16 --perform_validation --D 4 --num_epoch 10  --noisy_train data1/img_noisy_train_1-2-3-4-5_0-1-1.5-2-2.5_two_50_50_384.npy --clean_train data1/img_clean_train_1-2-3-4-5_0-1-1.5-2-2.5_two_50_50_384.npy


