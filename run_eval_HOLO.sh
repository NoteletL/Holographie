#!/bin/bash

#SBATCH -p gpu
#SBATCH --gres gpu:rtx6000:1
#SBATCH -o test.out
#SBATCH --mail-user=mano.brabant.etu@univ-lemans.fr


#noisyImg=$1
#cleanImg=$2
#runTest=/lium/raid01_c/tahon/holography/checkpoints/run-test2020-04-12_12\:14\:29.082341/
runTest=$1
epoch=$2
D=$3

python main_holo.py --is_training --input_dir $runTest --epoch $epoch --D $D 

