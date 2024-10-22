#!/bin/bash

#SBATCH -p gpu
#SBATCH --gres gpu:rtx6000:1

#noisyImg=$1
#cleanImg=$2
#runTest=/lium/raid01_c/tahon/holography/checkpoints/run-test2020-04-12_12\:14\:29.082341/
runTest=$1
epoch=$2
D=$3
nbIteration=$4

#test1=/info/etu/m1/s171085/Projets/Portage-Keras-PyTorch/Portage-reseau-de-neurones-de-Keras-vers-PyTorch/dncnn-tensorflow-holography-master/Holography/DATAEVAL/DATAEVAL/DATA_1_Phase_Type1_2_0.25_1.5_4_50.mat
#test2=/info/etu/m1/s171085/Projets/Portage-Keras-PyTorch/Portage-reseau-de-neurones-de-Keras-vers-PyTorch/dncnn-tensorflow-holography-master/Holography/DATAEVAL/DATAEVAL/DATA_20_Phase_Type4_2_0.25_2.5_4_100.mat
#test3=/info/etu/m1/s171085/Projets/Portage-Keras-PyTorch/Portage-reseau-de-neurones-de-Keras-vers-PyTorch/dncnn-tensorflow-holography-master/Holography/DATAEVAL/DATAEVAL/VibPhaseDATA.mat
#keyNoisy='Phaseb'
#keyClean='Phase'


python main_holo.py --test_mode --input_dir $runTest --epoch $epoch --D $D --nb_iteration $nbIteration
