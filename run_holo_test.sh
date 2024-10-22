#!/bin/bash

#SBATCH -p gpu
#SBATCH --gres gpu:rtx6000:1
#SBATCH -o test.out

#noisyImg=$1
#cleanImg=$2
#runTest=/lium/raid01_c/tahon/holography/checkpoints/run-test2020-04-12_12\:14\:29.082341/
runTest=$1
epoch=$2
D=$3

for num in 1 2 3 4 5; do
    for lambda in 0 1 1p5 2 2p5; do
        noisyImg=/info/etu/m1/s171085/Projets/Portage-Keras-PyTorch/Portage-reseau-de-neurones-de-Keras-vers-PyTorch/dncnn-tensorflow-holography-master/Holography/DATABASE/PATTERN$num/MFH_$lambda/NoisyPhase.mat
        #noisyImg=/lium/raid01_c/tahon/holography/HOLODEEPmat/PATTERN$num/MFH_$lambda/run-test2020-04-12_12\:14\:29.082341/run-test2020-04-12_12\:14\:29.082341/NoisyPhase.mat-27000.mat-27000.mat
        cleanImg=/info/etu/m1/s171085/Projets/Portage-Keras-PyTorch/Portage-reseau-de-neurones-de-Keras-vers-PyTorch/dncnn-tensorflow-holography-master/Holography/DATABASE//PATTERN$num/PhaseDATA.mat
        echo $noisyImg 
        python main_holo.py --is_training --test_noisy_img $noisyImg --test_noisy_key 'NoisyPhase' --test_clean_img $cleanImg --test_clean_key 'Phase' --input_dir $runTest --epoch $epoch --D $D
    done
done

test1=/info/etu/m1/s171085/Projets/Portage-Keras-PyTorch/Portage-reseau-de-neurones-de-Keras-vers-PyTorch/dncnn-tensorflow-holography-master/Holography/DATAEVAL/DATAEVAL/DATA_1_Phase_Type1_2_0.25_1.5_4_50.mat
test2=/info/etu/m1/s171085/Projets/Portage-Keras-PyTorch/Portage-reseau-de-neurones-de-Keras-vers-PyTorch/dncnn-tensorflow-holography-master/Holography/DATAEVAL/DATAEVAL/DATA_20_Phase_Type4_2_0.25_2.5_4_100.mat
test3=/info/etu/m1/s171085/Projets/Portage-Keras-PyTorch/Portage-reseau-de-neurones-de-Keras-vers-PyTorch/dncnn-tensorflow-holography-master/Holography/DATAEVAL/DATAEVAL/VibPhaseDATA.mat
keyNoisy='Phaseb'
keyClean='Phase'
echo $test1

python main_holo.py --is_training --test_noisy_img $test1 --test_noisy_key $keyNoisy --test_clean_img $test1 --test_clean_key $keyClean --input_dir $runTest --epoch $epoch --D $D 

python main_holo.py --is_training --test_noisy_img $test2 --test_noisy_key $keyNoisy --test_clean_img $test2 --test_clean_key $keyClean --input_dir $runTest --epoch $epoch --D $D 

python main_holo.py --is_training --test_noisy_img $test3 --test_noisy_key $keyNoisy --test_clean_img $test3 --test_clean_key $keyClean --input_dir $runTest --epoch $epoch --D $D 
