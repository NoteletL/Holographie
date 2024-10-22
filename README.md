# DnCNN-PyTorch
A PyTorch implement of the TIP2017 paper [Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising](http://www4.comp.polyu.edu.hk/~cslzhang/paper/DnCNN.pdf)
This project aims at automatically denoising phase images using a CNN residual filter.


## Prerequisites TODO
This project needs the following python3 packages
```
conda install -c pytorch pytorch 
conda install pillow
conda install matplotlib
conda install -c pytorch torchvision
pip install argparse yaml time scipy pandas
```
Python == 3.9
PyTorch == 1.7.1

## Parameters
Modifiable parameters are located in the config file under YAML format.
An example is given in dataset.yaml


## Localization of data and generation of patches
Data is either .tiff images or .mat MATLAB matrices. 
While tiff matrices contains discrete pixels values, matlab matrices contains directly the phase image.
Four databases are available for the moment:
* HOLODEEPmat: MATLAB images for training and development purposes: 5 patterns and 5 noise level, 1024x1024
* DATAEVAL: 3 MATLAB images for evaluation purposes: Data1, Data20 and VibMap.
* DB99: 128 MATLAB images , 1024x1024
* NATURAL: 400 images in B&W for image denoising, noisy images are obtained with additive Gaussian noise: 180x180

The train/dev/test datasets are explicited in a csv file containing the following header and the list of corresponding image lcoations
```
clean, noisy, Ns
NOISEFREE.tiff,NOISY.tiff,0
```
The Ns value corresponds to the speckle grain size used to generate the noisy image from the clean image.
Images from HOLODEEP are referred by their name according to their pattern number (argument.train_patterns from 1 to 5), their noise level (hparams.train_noises 0, 1, 1.5, 2 or 2.5). Images for train and developpement (or test) and final evaluation (eval) are given by "train", "test" or "eval" suffix.

All phase data is converted using sine or cosine functions and normalized between 0 and 1 for being in agreement with the output of the network. A propotional coefficient is applied on the predicted image to rescale the phase amplitude between -pi and pi.

Patch size is given as an argument to the DataLoader directly in the config file.

## Manual
In order to train models you have to manualy download an holography database and give the path to that directory for training database (args.train_dir) evaluating database (args.eval_dir) and testing database (args.test_dir).

A database can be found on skinner /info/etu/m1/s171085/Projets/Portage-Keras-PyTorch/Portage-reseau-de-neurones-de-Keras-vers-PyTorch/dncnn-tensorflow-holography-master/Holography or here https://lium-cloud.univ-lemans.fr/index.php/s/8gLGetFsFcGQgMo

The application is used with the main_holo.py script with different arguments from the argument.py script.
To start a training with default parameters you can use the command.
```
#launch a training
python main_holo.py --config dataset.yaml 
```

You can precise how training data is pre-processed in the DataLoader.
```
# Training set
train:
  path: "../XP_holography/CORPUS/DATABASE/" # folder where the training images are located according to the csv file
  csv: "holodeep_mat.csv"         # list of the training images (clean, noisy, Ns)
  extension: "mat"                # format of the training images (matlab or tiff)
  matlab_key_noisy: "NoisyPhase"  # key value for noisy images in case of matlab files. In case of tiff set to null
  matlab_key_clean: "Phase"       # key value for noisy images in case of matlab files. In case of tiff set to null

  patch:                    # parameters for patch extraction
    nb_patch_per_image: 2   # number of patches per image
    size: 50                # size of the squared patches
    stride: 50              # stride between two consecutive patches
    step: 0                 # step at the begining and end of each image

  augmentation: "add45,add45transpose,cossin,flip,rot90"
                            # data augmentation function applied on all patches.
```
The possible data augmentation functions are the followings:
* *add45*: adds an angle of pi/4 to the current phase
* *transpose*: transposes the patch in the phase domain
* *cossin*: compute the cosine and sine transforms of the phase values. If no, compute the cosine or sine transforms
* *flip*: flip up down the cosine or sine image
* *rot90* (90, 180, 270): rotate the cosine or sine image of 90 degrees (or 180, or 270)
* *add45transpose*: combines both augmentation functions


You can precise in the config file the different hyperparameters for the training:
```
# training parameters
model:
  batch_size: 64    # number of patches in each batch
  num_epochs: 2     # number of epoch the model will train
  D: 4              # number ores block 
  C: 64             # kernel size for convolutional layer (not tested)
  lr: 0.001         # learning rate
  input_dir:  "./PyTorchCheckpoint/" 
                    # folder where are saved the models during training
  output_dir: null  # directory of saved checkpoints for denoising operation or retraining
  start_epoch: null #'epoch\'s number from which we going to retrain' if > 0 a model has to be saved in the input_dir
  freq_save: 1      # number of epochs needed before saving a model
  perform_validation: True 
                    # perform validation on the validation set at each freq_save epoch.
  graph: True       # plot the loss function according to epochs. 
```



The arguments input_dir and epoch are used for re-training and de-noising operation.
In input_dir give the path to the model you want to use, and in epoch give the number from which you want to re-train or do a de-noising operation.
The new model will be saved in the input_dir directory.
```
#re launch a training strating from the model experiment_xxx at the epoch 130 
model:
  input_dir:  "./PyTorchExperiments/experiment_xxx/"
  output_dir: null # directory of saved checkpoints for denoising operation or retraining
  start_epoch: 130 #'epoch\'s number from which we going to retrain' if > 0 a model has to be saved in the input_dir
```

To do a de-noising operation you have to use the test_mode argument.
You can use the argument test_noisy_img, test_noisy_key, test_clean_img and test_clean_key to precise which image you want to de-noise
The clean image will only be used to give the level of remaining noise after the de-noising operation. If you don't have a clean reference just give the noisy reference again.
```
python main_holo.py --config dataset.yaml --test_mode

#launch a denoising operation on the image DATA_1_Phase_Type1_2_0.25_1.5_4_50.mat with 3 iterations
#with the model experiment_xxx at the epoch 130
model:
  input_dir:  "./PyTorchExperiments/experiment_xxx/"
  output_dir: null # directory of saved checkpoints for denoising operation or retraining
  start_epoch: 130 #'epoch\'s number from which we going to retrain' if > 0 a model has to be saved in the input_dir
# Test set
test:
  path: "../XP_holography/"
  csv: "holodeep_test.csv"
  extension: "tiff"
  matlab_key_noisy: "NoisyPhase"
  matlab_key_clean: "Phase"
  nb_iteration: 3
  save_test_dir: "./TestImages/" # directory of saved test images after denoising operation

```
The results of those de-noising operation can be found in a TestImages directory

## Results
Results obtained with different configurations. Values are given in terms of Phase STD
Durations are given in Day-Hours:Min:Sec with one GPU.
```
                                                                                                    | PATTERNS 1-5  || TEST DATA
Epochs  | lr    | train patterns    | train noise       | nb batches    | layers    |  duration     | MOY           || DATA1 | DATA20 | VibMap ||
3       | 0.001 | [1-2-3-4-5]       | [0,1,1.5,2,2.5]   | 384           | 16        | 00-00:27:47   | 0.036         || 0.065 | 0.606  | 0.131  ||
100     | 0.001 | [1-2-3-4-5]       | [0-1-1.5-2-2.5]   | 384           | 16        | 00-15:25:35   | 0.029         || 0.054 | 0.599  | 0.139  ||
```
