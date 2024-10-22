#
# This file is part of DnCnn4Holo.
#
# Adapted from https://github.com/wbhu/DnCNN-tensorflow by Hu Wenbo
#
# DnCnn4Holo is a python script for phase image denoising.
# Home page: https://git-lium.univ-lemans.fr/tahon/dncnn-tensorflow-holography
#
# DnCnn4Holo is free software: you can redistribute it and/or modify
# it under the terms of the GNU LLesser General Public License as
# published by the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.
#
# DnCnn4Holo is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DnCnn4Holo.  If not, see <http://www.gnu.org/licenses/>.

"""
Copyright 2019-2020 Marie Tahon

    :mod:`utils.py` definition of util function for DnCnn4Holo

"""

import numpy as np
from PIL import Image
from scipy.io import savemat



__license__ = "LGPL"
__author__ = "Marie Tahon"
__copyright__ = "Copyright 2019-2020 Marie Tahon"
__maintainer__ = "Marie Tahon"
__email__ = "marie.tahon@univ-lemans.fr"
__status__ = "Production"
#__docformat__ = 'reStructuredText'



def extract_sess_name(lp, ln, pt, stride, ps, np):
    """DEPRECATED
    This method return a sessions name with his given parameters

    Arguments:
        lp (list[int])  : The different patterns used for training
        ln (str)        : The different noises used for training
        pt (str)        : The type of the pahse used
        stride (int)    : The stride of the patches
        ps (int)        : The size of the patches
        np (int)        : The number of patch per image
    """
#example of the call of the function:
#sess_name = extract_sess_name(hparams.train_patterns, hparams.train_noise, hparams.phase_type, hparams.stride, hparams.patch_size, hparams.patch_per_image)
    #return '-'.join(map(str, lp)) + '_' + '-'.join(map(str, ln)) + '_' + pt + '_' + str(stride) + '_' + str(ps) + '_' + str(np)
    return '-'.join(map(str, lp)) + '_' + ln + '_' + pt + '_' + str(stride) + '_' + str(ps) + '_' + str(np)


def wrap_phase(x):
    """
    This method return the given numpy.array wrap between [-pi; pi]

    Arguments:
        x (numpy.array) : The numpy.array to wrap
    """
    return (x + np.pi) % (2 * np.pi) - np.pi


def phase_to_image(data, name):
    """
    This method save a numpy.array in a .tiff format with a given name

    Arguments:
        data (numpy.array)  : The numpy.array to save
        name (str)          : The saving name
    """
    #normalize brute phase between -pi and pi between 0 and 1
    #data = (data - data.min())/ (data.max() - data.min())
    #if not (data.min() >= -np.pi) and (data.max() <= np.pi):
    #    data = np.unwrap(data)

    data = wrap_phase(data)
    data = min_max_norm(data) #resale entre 0 et 1
    np.clip(data, 0, 1) #supprime les valeurs inférieures à 0 et supérieures à 1
    #print(data.min(), data.max())
    data = (data * 255).astype('uint8') #formate les données pour faire une image.
    im = Image.fromarray(data)
    im.save(name, 'tiff')


def min_max_norm(X):
    return (X - X.min())/(X.max() - X.min())

def norm_to_sincos(X):
    return 2* X -1

def norm_to_phase(X):
    #assume normalized value is between -0.5 and 0.5
    #return (2 * np.pi * X ) - np.pi
    return 2 *np.pi * X

def phase_to_norm(X):
    return (X + np.pi) / (2* np.pi)

def sincos_to_norm(X):
    return (X + 1) / 2


def save_images(filepath, ground_truth, noisy_image=np.array([]), clean_image=np.array([])):
    # assert the pixel value range is 0-255
    if not ground_truth.any():
        cat_image = ground_truth
    elif (noisy_image.size == 0) and (clean_image.size == 0):
        cat_image = ground_truth
    else:
        cat_image = np.concatenate([ground_truth, noisy_image, clean_image], axis=1)
    phase_to_image(cat_image, filepath)

def save_MAT_images(filepath, values):
    print(values.shape)
    #save values numpy array into matlab format (in order to perform iterations on predicted images)
    print("original size: ", values.shape)
    mdict = {'NoisyPhase': values}
    savemat(filepath, mdict, appendmat = False)

def rad_to_flat(img):
    return ((np.cos(img) + 1) / 2) * 255

def cal_psnr(im1, im2):
    # assert pixel value range is 0-255 and type is uint8
    mse = ((im1.astype(np.float) - im2.astype(np.float)) ** 2).mean()
    psnr = 10 * np.log10(255 ** 2 / mse)
    return psnr


def cal_std_phase(im1, im2):
    #assert pixel value range is 0-255 and type is uint8
    diff = im1 - im2 #im phase entre -pi et pi
    mse = np.angle(np.exp(1j * diff)) # difference de phase brute entre -2pi et 2pi
    dev = np.std(mse)
    return dev



