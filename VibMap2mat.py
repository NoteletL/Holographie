import numpy as np
from scipy.io import loadmat, savemat
from PIL import Image


vibmap = loadmat('/lium/raid01_c/tahon/holography/DATAEVAL/VibPhaseDATA.mat')
mask = loadmat('/lium/raid01_c/tahon/holography/DATAEVAL/Mask1024_Rect.mat')

new_vibmap = {}
for key in ['Phase', 'PhiCal', 'Phaseb', 'BRUIT']:
    new_vibmap[key] = np.array(vibmap[key]) * np.array(mask['Mask'])
savemat('/lium/raid01_c/tahon/holography/DATAEVAL/VibPhaseDATA_masked.mat', new_vibmap, appendmat = False)

#check if savemat is correct
#nv = loadmat('/lium/raid01_c/tahon/holography/DATAEVAL/VibPhaseDATA_masked.mat')
#print(nv.keys())
#print(np.array(nv['Phase']).shape)
