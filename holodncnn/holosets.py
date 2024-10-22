import os
import torch
import torch.utils.data as td
import pandas as pd
from scipy.io import loadmat
import torchvision.transforms as transforms
from torchvision.io import read_image
import numpy as np
from PIL import Image


def extract_patches_from_images(original_image, patch_size, patch_stride, patch_step, nb_patch_per_image, idx):
    original_size = original_image.size(dim=1)
    nb_total_patch = int((original_size - patch_size) / patch_stride + 1) * int(
        (original_size - patch_size) / patch_stride + 1)
    # (1024 - 50)/50 + 1 = 20 -> 20*20 = 400 patch per img => we assume here that original images are squared and patches also.
    patches = torch.zeros(nb_total_patch, patch_size, patch_size)
    cpt = 0
    for x in range(0 + patch_step, original_size - patch_size, patch_stride):
        for y in range(0 + patch_step, original_size - patch_size, patch_stride):
            patches[cpt, :, :] = original_image[:, x:x + patch_size, y:y + patch_size]
            cpt += 1
    # shuffle the different patches of an image similarly on noisy and clean images
    perm_idx = np.random.RandomState(seed=23).permutation(nb_total_patch)[:nb_patch_per_image]
    return patches[perm_idx[idx], :, :].unsqueeze(0)


def data_augmentation(patch, transformation):
    """Phase augmentation and convert to cosinus or sinus or both values.
    Arguments:
        patches: torch tensor of size (nb_patch_per_image, patch_size, patch_size) for a single image
        transformation: transformation from the possible transformations (phase and rotations) applied to the given patch
    """
    # phase augmentation and conversion to cos /sin values
    if (transformation is not None) and (transformation != ''):
        if 'add45' in transformation:
            #patches = torch.cat((patches, patches + torch.pi / 4), 0)
            patch = patch + torch.pi/4
        if 'transpose' in transformation:
            #patches = torch.cat((patches, torch.transpose(patches, 1, 2)), 0)
            patch = torch.transpose(patch, 1, 2)

        nb_patch = patch.size(1) // 2
        patch = torch.cat((torch.cos(patch[:nb_patch, :, :]), torch.sin(patch[nb_patch:, :, :])), 0)
        # image rotations
        if 'flip' in transformation:
            # patches = torch.cat((patches, torch.flipud(patches)))
            patch = torch.flipud(patch)
        if 'rot90' in transformation:
            # patches = torch.cat((patches, torch.rot90(patches, 1, [1, 2])))
            patch = torch.rot90(patch, 1, [1, 2])
        if 'rot180' in transformation:
            # patches = torch.cat((patches, torch.rot90(patches, 2, [1, 2])))
            patch = torch.rot90(patch, 2, [1, 2])
        if 'rot270' in transformation:
            # patch = torch.cat((patch, torch.rot90(patch, 3, [1, 2])))
            patch = torch.rot90(patch, 3, [1, 2])
        # print("[*] augmentation process is done with", transformation)
    else:
        nb_patch = patch.size(1) // 2
        patch = torch.cat((torch.cos(patch[:nb_patch, :, :]), torch.sin(patch[nb_patch:, :, :])), 0)
        # print("[*] no augmentation process")

    return patch



class TrainHoloset(td.Dataset):
    """ This class allow us to load and use the data needed for model training
        It loads and normalize holographic phase images.
        The final images are B&W images with float values between -pi and pi
    """

    def __init__(self, img_dir, img_files, extension, key_clean, key_noisy, augmentation, nb_patch_per_image,
                 patch_size, patch_stride, patch_step):
        self.img_dir = img_dir
        self.img_files = pd.read_csv(img_files)
        self.extension = extension
        self.key_clean = key_clean
        self.key_noisy = key_noisy
        self.augmentation = (augmentation + ',').split(',')
        self.nb_patch_per_image = nb_patch_per_image
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.patch_step = patch_step

    def __len__(self):
        return len(self.img_files) * self.nb_patch_per_image * self.getAugmentationNb()

    def getTrainingName(self):
        return self.img_dir

    def getAugmentationNb(self):
        """
        nb_augmentations = 1
        t = self.augmentation
        if (t is not None) and (t != ''):
            if 'add45' in t:
                nb_augmentations *= 2
            if 'transpose' in t:
                nb_augmentations *= 2

            # image rotations
            if 'flip' in t:
                nb_augmentations *= 2
            if 'rot90' in t:
                nb_augmentations *= 2
            if 'rot180' in t:
                nb_augmentations *= 2
            if 'rot270' in t:
                nb_augmentations *= 2
            """
        # nb_augmentations = 2 ** len(self.augmentation)
        # print("[*] augmentation process is done with", nb_augmentations, "augmentations")
        return len(self.augmentation)

    def __getitem__(self, item):
        # item = idx_img * nb_patch-per_image * nb_augmentation +
        #        idx_aug * nb_augmentation +
        #        idx_patch
        #for idx in range(len(img_files))
        naug = self.getAugmentationNb()
        a = (item // self.nb_patch_per_image)
        idx_patch = item - a * self.nb_patch_per_image  # select the patch
        idx_img = a // naug
        idx_aug = a - idx_img * naug
        # print(item, idx_img, idx_aug, idx_patch)

        if self.extension == 'mat':
            transform = transforms.Compose([transforms.ToTensor()])
            img_clean = loadmat(os.path.join(self.img_dir, self.img_files.iloc[idx_img, 0]))[self.key_clean]
            img_noisy = loadmat(os.path.join(self.img_dir, self.img_files.iloc[idx_img, 1]))[self.key_noisy]
            img_clean = transform(img_clean)
            img_noisy = transform(img_noisy)

        elif (self.extension == 'tif') | (self.extension == 'tiff'):
            transform = transforms.Compose([transforms.PILToTensor()])
            img_clean = Image.open(os.path.join(self.img_dir, self.img_files.iloc[idx_img, 0])).convert('L')
            img_noisy = Image.open(os.path.join(self.img_dir, self.img_files.iloc[idx_img, 1])).convert('L')
            img_clean = (transform(img_clean).to(torch.float32) * torch.pi / 128.0) - torch.pi
            img_noisy = (transform(img_noisy).to(torch.float32) * torch.pi / 128.0) - torch.pi

        else:
            img_clean = read_image(os.path.join(self.img_dir, self.img_files.iloc[idx_img, 0]))
            img_noisy = read_image(os.path.join(self.img_dir, self.img_files.iloc[idx_img, 1]))
            img_clean = (img_clean.to(torch.float32) * torch.pi / 128.0) - torch.pi
            img_noisy = (img_noisy.to(torch.float32) * torch.pi / 128.0) - torch.pi
            print('unknown extension => TODO')

        assert (img_clean.size() == img_noisy.size()), "training clean and noise have not same sizes"

        assert (img_clean.size(dim=1) == img_clean.size(dim=2)), "training images are not squared"

        clean_patch = extract_patches_from_images(img_clean, self.patch_size, self.patch_stride, self.patch_step,
                                                    self.nb_patch_per_image,
                                                    idx_patch)
        noisy_patch = extract_patches_from_images(img_noisy, self.patch_size, self.patch_stride, self.patch_step,
                                                    self.nb_patch_per_image,
                                                    idx_patch)

        # print(clean_patch.size())

        clean_patch = data_augmentation(clean_patch, self.augmentation[idx_aug])
        noisy_patch = data_augmentation(noisy_patch, self.augmentation[idx_aug])

        # noise_simu = self.img_files.iloc[item, 2]

        # print(clean_patches.size())
        # print(self.augmentation)

        return noisy_patch, clean_patch


class EvalHoloset(td.Dataset):
    """ This class allow us to load and use the data needed for model training
        It loads and normalize holographic phase images.
        The final images are B&W images with float values between -pi and pi
    """

    def __init__(self, img_dir, img_files, extension, key_clean, key_noisy):
        self.img_dir = img_dir
        self.img_files = pd.read_csv(img_files)
        self.extension = extension
        self.key_clean = key_clean
        self.key_noisy = key_noisy
        # check if the clean reference is given in the csv file
        if len(self.img_files.clean.value_counts()) > 0:
            self.ref = True
        else:
            self.ref = False

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, item):
        clean_file = self.img_files.iloc[item, 0]
        noisy_file = self.img_files.iloc[item, 1]
        if not self.ref:
            img_clean = None

        if self.extension == 'mat':
            transform = transforms.Compose([transforms.ToTensor()])
            img_noisy = loadmat(os.path.join(self.img_dir, noisy_file))[self.key_noisy]
            img_noisy = transform(img_noisy)
            if self.ref:
                img_clean = loadmat(os.path.join(self.img_dir, clean_file))[self.key_clean]
                img_clean = transform(img_clean)

        elif (self.extension == 'tif') | (self.extension == 'tiff'):
            transform = transforms.Compose([transforms.PILToTensor()])
            img_noisy = Image.open(os.path.join(self.img_dir, noisy_file)).convert('L')
            img_noisy = (transform(img_noisy).to(torch.float32) * torch.pi / 128.0) - torch.pi
            if self.ref:
                img_clean = Image.open(os.path.join(self.img_dir, clean_file)).convert('L')
                img_clean = (transform(img_clean).to(torch.float32) * torch.pi / 128.0) - torch.pi

        else:
            img_noisy = read_image(os.path.join(self.img_dir, noisy_file))
            img_noisy = (img_noisy.to(torch.float32) * torch.pi / 128.0) - torch.pi
            if self.ref:
                img_clean = read_image(os.path.join(self.img_dir, clean_file))
                img_clean = (img_clean.to(torch.float32) * torch.pi / 128.0) - torch.pi
            print('unknown extension => TODO')

        if self.ref:
            assert (img_noisy.size() == img_clean.size()), "eval clean and noise have not same sizes"

        assert (img_noisy.size(dim=1) == img_noisy.size(dim=2)), "eval images are not squared"

        #img_noisy_cos = torch.cos(img_noisy)
        #img_noisy_sin = torch.sin(img_noisy)

        return img_noisy, img_clean