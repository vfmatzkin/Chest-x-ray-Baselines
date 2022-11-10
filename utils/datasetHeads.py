import os

import SimpleITK as sitk
import numpy as np
import torch
import vedo
from torch.utils.data import Dataset
from ctunet.pytorch.transforms import random_blank_patch


def check_files_exist(files):
    if type(files) is not list:
        files = [files]
    for f in files:
        if not os.path.exists(f):
            raise FileNotFoundError(f)


class MeshHeadsDataset(Dataset):
    # self, images, img_path, label_path, transform=None, heart = False
    def __init__(self, images, img_path, label_path=None, transform=None,
                 heart=False):
        self.images = images
        self.img_path = img_path
        self.label_path = label_path if label_path else img_path
        self.heart = heart
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        idx = idx.tolist() if torch.is_tensor(idx) else idx
        img_name = self.images[idx]
        img_path = os.path.abspath(os.path.join(self.img_path, img_name))

        # TODO Don't hardcode this
        landmarks_path = os.path.abspath(
            img_path.replace('.nii.gz',
                             '_decimated_1perc_dfm.vtk')
        )

        check_files_exist([img_path, landmarks_path])

        image = torch.tensor(
            sitk.GetArrayFromImage(sitk.ReadImage(img_path)),
            dtype=torch.float
        ).unsqueeze(0)
        landmarks = vedo.Mesh(landmarks_path).points()

        sample = {
            'image': image,
            'landmarks': landmarks
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        return {'image': torch.from_numpy(sample['image']).float()}

# TODO Copied this from ctunet.transforms. Adapt to this project samples.
class SkullRandomHole(object):
    """ Simulate craniectomies placing random binary shapes.

    Given a batch of 3D images (PyTorch tensors), crop a random cube or box
    placed in a random position of the image with the sizes given in d.

    :param img: Input image.
    :param p: probability of adding the noise (by default flip a coin).
    :param return_flap: Return extracted bone flap.
    """

    def __init__(self, p=1, double_output=False):
        self.p = p
        self.double_output = double_output

    def __call__(self, sample):
        img = sample['image']
        if not type(img) == torch.Tensor:
            raise TypeError(f"Expected 'torch.Tensor'. Got {type(img)}.")
        is_batch = len(img.shape) == 4
        batch_size = img.shape[0] if is_batch else 1
        brk_sk = np.copy((img if is_batch else img.unsqueeze(0))).astype(
            np.uint8)  # Broken skull
        flap = np.copy((img if is_batch else img.unsqueeze(0))).astype(
            np.uint8)  # Initialize flap
        if self.double_output:
            full_sk = torch.ByteTensor(
                np.copy((img if is_batch else img.unsqueeze(0))).astype(
                    np.uint8))  # Save full skull
        for i in range(batch_size):  # Apply the transform for each img
            np_img = brk_sk[i, :, :, :]
            brk_sk[i], flap[i] = random_blank_patch(np_img, self.p, True)
        brk_sk = torch.ByteTensor(brk_sk)
        if not is_batch:
            brk_sk = brk_sk[0]
            flap = flap[0]
        flap = torch.ByteTensor(flap)

        if self.double_output:
            sample = {'image': brk_sk, 'target': (full_sk, flap)}
        else:
            sample = {'image': brk_sk, 'target': flap}
        return sample
