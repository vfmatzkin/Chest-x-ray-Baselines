import os

import SimpleITK as sitk
import torch
import vedo
from ctunet.pytorch.transforms import random_blank_patch
from torch.utils.data import Dataset

from .utils import check_files_exist


class MeshHeadsDataset(Dataset):
    def __init__(self, images, img_path, label_path=None, transform=None,
                 test=False):
        self.images = images
        self.img_path = img_path
        self.label_path = label_path if label_path else img_path
        self.transform = transform
        self.test = test

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        idx = idx.tolist() if torch.is_tensor(idx) else idx
        img_name = self.images[idx]
        img_path = os.path.abspath(os.path.join(self.img_path, img_name))

        # TODO Don't hardcode this
        lmk_pth = os.path.abspath(
            img_path.replace('.nii.gz',
                             '_decimated_1perc_dfm.vtk')
        )

        check_files_exist([img_path, lmk_pth], False, not self.test)
        
        image = torch.tensor(
            sitk.GetArrayFromImage(sitk.ReadImage(img_path)),
            dtype=torch.float
        ).unsqueeze(0)
        
        lmk = vedo.Mesh(lmk_pth).points() if os.path.exists(lmk_pth) else None
            
        sample = {
            'image': image,
            'landmarks': lmk
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        return {'image': torch.from_numpy(sample['image']).float()}


class SkullRandomHole(object):
    """ Simulate craniectomies placing random binary shapes.

    Given a batch of 3D images (PyTorch tensors), crop a random cube or box
    placed in a random position of the image with the sizes given in d.

    :param img: Input image.
    :param p: probability of adding the noise (by default flip a coin).
    :param return_flap: Return extracted bone flap.
    """

    def __init__(self, p=1):
        self.p = p

    def __call__(self, sample):
        img = sample['image']
        if not type(img) == torch.Tensor:
            raise TypeError(f"Expected 'torch.Tensor'. Got {type(img)}.")
        is_batch = len(img.shape) == 4
        batch_size = img.shape[0] if is_batch else 1

        brk_sk = torch.FloatTensor(
            (img if is_batch else img.unsqueeze(0))
        )  # Broken skull

        for i in range(batch_size):  # Apply the transform for each img
            np_img = brk_sk[i].numpy()
            brk_sk[i] = torch.FloatTensor(random_blank_patch(np_img, self.p))

        sample['image'] = brk_sk[0] if not is_batch else brk_sk

        return sample
