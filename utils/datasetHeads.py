import os

import torch
import vedo
from torch.utils.data import Dataset


class MeshHeadsDataset(Dataset):
    def __init__(self, images, images_folder, transform=None):
        self.images = images
        self.img_path = images_folder
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        idx = idx.tolist() if torch.is_tensor(idx) else idx
        img_name = self.images[idx]
        img_path = os.path.join(self.img_path, img_name)

        image = vedo.Mesh(img_path)

        sample = {'image': image}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        return {'image': torch.from_numpy(sample['image']).float()}
