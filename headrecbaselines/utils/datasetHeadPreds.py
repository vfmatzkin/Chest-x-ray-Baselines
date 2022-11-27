""" Predictions dataset

It could contain mesh predictions or binarized mesh predictions (in case it's
provided both the path and suffix of the binarized mesh predictions).

Is expected that both the predictions and the Ground Truth have the same suffix

"""

import os

import SimpleITK as sitk
import torch
import vedo
from torch.utils.data import Dataset


class HeadPredsDataset(Dataset):
    def __init__(self, meshes, dir_path, gt_path=None, gt_suffix=None,
                 bin_preds_path=None, bin_preds_suffix=None):
        self.meshes = meshes  # .stl filenames
        self.dir_path = dir_path  # path to the images
        self.gt_path = gt_path if gt_path else None
        self.gt_suffix = gt_suffix if gt_suffix else None
        self.bin_preds_path = bin_preds_path if bin_preds_path else None
        self.bin_preds_suffix = bin_preds_suffix if bin_preds_suffix else None

    def __len__(self):
        return len(self.meshes)

    def __getitem__(self, idx):
        idx = idx.tolist() if torch.is_tensor(idx) else idx
        mesh_fname = self.meshes[idx]
        msh_pth = os.path.abspath(os.path.join(self.dir_path, mesh_fname))
        bin_fname = mesh_fname.replace('.stl', self.bin_preds_suffix) \
            if self.bin_preds_suffix else None
        bin_pth = os.path.join(self.bin_preds_path, bin_fname) \
            if self.bin_preds_path else None

        gt_fname = mesh_fname.replace('.stl', self.gt_suffix) \
            if self.gt_suffix else mesh_fname
        gt_pth = os.path.join(self.gt_path, gt_fname) \
            if self.gt_path else None

        msh = vedo.Mesh(msh_pth).points() if os.path.exists(msh_pth) else None
        bin = torch.tensor(
            sitk.GetArrayFromImage(sitk.ReadImage(bin_pth)),
            dtype=torch.int
        ) if bin_pth and os.path.exists(bin_pth) else None
        gt = vedo.Mesh(gt_pth).points() if gt_pth else None
        gt_bin_fname = os.path.join(self.gt_path, 
                                    mesh_fname.replace('.stl', '.nii.gz'))
        gt_bin = torch.tensor(
            sitk.GetArrayFromImage(sitk.ReadImage(gt_bin_fname)),
            dtype=torch.int
        ) if gt_bin_fname and os.path.exists(gt_bin_fname) else None

        sample = {
            'mesh': msh,
            'bin': bin,
            'gt': gt,
            'gt_bin': gt_bin
        }

        return sample
