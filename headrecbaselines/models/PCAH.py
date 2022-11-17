""" Convolutional PCA encoder for 3D skull reconstruction. """

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f


class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock3D, self).__init__()

        self.skip = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels, track_running_stats=False))
        else:
            self.skip = None

        self.block = nn.Sequential(
            nn.BatchNorm3d(in_channels, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1)
        )

    def forward(self, x):
        identity = x
        out = self.block(x)

        if self.skip is not None:
            identity = self.skip(x)

        out += identity
        out = f.relu(out)

        return out


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()

        self.interpolate = config['interpolate']
        self.h = config['h']
        self.w = config['w']
        self.z = config['latents']
        self.slices = config['slices']

        self.residual1 = ResidualBlock3D(in_channels=1, out_channels=8)
        self.residual2 = ResidualBlock3D(in_channels=8, out_channels=16)
        self.residual3 = ResidualBlock3D(in_channels=16, out_channels=32)
        self.residual4 = ResidualBlock3D(in_channels=32, out_channels=64)
        self.residual5 = ResidualBlock3D(in_channels=64, out_channels=128)
        self.residual6 = ResidualBlock3D(in_channels=128, out_channels=128)

        # Input shape is slices x h x w
        self.maxpool = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))

        h2 = self.h // 32
        w2 = self.w // 32
        s2 = self.slices // 32

        self.mu = nn.Linear(128 * h2 * w2 * s2, self.z)

    def forward(self, x):
        if self.interpolate:
            x = f.interpolate(x, size=(self.slices, self.h, self.w))
        x = self.residual1(x)
        x = self.maxpool(x)
        x = self.residual2(x)
        x = self.maxpool(x)
        l3 = self.residual3(x)
        x = self.maxpool(l3)
        l4 = self.residual4(x)
        x = self.maxpool(l4)
        l5 = self.residual5(x)
        x = self.maxpool(l5)
        l6 = self.residual6(x)

        x = x.view(l6.size(0), -1)

        mu = self.mu(x)

        return mu


class DecoderPCA(nn.Module):
    def __init__(self, config):
        super(DecoderPCA, self).__init__()

        device = config['device']

        matrix = np.load('trained/heads_pca_components.npy')
        mean = np.load('trained/heads_pca_mean.npy')

        self.matrix = torch.from_numpy(matrix).float().to(device)
        self.mean = torch.from_numpy(mean).float().to(device)

    def forward(self, x):
        return torch.matmul(x, self.matrix) + self.mean


class PCAH_Net(nn.Module):
    def __init__(self, config):
        super(PCAH_Net, self).__init__()

        self.encoder = Encoder(config)
        self.decoder = DecoderPCA(config)

    def forward(self, x):
        x = self.encoder(x)  # returns mu
        x = self.decoder(x)
        return x
