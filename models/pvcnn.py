import functools

import torch
import torch.nn as nn
from .shared_mlp import SharedMLP


class PVConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, resolution, gvconv, with_se=False, normalize=True, eps=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.resolution = resolution
        self.gvconv = gvconv

        self.point_features = SharedMLP(in_channels, out_channels)

    def forward(self, inputs, origin_input):
        features = inputs

        point_features = self.point_features(features, origin_input)
        fused_features = point_features
        return fused_features


def create_pointnet_components(blocks, in_channels, gvconv, with_se=False, normalize=True, eps=0, width_multiplier=1,
                               voxel_resolution_multiplier=1):

    r, vr = width_multiplier, voxel_resolution_multiplier

    layers, concat_channels = [], 0
    for out_channels, num_blocks, voxel_resolution in blocks:
        out_channels = int(r * out_channels)
        block = functools.partial(PVConv, kernel_size=3, resolution=int(vr * voxel_resolution),
                                  with_se=with_se, normalize=normalize, eps=eps, gvconv=gvconv)
        for _ in range(num_blocks):
            layers.append(block(in_channels, out_channels))
            in_channels = out_channels
            concat_channels += out_channels
    return layers, in_channels, concat_channels
