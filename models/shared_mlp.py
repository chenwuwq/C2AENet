import time

import torch.nn as nn
import torch
import os
from .utils import *
from torch.nn import Module, Linear, ModuleList
import pytorch3d.ops
from matplotlib import pyplot as plt


def get_knn_idx(x, y, k, offset=0):
    _, knn_idx, _ = pytorch3d.ops.knn_points(x, y, K=k + offset)
    return knn_idx[:, :, offset:]


def knn_group(x: torch.FloatTensor, idx: torch.LongTensor):
    """
    :param  x:      (B, N, F)
    :param  idx:    (B, M, k)
    :return (B, M, k, F)
    """
    B, N, F = tuple(x.size())
    _, M, k = tuple(idx.size())

    x = x.unsqueeze(1).expand(B, M, N, F)
    idx = idx.unsqueeze(3).expand(B, M, k, F)
    # 从完整数据中按索引取值
    return torch.gather(x, dim=2, index=idx)



class GConv(Module):

    def __init__(self, in_channels, out_channels, knn=16, num_fc_layers=3, aggr='max', activation='relu'):
        super().__init__()
        self.in_channels = in_channels
        self.knn = knn

        self.conv1 = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.att_conv = nn.Conv2d(in_channels, 1, kernel_size=1, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.mlp = FCLayer(knn*2, knn*2, bias=True, activation=activation)

    def get_edge_feature(self, x, knn_idx):
        """
        :param  x:          (B, N, d)
        :param  knn_idx:    (B, N, K)
        :return (B, N, K, 2*d)
        """
        knn_feat = knn_group(x, knn_idx)  # (batch_size, num_points, knn_points, channels)
        # 将输入tensor的维度扩展为与括号内指定tensor相同的size
        x_tiled = x.unsqueeze(-2).expand_as(knn_feat)  # tiled for the same data
        edge_feat = knn_feat - x_tiled
        return edge_feat, x_tiled, knn_feat

    def forward(self, input):
        """
        :param  x:  (B, N, d)
        :return (B, N, d+L*c)
        """
        x, origin_x = input
        x = torch.transpose(x, 1, 2)
        knn_idx = get_knn_idx(x, x, k=self.knn, offset=1)  # (B, N, knn_size)
        origin_x = torch.transpose(origin_x, 1, 2)
        origin_knn_idx = get_knn_idx(origin_x, origin_x, k=self.knn, offset=1)  # (B, N, knn_size)

        # First Layer
        edge_feat, x_tiled, knn_feat = self.get_edge_feature(x, knn_idx)  # (B, N, K, c)
        origin_edge_feat, x_tiled, knn_feat2 = self.get_edge_feature(x, origin_knn_idx)  # (B, N, K, c)
        edge_feat = torch.cat((edge_feat, origin_edge_feat), dim=-2)
        x_tiled2 = torch.cat((x_tiled, x_tiled), dim=-2)
        edge_feat = edge_feat.permute(0, 3, 1, 2)
        x_tiled2 = x_tiled2.permute(0, 3, 1, 2)

        point_att = self.sigmoid(self.att_conv(edge_feat))
        edge_feat = point_att * edge_feat + edge_feat

        cat_feature = torch.cat([edge_feat, x_tiled2], dim=1)
        conv_feature = self.conv1(cat_feature)
        conv_feature = self.relu(self.bn1(conv_feature))
        conv_feature = self.conv2(conv_feature)
        conv_feature = self.relu(self.bn2(conv_feature))

        graph_conv_feature = self.mlp(conv_feature)
        out, _ = graph_conv_feature.max(dim=-1, keepdim=False)
        return out


class SharedMLP(nn.Module):
    def __init__(self, in_channels, out_channels, dim=1):
        super().__init__()
        if dim == 1:
            conv = nn.Conv1d
            bn = nn.BatchNorm1d
        elif dim == 2:
            conv = nn.Conv2d
            bn = nn.BatchNorm2d
        else:
            raise ValueError
        if not isinstance(out_channels, (list, tuple)):
            out_channels = [out_channels]
        layers = []
        for oc in out_channels:
            layers.extend([
                GConv(in_channels, oc),
            ])
            in_channels = oc
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs, origin_input):
        return self.layers((inputs, origin_input))

