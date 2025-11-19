import torch
import torch.nn as nn
from torch.nn import Module, Linear, ModuleList
import pytorch3d.ops
import torch.nn.functional as F
from .utils import *
from .pvcnn import create_pointnet_components


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


class cscc_layer(Module):
    def __init__(self, out_channel):
        super().__init__()
        self.fc = FCLayer(out_channel, out_channel, activation='relu')

    def forward(self, x):
        x = self.fc(x)
        return x


class C2AENet(Module):
    # stage 4
    def __init__(self, in_channels=3, width_multiplier=1, voxel_resolution_multiplier=1, dynamic_graph=True,
                 conv_channels=24, num_convs=4, conv_num_fc_layers=3,
                 conv_growth_rate=12, conv_knn=16, conv_aggr='max', activation='relu'):
        super().__init__()
        # out_channels, num_blocks, voxel_resolution
        self.blocks = ((64, 1, 16), (128, 1, 8), (256, 1, 8))
        self.in_channels = in_channels

        layers, channels_point, concat_channels_point = create_pointnet_components(
            blocks=self.blocks, in_channels=self.in_channels, with_se=True, normalize=False,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier,
            gvconv="graph"
        )
        layers2, channels_point, concat_channels_point = create_pointnet_components(
            blocks=self.blocks, in_channels=self.in_channels, with_se=True, normalize=False,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier,
            gvconv="graph"
        )
        layers3, channels_point, concat_channels_point = create_pointnet_components(
            blocks=self.blocks, in_channels=self.in_channels, with_se=True, normalize=False,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier,
            gvconv="graph"
        )
        layers4, channels_point, concat_channels_point = create_pointnet_components(
            blocks=self.blocks, in_channels=self.in_channels, with_se=True, normalize=False,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier,
            gvconv="graph"
        )
        self.point_features = nn.ModuleList(layers)
        self.point_features2 = nn.ModuleList(layers2)
        self.point_features3 = nn.ModuleList(layers3)
        self.point_features4 = nn.ModuleList(layers4)

        self.CSCC_E1 = nn.ModuleList([cscc_layer(out_channel=64), cscc_layer(out_channel=128),
                                      cscc_layer(out_channel=256), cscc_layer(out_channel=256),
                                      cscc_layer(out_channel=128), cscc_layer(out_channel=64),
                                      ])
        self.CSCC_E2 = nn.ModuleList([cscc_layer(out_channel=64), cscc_layer(out_channel=128),
                                      cscc_layer(out_channel=256), cscc_layer(out_channel=256),
                                      cscc_layer(out_channel=128), cscc_layer(out_channel=64),
                                      ])
        self.CSCC_E3 = nn.ModuleList([cscc_layer(out_channel=64), cscc_layer(out_channel=128),
                                      cscc_layer(out_channel=256), cscc_layer(out_channel=256),
                                      cscc_layer(out_channel=128), cscc_layer(out_channel=64),
                                      ])

        self.CSCC_D1 = cscc_layer(out_channel=512)
        self.CSCC_D2 = cscc_layer(out_channel=512)
        self.CSCC_D3 = cscc_layer(out_channel=512)

        self.fc_cat = FCLayer(448, 512, bias=True, activation='relu')
        self.fc1_1 = FCLayer(512, 256, bias=True, activation='relu')
        self.fc1_2 = FCLayer(256, 128, bias=True, activation='relu')
        self.fc1_3 = FCLayer(128, 64, bias=True, activation='relu')
        self.fc1_4 = FCLayer(64, 3, bias=True, activation=None)

        self.fc_cat2 = FCLayer(448, 512, bias=True, activation='relu')
        self.fc2_fuse = FCLayer(1024, 512, bias=True, activation='relu')
        self.fc2_1 = FCLayer(512, 256, bias=True, activation='relu')
        self.fc2_2 = FCLayer(256, 128, bias=True, activation='relu')
        self.fc2_3 = FCLayer(128, 64, bias=True, activation='relu')
        self.fc2_4 = FCLayer(64, 3, bias=True, activation=None)

        self.fc_cat3 = FCLayer(448, 512, bias=True, activation='relu')
        self.fc3_fuse = FCLayer(1024, 512, bias=True, activation='relu')
        self.fc3_1 = FCLayer(512, 256, bias=True, activation='relu')
        self.fc3_2 = FCLayer(256, 128, bias=True, activation='relu')
        self.fc3_3 = FCLayer(128, 64, bias=True, activation='relu')
        self.fc3_4 = FCLayer(64, 3, bias=True, activation=None)

        self.fc_cat4 = FCLayer(448, 512, bias=True, activation='relu')
        self.fc4_fuse = FCLayer(1024, 512, bias=True, activation='relu')
        self.fc4_1 = FCLayer(512, 256, bias=True, activation='relu')
        self.fc4_2 = FCLayer(256, 128, bias=True, activation='relu')
        self.fc4_3 = FCLayer(128, 64, bias=True, activation='relu')
        self.fc4_4 = FCLayer(64, 3, bias=True, activation=None)

    @property
    def out_channels(self):
        return self.convs[-1].out_channels

    def forward(self, x):
        # === stage 1 ===
        origin_feature = torch.transpose(x, 1, 2)
        features = torch.transpose(x, 1, 2)  # (bs, 3, num_points)
        out_features_list = []
        for i in range(len(self.point_features)):
            features = self.point_features[i](features, origin_feature)
            out_features_list.append(features)

        out_feature = torch.cat(out_features_list, dim=1)
        out_feature = torch.transpose(out_feature, 1, 2)
        out_feature = self.fc_cat(out_feature)

        # decoder
        decoder_features_list = []
        out1 = self.fc1_1(out_feature)
        decoder_features_list.append(out1)
        out1 = self.fc1_2(out1)
        decoder_features_list.append(out1)
        out1 = self.fc1_3(out1)
        decoder_features_list.append(out1)
        out1 = self.fc1_4(out1)
        stage_x = x + out1

        # === stage 2 ===
        stage2_origin_feature = torch.transpose(stage_x, 1, 2)
        stage2_features = torch.transpose(stage_x, 1, 2)
        stage2_out_features_list = []
        for i in range(len(self.point_features2)):
            stage2_features = self.point_features2[i](stage2_features, stage2_origin_feature)
            # CSCC_E
            stage2_features = stage2_features + torch.transpose(
                self.CSCC_E1[i](torch.transpose(out_features_list[i], 1, 2)), 1, 2) + torch.transpose(
                self.CSCC_E1[-i - 1](decoder_features_list[-i - 1]), 1, 2)
            stage2_out_features_list.append(stage2_features)

        stage2_out_feature = torch.cat(stage2_out_features_list, dim=1)
        stage2_out_feature = torch.transpose(stage2_out_feature, 1, 2)
        stage2_out_feature = self.fc_cat2(stage2_out_feature)

        # CSCC_D
        CSCC_D_stage_x = self.CSCC_D1(out_feature)
        stage2_out_feature = torch.cat((stage2_out_feature, CSCC_D_stage_x), dim=2)
        stage2_out_feature = self.fc2_fuse(stage2_out_feature)

        stage2_decoder_features_list = []
        out2 = self.fc2_1(stage2_out_feature)
        stage2_decoder_features_list.append(out2)
        out2 = self.fc2_2(out2)
        stage2_decoder_features_list.append(out2)
        out2 = self.fc2_3(out2)
        stage2_decoder_features_list.append(out2)
        out2 = self.fc2_4(out2)
        stage2_x = stage_x + out2

        # === stage 3 ===
        stage3_origin_feature = torch.transpose(stage2_x, 1, 2)
        stage3_features = torch.transpose(stage2_x, 1, 2)
        stage3_out_features_list = []
        for i in range(len(self.point_features3)):
            stage3_features = self.point_features3[i](stage3_features, stage3_origin_feature)
            # CSCC_E
            stage3_features = stage3_features + torch.transpose(
                self.CSCC_E2[i](torch.transpose(stage2_out_features_list[i], 1, 2)), 1,
                2) + torch.transpose(
                self.CSCC_E2[-i - 1](stage2_decoder_features_list[-i - 1]), 1, 2)
            stage3_out_features_list.append(stage3_features)

        stage3_out_feature = torch.cat(stage3_out_features_list, dim=1)
        stage3_out_feature = torch.transpose(stage3_out_feature, 1, 2)
        stage3_out_feature = self.fc_cat3(stage3_out_feature)

        # CSCC_D
        CSCC_D_stage2_x = self.CSCC_D2(stage2_out_feature)
        stage3_out_feature = torch.cat((stage3_out_feature, CSCC_D_stage2_x), dim=2)
        stage3_out_feature = self.fc3_fuse(stage3_out_feature)

        stage3_decoder_features_list = []
        out3 = self.fc3_1(stage3_out_feature)
        stage3_decoder_features_list.append(out3)
        out3 = self.fc3_2(out3)
        stage3_decoder_features_list.append(out3)
        out3 = self.fc3_3(out3)
        stage3_decoder_features_list.append(out3)
        out3 = self.fc3_4(out3)
        stage3_x = stage2_x + out3

        # === stage 4 ===
        stage4_origin_feature = torch.transpose(stage3_x, 1, 2)
        stage4_features = torch.transpose(stage3_x, 1, 2)
        stage4_out_features_list = []
        for i in range(len(self.point_features4)):
            stage4_features = self.point_features4[i](stage4_features, stage4_origin_feature)
            # CSCC_E
            stage4_features = stage4_features + torch.transpose(
                self.CSCC_E3[i](torch.transpose(stage3_out_features_list[i], 1, 2)), 1,
                2) + torch.transpose(
                self.CSCC_E3[-i - 1](stage3_decoder_features_list[-i - 1]), 1, 2)
            stage4_out_features_list.append(stage4_features)

        stage4_out_feature = torch.cat(stage4_out_features_list, dim=1)
        stage4_out_feature = torch.transpose(stage4_out_feature, 1, 2)
        stage4_out_feature = self.fc_cat4(stage4_out_feature)

        # CSCC_D
        CSCC_D_stage3_x = self.CSCC_D3(stage3_out_feature)
        stage4_out_feature = torch.cat((stage4_out_feature, CSCC_D_stage3_x), dim=2)
        stage4_out_feature = self.fc4_fuse(stage4_out_feature)

        stage4_decoder_features_list = []
        out4 = self.fc4_1(stage4_out_feature)
        stage4_decoder_features_list.append(out4)
        out4 = self.fc4_2(out4)
        stage4_decoder_features_list.append(out4)
        out4 = self.fc4_3(out4)
        stage4_decoder_features_list.append(out4)
        out4 = self.fc4_4(out4)
        stage4_x = stage3_x + out4

        return [out1, out2, out3, out4]


