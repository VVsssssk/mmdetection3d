# Copyright (c) OpenMMLab. All rights reserved.
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks import build_norm_layer
from mmcv.ops import PointsSampler, QueryAndGroup, gather_points
from mmengine.model import BaseModule

from mmdet3d.models.layers.pointnet_modules import build_sa_module
from mmdet3d.registry import MODELS


def bilinear_interpolate_torch(im, x, y):
    """
    Args:
        im: (H, W, C) [y, x]
        x: (N)
        y: (N)

    Returns:

    """
    x0 = torch.floor(x).long()
    x1 = x0 + 1

    y0 = torch.floor(y).long()
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, im.shape[1] - 1)
    x1 = torch.clamp(x1, 0, im.shape[1] - 1)
    y0 = torch.clamp(y0, 0, im.shape[0] - 1)
    y1 = torch.clamp(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1.type_as(x) - x) * (y1.type_as(y) - y)
    wb = (x1.type_as(x) - x) * (y - y0.type_as(y))
    wc = (x - x0.type_as(x)) * (y1.type_as(y) - y)
    wd = (x - x0.type_as(x)) * (y - y0.type_as(y))
    ans = torch.t((torch.t(Ia) * wa)) + torch.t(torch.t(Ib) * wb) + torch.t(
        torch.t(Ic) * wc) + torch.t(torch.t(Id) * wd)
    return ans


@MODELS.register_module()
class VoxelSetAbstraction(BaseModule):

    def __init__(self,
                 voxel_size,
                 point_cloud_range,
                 keypoints_sampler=None,
                 rawpoints_dim=None,
                 voxel_sa_cfg_list=None,
                 rawpoint_sa_cfg=None,
                 bev_sa_cfg=None,
                 voxel_center_as_source=False,
                 out_channels=128,
                 norm_cfg=dict(type='BN1d', eps=1e-5, momentum=0.1)):
        super().__init__()
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.voxel_center_as_source = voxel_center_as_source

        self.voxel_sa_cfg_list = voxel_sa_cfg_list
        self.rawpoint_sa_cfg = rawpoint_sa_cfg
        self.bev_sa_cfg = bev_sa_cfg

        self.keypoints_sampler = PointsSampler(**keypoints_sampler)
        in_channels = 0
        if bev_sa_cfg is not None:
            in_channels += bev_sa_cfg.in_channels

        if rawpoint_sa_cfg is not None:
            mlp_channels = []
            for mlp_channel in self.rawpoint_sa_cfg.mlp_channels:
                mlp_channel = [rawpoints_dim - 3] + list(mlp_channel)
                mlp_channels.append(mlp_channel)
            self.rawpoint_sa_cfg.update(mlp_channels=mlp_channels)
            self.rawpoints_sa_layer = build_sa_module(self.rawpoint_sa_cfg)
            in_channels += sum(
                [x[-1] for x in self.rawpoints_sa_layer.mlp_channels])

        if voxel_sa_cfg_list is not None:
            self.voxel_sa_layers = nn.ModuleList()
            self.downsample_times_map = []
            self.source_feats_idx = []
            for idx, voxel_sa_cfg in enumerate(voxel_sa_cfg_list):
                self.downsample_times_map.append(
                    voxel_sa_cfg.pop('scale_factor'))
                self.source_feats_idx.append(
                    voxel_sa_cfg.pop('source_feats_idx', idx))
                cur_layer = build_sa_module(voxel_sa_cfg)
                self.voxel_sa_layers.append(cur_layer)
                in_channels += sum([x[-1] for x in cur_layer.mlp_channels])
        self.point_feature_fusion = nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=False),
            build_norm_layer(norm_cfg, out_channels)[1], nn.ReLU())

    def interpolate_from_bev_features(self, keypoints, bev_features,
                                      scale_factor):
        point_bev_features_list = []
        batch_size = bev_features.shape[0]
        for k in range(batch_size):
            cur_x_idxs = (keypoints[k, :, 0] -
                          self.point_cloud_range[0]) / self.voxel_size[0]
            cur_y_idxs = (keypoints[k, :, 1] -
                          self.point_cloud_range[1]) / self.voxel_size[1]

            cur_x_idxs = cur_x_idxs / scale_factor
            cur_y_idxs = cur_y_idxs / scale_factor

            cur_bev_features = bev_features[k].permute(1, 2, 0)  # (H, W, C)
            point_bev_features = bilinear_interpolate_torch(
                cur_bev_features, cur_x_idxs, cur_y_idxs)
            point_bev_features_list.append(point_bev_features)

        point_bev_features = torch.stack(
            point_bev_features_list, dim=0)  # (N1 + N2 + ..., C)
        return point_bev_features

    def aggregate_keypoint_features_from_one_source(self,
                                                    aggregate_func,
                                                    points_xyz,
                                                    features=None,
                                                    target_xyz=None):
        new_xyz, pooled_features, indices = aggregate_func(
            points_xyz=points_xyz, features=features, target_xyz=target_xyz)
        return new_xyz, pooled_features.transpose(1, 2), indices

    def _sample_points(self, raw_points):
        points_xyz = raw_points[:, :, :3].contiguous()
        points_flipped = raw_points.transpose(1, 2).contiguous()
        indices = self.keypoints_sampler(points_xyz, None)
        new_points = gather_points(points_flipped,
                                   indices).transpose(1, 2).contiguous()
        return new_points, indices

    def concat_batch_points(self, batch_points, pad_type='zeros'):
        pad_points_list = []
        max_points_num = max(points.shape[0] for points in batch_points)
        for points in batch_points:
            pad_points_list.append(
                F.pad(points, (0, 0, 0, max_points_num - points.shape[0])))
        pad_batch_points = torch.stack(pad_points_list, dim=0)
        return pad_batch_points

    def get_voxel_centers(self, coors, features, scale_factor=1):
        assert coors.shape[1] == 4
        batch_size = coors[-1][0] + 1
        batch_coors = []
        batch_feats = []
        voxel_centers = coors[:, [3, 2, 1]].float()  # (xyz)
        voxel_size = voxel_centers.new_tensor(self.voxel_size)
        pc_range_min = voxel_centers.new_tensor(self.point_cloud_range[:3])

        voxel_centers = voxel_centers * voxel_size * scale_factor + pc_range_min

        voxel_centers.add_(0.5 * voxel_size * scale_factor)
        for i in range(batch_size):
            batch_coors.append(voxel_centers[coors[:, 0] == i])
            batch_feats.append(features[coors[:, 0] == i])
        batch_coors = self.concat_batch_points(batch_coors)
        batch_feats = self.concat_batch_points(batch_feats)
        return batch_coors, batch_feats

    def forward(self, batch_inputs_dict, feats_dict, rpn_results_list):
        """
        Args:
            batch_dict:
                batch_size:
                keypoints: (B, num_keypoints, 3)
                multi_scale_3d_features: {
                        'x_conv4': ...
                    }
                points: optional (N, 1 + 3 + C) [bs_idx, x, y, z, ...]
                spatial_features: optional
                spatial_features_stride: optional

        Returns:
            point_features: (N, C)
            point_coords: (N, 4)

        """
        batch_points = self.concat_batch_points(batch_inputs_dict['points'])
        point_features_list = []
        keypoints, _ = self._sample_points(batch_points)
        num_keypoints = keypoints.shape[1]

        if self.bev_sa_cfg:
            spatial_feats = feats_dict['spatial_feats']
            keypoint_bev_features = self.interpolate_from_bev_features(
                keypoints, spatial_feats, self.bev_sa_cfg.scale_factor)
            point_features_list.append(keypoint_bev_features)

        if self.rawpoint_sa_cfg:
            _, pooled_features, _ = \
                self.aggregate_keypoint_features_from_one_source(self.rawpoints_sa_layer,
                                                                 points_xyz=batch_points[:, :, :3].contiguous(),
                                                                 features=batch_points[:, :, 3:].transpose(1, 2),
                                                                 target_xyz=keypoints[:, :, :3].contiguous())
            point_features_list.append(pooled_features)

        for i, voxel_sa_layer in enumerate(self.voxel_sa_layers):
            cur_coords = feats_dict['multi_scale_3d_feats'][i].indices
            cur_features = feats_dict['multi_scale_3d_feats'][
                i].features.contiguous()
            voxel_xyz, voxel_feats = self.get_voxel_centers(
                coors=cur_coords,
                features=cur_features,
                scale_factor=self.downsample_times_map[i])
            _, pooled_features, _ = self.aggregate_keypoint_features_from_one_source(
                self.voxel_sa_layers[i],
                points_xyz=voxel_xyz.contiguous(),
                features=voxel_feats.transpose(1, 2).contiguous(),
                target_xyz=keypoints[:, :, :3].contiguous())
            point_features_list.append(pooled_features)
        batch_size = len(batch_inputs_dict['points'])
        keypoint_features = torch.cat(
            point_features_list, dim=-1).view(batch_size * num_keypoints, -1)
        # keypoint_features = torch.load('point_feats.pkl')
        fusion_point_features = self.point_feature_fusion(keypoint_features)

        return dict(
            keypoint_features=keypoint_features,
            fusion_keypoint_features=fusion_point_features,
            keypoints=keypoints)
