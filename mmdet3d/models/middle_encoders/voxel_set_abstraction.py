# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks import build_norm_layer
from mmcv.ops import PointsSampler, gather_points
from mmengine.model import BaseModule, ModuleList
from torch import Tensor

from mmdet3d.models.layers.pointnet_modules import build_sa_module
from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import InstanceData
from mmdet3d.models.layers.pointnet_modules.pointnet2_modules import GuidedSAModuleMSG
from mmcv.ops.furthest_point_sample import furthest_point_sample
def bilinear_interpolate_torch(im, x, y):
    """Bilinear interpolate for.

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
    """Voxel set abstraction module for PVRCNN and PVRCNN++.

    Args:
        keypoints_sampler (dict or ConfigDict): Key point sampler config.
            It is used to build `PointsSampler` to sample key points from
            raw points.
        voxel_size (list[float]): Size of voxels. Defaults to
            [0.05, 0.05, 0.1].
        point_cloud_range (list[float]): Point cloud range. Defaults to
            [0, -40, -3, 70.4, 40, 1].
        rawpoint_sa_cfg (dict or ConfigDict, optional): SA module cfg. Used to
            gather key points features from raw points. Default to None.
        voxel_sa_cfg_list (List[dict or ConfigDict], optional): List of SA
            module cfg. Used to gather key points features from multi-level
            voxel features. Default to None.
        bev_cfg (dict or ConfigDict, optional): Bev features encode cfg. Used
            to gather key points features from Bev features. Default to None.
        sample_mode (str): Key points sample mode include
            `raw_points` and `voxel_centers` modes. If used `raw_points`
            the module will use keypoints_sampler to gather key points from
            raw points. Else if used `voxel_centers`, the module will use
            voxel centers as key points to extract features. Default to
            `raw_points.`
        fused_out_channels (int): Key points feature output channel
            num after fused. Default to 128.
        norm_cfg (dict[str]): Config of normalization layer. Default
            used dict(type='BN1d', eps=1e-5, momentum=0.1)
    """

    def __init__(self, num_keypoints, out_channels, voxel_size,
                 point_cloud_range, voxel_sa_configs, rawpoint_sa_config=None,
                 bev_sa_config=None, voxel_center_as_source=False,
                 norm_cfg=dict(type='BN2d', eps=1e-5, momentum=0.01),
                 voxel_center_align='half',
                 debug=False):
        super().__init__()
        self.debug = debug
        assert voxel_center_align in ('half', 'halfmin')
        self.voxel_center_align = voxel_center_align
        self.num_keypoints = num_keypoints
        self.out_channels = out_channels
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.voxel_center_as_source = voxel_center_as_source

        self.voxel_sa_configs = voxel_sa_configs
        self.rawpoint_sa_config = rawpoint_sa_config
        self.bev_sa_config = bev_sa_config

        self.voxel_sa_layers = nn.ModuleList()

        gathered_channels = 0

        self.rawpoints_sa = (rawpoint_sa_config is not None)
        if rawpoint_sa_config is not None:
            self.rawpoints_sa_layer = GuidedSAModuleMSG(
                in_channels=rawpoint_sa_config.in_channels,
                radii=rawpoint_sa_config.pool_radius,
                nsamples=rawpoint_sa_config.samples,
                mlps=rawpoint_sa_config.mlps,
                use_xyz=True,
                pool_method='max',
                norm_cfg=norm_cfg)
            gathered_channels += sum([x[-1] for x in rawpoint_sa_config.mlps])

        for voxel_sa_config in voxel_sa_configs:
            cur_layer = GuidedSAModuleMSG(
                in_channels=voxel_sa_config.in_channels,
                radii=voxel_sa_config.pool_radius,
                nsamples=voxel_sa_config.samples,
                mlps=voxel_sa_config.mlps,
                use_xyz=True,
                pool_method='max',
                norm_cfg=norm_cfg)
            self.voxel_sa_layers.append(cur_layer)
            gathered_channels += sum([x[-1] for x in voxel_sa_config.mlps])

        self.bev_sa = (bev_sa_config is not None)
        if bev_sa_config is not None:
            self.bev_sa = True
            gathered_channels += bev_sa_config.in_channels

        norm_cfg=dict(type='BN1d', eps=1e-5, momentum=0.01)
        self.point_feature_fusion = nn.Sequential(
            nn.Linear(gathered_channels, out_channels, bias=False),
            build_norm_layer(norm_cfg, out_channels)[1],
            nn.ReLU())

    # def interpolate_from_bev_features(self, keypoints, bev_features,
    #                                   scale_factor):
    #     _, _, y_grid, x_grid = bev_features.shape
    #
    #     voxel_size_xy = keypoints.new_tensor(self.voxel_size[:2])
    #
    #     bev_tl_grid_cxy = keypoints.new_tensor(self.point_cloud_range[:2])
    #     bev_br_grid_cxy = keypoints.new_tensor(self.point_cloud_range[3:5])
    #     if self.voxel_center_align == 'half':
    #         bev_tl_grid_cxy.add_(0.5 * voxel_size_xy * scale_factor)
    #         bev_br_grid_cxy.sub_(0.5 * voxel_size_xy * scale_factor)
    #     elif self.voxel_center_align == 'halfmin':
    #         bev_tl_grid_cxy.add_(0.5 * voxel_size_xy)
    #         bev_br_grid_cxy.sub_(voxel_size_xy * (scale_factor - 0.5))
    #
    #     xy = keypoints[..., :2]
    #
    #     grid_sample_xy = (xy - bev_tl_grid_cxy[None, None, :]) / (
    #         (bev_br_grid_cxy - bev_tl_grid_cxy)[None, None, :])
    #
    #     grid_sample_xy = (grid_sample_xy * 2 - 1).unsqueeze(1)
    #     point_bev_features = F.grid_sample(bev_features,
    #                                        grid=grid_sample_xy,
    #                                        align_corners=True)
    #     return point_bev_features.squeeze(2).permute(0, 2, 1).contiguous()
    def interpolate_from_bev_features(self, keypoints, bev_features, batch_size, bev_stride):
        """
        Args:
            keypoints: (N1 + N2 + ..., 4)
            bev_features: (B, C, H, W)
            batch_size:
            bev_stride:

        Returns:
            point_bev_features: (N1 + N2 + ..., C)
        """
        x_idxs = (keypoints[..., 0] - self.point_cloud_range[0]) / self.voxel_size[0]
        y_idxs = (keypoints[..., 1] - self.point_cloud_range[1]) / self.voxel_size[1]

        x_idxs = x_idxs / bev_stride
        y_idxs = y_idxs / bev_stride

        point_bev_features_list = []
        for k in range(batch_size):
            cur_x_idxs = x_idxs[k,...]
            cur_y_idxs = y_idxs[k,...]
            cur_bev_features = bev_features[k].permute(1, 2, 0)  # (H, W, C)
            point_bev_features = bilinear_interpolate_torch(cur_bev_features, cur_x_idxs, cur_y_idxs)
            point_bev_features_list.append(point_bev_features)

        point_bev_features = torch.cat(point_bev_features_list, dim=0)  # (N1 + N2 + ..., C)
        return point_bev_features.view(batch_size,keypoints.shape[1],-1)

    def get_voxel_centers(self, coors, scale_factor):
        assert coors.shape[1] == 4
        voxel_centers = coors[:, [3, 2, 1]].float()  # (xyz)
        voxel_size = voxel_centers.new_tensor(self.voxel_size)
        pc_range_min = voxel_centers.new_tensor(self.point_cloud_range[:3])

        voxel_centers = voxel_centers * voxel_size * scale_factor + pc_range_min
        voxel_centers.add_(0.5 * voxel_size * scale_factor)
        # if self.voxel_center_align == 'half':
        #     voxel_centers.add_(0.5 * voxel_size * scale_factor)
        # elif self.voxel_center_align == 'halfmin':
        #     voxel_centers.add_(0.5 * voxel_size)
        # else:
        #     raise NotImplementedError
        return voxel_centers

    def get_sampled_points(self, points, coors):
        assert points is not None or coors is not None
        if self.voxel_center_as_source:
            _src_points = self.get_voxel_centers(coors=coors, scale_factor=1)
            batch_size = coors[-1, 0].item() + 1
            src_points = [_src_points[coors[:, 0] == b] for b in
                          range(batch_size)]
        else:
            src_points = [p[..., :3] for p in points]

        keypoints_list = []
        for points_to_sample in src_points:
            num_points = points_to_sample.shape[0]
            cur_pt_idxs = furthest_point_sample(
                points_to_sample.unsqueeze(dim=0).contiguous(),
                self.num_keypoints).long()[0]

            if num_points < self.num_keypoints:
                times = int(self.num_keypoints / num_points) + 1
                non_empty = cur_pt_idxs[:num_points]
                cur_pt_idxs = non_empty.repeat(times)[:self.num_keypoints]

            keypoints = points_to_sample[cur_pt_idxs]

            keypoints_list.append(keypoints)
        keypoints = torch.stack(keypoints_list, dim=0)  # (B, M, 3)
        return keypoints


    def forward(self, batch_inputs_dict: dict, feats_dict: dict,
                rpn_results_list: List[InstanceData]) -> dict:
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
        points = batch_inputs_dict['points']
        voxel_encode_features = feats_dict['multi_scale_3d_feats']
        bev_encode_features = feats_dict['spatial_feats']
        keypoints = self.get_sampled_points(points, None)

        point_features_list = []
        batch_size = len(points)
        if self.bev_sa:
            point_bev_features = self.interpolate_from_bev_features(
                keypoints, bev_encode_features, batch_size, self.bev_sa_config.scale_factor)
            point_features_list.append(point_bev_features)

        batch_size, num_keypoints, _ = keypoints.shape
        key_xyz = keypoints.view(-1, 3)
        key_xyz_batch_cnt = key_xyz.new_zeros(batch_size).int().fill_(
            num_keypoints)

        if self.rawpoints_sa:
            batch_points = torch.cat(points, dim=0)
            batch_cnt = [len(p) for p in points]
            xyz = batch_points[:, :3].contiguous()
            features = None
            if batch_points.size(1) > 0:
                features = batch_points[:, 3:].contiguous()
            xyz_batch_cnt = xyz.new_tensor(batch_cnt, dtype=torch.int32)

            pooled_points, pooled_features = self.rawpoints_sa_layer(
                xyz=xyz.contiguous(),
                xyz_batch_cnt=xyz_batch_cnt,
                new_xyz=key_xyz.contiguous(),
                new_xyz_batch_cnt=key_xyz_batch_cnt,
                features=features.contiguous(),
            )
            point_features_list.append(
                pooled_features.view(batch_size, num_keypoints, -1))

        for k, voxel_sa_layer in enumerate(self.voxel_sa_layers):
            cur_coords = voxel_encode_features[k].indices
            xyz = self.get_voxel_centers(
                coors=cur_coords,
                scale_factor=self.voxel_sa_configs[k].scale_factor
            ).contiguous()
            xyz_batch_cnt = xyz.new_zeros(batch_size).int()
            for bs_idx in range(batch_size):
                xyz_batch_cnt[bs_idx] = (cur_coords[:, 0] == bs_idx).sum()

            pooled_points, pooled_features = voxel_sa_layer(
                xyz=xyz.contiguous(),
                xyz_batch_cnt=xyz_batch_cnt,
                new_xyz=key_xyz.contiguous(),
                new_xyz_batch_cnt=key_xyz_batch_cnt,
                features=voxel_encode_features[k].features.contiguous(),
            )
            point_features_list.append(
                pooled_features.view(batch_size, num_keypoints, -1))

        point_features = torch.cat(point_features_list, dim=-1).view(
            batch_size * num_keypoints, -1)

        fusion_point_features = self.point_feature_fusion(point_features)

        bid = torch.arange(batch_size * num_keypoints,
                           device=keypoints.device) // num_keypoints
        key_bxyz = torch.cat((bid.to(key_xyz.dtype).unsqueeze(dim=-1),
                              key_xyz), dim=-1)

        return dict(keypoint_features=point_features,
                    fusion_keypoint_features=fusion_point_features,
                    keypoints=key_bxyz)
