# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner import BaseModule

from mmdet3d.models.layers.pointnet_modules import build_sa_module
from mmdet3d.registry import MODELS
from mmdet3d.structures.bbox_3d import rotation_3d_in_axis


@MODELS.register_module()
class Batch3DRoIGridExtractor(BaseModule):

    def __init__(self, sa_module_cfg, grid_size=6, init_cfg=None):
        super(Batch3DRoIGridExtractor, self).__init__(init_cfg=init_cfg)
        self.roi_grid_pool_layer = build_sa_module(sa_module_cfg)
        self.grid_size = grid_size

    def forward(self, feats, coordinate, rois):

        roi_grid = self.get_dense_grid_points(rois[:, 1:])
        batch_size = coordinate.shape[0]
        grid_points = roi_grid.reshape(batch_size, -1, 3)
        feats = feats.view(batch_size, -1, feats.shape[-1])

        _, pooled_features, _ = self.roi_grid_pool_layer(
            points_xyz=coordinate[:, :, :3].contiguous(),  # B, N, 3
            target_xyz=grid_points,  #B, M, 3
            features=feats.transpose(1, 2).contiguous())  # B, C, N

        pooled_features = pooled_features.transpose(1, 2)
        pooled_features = pooled_features.view(-1, self.grid_size,
                                               self.grid_size, self.grid_size,
                                               pooled_features.shape[-1])
        # (BxN, 6, 6, 6, C)
        return pooled_features

    def get_dense_grid_points(self, rois):
        faked_features = rois.new_ones(
            (self.grid_size, self.grid_size, self.grid_size))
        dense_idx = faked_features.nonzero()
        dense_idx = dense_idx.repeat(rois.size(0), 1, 1).float()
        dense_idx = ((dense_idx + 0.5) / self.grid_size)
        dense_idx[..., :3] -= 0.5

        roi_ctr = rois[:, :3]
        roi_dim = rois[:, 3:6]
        roi_grid_points = dense_idx * roi_dim.view(-1, 1, 3)
        roi_grid_points = rotation_3d_in_axis(
            roi_grid_points, rois[:, 6], axis=2)
        roi_grid_points += roi_ctr.view(-1, 1, 3)

        return roi_grid_points
