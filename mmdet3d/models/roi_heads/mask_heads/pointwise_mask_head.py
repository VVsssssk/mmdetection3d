# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.cnn.bricks import build_norm_layer
from mmengine.model import BaseModule
from torch import nn as nn
from torch.nn import functional as F

from mmdet3d.models.builder import build_loss
from mmdet3d.registry import MODELS
from mmdet.models.utils import multi_apply


@MODELS.register_module()
class PointwiseMaskHead(BaseModule):

    def __init__(self,
                 in_channels,
                 num_classes=1,
                 mlps=(256, 256),
                 extra_width=0.1,
                 class_agnostic=False,
                 norm_cfg=dict(type='BN1d', eps=1e-5, momentum=0.1),
                 init_cfg=None,
                 loss_seg=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     reduction='sum',
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0)):
        super(PointwiseMaskHead, self).__init__(init_cfg=init_cfg)
        self.extra_width = extra_width
        self.class_agnostic = class_agnostic
        self.num_classes = num_classes

        self.in_channels = in_channels
        self.use_sigmoid_cls = loss_seg.get('use_sigmoid', False)

        out_channels = 1 if class_agnostic else num_classes
        if self.use_sigmoid_cls:
            self.out_channels = out_channels
        else:
            self.out_channels = out_channels + 1

        mlps_layers = []
        cin = in_channels
        for cout in mlps:
            mlps_layers.extend([
                nn.Linear(cin, cout, bias=False),
                build_norm_layer(norm_cfg, cout)[1],
                nn.ReLU()
            ])
            cin = cout
        mlps_layers.append(nn.Linear(cin, self.out_channels, bias=True))

        self.seg_cls_layer = nn.Sequential(*mlps_layers)

        self.loss_seg = build_loss(loss_seg)

    def forward(self, feats):
        seg_preds = self.seg_cls_layer(feats)  # (N, 1)
        return dict(seg_preds=seg_preds)

    def get_targets_single(self, point_xyz, gt_bboxes_3d, gt_labels_3d):
        """generate segmentation and part prediction targets for a single
        sample.

        Args:
            voxel_centers (torch.Tensor): The center of voxels in shape
                (voxel_num, 3).
            gt_bboxes_3d (:obj:`BaseInstance3DBoxes`): Ground truth boxes in
                shape (box_num, 7).
            gt_labels_3d (torch.Tensor): Class labels of ground truths in
                shape (box_num).

        Returns:
            tuple[torch.Tensor]: Segmentation targets with shape [voxel_num]
                part prediction targets with shape [voxel_num, 3]
        """
        point_cls_labels_single = point_xyz.new_zeros(
            point_xyz.shape[0]).long()
        enlarged_gt_boxes = gt_bboxes_3d.enlarged_box(self.extra_width)

        box_idxs_of_pts = gt_bboxes_3d.points_in_boxes_part(point_xyz).long()
        extend_box_idxs_of_pts = enlarged_gt_boxes.points_in_boxes_part(
            point_xyz).long()
        box_fg_flag = box_idxs_of_pts >= 0
        fg_flag = box_fg_flag.clone()
        ignore_flag = fg_flag ^ (extend_box_idxs_of_pts >= 0)
        point_cls_labels_single[ignore_flag] = -1
        gt_box_of_fg_points = gt_labels_3d[box_idxs_of_pts[fg_flag]]
        point_cls_labels_single[
            fg_flag] = 1 if self.num_classes == 1 else gt_box_of_fg_points.long(
            )
        return point_cls_labels_single,

    def get_targets(self, points_bxyz, batch_gt_instances_3d):
        """generate segmentation and part prediction targets.

        Args:
            xyz (torch.Tensor): The center of voxels in shape
                (B, num_points, 3).
            gt_bboxes_3d (:obj:`BaseInstance3DBoxes`): Ground truth boxes in
                shape (box_num, 7).
            gt_labels_3d (torch.Tensor): Class labels of ground truths in
                shape (box_num).

        Returns:
            dict: Prediction targets

                - seg_targets (torch.Tensor): Segmentation targets
                    with shape [voxel_num].
                - part_targets (torch.Tensor): Part prediction targets
                    with shape [voxel_num, 3].
        """
        batch_size = len(batch_gt_instances_3d)
        points_xyz_list = []
        gt_bboxes_3d = []
        gt_labels_3d = []
        for idx in range(batch_size):
            coords_idx = points_bxyz[:, 0] == idx
            points_xyz_list.append(points_bxyz[coords_idx][..., 1:])
            gt_bboxes_3d.append(batch_gt_instances_3d[idx].bboxes_3d)
            gt_labels_3d.append(batch_gt_instances_3d[idx].labels_3d)
        seg_targets, = multi_apply(self.get_targets_single, points_xyz_list,
                                   gt_bboxes_3d, gt_labels_3d)
        seg_targets = torch.cat(seg_targets, dim=0)
        return dict(seg_targets=seg_targets)

    def loss(self, semantic_results, semantic_targets):
        seg_preds = semantic_results['seg_preds']
        seg_targets = semantic_targets['seg_targets']

        positives = (seg_targets > 0)

        negative_cls_weights = (seg_targets == 0).float()
        seg_weights = (negative_cls_weights + 1.0 * positives).float()
        pos_normalizer = positives.sum(dim=0).float()
        seg_weights /= torch.clamp(pos_normalizer, min=1.0)

        seg_preds = torch.sigmoid(seg_preds)
        loss_seg = self.loss_seg(seg_preds, (~positives).long(), seg_weights)
        return dict(loss_semantic=loss_seg)
