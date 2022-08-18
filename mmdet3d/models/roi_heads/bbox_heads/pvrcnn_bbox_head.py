# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from mmcv.cnn.bricks import build_norm_layer
from mmengine.data import InstanceData
from mmengine.model import BaseModule
from torch import nn as nn
from mmcv.cnn import ConvModule
from mmdet3d.models.builder import build_loss
from mmdet3d.models.layers import nms_bev, nms_normal_bev
from mmdet3d.registry import MODELS, TASK_UTILS
from mmdet3d.structures.bbox_3d import (LiDARInstance3DBoxes,
                                        rotation_3d_in_axis, xywhr2xyxyr)
from mmdet.models.utils import multi_apply


@MODELS.register_module()
class PVRCNNBboxHead(BaseModule):
    """PartA2 RoI head.

    Args:
        num_classes (int): The number of classes to prediction.
        seg_in_channels (int): Input channels of segmentation
            convolution layer.
        part_in_channels (int): Input channels of part convolution layer.
        seg_conv_channels (list(int)): Out channels of each
            segmentation convolution layer.
        part_conv_channels (list(int)): Out channels of each
            part convolution layer.
        merge_conv_channels (list(int)): Out channels of each
            feature merged convolution layer.
        down_conv_channels (list(int)): Out channels of each
            downsampled convolution layer.
        shared_fc_channels (list(int)): Out channels of each shared fc layer.
        cls_channels (list(int)): Out channels of each classification layer.
        reg_channels (list(int)): Out channels of each regression layer.
        dropout_ratio (float): Dropout ratio of classification and
            regression layers.
        roi_feat_size (int): The size of pooled roi features.
        with_corner_loss (bool): Whether to use corner loss or not.
        bbox_coder (:obj:`BaseBBoxCoder`): Bbox coder for box head.
        conv_cfg (dict): Config dict of convolutional layers
        norm_cfg (dict): Config dict of normalization layers
        loss_bbox (dict): Config dict of box regression loss.
        loss_cls (dict): Config dict of classifacation loss.
    """

    def __init__(self,
                 in_channels,
                 grid_size,
                 num_classes,
                 class_agnostic=True,
                 shared_fc=(256, 256),
                 cls_fc=(256, 256),
                 reg_fc=(256, 256),
                 dropout=0.3,
                 with_corner_loss=True,
                 bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
                 norm_cfg=dict(type='BN1d', eps=1e-5, momentum=0.1),
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     reduction='none',
                     loss_weight=1.0),
                 init_cfg=None):
        super(PVRCNNBboxHead, self).__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.with_corner_loss = with_corner_loss
        self.class_agnostic = class_agnostic
        self.bbox_coder = TASK_UTILS.build(bbox_coder)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_cls = build_loss(loss_cls)
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)

        cls_out_channels = 1 if class_agnostic else num_classes
        self.reg_out_channels = self.bbox_coder.code_size * cls_out_channels
        if self.use_sigmoid_cls:
            self.cls_out_channels = cls_out_channels
        else:
            self.cls_out_channels = cls_out_channels + 1

        self.dropout = dropout
        self.grid_size = grid_size
        self.norm_cfg = norm_cfg

        in_channels *= (self.grid_size**3)
        self.in_channels = in_channels

        self.shared_fc_layer = self.make_fc_layers(in_channels, shared_fc,
                                                   range(len(shared_fc) - 1))
        self.cls_layers = self.make_fc_layers(shared_fc[-1], cls_fc, range(1))
        self.cls_out = nn.Conv1d(
            cls_fc[-1], self.cls_out_channels, 1, bias=True)
        self.reg_layers = self.make_fc_layers(shared_fc[-1], reg_fc, range(1))
        self.reg_out = nn.Conv1d(
            reg_fc[-1], self.reg_out_channels, 1, bias=True)

        if init_cfg is None:
            self.init_cfg = dict(
                type='Xavier',
                layer=['Conv2d', 'Conv1d'],
                distribution='uniform')

    def make_fc_layers(self, in_channels, fc, apply_dropout_indices):
        fc_layers = []
        pre_channel = in_channels
        for k, fck in enumerate(fc):
            fc_layers.extend([
                nn.Conv1d(pre_channel, fck, 1, bias=False),
                # ConvModule(
                #     pre_channel,
                #     fck,
                #     kernel_size=1,
                #     stride=1,
                #     inplace=False,
                #     bias=False,
                #     conv_cfg=dict(type='Conv1d')),
                build_norm_layer(self.norm_cfg, fck)[1],
                nn.ReLU()
            ])
            pre_channel = fck
            if self.dropout >= 0 and k in apply_dropout_indices:
                fc_layers.append(nn.Dropout(self.dropout))
        fc_layers = nn.Sequential(*fc_layers)
        return fc_layers

    def forward(self, feats):
        # (B * N, 6, 6, 6, C)
        rcnn_batch_size = feats.shape[0]
        feats = feats.permute(0, 4, 1, 2,
                              3).contiguous().view(rcnn_batch_size, -1, 1)
        # (BxN, C*6*6*6)
        shared_feats = self.shared_fc_layer(feats)
        cls_feats = self.cls_layers(shared_feats)
        reg_feats = self.reg_layers(shared_feats)
        bbox_score = self.cls_out(cls_feats).transpose(1,
                                                       2).contiguous().squeeze(
                                                           dim=1)  # (B, 1)
        bbox_reg = self.reg_out(reg_feats).transpose(1,
                                                     2).contiguous().squeeze(
                                                         dim=1)  # (B, C)
        # bbox_dict = torch.load('bbox_head.pkl')
        # bbox_score=bbox_dict['bbox_score']
        # bbox_reg=bbox_dict['bbox_reg']
        return bbox_score, bbox_reg

    def loss(self, cls_score, bbox_pred, rois, labels, bbox_targets,
             pos_gt_bboxes, reg_mask, label_weights, bbox_weights):
        """Coumputing losses.

        Args:
            cls_score (torch.Tensor): Scores of each roi.
            bbox_pred (torch.Tensor): Predictions of bboxes.
            rois (torch.Tensor): Roi bboxes.
            labels (torch.Tensor): Labels of class.
            bbox_targets (torch.Tensor): Target of positive bboxes.
            pos_gt_bboxes (torch.Tensor): Ground truths of positive bboxes.
            reg_mask (torch.Tensor): Mask for positive bboxes.
            label_weights (torch.Tensor): Weights of class loss.
            bbox_weights (torch.Tensor): Weights of bbox loss.

        Returns:
            dict: Computed losses.

                - loss_cls (torch.Tensor): Loss of classes.
                - loss_bbox (torch.Tensor): Loss of bboxes.
                - loss_corner (torch.Tensor): Loss of corners.
        """
        losses = dict()
        rcnn_batch_size = cls_score.shape[0]

        # calculate class loss
        cls_flat = cls_score.view(-1)
        loss_cls = self.loss_cls(cls_flat, labels, label_weights)
        losses['loss_cls'] = loss_cls

        # calculate regression loss
        code_size = self.bbox_coder.code_size
        pos_inds = (reg_mask > 0)
        if pos_inds.any() == 0:
            # fake a part loss
            losses['loss_bbox'] = 0 * bbox_pred.sum()
            if self.with_corner_loss:
                losses['loss_corner'] =  0 * bbox_pred.sum()
        else:
            pos_bbox_pred = bbox_pred.view(rcnn_batch_size, -1)[pos_inds]
            bbox_weights_flat = bbox_weights[pos_inds].view(-1, 1).repeat(
                1, pos_bbox_pred.shape[-1])
            loss_bbox = self.loss_bbox(
                pos_bbox_pred.unsqueeze(dim=0), bbox_targets.unsqueeze(dim=0),
                bbox_weights_flat.unsqueeze(dim=0))
            losses['loss_bbox'] = loss_bbox

            if self.with_corner_loss:
                pos_roi_boxes3d = rois[..., 1:].view(-1, code_size)[pos_inds]
                pos_roi_boxes3d = pos_roi_boxes3d.view(-1, code_size)
                batch_anchors = pos_roi_boxes3d.clone().detach()
                pos_rois_rotation = pos_roi_boxes3d[..., 6].view(-1)
                roi_xyz = pos_roi_boxes3d[..., 0:3].view(-1, 3)
                batch_anchors[..., 0:3] = 0
                # decode boxes
                pred_boxes3d = self.bbox_coder.decode(
                    batch_anchors,
                    pos_bbox_pred.view(-1, code_size),bottom_center=True).view(-1, code_size)

                pred_boxes3d[..., 0:3] = rotation_3d_in_axis(
                    pred_boxes3d[..., 0:3].unsqueeze(1),
                    pos_rois_rotation,
                    axis=2).squeeze(1)

                pred_boxes3d[:, 0:3] += roi_xyz

                # calculate corner loss
                loss_corner = self.get_corner_loss_lidar(
                    pred_boxes3d, pos_gt_bboxes)
                losses['loss_corner'] = loss_corner.mean()

        return losses

    def get_targets(self, sampling_results, rcnn_train_cfg, concat=True):
        """Generate targets.

        Args:
            sampling_results (list[:obj:`SamplingResult`]):
                Sampled results from rois.
            rcnn_train_cfg (:obj:`ConfigDict`): Training config of rcnn.
            concat (bool): Whether to concatenate targets between batches.

        Returns:
            tuple[torch.Tensor]: Targets of boxes and class prediction.
        """
        pos_bboxes_list = [res.pos_bboxes for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        iou_list = [res.iou for res in sampling_results]
        targets = multi_apply(
            self._get_target_single,
            pos_bboxes_list,
            pos_gt_bboxes_list,
            iou_list,
            cfg=rcnn_train_cfg)

        (label, bbox_targets, pos_gt_bboxes, reg_mask, label_weights,
         bbox_weights) = targets

        if concat:
            label = torch.cat(label, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            pos_gt_bboxes = torch.cat(pos_gt_bboxes, 0)
            reg_mask = torch.cat(reg_mask, 0)

            label_weights = torch.cat(label_weights, 0)
            label_weights /= torch.clamp(label_weights.sum(), min=1.0)

            bbox_weights = torch.cat(bbox_weights, 0)
            bbox_weights /= torch.clamp(bbox_weights.sum(), min=1.0)

        return (label, bbox_targets, pos_gt_bboxes, reg_mask, label_weights,
                bbox_weights)

    def _get_target_single(self, pos_bboxes, pos_gt_bboxes, ious, cfg):
        """Generate training targets for a single sample.

        Args:
            pos_bboxes (torch.Tensor): Positive boxes with shape
                (N, 7).
            pos_gt_bboxes (torch.Tensor): Ground truth boxes with shape
                (M, 7).
            ious (torch.Tensor): IoU between `pos_bboxes` and `pos_gt_bboxes`
                in shape (N, M).
            cfg (dict): Training configs.

        Returns:
            tuple[torch.Tensor]: Target for positive boxes.
                (label, bbox_targets, pos_gt_bboxes, reg_mask, label_weights,
                bbox_weights)
        """
        cls_pos_mask = ious > cfg.cls_pos_thr
        cls_neg_mask = ious < cfg.cls_neg_thr
        interval_mask = (cls_pos_mask == 0) & (cls_neg_mask == 0)

        # iou regression target
        label = (cls_pos_mask > 0).float()
        label[interval_mask] = ious[interval_mask] * 2 - 0.5
        # label weights
        label_weights = (label >= 0).float()

        # box regression target
        reg_mask = pos_bboxes.new_zeros(ious.size(0)).long()
        reg_mask[0:pos_gt_bboxes.size(0)] = 1
        bbox_weights = (reg_mask > 0).float()
        if reg_mask.bool().any():
            pos_gt_bboxes_ct = pos_gt_bboxes.clone().detach()
            roi_center = pos_bboxes[..., 0:3]
            roi_ry = pos_bboxes[..., 6] % (2 * np.pi)

            # canonical transformation
            pos_gt_bboxes_ct[..., 0:3] -= roi_center
            pos_gt_bboxes_ct[..., 6] -= roi_ry
            pos_gt_bboxes_ct[..., 0:3] = rotation_3d_in_axis(
                pos_gt_bboxes_ct[..., 0:3].unsqueeze(1), -roi_ry,
                axis=2).squeeze(1)

            # flip orientation if rois have opposite orientation
            ry_label = pos_gt_bboxes_ct[..., 6] % (2 * np.pi)  # 0 ~ 2pi
            opposite_flag = (ry_label > np.pi * 0.5) & (ry_label < np.pi * 1.5)
            ry_label[opposite_flag] = (ry_label[opposite_flag] + np.pi) % (
                2 * np.pi)  # (0 ~ pi/2, 3pi/2 ~ 2pi)
            flag = ry_label > np.pi
            ry_label[flag] = ry_label[flag] - np.pi * 2  # (-pi/2, pi/2)
            ry_label = torch.clamp(ry_label, min=-np.pi / 2, max=np.pi / 2)
            pos_gt_bboxes_ct[..., 6] = ry_label

            rois_anchor = pos_bboxes.clone().detach()
            rois_anchor[:, 0:3] = 0
            rois_anchor[:, 6] = 0
            bbox_targets = self.bbox_coder.encode(rois_anchor,
                                                  pos_gt_bboxes_ct)
        else:
            # no fg bbox
            bbox_targets = pos_gt_bboxes.new_empty((0, 7))

        return (label, bbox_targets, pos_gt_bboxes, reg_mask, label_weights,
                bbox_weights)

    def get_corner_loss_lidar(self, pred_bbox3d, gt_bbox3d, delta=1):
        """Calculate corner loss of given boxes.

        Args:
            pred_bbox3d (torch.FloatTensor): Predicted boxes in shape (N, 7).
            gt_bbox3d (torch.FloatTensor): Ground truth boxes in shape (N, 7).

        Returns:
            torch.FloatTensor: Calculated corner loss in shape (N).
        """
        assert pred_bbox3d.shape[0] == gt_bbox3d.shape[0]

        # This is a little bit hack here because we assume the box for
        # Part-A2 is in LiDAR coordinates
        gt_boxes_structure = LiDARInstance3DBoxes(gt_bbox3d)
        pred_box_corners = LiDARInstance3DBoxes(pred_bbox3d).corners
        gt_box_corners = gt_boxes_structure.corners

        # This flip only changes the heading direction of GT boxes
        gt_bbox3d_flip = gt_boxes_structure.clone()
        gt_bbox3d_flip.tensor[:, 6] += np.pi
        gt_box_corners_flip = gt_bbox3d_flip.corners

        corner_dist = torch.min(
            torch.norm(pred_box_corners - gt_box_corners, dim=2),
            torch.norm(pred_box_corners - gt_box_corners_flip,
                       dim=2))  # (N, 8)
        # huber loss
        abs_error = torch.abs(corner_dist)
        quadratic = torch.clamp(abs_error, max=delta)
        linear = (abs_error - quadratic)
        corner_loss = 0.5 * quadratic**2 + delta * linear

        return corner_loss.mean(dim=1)

    def get_results(self,
                    rois,
                    bbox_scores,
                    bbox_reg,
                    class_labels,
                    cls_preds,
                    img_metas,
                    cfg=None):
        """Generate bboxes from bbox head predictions.

        Args:
            rois (torch.Tensor): Roi bounding boxes.
            cls_score (torch.Tensor): Scores of bounding boxes.
            bbox_pred (torch.Tensor): Bounding boxes predictions
            class_labels (torch.Tensor): Label of classes
            class_pred (torch.Tensor): Score for nms.
            img_metas (list[dict]): Point cloud and image's meta info.
            cfg (:obj:`ConfigDict`): Testing config.

        Returns:
            list[tuple]: Decoded bbox, scores and labels after nms.
        """
        roi_batch_id = rois[..., 0]
        roi_boxes = rois[..., 1:]  # boxes without batch id
        batch_size = int(roi_batch_id.max().item() + 1)

        # decode boxes
        roi_ry = roi_boxes[..., 6].view(-1)
        roi_xyz = roi_boxes[..., 0:3].view(-1, 3)
        local_roi_boxes = roi_boxes.clone().detach()
        local_roi_boxes[..., 0:3] = 0
        rcnn_boxes3d = self.bbox_coder.decode(
            local_roi_boxes, bbox_reg, bottom_center=True)
        rcnn_boxes3d[..., 0:3] = rotation_3d_in_axis(
            rcnn_boxes3d[..., 0:3].unsqueeze(1), roi_ry, axis=2).squeeze(1)
        rcnn_boxes3d[:, 0:3] += roi_xyz

        # post processing
        result_list = []
        for batch_id in range(batch_size):
            cur_class_labels = class_labels[batch_id]
            cur_class_score = cls_preds[batch_id]
            cur_bbox_score = bbox_scores[roi_batch_id == batch_id]
            cur_bbox_score = cur_bbox_score.sigmoid().squeeze(dim=-1)
            cur_rcnn_boxes3d = rcnn_boxes3d[roi_batch_id == batch_id]

            cls_scores, _ = torch.max(cur_class_score, dim=-1)
            selected = self.class_agnostic_nms(
                obj_scores=cur_bbox_score,
                bbox=cur_rcnn_boxes3d,
                input_meta=img_metas[batch_id],
                cfg=cfg)
            # selected = self.multi_class_nms(cur_box_prob, cur_rcnn_boxes3d,
            #                                 cfg.score_thr, cfg.nms_thr,
            #                                 img_metas[batch_id],
            #                                 cfg.use_rotate_nms)
            selected_bboxes = cur_rcnn_boxes3d[selected]
            selected_label_preds = cur_class_labels[selected]
            selected_scores = cls_scores[selected]

            # selected_label_preds -= 1
            # selected_label_preds[selected_label_preds == -1] = 2
            results = InstanceData()
            results.bboxes_3d = img_metas[batch_id]['box_type_3d'](
                selected_bboxes, self.bbox_coder.code_size)
            results.scores_3d = selected_scores
            results.labels_3d = selected_label_preds

            result_list.append(results)
        return result_list

    def class_agnostic_nms(self, obj_scores, bbox, cfg, input_meta):
        """Class agnostic nms.

        Args:
            obj_scores (torch.Tensor): Objectness score of bounding boxes.
            sem_scores (torch.Tensor): Semantic class score of bounding boxes.
            bbox (torch.Tensor): Predicted bounding boxes.

        Returns:
            tuple[torch.Tensor]: Bounding boxes, scores and labels.
        """
        if cfg.use_rotate_nms:
            nms_func = nms_bev
        else:
            nms_func = nms_normal_bev

        bbox = input_meta['box_type_3d'](
            bbox.clone(),
            box_dim=bbox.shape[-1],
            with_yaw=True,
            origin=(0.5, 0.5, 0.5))

        if cfg.score_thr is not None:
            scores_mask = (obj_scores >= cfg.score_thr)
            obj_scores = obj_scores[scores_mask]
            bbox = bbox[scores_mask]
        selected = []
        if obj_scores.shape[0] > 0:
            box_scores_nms, indices = torch.topk(
                obj_scores, k=min(4096, obj_scores.shape[0]))
            bbox_bev = bbox.bev[indices]
            bbox_for_nms = xywhr2xyxyr(bbox_bev)

            keep = nms_func(bbox_for_nms, box_scores_nms, cfg.nms_thr)
            selected = indices[keep]
        if cfg.score_thr is not None:
            original_idxs = scores_mask.nonzero().view(-1)
            selected = original_idxs[selected]
        return selected
