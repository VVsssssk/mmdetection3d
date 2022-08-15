# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from mmcv.ops import Voxelization

from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList
from .two_stage import TwoStage3DDetector


@MODELS.register_module()
class PointVoxelRCNN(TwoStage3DDetector):

    def __init__(self,
                 voxel_layer: dict,
                 voxel_encoder: dict,
                 middle_encoder: dict,
                 backbone: dict,
                 neck: dict = None,
                 rpn_head: dict = None,
                 keypoints_encoder: dict = None,
                 roi_head: dict = None,
                 train_cfg: dict = None,
                 test_cfg: dict = None,
                 init_cfg: dict = None,
                 data_preprocessor: Optional[dict] = None):
        super().__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor)
        self.voxel_layer = Voxelization(**voxel_layer)
        self.voxel_encoder = MODELS.build(voxel_encoder)
        self.middle_encoder = MODELS.build(middle_encoder)
        self.keypoints_encoder = MODELS.build(keypoints_encoder)

    def predict(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
                **kwargs) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points', 'imgs' keys.

                    - points (list[torch.Tensor]): Point cloud of each sample.
                    - imgs (torch.Tensor, optional): Image of each sample.

            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                samples. It usually includes information such as
                `gt_instance_3d`, `gt_panoptic_seg_3d` and `gt_sem_seg_3d`.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input samples. Each Det3DDataSample usually contain
            'pred_instances_3d'. And the ``pred_instances_3d`` usually
            contains following keys.

                - scores_3d (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels_3d (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes_3d (Tensor): Contains a tensor with shape
                    (num_instances, C) where C >=7.
        """
        feats_dict = self.extract_feat(batch_inputs_dict)

        if self.with_rpn:
            rpn_results_list = self.rpn_head.predict(feats_dict,
                                                     batch_data_samples)
        else:
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        points_feats_dict = self.extract_points_feat(batch_inputs_dict,
                                                     feats_dict,
                                                     rpn_results_list)

        results_list = self.roi_head.predict(points_feats_dict,
                                             rpn_results_list,
                                             batch_data_samples)

        # connvert to Det3DDataSample
        results_list = self.convert_to_datasample(results_list)

        return results_list

    @torch.no_grad()
    def voxelize(self, points):
        """Apply hard voxelization to points."""
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return dict(voxels=voxels, num_points=num_points, coors=coors_batch)

    def extract_feat(self, batch_inputs_dict: Dict):
        """Extract features from points."""
        feats_dict = dict()
        voxel_dict = self.voxelize(batch_inputs_dict['points'])
        # import torch
        # voxel_dict = torch.load('voxel_dict.pkl')
        # voxel_dict['coors'] = voxel_dict['coors'].int()
        # voxel_dict['num_points'] = voxel_dict['num_points'].int()
        feats_dict['voxel_dict'] = voxel_dict
        voxel_features = self.voxel_encoder(voxel_dict['voxels'],
                                            voxel_dict['num_points'],
                                            voxel_dict['coors'])
        batch_size = voxel_dict['coors'][-1, 0].item() + 1
        feats_dict['spatial_feats'], feats_dict[
            'multi_scale_3d_feats'] = self.middle_encoder(
                voxel_features, voxel_dict['coors'], batch_size)
        x = self.backbone(feats_dict['spatial_feats'])
        if self.with_neck:
            neck_feats = self.neck(x)
            feats_dict['neck_feats'] = neck_feats
        return feats_dict

    def extract_points_feat(self, batch_inputs_dict, feats_dict,
                            rpn_results_list):
        return self.keypoints_encoder(batch_inputs_dict, feats_dict,
                                      rpn_results_list)

    def loss(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
             **kwargs):
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points', 'imgs' keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
                - imgs (torch.Tensor, optional): Image of each sample.

            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                samples. It usually includes information such as
                `gt_instance_3d`, `gt_panoptic_seg_3d` and `gt_sem_seg_3d`.

        Returns:
            dict: A dictionary of loss components.
        """
        feats_dict = self.extract_feat(batch_inputs_dict)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_data_samples = copy.deepcopy(batch_data_samples)

            rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
                feats_dict,
                rpn_data_samples,
                proposal_cfg=proposal_cfg,
                **kwargs)
            # avoid get same name with roi_head loss
            keys = rpn_losses.keys()
            for key in keys:
                if 'loss' in key and 'rpn' not in key:
                    rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
            losses.update(rpn_losses)
        else:
            # TODO: Not support currently, should have a check at Fast R-CNN
            assert batch_data_samples[0].get('proposals', None) is not None
            # use pre-defined proposals in InstanceData for the second stage
            # to extract ROI features.
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        points_feats_dict = self.extract_points_feat(batch_inputs_dict,
                                                     feats_dict,
                                                     rpn_results_list)

        roi_losses = self.roi_head.loss(points_feats_dict, rpn_results_list,
                                        batch_data_samples)
        losses.update(roi_losses)

        return losses
