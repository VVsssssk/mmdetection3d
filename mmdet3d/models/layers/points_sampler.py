# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import List

import numpy as np
import torch
from mmcv.ops.furthest_point_sample import furthest_point_sample
from torch import Tensor
from torch import nn as nn

from mmdet3d.registry import MODELS


@MODELS.register_module()
class FPSSampler(nn.Module):
    """Using Euclidean distances of points for FPS."""

    def __init__(self, num_keypoints) -> None:
        super().__init__()
        self.num_keypoints = num_keypoints

    def forward(self, points_list: List[Tensor], **kwargs) -> Tensor:
        """Sampling points with D-FPS."""
        sampled_points = []
        for batch_idx in range(len(points_list)):
            points = points_list[batch_idx]
            num_points = points.shape[0]
            fps_idx = furthest_point_sample(
                points.unsqueeze(dim=0).contiguous(),
                self.num_keypoints).long()[0]
            if num_points < self.num_keypoints:
                times = int(self.num_keypoints / num_points) + 1
                non_empty = fps_idx[:num_points]
                fps_idx = non_empty.repeat(times)[:self.num_keypoints]
            key_points = points[fps_idx]
            sampled_points.append(key_points)
        return sampled_points


@MODELS.register_module()
class SPCSampler(nn.Module):
    """Using Euclidean distances of points for FPS."""

    def __init__(
        self,
        num_keypoints,
        sample_radius_with_roi,
        num_sectors,
        num_max_points_of_part=200000,
    ) -> None:
        super().__init__()
        self.num_keypoints = num_keypoints
        self.sample_radius_with_roi = sample_radius_with_roi
        self.num_max_points_of_part = num_max_points_of_part
        self.num_sectors = num_sectors

    def sample_points_with_roi(self, rois, points):
        """
        Args:
            rois: (M, 7 + C)
            points: (N, 3)
            sample_radius_with_roi:
            num_max_points_of_part:

        Returns:
            sampled_points: (N_out, 3)
        """
        if points.shape[0] < self.num_max_points_of_part:
            distance = (points[:, None, :] - rois[None, :, 0:3]).norm(dim=-1)
            min_dis, min_dis_roi_idx = distance.min(dim=-1)
            roi_max_dim = (rois[min_dis_roi_idx, 3:6] / 2).norm(dim=-1)
            point_mask = min_dis < roi_max_dim + self.sample_radius_with_roi
        else:
            point_mask_list = []
            for start_idx in range(len(points.shape[0])):
                distance = (points[start_idx:start_idx +
                                   self.num_max_points_of_part, None, :] -
                            rois[None, :, 0:3]).norm(dim=-1)
                min_dis, min_dis_roi_idx = distance.min(dim=-1)
                roi_max_dim = (rois[min_dis_roi_idx, 3:6] / 2).norm(dim=-1)
                cur_point_mask = min_dis < roi_max_dim + self.sample_radius_with_roi
                point_mask_list.append(cur_point_mask)
                start_idx += self.num_max_points_of_part
            point_mask = torch.cat(point_mask_list, dim=0)

        sampled_points = points[:1] if point_mask.sum() == 0 else points[
            point_mask, :]

        return sampled_points, point_mask

    def sector_fps(self, points):
        """
        Args:
            points: (N, 3)
            num_sampled_points: int
            num_sectors: int

        Returns:
            sampled_points: (N_out, 3)
        """
        sector_size = np.pi * 2 / self.num_sectors
        point_angles = torch.atan2(points[:, 1], points[:, 0]) + np.pi
        sector_idx = (point_angles / sector_size).floor().clamp(
            min=0, max=self.num_sectors)
        xyz_points_list = []
        xyz_batch_cnt = []
        num_sampled_points_list = []
        for k in range(self.num_sectors):
            mask = (sector_idx == k)
            cur_num_points = mask.sum().item()
            if cur_num_points > 0:
                xyz_points_list.append(points[mask])
                xyz_batch_cnt.append(cur_num_points)
                ratio = cur_num_points / points.shape[0]
                num_sampled_points_list.append(
                    min(cur_num_points, math.ceil(ratio * self.num_keypoints)))

        if len(xyz_batch_cnt) == 0:
            xyz_points_list.append(points)
            xyz_batch_cnt.append(len(points))
            num_sampled_points_list.append(self.num_keypoints)
            print(
                f'Warning: empty sector points detected in SectorFPS: points.shape={points.shape}'
            )

        xyz = torch.cat(xyz_points_list, dim=0)
        xyz_batch_cnt = torch.tensor(xyz_batch_cnt, device=points.device).int()
        sampled_points_batch_cnt = torch.tensor(
            num_sampled_points_list, device=points.device).int()

        sampled_pt_idxs = furthest_point_sample(xyz.contiguous(),
                                                sampled_points_batch_cnt,
                                                xyz_batch_cnt).long()

        sampled_points = xyz[sampled_pt_idxs]

        return sampled_points

    def sectorized_proposal_centric_sampling(self, roi_boxes, points):
        sampled_points, _ = self.sample_points_with_roi(
            rois=roi_boxes, points=points)
        sampled_points = self.sector_fps(points=sampled_points)
        return sampled_points

    def forward(self, points_list: List[Tensor], rpn_results_list) -> Tensor:
        """Sampling points with D-FPS."""
        sampled_points = []
        roi_boxes_list = []
        for proposal in rpn_results_list:
            roi_boxes = proposal.bboxes_3d.tensor.clone()
            roi_boxes[:, 2] = roi_boxes[:, 2] + roi_boxes[:, 5] / 2
            roi_boxes_list.append(roi_boxes)
        for batch_idx in range(len(points_list)):
            points = points_list[batch_idx]
            cur_keypoints = self.sectorized_proposal_centric_sampling(
                roi_boxes=roi_boxes_list[batch_idx], points=points)
            # bs_idxs = cur_keypoints.new_ones(cur_keypoints.shape[0]) * batch_idx
            # keypoints = torch.cat((bs_idxs[:, None], cur_keypoints), dim=1)
            sampled_points.append(cur_keypoints)
        return sampled_points
