import torch.nn as nn
from typing import List
from mmcv.cnn.bricks import build_norm_layer, is_norm
import torch
from torch.autograd import Function
from . import vsa_utils
from mmcv.cnn import ConvModule

class BallQuery(Function):

    @staticmethod
    def forward(ctx, radius: float, nsample: int, xyz: torch.Tensor,
                xyz_batch_cnt: torch.Tensor,
                new_xyz: torch.Tensor, new_xyz_batch_cnt):
        """
        Args:
            ctx:
            radius: float, radius of the balls
            nsample: int, maximum number of features in the balls
            xyz: (N1 + N2 ..., 3) xyz coordinates of the features
            xyz_batch_cnt: (batch_size), [N1, N2, ...]
            new_xyz: (M1 + M2 ..., 3) centers of the ball query
            new_xyz_batch_cnt: (batch_size), [M1, M2, ...]

        Returns:
            idx: (M1 + M2, nsample) tensor with the indicies of the features that form the query balls
        """
        idx = new_xyz.new_zeros((new_xyz.shape[0], nsample), dtype=torch.int32)

        vsa_utils.ball_query(radius, nsample, new_xyz, new_xyz_batch_cnt, xyz,
                             xyz_batch_cnt, idx)
        empty_ball_mask = (idx[:, 0] == -1)
        idx[empty_ball_mask] = 0
        return idx, empty_ball_mask

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


ball_query = BallQuery.apply


class GroupingOperation(Function):

    @staticmethod
    def forward(ctx, features: torch.Tensor, features_batch_cnt: torch.Tensor,
                idx: torch.Tensor, idx_batch_cnt: torch.Tensor):
        """
        Args:
            ctx:
            features: (N1 + N2 ..., C) tensor of features to group
            features_batch_cnt: (batch_size) [N1 + N2 ...] tensor containing the indicies of features to group with
            idx: (M1 + M2 ..., nsample) tensor containing the indicies of features to group with
            idx_batch_cnt: (batch_size) [M1 + M2 ...] tensor containing the indicies of features to group with

        Returns:
            output: (M1 + M2, C, nsample) tensor
        """
        assert features.shape[0] == features_batch_cnt.sum(), \
            'features: %s, features_batch_cnt: %s' % (
                str(features.shape), str(features_batch_cnt))
        assert idx.shape[0] == idx_batch_cnt.sum(), \
            'idx: %s, idx_batch_cnt: %s' % (str(idx.shape), str(idx_batch_cnt))

        M, nsample = idx.size()
        N, C = features.size()
        output = features.new_empty((M, C, nsample))

        vsa_utils.group_points(features, features_batch_cnt, idx, idx_batch_cnt,
                               output)

        ctx.for_backwards = (N, idx, features_batch_cnt, idx_batch_cnt)
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        """
        Args:
            ctx:
            grad_out: (M1 + M2 ..., C, nsample) tensor of the gradients of the output from forward

        Returns:
            grad_features: (N1 + N2 ..., C) gradient of the features
        """
        grad_out = grad_out.contiguous()
        N, idx, features_batch_cnt, idx_batch_cnt = ctx.for_backwards

        M, C, nsample = grad_out.size()
        grad_features = grad_out.new_zeros((N, C))

        vsa_utils.group_points_grad(grad_out, idx, idx_batch_cnt,
                                    features_batch_cnt, grad_features)
        return grad_features, None, None, None

grouping = GroupingOperation.apply

class QueryAndGroup(nn.Module):
    def __init__(self, radius: float, nsample: int, use_xyz: bool = True,
                 debug: bool = False):
        """
        Args:
            radius: float, radius of ball
            nsample: int, maximum number of features to gather in the ball
            use_xyz:
        """
        super().__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz
        self.debug = debug

    def forward(self, xyz: torch.Tensor, xyz_batch_cnt: torch.Tensor,
                new_xyz: torch.Tensor, new_xyz_batch_cnt: torch.Tensor,
                features: torch.Tensor = None):
        """
        Args:
            xyz: (N1 + N2 ..., 3) xyz coordinates of the features
            xyz_batch_cnt: (batch_size), [N1, N2, ...]
            new_xyz: (M1 + M2 ..., 3) centers of the ball query
            new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
            features: (N1 + N2 ..., C) tensor of features to group

        Returns:
            new_features: (M1 + M2, C, nsample) tensor
        """
        assert xyz.shape[
                   0] == xyz_batch_cnt.sum(), 'xyz: %s, xyz_batch_cnt: %s' % (
            str(xyz.shape), str(new_xyz_batch_cnt))
        assert new_xyz.shape[0] == new_xyz_batch_cnt.sum(), \
            'new_xyz: %s, new_xyz_batch_cnt: %s' % (
                str(new_xyz.shape), str(new_xyz_batch_cnt))

        # idx: (M1 + M2 ..., nsample), empty_ball_mask: (M1 + M2 ...)
        idx, empty_ball_mask = ball_query(self.radius, self.nsample, xyz,
                                          xyz_batch_cnt, new_xyz,
                                          new_xyz_batch_cnt)
        grouped_xyz = grouping(xyz, xyz_batch_cnt, idx,
                               new_xyz_batch_cnt)  # (M1 + M2, 3, nsample)
        if self.debug:
            import open3d as o3d
            import numpy as np
            kpts = new_xyz.cpu().detach().numpy().astype(np.float64)
            pts = grouped_xyz.cpu().detach().numpy().astype(np.float64)

            old = 0
            for cnt in new_xyz_batch_cnt:
                b_kpts = kpts[old: old + cnt]
                b_pts = pts[old: old + cnt].transpose(0, 2, 1).reshape(-1, 3)
                old += cnt

                kpc = o3d.geometry.PointCloud(
                    points=o3d.utility.Vector3dVector(b_kpts))
                pc = o3d.geometry.PointCloud(
                    points=o3d.utility.Vector3dVector(b_pts))

                kcolor = np.zeros_like(b_kpts)
                kcolor[..., 2] = 1
                kcolor = o3d.utility.Vector3dVector(kcolor)
                kpc.colors = kcolor

                color = np.zeros_like(b_pts)
                color[..., 0] = 1
                color = o3d.utility.Vector3dVector(color)
                pc.colors = color

                o3d.visualization.draw_geometries([pc, kpc])

        grouped_xyz -= new_xyz.unsqueeze(-1)

        grouped_xyz[empty_ball_mask] = 0

        if features is not None:
            grouped_features = grouping(features, xyz_batch_cnt, idx,
                                        new_xyz_batch_cnt)  # (M1 + M2, C, nsample)
            grouped_features[empty_ball_mask] = 0
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features],
                                         dim=1)  # (M1 + M2 ..., C + 3, nsample)
            else:
                new_features = grouped_features
        else:
            assert self.use_xyz, "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        return new_features, idx



class GuidedSAModuleMSG(nn.Module):

    def __init__(self, in_channels: int, radii: List[float],
                 nsamples: List[int], mlps: List[List[int]],
                 use_xyz: bool = True, pool_method='max',
                 norm_cfg: dict = dict(type='BN2d', eps=1e-5, momentum=0.01)):
        """
        Args:
            radii: list of float, list of radii to group with
            nsamples: list of int, number of samples in each ball query
            mlps: list of list of int, spec of the pointnet before the global pooling for each scale
            use_xyz:
            pool_method: max / avg
        """
        super().__init__()
        assert len(radii) == len(nsamples) == len(mlps)

        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            cin = in_channels
            if use_xyz:
                cin += 3
            radius = radii[i]
            nsample = nsamples[i]
            mlp_spec = mlps[i]

            self.groupers.append(
                QueryAndGroup(radius, nsample, use_xyz=use_xyz))

            mlp = nn.Sequential()
            for i in range(len(mlp_spec)):
                cout = mlp_spec[i]
                mlp.add_module(
                    f'layer{i}',
                    ConvModule(
                        cin,
                        cout,
                        kernel_size=(1, 1),
                        stride=(1, 1),
                        conv_cfg=dict(type='Conv2d'),
                        norm_cfg=norm_cfg,
                        bias=False))
                cin = cout
            self.mlps.append(mlp)
        self.pool_method = pool_method
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if is_norm(m):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt,
                features=None, empty_voxel_set_zeros=True):
        """
        :param xyz: (N1 + N2 ..., 3) tensor of the xyz coordinates of the features
        :param xyz_batch_cnt: (batch_size), [N1, N2, ...]
        :param new_xyz: (M1 + M2 ..., 3)
        :param new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
        :param features: (N1 + N2 ..., C) tensor of the descriptors of the the features
        :return:
            new_xyz: (M1 + M2 ..., 3) tensor of the new features' xyz
            new_features: (M1 + M2 ..., \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        new_features_list = []
        for k in range(len(self.groupers)):
            new_features, ball_idxs = self.groupers[k](
                xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt, features
            )  # (M1 + M2, Cin, nsample)
            new_features = new_features.permute(1, 0, 2).unsqueeze(dim=0)
            new_features = self.mlps[k](new_features)
            # (M1 + M2 ..., Cout, nsample)

            if self.pool_method == 'max':
                new_features = new_features.max(-1).values
            elif self.pool_method == 'avg':
                new_features = new_features.mean(-1)
            else:
                raise NotImplementedError
            new_features = new_features.squeeze(dim=0).permute(1, 0)
            new_features_list.append(new_features)

        new_features = torch.cat(new_features_list, dim=1)  # (M1 + M2 ..., C)

        return new_xyz, new_features
