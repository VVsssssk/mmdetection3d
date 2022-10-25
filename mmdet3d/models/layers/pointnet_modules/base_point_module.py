# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.ops import PointsSampler as Points_Sampler
from torch import nn as nn

from mmdet3d.registry import MODELS


@MODELS.register_module()
class MLPEncodeLayer(nn.Module):


@MODELS.register_module()
class StackSAModule(nn.Module):
    def __init__(self,
                 radii,
                 sample_nums,
                 encode_layer=None,
                 points_sampler=None):
        super.__init__()

        assert len(radii) == len(sample_nums)

        self.encode_layer = MODELS.build(encode_layer)

        if points_sampler is not None:
            self.points_sampler = Points_Sampler(points_sampler)
        else:
            self.points_sampler = None
