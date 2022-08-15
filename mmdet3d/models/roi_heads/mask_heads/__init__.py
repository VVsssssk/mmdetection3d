# Copyright (c) OpenMMLab. All rights reserved.
from .pointwise_mask_head import PointwiseMaskHead
from .pointwise_semantic_head import PointwiseSemanticHead
from .primitive_head import PrimitiveHead

__all__ = ['PointwiseSemanticHead', 'PrimitiveHead', 'PointwiseMaskHead']
