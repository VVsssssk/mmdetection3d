import unittest

import torch
from mmengine import DefaultScope

from mmdet3d.registry import MODELS
from tests.utils.model_utils import (_create_detector_inputs,
                                     _get_detector_cfg, _setup_seed)


class TestPVRCNN(unittest.TestCase):

    def test_pvrcnn(self):
        import mmdet3d.models

        assert hasattr(mmdet3d.models, 'PointVoxelRCNN')
        DefaultScope.get_instance('test_pvrcnn_plusplus', scope_name='mmdet3d')
        _setup_seed(0)
        pvrcnn_cfg = _get_detector_cfg(
            'pvrcnn/pvrcnn_plusplus_8xb2-80e_waymo-3class.py')
        model = MODELS.build(pvrcnn_cfg)
        num_gt_instance = 10
        packed_inputs = _create_detector_inputs(
            num_points=9000, points_feat_dim=5, num_gt_instance=num_gt_instance, gt_bboxes_dim=7)

        # TODO: Support aug data test
        # aug_packed_inputs = [
        #     _create_detector_inputs(num_gt_instance=num_gt_instance),
        #     _create_detector_inputs(num_gt_instance=num_gt_instance + 1)
        # ]
        # test_aug_test
        # metainfo = {
        #     'pcd_scale_factor': 1,
        #     'pcd_horizontal_flip': 1,
        #     'pcd_vertical_flip': 1,
        #     'box_type_3d': LiDARInstance3DBoxes
        # }
        # for item in aug_packed_inputs:
        #     for batch_id in len(item['data_samples']):
        #         item['data_samples'][batch_id].set_metainfo(metainfo)

        if torch.cuda.is_available():
            model = model.cuda()
            # test simple_test
            with torch.no_grad():
                data = model.data_preprocessor(packed_inputs, True)
                torch.cuda.empty_cache()
                results = model.forward(**data, mode='predict')
            self.assertEqual(len(results), 1)
            self.assertIn('bboxes_3d', results[0].pred_instances_3d)
            self.assertIn('scores_3d', results[0].pred_instances_3d)
            self.assertIn('labels_3d', results[0].pred_instances_3d)

            # save the memory
            with torch.no_grad():
                losses = model.forward(**data, mode='loss')
                torch.cuda.empty_cache()
            self.assertGreater(losses['rpn_task0.loss_heatmap'], 0)
            self.assertGreater(losses['rpn_task0.loss_bbox'], 0)
            self.assertGreater(losses['loss_semantic'], 0)
            self.assertGreaterEqual(losses['loss_bbox'], 0)
            self.assertGreaterEqual(losses['loss_cls'], 0)
            self.assertGreaterEqual(losses['loss_corner'], 0)
