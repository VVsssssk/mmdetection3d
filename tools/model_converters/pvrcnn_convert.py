import mmcv
import torch

from mmdet3d.registry import MODELS
from mmdet3d.utils import register_all_modules

source_map = [
    'neck', 'backbone', 'rpn_head', 'roi_head.semantic_head', 'middle_encoder',
    'keypoints_encoder'
]
dist_map = [
    'backbone_2d.deblocks', 'backbone_2d.blocks', 'dense_head', 'point_head',
    'backbone_3d', 'pfe'
]


def main():
    register_all_modules()
    cfg = mmcv.Config.fromfile('configs/pvrcnn/pvrcnn_kitti-3d-3class.py')
    mmdet_model = MODELS.build(cfg.model).state_dict()
    pcd_model = torch.load('./pv_rcnn_8369.pth')['model_state']
    pcd_model.pop('global_step')
    for i in range(len(dist_map)):
        source_key = []
        dict_key = []
        for k in mmdet_model.keys():
            if source_map[i] in k:
                source_key.append(k)
        for k in pcd_model.keys():
            if dist_map[i] in k:
                dict_key.append(k)
        assert len(source_key) == len(dict_key)
        for j in range(len(source_key)):
            mmdet_model[source_key[j]] = pcd_model[dict_key[j]]
    source_key = []
    dict_key = []
    for k in mmdet_model.keys():
        if 'roi_head.bbox_roi_extractor' in k:
            source_key.append(k)
    for k in pcd_model.keys():
        if 'roi_head.roi_grid_pool_layer' in k:
            dict_key.append(k)
    assert len(source_key) == len(dict_key)
    for j in range(len(source_key)):
        mmdet_model[source_key[j]] = pcd_model[dict_key[j]]

    source_key = []
    dict_key = []
    for k in mmdet_model.keys():
        if 'roi_head.bbox_head' in k:
            source_key.append(k)
    for k in pcd_model.keys():
        if 'roi_head' in k and 'roi_head.roi_grid_pool_layer' not in k:
            dict_key.append(k)
    assert len(source_key) == len(dict_key)
    for j in range(len(source_key)):
        mmdet_model[source_key[j]] = pcd_model[dict_key[j]]

    source_key = []
    dict_key = []
    for k in mmdet_model.keys():
        if 'keypoints_encoder.rawpoints_sa_layer' in k:
            source_key.append(k)
    for k in pcd_model.keys():
        if 'pfe.SA_rawpoints' in k:
            dict_key.append(k)
    assert len(source_key) == len(dict_key)
    for j in range(len(source_key)):
        mmdet_model[source_key[j]] = pcd_model[dict_key[j]]

    source_key = []
    dict_key = []
    for k in mmdet_model.keys():
        if 'keypoints_encoder.voxel_sa_layers' in k:
            source_key.append(k)
    for k in pcd_model.keys():
        if 'pfe.SA_layers' in k:
            dict_key.append(k)
    assert len(source_key) == len(dict_key)
    for j in range(len(source_key)):
        mmdet_model[source_key[j]] = pcd_model[dict_key[j]]

    torch.save(mmdet_model, 'new_pvrcnn.pth')


if __name__ == '__main__':
    main()
