_base_ = [
    '../_base_/datasets/waymoD5-3d-3class.py',
    '../_base_/schedules/cyclic-40e.py', '../_base_/default_runtime.py'
]

voxel_size = [0.1, 0.1, 0.15]
point_cloud_range = [-75.2, -75.2, -2, 75.2, 75.2, 4]

data_root = 'data/waymo/kitti_format/'
class_names = ['Car', 'Pedestrian', 'Cyclist']
metainfo = dict(CLASSES=class_names)
db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'waymo_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(Car=5, Pedestrian=5, Cyclist=5)),
    classes=class_names,
    sample_groups=dict(Car=15, Pedestrian=10, Cyclist=10),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4]))

train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=6, use_dim=5),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ObjectSample', db_sampler=db_sampler),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=6, use_dim=5),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range)
        ]),
    dict(type='Pack3DDetInputs', keys=['points'])
]

model = dict(
    type='PointVoxelRCNN',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=5,  # max_points_per_voxel
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=(150000, 150000))),
    voxel_encoder=dict(type='HardSimpleVFE', num_features=5),
    middle_encoder=dict(
        type='SparseEncoder',
        in_channels=5,
        sparse_shape=[41, 1504, 1504],
        order=('conv', 'norm', 'act'),
        encoder_paddings=((0, 0, 0), ((1, 1, 1), 0, 0), ((1, 1, 1), 0, 0),
                          ((0, 1, 1), 0, 0)),
        return_middle_feats=True),
    backbone=dict(
        type='SECOND',
        in_channels=256,
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        out_channels=[128, 256]),
    neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        upsample_strides=[1, 2],
        out_channels=[256, 256]),
    rpn_head=dict(
        type='CenterHead',
        in_channels=sum([256, 256]),
        tasks=[
            dict(num_class=3, class_names=class_names),
        ],
        common_heads=dict(reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            post_center_range=[-75.2, -75.2, -2, 75.2, 75.2, 4],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            pc_range=point_cloud_range[:2],
            code_size=7),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(
            type='mmdet.GaussianFocalLoss', reduction='mean', loss_weight=1),
        loss_bbox=dict(type='mmdet.L1Loss', reduction='mean', loss_weight=2),
        norm_bbox=True),
    points_encoder=dict(
        type='VoxelSetAbstraction',
        keypoints_sampler=dict(
            type='SPCSampler',
            num_keypoints=4096,
            num_sectors=6,
            sample_radius_with_roi=1.6),
        fused_out_channel=90,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        bev_feat_channel=256,
        bev_scale_factor=8,
        voxel_sa_cfgs_list=[
            dict(
                type='VectorPoolAggregationModuleMSG',
                source_feats_index=2,
                scale_factor=4,
                in_channels=64,
                mlp_channels=[[128]],
                local_aggregation_type='local_interpolation',
                num_channels_of_local_aggregation=32,
                num_reduced_channels=32,
                filter_neighbor_with_roi=True,
                radius_of_neighbor_with_roi=4.0,
                groups_cfg_list=[
                    dict(
                        num_local_voxel=[3, 3, 3],
                        post_mlps=[64, 64],
                        max_neighbor_distance=1.2,
                        neighbor_nsample=-1),
                    dict(
                        num_local_voxel=[3, 3, 3],
                        post_mlps=[64, 64],
                        max_neighbor_distance=2.4,
                        neighbor_nsample=-1),
                ]),
            dict(
                type='VectorPoolAggregationModuleMSG',
                source_feats_index=3,
                scale_factor=8,
                in_channels=64,
                mlp_channels=[[128]],
                local_aggregation_type='local_interpolation',
                num_channels_of_local_aggregation=32,
                num_reduced_channels=32,
                filter_neighbor_with_roi=True,
                radius_of_neighbor_with_roi=6.4,
                groups_cfg_list=[
                    dict(
                        num_local_voxel=[3, 3, 3],
                        post_mlps=[64, 64],
                        max_neighbor_distance=2.4,
                        neighbor_nsample=-1),
                    dict(
                        num_local_voxel=[3, 3, 3],
                        post_mlps=[64, 64],
                        max_neighbor_distance=4.8,
                        neighbor_nsample=-1),
                ]),
        ],
        rawpoints_sa_cfgs=dict(
            type='VectorPoolAggregationModuleMSG',
            in_channels=2,
            mlp_channels=[[32]],
            local_aggregation_type='local_interpolation',
            num_channels_of_local_aggregation=32,
            filter_neighbor_with_roi=True,
            radius_of_neighbor_with_roi=2.4,
            groups_cfg_list=[
                dict(
                    num_local_voxel=[2, 2, 2],
                    post_mlps=[32, 32],
                    max_neighbor_distance=0.2,
                    neighbor_nsample=-1),
                dict(
                    num_local_voxel=[3, 3, 3],
                    post_mlps=[32, 32],
                    max_neighbor_distance=0.4,
                    neighbor_nsample=-1),
            ])),
    roi_head=dict(
        type='PVRCNNRoiHead',
        num_classes=3,
        semantic_head=dict(
            type='ForegroundSegmentationHead',
            in_channels=544,
            extra_width=0.1,
            loss_seg=dict(
                type='mmdet.FocalLoss',
                use_sigmoid=True,
                reduction='sum',
                gamma=2.0,
                alpha=0.25,
                activated=True,
                loss_weight=1.0)),
        bbox_roi_extractor=dict(
            type='Batch3DRoIGridExtractor',
            grid_size=6,
            roi_layer=dict(
                type='VectorPoolAggregationModuleMSG',
                in_channels=90,
                mlp_channels=[[128]],
                local_aggregation_type='voxel_random_choice',
                # local_aggregation_type='local_interpolation',
                num_channels_of_local_aggregation=32,
                num_reduced_channels=30,
                groups_cfg_list=[
                    dict(
                        num_local_voxel=[3, 3, 3],
                        post_mlps=[64, 64],
                        max_neighbor_distance=0.8,
                        neighbor_nsample=32),
                    dict(
                        num_local_voxel=[3, 3, 3],
                        post_mlps=[64, 64],
                        max_neighbor_distance=1.6,
                        neighbor_nsample=32),
                ])),
        bbox_head=dict(
            type='PVRCNNBBoxHead',
            in_channels=128,
            grid_size=6,
            num_classes=3,
            class_agnostic=True,
            shared_fc_channels=(256, 256),
            reg_channels=(256, 256),
            cls_channels=(256, 256),
            dropout_ratio=0.3,
            with_corner_loss=True,
            bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
            loss_bbox=dict(
                type='mmdet.SmoothL1Loss',
                beta=1.0 / 9.0,
                reduction='sum',
                loss_weight=1.0),
            loss_cls=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=True,
                reduction='sum',
                loss_weight=1.0))),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            grid_size=[1504, 1504, 40],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_size_factor=8,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        rpn_proposal=dict(
            post_center_limit_range=[-75.2, -75.2, -2, 75.2, 75.2, 4],
            pc_range=point_cloud_range[:2],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            nms_type='rotate',
            pre_max_size=9000,
            post_max_size=512,
            nms_thr=0.8),
        rcnn=dict(
            assigner=[
                dict(  # for Pedestrian
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(
                        type='BboxOverlaps3D', coordinate='lidar'),
                    pos_iou_thr=0.55,
                    neg_iou_thr=0.55,
                    min_pos_iou=0.55,
                    ignore_iof_thr=-1),
                dict(  # for Cyclist
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(
                        type='BboxOverlaps3D', coordinate='lidar'),
                    pos_iou_thr=0.55,
                    neg_iou_thr=0.55,
                    min_pos_iou=0.55,
                    ignore_iof_thr=-1),
                dict(  # for Car
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(
                        type='BboxOverlaps3D', coordinate='lidar'),
                    pos_iou_thr=0.55,
                    neg_iou_thr=0.55,
                    min_pos_iou=0.55,
                    ignore_iof_thr=-1)
            ],
            sampler=dict(
                type='IoUNegPiecewiseSampler',
                num=128,
                pos_fraction=0.5,
                neg_piece_fractions=[0.8, 0.2],
                neg_iou_piece_thrs=[0.55, 0.1],
                neg_pos_ub=-1,
                add_gt_as_proposals=False,
                return_iou=True),
            cls_pos_thr=0.75,
            cls_neg_thr=0.25)),
    test_cfg=dict(
        rpn=dict(
            post_center_limit_range=[-75.2, -75.2, -2, 75.2, 75.2, 4],
            pc_range=point_cloud_range[:2],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],  # ?
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            nms_type='rotate',
            pre_max_size=1024,
            post_max_size=100,
            nms_thr=0.7),
        rcnn=dict(
            use_rotate_nms=True,
            use_raw_score=True,
            nms_thr=0.7,
            score_thr=0.1)))
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(dataset=dict(pipeline=train_pipeline, metainfo=metainfo)))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline, metainfo=metainfo))
eval_dataloader = dict(dataset=dict(pipeline=test_pipeline, metainfo=metainfo))
lr = 0.001
optim_wrapper = dict(optimizer=dict(lr=lr))
train_cfg = dict(by_epoch=True, max_epochs=15, val_interval=1)
param_scheduler = [
    # learning rate scheduler
    # During the first 16 epochs, learning rate increases from 0 to lr * 10
    # during the next 24 epochs, learning rate decreases from lr * 10 to
    # lr * 1e-4
    dict(
        type='CosineAnnealingLR',
        T_max=6,
        eta_min=lr * 10,
        begin=0,
        end=6,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=9,
        eta_min=lr * 1e-4,
        begin=6,
        end=15,
        by_epoch=True,
        convert_to_iter_based=True),
    # momentum scheduler
    # During the first 16 epochs, momentum increases from 0 to 0.85 / 0.95
    # during the next 24 epochs, momentum increases from 0.85 / 0.95 to 1
    dict(
        type='CosineAnnealingMomentum',
        T_max=6,
        eta_min=0.85 / 0.95,
        begin=0,
        end=6,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=9,
        eta_min=1,
        begin=6,
        end=15,
        by_epoch=True,
        convert_to_iter_based=True)
]
