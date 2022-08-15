voxel_size = [0.05, 0.05, 0.1]
point_cloud_range = [0, -40, -3, 70.4, 40, 1]
model = dict(
    type='PointVoxelRCNN',
    data_preprocessor=dict(type='Det3DDataPreprocessor'),
    voxel_layer=dict(
        max_num_points=5,
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        max_voxels=(16000, 40000),
        deterministic=False),
    voxel_encoder=dict(type='HardSimpleVFE'),
    voxel_middle_encoder=dict(
        type='SparseEncoder',
        in_channels=4,
        sparse_shape=[41, 1600, 1408],
        order=('conv', 'norm', 'act'),
        mlvl_outputs=True),
    keypoints_encoder=dict(
        type='VoxelSetAbstraction',
        num_keypoints=2048,
        out_channels=128,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        voxel_sa_configs=[
            dict(
                scale_factor=1,
                in_channels=16,
                pool_radius=(0.4, 0.8),
                samples=(16, 16),
                mlps=((16, 16), (16, 16))),
            dict(
                scale_factor=2,
                in_channels=32,
                pool_radius=(0.8, 1.2),
                samples=(16, 32),
                mlps=((32, 32), (32, 32))),
            dict(
                scale_factor=4,
                in_channels=64,
                pool_radius=(1.2, 2.4),
                samples=(16, 32),
                mlps=((64, 64), (64, 64))),
            dict(
                scale_factor=8,
                in_channels=64,
                pool_radius=(2.4, 4.8),
                samples=(16, 32),
                mlps=((64, 64), (64, 64)))
        ],
        rawpoint_sa_config=dict(
            in_channels=1,
            pool_radius=(0.4, 0.8),
            samples=(16, 16),
            mlps=((16, 16), (16, 16))),
        bev_sa_config=dict(scale_factor=8, in_channels=256),
    ),
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
        type='PartA2RPNHead',
        num_classes=3,
        in_channels=512,
        feat_channels=512,
        use_direction_classifier=True,
        anchor_generator=dict(
            type='Anchor3DRangeGenerator',
            ranges=[[0.2, -39.8, -0.6, 70.2, 39.8, -0.6],
                    [0.2, -39.8, -0.6, 70.2, 39.8, -0.6],
                    [0.2, -39.8, -1.78, 70.2, 39.8, -1.78]],
            sizes=[[0.8, 0.6, 1.73], [1.76, 0.6, 1.73], [3.9, 1.6, 1.56]],
            rotations=[0, 1.57],
            reshape_out=False),
        diff_rad_by_sin=True,
        assigner_per_size=True,
        assign_per_class=True,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2)),
    roi_head=dict(
        type='PVRCNNROIHead',
        num_classes=3,
        semantic_head=dict(
            type='PointwiseMaskHead',
            in_channels=640,
            extra_width=0.2,
            class_agnostic=True,
            loss_seg=dict(
                type='FocalLoss',
                use_sigmoid=True,
                reduction='sum',
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0)),
        roi_extractor=dict(
            type='Batch3DRoIGridExtractor',
            in_channels=128,
            pool_radius=(0.8, 1.6),
            samples=(16, 16),
            mlps=((64, 64), (64, 64)),
            grid_size=6,
            mode='max'),
        bbox_head=dict(
            type='PVRCNNBboxHead',
            in_channels=128,
            grid_size=6,
            num_classes=3,
            class_agnostic=True,
            reg_fc=(256, 256),
            cls_fc=(256, 256),
            dropout=0.3,
            with_corner_loss=True,
            bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
            loss_bbox=dict(
                type='SmoothL1Loss',
                beta=1.0 / 9.0,
                reduction='sum',
                loss_weight=1.0),
            loss_cls=dict(
                type='CrossEntropyLoss',
                use_sigmoid=True,
                reduction='sum',
                loss_weight=1.0))))
