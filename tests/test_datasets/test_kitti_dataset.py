tensor(82933.4844, device='cuda:0')
point_bev_features: tensor(37229.7461, device='cuda:0')
pooled_features: tensor(73227.1562, device='cuda:0')
voxel_encode_features0: tensor(71103.4531, device='cuda:0')
pooled_features0: tensor(59361.9648, device='cuda:0')
voxel_encode_features1: tensor(173550.1250, device='cuda:0')
pooled_features1: tensor(49360.9570, device='cuda:0')
voxel_encode_features2: tensor(82281.6641, device='cuda:0')
pooled_features2: tensor(76148.5859, device='cuda:0')
voxel_encode_features3: tensor(93328.3281, device='cuda:0')
pooled_features3: tensor(94469.5000, device='cuda:0')
point_features_before_fusion tensor(389797.9375, device='cuda:0')

  (voxel_sa_layers): ModuleList(
    (0): GuidedSAModuleMSG(
      (groupers): ModuleList(
        (0): QueryAndGroup()
        (1): QueryAndGroup()
      )
      (mlps): ModuleList(
        (0): Sequential(
          (layer0): ConvModule(
            (conv): Conv2d(19, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): ReLU(inplace=True)
          )
          (layer1): ConvModule(
            (conv): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): ReLU(inplace=True)
          )
        )
        (1): Sequential(
          (layer0): ConvModule(
            (conv): Conv2d(19, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): ReLU(inplace=True)
          )
          (layer1): ConvModule(
            (conv): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): ReLU(inplace=True)
          )
        )
      )
    )
    (1): GuidedSAModuleMSG(
      (groupers): ModuleList(
        (0): QueryAndGroup()
        (1): QueryAndGroup()
      )
      (mlps): ModuleList(
        (0): Sequential(
          (layer0): ConvModule(
            (conv): Conv2d(35, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): ReLU(inplace=True)
          )
          (layer1): ConvModule(
            (conv): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): ReLU(inplace=True)
          )
        )
        (1): Sequential(
          (layer0): ConvModule(
            (conv): Conv2d(35, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): ReLU(inplace=True)
          )
          (layer1): ConvModule(
            (conv): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): ReLU(inplace=True)
          )
        )
      )
    )
    (2): GuidedSAModuleMSG(
      (groupers): ModuleList(
        (0): QueryAndGroup()
        (1): QueryAndGroup()
      )
      (mlps): ModuleList(
        (0): Sequential(
          (layer0): ConvModule(
            (conv): Conv2d(67, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): ReLU(inplace=True)
          )
          (layer1): ConvModule(
            (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): ReLU(inplace=True)
          )
        )
        (1): Sequential(
          (layer0): ConvModule(
            (conv): Conv2d(67, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): ReLU(inplace=True)
          )
          (layer1): ConvModule(
            (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): ReLU(inplace=True)
          )
        )
      )
    )
    (3): GuidedSAModuleMSG(
      (groupers): ModuleList(
        (0): QueryAndGroup()
        (1): QueryAndGroup()
      )
      (mlps): ModuleList(
        (0): Sequential(
          (layer0): ConvModule(
            (conv): Conv2d(67, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): ReLU(inplace=True)
          )
          (layer1): ConvModule(
            (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): ReLU(inplace=True)
          )
        )
        (1): Sequential(
          (layer0): ConvModule(
            (conv): Conv2d(67, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): ReLU(inplace=True)
          )
          (layer1): ConvModule(
            (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): ReLU(inplace=True)
          )
        )
      )
    )
  )
