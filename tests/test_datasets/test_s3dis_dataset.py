keypoints: tensor(82933.4922, device='cuda:0')
point_bev_features: tensor(37229.7461, device='cuda:0')
pooled_features: tensor(73227.1562, device='cuda:0')
cur_features0: tensor(71103.4531, device='cuda:0')
pooled_features0: tensor(59361.9648, device='cuda:0')
cur_features1: tensor(173550.1250, device='cuda:0')
pooled_features1: tensor(56390.8203, device='cuda:0')
cur_features2: tensor(82281.6641, device='cuda:0')
pooled_features2: tensor(83988.9609, device='cuda:0')
cur_features3: tensor(93328.3281, device='cuda:0')
pooled_features3: tensor(109429.9062, device='cuda:0')
point_features_before_fusion tensor(419628.5625, device='cuda:0')

VoxelSetAbstraction(
  (SA_layers): ModuleList(
    (0): StackSAModuleMSG(
      (groupers): ModuleList(
        (0): QueryAndGroup()
        (1): QueryAndGroup()
      )
      (mlps): ModuleList(
        (0): Sequential(
          (0): Conv2d(19, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
          (3): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (4): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (5): ReLU()
        )
        (1): Sequential(
          (0): Conv2d(19, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
          (3): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (4): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (5): ReLU()
        )
      )
    )
    (1): StackSAModuleMSG(
      (groupers): ModuleList(
        (0): QueryAndGroup()
        (1): QueryAndGroup()
      )
      (mlps): ModuleList(
        (0): Sequential(
          (0): Conv2d(35, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
          (3): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (4): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (5): ReLU()
        )
        (1): Sequential(
          (0): Conv2d(35, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
          (3): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (4): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (5): ReLU()
        )
      )
    )
    (2): StackSAModuleMSG(
      (groupers): ModuleList(
        (0): QueryAndGroup()
        (1): QueryAndGroup()
      )
      (mlps): ModuleList(
        (0): Sequential(
          (0): Conv2d(67, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
          (3): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (5): ReLU()
        )
        (1): Sequential(
          (0): Conv2d(67, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
          (3): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (5): ReLU()
        )
      )
    )
    (3): StackSAModuleMSG(
      (groupers): ModuleList(
        (0): QueryAndGroup()
        (1): QueryAndGroup()
      )
      (mlps): ModuleList(
        (0): Sequential(
          (0): Conv2d(67, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
          (3): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (5): ReLU()
        )
        (1): Sequential(
          (0): Conv2d(67, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
          (3): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (5): ReLU()
        )
      )
    )
  )
