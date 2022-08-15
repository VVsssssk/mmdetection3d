# Copyright (c) OpenMMLab. All rights reserved.
import mmcv

import mmdet
from .version import __version__, short_version


def digit_version(version_str):
    digit_version = []
    for x in version_str.split('.'):
        if x.isdigit():
            digit_version.append(int(x))
        elif x.find('rc') != -1:
            patch_version = x.split('rc')
            digit_version.append(int(patch_version[0]) - 1)
            digit_version.append(int(patch_version[1]))
    return digit_version


mmcv_minimum_version = '2.0.0rc0'
<<<<<<< HEAD
mmcv_maximum_version = '2.0.0rc0'
=======
mmcv_maximum_version = '2.1.0'
>>>>>>> 84b132b03846f6c64817248e62e0d78ec41b72f0
mmcv_version = digit_version(mmcv.__version__)


assert (mmcv_version >= digit_version(mmcv_minimum_version)
        and mmcv_version < digit_version(mmcv_maximum_version)), \
    f'MMCV=={mmcv.__version__} is used but incompatible. ' \
    f'Please install mmcv>={mmcv_minimum_version}, <={mmcv_maximum_version}.'

mmdet_minimum_version = '3.0.0rc0'
mmdet_maximum_version = '3.1.0'
mmdet_version = digit_version(mmdet.__version__)
assert (mmdet_version >= digit_version(mmdet_minimum_version)
        and mmdet_version < digit_version(mmdet_maximum_version)), \
    f'MMDET=={mmdet.__version__} is used but incompatible. ' \
    f'Please install mmdet>={mmdet_minimum_version}, ' \
    f'<={mmdet_maximum_version}.'

__all__ = ['__version__', 'short_version']
