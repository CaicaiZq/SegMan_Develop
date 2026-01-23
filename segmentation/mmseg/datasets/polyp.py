# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from .builder import DATASETS
from .custom import CustomDataset

@DATASETS.register_module()
class PolypDataset(CustomDataset):
    """Polyp dataset for medical image segmentation.
    
    CLASSES 对应你清洗后的像素值：0 为 background, 1 为 polyp。
    PALETTE 定义可视化颜色：黑色背景，白色息肉。
    """

    CLASSES = ('background', 'polyp')

    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 **kwargs):
        super(PolypDataset, self).__init__(
            img_suffix=img_suffix, 
            seg_map_suffix=seg_map_suffix, 
            reduce_zero_label=False, # 必须为 False，因为 0 是你的背景类
            **kwargs)