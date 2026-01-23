dataset_type = 'PolypDataset'
data_root = 'data/polypdataset/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (352, 352)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(512, 512), ratio_range=(0.8, 1.2),keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.7),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'), # 水平翻转
    dict(type='RandomFlip', prob=0.5, direction='vertical'), # 垂直翻转
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(352, 352),
        img_ratios=[1.0], # 单尺度测试填 1.0，多尺度测试可填 [0.75, 1.0, 1.25]
        flip=False,       # 验证时通常关闭随机翻转，保证评估唯一性
        transforms=[
            dict(type='Resize', keep_ratio=True), 
            
            # 随机翻转在包装内必须有，但 flip=False 会关闭它
            dict(type='RandomFlip'), 
            
            # 归一化参数必须与训练完全一致
            dict(type='Normalize', **img_norm_cfg),
            
            # 填充：由于开启了 keep_ratio，长方形图片 Resize 后会有空余
            # 必须 Pad 到 32 的倍数，或者直接 Pad 到 352
            dict(type='Pad', size_divisor=32), 
            
            # 格式打包
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
# sub_datasets = ['CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'CVC-300', 'ETIS-LaribPolypDB']
data = dict(
    samples_per_gpu=32,   # 显存不够可以调小 batch size
    workers_per_gpu=8,
    train=dict(
        type='RepeatDataset',
            times=5,  # 对于1200张图，10倍非常合适
            dataset=dict(
                type=dataset_type,
                data_root=data_root,
                img_dir='train/images',
                ann_dir='train/masks',
                pipeline=train_pipeline)
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='val/images',
        ann_dir='val/masks',
        pipeline=test_pipeline
    ),
    test=dict(
            type=dataset_type,
            data_root=data_root,
            img_dir='test/CVC-ColonDB/images',
            ann_dir='test/CVC-ColonDB/masks',
            pipeline=test_pipeline
        )
)
