_base_ = [
    '../../_base_/models/segman.py',
    '../../_base_/datasets/polyp.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_160k_adamw_polyp.py'
]

# model settings
norm_cfg = dict(type='BN', requires_grad=True)
# norm_cfg = dict(type='GN', num_groups=8, requires_grad=True)

model = dict(
    type='EncoderDecoder',
        backbone=dict(
        type='SegMANEncoder_s',
        pretrained='./pretrained/SegMAN_Encoder_s.pth.tar',
        style='pytorch',
        ),
    decode_head=dict(
        type='SegMANDecoder',
        in_channels=[64, 144, 288, 512],
        in_index=[0, 1, 2, 3],
        channels=152,
        feat_proj_dim=288,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            dict(type='DiceLoss', loss_weight=1.0) 
        ]),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.0002, betas=(0.9, 0.999), weight_decay=0.001,
                 paramwise_cfg=dict(custom_keys={'pos_block': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'head': dict(lr_mult=10.)
                                                 }))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

data = dict(samples_per_gpu=8,  ##total_batch_size
            workers_per_gpu=3
        ) #
evaluation = dict(
    interval=2000,                # 建议调小间隔，2000次验一次，更早发现好模型
    metric=['mDice', 'mIoU'],     # 同时计算 Dice 和 IoU
    save_best='mDice',            # 重点！以 Dice 分数最高为准保存 best_mDice.pth
    rule='greater'                # 分数越大越好
)
runner = dict(_delete_=True, type='IterBasedRunner', max_iters=36000)
checkpoint_config = dict(_delete_=True, by_epoch=False, interval=2000)

