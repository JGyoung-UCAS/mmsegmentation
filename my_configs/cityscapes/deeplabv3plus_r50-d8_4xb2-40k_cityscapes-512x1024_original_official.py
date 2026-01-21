_base_ = [
    '../../configs/_base_/models/deeplabv3plus_r50-d8.py',
    '../../configs/_base_/datasets/cityscapes.py', '../../configs/_base_/default_runtime.py',
    '../../configs/_base_/schedules/schedule_40k.py'
]
crop_size = (512, 1024)
data_preprocessor = dict(size=crop_size)


model = dict(pretrained='/data/young/fork_code/mmsegmentation/local_results/ss/cityscapes/deeplabv3plus_r50-d8_4xb2-40k_cityscapes-512x1024_original_official/deeplabv3plus_r50-d8_512x1024_40k_cityscapes_20200605_094610-d222ffcd.pth',
    backbone=dict(frozen_stages=0),
    data_preprocessor=data_preprocessor)
