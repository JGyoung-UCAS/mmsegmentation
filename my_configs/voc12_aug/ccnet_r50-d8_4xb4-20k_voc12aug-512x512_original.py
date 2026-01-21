_base_ = [
    '../../configs/_base_/models/ccnet_r50-d8.py',
    '../../configs/_base_/datasets/pascal_voc12_aug.py', '../../configs/_base_/default_runtime.py',
    '../../configs/_base_/schedules/schedule_20k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size,size_divisor=None)

model = dict(pretrained='/data/young/codes/Robust_Vision/local_results/robust_exp/vr/train_from_scratch/in1k/resnet50_imagenet_baseline_mm/resnet50_vlc_baseline.pth',
    backbone=dict(frozen_stages=0),
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=21),
    auxiliary_head=dict(num_classes=21))

