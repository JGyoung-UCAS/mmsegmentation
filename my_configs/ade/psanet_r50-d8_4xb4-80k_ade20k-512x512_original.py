_base_ = [
    '../../configs/_base_/models/psanet_r50-d8.py', '../../configs/_base_/datasets/ade20k.py',
    '../../configs/_base_/default_runtime.py', '../../configs/_base_/schedules/schedule_80k.py'
]

crop_size = (512, 512)
data_preprocessor = dict(size=crop_size,size_divisor=None)

model = dict(pretrained='/data/young/codes/Robust_Vision/local_results/robust_exp/vr/train_from_scratch/in1k/resnet50_imagenet_baseline_mm/resnet50_vlc_baseline.pth',
    backbone=dict(frozen_stages=0),
    data_preprocessor=data_preprocessor,
    decode_head=dict(mask_size=(66, 66),num_classes=150),
    auxiliary_head=dict(num_classes=150))
