_base_ = [
    '../../configs/_base_/models/gcnet_r50-d8.py', '../../configs/_base_/datasets/cityscapes.py',
    '../../configs/_base_/default_runtime.py', '../../configs/_base_/schedules/schedule_40k.py'
]
crop_size = (512, 1024)
data_preprocessor = dict(size=crop_size)

model = dict(pretrained='/data/young/codes/Robust_Vision/local_results/robust_exp/vr/train_from_scratch/in1k/resnet50_imagenet_baseline_mm/resnet50_vlc_baseline.pth',
    backbone=dict(frozen_stages=0),
    data_preprocessor=data_preprocessor)
