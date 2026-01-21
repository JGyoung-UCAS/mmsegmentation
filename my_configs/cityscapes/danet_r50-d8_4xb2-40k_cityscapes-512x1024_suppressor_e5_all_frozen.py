_base_ = [
    '../../configs/_base_/models/danet_r50-d8.py', '../../configs/_base_/datasets/cityscapes.py',
    '../../configs/_base_/default_runtime.py', '../../configs/_base_/schedules/schedule_40k.py'
]
crop_size = (512, 1024)
data_preprocessor = dict(size=crop_size)

model = dict(pretrained='/data/young/codes/Robust_Vision/local_results/robust_exp/vr/finetune/partial_finetune/in1k/resnet50_mm_imagenet_conv1_top32_stem8_463_e5_replace_diversify_2/resnet50_vlc_diversify.pth',
    backbone=dict(frozen_stages=0),
    data_preprocessor=data_preprocessor)
