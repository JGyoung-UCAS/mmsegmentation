_base_ = [
    '../../configs/_base_/models/pspnet_r50-d8.py',
    '../../configs/_base_/datasets/pascal_voc12_aug.py', '../../configs/_base_/default_runtime.py',
    '../../configs/_base_/schedules/schedule_20k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size,size_divisor=None)

model = dict(pretrained='/data/young/fork_code/mmsegmentation/local_results/ss/voc12_aug/pspnet_r50-d8_4xb4-20k_voc12aug-512x512_original_official/pspnet_r50-d8_512x512_20k_voc12aug_20200617_101958-ed5dfbd9.pth',
    backbone=dict(frozen_stages=0),
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=21),
    auxiliary_head=dict(num_classes=21))
