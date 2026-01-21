_base_ = [
    '../../configs/_base_/models/deeplabv3plus_r50-d8.py', '../../configs/_base_/datasets/ade20k.py',
    '../../configs/_base_/default_runtime.py', '../../configs/_base_/schedules/schedule_80k.py'
]
crop_size = (512, 512)



# model=dict(pretrained=None,backbone=dict(frozen_stages=0,init_cfg=dict(type='Pretrained',
# checkpoint='/data/young/codes/Robust_Vision/local_results/robust_exp/vr/train_from_scratch/in1k/resnet50_imagenet_baseline_mm/resnet50_vlc_baseline.pth')))

data_preprocessor = dict(size=crop_size,size_divisor=None)

model = dict(pretrained='/data/young/fork_code/mmsegmentation/local_results/ss/ADE20k/deeplabv3plus_r50-d8_4xb4-80k_ade20k-512x512_original_official/deeplabv3plus_r50-d8_512x512_80k_ade20k_20200614_185028-bf1400d8.pth',
    backbone=dict(frozen_stages=0),
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=150),
    auxiliary_head=dict(num_classes=150))

#size_divisor=None
# model=dict(backbone=dict(frozen_stages=0,init_cfg=dict(type='Pretrained',
# checkpoint='/data/young/codes/Robust_Vision/local_results/robust_exp/vr/train_from_scratch/in1k/resnet50_imagenet_baseline_mm/model_best.pth.tar')))


#model=dict()
#model = dict(pretrained='torchvision://resnet50', backbone=dict(type='ResNet'))