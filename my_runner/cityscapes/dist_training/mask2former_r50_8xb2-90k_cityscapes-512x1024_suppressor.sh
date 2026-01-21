#!/usr/bin/env bash

CONFIG='my_configs/cityscapes/mask2former_r50_8xb2-90k_cityscapes-512x1024_suppressor.py'
GPUS=4
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
GPUS_ids="6,7,8,9"
WORK_DIR='local_results/ss/cityscapes/mask2former_r50_8xb2-90k_cityscapes-512x1024_suppressor'


#PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=$GPUS_ids python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    tools/train.py \
    $CONFIG \
    --work-dir $WORK_DIR \
    --launcher pytorch ${@:3}
