#!/usr/bin/env bash

CONFIG='my_configs/cityscapes/danet_r50-d8_4xb2-40k_cityscapes-512x1024_original.py'
GPUS=4
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
GPUS_ids="1,2,3,4"
WORK_DIR='local_results/ss/cityscapes/danet_r50-d8_4xb2-40k_cityscapes-512x1024_original'


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
