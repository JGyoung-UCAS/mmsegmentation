#!/usr/bin/env bash

CONFIG='my_configs/ade/ccnet_r50-d8_4xb4-80k_ade20k-512x512_suppressor_e5_all_frozen.py'
GPUS=4
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
GPUS_ids="3,4,8,9"
WORK_DIR='local_results/ss/ADE20k/ccnet_r50-d8_4xb4-80k_ade20k-512x512_suppressor_e5_all_frozen'


#PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=$GPUS_ids python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    tools/train.py \
    $CONFIG \
    --resume \
    --work-dir $WORK_DIR \
    --launcher pytorch ${@:3}
