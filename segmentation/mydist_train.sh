#!/usr/bin/env bash

if [ $# -lt 1 ]; then
  echo "Usage: $0 EXP_NAME"
  exit 1
fi

EXP_NAME=$1
CONFIG=local_configs/segman/small/segman_s_polyp.py
WORK_DIR=outputs/${EXP_NAME}

###### 选择 GPU（物理 1,2）
export CUDA_VISIBLE_DEVICES=1,2

###### 使用 GPU 数量
GPUS=2

echo "====================================="
echo "Experiment Name : ${EXP_NAME}"
echo "Config File     : ${CONFIG}"
echo "Work Dir        : ${WORK_DIR}"
echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES}"
echo "GPUS            = ${GPUS}"
echo "====================================="

bash tools/dist_train.sh ${CONFIG} ${GPUS} \
    --work-dir ${WORK_DIR}\
    --cfg-options \
    runner.max_iters=20000 \
    optimizer.lr=0.0004 \
    data.samples_per_gpu=24 \
    data.workers_per_gpu=12
