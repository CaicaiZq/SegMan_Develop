#!/bin/bash

# 使用方法：
# ./train.sh EXP_NAME

if [ $# -lt 1 ]; then
  echo "Usage: $0 EXP_NAME"
  exit 1
fi

EXP_NAME=$1
CONFIG=local_configs/segman/small/segman_s_polyp.py
WORK_DIR=outputs/${EXP_NAME}

echo "====================================="
echo "Experiment Name : ${EXP_NAME}"
echo "Config File     : ${CONFIG}"
echo "Work Dir        : ${WORK_DIR}"
echo "====================================="

python tools/train.py ${CONFIG} \
  --work-dir ${WORK_DIR} \
  --cfg-options \
    runner.max_iters=36000 \
    optimizer.lr=0.0002 \
    data.samples_per_gpu=8 \
    data.workers_per_gpu=8
