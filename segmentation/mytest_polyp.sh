#!/bin/bash

# 使用方法：
# ./test.sh EXP_NAME CHECKPOINT_PATH

if [ $# -lt 2 ]; then
  echo "Usage: $0 CHECKPOINT_PATH EXP_NAME"
  exit 1
fi

CKPT=$1
EXP_NAME=$2


CONFIG=local_configs/segman/small/segman_s_polyp.py
WORK_DIR=test_eval/${EXP_NAME}

echo "====================================="
echo "Experiment Name : ${EXP_NAME}"
echo "Config File     : ${CONFIG}"
echo "Checkpoint      : ${CKPT}"
echo "Eval Metrics    : mDice mIoU"
echo "Work Dir        : ${WORK_DIR}"
echo "====================================="

python tools/test.py \
  ${CONFIG} \
  --checkpoint ${CKPT} \
  --eval mDice mIoU \
  --work-dir ${WORK_DIR}
