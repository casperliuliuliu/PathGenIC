#!/bin/bash

BASE_PATH="your_base_path"
WEIGHT_PATH="qllava-v1.5-7b_lora-0109-1338"

export CUDA_VISIBLE_DEVICES=0

python -m llava.serve.cli \
    --is-wsi-feature \
    --model-base $BASE_PATH/Checkpoint/Quilt-Llava-v1.5-7b \
    --model-path $BASE_PATH/Checkpoint/casper-quilt-llava/$WEIGHT_PATH \
    --additional-model-path $BASE_PATH/Checkpoint/casper-quilt-llava/$WEIGHT_PATH/additional_module.bin \
    --mm-projector-path $BASE_PATH/Checkpoint/casper-quilt-llava/$WEIGHT_PATH/non_lora_trainables.bin \
    --image-file $BASE_PATH/Data/histgen/HistGen/DINOv2_Features/TCGA-23-1027-01Z-00-DX1.53F9DFF4-6811-4184-B2FD-1F6706B948FD.pt \
    --temperature 0.3 \
    --max-seq-length 100 \
    --forget-what-we-said \
    --use-guideline \
    --feedback \
