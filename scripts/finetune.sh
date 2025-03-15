#!/bin/bash

BASE_PATH="your_base_path"

################## VICUNA ##################
PROMPT_VERSION=v1
# MODEL_VERSION="vicuna-v1-3-7b"

################## VICUNA ##################


deepspeed --include localhost:1,2 \
    --master_port 45216 \
    llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True \
    --model_name_or_path $BASE_PATH/Checkpoint/Quilt-Llava-v1.5-7b \
    --version $PROMPT_VERSION \
    --data_path $BASE_PATH/Data/histgen/train_1120.json \
    --image_folder ./playground/data/lil_test \
    --vision_tower openai/clip-vit-large-patch14 \
    --pretrain_mm_mlp_adapter $BASE_PATH/Checkpoint/Quilt-Llava-v1.5-7b/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir $BASE_PATH/Checkpoint/casper-quilt-llava/qllava-v1.5-7b_lora-0109-1338 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --dataloader_num_workers 4 \
    --report_to wandb \
    --is_wsi_feature True \
    --wsi_encoder True \
    --in_context_example True \
    --category_guidelines True \
    --training_data True \
    --feedback_refinement True \

