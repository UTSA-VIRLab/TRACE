#!/bin/bash

# Temporal Grounding Training Script
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export WANDB_PROJECT="temporal_grounding"

# Use base Vicuna model (LoRA will be applied during training)
BASE_MODEL="lmsys/vicuna-7b-v1.5"
# Pretrained adapter and projector from RaDialog
PRETRAIN_ADAPTER="/raid/den365/RaDialog_v2/checkpoints/radialog-v2"
DATA_PATH="/raid/den365/RaDialog_v2/data/temporal_grounding/train.json"
IMAGE_FOLDER="/raid/den365/physionet.org/files/mimic-cxr-jpg/2.1.0"
OUTPUT_DIR="/raid/den365/RaDialog_v2/trace_model/checkpoints/temporal_grounding_v1"

cd /raid/den365/RaDialog_v2/trace_model

deepspeed --num_gpus=6 --master_port=29501 llava/train/train.py \
    --deepspeed scripts/zero2.json \
    --lora_enable True \
    --lora_r 128 \
    --lora_alpha 256 \
    --mm_projector_lr 2e-5 \
    --model_name_or_path $BASE_MODEL \
    --pretrain_mm_mlp_adapter "${PRETRAIN_ADAPTER}/non_lora_trainables.bin" \
    --version v1 \
    --data_path $DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --vision_tower biovil \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 3 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name "temporal_grounding_v1" \
    --mv_type "concat"
