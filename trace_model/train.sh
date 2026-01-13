#!/bin/bash

# ===== CONFIGURE THESE =====
NUM_GPUS=6
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

export GPUS_PER_NODE=$NUM_GPUS
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29713
export PYTHONPATH="/raid/den365/RaDialog_v2:$PYTHONPATH"
export WANDB_MODE=disabled

python -m torch.distributed.run \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    llava/train/train.py \
    --lora_enable True \
    --lora_r 128 \
    --lora_alpha 256 \
    --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version v1 \
    --data_path /raid/den365/RaDialog_v2/data/mimic_cxr_instruct_llava_v2_valid.json \
    --image_folder /raid/den365/physionet.org/files/mimic-cxr-jpg/2.1.0 \
    --vision_tower biovil \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/radialog_v2_custom \
    --num_train_epochs 5 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1500 \
    --learning_rate 2e-5 \
    --max_grad_norm 0.1 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 1300 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none \
    --run_name radialog_v2_custom \
    --unfreeze_vision_tower_layers