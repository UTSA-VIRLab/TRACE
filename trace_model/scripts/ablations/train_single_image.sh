#!/bin/bash
cd /raid/den365/RaDialog_v2/trace_model

deepspeed --num_gpus 6 llava/train/train.py \
    --deepspeed scripts/zero2.json \
    --lora_enable True --lora_r 128 --lora_alpha 256 \
    --mm_projector_lr 2e-5 \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --pretrain_mm_mlp_adapter /raid/den365/RaDialog_v2/trace_model/checkpoints/radialog_v2_custom/non_lora_trainables.bin \
    --version v1 \
    --data_path /raid/den365/RaDialog_v2/data/temporal_grounding/train_single_image.json \
    --image_folder /raid/den365/physionet.org/files/mimic-cxr-jpg/2.1.0 \
    --vision_tower biovil --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 --mm_use_im_start_end False --mm_use_im_patch_token False \
    --image_aspect_ratio pad --group_by_modality_length True --bf16 True \
    --output_dir /raid/den365/RaDialog_v2/trace_model/checkpoints/ablation_single_image \
    --num_train_epochs 3 --per_device_train_batch_size 4 --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 --save_strategy steps --save_steps 500 --save_total_limit 2 \
    --learning_rate 2e-4 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type cosine \
    --logging_steps 10 --tf32 True --model_max_length 2048 --gradient_checkpointing True \
    --dataloader_num_workers 4 --lazy_preprocess True --report_to none \
    --run_name ablation_single_image
