"""
Custom training script for TemporalGroundNet with auxiliary losses.
Adds box prediction loss and change classification loss on top of LM loss.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Dict, List
import transformers
from transformers import Trainer

# Import base training components
from llava.train.train import *
from llava.model.temporal_grounding import compute_temporal_grounding_loss, CHANGE_LABELS


class TemporalGroundingTrainer(Trainer):
    """Custom trainer that adds auxiliary losses for temporal grounding."""
    
    def __init__(self, *args, box_loss_weight=0.5, change_loss_weight=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.box_loss_weight = box_loss_weight
        self.change_loss_weight = change_loss_weight
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute LM loss + auxiliary losses."""
        # Standard forward pass
        outputs = model(**inputs)
        lm_loss = outputs.loss
        
        # Get grounding outputs if available
        grounding_outputs = getattr(model, '_temporal_grounding_outputs', None)
        
        total_loss = lm_loss
        
        if grounding_outputs is not None and 'target_boxes' in inputs:
            # Compute auxiliary losses
            target_boxes = inputs['target_boxes']
            target_classes = inputs['target_classes']
            target_mask = inputs['target_mask']
            
            aux_losses = compute_temporal_grounding_loss(
                grounding_outputs,
                target_boxes,
                target_classes,
                target_mask
            )
            
            # Add weighted auxiliary losses
            aux_loss = (self.box_loss_weight * (aux_losses['box_loss'] + aux_losses['giou_loss']) +
                       self.change_loss_weight * aux_losses['class_loss'])
            
            total_loss = lm_loss + aux_loss
            
            # Log auxiliary losses
            if self.state.global_step % 100 == 0:
                print(f"LM: {lm_loss.item():.4f}, Box: {aux_losses['box_loss'].item():.4f}, "
                      f"GIoU: {aux_losses['giou_loss'].item():.4f}, Class: {aux_losses['class_loss'].item():.4f}")
            
            # Clear grounding outputs
            model._temporal_grounding_outputs = None
        
        return (total_loss, outputs) if return_outputs else total_loss


class TemporalDataset(LazySupervisedDataset):
    """Dataset that also returns target boxes and change labels."""
    
    def __getitem__(self, i):
        # Get base item
        item = super().__getitem__(i)
        
        # Get source data for targets
        source = self.list_data_dict[i]
        
        # Parse target boxes and labels
        if 'boxes' in source and 'change_labels' in source:
            boxes = source['boxes']
            change_labels = source['change_labels']
            
            # Convert to tensors
            max_boxes = 20  # Same as num_queries in TemporalGroundingHead
            
            target_boxes = torch.zeros(max_boxes, 4)
            target_classes = torch.zeros(max_boxes, dtype=torch.long)
            target_mask = torch.zeros(max_boxes, dtype=torch.bool)
            
            num_boxes = min(len(boxes), max_boxes)
            for j in range(num_boxes):
                target_boxes[j] = torch.tensor(boxes[j])
                # Map change label to class index
                label = change_labels[j].lower()
                if 'improved' in label:
                    target_classes[j] = 0
                elif 'worsened' in label:
                    target_classes[j] = 1
                else:  # no change / stable
                    target_classes[j] = 2
                target_mask[j] = True
            
            item['target_boxes'] = target_boxes
            item['target_classes'] = target_classes
            item['target_mask'] = target_mask
        
        return item


@dataclass
class TemporalTrainingArguments(TrainingArguments):
    box_loss_weight: float = field(default=0.5)
    change_loss_weight: float = field(default=0.5)


def train():
    global local_rank
    
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TemporalTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type
            )
        ))

    if model_args.vision_tower is not None:
        if 'mpt' in model_args.model_name_or_path:
            config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config.attn_config['attn_impl'] = training_args.mpt_attn_impl
            model = LlavaMPTForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
        else:
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            **bnb_model_from_pretrained_args
        )
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    if 'mpt' in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )
        
        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = None
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    # Use temporal dataset
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    
    # Use custom trainer with auxiliary losses
    trainer = TemporalGroundingTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        box_loss_weight=training_args.box_loss_weight,
        change_loss_weight=training_args.change_loss_weight,
        **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
