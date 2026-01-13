"""
LLaVA-Llama model with temporal grounding auxiliary losses.
"""

from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


class LlavaConfig(LlamaConfig):
    model_type = "llava"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig, mv_type='none'):
        super(LlavaLlamaModel, self).__init__(config, mv_type)


class LlavaLlamaForCausalLMTemporal(LlavaMetaForCausalLM, LlamaForCausalLM):
    """LLaVA model with temporal grounding auxiliary loss."""
    config_class = LlavaConfig

    def __init__(self, config, mv_type='none'):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config, mv_type)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Store auxiliary loss weights
        self.change_loss_weight = 0.1  # Weight for change classification loss
        
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        prev_images: Optional[torch.FloatTensor] = None,
        target_change_labels: Optional[torch.LongTensor] = None,  # For auxiliary loss
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                prev_images
            )

        # Standard forward pass
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # Add auxiliary change classification loss if we have grounding outputs and targets
        if hasattr(self, '_temporal_grounding_outputs') and self._temporal_grounding_outputs is not None:
            grounding_outputs = self._temporal_grounding_outputs
            
            if target_change_labels is not None and 'patch_changes' in grounding_outputs:
                # Compute change classification loss
                # patch_changes: [B, N, 3] - per-patch change predictions
                # We use global average pooling to get [B, 3] and compare with target
                patch_changes = grounding_outputs['patch_changes']  # [B, N, 3]
                global_change_logits = patch_changes.mean(dim=1)  # [B, 3]
                
                change_loss = F.cross_entropy(global_change_logits, target_change_labels)
                
                # Add to total loss
                if outputs.loss is not None:
                    outputs.loss = outputs.loss + self.change_loss_weight * change_loss
            
            # Clear for next forward pass
            self._temporal_grounding_outputs = None
        
        return outputs

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        prev_images = kwargs.pop("prev_images", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        if prev_images is not None:
            _inputs['prev_images'] = prev_images
        return _inputs


AutoConfig.register("llava", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLMTemporal)
