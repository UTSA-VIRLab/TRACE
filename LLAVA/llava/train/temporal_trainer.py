"""
Custom trainer for temporal grounding with auxiliary losses.
"""

import torch
import torch.nn.functional as F
from transformers import Trainer
from typing import Dict, Optional, List, Union, Any


def get_grounding_outputs(model):
    """Navigate model hierarchy to find _temporal_grounding_outputs."""
    # Try different paths through model wrappers
    candidates = [model]
    
    # DeepSpeed wraps the model
    if hasattr(model, 'module'):
        candidates.append(model.module)
    
    # PEFT wraps the model
    if hasattr(model, 'base_model'):
        candidates.append(model.base_model)
        if hasattr(model.base_model, 'model'):
            candidates.append(model.base_model.model)
    
    # Direct model access
    if hasattr(model, 'model'):
        candidates.append(model.model)
        if hasattr(model.model, 'model'):
            candidates.append(model.model.model)
    
    # Check each candidate
    for m in candidates:
        if hasattr(m, '_temporal_grounding_outputs') and m._temporal_grounding_outputs is not None:
            return m._temporal_grounding_outputs, m
    
    return None, None


def clear_grounding_outputs(model):
    """Clear grounding outputs from all model wrappers."""
    candidates = [model]
    if hasattr(model, 'module'):
        candidates.append(model.module)
    if hasattr(model, 'base_model'):
        candidates.append(model.base_model)
        if hasattr(model.base_model, 'model'):
            candidates.append(model.base_model.model)
    if hasattr(model, 'model'):
        candidates.append(model.model)
        if hasattr(model.model, 'model'):
            candidates.append(model.model.model)
    
    for m in candidates:
        if hasattr(m, '_temporal_grounding_outputs'):
            m._temporal_grounding_outputs = None


class TemporalGroundingTrainer(Trainer):
    """Trainer with auxiliary losses for temporal grounding."""
    
    def __init__(self, *args, box_loss_weight=1.0, change_loss_weight=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.box_loss_weight = box_loss_weight
        self.change_loss_weight = change_loss_weight
        self.aux_loss_logged = {'box': 0, 'change': 0, 'count': 0}
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute LM loss + auxiliary losses."""
        
        # Extract targets before forward pass
        target_boxes = inputs.pop('target_boxes', None)
        target_classes = inputs.pop('target_classes', None)
        target_mask = inputs.pop('target_mask', None)
        gt_change = inputs.pop('gt_change', None)
        
        # Standard forward pass
        outputs = model(**inputs)
        lm_loss = outputs.loss
        
        total_loss = lm_loss
        
        # Get grounding outputs from the model
        grounding_outputs, source_model = get_grounding_outputs(model)
        
        # Compute auxiliary losses if we have both predictions and targets
        if grounding_outputs is not None and gt_change is not None:
            aux_loss = torch.tensor(0.0, device=lm_loss.device, dtype=lm_loss.dtype)
            
            # Change classification loss (use patch-level predictions)
            if 'patch_changes' in grounding_outputs:
                patch_changes = grounding_outputs['patch_changes']  # [B, N, 3]
                # Global average to get [B, 3]
                global_logits = patch_changes.mean(dim=1)
                
                # Make sure dominant_change is on same device
                gt_change = gt_change.to(global_logits.device)
                
                change_loss = F.cross_entropy(global_logits, gt_change)
                aux_loss = aux_loss + self.change_loss_weight * change_loss
                
                self.aux_loss_logged['change'] += change_loss.item()
            
            # Box prediction loss
            if 'boxes' in grounding_outputs and target_boxes is not None and target_mask is not None:
                pred_boxes = grounding_outputs['boxes']  # [B, Q, 4]
                target_boxes = target_boxes.to(pred_boxes.device)
                target_mask = target_mask.to(pred_boxes.device)
                
                # Simple L1 loss on matched boxes
                batch_size = pred_boxes.shape[0]
                box_loss = torch.tensor(0.0, device=pred_boxes.device, dtype=pred_boxes.dtype)
                
                for b in range(batch_size):
                    mask = target_mask[b]
                    num_targets = mask.sum().int()
                    if num_targets > 0:
                        pred = pred_boxes[b, :num_targets]
                        tgt = target_boxes[b, :num_targets]
                        box_loss = box_loss + F.l1_loss(pred, tgt)
                
                box_loss = box_loss / batch_size
                aux_loss = aux_loss + self.box_loss_weight * box_loss
                
                self.aux_loss_logged['box'] += box_loss.item()
            
            self.aux_loss_logged['count'] += 1
            total_loss = lm_loss + aux_loss
            
            # Clear grounding outputs
            clear_grounding_outputs(model)
        
        # Log every 100 steps
        if self.state.global_step % 100 == 0 and self.aux_loss_logged['count'] > 0:
            avg_box = self.aux_loss_logged['box'] / self.aux_loss_logged['count']
            avg_change = self.aux_loss_logged['change'] / self.aux_loss_logged['count']
            print(f"Step {self.state.global_step}: LM={lm_loss.item():.4f}, Box={avg_box:.4f}, Change={avg_change:.4f}")
            self.aux_loss_logged = {'box': 0, 'change': 0, 'count': 0}
        
        return (total_loss, outputs) if return_outputs else total_loss
