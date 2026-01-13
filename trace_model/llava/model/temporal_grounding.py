"""
Temporal Grounding Modules for TemporalGroundNet
Novel contribution: Grounded temporal change detection in radiology

Components:
1. TemporalCrossAttention - compares prior/current image features
2. ChangeAwareBoxPredictor - predicts bounding boxes for changed regions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

class TemporalCrossAttention(nn.Module):
    """
    Cross-attention between prior and current image features.
    Identifies WHERE changes occurred between studies.
    """
    def __init__(
        self,
        hidden_size: int = 512,  # BioViL-T feature size
        num_heads: int = 8,
        dropout: float = 0.1,
        num_change_classes: int = 3  # improved, worsened, no_change
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # Cross-attention: current queries, prior keys/values
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Self-attention on difference features
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Change detection head (per-patch classification)
        self.change_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_change_classes)
        )
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),  # current + prior + diff
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size)
        )
        
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
    def forward(
        self,
        current_features: torch.Tensor,  # [B, N, D] current image patches
        prior_features: torch.Tensor,     # [B, N, D] prior image patches
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            current_features: Current image patch features [B, N, D]
            prior_features: Prior image patch features [B, N, D]
            
        Returns:
            fused_features: Change-aware features [B, N, D]
            change_logits: Per-patch change classification [B, N, 3]
            attention_weights: Cross-attention weights [B, N, N] (optional)
        """
        B, N, D = current_features.shape
        
        # Cross-attention: current attends to prior
        cross_out, attn_weights = self.cross_attn(
            query=current_features,
            key=prior_features,
            value=prior_features,
            need_weights=return_attention
        )
        cross_out = self.norm1(current_features + cross_out)
        
        # Compute difference features
        diff_features = current_features - prior_features
        
        # Self-attention on difference to capture spatial patterns
        diff_out, _ = self.self_attn(
            query=diff_features,
            key=diff_features,
            value=diff_features
        )
        diff_out = self.norm2(diff_features + diff_out)
        
        # Fuse all features
        concat_features = torch.cat([current_features, cross_out, diff_out], dim=-1)
        fused_features = self.fusion(concat_features)
        
        # Per-patch change classification
        change_logits = self.change_classifier(fused_features)
        
        return fused_features, change_logits, attn_weights if return_attention else None


class ChangeAwareBoxPredictor(nn.Module):
    """
    DETR-style box prediction for changed regions.
    Predicts bounding boxes where temporal changes occurred.
    """
    def __init__(
        self,
        hidden_size: int = 512,
        num_queries: int = 20,  # max boxes to predict
        num_heads: int = 8,
        num_decoder_layers: int = 3,
        num_change_classes: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_queries = num_queries
        
        # Learnable box queries
        self.box_queries = nn.Embedding(num_queries, hidden_size)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        # Box prediction head (x1, y1, x2, y2)
        self.box_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 4),
            nn.Sigmoid()  # normalize to [0, 1]
        )
        
        # Change classification head
        self.class_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, num_change_classes + 1)  # +1 for "no object"
        )
        
        # Confidence head
        self.conf_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        temporal_features: torch.Tensor,  # [B, N, D] from TemporalCrossAttention
        change_logits: torch.Tensor = None  # [B, N, 3] optional guidance
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            temporal_features: Change-aware features from TemporalCrossAttention
            change_logits: Per-patch change predictions for guidance
            
        Returns:
            boxes: Predicted bounding boxes [B, num_queries, 4]
            class_logits: Change type classification [B, num_queries, num_classes+1]
            confidence: Box confidence scores [B, num_queries, 1]
        """
        B = temporal_features.shape[0]
        
        # Expand queries for batch
        queries = self.box_queries.weight.unsqueeze(0).expand(B, -1, -1)
        
        # Decode boxes from temporal features
        decoder_out = self.decoder(
            tgt=queries,
            memory=temporal_features
        )
        
        # Predict boxes, classes, and confidence
        boxes = self.box_head(decoder_out)
        class_logits = self.class_head(decoder_out)
        confidence = self.conf_head(decoder_out)
        
        return boxes, class_logits, confidence


class TemporalGroundingHead(nn.Module):
    """
    Complete temporal grounding module.
    Combines cross-attention and box prediction.
    """
    def __init__(
        self,
        hidden_size: int = 512,
        num_heads: int = 8,
        num_queries: int = 20,
        num_change_classes: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.temporal_attention = TemporalCrossAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            num_change_classes=num_change_classes
        )
        
        self.box_predictor = ChangeAwareBoxPredictor(
            hidden_size=hidden_size,
            num_queries=num_queries,
            num_heads=num_heads,
            num_change_classes=num_change_classes,
            dropout=dropout
        )
        
    def forward(
        self,
        current_features: torch.Tensor,
        prior_features: torch.Tensor,
        return_attention: bool = False
    ):
        """
        Full forward pass for temporal grounding.
        
        Returns:
            fused_features: For LLM input [B, N, D]
            boxes: Predicted change boxes [B, Q, 4]
            box_classes: Change type per box [B, Q, C+1]
            box_confidence: Confidence per box [B, Q, 1]
            patch_changes: Per-patch change logits [B, N, 3]
            attention: Cross-attention weights (optional)
        """
        # Get temporal features and patch-level changes
        fused_features, patch_changes, attention = self.temporal_attention(
            current_features, prior_features, return_attention
        )
        
        # Predict boxes for changed regions
        boxes, box_classes, box_confidence = self.box_predictor(
            fused_features, patch_changes
        )
        
        return {
            'fused_features': fused_features,
            'boxes': boxes,
            'box_classes': box_classes,
            'box_confidence': box_confidence,
            'patch_changes': patch_changes,
            'attention': attention
        }


def compute_temporal_grounding_loss(
    outputs: dict,
    target_boxes: torch.Tensor,      # [B, max_boxes, 4]
    target_classes: torch.Tensor,    # [B, max_boxes] 
    target_mask: torch.Tensor,       # [B, max_boxes] valid box mask
    box_weight: float = 2.0,
    giou_weight: float = 1.0,
    class_weight: float = 1.0
) -> dict:
    """
    Compute losses for temporal grounding.
    Uses Hungarian matching like DETR.
    """
    from scipy.optimize import linear_sum_assignment
    
    pred_boxes = outputs['boxes']  # [B, Q, 4]
    pred_classes = outputs['box_classes']  # [B, Q, C+1]
    
    B, Q, _ = pred_boxes.shape
    device = pred_boxes.device
    
    total_box_loss = 0
    total_giou_loss = 0
    total_class_loss = 0
    
    for b in range(B):
        # Get valid targets for this sample
        valid_mask = target_mask[b].bool()
        num_targets = valid_mask.sum().item()
        
        if num_targets == 0:
            # No targets - all predictions should be "no object"
            no_obj_target = torch.full((Q,), pred_classes.shape[-1] - 1, 
                                       device=device, dtype=torch.long)
            total_class_loss += F.cross_entropy(pred_classes[b], no_obj_target)
            continue
        
        tgt_boxes = target_boxes[b][valid_mask]  # [num_targets, 4]
        tgt_classes = target_classes[b][valid_mask]  # [num_targets]
        
        # Compute cost matrix for Hungarian matching
        with torch.no_grad():
            # L1 cost
            cost_bbox = torch.cdist(pred_boxes[b], tgt_boxes, p=1)  # [Q, num_targets]
            
            # Class cost
            cost_class = -pred_classes[b][:, tgt_classes.long()].softmax(-1)  # [Q, num_targets]
            
            # Combined cost
            C = cost_bbox * box_weight + cost_class * class_weight
            
            # Hungarian matching
            pred_idx, tgt_idx = linear_sum_assignment(C.cpu().numpy())
            pred_idx = torch.as_tensor(pred_idx, device=device)
            tgt_idx = torch.as_tensor(tgt_idx, device=device)
        
        # Compute losses on matched pairs
        matched_pred_boxes = pred_boxes[b][pred_idx]
        matched_tgt_boxes = tgt_boxes[tgt_idx]
        
        # L1 box loss
        total_box_loss += F.l1_loss(matched_pred_boxes, matched_tgt_boxes)
        
        # GIoU loss
        total_giou_loss += (1 - box_giou(matched_pred_boxes, matched_tgt_boxes)).mean()
        
        # Class loss (matched + unmatched)
        target_classes_full = torch.full((Q,), pred_classes.shape[-1] - 1,
                                         device=device, dtype=torch.long)
        target_classes_full[pred_idx] = tgt_classes[tgt_idx].long()
        total_class_loss += F.cross_entropy(pred_classes[b], target_classes_full)
    
    return {
        'box_loss': total_box_loss / B * box_weight,
        'giou_loss': total_giou_loss / B * giou_weight,
        'class_loss': total_class_loss / B * class_weight,
        'total_loss': (total_box_loss * box_weight + 
                      total_giou_loss * giou_weight + 
                      total_class_loss * class_weight) / B
    }


def box_giou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute GIoU between two sets of boxes."""
    # boxes format: [x1, y1, x2, y2], normalized to [0, 1]
    
    # Intersection
    inter_x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
    inter_y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
    inter_x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
    inter_y2 = torch.min(boxes1[:, 3], boxes2[:, 3])
    
    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    
    # Union
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union_area = area1 + area2 - inter_area
    
    iou = inter_area / (union_area + 1e-6)
    
    # Enclosing box
    enclose_x1 = torch.min(boxes1[:, 0], boxes2[:, 0])
    enclose_y1 = torch.min(boxes1[:, 1], boxes2[:, 1])
    enclose_x2 = torch.max(boxes1[:, 2], boxes2[:, 2])
    enclose_y2 = torch.max(boxes1[:, 3], boxes2[:, 3])
    
    enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1)
    
    giou = iou - (enclose_area - union_area) / (enclose_area + 1e-6)
    
    return giou


# Change label mapping
CHANGE_LABELS = {
    'improved': 0,
    'worsened': 1,
    'no change': 2
}
