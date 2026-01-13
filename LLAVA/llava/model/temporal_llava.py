"""
TemporalGroundNet: Integration of temporal grounding into LLaVA/RaDialog
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Tuple

from .temporal_grounding import TemporalGroundingHead, CHANGE_LABELS


class TemporalGroundingWrapper(nn.Module):
    """
    Wraps temporal grounding head for integration with RaDialog.
    Processes prior/current image features before they go to LLM.
    """
    def __init__(
        self,
        hidden_size: int = 512,
        num_queries: int = 20,
        enabled: bool = True
    ):
        super().__init__()
        self.enabled = enabled
        self.hidden_size = hidden_size
        
        if enabled:
            self.grounding_head = TemporalGroundingHead(
                hidden_size=hidden_size,
                num_queries=num_queries,
                num_change_classes=3  # improved, worsened, no_change
            )
        
    def forward(
        self,
        current_features: torch.Tensor,
        prior_features: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        Process temporal features.
        
        Args:
            current_features: [B, N, D] current image features
            prior_features: [B, N, D] prior image features (optional)
            
        Returns:
            dict with:
                - features: processed features for LLM [B, N, D]
                - grounding_outputs: box predictions etc (if prior available)
        """
        if not self.enabled or prior_features is None:
            return {
                'features': current_features,
                'grounding_outputs': None
            }
        
        # Run temporal grounding
        outputs = self.grounding_head(current_features, prior_features)
        
        return {
            'features': outputs['fused_features'],
            'grounding_outputs': outputs
        }


def format_box_tokens(boxes: torch.Tensor, confidences: torch.Tensor, threshold: float = 0.5) -> List[str]:
    """
    Convert predicted boxes to text tokens for LLM output.
    
    Args:
        boxes: [num_queries, 4] predicted boxes (x1,y1,x2,y2 normalized)
        confidences: [num_queries, 1] confidence scores
        threshold: minimum confidence to include box
        
    Returns:
        List of box token strings like "<box>0.2,0.3,0.5,0.6</box>"
    """
    box_strings = []
    
    for i in range(boxes.shape[0]):
        if confidences[i, 0] >= threshold:
            x1, y1, x2, y2 = boxes[i].tolist()
            box_str = f"<box>{x1:.3f},{y1:.3f},{x2:.3f},{y2:.3f}</box>"
            box_strings.append(box_str)
    
    return box_strings


def parse_box_tokens(text: str) -> List[List[float]]:
    """
    Extract boxes from text containing <box>...</box> tokens.
    
    Args:
        text: Text with box tokens
        
    Returns:
        List of [x1, y1, x2, y2] coordinates
    """
    import re
    pattern = r'<box>([\d.,]+)</box>'
    matches = re.findall(pattern, text)
    
    boxes = []
    for match in matches:
        coords = [float(x) for x in match.split(',')]
        if len(coords) == 4:
            boxes.append(coords)
    
    return boxes


def prepare_temporal_targets(
    batch: Dict,
    device: torch.device,
    max_boxes: int = 20
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare target boxes and labels from batch data.
    
    Args:
        batch: Dictionary containing 'boxes', 'change_labels'
        device: Target device
        max_boxes: Maximum boxes per sample
        
    Returns:
        target_boxes: [B, max_boxes, 4]
        target_classes: [B, max_boxes]
        target_mask: [B, max_boxes] boolean mask for valid boxes
    """
    B = len(batch.get('boxes', []))
    
    target_boxes = torch.zeros(B, max_boxes, 4, device=device)
    target_classes = torch.zeros(B, max_boxes, dtype=torch.long, device=device)
    target_mask = torch.zeros(B, max_boxes, dtype=torch.bool, device=device)
    
    for i in range(B):
        boxes = batch['boxes'][i] if 'boxes' in batch else []
        labels = batch['change_labels'][i] if 'change_labels' in batch else []
        
        num_boxes = min(len(boxes), max_boxes)
        
        for j in range(num_boxes):
            target_boxes[i, j] = torch.tensor(boxes[j], device=device)
            target_classes[i, j] = CHANGE_LABELS.get(labels[j], 2)  # default to no_change
            target_mask[i, j] = True
    
    return target_boxes, target_classes, target_mask
