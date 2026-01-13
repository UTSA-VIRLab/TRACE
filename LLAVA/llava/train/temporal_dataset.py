"""
Temporal dataset that returns target boxes and change labels for auxiliary losses.
"""

import torch
import json
import copy
from torch.utils.data import Dataset

CHANGE_TO_IDX = {
    'improved': 0,
    'worsened': 1, 
    'no change': 2,
    'stable': 2,
}

def parse_targets_from_sample(sample, max_boxes=20):
    """Extract target boxes and change labels from a sample."""
    boxes = sample.get('boxes', [])
    change_labels = sample.get('change_labels', [])
    
    target_boxes = torch.zeros(max_boxes, 4)
    target_classes = torch.zeros(max_boxes, dtype=torch.long)
    target_mask = torch.zeros(max_boxes, dtype=torch.bool)
    
    num_boxes = min(len(boxes), max_boxes)
    for j in range(num_boxes):
        if j < len(boxes):
            target_boxes[j] = torch.tensor(boxes[j])
        if j < len(change_labels):
            label = change_labels[j].lower()
            if 'improv' in label:
                target_classes[j] = 0
            elif 'worsen' in label:
                target_classes[j] = 1
            else:
                target_classes[j] = 2
        target_mask[j] = True
    
    # Get dominant change for global classification
    if change_labels:
        from collections import Counter
        normalized = []
        for label in change_labels:
            label = label.lower()
            if 'worsen' in label:
                normalized.append(1)
            elif 'improv' in label:
                normalized.append(0)
            else:
                normalized.append(2)
        dominant = Counter(normalized).most_common(1)[0][0]
    else:
        dominant = 2  # stable by default
    
    return {
        'target_boxes': target_boxes,
        'target_classes': target_classes,
        'target_mask': target_mask,
        'dominant_change': torch.tensor(dominant, dtype=torch.long)
    }
