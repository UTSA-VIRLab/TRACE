"""Evaluate single image ablation model"""
import json
import pathlib
import torch
from PIL import Image
from skimage import io
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor, CenterCrop
from tqdm import tqdm
import re
import sys
from functools import wraps

sys.path.insert(0, '/raid/den365/RaDialog_v2')

from LLAVA.llava.constants import IMAGE_TOKEN_INDEX
from LLAVA.llava.conversation import conv_vicuna_v1
from LLAVA.llava.mm_utils import tokenizer_image_token, process_image_biovil
from LLAVA.llava.model.builder import load_pretrained_model
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

class ExpandChannels:
    def __call__(self, data):
        return data.expand(3, -1, -1)

def patch_model_forward(model):
    """Fix for newer transformers versions"""
    original_forward = model.forward
    @wraps(original_forward)
    def patched_forward(*args, **kwargs):
        kwargs.pop('cache_position', None)
        kwargs.pop('num_logits_to_keep', None)
        return original_forward(*args, **kwargs)
    model.forward = patched_forward
    return model

def remap_to_uint8(array):
    array = array.astype(float)
    array -= array.min()
    array /= (array.max() + 1e-8)
    array *= 255
    return array.astype(np.uint8)

def load_img(path):
    img = io.imread(path)
    img = Image.fromarray(remap_to_uint8(img)).convert("L")
    return img

def extract_boxes(text):
    pattern = r'<box>([\d.,]+)</box>'
    boxes = []
    for match in re.finditer(pattern, text):
        coords = [float(x) for x in match.group(1).split(',')]
        if len(coords) == 4:
            boxes.append(coords)
    return boxes

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

def get_change_type(text):
    text_lower = text.lower()
    if 'worsening' in text_lower or 'worse' in text_lower or 'increased' in text_lower:
        return 'worsening'
    elif 'improvement' in text_lower or 'improving' in text_lower or 'improved' in text_lower or 'decreased' in text_lower or 'resolved' in text_lower:
        return 'improvement'
    else:
        return 'stable'

vis_transforms = Compose([Resize(512), CenterCrop(448), ToTensor(), ExpandChannels()])
mimic_base = "/raid/den365/physionet.org/files/mimic-cxr-jpg/2.1.0"

def main(num_samples=None):
    device = "cuda:0"
    model_path = pathlib.Path("./LLAVA/checkpoints/ablation_single_image")
    
    print("Loading model...")
    tokenizer, model, _, _ = load_pretrained_model(
        model_path, model_base="lmsys/vicuna-7b-v1.5", 
        model_name="llava-lora", device=device
    )
    model = patch_model_forward(model)
    print("Model loaded")
    
    # Load test data (use test_single_image.json for single-image ablation)
    with open('./data/temporal_grounding/test_single_image.json') as f:
        test_data = json.load(f)
    
    if num_samples:
        test_data = test_data[:num_samples]
    
    print(f"Evaluating on {len(test_data)} samples...")
    
    bleu_scores = []
    ious = []
    change_matrix = {
        'worsening': {'worsening': 0, 'improvement': 0, 'stable': 0},
        'improvement': {'worsening': 0, 'improvement': 0, 'stable': 0},
        'stable': {'worsening': 0, 'improvement': 0, 'stable': 0}
    }
    smooth = SmoothingFunction().method1
    errors = 0
    
    for sample in tqdm(test_data):
        try:
            curr_tensor = process_image_biovil([load_img(f"{mimic_base}/{sample['image']}")], vis_transforms).to(device, dtype=torch.bfloat16)
            
            prompt_text = sample['conversations'][0]['value']
            conv = conv_vicuna_v1.copy()
            conv.append_message(conv.roles[0], prompt_text)
            conv.append_message(conv.roles[1], None)
            input_ids = tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)
            
            with torch.inference_mode():
                output_ids = model.generate(input_ids, images=curr_tensor, do_sample=False, max_new_tokens=256, use_cache=True)
            
            pred = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip().replace("</s>", "")
            ref = sample['conversations'][1]['value']
            
            # BLEU
            pred_tokens = pred.lower().split()
            ref_tokens = ref.lower().split()
            if pred_tokens and ref_tokens:
                bleu = sentence_bleu([ref_tokens], pred_tokens, weights=(0.25,0.25,0.25,0.25), smoothing_function=smooth)
                bleu_scores.append(bleu)
            
            # IoU
            pred_boxes = extract_boxes(pred)
            ref_boxes = extract_boxes(ref)
            if pred_boxes and ref_boxes:
                best_iou = max(compute_iou(pb, rb) for pb in pred_boxes for rb in ref_boxes)
                ious.append(best_iou)
            
            # Change detection
            pred_change = get_change_type(pred)
            ref_change = get_change_type(ref)
            change_matrix[ref_change][pred_change] += 1
                
        except Exception as e:
            errors += 1
            continue
    
    # Results
    total = sum(sum(v.values()) for v in change_matrix.values())
    correct = sum(change_matrix[c][c] for c in change_matrix)
    
    print(f"\n{'='*60}")
    print(f"SINGLE IMAGE ABLATION RESULTS")
    print(f"{'='*60}")
    print(f"Samples: {total}, Errors: {errors}")
    print(f"\nBLEU-4: {np.mean(bleu_scores):.4f}")
    print(f"\nChange Detection Accuracy: {100*correct/total:.1f}%")
    print(f"\nConfusion Matrix:")
    print(f"{'GT/Pred':<15} {'worsening':<12} {'improvement':<12} {'stable':<12}")
    for gt in ['worsening', 'improvement', 'stable']:
        row = change_matrix[gt]
        print(f"{gt:<15} {row['worsening']:<12} {row['improvement']:<12} {row['stable']:<12}")
    
    print(f"\nPer-class accuracy:")
    for cls in ['worsening', 'improvement', 'stable']:
        t = sum(change_matrix[cls].values())
        c = change_matrix[cls][cls]
        print(f"  {cls}: {c}/{t} = {100*c/t:.1f}%" if t > 0 else f"  {cls}: N/A")
    
    if ious:
        print(f"\nGrounding:")
        print(f"  Avg IoU: {np.mean(ious):.3f}")
        print(f"  IoU>0.5: {100*sum(1 for x in ious if x>0.5)/len(ious):.1f}%")
    
    # Save results
    results = {
        'bleu4': float(np.mean(bleu_scores)),
        'change_accuracy': correct/total,
        'change_matrix': change_matrix,
        'iou_mean': float(np.mean(ious)) if ious else 0,
        'iou_above_5': sum(1 for x in ious if x>0.5)/len(ious) if ious else 0
    }
    with open('./data/temporal_grounding/ablation_single_image_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to ablation_single_image_results.json")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=None, help='Number of samples (default: all)')
    args = parser.parse_args()
    main(args.samples)
