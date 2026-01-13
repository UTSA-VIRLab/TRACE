"""Comprehensive evaluation with NLG + Clinical Efficacy metrics - FIXED"""
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
import os
from functools import wraps

sys.path.insert(0, '/raid/den365/RaDialog_v2')

from LLAVA.llava.constants import IMAGE_TOKEN_INDEX
from LLAVA.llava.conversation import conv_vicuna_v1
from LLAVA.llava.mm_utils import tokenizer_image_token, process_image_biovil
from LLAVA.llava.model.builder import load_pretrained_model
from collections import defaultdict

# NLG Metrics
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import nltk
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Try to import clinical metrics
try:
    from radgraph import F1RadGraph
    RADGRAPH_AVAILABLE = True
    print("RadGraph available!")
except ImportError:
    RADGRAPH_AVAILABLE = False
    print("RadGraph not available")

try:
    from f1chexbert import F1CheXbert
    CHEXBERT_AVAILABLE = True
    print("CheXbert available!")
except ImportError:
    CHEXBERT_AVAILABLE = False
    print("CheXbert not available")

def patch_model_forward(model):
    """Patch model forward to ignore unknown kwargs like cache_position"""
    original_forward = model.forward
    
    @wraps(original_forward)
    def patched_forward(*args, **kwargs):
        # Remove unsupported arguments
        kwargs.pop('cache_position', None)
        kwargs.pop('num_logits_to_keep', None)
        return original_forward(*args, **kwargs)
    
    model.forward = patched_forward
    return model

class ExpandChannels:
    def __call__(self, data):
        return data.expand(3, -1, -1)

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
    elif 'improving' in text_lower or 'improved' in text_lower or 'decreased' in text_lower or 'resolved' in text_lower:
        return 'improvement'
    else:
        return 'stable'

def remove_boxes_from_text(text):
    return re.sub(r'<box>[\d.,]+</box>', '', text).strip()

def compute_nlg_metrics(predictions, references):
    """Compute all NLG metrics"""
    if not predictions or not references:
        return {k: 0 for k in ['bleu1', 'bleu2', 'bleu3', 'bleu4', 'corpus_bleu4', 'meteor', 'rouge1', 'rouge2', 'rougeL']}
    
    smooth = SmoothingFunction().method1
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    bleu1_scores, bleu2_scores, bleu3_scores, bleu4_scores = [], [], [], []
    meteor_scores_list = []
    rouge1_scores, rouge2_scores, rougeL_scores = [], [], []
    
    all_refs = []
    all_preds = []
    
    for pred, ref in zip(predictions, references):
        pred_clean = remove_boxes_from_text(pred)
        ref_clean = remove_boxes_from_text(ref)
        
        pred_tokens = pred_clean.lower().split()
        ref_tokens = ref_clean.lower().split()
        
        if not pred_tokens or not ref_tokens:
            continue
        
        all_refs.append([ref_tokens])
        all_preds.append(pred_tokens)
        
        # BLEU
        try:
            bleu1_scores.append(sentence_bleu([ref_tokens], pred_tokens, weights=(1,0,0,0), smoothing_function=smooth))
            bleu2_scores.append(sentence_bleu([ref_tokens], pred_tokens, weights=(0.5,0.5,0,0), smoothing_function=smooth))
            bleu3_scores.append(sentence_bleu([ref_tokens], pred_tokens, weights=(0.33,0.33,0.33,0), smoothing_function=smooth))
            bleu4_scores.append(sentence_bleu([ref_tokens], pred_tokens, weights=(0.25,0.25,0.25,0.25), smoothing_function=smooth))
        except:
            pass
        
        # METEOR
        try:
            meteor_scores_list.append(meteor_score([ref_tokens], pred_tokens))
        except:
            pass
        
        # ROUGE
        try:
            rouge_result = rouge.score(ref_clean, pred_clean)
            rouge1_scores.append(rouge_result['rouge1'].fmeasure)
            rouge2_scores.append(rouge_result['rouge2'].fmeasure)
            rougeL_scores.append(rouge_result['rougeL'].fmeasure)
        except:
            pass
    
    # Corpus BLEU
    try:
        corpus_bleu4 = corpus_bleu(all_refs, all_preds, weights=(0.25,0.25,0.25,0.25), smoothing_function=smooth) if all_refs else 0
    except:
        corpus_bleu4 = 0
    
    return {
        'bleu1': np.mean(bleu1_scores) if bleu1_scores else 0,
        'bleu2': np.mean(bleu2_scores) if bleu2_scores else 0,
        'bleu3': np.mean(bleu3_scores) if bleu3_scores else 0,
        'bleu4': np.mean(bleu4_scores) if bleu4_scores else 0,
        'corpus_bleu4': corpus_bleu4,
        'meteor': np.mean(meteor_scores_list) if meteor_scores_list else 0,
        'rouge1': np.mean(rouge1_scores) if rouge1_scores else 0,
        'rouge2': np.mean(rouge2_scores) if rouge2_scores else 0,
        'rougeL': np.mean(rougeL_scores) if rougeL_scores else 0,
    }

def compute_clinical_metrics(predictions, references):
    """Compute clinical efficacy metrics"""
    clinical_metrics = {}
    
    if not predictions:
        return clinical_metrics
    
    preds_clean = [remove_boxes_from_text(p) for p in predictions]
    refs_clean = [remove_boxes_from_text(r) for r in references]
    
    if RADGRAPH_AVAILABLE:
        print("Computing RadGraph F1...")
        try:
            f1radgraph = F1RadGraph(reward_level="partial")
            _, _, radgraph_f1, _ = f1radgraph(hyps=preds_clean, refs=refs_clean)
            clinical_metrics['radgraph_f1'] = float(radgraph_f1)
            print(f"RadGraph F1: {radgraph_f1:.4f}")
        except Exception as e:
            print(f"RadGraph error: {e}")
            clinical_metrics['radgraph_f1'] = None
    
    if CHEXBERT_AVAILABLE:
        print("Computing CheXbert F1...")
        try:
            f1chexbert = F1CheXbert()
            chexbert_f1 = f1chexbert(hyps=preds_clean, refs=refs_clean)
            clinical_metrics['chexbert_f1'] = float(chexbert_f1)
            print(f"CheXbert F1: {chexbert_f1:.4f}")
        except Exception as e:
            print(f"CheXbert error: {e}")
            clinical_metrics['chexbert_f1'] = None
    
    return clinical_metrics

vis_transforms = Compose([Resize(512), CenterCrop(448), ToTensor(), ExpandChannels()])
mimic_base = "/raid/den365/physionet.org/files/mimic-cxr-jpg/2.1.0"

def main():
    device = "cuda:0"
    model_path = pathlib.Path("./LLAVA/checkpoints/biovilt_mistral")
    
    print("Loading model...")
    tokenizer, model, _, _ = load_pretrained_model(
        model_path, model_base="mistralai/Mistral-7B-v0.1", 
        model_name="llava-lora-temporal", device=device
    )
    
    # PATCH: Fix cache_position error
    model = patch_model_forward(model)
    print("Model patched to handle cache_position")
    print("Model loaded")
    
    with open('./data/temporal_grounding/test.json') as f:
        test_data = json.load(f)
    
    print(f"="*60)
    print(f"EVALUATING ON FULL TEST SET: {len(test_data)} samples")
    print(f"="*60)
    
    predictions = []
    references = []
    ious = []
    change_matrix = defaultdict(lambda: defaultdict(int))
    all_results = []
    error_count = 0
    error_ids = []
    
    prompt_text = "<image> Compare this chest X-ray with the prior study and describe any interval changes with their locations."
    
    for idx, sample in enumerate(tqdm(test_data, desc="Evaluating")):
        try:
            curr_tensor = process_image_biovil([load_img(f"{mimic_base}/{sample['image']}")], vis_transforms).to(device, dtype=torch.bfloat16)
            prev_tensor = process_image_biovil([load_img(f"{mimic_base}/{sample['prev_image']}")], vis_transforms).to(device, dtype=torch.bfloat16)
            
            conv = conv_vicuna_v1.copy()
            conv.append_message(conv.roles[0], prompt_text)
            conv.append_message(conv.roles[1], None)
            input_ids = tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)
            
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids, 
                    images=curr_tensor, 
                    prev_images=prev_tensor, 
                    do_sample=False, 
                    max_new_tokens=256,
                    use_cache=True
                )
            
            pred = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip().replace("</s>", "")
            ref = sample['conversations'][1]['value']
            
            predictions.append(pred)
            references.append(ref)
            
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
            
            all_results.append({
                'id': sample['id'],
                'image': sample['image'],
                'prev_image': sample['prev_image'],
                'prediction': pred,
                'reference': ref,
                'pred_change': pred_change,
                'ref_change': ref_change
            })
            
            # Checkpoint
            if (idx + 1) % 5000 == 0:
                print(f"\n--- Checkpoint at {idx + 1} samples ---")
                print(f"Processed: {len(predictions)}, Errors: {error_count}")
                nlg = compute_nlg_metrics(predictions, references)
                print(f"BLEU-4: {nlg['bleu4']:.4f}, ROUGE-L: {nlg['rougeL']:.4f}, METEOR: {nlg['meteor']:.4f}")
                if ious:
                    iou_5 = sum(1 for x in ious if x > 0.5) / len(ious)
                    print(f"IoU>0.5: {100*iou_5:.1f}%")
                
                # Save checkpoint
                checkpoint = {
                    'samples_processed': len(predictions),
                    'errors': error_count,
                    'nlg_metrics': nlg,
                    'predictions': all_results[-1000:]  # Last 1000 to save space
                }
                with open(f'data/temporal_grounding/checkpoint_{idx+1}.json', 'w') as f:
                    json.dump(checkpoint, f)
                
        except Exception as e:
            error_count += 1
            error_ids.append(sample.get('id', f'idx_{idx}'))
            if error_count <= 10:
                print(f"\nError on {sample.get('id', 'unknown')}: {e}")
            continue
    
    # Check errors
    if error_count > 0:
        print(f"\n⚠️ WARNING: {error_count} errors ({100*error_count/len(test_data):.2f}%)")
    
    # Compute all metrics
    print(f"\n{'='*60}")
    print("Computing comprehensive metrics...")
    print(f"{'='*60}")
    
    nlg_metrics = compute_nlg_metrics(predictions, references)
    clinical_metrics = compute_clinical_metrics(predictions, references)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"FULL TEST SET RESULTS")
    print(f"{'='*60}")
    print(f"Total samples: {len(test_data)}")
    print(f"Successfully evaluated: {len(predictions)}")
    print(f"Errors: {error_count} ({100*error_count/len(test_data):.2f}%)")
    
    print(f"\n[NLG METRICS]")
    print(f"BLEU-1: {nlg_metrics['bleu1']:.4f}")
    print(f"BLEU-2: {nlg_metrics['bleu2']:.4f}")
    print(f"BLEU-3: {nlg_metrics['bleu3']:.4f}")
    print(f"BLEU-4: {nlg_metrics['bleu4']:.4f}")
    print(f"METEOR: {nlg_metrics['meteor']:.4f}")
    print(f"ROUGE-1: {nlg_metrics['rouge1']:.4f}")
    print(f"ROUGE-2: {nlg_metrics['rouge2']:.4f}")
    print(f"ROUGE-L: {nlg_metrics['rougeL']:.4f}")
    
    print(f"\n[CLINICAL EFFICACY METRICS]")
    if clinical_metrics.get('radgraph_f1') is not None:
        print(f"RadGraph F1: {clinical_metrics['radgraph_f1']:.4f}")
    else:
        print("RadGraph F1: N/A")
    if clinical_metrics.get('chexbert_f1') is not None:
        print(f"CheXbert F1: {clinical_metrics['chexbert_f1']:.4f}")
    else:
        print("CheXbert F1: N/A")
    
    print(f"\n[GROUNDING]")
    if ious:
        iou_mean = np.mean(ious)
        iou_above_5 = sum(1 for x in ious if x > 0.5)
        iou_above_3 = sum(1 for x in ious if x > 0.3)
        print(f"Samples with boxes: {len(ious)}")
        print(f"Avg IoU: {iou_mean:.3f}")
        print(f"IoU > 0.5: {iou_above_5}/{len(ious)} ({100*iou_above_5/len(ious):.1f}%)")
        print(f"IoU > 0.3: {iou_above_3}/{len(ious)} ({100*iou_above_3/len(ious):.1f}%)")
    else:
        iou_mean = 0
        iou_above_5 = 0
        iou_above_3 = 0
    
    print(f"\n[CHANGE DETECTION CONFUSION MATRIX]")
    classes = ['worsening', 'improvement', 'stable']
    print(f"{'GT/Pred':<12}", end="")
    for c in classes:
        print(f"{c:<12}", end="")
    print()
    for gt in classes:
        print(f"{gt:<12}", end="")
        for pred in classes:
            print(f"{change_matrix[gt][pred]:<12}", end="")
        print()
    
    print(f"\n[CHANGE DETECTION ACCURACY]")
    total_correct = 0
    total_samples_change = 0
    for gt in classes:
        total = sum(change_matrix[gt].values())
        correct = change_matrix[gt][gt]
        total_correct += correct
        total_samples_change += total
        if total > 0:
            print(f"{gt}: {correct}/{total} = {100*correct/total:.1f}%")
    
    overall_acc = 100*total_correct/total_samples_change if total_samples_change > 0 else 0
    print(f"Overall: {total_correct}/{total_samples_change} = {overall_acc:.1f}%")
    
    # Save results
    final_results = {
        'total_samples': len(test_data),
        'evaluated_samples': len(predictions),
        'errors': error_count,
        'error_rate': error_count/len(test_data) if test_data else 0,
        'nlg_metrics': nlg_metrics,
        'clinical_metrics': clinical_metrics,
        'grounding': {
            'samples_with_boxes': len(ious),
            'iou_mean': float(iou_mean) if ious else 0,
            'iou_above_5': iou_above_5/len(ious) if ious else 0,
            'iou_above_3': iou_above_3/len(ious) if ious else 0,
        },
        'change_detection': {
            'accuracy': total_correct/total_samples_change if total_samples_change > 0 else 0,
            'confusion_matrix': {k: dict(v) for k, v in change_matrix.items()},
        },
        'predictions': all_results
    }
    
    with open('data/temporal_grounding/full_test_results_clinical.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Results saved to data/temporal_grounding/full_test_results_clinical.json")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
