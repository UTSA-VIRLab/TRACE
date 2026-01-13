"""
Evaluation script for LLM ablation models (Mistral, Llama3.1, Gemma)
Fixed dtype handling - everything in bfloat16
"""
import json
import pathlib
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from skimage import io
from tqdm import tqdm
import re
from torchvision.transforms import Compose, Resize, ToTensor, CenterCrop

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from health_multimodal.image.model.pretrained import get_biovil_t_image_encoder

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
    return Image.fromarray(remap_to_uint8(img)).convert("L")

def extract_boxes(text):
    boxes = []
    for match in re.finditer(r'<box>([\d.,]+)</box>', text):
        coords = [float(x) for x in match.group(1).split(',')]
        if len(coords) == 4:
            boxes.append(coords)
    return boxes

def compute_iou(box1, box2):
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

def get_change_type(text):
    t = text.lower()
    if any(w in t for w in ['worsening', 'worse', 'increased']):
        return 'worsening'
    elif any(w in t for w in ['improvement', 'improved', 'decreased', 'resolved']):
        return 'improvement'
    return 'stable'

class MLPProjector(nn.Module):
    def __init__(self, vision_dim, llm_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(vision_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim)
        )
    def forward(self, x):
        return self.proj(x)

def evaluate_model(checkpoint_path, llm_base, output_path, device="cuda:0"):
    print(f"\n{'='*60}")
    print(f"Evaluating: {checkpoint_path}")
    print(f"LLM Base: {llm_base}")
    print(f"{'='*60}\n")
    
    vis_transforms = Compose([Resize(512), CenterCrop(448), ToTensor(), ExpandChannels()])
    mimic_base = "/raid/den365/physionet.org/files/mimic-cxr-jpg/2.1.0"
    
    # Load BioViL-T
    print("Loading BioViL-T encoder...")
    biovil = get_biovil_t_image_encoder().to(device).eval()
    
    @torch.no_grad()
    def encode_image(img):
        tensor = vis_transforms(img).unsqueeze(0).to(device)
        out = biovil(tensor)
        emb = out.patch_embeddings.flatten(2).permute(0, 2, 1)  # [1, 196, 512]
        return emb
    
    # Load tokenizer
    print(f"Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(llm_base, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load LLM
    print(f"Loading LLM...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    llm = AutoModelForCausalLM.from_pretrained(
        llm_base, quantization_config=bnb_config,
        device_map={"": device}, trust_remote_code=True,
    )
    llm_dim = llm.config.hidden_size
    
    # Load LoRA
    print(f"Loading LoRA weights...")
    llm = PeftModel.from_pretrained(llm, checkpoint_path)
    llm.eval()
    
    # Load projector
    print("Loading projector...")
    projector = MLPProjector(512, llm_dim).to(device).to(torch.bfloat16)
    projector.load_state_dict(torch.load(f"{checkpoint_path}/projector.pt", map_location=device))
    projector.eval()
    
    print("Model loaded!")
    
    # Test dtype
    test_img = Image.fromarray(np.random.randint(0,255,(512,512),dtype=np.uint8)).convert("L")
    test_feat = encode_image(test_img)
    print(f"Encoder output dtype: {test_feat.dtype}")
    test_proj = projector(test_feat.to(torch.bfloat16))
    print(f"Projector output dtype: {test_proj.dtype}")
    test_ids = tokenizer("test", return_tensors="pt").input_ids.to(device)
    test_emb = llm.get_input_embeddings()(test_ids)
    print(f"LLM embedding dtype: {test_emb.dtype}")
    
    # Load test data
    with open('./data/temporal_grounding/test.json') as f:
        test_data = json.load(f)
    print(f"Test samples: {len(test_data)}")
    
    predictions, ious = [], []
    change_matrix = {c: {c2: 0 for c2 in ['worsening','improvement','stable']} 
                     for c in ['worsening','improvement','stable']}
    errors = 0
    
    for idx, sample in enumerate(tqdm(test_data, desc="Evaluating")):
        try:
            curr_img = load_img(f"{mimic_base}/{sample['image']}")
            prev_img = load_img(f"{mimic_base}/{sample['prev_image']}")
            
            with torch.no_grad():
                curr_feat = encode_image(curr_img)
                prev_feat = encode_image(prev_img)
                combined = torch.cat([prev_feat, curr_feat], dim=1).to(torch.bfloat16)
                projected = projector(combined)
            
            prompt = sample['conversations'][0]['value'].replace("<image>\n", "").replace("<image>", "")
            text_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            text_emb = llm.get_input_embeddings()(text_ids)
            
            # Match dtypes
            projected = projected.to(text_emb.dtype)
            inputs_embeds = torch.cat([projected, text_emb], dim=1)
            
            with torch.no_grad():
                outputs = llm.generate(
                    inputs_embeds=inputs_embeds,
                    max_new_tokens=256,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            
            pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
            ref = sample['conversations'][1]['value']
            
            pred_change = get_change_type(pred)
            ref_change = get_change_type(ref)
            change_matrix[ref_change][pred_change] += 1
            
            pred_boxes, ref_boxes = extract_boxes(pred), extract_boxes(ref)
            if pred_boxes and ref_boxes:
                ious.append(compute_iou(pred_boxes[0], ref_boxes[0]))
            
            predictions.append({'prediction': pred, 'reference': ref})
            
            if (idx+1) % 1000 == 0:
                correct = sum(change_matrix[c][c] for c in change_matrix)
                total = sum(sum(v.values()) for v in change_matrix.values())
                print(f"\n[{idx+1}] Acc: {100*correct/total:.1f}%")
                
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"Error {idx}: {e}")
    
    # Results
    print(f"\n{'='*60}\nRESULTS\n{'='*60}")
    
    correct = sum(change_matrix[c][c] for c in change_matrix)
    total = sum(sum(v.values()) for v in change_matrix.values())
    print(f"Change Acc: {100*correct/total:.1f}%")
    
    print(f"\nPer-class:")
    for cls in ['worsening', 'improvement', 'stable']:
        t = sum(change_matrix[cls].values())
        c = change_matrix[cls][cls]
        print(f"  {cls}: {c}/{t} = {100*c/t:.1f}%" if t > 0 else f"  {cls}: 0")
    
    if ious:
        print(f"\nGrounding:")
        print(f"  Mean IoU: {np.mean(ious):.3f}")
        print(f"  IoU>0.5: {100*sum(i>0.5 for i in ious)/len(ious):.1f}%")
    
    # NLG
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        refs = [[p['reference'].split()] for p in predictions]
        preds = [p['prediction'].split() for p in predictions]
        bleu4 = corpus_bleu(refs, preds, weights=(0.25,0.25,0.25,0.25), 
                           smoothing_function=SmoothingFunction().method1)
        print(f"\nBLEU-4: {bleu4:.3f}")
    except: pass
    
    # Save
    results = {
        'change_acc': correct/total if total else 0,
        'confusion': {k: dict(v) for k,v in change_matrix.items()},
        'iou_mean': float(np.mean(ious)) if ious else 0,
        'iou_above_5': sum(i>0.5 for i in ious)/len(ious) if ious else 0,
        'errors': errors,
    }
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--llm_base', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    evaluate_model(args.checkpoint, args.llm_base, args.output, args.device)
