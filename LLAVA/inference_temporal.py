#!/usr/bin/env python3
"""Simple inference script for temporal grounding model"""

import sys
import torch
import json
from PIL import Image
from pathlib import Path
from transformers import AutoTokenizer
from peft import PeftModel

# Add paths
sys.path.insert(0, '/raid/den365/RaDialog_v2/LLAVA')

from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token

# BioViL-T imports
from biovil_t.model import ImageModel
from biovil_t.pretrained import _download_biovil_t_image_model_weights
from biovil_t.types import ImageEncoderType
from torchvision.transforms import Compose, Resize, ToTensor, CenterCrop

class ExpandChannels:
    def __call__(self, data):
        return data.expand(3, -1, -1)

def create_transform(resize=512, crop=448):
    return Compose([Resize(resize), CenterCrop(crop), ToTensor(), ExpandChannels()])

def load_model(model_path, model_base="lmsys/vicuna-7b-v1.5"):
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
    
    print("Loading base model...")
    model = LlavaLlamaForCausalLM.from_pretrained(
        model_base,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Load vision tower (BioViL-T)
    print("Loading BioViL-T vision encoder...")
    biovilt_path = _download_biovil_t_image_model_weights()
    vision_tower = ImageModel(
        img_encoder_type=ImageEncoderType.RESNET50_MULTI_IMAGE,
        joint_feature_size=128,
        pretrained_model_path=biovilt_path
    )
    vision_tower = vision_tower.to(model.device)
    vision_tower.eval()
    for p in vision_tower.parameters():
        p.requires_grad = False
    model.model.vision_tower = vision_tower
    
    # Load mm_projector
    print("Loading projector weights...")
    non_lora_weights = torch.load(f"{model_path}/non_lora_trainables.bin", map_location="cpu")
    non_lora_weights = {k.replace("model.", ""): v for k, v in non_lora_weights.items()}
    
    # Build projector
    from torch import nn
    mm_projector = nn.Sequential(
        nn.Linear(512, 4096),
        nn.GELU(),
        nn.Linear(4096, 4096)
    )
    
    proj_weights = {k.replace("mm_projector.", "").replace("base_", ""): v for k, v in non_lora_weights.items() if "mm_projector" in k}
    mm_projector.load_state_dict(proj_weights)
    mm_projector = mm_projector.to(model.device, dtype=torch.float16)
    model.model.mm_projector = mm_projector
    
    # Load LoRA adapter
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, model_path)
    model = model.merge_and_unload()
    
    model.eval()
    print("Model loaded!")
    
    return model, tokenizer, vision_tower

def inference(model, tokenizer, vision_tower, image_path, prompt, device="cuda"):
    # Load and preprocess image
    transform = create_transform()
    image = Image.open(image_path).convert('L')  # Grayscale for X-ray
    image_tensor = transform(image).unsqueeze(0).to(device, dtype=torch.float32)
    
    # Get image features
    with torch.no_grad():
        image_features = vision_tower(image_tensor)
        image_features = image_features.patch_embeddings.flatten(2).transpose(1, 2).to(torch.float16)
        image_features = model.model.mm_projector(image_features)
    
    # Prepare prompt
    conv = conv_templates["v1"].copy()
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + prompt)
    conv.append_message(conv.roles[1], None)
    full_prompt = conv.get_prompt()
    
    input_ids = tokenizer_image_token(full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
    input_ids = input_ids.unsqueeze(0).to(device)
    
    # Insert image features into embeddings
    embed_tokens = model.model.model.embed_tokens
    inputs_embeds = embed_tokens(input_ids)
    
    # Find image token position and replace
    image_token_mask = input_ids == IMAGE_TOKEN_INDEX
    num_image_tokens = image_features.shape[1]
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("ASSISTANT:")[-1].strip()

if __name__ == "__main__":
    model_path = "/raid/den365/RaDialog_v2/LLAVA/checkpoints/temporal_grounding_v1"
    
    # Load test sample
    with open('/raid/den365/RaDialog_v2/data/temporal_grounding/test.json') as f:
        test_data = json.load(f)
    
    sample = test_data[0]
    mimic_base = '/raid/den365/physionet.org/files/mimic-cxr-jpg/2.1.0'
    image_path = f"{mimic_base}/{sample['image']}"
    
    print("=== Test Sample ===")
    print(f"Image: {sample['image']}")
    print(f"\nGround truth:\n{sample['conversations'][1]['value'][:400]}...")
    
    # Load model
    model, tokenizer, vision_tower = load_model(model_path)
    
    # Run inference
    prompt = "Compare this chest X-ray with the prior study and describe any interval changes with their locations."
    print("\nGenerating...")
    output = inference(model, tokenizer, vision_tower, image_path, prompt)
    
    print("\n=== Generated ===")
    print(output)
