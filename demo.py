"""
TRACE Demo: Temporal Radiology with Anatomical Change Explanation
Interactive demo for grounded temporal change detection in chest X-rays
FIXED VERSION
"""

import gradio as gr
import torch
import re
import os
import sys
import json
import random
from PIL import Image, ImageDraw, ImageFont
from skimage import io
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor, CenterCrop
import pathlib

sys.path.insert(0, '/raid/den365/RaDialog_v2')

from LLAVA.llava.constants import IMAGE_TOKEN_INDEX
from LLAVA.llava.conversation import conv_vicuna_v1
from LLAVA.llava.mm_utils import tokenizer_image_token, process_image_biovil
from LLAVA.llava.model.builder import load_pretrained_model
from functools import wraps

# Colors for different change types
COLORS = {
    'worsening': '#FF4444',  # Red
    'improvement': '#44FF44',  # Green
    'stable': '#4444FF',  # Blue
    'default': '#FFFF44'  # Yellow
}

class ExpandChannels:
    def __call__(self, data):
        return data.expand(3, -1, -1)

def patch_model_forward(model):
    """Patch model forward to ignore unknown kwargs"""
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

def load_img_from_path(path):
    """Load image from file path"""
    img = io.imread(path)
    img = Image.fromarray(remap_to_uint8(img)).convert("L")
    return img

def load_img_from_pil(pil_img):
    """Convert PIL image to grayscale"""
    return pil_img.convert("L")

def extract_boxes_and_text(text):
    """Extract bounding boxes and associated text from model output"""
    pattern = r'([^<]*)<box>([\d.,]+)</box>'
    results = []
    
    for match in re.finditer(pattern, text):
        description = match.group(1).strip()
        coords = [float(x) for x in match.group(2).split(',')]
        if len(coords) == 4:
            # Determine change type
            desc_lower = description.lower()
            if 'worsening' in desc_lower or 'worse' in desc_lower or 'increased' in desc_lower:
                change_type = 'worsening'
            elif 'improvement' in desc_lower or 'improved' in desc_lower or 'decreased' in desc_lower or 'resolved' in desc_lower:
                change_type = 'improvement'
            elif 'stable' in desc_lower or 'no change' in desc_lower:
                change_type = 'stable'
            else:
                change_type = 'default'
            
            results.append({
                'description': description,
                'box': coords,
                'change_type': change_type
            })
    
    return results

def draw_boxes_on_image(image, findings, image_size=448):
    """Draw bounding boxes on image with labels"""
    # Convert to RGB for colored boxes
    if image.mode == 'L':
        image = image.convert('RGB')
    
    # Resize to display size
    display_size = 512
    image = image.resize((display_size, display_size), Image.LANCZOS)
    
    draw = ImageDraw.Draw(image)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        font = ImageFont.load_default()
        small_font = font
    
    for i, finding in enumerate(findings):
        box = finding['box']
        color = COLORS[finding['change_type']]
        
        # Scale box coordinates to display size
        x1 = int(box[0] * display_size)
        y1 = int(box[1] * display_size)
        x2 = int(box[2] * display_size)
        y2 = int(box[3] * display_size)
        
        # Draw box
        for offset in range(3):  # Thicker border
            draw.rectangle([x1-offset, y1-offset, x2+offset, y2+offset], outline=color)
        
        # Draw label background
        label = f"{i+1}. {finding['change_type'].upper()}"
        bbox = draw.textbbox((x1, y1-20), label, font=small_font)
        draw.rectangle([bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2], fill=color)
        draw.text((x1, y1-20), label, fill='black', font=small_font)
    
    return image

def format_output_text(findings, raw_output):
    """Format findings as readable HTML"""
    if not findings:
        html = "<p>No findings with bounding boxes detected.</p>"
    else:
        html = "<div style='font-family: Arial, sans-serif;'>"
        html += "<h3>📋 Detected Findings:</h3>"
        
        for i, finding in enumerate(findings):
            color = COLORS[finding['change_type']]
            change_emoji = {
                'worsening': '🔴 ↑ WORSENING',
                'improvement': '🟢 ↓ IMPROVING',
                'stable': '🔵 → STABLE',
                'default': '🟡 FINDING'
            }[finding['change_type']]
            
            html += f"""
            <div style='margin: 10px 0; padding: 10px; border-left: 4px solid {color}; background: #f5f5f5;'>
                <strong>{change_emoji}</strong><br>
                {finding['description']}<br>
                <small style='color: #666;'>
                    📍 Box: [{finding['box'][0]:.3f}, {finding['box'][1]:.3f}, {finding['box'][2]:.3f}, {finding['box'][3]:.3f}]
                </small>
            </div>
            """
        
        html += "</div>"
    
    html += f"<hr><h4>📝 Raw Model Output:</h4><pre style='background:#f0f0f0;padding:10px;overflow-x:auto;'>{raw_output}</pre>"
    return html

class TRACEDemo:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.vis_transforms = Compose([Resize(512), CenterCrop(448), ToTensor(), ExpandChannels()])
        self.mimic_base = "/raid/den365/physionet.org/files/mimic-cxr-jpg/2.1.0"
        
    def load_model(self):
        if self.model is not None:
            return "✅ Model already loaded!"
        
        print("Loading TRACE model...")
        model_path = pathlib.Path("/raid/den365/RaDialog_v2/LLAVA/checkpoints/temporal_grounding_v1")
        
        self.tokenizer, self.model, _, _ = load_pretrained_model(
            model_path, 
            model_base="lmsys/vicuna-7b-v1.5", 
            model_name="llava-lora-temporal", 
            device=self.device
        )
        self.model = patch_model_forward(self.model)
        print("Model loaded successfully!")
        return "✅ Model loaded successfully!"
    
    def predict(self, prior_image, current_image):
        """Run inference on image pair"""
        if self.model is None:
            return None, None, "⚠️ Please load the model first by clicking 'Load Model'!"
        
        if prior_image is None or current_image is None:
            return None, None, "⚠️ Please upload both prior and current images!"
        
        try:
            # Process images - FIXED: correct order now!
            prior_pil = load_img_from_pil(prior_image)
            current_pil = load_img_from_pil(current_image)
            
            # FIXED: curr_tensor uses current_pil, prev_tensor uses prior_pil
            curr_tensor = process_image_biovil([current_pil], self.vis_transforms).to(self.device, dtype=torch.bfloat16)
            prev_tensor = process_image_biovil([prior_pil], self.vis_transforms).to(self.device, dtype=torch.bfloat16)
            
            # Prepare prompt
            prompt_text = "<image> Compare this chest X-ray with the prior study and describe any interval changes with their locations."
            conv = conv_vicuna_v1.copy()
            conv.append_message(conv.roles[0], prompt_text)
            conv.append_message(conv.roles[1], None)
            
            input_ids = tokenizer_image_token(
                conv.get_prompt(), 
                self.tokenizer, 
                IMAGE_TOKEN_INDEX, 
                return_tensors='pt'
            ).unsqueeze(0).to(self.device)
            
            # Generate
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids, 
                    images=curr_tensor, 
                    prev_images=prev_tensor, 
                    do_sample=False, 
                    max_new_tokens=256,
                    use_cache=True
                )
            
            prediction = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip().replace("</s>", "")
            
            # Extract findings and draw boxes
            findings = extract_boxes_and_text(prediction)
            
            # Draw boxes on current image
            current_with_boxes = draw_boxes_on_image(current_pil.copy(), findings)
            
            # Format output
            output_html = format_output_text(findings, prediction)
            
            return current_with_boxes, output_html, "✅ Inference complete!"
            
        except Exception as e:
            import traceback
            error_msg = f"❌ Error: {str(e)}\n{traceback.format_exc()}"
            return None, error_msg, error_msg
    
    def load_example(self, example_idx):
        """Load an example from test set"""
        test_path = "/raid/den365/RaDialog_v2/data/temporal_grounding/test.json"
        
        with open(test_path) as f:
            test_data = json.load(f)
        
        # Get sample
        sample = test_data[example_idx]
        
        prev_path = os.path.join(self.mimic_base, sample['prev_image'])
        curr_path = os.path.join(self.mimic_base, sample['image'])
        reference = sample['conversations'][1]['value']
        
        prev_img = load_img_from_path(prev_path)
        curr_img = load_img_from_path(curr_path)
        
        return prev_img, curr_img, f"<b>Ground Truth:</b><br><pre>{reference}</pre>"

def create_demo():
    demo_app = TRACEDemo()
    
    with gr.Blocks(
        title="TRACE Demo",
        theme=gr.themes.Soft(),
        css="""
        .container { max-width: 1200px; margin: auto; }
        .header { text-align: center; padding: 20px; }
        .stats { background: #f0f0f0; padding: 15px; border-radius: 8px; margin: 10px 0; }
        """
    ) as demo:
        
        gr.HTML("""
        <div class="header">
            <h1>🔬 TRACE: Temporal Radiology with Anatomical Change Explanation</h1>
            <p style='font-size: 1.1em; color: #555;'>
                Upload a <b>prior</b> and <b>current</b> chest X-ray to detect and localize temporal changes
            </p>
        </div>
        <div class="stats">
            <b>Model Performance (Test Set):</b><br>
            📊 Overall Change Accuracy: <b>48.0%</b> | 
            🔴 Worsening: 37.4% | 
            🟢 Improvement: 26.3% | 
            🔵 Stable: 67.4%<br>
            📍 Grounding (IoU > 0.5): <b>90.2%</b> | 
            📝 BLEU-4: 0.260 | ROUGE-L: 0.494
        </div>
        """)
        
        with gr.Row():
            load_btn = gr.Button("🚀 Load Model", variant="primary", scale=1)
            status = gr.Textbox(label="Status", value="⏳ Model not loaded - click 'Load Model' to start", interactive=False, scale=2)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML("<h3>📥 Input Images</h3>")
                prior_img = gr.Image(label="Prior Study (Earlier)", type="pil", height=300)
                current_img = gr.Image(label="Current Study (Later)", type="pil", height=300)
                predict_btn = gr.Button("🔍 Analyze Changes", variant="primary", size="lg")
                
                gr.HTML("<h4>📂 Load Test Examples</h4>")
                with gr.Row():
                    example_idx = gr.Slider(0, 1000, step=1, value=0, label="Example Index")
                    load_example_btn = gr.Button("Load Example")
                ground_truth = gr.HTML(label="Ground Truth")
            
            with gr.Column(scale=1):
                gr.HTML("<h3>📤 Results</h3>")
                output_img = gr.Image(label="Current Study with Detected Changes", height=400)
                output_text = gr.HTML(label="Detected Findings")
        
        gr.HTML("""
        <div style='text-align:center; padding: 20px; color: #666;'>
            <hr>
            <p><b>TRACE</b> - Temporal Radiology with Anatomical Change Explanation</p>
            <p>🔴 Worsening | 🟢 Improvement | 🔵 Stable</p>
        </div>
        """)
        
        # Event handlers
        load_btn.click(fn=demo_app.load_model, outputs=[status])
        predict_btn.click(
            fn=demo_app.predict, 
            inputs=[prior_img, current_img], 
            outputs=[output_img, output_text, status]
        )
        load_example_btn.click(
            fn=demo_app.load_example,
            inputs=[example_idx],
            outputs=[prior_img, current_img, ground_truth]
        )
    
    return demo

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True
    )
