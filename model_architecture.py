import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Polygon
from PIL import Image
import numpy as np

# Set style
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

def draw_cube(ax, x, y, w, h, depth, facecolor, edgecolor):
    """Draw a 3D cube to represent tensors"""
    front = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                            facecolor=facecolor, edgecolor=edgecolor, linewidth=2)
    ax.add_patch(front)
    
    top_x = [x, x + depth * 0.3, x + w + depth * 0.3, x + w]
    top_y = [y + h, y + h + depth * 0.3, y + h + depth * 0.3, y + h]
    top = Polygon(list(zip(top_x, top_y)), facecolor=facecolor, edgecolor=edgecolor, 
                  linewidth=2, alpha=0.85)
    ax.add_patch(top)
    
    right_x = [x + w, x + w + depth * 0.3, x + w + depth * 0.3, x + w]
    right_y = [y, y + depth * 0.3, y + h + depth * 0.3, y + h]
    right = Polygon(list(zip(right_x, right_y)), facecolor=facecolor, edgecolor=edgecolor, 
                    linewidth=2, alpha=0.7)
    ax.add_patch(right)

def draw_arrow(ax, start, end, color='#455a64', lw=2.5):
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color=color, lw=lw))

# =============================================================================
# Create figure - VERY WIDE for no overlaps
# =============================================================================
fig, ax = plt.subplots(1, 1, figsize=(32, 12))
ax.set_xlim(0, 32)
ax.set_ylim(0, 12)
ax.axis('off')

# Colors
color_encoder = '#1976d2'
color_encoder_dark = '#0d47a1'
color_feature = '#bbdefb'
color_concat = '#fff3e0'
color_concat_dark = '#ef6c00'
color_mlp = '#ffe0b2'
color_mlp_dark = '#e65100'
color_llm = '#a5d6a7'
color_llm_dark = '#2e7d32'
color_output = '#e1bee7'
color_output_dark = '#7b1fa2'
color_gt = '#ef9a9a'
color_gt_dark = '#c62828'

# =============================================================================
# INPUT IMAGES - Load actual images
# =============================================================================
# Load images
try:
    prior_img = np.array(Image.open('paper_fig/improvement_prior.png').convert('L').resize((120, 120)))
    current_img = np.array(Image.open('paper_fig/improvement_current.png').convert('L').resize((120, 120)))
except FileNotFoundError:
    print("Warning: Images not found. Using placeholders.")
    prior_img = np.random.randint(50, 200, (120, 120), dtype=np.uint8)
    current_img = np.random.randint(50, 200, (120, 120), dtype=np.uint8)

# Prior image
ax_prior = fig.add_axes([0.02, 0.55, 0.08, 0.35])
ax_prior.imshow(prior_img, cmap='gray')
ax_prior.axis('off')
ax_prior.set_title(r'Prior $I_p$' + '\n448×448', fontsize=12, fontweight='bold', pad=5)

# Current image
ax_current = fig.add_axes([0.02, 0.12, 0.08, 0.35])
ax_current.imshow(current_img, cmap='gray')
ax_current.axis('off')
ax_current.set_title(r'Current $I_c$' + '\n448×448', fontsize=12, fontweight='bold', pad=5)

# =============================================================================
# VISION ENCODER
# =============================================================================
encoder_box = FancyBboxPatch((4, 2.5), 3, 7, boxstyle="round,pad=0.1",
                              facecolor=color_encoder, edgecolor=color_encoder_dark, linewidth=3)
ax.add_patch(encoder_box)

ax.text(5.5, 7.5, 'Vision Encoder', ha='center', va='center', 
        fontsize=16, fontweight='bold', color='white')
ax.text(5.5, 6.5, 'BioViL-T (ViT)', ha='center', va='center', 
        fontsize=14, color='white')

# Frozen badge
frozen_badge = FancyBboxPatch((4.3, 3.0), 2.4, 0.8, boxstyle="round,pad=0.05",
                               facecolor='#e8eaf6', edgecolor='#5c6bc0', linewidth=2)
ax.add_patch(frozen_badge)
ax.text(5.5, 3.4, 'FROZEN', ha='center', va='center', 
        fontsize=12, fontweight='bold', color='#3949ab')

# Arrows to encoder
draw_arrow(ax, (3.2, 7.5), (4, 7), lw=3)
draw_arrow(ax, (3.2, 4.5), (4, 5), lw=3)

# Shared weights
ax.text(3.5, 6.0, 'Shared', ha='center', va='center', fontsize=10, color='#555', style='italic')

# =============================================================================
# FEATURE TENSORS f(Ip) and f(Ic)
# =============================================================================
draw_cube(ax, 8, 7, 1.2, 2, 0.5, color_feature, color_encoder_dark)
ax.text(8.6, 8.5, r'$f(I_p)$', ha='center', va='center', fontsize=14, fontweight='bold')
ax.text(8.6, 7.3, '196×512', ha='center', va='center', fontsize=10, color='#444')

draw_cube(ax, 8, 3, 1.2, 2, 0.5, color_feature, color_encoder_dark)
ax.text(8.6, 4.5, r'$f(I_c)$', ha='center', va='center', fontsize=14, fontweight='bold')
ax.text(8.6, 3.3, '196×512', ha='center', va='center', fontsize=10, color='#444')

draw_arrow(ax, (7, 7), (8, 8), lw=2)
draw_arrow(ax, (7, 5), (8, 4), lw=2)

# =============================================================================
# CONCATENATION
# =============================================================================
concat_circle = Circle((10.5, 6), 0.6, facecolor=color_concat, 
                        edgecolor=color_concat_dark, linewidth=3)
ax.add_patch(concat_circle)
ax.text(10.5, 6, '⊕', ha='center', va='center', fontsize=28, 
        fontweight='bold', color=color_concat_dark)
ax.text(10.5, 5.0, 'Concat', ha='center', va='center', fontsize=11, fontweight='bold')

draw_arrow(ax, (9.4, 7.8), (10.0, 6.4), lw=2)
draw_arrow(ax, (9.4, 4.2), (10.0, 5.6), lw=2)

# =============================================================================
# CONCATENATED FEATURES F = [f(Ip); f(Ic)]
# =============================================================================
draw_cube(ax, 11.8, 4.5, 1.2, 3, 0.5, color_concat, color_concat_dark)
ax.text(12.4, 7.0, r'$F = [f(I_p); f(I_c)]$', ha='center', va='center', fontsize=12, fontweight='bold')
ax.text(12.4, 5.0, '392×512', ha='center', va='center', fontsize=10, color='#444')

draw_arrow(ax, (11.1, 6), (11.8, 6), lw=2)

# =============================================================================
# MLP PROJECTOR
# =============================================================================
mlp_box = FancyBboxPatch((14, 4), 2.5, 4, boxstyle="round,pad=0.1",
                          facecolor=color_mlp, edgecolor=color_mlp_dark, linewidth=3)
ax.add_patch(mlp_box)

ax.text(15.25, 7.0, 'MLP', ha='center', va='center', 
        fontsize=16, fontweight='bold', color='#333')
ax.text(15.25, 6.0, '2-Layer', ha='center', va='center', 
        fontsize=12, color='#555')
ax.text(15.25, 5.0, '512 → 4096', ha='center', va='center', 
        fontsize=11, color='#666')

draw_arrow(ax, (13.2, 6), (14, 6), lw=2)

# =============================================================================
# LLM DECODER
# =============================================================================
llm_box = FancyBboxPatch((17.5, 3), 4, 6, boxstyle="round,pad=0.1",
                          facecolor=color_llm, edgecolor=color_llm_dark, linewidth=3)
ax.add_patch(llm_box)

ax.text(19.5, 8.0, 'LLM Decoder', ha='center', va='center', 
        fontsize=16, fontweight='bold', color='#1b5e20')
ax.text(19.5, 7.0, 'Vicuna-7B', ha='center', va='center', 
        fontsize=14, fontweight='bold', color='#2e7d32')

# LoRA badge
lora_badge = FancyBboxPatch((18.0, 3.5), 3.0, 1.2, boxstyle="round,pad=0.05",
                             facecolor='#c8e6c9', edgecolor=color_llm_dark, linewidth=2)
ax.add_patch(lora_badge)
ax.text(19.5, 4.1, 'LoRA r=128, α=256', ha='center', va='center', 
        fontsize=11, fontweight='bold', color='#1b5e20')

draw_arrow(ax, (16.5, 6), (17.5, 6), lw=2)

# =============================================================================
# GROUND TRUTH (Training Target) - BELOW LLM
# =============================================================================
gt_box = FancyBboxPatch((17.5, 0.3), 4, 2.2, boxstyle="round,pad=0.1",
                         facecolor=color_gt, edgecolor=color_gt_dark, linewidth=2)
ax.add_patch(gt_box)

ax.text(19.5, 2.1, 'Ground Truth (Training)', ha='center', va='center', 
        fontsize=12, fontweight='bold', color='#333')
# Actual GT from improvement case
ax.text(19.5, 1.0, '"Interval improvement of lung opacity\n[0.098, 0.205, 0.433, 0.750] in right lung"', 
        ha='center', va='center', fontsize=9, color='#444', style='italic')

# Dashed arrow from GT to LLM
ax.annotate('', xy=(19.5, 3.0), xytext=(19.5, 2.5),
            arrowprops=dict(arrowstyle='->', color=color_gt_dark, lw=2, linestyle='--'))
ax.text(20.3, 2.75, 'Loss', ha='left', va='center', fontsize=11, 
        color=color_gt_dark, fontweight='bold')

# =============================================================================
# OUTPUT
# =============================================================================
output_box = FancyBboxPatch((22.5, 3), 4.5, 6, boxstyle="round,pad=0.1",
                             facecolor=color_output, edgecolor=color_output_dark, linewidth=3)
ax.add_patch(output_box)

ax.text(24.75, 8.2, 'Output', ha='center', va='center', 
        fontsize=16, fontweight='bold', color='#4a148c')

# Text section
ax.text(24.75, 7.0, 'Text:', ha='center', va='center', 
        fontsize=12, fontweight='bold', color='#666')
ax.text(24.75, 6.0, '"Interval improvement\nof lung opacity\nin right lung"', ha='center', va='center', 
        fontsize=10, color='#333')

# Box section - actual model output coordinates
ax.text(24.75, 4.5, 'Bounding Box:', ha='center', va='center', 
        fontsize=12, fontweight='bold', color='#666')
ax.text(24.75, 3.6, '[0.089, 0.219, 0.433, 0.759]', ha='center', va='center', 
        fontsize=11, color='#c62828', fontfamily='monospace', fontweight='bold')

draw_arrow(ax, (21.5, 6), (22.5, 6), lw=2)

# =============================================================================
# LEGEND
# =============================================================================
legend_y = 11.0
legend_items = [
    (4, color_encoder, color_encoder_dark, 'Frozen (0 trainable)'),
    (11, color_mlp, color_mlp_dark, 'Trainable MLP (~4M params)'),
    (19, color_llm, color_llm_dark, 'Trainable LoRA (~30M params)'),
]

for x, facecolor, edgecolor, label in legend_items:
    box = FancyBboxPatch((x, legend_y - 0.3), 0.8, 0.6, boxstyle="round,pad=0.02",
                          facecolor=facecolor, edgecolor=edgecolor, linewidth=2)
    ax.add_patch(box)
    ax.text(x + 1.0, legend_y, label, ha='left', va='center', fontsize=12)

plt.savefig('model_architecture.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("Figure saved: model_architecture.png")