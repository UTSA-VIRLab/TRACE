import matplotlib.pyplot as plt
from PIL import Image

# Set style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

# =============================================================================
# Data
# =============================================================================
examples = [
    {
        'label': 'Worsening',
        'prior_path': 'paper_fig/worsening_prior.png',
        'current_path': 'paper_fig/worsening_current.png',
        'result_path': 'paper_fig/worsening_result.png',
        'gt': "Interval worsening of lung opacity [0.174,0.295,0.473,0.933] in right lung. Interval worsening of lung opacity [0.129,0.866,0.219,0.955] in right costophrenic angle. Interval worsening of lung opacity [0.513,0.277,0.808,0.960] in left lung. Stable atelectasis [0.513,0.277,0.808,0.960] in left lung. Stable atelectasis [0.536,0.683,0.808,0.960] in left lower lung zone. Interval worsening of lung opacity [0.732,0.857,0.821,0.946] in left costophrenic angle.",
        'output': "Interval worsening of lung opacity [0.188,0.290,0.464,0.929] in right lung. Interval worsening of lung opacity [0.188,0.661,0.464,0.929] in right lower lung zone. Interval worsening of lung opacity [0.330,0.491,0.460,0.692] in right hilar structures. Interval worsening of lung opacity [0.500,0.250,0.839,0.839] in left lung. Interval worsening of lung opacity [0.540,0.629,0.839,0.839] in left lower lung zone. Interval worsening of lung opacity [0.513,0.496,0.665,0.656] in left hilar structures."
    },
    {
        'label': 'Stable',
        'prior_path': 'paper_fig/stable_prior.png',
        'current_path': 'paper_fig/stable_current.png',
        'result_path': 'paper_fig/stable_result.png',
        'gt': "Stable atelectasis [0.237,0.094,0.589,0.531] in right lung. Stable finding [0.237,0.094,0.589,0.531] in right lung. Stable atelectasis [0.268,0.375,0.545,0.429] in right mid lung zone. Stable atelectasis [0.237,0.429,0.536,0.531] in right lower lung zone. Stable lung opacity [0.576,0.134,0.902,0.723] in left lung. Stable finding [0.576,0.134,0.902,0.723] in left lung. Stable lung opacity [0.621,0.143,0.893,0.375] in left upper lung zone. Stable lung opacity [0.625,0.375,0.902,0.500] in left mid lung zone. Stable finding [0.438,0.442,0.777,0.701] in cardiac silhouette.",
        'output': "Stable lung opacity [0.250,0.125,0.580,0.598] in right lung. Stable lung opacity [0.250,0.460,0.540,0.598] in right lower lung zone. Stable lung opacity [0.205,0.545,0.295,0.634] in right costophrenic angle. Stable lung opacity [0.580,0.170,0.871,0.750] in left lung. Stable lung opacity [0.580,0.522,0.871,0.750] in left lower lung zone. Stable lung opacity [0.826,0.688,0.915,0.777] in left costophrenic angle."
    },
    {
        'label': 'Improvement',
        'prior_path': 'paper_fig/improvement_prior.png',
        'current_path': 'paper_fig/improvement_current.png',
        'result_path': 'paper_fig/improvement_result.png',
        'gt': "Interval improvement of lung opacity [0.098,0.205,0.433,0.750] in right lung. Interval worsening of lung opacity [0.098,0.205,0.433,0.750] in right lung. Interval improvement of lung opacity [0.281,0.402,0.429,0.567] in right hilar structures. Interval improvement of lung opacity [0.478,0.219,0.768,0.638] in left lung. Interval worsening of lung opacity [0.478,0.219,0.768,0.638] in left lung. Interval improvement of lung opacity [0.478,0.411,0.616,0.518] in left hilar structures.",
        'output': "Interval improvement of lung opacity [0.089,0.219,0.433,0.759] in right lung. Interval improvement of lung opacity [0.263,0.411,0.415,0.576] in right hilar structures. Interval improvement of lung opacity [0.469,0.219,0.777,0.759] in left lung. Interval improvement of lung opacity [0.482,0.411,0.621,0.576] in left hilar structures."
    }
]

# =============================================================================
# Helper function
# =============================================================================
def resize_image(img, size=(300, 300)):
    return img.resize(size, Image.LANCZOS)

IMG_SIZE = (300, 300)

# =============================================================================
# Create figure
# =============================================================================
fig, axes = plt.subplots(3, 4, figsize=(24, 12), 
                          gridspec_kw={'width_ratios': [1, 1, 1, 2.5]})

col_titles = ['Prior Image', 'Current Image', 'Prediction', 'Ground Truth / Model Output']

for row_idx, ex in enumerate(examples):
    # Load and resize images to same size
    prior_img = resize_image(Image.open(ex['prior_path']).convert('L'), IMG_SIZE)
    current_img = resize_image(Image.open(ex['current_path']).convert('L'), IMG_SIZE)
    result_img = resize_image(Image.open(ex['result_path']), IMG_SIZE)
    
    # Column 0: Prior image
    axes[row_idx, 0].imshow(prior_img, cmap='gray')
    axes[row_idx, 0].axis('off')
    
    # Column 1: Current image
    axes[row_idx, 1].imshow(current_img, cmap='gray')
    axes[row_idx, 1].axis('off')
    
    # Column 2: Result image
    axes[row_idx, 2].imshow(result_img)
    axes[row_idx, 2].axis('off')
    
    # Column 3: Text (GT and Output as paragraphs)
    axes[row_idx, 3].axis('off')
    
    # GT paragraph
    gt_text = f"Ground Truth: {ex['gt']}"
    axes[row_idx, 3].text(0.0, 0.95, gt_text, fontsize=15, color='black',
                          transform=axes[row_idx, 3].transAxes, va='top', ha='left',
                          wrap=True, fontfamily='serif',
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#cccccc'))
    
    # Output paragraph
    output_text = f"Model Output: {ex['output']}"
    axes[row_idx, 3].text(0.0, 0.45, output_text, fontsize=15, color='black',
                          transform=axes[row_idx, 3].transAxes, va='top', ha='left',
                          wrap=True, fontfamily='serif',
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#cccccc'))
    
    # Row label
    axes[row_idx, 0].set_ylabel(ex['label'], fontsize=14, fontweight='bold',
                                 rotation=90, labelpad=15, va='center')

# Column titles
for col_idx, title in enumerate(col_titles):
    axes[0, col_idx].set_title(title, fontsize=13, fontweight='bold', pad=10)

plt.tight_layout()
plt.subplots_adjust(wspace=0.1, hspace=0.12)

plt.savefig('qualitative_results.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

print("Figure saved as 'qualitative_results.png'")