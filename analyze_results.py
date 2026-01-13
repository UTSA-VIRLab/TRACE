"""Complete analysis of TRACE results"""
import json
from collections import defaultdict
import numpy as np

with open('data/temporal_grounding/full_test_results_final.json') as f:
    results = json.load(f)

def get_change_type(text):
    text_lower = text.lower()
    if 'worsening' in text_lower or 'worse' in text_lower:
        return 'worsening'
    elif 'improvement' in text_lower or 'improved' in text_lower:
        return 'improvement'
    else:
        return 'stable'

def extract_anatomy(text):
    anatomies = ['right lung', 'left lung', 'right lower lung', 'left lower lung',
                 'cardiac silhouette', 'mediastinum', 'right hilar', 'left hilar']
    text_lower = text.lower()
    for anat in anatomies:
        if anat in text_lower:
            return anat
    return 'other'

# Build confusion matrix
classes = ['worsening', 'improvement', 'stable']
cm = defaultdict(lambda: defaultdict(int))
anatomy_stats = defaultdict(lambda: {'correct': 0, 'total': 0})

for pred_data in results['predictions']:
    pred = get_change_type(pred_data['prediction'])
    ref = get_change_type(pred_data['reference'])
    cm[ref][pred] += 1
    
    anatomy = extract_anatomy(pred_data['reference'])
    anatomy_stats[anatomy]['total'] += 1
    if pred == ref:
        anatomy_stats[anatomy]['correct'] += 1

print("="*60)
print("1. CONFUSION MATRIX")
print("="*60)
print(f"{'GT/Pred':<15} {'worsening':<12} {'improvement':<12} {'stable':<12}")
for gt in classes:
    print(f"{gt:<15} {cm[gt]['worsening']:<12} {cm[gt]['improvement']:<12} {cm[gt]['stable']:<12}")

print("\n" + "="*60)
print("2. PER-CLASS METRICS")
print("="*60)
print(f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")

for cls in classes:
    tp = cm[cls][cls]
    fp = sum(cm[other][cls] for other in classes if other != cls)
    fn = sum(cm[cls][other] for other in classes if other != cls)
    support = sum(cm[cls].values())
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"{cls:<15} {precision:>10.3f} {recall:>10.3f} {f1:>10.3f} {support:>10}")

print("\n" + "="*60)
print("3. PER-ANATOMY ACCURACY")
print("="*60)
print(f"{'Anatomy':<25} {'Accuracy':>10} {'Support':>10}")

sorted_stats = sorted(anatomy_stats.items(), key=lambda x: x[1]['total'], reverse=True)
for anatomy, stats in sorted_stats:
    if stats['total'] >= 50:
        acc = 100 * stats['correct'] / stats['total']
        print(f"{anatomy:<25} {acc:>9.1f}% {stats['total']:>10}")

print("\n" + "="*60)
print("4. SUMMARY FOR PAPER")
print("="*60)
print(f"Overall Change Accuracy: {results['change_detection']['accuracy']*100:.1f}%")
print(f"Grounding IoU > 0.5: {results['grounding']['iou_above_5']*100:.1f}%")
print(f"BLEU-4: {results['nlg_metrics']['bleu4']:.3f}")
print(f"ROUGE-L: {results['nlg_metrics']['rougeL']:.3f}")
print(f"RadGraph F1: {results['clinical_metrics']['radgraph_f1']:.3f}")
