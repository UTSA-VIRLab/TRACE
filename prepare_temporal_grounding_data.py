#!/usr/bin/env python3
"""
Prepare Temporal Grounding Dataset for TemporalGroundNet
Uses OFFICIAL Chest ImaGenome splits (patient-disjoint, aligned with MIMIC-CXR)
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter
import pandas as pd
from tqdm import tqdm

def parse_comparison_cue(cue):
    """Parse comparison cue like 'comparison|yes|improved' -> 'improved'"""
    parts = cue.split('|')
    if len(parts) >= 3 and parts[0] == 'comparison' and parts[1] == 'yes':
        return parts[2]
    return None

def extract_temporal_annotations(scene_graph):
    """Extract temporal annotations with bboxes from scene graph."""
    annotations = []
    objects = {obj['bbox_name']: obj for obj in scene_graph.get('objects', [])}
    
    for attr in scene_graph.get('attributes', []):
        bbox_name = attr.get('bbox_name')
        if not bbox_name or bbox_name not in objects:
            continue
            
        obj = objects[bbox_name]
        x1, y1, x2, y2 = obj['x1'], obj['y1'], obj['x2'], obj['y2']
        bbox_norm = [x1/224, y1/224, x2/224, y2/224]
        
        comparison_cues = attr.get('comparison_cues', [])
        phrases = attr.get('phrases', [])
        attributes_list = attr.get('attributes', [])
        
        for i, cue_list in enumerate(comparison_cues):
            for cue in cue_list:
                change_label = parse_comparison_cue(cue)
                if change_label:
                    findings = []
                    if i < len(attributes_list):
                        for a in attributes_list[i]:
                            if 'anatomicalfinding|yes|' in a:
                                findings.append(a.split('|')[-1])
                    
                    phrase = phrases[i] if i < len(phrases) else ""
                    annotations.append({
                        'bbox_name': bbox_name,
                        'bbox': bbox_norm,
                        'original_bbox': [obj.get('original_x1', x1), obj.get('original_y1', y1),
                                         obj.get('original_x2', x2), obj.get('original_y2', y2)],
                        'findings': findings,
                        'change_label': change_label,
                        'phrase': phrase.strip()
                    })
    return annotations

def format_grounded_report(annotations):
    """Format annotations into grounded report string."""
    if not annotations:
        return "No significant interval change compared to prior."
    
    change_map = {'improved': 'Interval improvement of', 'worsened': 'Interval worsening of', 'no change': 'Stable'}
    
    # Deduplicate
    seen = set()
    sentences = []
    for ann in annotations:
        key = (ann['bbox_name'], ann['change_label'], tuple(ann['findings'][:1]))
        if key in seen:
            continue
        seen.add(key)
        
        change_prefix = change_map.get(ann['change_label'], 'Stable')
        location = ann['bbox_name'].replace('_', ' ')
        finding = ann['findings'][0].replace('_', ' ') if ann['findings'] else 'finding'
        bbox_str = ','.join([f"{x:.3f}" for x in ann['bbox']])
        sentences.append(f"{change_prefix} {finding} <box>{bbox_str}</box> in {location}.")
    
    return ' '.join(sentences)

def load_official_splits(splits_dir):
    """Load official Chest ImaGenome splits."""
    splits = {}
    for split_name, filename in [('train', 'train.csv'), ('val', 'valid.csv'), ('test', 'test.csv')]:
        df = pd.read_csv(splits_dir / filename)
        splits[split_name] = set(df['subject_id'].unique())
        print(f"{split_name}: {len(splits[split_name])} patients")
    return splits

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chest_imagenome_path', type=str, default='/raid/den365/dataset/chest-imagenome/1.0.0')
    parser.add_argument('--output_path', type=str, default='./data/temporal_grounding')
    parser.add_argument('--max_samples_per_split', type=int, default=None)
    args = parser.parse_args()
    
    chest_imagenome = Path(args.chest_imagenome_path)
    scene_graph_dir = chest_imagenome / 'silver_dataset' / 'scene_graph'
    splits_dir = chest_imagenome / 'silver_dataset' / 'splits'
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Loading official splits...")
    splits = load_official_splits(splits_dir)
    
    print("\nLoading scene graphs...")
    files = list(scene_graph_dir.glob("*.json"))
    print(f"Found {len(files)} scene graph files")
    
    # Group by patient
    patient_studies = defaultdict(list)
    print("Grouping studies by patient...")
    for f in tqdm(files):
        try:
            with open(f) as fp:
                data = json.load(fp)
            patient_id = data.get('patient_id')
            if patient_id:
                patient_studies[patient_id].append({
                    'image_id': data.get('image_id'),
                    'study_id': data.get('study_id'),
                    'StudyOrder': data.get('StudyOrder', 0),
                    'data': data
                })
        except:
            continue
    
    print(f"Found {len(patient_studies)} unique patients")
    
    # Create temporal pairs with official splits
    print("\nCreating temporal grounding pairs...")
    split_samples = {'train': [], 'val': [], 'test': []}
    
    for patient_id, studies in tqdm(patient_studies.items()):
        if len(studies) < 2:
            continue
        
        # Determine split
        split_name = None
        for sn in ['train', 'val', 'test']:
            if patient_id in splits[sn]:
                split_name = sn
                break
        if not split_name:
            continue
        
        if args.max_samples_per_split and len(split_samples[split_name]) >= args.max_samples_per_split:
            continue
        
        studies_sorted = sorted(studies, key=lambda x: x['StudyOrder'])
        
        for i, current in enumerate(studies_sorted):
            if current['StudyOrder'] <= 1 or i == 0:
                continue
            
            prior = studies_sorted[i-1]
            annotations = extract_temporal_annotations(current['data'])
            if not annotations:
                continue
            
            sample = {
                'id': f"{patient_id}_{current['study_id']}",
                'current_image': current['image_id'],
                'prior_image': prior['image_id'],
                'conversations': [
                    {'from': 'human', 'value': '<prior><current> Compare this chest X-ray with the prior study and describe any interval changes with their locations.'},
                    {'from': 'gpt', 'value': format_grounded_report(annotations)}
                ],
                'boxes': [a['bbox'] for a in annotations],
                'change_labels': [a['change_label'] for a in annotations],
                'bbox_names': [a['bbox_name'] for a in annotations],
                'findings': [a['findings'] for a in annotations]
            }
            split_samples[split_name].append(sample)
    
    # Save
    total = 0
    for split_name, samples in split_samples.items():
        with open(output_path / f'{split_name}.json', 'w') as f:
            json.dump(samples, f, indent=2)
        n_ann = sum(len(s['boxes']) for s in samples)
        print(f"{split_name}: {len(samples)} samples, {n_ann} annotations")
        total += len(samples)
    
    # Stats
    all_changes = [c for s in split_samples.values() for sample in s for c in sample['change_labels']]
    print(f"\nTotal: {total} samples")
    print("Change distribution:", Counter(all_changes))
    print(f"Output: {output_path}")

if __name__ == '__main__':
    main()
