"""
Ablation study for embedding norms.

This module analyzes the L2 norms of entity and relation embeddings across different models.
For relations: saves norms with labels and creates comparative visualizations.
For entities: creates distribution plots without individual labels.
"""
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from argparse import Namespace
import glob

from kge.data import load_entity_relation_dicts, load_triples
from .stability_utils import load_model_from_checkpoint
import matplotlib.colors as mcolors


def load_relation_labels(data_dir: str, use_inverse: bool = False) -> Dict[int, str]:
    """
    Load relation labels from id2name_rel.txt and relation2id.txt.
    
    Args:
        data_dir: Path to data directory
        use_inverse: Whether to include inverse relations
        
    Returns:
        Dictionary mapping relation IDs to human-readable labels
    """
    relation_labels = {}
    
    # First load relation2id to get the mapping
    relation2id_path = os.path.join(data_dir, 'relation2id.txt')
    relation2id = {}
    if os.path.exists(relation2id_path):
        with open(relation2id_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split('\t')
                    if len(parts) == 2:
                        rel_code, rel_id = parts
                        relation2id[rel_code] = int(rel_id)
    
    # Then load id2name_rel.txt to get human-readable names
    id2name_path = os.path.join(data_dir, 'id2name_rel.txt')
    if os.path.exists(id2name_path):
        with open(id2name_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split('\t')
                    if len(parts) == 2:
                        rel_id_str, rel_name = parts
                        relation_labels[int(rel_id_str)] = rel_name
    
    # Handle inverse relations if needed
    if use_inverse:
        num_base_relations = len(relation2id)
        for rel_id in range(num_base_relations):
            if rel_id in relation_labels:
                # Inverse relation gets the same label with _inv suffix
                relation_labels[rel_id + num_base_relations] = relation_labels[rel_id] + '_inv'
    
    if not relation_labels:
        print(f"[WARN] Could not load relation labels from {data_dir}, using generic labels")
    
    return relation_labels


def count_triples_per_relation(data_dir: str, use_inverse: bool = False) -> Dict[int, int]:
    """
    Count the number of triples per relation in the training set.
    For inverse relations, the count is the same as the base relation.
    
    Args:
        data_dir: Path to data directory
        use_inverse: Whether inverse relations are used
        
    Returns:
        Dictionary mapping relation IDs to triple counts
    """
    triple_counts = {}
    
    # Load train triples
    train_path = os.path.join(data_dir, 'train.txt')
    if not os.path.exists(train_path):
        print(f"[WARN] train.txt not found in {data_dir}")
        return triple_counts
    
    train_triples = load_triples(train_path)
    
    # Load relation2id for counting
    entity2id, relation2id = load_entity_relation_dicts(data_dir)
    
    # Count triples per base relation
    for h, r, t in train_triples:
        rel_id = relation2id.get(r, -1)
        if rel_id >= 0:
            triple_counts[rel_id] = triple_counts.get(rel_id, 0) + 1
    
    # For inverse relations, copy the count from the base relation
    if use_inverse:
        num_base_relations = len(relation2id)
        for rel_id in range(num_base_relations):
            if rel_id in triple_counts:
                # Inverse relation has the same count
                triple_counts[rel_id + num_base_relations] = triple_counts[rel_id]
    
    print(f"[INFO] Counted triples for {len(triple_counts)} relations (use_inverse={use_inverse})")
    return triple_counts


def count_entity_degrees(data_dir: str) -> Dict[int, int]:
    """
    Count the degree (number of appearances) of each entity in the training set.
    Degree = number of times entity appears as head or tail.
    
    Args:
        data_dir: Path to data directory
        
    Returns:
        Dictionary mapping entity IDs to degrees
    """
    entity_degrees = {}
    
    # Load train triples
    train_path = os.path.join(data_dir, 'train.txt')
    if not os.path.exists(train_path):
        print(f"[WARN] train.txt not found in {data_dir}")
        return entity_degrees
    
    train_triples = load_triples(train_path)
    
    # Load entity2id for counting
    entity2id, _ = load_entity_relation_dicts(data_dir)
    
    # Count degrees
    for h, r, t in train_triples:
        h_id = entity2id.get(h, -1)
        t_id = entity2id.get(t, -1)
        
        if h_id >= 0:
            entity_degrees[h_id] = entity_degrees.get(h_id, 0) + 1
        if t_id >= 0:
            entity_degrees[t_id] = entity_degrees.get(t_id, 0) + 1
    
    print(f"[INFO] Counted degrees for {len(entity_degrees)} entities")
    return entity_degrees


def extract_embeddings(model: torch.nn.Module) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract entity and relation embeddings from a model.
    
    Args:
        model: Trained KGE model
        
    Returns:
        Tuple of (entity_embeddings, relation_embeddings) as numpy arrays
    """
    # Get embeddings based on model type
    if hasattr(model, 'entity_emb'):
        entity_emb = model.entity_emb.weight.data.cpu().numpy()
    elif hasattr(model, 'entity_embeddings'):
        entity_emb = model.entity_embeddings.weight.data.cpu().numpy()
    else:
        raise ValueError("Model does not have entity embeddings")
    
    if hasattr(model, 'relation_emb'):
        relation_emb = model.relation_emb.weight.data.cpu().numpy()
    elif hasattr(model, 'relation_embeddings'):
        relation_emb = model.relation_embeddings.weight.data.cpu().numpy()
    else:
        raise ValueError("Model does not have relation embeddings")
    
    return entity_emb, relation_emb


def compute_norms(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute L2 norms of embeddings.
    
    Args:
        embeddings: Array of shape (n, d) where n is number of embeddings and d is dimension
        
    Returns:
        Array of shape (n,) containing L2 norms
    """
    return np.linalg.norm(embeddings, axis=1)


def analyze_relation_norms(
    model_dirs: List[str],
    data_dir: str,
    output_dir: str,
    model_family: str,
    norm_method: str = 'none',
    device: str = 'cpu'
) -> Dict:
    """
    Analyze relation embedding norms across multiple models.
    
    Args:
        model_dirs: List of model checkpoint directories
        data_dir: Path to data directory
        output_dir: Directory to save results
        model_family: Model family name (e.g., 'DistMult')
        norm_method: Normalization method - 'none', 'minmax', 'minmax_entity_only', 'zscore_entity_only', 'ratio_entity'
        device: Device to load models on
        
    Returns:
        Dictionary containing norm analysis results
    """
    # Determine use_inverse from first model config
    use_inverse = False
    if model_dirs:
        try:
            _, config = load_model_from_checkpoint(model_dirs[0], device)
            use_inverse = config.use_inverse
            print(f"[INFO] use_inverse={use_inverse}")
        except:
            pass
    
    # Load relation labels
    relation_labels = load_relation_labels(data_dir, use_inverse)
    print(f"[INFO] Loaded {len(relation_labels)} relation labels")
    
    # Count triples per relation
    triple_counts = count_triples_per_relation(data_dir, use_inverse)
    
    # Collect norms from all models
    all_relation_norms = {}
    all_entity_norms = {}  # For normalization
    model_names = []
    
    for model_dir in model_dirs:
        try:
            print(f"[INFO] Loading model from {model_dir}")
            model, config = load_model_from_checkpoint(model_dir, device)
            
            # Extract embeddings
            entity_emb, relation_emb = extract_embeddings(model)
            
            # Compute norms
            relation_norms = compute_norms(relation_emb)
            entity_norms = compute_norms(entity_emb)
            
            # Normalize if requested
            if norm_method == 'minmax':
                # Combine all norms to find global min/max
                all_norms = np.concatenate([entity_norms, relation_norms])
                min_norm = all_norms.min()
                max_norm = all_norms.max()
                
                # Normalize to [0, 1]
                if max_norm > min_norm:
                    relation_norms = (relation_norms - min_norm) / (max_norm - min_norm)
                    entity_norms = (entity_norms - min_norm) / (max_norm - min_norm)
                
                print(f"[INFO] MinMax normalized (entities+relations) from [{min_norm:.3f}, {max_norm:.3f}] to [0, 1]")
            
            elif norm_method == 'minmax_entity_only':
                # Use entity norms only for min/max
                min_norm = entity_norms.min()
                max_norm = entity_norms.max()
                
                # Apply to both entities and relations
                if max_norm > min_norm:
                    entity_norms = (entity_norms - min_norm) / (max_norm - min_norm)
                    relation_norms = (relation_norms - min_norm) / (max_norm - min_norm)
                
                print(f"[INFO] MinMax normalized (entity-based) from [{min_norm:.3f}, {max_norm:.3f}] to [0, 1]")
            
            elif norm_method == 'zscore_entity_only':
                # Use entity norms only for mean/std
                mean_norm = entity_norms.mean()
                std_norm = entity_norms.std()
                
                # Apply z-score to both entities and relations
                if std_norm > 0:
                    entity_norms = (entity_norms - mean_norm) / std_norm
                    relation_norms = (relation_norms - mean_norm) / std_norm
                
                print(f"[INFO] Z-score normalized (entity-based) with mean={mean_norm:.3f}, std={std_norm:.3f}")
            
            elif norm_method == 'ratio_entity':
                # Divide relation norms by mean entity norm
                mean_entity_norm = entity_norms.mean()
                
                if mean_entity_norm > 0:
                    relation_norms = relation_norms / mean_entity_norm
                    entity_norms = entity_norms / mean_entity_norm
                
                print(f"[INFO] Ratio normalized (divided by mean entity norm={mean_entity_norm:.3f})")
            
            elif norm_method == 'max_relation':
                # Divide relation norms by max relation norm (per model)
                max_relation_norm = relation_norms.max()
                
                if max_relation_norm > 0:
                    relation_norms = relation_norms / max_relation_norm
                    # Entity norms are NOT normalized in this method
                
                print(f"[INFO] Max relation normalized (divided by max relation norm={max_relation_norm:.3f})")
            
            # Store results
            model_name = os.path.basename(model_dir)
            model_names.append(model_name)
            all_relation_norms[model_name] = relation_norms.tolist()
            all_entity_norms[model_name] = entity_norms
            
            print(f"[INFO] {model_name}: {len(relation_norms)} relations, "
                  f"norm range [{relation_norms.min():.3f}, {relation_norms.max():.3f}]")
            
        except Exception as e:
            print(f"[ERROR] Failed to process {model_dir}: {e}")
            continue
    
    # Create list of all relation data sorted by norm
    relation_data_all = {}
    for model_name, norms in all_relation_norms.items():
        relation_data = []
        for rel_id, norm_value in enumerate(norms):
            label = relation_labels.get(rel_id, f"relation_{rel_id}")
            count = triple_counts.get(rel_id, 0)
            relation_data.append({
                "relation_id": rel_id,
                "label": label,
                "norm": norm_value,
                "triple_count": count
            })
        
        # Sort by norm (descending)
        relation_data.sort(key=lambda x: x['norm'], reverse=True)
        relation_data_all[model_name] = relation_data
    
    # Create model family directory
    family_dir = os.path.join(output_dir, model_family)
    os.makedirs(family_dir, exist_ok=True)
    
    # Save all relation data to one JSON file
    suffix = f'_{norm_method}' if norm_method != 'none' else ''
    json_path = os.path.join(family_dir, f'relation_norms_sorted{suffix}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(relation_data_all, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved sorted relation norms to {json_path}")
    
    return {
        'relation_norms': all_relation_norms,
        'entity_norms': all_entity_norms,
        'relation_labels': relation_labels,
        'triple_counts': triple_counts,
        'model_names': model_names,
        'model_family': model_family,
        'norm_method': norm_method
    }


def visualize_relation_norms(
    all_relation_norms: Dict[str, List[float]],
    relation_labels: Dict[int, str],
    triple_counts: Dict[int, int],
    model_names: List[str],
    output_dir: str,
    model_family: str,
    norm_method: str = 'none'
):
    """
    Create visualization of relation norms across models.
    Relations are sorted by norm (descending).
    Bar colors use a gradient based on triple count.
    All subplots share the same y-axis scale.
    
    Args:
        all_relation_norms: Dictionary mapping model names to relation norms
        relation_labels: Dictionary mapping relation IDs to labels
        triple_counts: Dictionary mapping relation IDs to triple counts
        model_names: List of model names (for ordering)
        output_dir: Directory to save plots
    """
    if not model_names:
        print("[WARN] No models to visualize")
        return
    
    num_models = len(model_names)
    
    # Determine global y-axis limits
    all_norms_flat = []
    for norms in all_relation_norms.values():
        all_norms_flat.extend(norms)
    # Allow negative values for z-score normalization
    y_min = min(all_norms_flat)
    y_max = max(all_norms_flat)
    margin = 0.1 * (y_max - y_min) if y_max != y_min else 0.1
    y_min -= margin
    y_max += margin
    
    # Create colormap based on triple counts
    all_counts = [triple_counts.get(i, 0) for i in range(len(next(iter(all_relation_norms.values()))))]
    max_count = max(all_counts) if all_counts else 1
    min_count = min([c for c in all_counts if c > 0]) if any(c > 0 for c in all_counts) else 0
    
    # Use a colormap from light (few triples) to dark (many triples)
    cmap = plt.cm.YlOrRd  # Yellow to Orange to Red
    norm_colors = mcolors.Normalize(vmin=min_count, vmax=max_count)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 8), sharey=True)
    
    # Handle case of single model
    if num_models == 1:
        axes = [axes]
    
    for idx, model_name in enumerate(model_names):
        ax = axes[idx]
        norms = all_relation_norms[model_name]
        
        # Sort relations by norm (descending)
        num_relations = len(norms)
        relation_data = [(i, norms[i], relation_labels.get(i, f"R{i}"), triple_counts.get(i, 0)) 
                        for i in range(num_relations)]
        relation_data.sort(key=lambda x: x[1], reverse=True)
        
        sorted_rel_ids = [x[0] for x in relation_data]
        sorted_norms = [x[1] for x in relation_data]
        sorted_labels = [x[2] for x in relation_data]
        sorted_counts = [x[3] for x in relation_data]
        
        # Create colors based on triple counts
        colors = [cmap(norm_colors(count)) for count in sorted_counts]
        
        # Create bar plot
        x_positions = list(range(num_relations))
        bars = ax.bar(x_positions, sorted_norms, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Customize subplot
        ax.set_xlabel('Relations (sorted by norm)', fontsize=12)
        ax.set_ylabel('L2 Norm' if idx == 0 else '', fontsize=12)
        ax.set_title(f'{model_name}', fontsize=14, fontweight='bold')
        ax.set_ylim([y_min, y_max])
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels on top of bars (only show if space permits)
        if num_relations <= 30:
            for bar_idx, (bar, norm) in enumerate(zip(bars, sorted_norms)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{norm:.1f}',
                       ha='center', va='bottom', fontsize=7, rotation=90)
        
        # Set x-tick labels - show ALL labels
        ax.set_xticks(x_positions)
        # Adjust font size based on number of relations
        label_fontsize = max(5, min(8, 200 // num_relations))
        ax.set_xticklabels(sorted_labels, rotation=90, fontsize=label_fontsize, ha='right')
        
        # Add colorbar for the last subplot
        if idx == num_models - 1:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_colors)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, pad=0.02)
            cbar.set_label('Triple Count', rotation=270, labelpad=20, fontsize=10)
    
    title_suffix = f' ({norm_method})' if norm_method != 'none' else ''
    plt.suptitle(f'Relation Embedding Norms{title_suffix} (sorted by norm)', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save figure in model family directory
    family_dir = os.path.join(output_dir, model_family)
    os.makedirs(family_dir, exist_ok=True)
    suffix = f'_{norm_method}' if norm_method != 'none' else ''
    output_path = os.path.join(family_dir, f'relation_norms_sorted{suffix}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[INFO] Saved relation norms visualization to {output_path}")
    
    plt.close()


def visualize_entity_norms(
    all_entity_norms: Dict[str, np.ndarray],
    model_names: List[str],
    output_dir: str,
    model_family: str,
    norm_method: str = 'none'
):
    """
    Visualize entity embedding norm distributions across multiple models.
    Creates distribution plots without individual labels.
    
    Args:
        all_entity_norms: Dictionary mapping model names to entity norm arrays
        model_names: List of model names
        output_dir: Directory to save results
        model_family: Model family name (e.g., 'DistMult')
        normalize: Whether norms are normalized
    """
    
    if not model_names:
        print("[WARN] No models to visualize")
        return
    
    num_models = len(model_names)
    
    # Determine global x-axis limits for histogram
    all_norms_flat = np.concatenate([norms for norms in all_entity_norms.values()])
    x_min = all_norms_flat.min()
    x_max = all_norms_flat.max()
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 6), sharey=True)
    
    # Handle case of single model
    if num_models == 1:
        axes = [axes]
    
    # Determine bins for consistency across plots
    bins = np.linspace(x_min, x_max, 50)
    
    for idx, model_name in enumerate(model_names):
        ax = axes[idx]
        norms = all_entity_norms[model_name]
        
        # Create histogram
        ax.hist(norms, bins=bins, color='coral', alpha=0.7, edgecolor='black')
        
        # Add statistics
        mean_norm = norms.mean()
        std_norm = norms.std()
        median_norm = np.median(norms)
        
        # Add vertical lines for statistics
        ax.axvline(mean_norm, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_norm:.2f}')
        ax.axvline(median_norm, color='blue', linestyle='--', linewidth=2, label=f'Median: {median_norm:.2f}')
        
        # Customize subplot
        ax.set_xlabel('L2 Norm', fontsize=12)
        ax.set_ylabel('Frequency' if idx == 0 else '', fontsize=12)
        ax.set_title(f'{model_name}', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.legend(fontsize=10, loc='upper right')
        
        # Add text box with statistics
        stats_text = f'Std: {std_norm:.2f}\nMin: {norms.min():.2f}\nMax: {norms.max():.2f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
               fontsize=9)
    
    title_suffix = f' ({norm_method})' if norm_method != 'none' else ''
    plt.suptitle(f'Entity Embedding Norm Distributions{title_suffix}', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save figure in model family directory
    family_dir = os.path.join(output_dir, model_family)
    os.makedirs(family_dir, exist_ok=True)
    suffix = f'_{norm_method}' if norm_method != 'none' else ''
    output_path = os.path.join(family_dir, f'entity_norm_distributions{suffix}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[INFO] Saved entity norm distributions to {output_path}")
    
    plt.close()


def analyze_dimension_distributions(
    model_dirs: List[str],
    data_dir: str,
    output_dir: str,
    model_family: str,
    norm_method: str = 'none',
    device: str = 'cpu'
) -> Dict:
    """
    Analyze dimension-wise value distributions for entity and relation embeddings.
    Creates histograms for each dimension across models.
    
    Args:
        model_dirs: List of model checkpoint directories
        data_dir: Path to data directory
        output_dir: Directory to save results
        model_family: Model family name
        norm_method: Normalization method
        device: Device to load models on
        
    Returns:
        Dictionary containing dimension-wise statistics
    """
    # Collect embeddings from all models
    all_entity_embs = {}
    all_relation_embs = {}
    model_names = []
    
    for model_dir in model_dirs:
        try:
            print(f"[INFO] Loading embeddings from {model_dir}")
            model, config = load_model_from_checkpoint(model_dir, device)
            
            # Extract embeddings
            entity_emb, relation_emb = extract_embeddings(model)
            
            # Apply normalization if requested (same as in analyze_relation_norms)
            entity_norms = compute_norms(entity_emb)
            relation_norms = compute_norms(relation_emb)
            
            if norm_method == 'minmax':
                all_norms = np.concatenate([entity_norms, relation_norms])
                min_norm = all_norms.min()
                max_norm = all_norms.max()
                if max_norm > min_norm:
                    entity_emb = (entity_emb - entity_emb.min()) / (entity_emb.max() - entity_emb.min())
                    relation_emb = (relation_emb - relation_emb.min()) / (relation_emb.max() - relation_emb.min())
            
            elif norm_method == 'minmax_entity_only':
                min_val = entity_emb.min()
                max_val = entity_emb.max()
                if max_val > min_val:
                    entity_emb = (entity_emb - min_val) / (max_val - min_val)
                    relation_emb = (relation_emb - min_val) / (max_val - min_val)
            
            elif norm_method == 'zscore_entity_only':
                mean_val = entity_emb.mean()
                std_val = entity_emb.std()
                if std_val > 0:
                    entity_emb = (entity_emb - mean_val) / std_val
                    relation_emb = (relation_emb - mean_val) / std_val
            
            elif norm_method == 'ratio_entity':
                mean_entity_norm = entity_norms.mean()
                if mean_entity_norm > 0:
                    entity_emb = entity_emb / mean_entity_norm
                    relation_emb = relation_emb / mean_entity_norm
            
            elif norm_method == 'max_relation':
                # Divide by max relation norm (per model)
                max_relation_norm = relation_norms.max()
                if max_relation_norm > 0:
                    relation_emb = relation_emb / max_relation_norm
                    # Entity embeddings are NOT normalized in this method
            
            model_name = os.path.basename(model_dir)
            model_names.append(model_name)
            all_entity_embs[model_name] = entity_emb
            all_relation_embs[model_name] = relation_emb
            
            print(f"[INFO] {model_name}: entity shape {entity_emb.shape}, relation shape {relation_emb.shape}")
            
        except Exception as e:
            print(f"[ERROR] Failed to process {model_dir}: {e}")
            continue
    
    if not model_names:
        print("[WARN] No models processed")
        return {}
    
    # Get embedding dimensions
    first_model = model_names[0]
    entity_dim = all_entity_embs[first_model].shape[1]
    relation_dim = all_relation_embs[first_model].shape[1]
    
    # Save dimension-wise statistics to JSON
    dim_stats = {
        'entity_dimensions': {},
        'relation_dimensions': {}
    }
    
    for model_name in model_names:
        dim_stats['entity_dimensions'][model_name] = []
        dim_stats['relation_dimensions'][model_name] = []
        
        for d in range(entity_dim):
            values = all_entity_embs[model_name][:, d]
            dim_stats['entity_dimensions'][model_name].append({
                'dimension': d,
                'mean': float(values.mean()),
                'std': float(values.std()),
                'min': float(values.min()),
                'max': float(values.max())
            })
        
        for d in range(relation_dim):
            values = all_relation_embs[model_name][:, d]
            dim_stats['relation_dimensions'][model_name].append({
                'dimension': d,
                'mean': float(values.mean()),
                'std': float(values.std()),
                'min': float(values.min()),
                'max': float(values.max())
            })
    
    # Save to JSON
    family_dir = os.path.join(output_dir, model_family)
    os.makedirs(family_dir, exist_ok=True)
    suffix = f'_{norm_method}' if norm_method != 'none' else ''
    json_path = os.path.join(family_dir, f'dimension_stats{suffix}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(dim_stats, f, indent=2)
    print(f"[INFO] Saved dimension statistics to {json_path}")
    
    return {
        'entity_embeddings': all_entity_embs,
        'relation_embeddings': all_relation_embs,
        'model_names': model_names,
        'entity_dim': entity_dim,
        'relation_dim': relation_dim
    }


def visualize_dimension_distributions(
    dim_data: Dict,
    output_dir: str,
    model_family: str,
    norm_method: str = 'none',
    embedding_type: str = 'entity'
):
    """
    Visualize dimension-wise value distributions.
    One row per dimension, one column per model.
    
    Args:
        dim_data: Dictionary from analyze_dimension_distributions
        output_dir: Directory to save plots
        model_family: Model family name
        norm_method: Normalization method
        embedding_type: 'entity' or 'relation'
    """
    if embedding_type == 'entity':
        all_embs = dim_data['entity_embeddings']
        num_dims = dim_data['entity_dim']
    else:
        all_embs = dim_data['relation_embeddings']
        num_dims = dim_data['relation_dim']
    
    model_names = dim_data['model_names']
    num_models = len(model_names)
    
    if num_models == 0 or num_dims == 0:
        print("[WARN] No data to visualize")
        return
    
    # Limit to first 10 dimensions to avoid too large plots
    max_dims = min(10, num_dims)
    
    # Create figure: rows = dimensions, columns = models
    fig, axes = plt.subplots(max_dims, num_models, 
                            figsize=(4 * num_models, 2.5 * max_dims),
                            squeeze=False)
    
    for dim_idx in range(max_dims):
        for model_idx, model_name in enumerate(model_names):
            ax = axes[dim_idx, model_idx]
            
            # Get values for this dimension
            values = all_embs[model_name][:, dim_idx]
            
            # Create histogram
            ax.hist(values, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
            
            # Add statistics
            mean_val = values.mean()
            std_val = values.std()
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, label=f'Mean={mean_val:.2f}')
            
            # Labels
            if dim_idx == 0:
                ax.set_title(f'{model_name}', fontsize=10, fontweight='bold')
            if model_idx == 0:
                ax.set_ylabel(f'Dim {dim_idx}', fontsize=9, fontweight='bold')
            
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.tick_params(labelsize=7)
            
            if dim_idx == 0 and model_idx == 0:
                ax.legend(fontsize=7, loc='upper right')
    
    title_suffix = f' ({norm_method})' if norm_method != 'none' else ''
    plt.suptitle(f'{embedding_type.capitalize()} Dimension Distributions{title_suffix} - {model_family}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    family_dir = os.path.join(output_dir, model_family)
    os.makedirs(family_dir, exist_ok=True)
    suffix = f'_{norm_method}' if norm_method != 'none' else ''
    output_path = os.path.join(family_dir, f'{embedding_type}_dimension_distributions{suffix}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[INFO] Saved {embedding_type} dimension distributions to {output_path}")
    
    plt.close()


def visualize_entity_norms_bar(
    all_entity_norms: Dict[str, np.ndarray],
    entity_degrees: Dict[int, int],
    model_names: List[str],
    output_dir: str,
    model_family: str,
    norm_method: str = 'none'
):
    """
    Create bar plot of entity norms sorted by norm value.
    Bar colors use a gradient based on entity degree.
    
    Args:
        all_entity_norms: Dictionary mapping model names to entity norm arrays
        entity_degrees: Dictionary mapping entity IDs to degrees
        model_names: List of model names
        output_dir: Directory to save results
        model_family: Model family name (e.g., 'DistMult')
        norm_method: Normalization method used
    """
    if not model_names:
        print("[WARN] No models to visualize")
        return
    
    num_models = len(model_names)
    
    # Determine global y-axis limits
    all_norms_flat = np.concatenate([norms for norms in all_entity_norms.values()])
    y_min = all_norms_flat.min()
    y_max = all_norms_flat.max()
    margin = 0.1 * (y_max - y_min) if y_max != y_min else 0.1
    y_min -= margin
    y_max += margin
    
    # Get entity degrees
    all_degrees = [entity_degrees.get(i, 0) for i in range(len(next(iter(all_entity_norms.values()))))]
    max_degree = max(all_degrees) if all_degrees else 1
    min_degree = min([d for d in all_degrees if d > 0]) if any(d > 0 for d in all_degrees) else 0
    
    # Use colormap from light (low degree) to dark (high degree)
    cmap = plt.cm.Blues
    norm_colors = mcolors.Normalize(vmin=min_degree, vmax=max_degree)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, num_models, figsize=(8 * num_models, 8), sharey=True)
    
    # Handle case of single model
    if num_models == 1:
        axes = [axes]
    
    for idx, model_name in enumerate(model_names):
        ax = axes[idx]
        norms = all_entity_norms[model_name]
        num_entities = len(norms)
        
        # Create entity data: (entity_id, norm, degree)
        entity_data = [(i, norms[i], entity_degrees.get(i, 0)) for i in range(num_entities)]
        
        # Sort by norm (descending)
        entity_data.sort(key=lambda x: x[1], reverse=True)
        
        sorted_entity_ids = [x[0] for x in entity_data]
        sorted_norms = [x[1] for x in entity_data]
        sorted_degrees = [x[2] for x in entity_data]
        
        # Create colors based on degrees
        colors = [cmap(norm_colors(degree)) for degree in sorted_degrees]
        
        # Create bar plot
        x_positions = list(range(num_entities))
        bars = ax.bar(x_positions, sorted_norms, color=colors, alpha=0.8, edgecolor='none', linewidth=0)
        
        # Customize subplot
        ax.set_xlabel('Entities (sorted by norm)', fontsize=12)
        ax.set_ylabel('L2 Norm' if idx == 0 else '', fontsize=12)
        ax.set_title(f'{model_name}', fontsize=14, fontweight='bold')
        ax.set_ylim([y_min, y_max])
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # No x-tick labels for entities (too many)
        ax.set_xticks([])
        
        # Add colorbar for the last subplot
        if idx == num_models - 1:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_colors)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, pad=0.02)
            cbar.set_label('Entity Degree', rotation=270, labelpad=20, fontsize=10)
    
    title_suffix = f' ({norm_method})' if norm_method != 'none' else ''
    plt.suptitle(f'Entity Embedding Norms{title_suffix} (sorted by norm)', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save figure in model family directory
    family_dir = os.path.join(output_dir, model_family)
    os.makedirs(family_dir, exist_ok=True)
    suffix = f'_{norm_method}' if norm_method != 'none' else ''
    output_path = os.path.join(family_dir, f'entity_norms_bar{suffix}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[INFO] Saved entity norms bar plot to {output_path}")
    
    plt.close()


def create_global_plots(
    families_results: Dict,
    output_dir: str,
    data_name: str,
    norm_method: str,
    entity_degrees: Dict[int, int]
):
    """
    Create global plots that aggregate results across multiple model families.
    Each plot type gets one global visualization.
    
    Args:
        families_results: Dictionary mapping family names to their results
        output_dir: Output directory
        data_name: Dataset name
        norm_method: Normalization method
        entity_degrees: Entity degree dictionary
    """
    if not families_results:
        return
    
    family_names = list(families_results.keys())
    num_families = len(family_names)
    
    # For each family, take only the first model to avoid clutter
    # Global relation norms plot
    print("[INFO] Creating global relation norms plot...")
    fig, axes = plt.subplots(1, num_families, figsize=(6 * num_families, 8), sharey=True)
    if num_families == 1:
        axes = [axes]
    
    for idx, family_name in enumerate(family_names):
        ax = axes[idx]
        results = families_results[family_name]
        
        # Take first model only
        model_names = results['model_names']
        if not model_names:
            continue
        
        first_model = model_names[0]
        norms = results['relation_norms'][first_model]
        relation_labels = results['relation_labels']
        triple_counts = results['triple_counts']
        
        # Sort relations by norm
        num_relations = len(norms)
        relation_data = [(i, norms[i], relation_labels.get(i, f"R{i}"), triple_counts.get(i, 0)) 
                        for i in range(num_relations)]
        relation_data.sort(key=lambda x: x[1], reverse=True)
        
        sorted_norms = [x[1] for x in relation_data]
        sorted_labels = [x[2] for x in relation_data]
        sorted_counts = [x[3] for x in relation_data]
        
        # Colors based on triple counts
        max_count = max(sorted_counts) if sorted_counts else 1
        min_count = min([c for c in sorted_counts if c > 0]) if any(c > 0 for c in sorted_counts) else 0
        cmap = plt.cm.YlOrRd
        norm_colors = mcolors.Normalize(vmin=min_count, vmax=max_count)
        colors = [cmap(norm_colors(count)) for count in sorted_counts]
        
        # Bar plot
        x_positions = list(range(num_relations))
        ax.bar(x_positions, sorted_norms, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Relations (sorted by norm)', fontsize=12)
        ax.set_ylabel('L2 Norm' if idx == 0 else '', fontsize=12)
        ax.set_title(f'{family_name}', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # X-tick labels
        ax.set_xticks(x_positions)
        label_fontsize = max(5, min(8, 200 // num_relations))
        ax.set_xticklabels(sorted_labels, rotation=90, fontsize=label_fontsize, ha='right')
    
    title_suffix = f' ({norm_method})' if norm_method != 'none' else ''
    plt.suptitle(f'Relation Embedding Norms{title_suffix} - All Families',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    suffix = f'_{norm_method}' if norm_method != 'none' else ''
    output_path = os.path.join(output_dir, f'global_relation_norms{suffix}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[INFO] Saved global relation norms to {output_path}")
    plt.close()
    
    # Global entity norm distributions
    print("[INFO] Creating global entity norm distributions...")
    fig, axes = plt.subplots(1, num_families, figsize=(6 * num_families, 6), sharey=True)
    if num_families == 1:
        axes = [axes]
    
    # Determine global bins
    all_norms = []
    for family_name in family_names:
        results = families_results[family_name]
        if results['model_names']:
            first_model = results['model_names'][0]
            all_norms.extend(results['entity_norms'][first_model])
    
    all_norms = np.array(all_norms)
    bins = np.linspace(all_norms.min(), all_norms.max(), 50)
    
    for idx, family_name in enumerate(family_names):
        ax = axes[idx]
        results = families_results[family_name]
        
        if not results['model_names']:
            continue
        
        first_model = results['model_names'][0]
        norms = results['entity_norms'][first_model]
        
        ax.hist(norms, bins=bins, color='coral', alpha=0.7, edgecolor='black')
        
        mean_norm = norms.mean()
        median_norm = np.median(norms)
        std_norm = norms.std()
        
        ax.axvline(mean_norm, color='red', linestyle='--', linewidth=2, label=f'Mean={mean_norm:.2f}')
        ax.axvline(median_norm, color='blue', linestyle=':', linewidth=2, label=f'Median={median_norm:.2f}')
        
        ax.set_xlabel('L2 Norm', fontsize=12)
        ax.set_ylabel('Frequency' if idx == 0 else '', fontsize=12)
        ax.set_title(f'{family_name}', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.legend(fontsize=10, loc='upper right')
    
    title_suffix = f' ({norm_method})' if norm_method != 'none' else ''
    plt.suptitle(f'Entity Embedding Norm Distributions{title_suffix} - All Families',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'global_entity_norm_distributions{suffix}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[INFO] Saved global entity distributions to {output_path}")
    plt.close()
    
    # Global entity norms bar
    print("[INFO] Creating global entity norms bar...")
    fig, axes = plt.subplots(1, num_families, figsize=(8 * num_families, 8), sharey=True)
    if num_families == 1:
        axes = [axes]
    
    for idx, family_name in enumerate(family_names):
        ax = axes[idx]
        results = families_results[family_name]
        
        if not results['model_names']:
            continue
        
        first_model = results['model_names'][0]
        norms = results['entity_norms'][first_model]
        num_entities = len(norms)
        
        # Create entity data sorted by norm
        entity_data = [(i, norms[i], entity_degrees.get(i, 0)) for i in range(num_entities)]
        entity_data.sort(key=lambda x: x[1], reverse=True)
        
        sorted_norms = [x[1] for x in entity_data]
        sorted_degrees = [x[2] for x in entity_data]
        
        # Colors based on degrees
        max_degree = max(sorted_degrees) if sorted_degrees else 1
        min_degree = min([d for d in sorted_degrees if d > 0]) if any(d > 0 for d in sorted_degrees) else 0
        cmap = plt.cm.Blues
        norm_colors = mcolors.Normalize(vmin=min_degree, vmax=max_degree)
        colors = [cmap(norm_colors(degree)) for degree in sorted_degrees]
        
        x_positions = list(range(num_entities))
        ax.bar(x_positions, sorted_norms, color=colors, alpha=0.8, edgecolor='none', linewidth=0)
        
        ax.set_xlabel('Entities (sorted by norm)', fontsize=12)
        ax.set_ylabel('L2 Norm' if idx == 0 else '', fontsize=12)
        ax.set_title(f'{family_name}', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_xticks([])
    
    title_suffix = f' ({norm_method})' if norm_method != 'none' else ''
    plt.suptitle(f'Entity Embedding Norms{title_suffix} - All Families',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'global_entity_norms_bar{suffix}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[INFO] Saved global entity bar plot to {output_path}")
    plt.close()


def run_ablation_norm(args):
    """
    Main function to run norm ablation study.
    Analyzes norms with 4 different normalization methods for each model family.
    
    Args:
        args: Command line arguments containing:
            - data_dir: Path to dataset
            - model_types: List of model families to analyze
            - stability_base_dir: Base directory for stability runs
            - model_pattern: Pattern to filter model directories
            - use_gpu: Whether to use GPU
    """
    device = 'cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Using device: {device}")
    
    # Determine model families to analyze using model_types
    if hasattr(args, 'model_types') and args.model_types:
        model_families = args.model_types
    elif hasattr(args, 'model_families') and args.model_families:
        model_families = args.model_families
    elif hasattr(args, 'model'):
        model_families = [args.model]
    else:
        # Default: analyze all available models
        model_families = ['TransE', 'DistMult', 'ConvE', 'RGCN', 'Transformer']
    
    print(f"[INFO] Analyzing model families: {model_families}")
    
    # Get dataset name
    data_name = args.data_dir.split('/')[-1]
    
    # Create output directory
    output_dir = f"ablation_results/norm_analysis_{data_name}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Results will be saved to {output_dir}")
    
    # Count entity degrees (once for all)
    print("\n" + "="*70)
    print("COUNTING ENTITY DEGREES")
    print("="*70)
    entity_degrees = count_entity_degrees(args.data_dir)
    
    # Normalization methods to apply
    norm_methods = ['none', 'minmax', 'minmax_entity_only', 'zscore_entity_only', 'ratio_entity', 'max_relation']
    
    # Store results for global plots
    all_families_results = {norm_method: {} for norm_method in norm_methods}
    
    # Process each model family separately
    for model_family in model_families:
        print("\n" + "="*70)
        print(f"PROCESSING MODEL FAMILY: {model_family}")
        print("="*70)
        
        # Collect model directories for this family
        if hasattr(args, 'stability_base_dir'):
            base_pattern = os.path.join(args.stability_base_dir, f"{data_name}_{model_family}_*")
        else:
            base_pattern = f"24_09/stability/{data_name}_{model_family}_*"
        
        matching_dirs = glob.glob(base_pattern)
        
        # Filter by model_pattern if specified
        if hasattr(args, 'model_pattern') and args.model_pattern != '*':
            pattern = args.model_pattern
            matching_dirs = [d for d in matching_dirs if pattern.replace('*', '') in d]
        
        # Keep only directories with model.pth
        valid_dirs = [d for d in matching_dirs if os.path.exists(os.path.join(d, 'model.pth'))]
        
        if not valid_dirs:
            print(f"[WARN] No valid model directories found for {model_family}")
            continue
        
        print(f"[INFO] Found {len(valid_dirs)} model directories for {model_family}")
        # Take first N models (to avoid too many subplots)
        model_dirs = valid_dirs[:5]
        print(f"[INFO] Processing {len(model_dirs)} models")
        
        # Process each normalization method
        for norm_method in norm_methods:
            method_name = norm_method if norm_method != 'none' else 'RAW'
            print(f"\n--- {model_family}: {method_name.upper()} ---")
            
            results = analyze_relation_norms(
                model_dirs, args.data_dir, output_dir, model_family, 
                norm_method=norm_method, device=device
            )
            
            if results['relation_norms']:
                # Visualize relation norms
                visualize_relation_norms(
                    results['relation_norms'],
                    results['relation_labels'],
                    results['triple_counts'],
                    results['model_names'],
                    output_dir,
                    model_family,
                    norm_method=norm_method
                )
                
                # Visualize entity norm distributions
                visualize_entity_norms(
                    results['entity_norms'],
                    results['model_names'],
                    output_dir,
                    model_family,
                    norm_method=norm_method
                )
                
                # Visualize entity norm bar plot
                visualize_entity_norms_bar(
                    results['entity_norms'],
                    entity_degrees,
                    results['model_names'],
                    output_dir,
                    model_family,
                    norm_method=norm_method
                )
                
                # Store results for global plots
                all_families_results[norm_method][model_family] = results
            
            # Dimension-wise distributions (only once per family, use 'none' method)
            if norm_method == 'none':
                print(f"\n--- {model_family}: DIMENSION DISTRIBUTIONS ---")
                dim_data = analyze_dimension_distributions(
                    model_dirs, args.data_dir, output_dir, model_family,
                    norm_method=norm_method, device=device
                )
                
                if dim_data:
                    # Visualize entity dimensions
                    visualize_dimension_distributions(
                        dim_data, output_dir, model_family,
                        norm_method=norm_method, embedding_type='entity'
                    )
                    
                    # Visualize relation dimensions
                    visualize_dimension_distributions(
                        dim_data, output_dir, model_family,
                        norm_method=norm_method, embedding_type='relation'
                    )
    
    # Create global plots if multiple families
    if len(model_families) > 1:
        print("\n" + "="*70)
        print("CREATING GLOBAL PLOTS ACROSS FAMILIES")
        print("="*70)
        
        for norm_method in norm_methods:
            if all_families_results[norm_method]:
                print(f"\n--- Global plots for {norm_method.upper()} ---")
                create_global_plots(all_families_results[norm_method], output_dir, 
                                   data_name, norm_method, entity_degrees)
    
    print("\n" + "="*70)
    print("ABLATION STUDY COMPLETE")
    print("="*70)
    print(f"Results saved to: {output_dir}")
    print(f"\nFor each model family, you will find (for each normalization method):")
    print("  - relation_norms_sorted[_METHOD].json")
    print("  - relation_norms_sorted[_METHOD].png")
    print("  - entity_norm_distributions[_METHOD].png")
    print("  - entity_norms_bar[_METHOD].png")
    print(f"\nNormalization methods: {', '.join(norm_methods)}")
    print(f"\nNote: 'max_relation' method normalizes each model's relation norms by its own max (range [0,1])")
