# stability_prediction.py
"""
Calculs des métriques sur les prédictions :
- Ambiguity
- Discrepancy
- Moyenne et écart-type des stds
- Jaccard, RBO, KL pour les prédictions
- Extraction et agrégation du MRR à partir des fichiers metrics.json des runs
"""
import os
import json
import numpy as np
from scipy.special import softmax
from scipy.stats import entropy
from typing import List, Dict
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

def compute_ambiguity(runs: List[Dict], k_values: List[int] = [1, 5, 10]) -> Dict[str, float]:
    """
    Compute ambiguity for multiple K values.
    
    Args:
        runs: List of run dictionaries containing 'ranks' array
        k_values: List of K values to compute ambiguity for
        
    Returns:
        Dictionary with ambiguity scores for each K value
    """

    
    ranks = [np.array(run['preds']['truth_ranks']) for run in runs]
    if len(ranks) < 2:
        return {f'ambiguity@{K}': "NaN" for K in k_values}
    ranks = np.stack(ranks)  # (n_runs, n_queries)
    n_runs, n_queries = ranks.shape
    
    results = {}
    for K in k_values:
        ambiguous = 0
        for i in range(n_queries):
            # Un point est ambigu s'il n'est pas toujours < K ou toujours >= K
            below = np.all(ranks[:, i] <= K)
            above = np.all(ranks[:, i] > K)
            if not (below or above):
                ambiguous += 1
        results[f'ambiguity@{K}'] = float(ambiguous / n_queries)
    
    return results

def compute_discrepancy(runs: List[Dict], k_values: List[int] = [1, 5, 10]) -> Dict[str, float]:
    """
    Compute discrepancy for multiple K values.
    
    Args:
        runs: List of run dictionaries containing 'ranks' array
        k_values: List of K values to compute discrepancy for
        
    Returns:
        Dictionary with discrepancy scores for each K value
    """
    
    ranks = [np.array(run['preds']['truth_ranks']) for run in runs]
    if len(ranks) < 2:
        return {f'discrepancy@{K}': "NaN" for K in k_values}
    ranks = np.stack(ranks)  # (n_runs, n_queries)
    n_runs, n_queries = ranks.shape
    
    results = {}
    for K in k_values:
        max_disc = 0.0
        for i in range(n_runs):
            for j in range(i+1, n_runs):
            # Pour chaque triple, diff = (un run < K, l'autre >= K) ou l'inverse
                # Pour chaque triple, diff = (un run < K, l'autre >= K) ou l'inverse
                diff_mask = (ranks[i] <= K) != (ranks[j] <= K)
                # if np.any(diff_mask):
                #     print(f"[DEBUG] Discrepancy found between run {i} and {j} for queries {np.where(diff_mask)[0]}")
                #     print(f"[DEBUG] Ranks for run {i}: {ranks[i][diff_mask]}")
                #     print(f"[DEBUG] Ranks for run {j}: {ranks[j][diff_mask]}")
                diff = np.sum(diff_mask) / n_queries
                if diff > max_disc:
                    max_disc = diff
        results[f'discrepancy@{K}'] = float(max_disc)
    
    return results

def compute_rank_stats(runs: List[Dict]) -> Dict[str, float]:
    """
    Calcule la moyenne et l'écart-type des stds des ranks et truths sur les runs.
    """
    out = {}

    # On suppose que chaque run a 'truth_ranks'
    if 'truth_ranks' in runs[0] and runs[0]['truth_ranks']:
        ranks_array = np.stack([np.array(run['truth_ranks']) for run in runs])
        per_triple_rank_std = np.std(ranks_array, axis=0, ddof=0)
        out['mean_of_rank_stds'] = float(np.mean(per_triple_rank_std))
        out['std_of_rank_stds'] = float(np.std(per_triple_rank_std, ddof=0))

    return out

def compute_pred_jaccard_and_overlap(preds1, preds2, k=10):
    """
    Compute both Jaccard and Overlap between top-K predictions of two models.
    Optimized to compute both metrics at once.
    
    Args:
        preds1: Dictionary containing 'top_k_entities' from first model
        preds2: Dictionary containing 'top_k_entities' from second model
        k: Number of top predictions to consider (must be <= k_value used during prediction)
    
    Returns:
        Tuple of (mean_jaccard, mean_overlap)
    """
    # Get top-k predictions for each query
    topk1 = np.array([preds1['top_k_entities'][i][:k] for i in range(len(preds1['top_k_entities']))])
    topk2 = np.array([preds2['top_k_entities'][i][:k] for i in range(len(preds2['top_k_entities']))])
    
    # Compute Jaccard and Overlap for each query
    jaccards = []
    overlaps = []
    for i in range(len(topk1)):
        set1 = set(topk1[i])
        set2 = set(topk2[i])
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        jaccards.append(intersection / union if union > 0 else 1.0)
        overlaps.append(intersection / k)
    
    return np.mean(jaccards), np.mean(overlaps)



def compute_pred_rbo(preds1, preds2, k=10):
    """
    Compute Rank-Biased Overlap (RBO) between top-K predictions of two models.
    Returns a value between 0 and 1, where 1 means identical rankings.
    
    Args:
        preds1: Dictionary containing 'top_k_entities' from first model
        preds2: Dictionary containing 'top_k_entities' from second model
        k: Number of top predictions to consider (must be <= k_value used during prediction)
    """
    # Get top-k predictions for each query
    topk1 = np.array([preds1['top_k_entities'][i][:k] for i in range(len(preds1['top_k_entities']))])
    topk2 = np.array([preds2['top_k_entities'][i][:k] for i in range(len(preds2['top_k_entities']))])
    
    rbo_scores = []
    
    for i in range(len(topk1)):
        score = 0.0
        for d in range(1, k + 1):
            # Get top d elements
            set1 = set(topk1[i, :d])
            set2 = set(topk2[i, :d])
            
            # Calculate overlap
            inter = len(set1 & set2)
            
            # Add weighted overlap to score
            score += inter / d
        
        # Normalize by k
        rbo_scores.append(score / k)
    
    # Return average RBO across all queries
    return np.mean(rbo_scores) if rbo_scores else 0.0

def compute_pred_kl(preds1, preds2, temperature=1.0):
    """
    Compute KL divergence between prediction distributions of two models.
    For entities not in top_k, we assume their score is -inf (which becomes 0 after softmax).
    
    Args:
        preds1: Dictionary containing 'top_k_entities' and 'top_k_scores' from first model
        preds2: Dictionary containing 'top_k_entities' and 'top_k_scores' from second model
        temperature: Temperature for softmax
    """
    # Get the union of all entities in top-k for both models
    all_entities = []
    for i in range(len(preds1['top_k_entities'])):
        entities1 = set(preds1['top_k_entities'][i])
        entities2 = set(preds2['top_k_entities'][i])
        all_entities.append(entities1.union(entities2))
    
    # For each query, create a vector with scores for all entities
    def get_full_scores(preds, query_idx, all_entities_query):
        # Initialize with -inf (will become 0 after softmax)
        full_scores = np.full(len(all_entities_query), -np.inf)
        entity_to_idx = {e: i for i, e in enumerate(sorted(all_entities_query))}
        
        # Fill in the scores we have
        for e, s in zip(preds['top_k_entities'][query_idx], preds['top_k_scores'][query_idx]):
            if e in entity_to_idx:
                full_scores[entity_to_idx[e]] = s
        
        return full_scores
    
    # Compute KL divergence for each query
    kl_divs = []
    for i in range(len(preds1['top_k_entities'])):
        # Get full score vectors for this query
        s1 = get_full_scores(preds1, i, all_entities[i])
        s2 = get_full_scores(preds2, i, all_entities[i])
        
        # Apply softmax with temperature
        def softmax_with_temp(x):
            e_x = np.exp((x - np.max(x)) / temperature)
            return e_x / e_x.sum()
        
        p = softmax_with_temp(s1)
        q = softmax_with_temp(s2)
        
        # Compute KL divergence
        kl = np.sum(p * (np.log(p + 1e-10) - np.log(q + 1e-10)))
        kl_divs.append(kl)
    
    return np.mean(kl_divs) if kl_divs else 0.0

def compute_super_pred_jaccard(preds_list, k=10):
    """
    Compute super-pred-jaccard@K = |A ∩ B ∩ ...| / |A ∪ B ∪ ...|
    and super-pred-overlap@K = |A ∩ B ∩ ...| / K
    This measures the agreement across ALL models simultaneously.
    Returns mean, std, and per-query values for both metrics.
    
    Args:
        preds_list: List of prediction dictionaries from different runs
        k: Number of top predictions to consider
    
    Returns:
        Tuple of (mean_super_jaccard, std_super_jaccard, super_jaccards_per_query,
                  mean_super_overlap, std_super_overlap, super_overlaps_per_query)
    """
    if len(preds_list) < 2:
        return 1.0, 0.0, [], 1.0, 0.0, []
    
    n_queries = len(preds_list[0]['top_k_entities'])
    super_jaccards = []
    super_overlaps = []
    
    for query_idx in range(n_queries):
        # Get top-k predictions from all models for this query
        all_sets = []
        for preds in preds_list:
            topk = set(preds['top_k_entities'][query_idx][:k])
            all_sets.append(topk)
        
        # Compute intersection of all sets
        intersection = all_sets[0]
        for s in all_sets[1:]:
            intersection = intersection & s
        
        # Compute union of all sets
        union = all_sets[0]
        for s in all_sets[1:]:
            union = union | s
        
        # Compute super jaccard for this query
        if len(union) > 0:
            super_jaccards.append(len(intersection) / len(union))
        else:
            super_jaccards.append(1.0)
        
        # Compute super overlap for this query
        super_overlaps.append(len(intersection) / k)
    
    return (np.mean(super_jaccards), np.std(super_jaccards), super_jaccards,
            np.mean(super_overlaps), np.std(super_overlaps), super_overlaps)


def plot_super_pred_jaccard_distribution(super_jaccards_per_k, k_values, output_path, normalize=False):
    """
    Plot the distribution of super_pred_jaccard values per query for each K.
    
    Args:
        super_jaccards_per_k: Dict mapping k -> list of super_jaccard values per query
        k_values: List of K values
        output_path: Path to save the plot
        normalize: If True, normalize frequencies by number of queries
    """
    fig, axes = plt.subplots(1, len(k_values), figsize=(5 * len(k_values), 4))
    
    # Handle single subplot case
    if len(k_values) == 1:
        axes = [axes]
    
    # Determine common Y-axis limits
    if normalize:
        # For normalized: always 0 to 1
        y_max = 1.0
    else:
        # For non-normalized: find max frequency across all K values
        max_freq = 0
        for k in k_values:
            values = super_jaccards_per_k[k]
            hist, _ = np.histogram(values, bins=50)
            max_freq = max(max_freq, hist.max())
        y_max = max_freq * 1.1  # Add 10% margin
    
    for idx, k in enumerate(k_values):
        ax = axes[idx]
        values = super_jaccards_per_k[k]
        n_queries = len(values)
        
        # Histogram
        if normalize:
            weights = np.ones_like(values) / n_queries
            ax.hist(values, bins=50, alpha=0.7, color='steelblue', edgecolor='black', weights=weights)
        else:
            ax.hist(values, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        
        # Add mean and std lines
        mean_val = np.mean(values)
        std_val = np.std(values)
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
        ax.axvline(mean_val - std_val, color='orange', linestyle=':', linewidth=1.5, label=f'Std: {std_val:.3f}')
        ax.axvline(mean_val + std_val, color='orange', linestyle=':', linewidth=1.5)
        
        ax.set_xlabel(f'K={k}', fontsize=12)
        
        # Set consistent Y-axis limits for all subplots
        ax.set_ylim(0, y_max)
        
        # Only show y-label and ticks for the first (leftmost) plot
        if idx == 0:
            if normalize:
                ax.set_ylabel('Normalized Frequency', fontsize=12)
            else:
                ax.set_ylabel('Frequency', fontsize=12)
        else:
            ax.set_yticklabels([])
        
        # No title
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Super-pred distribution plot saved to {output_path}")


def compute_all_prediction_metrics(preds_list: List[Dict], k_values: List[int] = [1, 5, 10],
                                    output_dir: str = None, model: str = None, 
                                    dataset: str = None, seed_type: str = None, 
                                    epoch: int = None, quality: str = None) -> Dict[str, float]:
    """
    Compute all prediction-based metrics for a list of predictions.
    
    Args:
        preds_list: List of prediction dictionaries from different runs
        k_values: List of K values to compute metrics for
        output_dir: Optional directory to save plots
    """
    results = {}
    n_models = len(preds_list)
    
    if n_models < 2:
        return {}
    
    # Store super_jaccard values per query for each k
    super_jaccards_per_k = {}
    super_overlaps_per_k = {}
    
    # For each K value, compute metrics
    for k in k_values:
        jaccards = []
        overlaps = []
        rbos = []
        kls = []
        
        # Compute pairwise metrics for each pair of runs
        for i in range(n_models):
            for j in range(i + 1, n_models):
                # Compute jaccard and overlap together (optimized)
                jaccard, overlap = compute_pred_jaccard_and_overlap(preds_list[i], preds_list[j], k)
                rbo = compute_pred_rbo(preds_list[i], preds_list[j], k)
                kl = compute_pred_kl(preds_list[i], preds_list[j])
                
                jaccards.append(jaccard)
                overlaps.append(overlap)
                rbos.append(rbo)
                kls.append(kl)
        
        # Compute super-pred-jaccard and super-pred-overlap (across all models)
        (super_jaccard_mean, super_jaccard_std, super_jaccards,
         super_overlap_mean, super_overlap_std, super_overlaps) = compute_super_pred_jaccard(preds_list, k)
        super_jaccards_per_k[k] = super_jaccards
        super_overlaps_per_k[k] = super_overlaps
        
        # Store results for this K value
        results[f'pred_jaccard@{k}'] = float(np.mean(jaccards))
        results[f'pred_jaccard@{k}_std'] = float(np.std(jaccards))
        results[f'pred_overlap@{k}'] = float(np.mean(overlaps))
        results[f'pred_overlap@{k}_std'] = float(np.std(overlaps))
        results[f'pred_rbo@{k}'] = float(np.mean(rbos))
        results[f'pred_rbo@{k}_std'] = float(np.std(rbos))
        results[f'pred_kl@{k}'] = float(np.mean(kls))
        results[f'pred_kl@{k}_std'] = float(np.std(kls))
        results[f'super_pred_jaccard@{k}_mean'] = float(super_jaccard_mean)
        results[f'super_pred_jaccard@{k}_std'] = float(super_jaccard_std)
        results[f'super_pred_overlap@{k}_mean'] = float(super_overlap_mean)
        results[f'super_pred_overlap@{k}_std'] = float(super_overlap_std)
    
    # Generate plot if output directory is provided
    if output_dir and super_jaccards_per_k:
        os.makedirs(output_dir, exist_ok=True)
        # Build filename with available parameters
        filename_parts = []
        if model:
            filename_parts.append(model)
        if dataset:
            filename_parts.append(dataset)
        if quality:
            filename_parts.append(quality)
        if seed_type:
            filename_parts.append(seed_type)
        if epoch:
            filename_parts.append(str(epoch))

        # Generate distribution plots ONLY for seed_type='all'
        if seed_type == 'all':
            # Generate Jaccard plots (normal and normalized)
            filename_jaccard = filename_parts.copy()
            filename_jaccard.append('super_pred_jaccard_distribution.png')
            plot_path = os.path.join(output_dir, '_'.join(filename_jaccard))
            plot_super_pred_jaccard_distribution(super_jaccards_per_k, k_values, plot_path, normalize=False)
            
            filename_jaccard_norm = filename_parts.copy()
            filename_jaccard_norm.append('super_pred_jaccard_distribution_normalized.png')
            plot_path_norm = os.path.join(output_dir, '_'.join(filename_jaccard_norm))
            plot_super_pred_jaccard_distribution(super_jaccards_per_k, k_values, plot_path_norm, normalize=True)
            
            # Generate Overlap plots (normal and normalized)
            filename_overlap = filename_parts.copy()
            filename_overlap.append('super_pred_overlap_distribution.png')
            plot_path = os.path.join(output_dir, '_'.join(filename_overlap))
            plot_super_pred_jaccard_distribution(super_overlaps_per_k, k_values, plot_path, normalize=False)
            
            filename_overlap_norm = filename_parts.copy()
            filename_overlap_norm.append('super_pred_overlap_distribution_normalized.png')
            plot_path_norm = os.path.join(output_dir, '_'.join(filename_overlap_norm))
            plot_super_pred_jaccard_distribution(super_overlaps_per_k, k_values, plot_path_norm, normalize=True)
    print("results:", results)
    return results

def compute_mrr_stats_from_runs(runs: List[Dict]) -> Dict[str, float]:
    """
    Récupère le MRR de chaque run dans metrics.json["pessimistic"]["MRR"],
    puis calcule la moyenne et l'écart-type pour le groupe.

    Args:
        runs: liste de dictionnaires contenant au moins la clé 'run_dir'.

    Returns:
        Dict avec 'mrr_mean' et 'mrr_std'. Si aucun MRR n'est trouvé, renvoie "NaN".
    """
    mrr_values = []
    hit_at_1_values = []
    hit_at_10_values = []
    mr_values = []
    for run in runs:
        run_dir = run.get('run_dir')
        if not run_dir:
            continue
        metrics_path = os.path.join(run_dir, 'metrics.json')
        if not os.path.exists(metrics_path):
            # Pas de metrics.json pour ce run
            continue
        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            mrr = metrics.get('pessimistic', {}).get('MRR', None)
            hit_at_1 = metrics.get('pessimistic', {}).get('Hit@1', None)
            hit_at_10 = metrics.get('pessimistic', {}).get('Hit@10', None)
            mr = metrics.get('pessimistic', {}).get('MR', None)
            if mrr is not None:
                # Force cast to float in case it's not
                mrr_values.append(float(mrr))
            if hit_at_1 is not None:
                hit_at_1_values.append(float(hit_at_1))
            if hit_at_10 is not None:
                hit_at_10_values.append(float(hit_at_10))
            if mr is not None:
                mr_values.append(float(mr))
        except Exception as e:
            # Ignore silently but could be logged by caller if needed
            pass

    if len(mrr_values) == 0:
        return {"mrr_mean": "NaN", "mrr_std": "NaN", "mrr_count": 0}

    return {
        "mrr_mean": float(np.mean(mrr_values)),
        "mrr_std": float(np.std(mrr_values, ddof=0)),
        "hit@1_mean": float(np.mean(hit_at_1_values)),
        "hit@1_std": float(np.std(hit_at_1_values, ddof=0)),
        "hit@10_mean": float(np.mean(hit_at_10_values)),
        "hit@10_std": float(np.std(hit_at_10_values, ddof=0)),
        "MR_mean": float(np.mean(mr_values)),
        "MR_std": float(np.std(mr_values, ddof=0)),
    }
