# stability_space.py
"""
Calculs des métriques dans l'espace d'embedding :
- Jaccard@K
- RBO@K
- CKA
"""
import numpy as np
from typing import Tuple, List, Dict, Union
import torch
from scipy.spatial.distance import cdist
from scipy.special import softmax
from scipy.stats import entropy
# from .cka import cka as compute_cka_official


def load_embedding_from_pth(pth_path: str):
    """
    Charge les embeddings à partir d'un fichier model.pth sous la clé 'entity_emb'.
    Retourne un np.ndarray.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(pth_path, map_location=device)
    # print(checkpoint.keys())
    if 'entity_emb.weight' not in checkpoint:
        raise KeyError(f"La clé 'entity_emb.weight' n'est pas présente dans {pth_path}")
    emb = checkpoint['entity_emb.weight']
    if isinstance(emb, torch.nn.Embedding):
        emb = emb.weight.data.cpu().numpy()
    elif hasattr(emb, 'cpu'):
        emb = emb.cpu().numpy()
    return emb

    

def compute_cka(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Calcule le CKA entre deux matrices d'embedding."""
    def center_gram(K):
        n = K.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        return H @ K @ H
    K1 = emb1 @ emb1.T
    K2 = emb2 @ emb2.T
    K1c = center_gram(K1)
    K2c = center_gram(K2)
    num = np.trace(K1c @ K2c)
    denom = np.sqrt(np.trace(K1c @ K1c) * np.trace(K2c @ K2c))
    return float(num / denom) if denom > 0 else 0.0

def compute_space_kl(pc1: dict, pc2: dict) -> float:
    """
    Compute the average KL divergence between distance distributions of two precomputed embeddings.
    
    Args:
        pc1: First precomputed data from _precompute_neighbors
        pc2: Second precomputed data from _precompute_neighbors
        
    Returns:
        Average KL divergence between distance distributions
    """
    p = pc1['softmax_dist']
    q = pc2['softmax_dist']
    
    # Compute KL divergence for each point and average
    # Using vectorized computation for better performance
    kl_divs = np.sum(p * (np.log(p + 1e-10) - np.log(q + 1e-10)), axis=1)
    
    return float(np.mean(kl_divs))


def _precompute_neighbors(emb, k_max, temperature=1.0):
    """
    Precompute distances, neighbors, ranks and KL-related data for an embedding matrix.
    
    Args:
        emb: Embedding matrix (n_samples, n_features)
        k_max: Maximum number of neighbors to consider
        temperature: Temperature for softmax normalization
        
    Returns:
        Dictionary containing precomputed data
    """
    # Compute pairwise distances
    dist = cdist(emb, emb, 'euclidean')
    np.fill_diagonal(dist, np.inf)  # Exclude self-distances
    
    # Compute neighbors and ranks
    neighbors = np.argsort(dist, axis=1)[:, :k_max]
    ranks = np.argsort(np.argsort(dist, axis=1), axis=1) + 1  # 1-based ranks
    
    # Precompute softmax for KL divergence
    softmax_dist = softmax(-dist / temperature, axis=1)
    
    return {
        'dist': dist,
        'neighbors': neighbors,
        'ranks': ranks,
        'softmax_dist': softmax_dist
    }


def _compute_jaccard(neigh1, neigh2, k):
    """Compute Jaccard similarity between two neighbor sets."""
    jaccards = []
    for i in range(neigh1.shape[0]):
        set1 = set(neigh1[i, :k])
        set2 = set(neigh2[i, :k])
        inter = len(set1 & set2)
        union = len(set1 | set2)
        jaccards.append(inter / union if union > 0 else 1.0)
    return np.mean(jaccards)


def _compute_rbo(neigh1, neigh2, k):
    """Compute Rank-Biased Overlap between two neighbor sets."""
    rbo_scores = []
    for i in range(neigh1.shape[0]):
        score = 0.0
        for d in range(1, k + 1):
            set1 = set(neigh1[i, :d])
            set2 = set(neigh2[i, :d])
            inter = len(set1 & set2)
            score += (inter / d)
        rbo_scores.append(score / k)
    return np.mean(rbo_scores)


def _compute_trust(neigh1, ranks1, neigh2, k):
    """Compute Trustworthiness between two neighbor sets."""
    n = neigh1.shape[0]
    trust = 0.0
    
    for i in range(n):
        # Find points in neigh2 that are not in neigh1
        intruders = np.setdiff1d(neigh2[i, :k], neigh1[i, :k])
        for j in intruders:
            trust += (ranks1[i, j] - k)
    
    # Normalization factor
    norm = n * k * (2 * n - 3 * k - 1)
    return 1 - 2 * trust / norm if norm > 0 else 0.0


def compute_all_neighbor_space_metrics(emb_list, k_values=[1, 5, 10]):
    """
    Compute all neighbor-based space metrics for a list of embeddings.
    
    Args:
        emb_list: List of embedding matrices (n_samples, n_features)
        k_values: List of k values to compute metrics for
        
    Returns:
        Dictionary containing all computed metrics
    """
    n_models = len(emb_list)
    if n_models < 2:
        return {}
    
    # Find maximum k value needed
    k_max = max(k_values)
    
    # Precompute neighbors, distances and KL-related data for each embedding
    precomputed = []
    for emb in emb_list:
        precomputed.append(_precompute_neighbors(emb, k_max, temperature=1.0))
    
    results = {}
    
    # Compute metrics for each k value
    for k in k_values:
        jaccards = []
        rbos = []
        trusts = []
        
        # Compute metrics for each pair of models
        for i in range(n_models):
            for j in range(i + 1, n_models):
                pc1, pc2 = precomputed[i], precomputed[j]
                
                # Compute Jaccard@k
                jaccard = _compute_jaccard(pc1['neighbors'], pc2['neighbors'], k)
                jaccards.append(jaccard)
                
                # Compute RBO@k
                rbo = _compute_rbo(pc1['neighbors'], pc2['neighbors'], k)
                rbos.append(rbo)
                
                # Compute Trust@k (in both directions and average)
                trust1 = _compute_trust(pc1['neighbors'], pc1['ranks'], pc2['neighbors'], k)
                trust2 = _compute_trust(pc2['neighbors'], pc2['ranks'], pc1['neighbors'], k)
                trusts.append((trust1 + trust2) / 2)
        
        # Store results for this k value
        if jaccards:
            results[f'jaccard@{k}_mean'] = float(np.mean(jaccards))
            results[f'jaccard@{k}_std'] = float(np.std(jaccards))
            results[f'rbo@{k}_mean'] = float(np.mean(rbos))
            results[f'rbo@{k}_std'] = float(np.std(rbos))
            results[f'trust@{k}_mean'] = float(np.mean(trusts))
            results[f'trust@{k}_std'] = float(np.std(trusts))
    
    # Compute CKA (only once, not per k)
    cka_list = []
    for i in range(n_models):
        for j in range(i + 1, n_models):
            cka = compute_cka(emb_list[i], emb_list[j])
            cka_list.append(cka)
    
    if cka_list:
        results['cka_mean'] = float(np.mean(cka_list))
        results['cka_std'] = float(np.std(cka_list))
    
    # Compute KL divergence (only once, not per k)
    kl_list = []
    for i in range(n_models):
        for j in range(i + 1, n_models):
            kl = compute_space_kl(precomputed[i], precomputed[j])
            kl_list.append(kl)
    
    if kl_list:
        results['space_kl_mean'] = float(np.mean(kl_list))
        results['space_kl_std'] = float(np.std(kl_list))
    
    return results
