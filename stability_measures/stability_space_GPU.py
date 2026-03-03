# stability_space.py
"""
Calculs des métriques dans l'espace d'embedding :
- Jaccard@K
- RBO@K
- CKA
"""
import numpy as np
from typing import List, Dict
import torch
import torch.nn.functional as F

# CF : https://openreview.net/pdf?id=KJNcAkY8tY4 eq. 1 and 2
def linear_hsic_unbiased(K, L):
    # K, L: tensor shape (n, n) Gram matrices, with diagonal zeros
    n = K.size(0)
    assert n > 3
    K = K.clone()
    L = L.clone()
    K.fill_diagonal_(0)
    L.fill_diagonal_(0)
    KL = K @ L
    tr_KL = torch.diagonal(KL, dim1=-2, dim2=-1).sum()
    sum_K = K.sum()
    sum_L = L.sum()
    sum_KL = KL.sum()
    term = tr_KL + (sum_K * sum_L) / ((n - 1) * (n - 2)) - (2.0 * sum_KL) / (n - 2)
    hsic = term / (n * (n - 3))
    return hsic

def cka_mini_batch(emb1, emb2, batch_size=256, device=None, eps=1e-8):
    """
    Compute the CKA linear in mini-batch mode between emb1 and emb2.
    emb1: Tensor size (N, d1)
    emb2: Tensor size (N, d2)
    batch_size: mini-batch size
    Passes over N examples in batches (without loader).
    Returns a scalar CKA.
    """
    if device is None:
        device = emb1.device
    N = emb1.size(0)
    assert emb2.size(0) == N
    # accumulations
    hsic_xy = 0.0
    hsic_xx = 0.0
    hsic_yy = 0.0
    count = 0
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        X = emb1[start:end].to(device)
        Y = emb2[start:end].to(device)
        K = X @ X.t()
        L = Y @ Y.t()
        # set diagonal zero in hsic function
        hs_xy = linear_hsic_unbiased(K, L)
        hs_xx = linear_hsic_unbiased(K, K)
        hs_yy = linear_hsic_unbiased(L, L)
        hsic_xy += hs_xy
        hsic_xx += hs_xx
        hsic_yy += hs_yy
        count += 1
    # moyenne
    hsic_xy /= count
    hsic_xx /= count
    hsic_yy /= count
    return hsic_xy / (torch.sqrt(hsic_xx * hsic_yy) + eps)



# ---------- Metric helpers ----------

def _batch_jaccard_and_overlap(neigh_i: np.ndarray, neigh_j: np.ndarray, k: int) -> tuple:
    """Compute both Jaccard and Overlap for a batch (optimized to compute both at once).
    Returns: (jaccards, overlaps)
    """
    batch_size = neigh_i.shape[0]
    jaccards = []
    overlaps = []
    for r in range(batch_size):
        set_i = set(neigh_i[r, :k])
        set_j = set(neigh_j[r, :k])
        inter = len(set_i & set_j)
        union = len(set_i | set_j)
        jaccards.append(inter / union if union else 1.0)
        overlaps.append(inter / k)
    return jaccards, overlaps


def _batch_rbo(neigh_i: np.ndarray, neigh_j: np.ndarray, k: int) -> List[float]:
    """RBO identical to CPU version but computed per entity and returned as list."""
    rbo_scores = []
    for r in range(neigh_i.shape[0]):
        score = 0.0
        for d in range(1, k + 1):
            inter_d = len(set(neigh_i[r, :d]) & set(neigh_j[r, :d]))
            score += inter_d / d
        rbo_scores.append(score / k)
    return rbo_scores


# def _batch_trust(ranks_i: np.ndarray, ranks_j: np.ndarray, k: int) -> List[float]:
#     penalties = []
#     for entity in range(ranks_i.shape[0]):
#         top_k_j = ranks_j[entity, :k]  # The k nearest neighbors of the entity in the j space
#         # Get the indices (ranks) of these neighbors in the order of ranks_i
#         ranks_in_i = np.where(np.isin(ranks_i[entity], top_k_j))[0]
#         # Filter those with rank > k, subtracting k from them
#         penalty = np.sum(ranks_in_i[ranks_in_i >= k] - k)
#         penalties.append(int(penalty))
#     return penalties


def _batch_trust(ranks_i: torch.Tensor, ranks_j: torch.Tensor, k: int) -> List[int]:

    penalties = []
    for entity in range(ranks_i.shape[0]):
        top_k_j = ranks_j[entity, :k]
        # For each index in ranks_i[entity], find its position if present in top_k_j
        is_in_top_k = torch.isin(ranks_i[entity], top_k_j)
        indices_in_i = torch.nonzero(is_in_top_k).squeeze()
        if indices_in_i.ndim == 0:
            indices_in_i = indices_in_i.unsqueeze(0)
        bad_ranks = indices_in_i[indices_in_i >= k]
        penalty = (bad_ranks - k).sum().item()
        penalties.append(int(penalty))
    return penalties



def _batch_kl(dist_i: torch.Tensor, dist_j: torch.Tensor) -> np.ndarray:
    """KL divergence per entity between two distance matrices (batch,n)."""
    eps = 1e-10
    soft_i = torch.softmax(-dist_i, dim=1)
    soft_j = torch.softmax(-dist_j, dim=1)
    kl = torch.sum(soft_i * (torch.log(soft_i + eps) - torch.log(soft_j + eps)), dim=1)
    return kl.detach().cpu().numpy()
# from .cka import CKACalculator

# Configuration du périphérique (GPU si disponible, sinon CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def to_tensor(x: np.ndarray) -> torch.Tensor:
    """Convertit un tableau numpy en tenseur PyTorch sur le bon périphérique."""
    return torch.as_tensor(x, device=device)

def to_numpy(x: torch.Tensor) -> np.ndarray:
    """Convertit un tenseur PyTorch en tableau numpy."""
    return x.cpu().numpy() if x.is_cuda else x.numpy()

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


def compute_all_neighbor_space_metrics(emb_list: List[np.ndarray], k_values: List[int] = [1, 5, 10], batch_size: int = 1024) -> Dict[str, float]:
    """
    Compute all neighbor-based space metrics for a list of embeddings.
    Version optimisée avec support GPU et calculs parallèles.
    
    Args:
        emb_list: List of embedding matrices (n_samples, n_features)
        k_values: List of k values to compute metrics for
        batch_size: Size of batches for processing
        
    Returns:
        Dictionary containing all computed metrics
    """
    n_models = len(emb_list)
    if n_models < 2:
        return {}
    
    n_samples = emb_list[0].shape[0]
    k_max = max(k_values)
    
    # Initialize result containers
    results = {}
    
    # Initialize accumulators for each metric and model pair
    individual_jaccard_l2 = {(i, j, k): [] for i in range(n_models) for j in range(i + 1, n_models) for k in k_values}
    individual_jaccard_cosine = {(i, j, k): [] for i in range(n_models) for j in range(i + 1, n_models) for k in k_values}
    individual_overlap = {(i, j, k): [] for i in range(n_models) for j in range(i + 1, n_models) for k in k_values}
    individual_rbo = {(i, j, k): [] for i in range(n_models) for j in range(i + 1, n_models) for k in k_values}
    individual_penalties = {(i, j, k): [] for i in range(n_models) for j in range(i + 1, n_models) for k in k_values}
    individual_kl = {(i, j): [] for i in range(n_models) for j in range(i + 1, n_models)}
    individual_second_order_global = {(i, j): [] for i in range(n_models) for j in range(i + 1, n_models)}
    individual_second_order_local = {(i, j, k): [] for i in range(n_models) for j in range(i + 1, n_models) for k in k_values}
    
    # Prepare tensors on device
    emb_tensors = [to_tensor(e) for e in emb_list]
    # Pre-normalized embeddings for cosine computations
    emb_norm = [F.normalize(e, dim=1) for e in emb_tensors]
    
    # Process in batches
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_indices = torch.arange(start, end, device=device)
        batch_size_actual = end - start
        
        precomputed = {
            'distances_l2': {},
            'distances_cosine': {},
            'sorted_indices_l2': {},
            'sorted_indices_cosine': {},
            'neighbors_l2': {},
            'neighbors_cosine': {}
        }
        
        # 1. Precompute distances and neighbors for each model (both L2 and cosine)
        for m in range(n_models):
            # Get batch embeddings
            E_m_batch = emb_tensors[m][start:end]  # (batch_size, dim)
            E_m_batch_norm = emb_norm[m][start:end]  # (batch_size, dim) normalized
            
            # Compute L2 distances from batch entities to all entities
            dist_l2_m = torch.cdist(E_m_batch, emb_tensors[m])  # (batch_size, n_samples)
            
            # Compute cosine distances (1 - cosine similarity)
            cosine_sim_m = E_m_batch_norm @ emb_norm[m].T  # (batch_size, n_samples)
            dist_cosine_m = 1.0 - cosine_sim_m  # (batch_size, n_samples)
            
            # Sort distances and get indices for L2
            sorted_dist_l2_m, sorted_indices_l2_m = torch.sort(dist_l2_m, dim=1)
            
            # Sort distances and get indices for cosine
            sorted_dist_cosine_m, sorted_indices_cosine_m = torch.sort(dist_cosine_m, dim=1)

            # Mask self-distances for L2
            row_ids = torch.arange(batch_size_actual, device=device)
            dist_l2_m[row_ids, batch_indices] = float('inf')
            
            # Mask self-distances for cosine
            dist_cosine_m[row_ids, batch_indices] = float('inf')
            
            precomputed['distances_l2'][m] = dist_l2_m
            precomputed['distances_cosine'][m] = dist_cosine_m
            precomputed['sorted_indices_l2'][m] = sorted_indices_l2_m
            precomputed['sorted_indices_cosine'][m] = sorted_indices_cosine_m
            precomputed['neighbors_l2'][m] = sorted_indices_l2_m[:, 1:k_max+1]  # (batch_size, k_max)
            precomputed['neighbors_cosine'][m] = sorted_indices_cosine_m[:, 1:k_max+1]  # (batch_size, k_max)
        
        # 2. Process each model pair with precomputed values
        for i in range(n_models):
            for j in range(i + 1, n_models):
                print(f"Processing batch {start}-{end} for model pair ({i+1},{j+1})")
                
                # L2-based data
                dist_l2_i = precomputed['distances_l2'][i]
                dist_l2_j = precomputed['distances_l2'][j]
                sorted_indices_l2_i = precomputed['sorted_indices_l2'][i]
                sorted_indices_l2_j = precomputed['sorted_indices_l2'][j]
                neighbors_l2_i = precomputed['neighbors_l2'][i]
                neighbors_l2_j = precomputed['neighbors_l2'][j]
                
                # Cosine-based data
                neighbors_cosine_i = precomputed['neighbors_cosine'][i]
                neighbors_cosine_j = precomputed['neighbors_cosine'][j]
                
                # Convert to numpy for metric computation
                neighbors_l2_i_np = to_numpy(neighbors_l2_i)
                neighbors_l2_j_np = to_numpy(neighbors_l2_j)
                neighbors_cosine_i_np = to_numpy(neighbors_cosine_i)
                neighbors_cosine_j_np = to_numpy(neighbors_cosine_j)
                
                # KL divergence (using L2 distances)
                kl_batch = _batch_kl(dist_l2_i, dist_l2_j)
                individual_kl[(i, j)].extend(kl_batch.tolist())

                # # --- Global second-order distances ---
                # # sim vectors against all entities in their own space
                # Ei_chunk = emb_norm[i][start:end]  # (b, d)
                # Ej_chunk = emb_norm[j][start:end]  # (b, d)
                # Si = Ei_chunk @ emb_norm[i].T      # (b, N)
                # Sj = Ej_chunk @ emb_norm[j].T      # (b, N)
                # Si_n = F.normalize(Si, dim=1)
                # Sj_n = F.normalize(Sj, dim=1)
                # row_cos = (Si_n * Sj_n).sum(dim=1)  # (b,)
                # row_dist = 1.0 - row_cos
                # individual_second_order_global[(i, j)].extend(to_numpy(row_dist))
                
                for k in k_values:
                    # L2-based neighbors
                    neigh_l2_i_k = neighbors_l2_i_np[:, :k]
                    neigh_l2_j_k = neighbors_l2_j_np[:, :k]
                    
                    # Cosine-based neighbors
                    neigh_cosine_i_k = neighbors_cosine_i_np[:, :k]
                    neigh_cosine_j_k = neighbors_cosine_j_np[:, :k]
                    
                    # Jaccard and Overlap with L2 distance (computed together for efficiency)
                    jaccards_l2, overlaps_l2 = _batch_jaccard_and_overlap(neigh_l2_i_k, neigh_l2_j_k, k)
                    individual_jaccard_l2[(i, j, k)].extend(jaccards_l2)
                    individual_overlap[(i, j, k)].extend(overlaps_l2)
                    
                    # Jaccard with cosine distance
                    jaccards_cosine, _ = _batch_jaccard_and_overlap(neigh_cosine_i_k, neigh_cosine_j_k, k)
                    individual_jaccard_cosine[(i, j, k)].extend(jaccards_cosine)
                    
                    # RBO (using L2)
                    individual_rbo[(i, j, k)].extend(_batch_rbo(neigh_l2_i_k, neigh_l2_j_k, k))
                    
                    # Trust (using L2)
                    penalties = _batch_trust(sorted_indices_l2_i, sorted_indices_l2_j, k)
                    individual_penalties[(i, j, k)].extend(penalties)

                    # # --- Local second-order distances (union of top-k) ---
                    # bsz = neighbors_i.shape[0]
                    # for r in range(bsz):
                    #     idx_i = neighbors_i[r, :k]
                    #     idx_j = neighbors_j[r, :k]
                    #     union_idx = torch.unique(torch.cat([idx_i, idx_j], dim=0))
                    #     # similarity vectors in respective spaces
                    #     sI = emb_norm[i][start + r:start + r + 1] @ emb_norm[i][union_idx].T  # (1, U)
                    #     sJ = emb_norm[j][start + r:start + r + 1] @ emb_norm[j][union_idx].T  # (1, U)
                    #     sI_n = F.normalize(sI, dim=1)
                    #     sJ_n = F.normalize(sJ, dim=1)
                    #     cos_val = (sI_n * sJ_n).sum(dim=1).item()
                    #     individual_second_order_local[(i, j, k)].append(1.0 - float(cos_val))
        
        # Libérer la mémoire GPU
        del precomputed
        print("GPU memory used: ", torch.cuda.memory_allocated() / 1e9)
        # Free GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Aggregate results
    # For each k value
    for k in k_values:
        jaccard_l2_all = []
        jaccard_cosine_all = []
        overlap_all = []
        rbo_all = []
        trust_all = []
        # second_local_all = []
        
        for i in range(n_models):
            for j in range(i + 1, n_models):
                # Get mean for this model pair
                jaccard_l2_mean_ij = np.mean(individual_jaccard_l2[(i, j, k)])
                jaccard_cosine_mean_ij = np.mean(individual_jaccard_cosine[(i, j, k)])
                overlap_mean_ij = np.mean(individual_overlap[(i, j, k)])
                rbo_mean_ij = np.mean(individual_rbo[(i, j, k)])
                
                # For trust, sum all penalties and normalize
                trust_sum_ij = np.sum(individual_penalties[(i, j, k)])
                norm_trust = 2 / (n_samples * k * (2 * n_samples - 3 * k - 1))
                trust_ij = 1 - trust_sum_ij * norm_trust
                
                jaccard_l2_all.append(jaccard_l2_mean_ij)
                jaccard_cosine_all.append(jaccard_cosine_mean_ij)
                overlap_all.append(overlap_mean_ij)
                rbo_all.append(rbo_mean_ij)
                trust_all.append(trust_ij)
                # Local second-order per pair mean
                # second_local_all.append(np.mean(individual_second_order_local[(i, j, k)]))
        
        # Store aggregated results
        results[f'space_jaccard_l2@{k}_mean'] = float(np.mean(jaccard_l2_all))
        results[f'space_jaccard_l2@{k}_std'] = float(np.std(jaccard_l2_all))
        results[f'space_jaccard_cosine@{k}_mean'] = float(np.mean(jaccard_cosine_all))
        results[f'space_jaccard_cosine@{k}_std'] = float(np.std(jaccard_cosine_all))
        results[f'space_overlap@{k}_mean'] = float(np.mean(overlap_all))
        results[f'space_overlap@{k}_std'] = float(np.std(overlap_all))
        results[f'rbo@{k}_mean'] = float(np.mean(rbo_all))
        results[f'rbo@{k}_std'] = float(np.std(rbo_all))
        results[f'trust@{k}_mean'] = float(np.mean(trust_all))
        results[f'trust@{k}_std'] = float(np.std(trust_all))
        # results[f'second_order_local@{k}_mean'] = float(np.mean(second_local_all))
        # results[f'second_order_local@{k}_std'] = float(np.std(second_local_all))
    
    # Aggregate KL divergence results
    kl_all = []
    for i in range(n_models):
        for j in range(i + 1, n_models):
            kl_mean_ij = np.mean(individual_kl[(i, j)])
            kl_all.append(kl_mean_ij)

    results['space_kl_mean'] = float(np.mean(kl_all))
    results['space_kl_std'] = float(np.std(kl_all))
    # # Aggregate Global second-order distances
    # second_global_all = []
    # for i in range(n_models):
    #     for j in range(i + 1, n_models):
    #         second_global_all.append(np.mean(individual_second_order_global[(i, j)]))
    # if second_global_all:
    #     results['second_order_global_mean'] = float(np.mean(second_global_all))
    #     results['second_order_global_std'] = float(np.std(second_global_all))
    # Compute CKA (single computation, not per k)
    cka_list = []
    for i in range(n_models):
        for j in range(i + 1, n_models):
            print(f"Computing CKA for model pair ({i+1},{j+1})")
            cka_val = cka_mini_batch(emb_tensors[i], emb_tensors[j])
            cka_list.append(cka_val.item())
    if cka_list:
        results['cka_mean'] = float(np.mean(cka_list))
        results['cka_std'] = float(np.std(cka_list))
    
    # Clean up GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return results
