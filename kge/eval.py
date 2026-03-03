import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm

def _prepare_triples(triples, device):
    if isinstance(triples, tuple):
        h = torch.from_numpy(triples[0]).to(device)
        r = torch.from_numpy(triples[1]).to(device)
        t = torch.from_numpy(triples[2]).to(device)
        return torch.stack((h, r, t), dim=1)
    return triples.to(device)


def _compute_ranks(scores, target_score):
    # _compute_ranks(scores: torch.Tensor, target_score: torch.Tensor) -> Tuple[float, float]:
    equal = (scores == target_score).sum().item()
    higher = (scores > target_score).sum().item()
    
    rank_pess = higher + equal
    rank_real = higher + (equal + 1) / 2.0
    
    return rank_pess, rank_real

def evaluate(model, triples, all_triples, filter=True, hits_k=[1,3,10], use_inverse=True, return_preds=False, k_value=50):
    model.eval()
    device = next(model.parameters()).device
    
    triples = _prepare_triples(triples, device)
    all_triples = _prepare_triples(all_triples, device)
    
    num_entities = model.num_entities
    num_relations = model.num_relations
    all_triple_set = set((h.item(), r.item(), t.item()) for h, r, t in all_triples)
    
    reciproc_rank_pess, ranks_pess, hits_pess = [], [], {k: [] for k in hits_k}
    reciproc_rank_real, ranks_real, hits_real = [], [], {k: [] for k in hits_k}
    ground_truth_scores = []
    top_k_entities = []
    top_k_scores = []
    
    with torch.no_grad():
        for h, r, t in tqdm(triples, desc="Evaluating"):

            # Tail prediction
            heads = h.unsqueeze(0)
            rels = r.unsqueeze(0)
            tails = torch.arange(num_entities, device=device).unsqueeze(0)
            
            scores = model(heads, rels, tails, score_mode="multi_tails").squeeze()
            
            if filter:
                h_val, r_val, t_val = h.item(), r.item(), t.item()
                mask = torch.tensor(
                    [(h_val, r_val, idx) in all_triple_set and idx != t_val 
                        for idx in range(num_entities)], 
                    device=device
                )
                scores = scores.masked_fill(mask, -float('inf'))
            target_score = scores[t_val]
            if return_preds:
                # print("h_val score:", scores[h_val].item())
                # print("t_val score:", scores[t_val].item())
                # print("target_score:", target_score.item())
                # print("argmax:", torch.argmax(scores).item())
                # print("argmin:", torch.argmin(scores).item())
                # print("max:", torch.max(scores).item())
                # print("min:", torch.min(scores).item())
                # print("scores[t_val]:", scores[t_val].item())

                top_k = torch.topk(scores, min(k_value, len(scores)), sorted=True)
                top_k_entities.append(top_k.indices.cpu().tolist())
                top_k_scores.append(top_k.values.cpu().tolist())
                ground_truth_scores.append(target_score.item())

            rank_pess, rank_real = _compute_ranks(scores, target_score)
            reciproc_rank_pess.append(1.0 / rank_pess)
            ranks_pess.append(rank_pess)
            reciproc_rank_real.append(1.0 / rank_real)
            ranks_real.append(rank_real)
            for k in hits_k:
                hits_pess[k].append(rank_pess <= k)
                hits_real[k].append(rank_real <= k)
            
            # Head prediction
            tails = t.unsqueeze(0)
            rels = r.unsqueeze(0)
            heads = torch.arange(num_entities, device=device).unsqueeze(0)
            scores = model(heads, rels, tails, score_mode="multi_heads").squeeze()

            if filter:
                t_val, r_val, h_val = t.item(), r.item(), h.item()
                mask = torch.tensor(
                    [(idx, r_val, t_val) in all_triple_set and idx != h_val 
                        for idx in range(num_entities)], 
                    device=device
                )
                scores = scores.masked_fill(mask, -float('inf'))
            target_score = scores[h_val]    
            if return_preds:
                top_k = torch.topk(scores, min(k_value, len(scores)), sorted=True)
                top_k_entities.append(top_k.indices.cpu().tolist())
                top_k_scores.append(top_k.values.cpu().tolist())
                ground_truth_scores.append(target_score.item())
            rank_pess, rank_real = _compute_ranks(scores, target_score)
            reciproc_rank_pess.append(1.0 / rank_pess)
            ranks_pess.append(rank_pess)
            reciproc_rank_real.append(1.0 / rank_real)
            ranks_real.append(rank_real)
            for k in hits_k:
                hits_pess[k].append(rank_pess <= k)
                hits_real[k].append(rank_real <= k)
    metrics = {
        "pessimistic": {
            "MRR": float(np.mean(reciproc_rank_pess)),
            "MR": float(np.mean(ranks_pess)),
            **{f"Hits@{k}": float(np.mean(hits_pess[k])) for k in hits_k}
        },
        "realistic": {
            "MRR": float(np.mean(reciproc_rank_real)),
            "MR": float(np.mean(ranks_real)),
            **{f"Hits@{k}": float(np.mean(hits_real[k])) for k in hits_k}
        }
    }
    
    torch.cuda.empty_cache()
    
    if return_preds:
        preds = {
            "truth_ranks": ranks_real,
            "truth_scores": ground_truth_scores,
            "top_k_entities": top_k_entities,
            "top_k_scores": top_k_scores,
            "k_value": k_value
        }
        return metrics, preds
    return metrics
