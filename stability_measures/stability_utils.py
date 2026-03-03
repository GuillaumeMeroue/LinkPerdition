import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import torch
from scipy.special import softmax
from scipy.stats import entropy
import sys

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kge.data import load_entity_relation_dicts, load_triples, convert_to_id_arrays
from kge.models import TransE, DistMult, ConvE
from kge.eval import evaluate
from training_utils import init_model
from argparse import Namespace

def load_model_from_checkpoint(
    run_dir: str,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Tuple[torch.nn.Module, Dict]:
    """
    Load a model from a checkpoint directory.
    
    Args:
        run_dir: Directory containing config.json and model.pth files
        device: Device to load the model on ('cuda' or 'cpu')
        
    Returns:
        Tuple of (model, config)
    """
    # Load config
    config_path = os.path.join(run_dir, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
        config = Namespace(**config)
    print(f"[INFO] Config loaded: {config}")
    
    # Load entity and relation mappings
    data_dir = config.data_dir
    entity2id, relation2id = load_entity_relation_dicts(data_dir)
    if config.use_inverse:
        # Add inverse relations
        num_rel = len(relation2id)
        for rel, idx in list(relation2id.items()):
            inv_rel = rel + '_inv'
            relation2id[inv_rel] = idx + num_rel
    # Initialize model
    model = init_model(config, entity2id, relation2id, config.init_function, config.seed_forward, config.seed_init)
    
    # Load model weights
    model_path = os.path.join(run_dir, 'model.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model, config


def load_test_triples(data_dir, config):
    """Load and prepare test triples for evaluation."""
    from kge.data import load_triples, convert_to_id_arrays, load_entity_relation_dicts
    
    # Load test triples
    test_triples_str = load_triples(os.path.join(data_dir, "test.txt"))
    train_triples_str = load_triples(os.path.join(data_dir, "train.txt"))
    valid_triples_str = load_triples(os.path.join(data_dir, "valid.txt"))
    entity2id, relation2id = load_entity_relation_dicts(data_dir)
    test_triples = convert_to_id_arrays(test_triples_str, entity2id, relation2id, 
                                      use_inverse=config.use_inverse, split="test")
    train_triples = convert_to_id_arrays(train_triples_str, entity2id, relation2id, 
                                      use_inverse=config.use_inverse, split="train")
    valid_triples = convert_to_id_arrays(valid_triples_str, entity2id, relation2id, 
                                      use_inverse=config.use_inverse, split="valid")
    
    all_h = np.concatenate([train_triples[0], valid_triples[0], test_triples[0]], axis=0)
    all_r = np.concatenate([train_triples[1], valid_triples[1], test_triples[1]], axis=0)
    all_t = np.concatenate([train_triples[2], valid_triples[2], test_triples[2]], axis=0)  
    all_triples = (all_h, all_r, all_t)
    
    return test_triples, all_triples, entity2id, relation2id


def compute_preds(model, test_triples, all_triples, config, run_dir=None, top_k=10):
    """
    Get prediction scores from a model for test triples.
    If run_dir is provided, will try to load from top_10_preds.pkl first,
    otherwise will compute and save predictions.
    """
    import torch
    import pickle
    import os
    from kge.eval import evaluate
    
    # Check if predictions exist on disk
    if run_dir is not None:
        preds_path = os.path.join(run_dir, f'top_{top_k}_preds.pkl')
        if os.path.exists(preds_path):
            print(f"[INFO] Loading predictions from {preds_path}")
            with open(preds_path, 'rb') as f:
                return pickle.load(f)
    
    # If not found on disk, compute predictions
    print(f"[INFO] Computing predictions for {run_dir or 'model'}")
    device = next(model.parameters()).device
    
    with torch.no_grad():
        _, preds = evaluate(
            model, 
            test_triples, 
            all_triples,
            use_inverse=config.use_inverse,
            return_preds=True,
            top_k=top_k
        )
    
    # Save predictions to disk if run_dir is provided
    if run_dir is not None:
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, f'top_{top_k}_preds.pkl'), 'wb') as f:
            pickle.dump(preds, f)
        print(f"[INFO] Saved predictions to {os.path.join(run_dir, f'top_{top_k}_preds.pkl')}")
    
    return preds


def get_preds_list(runs, device='cuda' if torch.cuda.is_available() else 'cpu', top_k=10):
    """
    Compute prediction scores for a list of model runs.
    
    Args:
        runs: List of run dictionaries containing 'run_dir' and 'data_dir'
        device: Device to run the models on ('cuda' or 'cpu')
        
    Returns:
        List of prediction score arrays
    """
    if not runs or len(runs) < 2:
        raise ValueError("At least two runs are required to compute stability metrics")

    try:
        _, config = load_model_from_checkpoint(runs[0]['run_dir'], device)
    except Exception as e:
        raise ValueError(f"Error loading model from checkpoint: {str(e)}")
    
    # Get data directory from first run
    data_dir = runs[0]['data_dir']
    if not data_dir or not os.path.exists(data_dir):
        raise ValueError(f"Invalid data directory: {data_dir}")
        
    # Load test triples and all triples for filtering
    test_triples, all_triples, entity2id, relation2id = load_test_triples(data_dir, config)

    all_scores = []
    
    # Process each run
    for run in runs:
        run_dir = run['run_dir']
        if not run_dir or not os.path.exists(run_dir):
            print(f"[WARN] Run directory not found: {run_dir}")
            continue
            
        try:
            # Load model and config
            model, config = load_model_from_checkpoint(run_dir, device)
            
            # Get prediction scores, passing run_dir for caching
            scores = compute_preds(model, test_triples, all_triples, config, run_dir, top_k)
            all_scores.append(scores)
            
        except Exception as e:
            print(f"[ERROR] Error processing run {run_dir}: {str(e)}")
            continue
    
    return all_scores
    

