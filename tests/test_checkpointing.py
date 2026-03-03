import torch
import numpy as np
import os
import sys
from pathlib import Path
import pytest

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kge.data import load_triples, load_entity_relation_dicts, convert_to_id_arrays
from training_utils import init_model, train_and_eval


@pytest.fixture(scope="module")
def setup_data(tmp_path_factory):
    data_dir = "data/kinship"
    entity2id, relation2id = load_entity_relation_dicts(data_dir)
    train_triples_str = load_triples(os.path.join(data_dir, "train.txt"))
    valid_triples_str = load_triples(os.path.join(data_dir, "valid.txt"))
    test_triples_str = load_triples(os.path.join(data_dir, "test.txt"))

    train_triples = convert_to_id_arrays(train_triples_str, entity2id, relation2id)
    valid_triples = convert_to_id_arrays(valid_triples_str, entity2id, relation2id)
    test_triples = convert_to_id_arrays(test_triples_str, entity2id, relation2id)

    return {
        "entity2id": entity2id,
        "relation2id": relation2id,
        "train_triples": train_triples,
        "valid_triples": valid_triples,
        "test_triples": test_triples,
        "data_dir": data_dir
    }

class Args:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def get_default_args():
    return {
        'model': 'TransE',
        'embedding_dim': 128,
        'transE_norm': 2,
        'loss_type': 'margin',
        'margin': 1.0,
        'batch_size': 256,
        'lr': 0.001,
        'reg_type': 'none',
        'reg_entity_weight': 0.0,
        'reg_relation_weight': 0.0,
        'dropout_entity': 0.2,
        'dropout_relation': 0.2,
        'seed_init': 42,
        'seed_neg': 42,
        'seed_order': 42,
        'seed_forward': 42,
        'max_epochs': 4,
        'eval_every': 5,
        'early_stop_metric': 'MRR',
        'early_stop_patience': 10,
        'early_stop_delta': 1e-4,
        'num_neg_h': 10,
        'num_neg_t': 10,
        'use_inverse': False,
        'log_to_wandb': False,
        'init_function': 'xavier_uniform',
        'max_hours': float('inf')
    }

def test_checkpoint_reproducibility(setup_data, tmp_path):
    # Common args
    args_dict = get_default_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_and_eval_keys = [
        'seed_neg', 'seed_order', 'seed_forward', 'seed_init', 'batch_size', 'lr', 'max_epochs',
        'eval_every', 'early_stop_metric', 'early_stop_patience', 'early_stop_delta',
        'embedding_dim', 'model_name', 'num_neg_h', 'num_neg_t', 'loss_type', 'margin', 'reg_type',
        'reg_entity_weight', 'reg_relation_weight', 'use_inverse', 'log_to_wandb', 'max_hours'
    ]

    # --- 1. Run for 4 epochs straight ---
    run_dir_full = tmp_path / "full_run"
    run_dir_full.mkdir()
    args_full_dict = args_dict.copy()
    args_full_dict['max_epochs'] = 4
    args_full = Args(**args_full_dict)

    model_full = init_model(args_full, setup_data['entity2id'], setup_data['relation2id'], args_full.init_function, args_full.seed_forward, args_full.seed_init).to(device)
    
    train_and_eval_args_full = {k: args_full_dict[k] for k in train_and_eval_keys if k in args_full_dict}
    train_and_eval_args_full['model_name'] = args_full_dict['model']

    train_and_eval(
        model=model_full,
        train_triples=setup_data['train_triples'],
        valid_triples=setup_data['valid_triples'],
        test_triples=setup_data['test_triples'],
        entity2id=setup_data['entity2id'],
        relation2id=setup_data['relation2id'],
        run_dir=str(run_dir_full),
        data_dir=setup_data['data_dir'],
        device=device,
        resume_checkpoint=False,
        **train_and_eval_args_full
    )

    # --- 2. Run for 2 epochs, then resume for 2 epochs ---
    run_dir_part1 = tmp_path / "run_part1"
    run_dir_part1.mkdir()
    args_part1_dict = args_dict.copy()
    args_part1_dict['max_epochs'] = 2
    args_part1 = Args(**args_part1_dict)
    model_part1 = init_model(args_part1, setup_data['entity2id'], setup_data['relation2id'], args_part1.init_function, args_part1.seed_forward, args_part1.seed_init).to(device)

    train_and_eval_args_part1 = {k: args_part1_dict[k] for k in train_and_eval_keys if k in args_part1_dict}
    train_and_eval_args_part1['model_name'] = args_part1_dict['model']

    train_and_eval(
        model=model_part1,
        train_triples=setup_data['train_triples'],
        valid_triples=setup_data['valid_triples'],
        test_triples=setup_data['test_triples'],
        entity2id=setup_data['entity2id'],
        relation2id=setup_data['relation2id'],
        run_dir=str(run_dir_part1),
        data_dir=setup_data['data_dir'],
        device=device,
        resume_checkpoint=False,
        **train_and_eval_args_part1
    )

    # Resume for another 2 epochs (total 4)
    run_dir_part2 = tmp_path / "run_part2"
    run_dir_part2.mkdir()
    args_part2_dict = args_dict.copy()
    args_part2_dict['max_epochs'] = 4
    args_part2 = Args(**args_part2_dict)
    model_reloaded = init_model(args_part2, setup_data['entity2id'], setup_data['relation2id'], args_part2.init_function, args_part2.seed_forward, args_part2.seed_init).to(device)

    # The resumed run will load the checkpoint from part1's directory
    import shutil
    shutil.copy(run_dir_part1 / "checkpoint.pth", run_dir_part2 / "checkpoint.pth")

    train_and_eval_args_part2 = {k: args_part2_dict[k] for k in train_and_eval_keys if k in args_part2_dict}
    train_and_eval_args_part2['model_name'] = args_part2_dict['model']

    train_and_eval(
        model=model_reloaded,
        train_triples=setup_data['train_triples'],
        valid_triples=setup_data['valid_triples'],
        test_triples=setup_data['test_triples'],
        entity2id=setup_data['entity2id'],
        relation2id=setup_data['relation2id'],
        run_dir=str(run_dir_part2),
        data_dir=setup_data['data_dir'],
        device=device,
        resume_checkpoint=True, # IMPORTANT
        **train_and_eval_args_part2
    )

    # --- 3. Compare results ---
    model_full.load_state_dict(torch.load(run_dir_full / "model.pth"))
    model_reloaded.load_state_dict(torch.load(run_dir_part2 / "model.pth"))

    
    # Compare model parameters
    for p1, p2 in zip(model_full.parameters(), model_reloaded.parameters()):
        assert torch.allclose(p1, p2), "Model parameters do not match after resuming from checkpoint."

    print("Checkpointing test passed successfully!")
