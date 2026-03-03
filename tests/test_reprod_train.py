
import numpy as np
import torch
from kge.models import DistMult
from kge.data import load_triples, load_entity_relation_dicts, convert_to_id_arrays, get_dataloader
from kge.utils import set_seed
from kge.train import get_loss_fn, init_training, one_epoch
from kge.eval import evaluate
import copy
import os
from tqdm import tqdm
from training_utils import init_model
import argparse

def val_evaluator(model, triples, all_triples, entity2id, relation2id, use_inverse=False):
    metrics = evaluate(model, triples, all_triples, use_inverse=use_inverse)
    print("Val Metrics (filtered):", metrics["pessimistic"])
    return metrics

def train_and_eval_model(args):
    data_dir = os.path.join("data", "nations")
    train_path = os.path.join(data_dir, "train.txt")
    valid_path = os.path.join(data_dir, "valid.txt")
    train_triples = load_triples(train_path)
    valid_triples = load_triples(valid_path)
    entity2id, relation2id = load_entity_relation_dicts(data_dir)
    h_train, r_train, t_train = convert_to_id_arrays(train_triples, entity2id, relation2id)
    h_valid, r_valid, t_valid = convert_to_id_arrays(valid_triples, entity2id, relation2id)

    # Préparation modèle et loaders avec init_training
    model = init_model(args, entity2id, relation2id, args.init_function, args.seed_forward, args.seed_init)
    (
        model,
        optimizer,
        train_loader,
        valid_loader,
        neg_sampler,
        valid_neg_sampler,
        train_generator,
        valid_generator,
    ) = init_training(
        model,
        (h_train, r_train, t_train),
        (h_valid, r_valid, t_valid),
        entity2id,
        relation2id,
        args.seed_neg,
        args.seed_order,
        args.seed_init,
        args.num_neg,
        args.num_neg,
        args.batch_size,
        args.lr,
        sampler_type="reproducible_on_the_fly",
    )
    loss_fn = get_loss_fn(args.loss_type)
    if args.use_gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)

    train_losses = []
    val_losses = []
    val_mrrs = []
    # Création de l'evaluator comme dans training_utils
    all_h = np.concatenate([train_triples[0], valid_triples[0]], axis=0)
    all_r = np.concatenate([train_triples[1], valid_triples[1]], axis=0)
    all_t = np.concatenate([train_triples[2], valid_triples[2]], axis=0)  
    all_triples = (all_h, all_r, all_t)
    evaluator=lambda m, v, e, r: val_evaluator(m, v, all_triples, e, r, use_inverse=args.use_inverse)

    for epoch in range(1, args.max_epochs+1):
        _, _, _, loss = one_epoch(
            model,
            optimizer,
            train_loader,
            neg_sampler,
            epoch,
            loss_fn,
            loss_type=args.loss_type,
            is_train=True
        )
        train_losses.append(loss)

        _, _, _, eval_loss = one_epoch(model, optimizer, valid_loader, valid_neg_sampler, epoch, loss_fn, loss_type=args.loss_type, is_train=False)
        val_losses.append(eval_loss)
        # Nouvelle version avec evaluator
        # metrics = evaluator(
        #     model,
        #     (h_valid, r_valid, t_valid),
        #     entity2id,
        #     relation2id,
        # )
        # Ancienne version :
        metrics = evaluate(
            model,
            (h_valid, r_valid, t_valid),
            (h_train, r_train, t_train),
            hits_k=[1, 3, 10],
            use_inverse=False
        )
        print(epoch, metrics)
        val_mrr = metrics['pessimistic']['MRR']
        val_mrrs.append(val_mrr)
    return train_losses, val_losses, val_mrrs


def test_best_TransE_training_reproducibility():
    """
    Teste que deux entraînements consécutifs de TransE avec la même config sur codex-s produisent les mêmes courbes de loss et de MRR.
    """ 
    args = {'lr': 0.1, 'oar': False, 'model': 'TransE', 'margin': 1, 'num_neg': 5, 'run_dir': None, 'use_gpu': False, 'data_dir': 'data/nations', 'reg_type': 'none', 'seed_neg': 42, 'sweep_id': None, 'loss_type': 'ce', 'num_neg_h': 5, 'num_neg_t': 5, 'seed_init': 42, 'seed_order': 42, 'seed_forward': 42, 'visualise': False, 'batch_size': 256, 'eval_every': 5, 'init_sweep': True, 'max_epochs': 500, 'seed_range': 5, 'transE_norm': 2, 'use_inverse': True, 'log_to_wandb': True, 'seed_varying': None, 'embedding_dim': 32, 'wandb_project': None, 'dropout_entity': 0.2, 'oar_besteffort': False, 'wandb_run_name': None, 'dropout_relation': 0.2, 'early_stop_delta': 0.001, 'early_stop_metric': 'MRR', 'reg_entity_weight': 0.0001, 'dropout_projection': 0.3, 'stability_measures': False, 'stabilité训练': False, 'dropout_feature_map': 0.2, 'early_stop_patience': 50, 'reg_relation_weight': 0.0001, 'convE_embedding_shape1': 16, 'init_function': 'xavier_normal'}
    args["max_epochs"] = 2
    args = argparse.Namespace(**args)
    
    losses1, val_losses1, mrrs1 = train_and_eval_model(args)
    losses2, val_losses2, mrrs2 = train_and_eval_model(args)
    print(losses1, losses2)
    
    # On tolère un écart numérique très faible (float)
    assert np.allclose(losses1, losses2, rtol=1e-6, atol=1e-7), f"Losses différentes : {losses1} vs {losses2}"
    assert np.allclose(val_losses1, val_losses2, rtol=1e-6, atol=1e-7), f"Val losses différentes : {val_losses1} vs {val_losses2}"
    assert np.allclose(mrrs1, mrrs2, rtol=1e-6, atol=1e-7), f"MRR différents : {mrrs1} vs {mrrs2}"

def test_best_DistMult_training_reproducibility():
    """
    Teste que deux entraînements consécutifs de DistMult avec la même config sur codex-s produisent les mêmes courbes de loss et de MRR.
    """ 
    args = {'lr': 0.01, 'oar': False, 'model': 'DistMult', 'margin': 1, 'num_neg': 5, 'run_dir': None, 'use_gpu': False, 'data_dir': 'data/nations', 'reg_type': 'none', 'seed_neg': 42, 'sweep_id': None, 'loss_type': 'ce', 'num_neg_h': 5, 'num_neg_t': 5, 'seed_init': 42, 'seed_order': 42, 'seed_forward': 42, 'visualise': False, 'batch_size': 256, 'eval_every': 5, 'init_sweep': True, 'max_epochs': 500, 'seed_range': 5, 'transE_norm': 1, 'use_inverse': True, 'log_to_wandb': True, 'seed_varying': None, 'embedding_dim': 32, 'wandb_project': None, 'dropout_entity': 0.2, 'oar_besteffort': False, 'wandb_run_name': None, 'dropout_relation': 0.2, 'early_stop_delta': 0.001, 'early_stop_metric': 'MRR', 'reg_entity_weight': 0.0001, 'dropout_projection': 0.3, 'stability_measures': False, 'stabilité训练': False, 'dropout_feature_map': 0.2, 'early_stop_patience': 50, 'reg_relation_weight': 0.0001, 'convE_embedding_shape1': 16, 'init_function': 'xavier_normal'}
    args["max_epochs"] = 2
    args = argparse.Namespace(**args)
    
    losses1, val_losses1, mrrs1 = train_and_eval_model(args)
    losses2, val_losses2, mrrs2 = train_and_eval_model(args)
    print(losses1, losses2)
    
    # On tolère un écart numérique très faible (float)
    assert np.allclose(losses1, losses2, rtol=1e-6, atol=1e-7), f"Losses différentes : {losses1} vs {losses2}"
    assert np.allclose(val_losses1, val_losses2, rtol=1e-6, atol=1e-7), f"Val losses différentes : {val_losses1} vs {val_losses2}"
    assert np.allclose(mrrs1, mrrs2, rtol=1e-6, atol=1e-7), f"MRR différents : {mrrs1} vs {mrrs2}"


def test_best_ComplEx_training_reproducibility():
    """
    Teste que deux entraînements consécutifs de ComplEx avec la même config sur nations produisent les mêmes courbes de loss et de MRR.
    """ 
    args = {'lr': 0.01, 'oar': False, 'model': 'ComplEx', 'margin': 1, 'num_neg': 5, 'run_dir': None, 'use_gpu': False, 'data_dir': 'data/nations', 'reg_type': 'none', 'seed_neg': 42, 'sweep_id': None, 'loss_type': 'ce', 'num_neg_h': 5, 'num_neg_t': 5, 'seed_init': 42, 'seed_order': 42, 'seed_forward': 42, 'visualise': False, 'batch_size': 256, 'eval_every': 5, 'init_sweep': True, 'max_epochs': 500, 'seed_range': 5, 'transE_norm': 1, 'use_inverse': True, 'log_to_wandb': True, 'seed_varying': None, 'embedding_dim': 32, 'wandb_project': None, 'dropout_entity': 0.2, 'oar_besteffort': False, 'wandb_run_name': None, 'dropout_relation': 0.2, 'early_stop_delta': 0.001, 'early_stop_metric': 'MRR', 'reg_entity_weight': 0.0001, 'dropout_projection': 0.3, 'stability_measures': False, 'stabilité训练': False, 'dropout_feature_map': 0.2, 'early_stop_patience': 50, 'reg_relation_weight': 0.0001, 'convE_embedding_shape1': 16, 'init_function': 'xavier_normal'}
    args["max_epochs"] = 2
    args = argparse.Namespace(**args)
    
    losses1, val_losses1, mrrs1 = train_and_eval_model(args)
    losses2, val_losses2, mrrs2 = train_and_eval_model(args)
    print(losses1, losses2)
    
    # On tolère un écart numérique très faible (float)
    assert np.allclose(losses1, losses2, rtol=1e-6, atol=1e-7), f"Losses différentes : {losses1} vs {losses2}"
    assert np.allclose(val_losses1, val_losses2, rtol=1e-6, atol=1e-7), f"Val losses différentes : {val_losses1} vs {val_losses2}"
    assert np.allclose(mrrs1, mrrs2, rtol=1e-6, atol=1e-7), f"MRR différents : {mrrs1} vs {mrrs2}"


def test_best_RotatE_training_reproducibility():
    """
    Teste que deux entraînements consécutifs de RotatE avec la même config sur nations produisent les mêmes courbes de loss et de MRR.
    """ 
    args = {'lr': 0.01, 'oar': False, 'model': 'RotatE', 'margin': 1, 'num_neg': 5, 'run_dir': None, 'use_gpu': False, 'data_dir': 'data/nations', 'reg_type': 'none', 'seed_neg': 42, 'sweep_id': None, 'loss_type': 'ce', 'num_neg_h': 5, 'num_neg_t': 5, 'seed_init': 42, 'seed_order': 42, 'seed_forward': 42, 'visualise': False, 'batch_size': 256, 'eval_every': 5, 'init_sweep': True, 'max_epochs': 500, 'seed_range': 5, 'transE_norm': 1, 'use_inverse': True, 'log_to_wandb': True, 'seed_varying': None, 'embedding_dim': 32, 'wandb_project': None, 'dropout_entity': 0.2, 'oar_besteffort': False, 'wandb_run_name': None, 'dropout_relation': 0.2, 'early_stop_delta': 0.001, 'early_stop_metric': 'MRR', 'reg_entity_weight': 0.0001, 'dropout_projection': 0.3, 'stability_measures': False, 'stabilité训练': False, 'dropout_feature_map': 0.2, 'early_stop_patience': 50, 'reg_relation_weight': 0.0001, 'convE_embedding_shape1': 16, 'init_function': 'xavier_normal', 'rotate_norm': 2, 'normalize_phases': True}
    args["max_epochs"] = 2
    args = argparse.Namespace(**args)
    
    losses1, val_losses1, mrrs1 = train_and_eval_model(args)
    losses2, val_losses2, mrrs2 = train_and_eval_model(args)
    print(losses1, losses2)
    
    # On tolère un écart numérique très faible (float)
    assert np.allclose(losses1, losses2, rtol=1e-6, atol=1e-7), f"Losses différentes : {losses1} vs {losses2}"
    assert np.allclose(val_losses1, val_losses2, rtol=1e-6, atol=1e-7), f"Val losses différentes : {val_losses1} vs {val_losses2}"
    assert np.allclose(mrrs1, mrrs2, rtol=1e-6, atol=1e-7), f"MRR différents : {mrrs1} vs {mrrs2}"


def test_best_ConvE_training_reproducibility():
    """
    Teste que deux entraînements consécutifs de ConvE avec la même config sur codex-s produisent les mêmes courbes de loss et de MRR.
    """ 
    args =  {'lr': 0.001, 'oar': False, 'model': 'ConvE', 'margin': 1, 'num_neg': 5, 'run_dir': None, 'use_gpu': False, 'data_dir': 'data/nations', 'reg_type': 'none', 'seed_neg': 42, 'sweep_id': None, 'loss_type': 'ce', 'num_neg_h': 5, 'num_neg_t': 5, 'seed_init': 42, 'seed_forward': 42, 'visualise': False, 'batch_size': 256, 'eval_every': 5, 'init_sweep': True, 'max_epochs': 500, 'seed_order': 42, 'seed_range': 5, 'transE_norm': 1, 'use_inverse': True, 'log_to_wandb': True, 'seed_varying': None, 'embedding_dim': 32, 'wandb_project': None, 'dropout_entity': 0.2, 'oar_besteffort': False, 'wandb_run_name': None, 'dropout_relation': 0.2, 'early_stop_delta': 0.001, 'early_stop_metric': 'MRR', 'reg_entity_weight': 0.0001, 'dropout_projection': 0.2, 'stability_measures': False, 'stability_training': False, 'dropout_feature_map': 0.2, 'early_stop_patience': 50, 'reg_relation_weight': 0.0001, 'convE_embedding_shape1': 16, 'init_function': 'xavier_normal'}
    args["max_epochs"] = 2
    args = argparse.Namespace(**args)
    
    losses1, val_losses1, mrrs1 = train_and_eval_model(args)
    losses2, val_losses2, mrrs2 = train_and_eval_model(args)
    print(losses1, losses2)
    
    # On tolère un écart numérique très faible (float)
    assert np.allclose(losses1, losses2, rtol=1e-6, atol=1e-7), f"Losses différentes : {losses1} vs {losses2}"
    assert np.allclose(val_losses1, val_losses2, rtol=1e-6, atol=1e-7), f"Val losses différentes : {val_losses1} vs {val_losses2}"
    assert np.allclose(mrrs1, mrrs2, rtol=1e-6, atol=1e-7), f"MRR différents : {mrrs1} vs {mrrs2}"


import pytest

def test_gpu_training_reproducibility():
    """
    Vérifie la reproductibilité sur GPU :
    - Sans options déterministes, deux runs donnent des résultats différents
    - Avec options déterministes, deux runs donnent les mêmes résultats
    """
    if not torch.cuda.is_available():
        pytest.skip("Pas de GPU disponible")
    seed = 1234

    # Args de base pour ConvE sur GPU
    args1 = {'lr': 0.01, 'oar': False, 'model': 'ConvE', 'margin': 1, 'num_neg': 500, 'run_dir': None, 'use_gpu': True, 'data_dir': 'data/codex-s', 'reg_type': 'none', 'sweep_id': None, 'loss_type': 'ce', 'num_neg_h': 500, 'num_neg_t': 500, 'visualise': False, 'batch_size': 64, 'eval_every': 1, 'max_epochs': 5, 'seed_range': 1, 'transE_norm': 1, 'use_inverse': True, 'log_to_wandb': False, 'seed_varying': None, 'embedding_dim': 32, 'wandb_project': None, 'dropout_entity': 0.1, 'oar_besteffort': False, 'wandb_run_name': None, 'dropout_relation': 0.1, 'early_stop_delta': 0.001, 'early_stop_metric': 'MRR', 'reg_entity_weight': 0.0001, 'dropout_projection': 0.1, 'stability_measures': False, 'stabilité训练': False, 'dropout_feature_map': 0.1, 'early_stop_patience': 10, 'reg_relation_weight': 0.0001, 'convE_embedding_shape1': 8, 'init_function': 'xavier_normal'}
    args1['seed_neg'] = seed
    args1['seed_init'] = seed
    args1['seed_order'] = seed
    args1['seed_forward'] = seed
    # 1. Sans seed ni options déterministes
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    args1 = argparse.Namespace(**args1)
    losses1, val_losses1, mrrs1 = train_and_eval_model(args1)
    losses2, val_losses2, mrrs2 = train_and_eval_model(args1)
    print(losses1, val_losses1, mrrs1)
    print(losses2, val_losses2, mrrs2)    
    # On attend des différences
    different = not (np.allclose(losses1, losses2, rtol=1e-6, atol=1e-7) and np.allclose(val_losses1, val_losses2, rtol=1e-6, atol=1e-7) and np.allclose(mrrs1, mrrs2, rtol=1e-6, atol=1e-7))
    assert different, f"Les runs sans seed ni options déterministes sont identiques, ce qui n'est pas attendu !"

    # 2. Avec seed et options déterministes
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    losses3, val_losses3, mrrs3 = train_and_eval_model(args1)
    losses4, val_losses4, mrrs4 = train_and_eval_model(args1)
    print(losses3, val_losses3, mrrs3)
    print(losses4, val_losses4, mrrs4)
    # On attend de l'identique
    assert np.allclose(losses3, losses4, rtol=1e-6, atol=1e-7), f"Losses différentes malgré seed et options déterministes : {losses3} vs {losses4}"
    assert np.allclose(val_losses3, val_losses4, rtol=1e-6, atol=1e-7), f"Val losses différentes malgré seed et options déterministes : {val_losses3} vs {val_losses4}"
    assert np.allclose(mrrs3, mrrs4, rtol=1e-6, atol=1e-7), f"MRR différents malgré seed et options déterministes : {mrrs3} vs {mrrs4}"
