import torch
import pytest
import numpy as np
from torch.utils.data import DataLoader
from kge.models import TransE
from kge.utils import compare_model_params, set_seed
from kge.data import (
    load_triples, 
    load_entity_relation_dicts, 
    ReproductibleOnTheFlyNegativeSampler,
    KGETriples,
    convert_to_id_arrays
)

import os
import argparse
from training_utils import init_model
from .test_utils import train_and_eval_model

SEED_INIT = 1
SEED_NEG = 11
SEED_ORDER = 111
SEED_FORWARD = 1111
def get_init_state(seed_init, seed_neg, seed_order, seed_forward, batch_size=8):
    data_dir = os.path.join("data", "nations")
    train_path = os.path.join(data_dir, "train.txt")
    triples = load_triples(train_path)
    entity2id, relation2id = load_entity_relation_dicts(data_dir)
    # Conversion en arrays d'IDs
    h_array, r_array, t_array = convert_to_id_arrays(triples, entity2id, relation2id)
    
    # Initialiser le modèle
    args = argparse.Namespace()
    args.model = "DistMult"
    args.data_dir = "data/nations"
    args.batch_size = batch_size
    args.embedding_dim = 16
    args.lr = 0.01
    args.num_neg = 5
    args.num_epochs = 6
    args.seed_init = seed_init
    args.seed_neg = seed_neg
    args.seed_order = seed_order
    args.seed_forward = seed_forward
    args.dropout_entity = 0.2
    args.dropout_relation = 0.2
    args.use_inverse = False  
    args.init_function = "xavier_normal"

    model = init_model(args, entity2id, relation2id, args.init_function, args.seed_forward, args.seed_init)

    
    # Initialiser l'optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Créer le dataloader avec indices grâce à get_dataloader
    from kge.data import get_dataloader
    dataloader, _ = get_dataloader((h_array, r_array, t_array), batch_size=batch_size, seed_order=seed_order, shuffle=True, return_indices=True)
    
    # Initialiser le sampler de négatifs
    neg_sampler = ReproductibleOnTheFlyNegativeSampler(
        num_entities=len(entity2id),
        seed=seed_neg,
        num_neg_h=5,
        num_neg_t=5,
        triples=(h_array, r_array, t_array)
    )
    
    # # Récupérer les paramètres initiaux
    # params = [p.detach().clone() for p in model.parameters()]
    
    # Récupérer l'ordre des batches et les triplets négatifs
    batch_orders = []
    all_neg_triples = set()
    
    for batch in dataloader:
        # batch: (batch_size, 4) (h, r, t, idx)
        h, r, t, idx = batch
        batch_pos = list(zip(h.tolist(), r.tolist(), t.tolist()))
        batch_orders.append(batch_pos)

        # Générer les négatifs pour ce batch avec la nouvelle API
        batch_tuple = (h, r, t, idx)
        negs = neg_sampler.sample(batch_tuple, mode="tail")
        # Les négatifs sont dans negs[:, 1:]
        for i in range(negs.size(0)):
            for neg_t in negs[i, 1:].tolist():
                all_neg_triples.add((h[i].item(), r[i].item(), neg_t))
    
    return model, batch_orders, all_neg_triples

# def params_equal(model1, model2):
#     for p1, p2 in zip(model1.parameters(), model2.parameters()):
#         if not torch.equal(p1, p2):
#             print("Paramètres différents :", p1.name, p2.name)
#             return False
#     return True


def params_equal(model1, model2):
    for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
        if not torch.equal(param1, param2):
            return False, f"Model weights differ at layer: {name1}"
    return True, None


def test_pipeline_param_seed():
    model1, _, _ = get_init_state(SEED_INIT, SEED_NEG, SEED_ORDER, SEED_FORWARD)
    model2, _, _ = get_init_state(SEED_INIT, SEED_NEG + 1, SEED_ORDER + 1, SEED_FORWARD + 1)
    model3, _, _ = get_init_state(SEED_INIT + 1, SEED_NEG, SEED_ORDER, SEED_FORWARD)


    # Même seed d'initialisation, autres seeds différentes → paramètres identiques
    equal, msg = params_equal(model1, model2)
    assert equal, f"Les paramètres devraient être identiques (même seed d'init) mais : {msg}"    
    
    # Seed d'initialisation différente, autres seeds identiques → paramètres différents
    equal, _ = params_equal(model1, model3)
    assert not equal, "Les paramètres devraient être différents (seed d'init différente)"

def test_pipeline_neg_seed():
    # Tester la reproductibilité des négatifs avec la même seed de négatifs
    _, _, negs1 = get_init_state(SEED_INIT, SEED_NEG, SEED_ORDER, SEED_FORWARD)
    _, _, negs2 = get_init_state(SEED_INIT + 1, SEED_NEG, SEED_ORDER + 1, SEED_FORWARD + 1)
    _, _, negs3 = get_init_state(SEED_INIT, SEED_NEG + 1, SEED_ORDER, SEED_FORWARD)
    
    # Même seed de négatifs → mêmes négatifs générés
    assert negs1 == negs2, "Les négatifs devraient être identiques avec la même seed de négatifs"
    
    # Seed de négatifs différente → négatifs différents
    assert negs1 != negs3, "Les négatifs devraient être différents avec une seed de négatifs différente"

def test_pipeline_order_seed():
    # Tester la reproductibilité de l'ordre des batches avec la même seed d'ordre
    _, order1, _ = get_init_state(SEED_INIT, SEED_NEG, SEED_ORDER, SEED_FORWARD)
    _, order2, _ = get_init_state(SEED_INIT + 1, SEED_NEG + 1, SEED_ORDER, SEED_FORWARD + 1)
    _, order3, _ = get_init_state(SEED_INIT, SEED_NEG, SEED_ORDER + 1, SEED_FORWARD)
    
    # Même seed d'ordre → même ordre des batches
    assert order1 == order2, "L'ordre des batches devrait être identique avec la même seed d'ordre"
    
    # Seed d'ordre différente → ordre des batches différent
    assert order1 != order3, "L'ordre des batches devrait être différent avec une seed d'ordre différente"


def test_pipeline_forward_seed():
    # Tester l'impact de la seed forward sur l'entraînement
    # On crée un modèle et on l'entraîne avec deux seeds forward différentes
    # On s'attend à ce que les courbes de loss soient différentes
    
    # Configuration de base pour l'entraînement
    args = {
        'model': 'DistMult',
        'embedding_dim': 32,
        'batch_size': 8,
        'max_epochs': 2,
        'lr': 0.001,
        'seed_init': SEED_INIT,
        'seed_neg': SEED_NEG,
        'seed_order': SEED_ORDER,
        'seed_forward': SEED_FORWARD,
        'loss_type': 'bce',
        'reg_type': 'none',
        'reg_entity_weight': 0.0,
        'reg_relation_weight': 0.0,
        'use_inverse': True,
        'log_to_wandb': False,
        'init_function': 'xavier_normal',
        'eval_every': 1,
        'early_stop_metric': 'loss',
        'early_stop_patience': 10,
        'early_stop_delta': 0.001,
        'dropout_feature_map': 0.2,
        'dropout_projection': 0.2,
        'dropout_entity': 0.2,
        'dropout_relation': 0.2,
        'convE_embedding_shape1': 16,
        'use_gpu': False,
        'num_neg': 5,
    }
    
    # Premier entraînement avec SEED_FORWARD
    args1 = args.copy()
    args1 = argparse.Namespace(**args1)
    train_losses1, val_losses1, _, _, _, _ = train_and_eval_model(args1)

    args2 = args.copy()
    args2 = argparse.Namespace(**args2)
    train_losses2, val_losses2, _, _, _, _ = train_and_eval_model(args2)

    # Deuxième entraînement avec SEED_FORWARD + 1
    args3 = args.copy()
    args3['seed_forward'] = SEED_FORWARD + 1
    args3 = argparse.Namespace(**args3)
    train_losses3, val_losses3, _, _, _, _ = train_and_eval_model(args3)

    assert np.allclose(train_losses1, train_losses2, rtol=1e-5, atol=1e-5), \
        "Loss curves should be the same with the same seed forward"
    assert np.allclose(val_losses1, val_losses2, rtol=1e-5, atol=1e-5), \
        "Loss curves should be the same with the same seed forward"

    assert not np.allclose(train_losses1, train_losses3, rtol=1e-5, atol=1e-5), \
        "Loss curves should be different with different seed forward"
    assert not np.allclose(val_losses1, val_losses3, rtol=1e-5, atol=1e-5), \
        "Loss curves should be different with different seed forward"