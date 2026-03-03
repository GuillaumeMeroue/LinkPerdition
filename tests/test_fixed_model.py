import os
import torch
import numpy as np
import pytest
from kge.eval import evaluate
from tests.test_utils import train_and_eval_model, load_train_val_data
from argparse import Namespace
from kge.data import load_triples, load_entity_relation_dicts

SEED = 42

# Test 1: 
# Allow to test the eval function (filter and all_triples_scores)
def test_evaluate_filters_train_triples():
    data_dir = os.path.join("data", "toy")
    entity2id, relation2id = load_entity_relation_dicts(data_dir)
    from kge.data import convert_to_id_arrays
    train_triples = load_triples(os.path.join(data_dir, "train.txt"))
    valid_triples = load_triples(os.path.join(data_dir, "valid.txt"))
    test_triples = load_triples(os.path.join(data_dir, "test.txt"))
    all_triples_txt = train_triples + valid_triples + test_triples
    h_train, r_train, t_train = convert_to_id_arrays(train_triples, entity2id, relation2id)
    h_test, r_test, t_test = convert_to_id_arrays(test_triples, entity2id, relation2id)
    all_triples = (h_train, r_train, t_train)

    args = Namespace(
        model="Fixed",
        embedding_dim=1,
        seed_forward=SEED,
        seed_init=SEED,
        use_inverse=False,
        dropout_entity=0.2,
        dropout_relation=0.2,
        init_function="xavier_normal",
        transE_norm=1,
    )
    from training_utils import init_model
    model = init_model(args, entity2id, relation2id, args.init_function, args.seed_forward, args.seed_init)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}

    # Build set of all (h, r, t) in train.txt U test.txt as IDs
    all_triples_set = set()
    for h, r, t in all_triples_txt:
        h_id = entity2id[h]
        r_id = relation2id[r.lower() if r.lower() in relation2id else r]
        t_id = entity2id[t]
        all_triples_set.add((h_id, r_id, t_id))

    metrics, preds = evaluate(model, (h_test, r_test, t_test), all_triples, return_preds=True, use_inverse=False, k_value=len(entity2id), filter=True)


    # For each test triple (h, r, t), check that for every entity e, if (h, r, e) in train.txt, then score == -inf
    for idx, (h, r, t) in enumerate(zip(h_test, r_test, t_test)):
        for e_id in entity2id.values():
            pos = preds['top_k_entities'][2*idx].index(e_id)
            score = preds['top_k_scores'][2*idx][pos]
            if (h, r, e_id) in all_triples_set and e_id != t:
                assert score == -float('inf'), f"Score not -inf for filtered entity: ({id2entity[h]},{id2relation[r]},{id2entity[e_id]}) in test triple ({id2entity[h]},{id2relation[r]},{id2entity[t]})"
            else:
                computed_score = model.forward(torch.tensor([h]), torch.tensor([r]), torch.tensor([e_id]), score_mode="triple")
                print(computed_score, score)
                assert computed_score == score, f"Score not fixed: ({id2entity[h]},{id2relation[r]},{id2entity[e_id]})"
            

            pos = preds['top_k_entities'][2*idx+1].index(e_id)
            score = preds['top_k_scores'][2*idx+1][pos]
            if (e_id, r, t) in all_triples_set and e_id != h:
                assert score == -float('inf'), f"Score not -inf for filtered entity: ({id2entity[e_id]},{id2relation[r]},{id2entity[t]}) in test triple ({id2entity[e_id]},{id2relation[r]},{id2entity[t]})"
            else:
                computed_score = model.forward(torch.tensor([e_id]), torch.tensor([r]), torch.tensor([t]), score_mode="triple")
                print(computed_score, score)
                assert computed_score == score, f"Score not fixed: ({id2entity[e_id]},{id2relation[r]},{id2entity[t]}), ({e_id, r, t}) ({computed_score},{score})"


# Test 2 : 
# Allow to test for seed_forward separation
def test_seed_forward_determinism():
    data_dir = os.path.join("data", "toy")
    # Paramètres communs
    args_base = dict(
        model="Fixed",
        embedding_dim=1,
        use_inverse=False,
        dropout_entity=0.5,
        dropout_relation=0.5,
        init_function="xavier_normal",
        transE_norm=1,
        batch_size=1000,
        lr=0.01,
        max_epochs=1,
        loss_type="bce",
        ce=None,
        num_neg=1,
        seed_neg=SEED,
        seed_order=SEED,
        seed_init=SEED,
        use_gpu=False,
    )
    # Même seed_forward -> loss identique
    args1 = Namespace(**{**args_base, 'seed_forward': SEED})
    args2 = Namespace(**{**args_base, 'seed_forward': SEED, 'seed_init': SEED+1, 'seed_order': SEED, 'seed_neg': SEED}) # Loss will be dependant of seed_neg and seed_order
    out1 = train_and_eval_model(args1, data_dir=data_dir)
    out2 = train_and_eval_model(args2, data_dir=data_dir)
    assert np.allclose(out1[0], out2[0]), f"Losses are different with same seed_forward: {out1[0]} vs {out2[0]}"
    assert np.allclose(out1[1], out2[1]), f"Val losses are different with same seed_forward: {out1[1]} vs {out2[1]}"
    assert np.allclose(out1[2], out2[2]), f"MRR are different with same seed_forward: {out1[2]} vs {out2[2]}"
    # Seed_forward différent -> loss différente (au moins une valeur)
    args3 = Namespace(**{**args_base, 'seed_forward': SEED+1})
    out3 = train_and_eval_model(args3, data_dir=data_dir)
    assert not np.allclose(out1[0], out3[0]), f"Losses are different with different seed_forward: {out1[0]} vs {out3[0]}"
    assert not np.allclose(out1[1], out3[1]), f"Val losses are different with different seed_forward: {out1[1]} vs {out3[1]}"
    assert np.allclose(out1[2], out3[2]), f"MRR are different with different seed_forward: {out1[2]} vs {out3[2]}" # There is no dropout for evaluation
