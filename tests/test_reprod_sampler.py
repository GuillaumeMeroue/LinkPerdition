import torch
import pytest
import numpy as np
from torch.utils.data import DataLoader
from kge.data import (
    load_triples, 
    load_entity_relation_dicts,
    ReproductibleOnTheFlyNegativeSampler,
    KGETriples
)
from kge.utils import set_seed

# Chemins vers les données de test
DATA_DIR = "data/WN18RR"

# Seeds pour les tests
SEED_INIT = 42
SEED_NEG = 123
SEED_ORDER = 456

@pytest.fixture
def load_test_data():
    # Charger les données de test
    train_path = f"{DATA_DIR}/train.txt"
    entity2id, relation2id = load_entity_relation_dicts(DATA_DIR)
    triples = load_triples(train_path)
    # Conversion en arrays d'IDs
    from kge.data import convert_to_id_arrays
    h_array, r_array, t_array = convert_to_id_arrays(triples, entity2id, relation2id)
    return triples, entity2id, relation2id, (h_array, r_array, t_array)

def test_reproducibility_same_seed(load_test_data):
    triples, entity2id, relation2id, id_arrays = load_test_data
    num_entities = len(entity2id)
    
    # Créer deux samplers avec la même seed
    sampler1 = ReproductibleOnTheFlyNegativeSampler(
        num_entities=num_entities,
        seed=SEED_NEG,
        num_neg_h=5,
        num_neg_t=5,
        triples=id_arrays
    )
    
    sampler2 = ReproductibleOnTheFlyNegativeSampler(
        num_entities=num_entities,
        seed=SEED_NEG,  # Même seed
        num_neg_h=5,
        num_neg_t=5,
        triples=id_arrays
    )
    
    # Créer un batch de test avec des indices
    batch_size = 32
    indices = torch.arange(batch_size)
    h = torch.randint(0, num_entities, (batch_size,))
    r = torch.randint(0, len(relation2id), (batch_size,))
    t = torch.randint(0, num_entities, (batch_size,))
    batch = list(zip(h, r, t, indices))
    
    # Générer des négatifs avec les deux samplers
    h, r, t, idx = zip(*batch)
    h = torch.stack(h)
    r = torch.stack(r)
    t = torch.stack(t)
    idx = torch.stack(idx)
    batch_tensors = (h, r, t, idx)
    negs1 = sampler1.sample(batch_tensors, mode="tail")
    negs2 = sampler2.sample(batch_tensors, mode="tail")
    
    # Vérifier que les négatifs sont identiques avec la même seed
    assert torch.equal(negs1, negs2), "Les négatifs devraient être identiques avec la même seed"

def test_different_seeds_produce_different_samples(load_test_data):
    triples, entity2id, relation2id, id_arrays = load_test_data
    num_entities = len(entity2id)
    
    # Créer deux samplers avec des seeds différentes
    sampler1 = ReproductibleOnTheFlyNegativeSampler(
        num_entities=num_entities,
        seed=SEED_NEG,
        num_neg_h=5,
        num_neg_t=5,
        triples=id_arrays
    )
    
    sampler2 = ReproductibleOnTheFlyNegativeSampler(
        num_entities=num_entities,
        seed=SEED_NEG + 1,  # Seed différente
        num_neg_h=5,
        num_neg_t=5,
        triples=id_arrays
    )
    
    # Créer un batch de test avec des indices
    batch_size = 32
    indices = torch.arange(batch_size)
    h = torch.randint(0, num_entities, (batch_size,))
    r = torch.randint(0, len(relation2id), (batch_size,))
    t = torch.randint(0, num_entities, (batch_size,))
    batch = list(zip(h, r, t, indices))
    
    # Générer des négatifs avec les deux samplers
    h, r, t, idx = zip(*batch)
    h = torch.stack(h)
    r = torch.stack(r)
    t = torch.stack(t)
    idx = torch.stack(idx)
    batch_tensors = (h, r, t, idx)
    negs1 = sampler1.sample(batch_tensors, mode="tail")
    negs2 = sampler2.sample(batch_tensors, mode="tail")
    
    # Vérifier que les négatifs sont différents avec des seeds différentes
    assert not torch.equal(negs1, negs2), "Les négatifs devraient être différents avec des seeds différentes"

def test_negative_samples_are_negative(load_test_data):
    triples, entity2id, relation2id, id_arrays = load_test_data
    num_entities = len(entity2id)
    
    sampler = ReproductibleOnTheFlyNegativeSampler(
        num_entities=num_entities,
        seed=SEED_NEG,
        num_neg_h=5,
        num_neg_t=5,
        triples=id_arrays
    )
    
    # Créer un batch de test avec des indices
    batch_size = 32
    indices = torch.arange(batch_size)
    h = torch.randint(0, num_entities, (batch_size,))
    r = torch.randint(0, len(relation2id), (batch_size,))
    t = torch.randint(0, num_entities, (batch_size,))
    batch = list(zip(h, r, t, indices))
    
    # Tester le mode "tail"
    h, r, t, idx = zip(*batch)
    h = torch.stack(h)
    r = torch.stack(r)
    t = torch.stack(t)
    idx = torch.stack(idx)
    batch_tensors = (h, r, t, idx)
    negs = sampler.sample(batch_tensors, mode="tail")
    for i, (_, _, t_i, _) in enumerate(batch):
        assert t_i not in negs[i, 1:], "Le vrai triplet ne devrait pas être dans les négatifs"
    
    # Tester le mode "head"
    negs = sampler.sample(batch_tensors, mode="head")
    for i, (h_i, _, _, _) in enumerate(batch):
        assert h_i not in negs[i, 1:], "Le vrai triplet ne devrait pas être dans les négatifs"

def test_dataloader_integration(load_test_data):
    triples, entity2id, relation2id, id_arrays = load_test_data
    num_entities = len(entity2id)
    
    # Créer un dataset qui retourne les indices
    dataset = KGETriples(id_arrays, return_indices=True)
    
    # Fonction pour créer un batch au bon format pour le sampler
    def collate_fn(batch):
        # batch est une liste de tuples (h, r, t, idx)
        h, r, t, idx = zip(*batch)
        return list(zip(
            torch.tensor(h, dtype=torch.long),
            torch.tensor(r, dtype=torch.long),
            torch.tensor(t, dtype=torch.long),
            torch.tensor(idx, dtype=torch.long)
        ))
    
    # Créer un dataloader avec une seed fixe
    generator = torch.Generator()
    generator.manual_seed(SEED_ORDER)
    dataloader = DataLoader(
        dataset, 
        batch_size=32, 
        shuffle=True, 
        generator=generator,
        collate_fn=collate_fn
    )
    
    # Créer le sampler
    sampler = ReproductibleOnTheFlyNegativeSampler(
        num_entities=num_entities,
        seed=SEED_NEG,
        num_neg_h=5,
        num_neg_t=5,
        triples=id_arrays
    )
    
    # Vérifier que le premier batch est reproductible
    batch1 = next(iter(dataloader))
    h, r, t, idx = zip(*batch1)
    h = torch.stack(h)
    r = torch.stack(r)
    t = torch.stack(t)
    idx = torch.stack(idx)
    batch1_tensors = (h, r, t, idx)
    negs1 = sampler.sample(batch1_tensors, mode="tail")
    
    # Recréer le dataloader avec la même seed
    generator = torch.Generator()
    generator.manual_seed(SEED_ORDER)
    dataloader = DataLoader(
        dataset, 
        batch_size=32, 
        shuffle=True, 
        generator=generator,
        collate_fn=collate_fn
    )
    
    batch2 = next(iter(dataloader))
    h2, r2, t2, idx2 = zip(*batch2)
    h2 = torch.stack(h2)
    r2 = torch.stack(r2)
    t2 = torch.stack(t2)
    idx2 = torch.stack(idx2)
    batch2_tensors = (h2, r2, t2, idx2)
    negs2 = sampler.sample(batch2_tensors, mode="tail")
    
    # Vérifier que les batches et les négatifs sont identiques
    assert len(batch1) == len(batch2), "Les batches devraient avoir la même taille"
    for i in range(len(batch1)):
        assert torch.equal(batch1[i][0], batch2[i][0]), f"Têtes différentes à l'index {i}"
        assert torch.equal(batch1[i][1], batch2[i][1]), f"Relations différentes à l'index {i}"
        assert torch.equal(batch1[i][2], batch2[i][2]), f"Queues différentes à l'index {i}"
    
    assert torch.equal(negs1, negs2), "Les négatifs devraient être identiques avec la même seed"
