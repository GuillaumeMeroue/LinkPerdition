import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class KGETriples(Dataset):
    def __init__(self, triples, return_indices=False):
        self.return_indices = return_indices
        self.heads, self.relations, self.tails = triples

    def __len__(self):
        return len(self.heads)

    def __getitem__(self, idx):
        h = self.heads[idx]
        r = self.relations[idx]
        t = self.tails[idx]
        
        if self.return_indices:
            return h, r, t, idx
        return h, r, t

def load_triples(path):
    triples = []
    with open(path) as f:
        for line in f:
            h, r, t = line.strip().split('\t')
            triples.append((h, r, t))
    return triples


def convert_to_id_arrays(triples, entity2id, relation2id, use_inverse=False, split="train"):
    """Converts string triples to numpy arrays of IDs.
    
    Args:
        triples: List of (h, r, t) string triples
        entity2id: Dictionary mapping entity strings to IDs
        relation2id: Dictionary mapping relation strings to IDs
        use_inverse: If True, add inverse triples [t, r^(-1), h] for each [h, r, t]
    
    Returns:
        Tuple of (heads, relations, tails) numpy arrays
    """
    heads = [entity2id[h] for h, r, t in triples]
    relations = [relation2id[r] for h, r, t in triples]
    tails = [entity2id[t] for h, r, t in triples]
    
    if use_inverse and (split == "train" or split == "valid"):
        # Add inverse relations
        num_relations = len(relation2id)
        # Create inverse relation IDs by adding num_relations to the original relation ID
        inv_relations = [r + num_relations for r in relations]
        # For inverse triples: [t, r^(-1), h]
        heads.extend(tails)
        relations.extend(inv_relations)
        tails.extend(heads[:len(triples)])  # Original heads become tails in inverse triples
    
    return np.array(heads, dtype=np.int64), np.array(relations, dtype=np.int64), np.array(tails, dtype=np.int64)


def load_entity_relation_dicts(data_dir):
    ent_path = os.path.join(data_dir, 'ent2id.txt')
    rel_path = os.path.join(data_dir, 'rel2id.txt')
    entity2id = {}
    relation2id = {}
    # if files exist, load them
    if os.path.exists(ent_path) and os.path.exists(rel_path):
        with open(ent_path) as f:
            for line in f:
                ent, eid = line.strip().split('\t')
                entity2id[ent] = int(eid)
        with open(rel_path) as f:
            for line in f:
                rel, rid = line.strip().split('\t')
                relation2id[rel] = int(rid)
        return entity2id, relation2id
    # else, build mappings from train, valid, test
    triple_files = ['train.txt', 'valid.txt', 'test.txt']
    all_entities = set()
    all_relations = set()
    for fname in triple_files:
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Le fichier {fname} est requis dans {data_dir}")
        with open(fpath) as f:
            for line in f:
                h, r, t = line.strip().split('\t')
                all_entities.add(h)
                all_entities.add(t)
                all_relations.add(r)
    entity2id = {ent: idx for idx, ent in enumerate(sorted(all_entities))}
    relation2id = {rel: idx for idx, rel in enumerate(sorted(all_relations))}
    # save mappings
    with open(ent_path, 'w') as f:
        for ent, eid in entity2id.items():
            f.write(f"{ent}\t{eid}\n")
    with open(rel_path, 'w') as f:
        for rel, rid in relation2id.items():
            f.write(f"{rel}\t{rid}\n")
    return entity2id, relation2id



def load_torch_geometric_data(data_dir, entity2id, relation2id):
    train_file = os.path.join(data_dir, 'train.txt')
    triples = []
    
    with open(train_file, 'r') as f:
        for line in f:
            h, r, t = line.strip().split('\t')
            h_idx = entity2id[h]
            r_idx = relation2id[r]
            t_idx = entity2id[t]
            triples.append((h_idx, r_idx, t_idx))
    
    num_edges = len(triples)
    edge_index = torch.zeros((2, num_edges), dtype=torch.long)
    edge_type = torch.zeros(num_edges, dtype=torch.long)
    
    for i, (h, r, t) in enumerate(triples):
        edge_index[0, i] = h  # head
        edge_index[1, i] = t  # tail
        edge_type[i] = r      # relation
    
    return edge_index, edge_type

class PrecomputedNegativeSampler:

    def __init__(self, num_entities, seed, num_neg_h=1, num_neg_t=1, triples=None):

        self.num_entities = num_entities
        self.seed = seed
        self.num_neg_h = num_neg_h
        self.num_neg_t = num_neg_t
        
        # Store all training triples as a set for fast lookup
        h, r, t = triples
        self.train_triples_set = {tuple(row) for row in np.stack([h, r, t], axis=1)}
        self.triples = np.stack([h, r, t], axis=1)
        
        # Precompute all negative samples
        self.neg_dict_head, self.neg_dict_tail = self._precompute_negatives()

    def get_state(self):
        # This sampler is stateless after initialization, but we add the methods for consistency
        return {}

    def set_state(self, state):
        # This sampler is stateless after initialization
        pass
    
    def _precompute_negatives(self):
        neg_dict_head = {}
        neg_dict_tail = {}
        
        for idx, (h, r, t) in enumerate(tqdm(self.triples, desc="Precomputing negative samples")):
            # Convert to Python native types for set lookup
            h, r, t = int(h), int(r), int(t)
            pos_triple = (h, r, t)
            
            # Precompute tail corruptions (h, r, ?)
            # Generate tail corruptions (h, r, ?)
            negs_tail = torch.zeros((self.num_neg_t, 3), dtype=torch.long)
            negs_tail[:, 0] = h  # h is the same
            negs_tail[:, 1] = r  # r is the same
            
            # Generate head corruptions (?, r, t)
            negs_head = torch.zeros((self.num_neg_h, 3), dtype=torch.long)
            negs_head[:, 1] = r  # r is the same
            negs_head[:, 2] = t  # t is the same
            
            # Generate negative tails
            g = torch.Generator()
            g.manual_seed(self.seed + idx)
            i = 0
            while i < self.num_neg_t:
                candidates = torch.randint(0, self.num_entities, (self.num_neg_t * 2,), generator=g, dtype=torch.long)
                for neg in candidates.tolist():
                    if (h, r, neg) not in self.train_triples_set: # and neg not in negs_tail[:i, 2]
                        negs_tail[i, 2] = neg
                        i += 1
                        if i >= self.num_neg_t:
                            break
            
            # Generate negative heads
            g = torch.Generator()
            g.manual_seed(self.seed + idx)
            i = 0
            while i < self.num_neg_h:
                candidates = torch.randint(0, self.num_entities, (self.num_neg_h * 2,), generator=g, dtype=torch.long)
                for neg in candidates.tolist():
                    if (neg, r, t) not in self.train_triples_set: # and neg not in negs_head[:i, 0]
                        negs_head[i, 0] = neg
                        i += 1
                        if i >= self.num_neg_h:
                            break
            
            neg_dict_tail[pos_triple] = negs_tail
            neg_dict_head[pos_triple] = negs_head
            
        return neg_dict_head, neg_dict_tail
    
    def sample(self, batch, mode="tail"):
        h_cpu, r_cpu, t_cpu, idx_cpu = batch
        batch_size = h_cpu.size(0)
        
        if mode == "triples":
            # Return positive and negative triples
            pos_triples = torch.stack([h_cpu, r_cpu, t_cpu], dim=1)  # [batch_size, 3]
            
            # Get tail corruptions
            tail_negs = []
            for i in range(batch_size):
                triple = (h_cpu[i].item(), r_cpu[i].item(), t_cpu[i].item())
                tail_negs.append(self.neg_dict_tail[triple])
            tail_negs = torch.stack(tail_negs, dim=0)  # [batch_size, num_neg_t, 3]
            
            # Get head corruptions
            head_negs = []
            for i in range(batch_size):
                triple = (h_cpu[i].item(), r_cpu[i].item(), t_cpu[i].item())
                head_negs.append(self.neg_dict_head[triple])
            head_negs = torch.stack(head_negs, dim=0)  # [batch_size, num_neg_h, 3]
            
            return {
                'pos': pos_triples,  # [batch_size, 3]
                'tail_neg': tail_negs,  # [batch_size, num_neg_t, 3]
                'head_neg': head_negs,  # [batch_size, num_neg_h, 3]
            }
        elif mode == "tail":
            num_neg = self.num_neg_t
            pos_entities = t_cpu.view(-1, 1)  # [batch_size, 1]
            neg_samples = torch.zeros((batch_size, num_neg), dtype=torch.long)
            
            for i in range(batch_size):
                triple = (h_cpu[i].item(), r_cpu[i].item(), t_cpu[i].item())
                neg_samples[i] = self.neg_dict_tail[triple][:, 2]  # Extract only the tail entities
                
            return torch.cat([pos_entities, neg_samples], dim=1)
            
        elif mode == "head":
            num_neg = self.num_neg_h
            pos_entities = h_cpu.view(-1, 1)  # [batch_size, 1]
            neg_samples = torch.zeros((batch_size, num_neg), dtype=torch.long)
            
            for i in range(batch_size):
                triple = (h_cpu[i].item(), r_cpu[i].item(), t_cpu[i].item())
                neg_samples[i] = self.neg_dict_head[triple][:, 0]  # Extract only the head entities
                
            return torch.cat([pos_entities, neg_samples], dim=1)
            
        else:
            raise ValueError(f"Unknown corruption mode: {mode}")

class ReproductibleOnTheFlyNegativeSampler:
    def __init__(self, num_entities, seed, num_neg_h=1, num_neg_t=1, triples=None):
        self.num_entities = num_entities
        self.seed = seed
        self.num_neg_h = num_neg_h
        self.num_neg_t = num_neg_t

        h, r, t = triples
        self.train_triples_set = {tuple(row) for row in np.stack([h, r, t], axis=1)}



    def get_state(self): # Will be used to get the internal counter in next version
        return {}

    def set_state(self, state): # Will be used to set the internal counter in next version
        pass

    def sample(self, batch, mode="tail"):
        h_cpu, r_cpu, t_cpu, idx_cpu = batch
        idx_cpu = idx_cpu.tolist()
        batch_size = h_cpu.size(0)

        if mode == "tail":
            num_neg = self.num_neg_t
            pos_entities_cpu = t_cpu
            neg_samples = torch.zeros((batch_size, num_neg), dtype=torch.long)

            for i in range(batch_size):
                # Utilise torch.Generator pour seed
                g = torch.Generator()
                g.manual_seed(self.seed + idx_cpu[i])
                # It would have been better if I used an internal counter but it was ok in this experiment as the triples order also change, so the negatives.
                h_i, r_i = h_cpu[i].item(), r_cpu[i].item()

                generated_negs = []
                # Oversample to find valid negatives faster
                oversample_factor = 2
                while len(generated_negs) < num_neg:
                    candidates = torch.randint(0, self.num_entities, (num_neg * oversample_factor,), generator=g, dtype=torch.long)
                    for neg in candidates.tolist():
                        if (h_i, r_i, neg) not in self.train_triples_set:
                            generated_negs.append(neg)
                        if len(generated_negs) >= num_neg:
                            break
                neg_samples[i, :] = torch.tensor(generated_negs[:num_neg], dtype=torch.long)

            pos_entities_tensor = pos_entities_cpu.view(-1, 1)
            return torch.cat([pos_entities_tensor, neg_samples], dim=1)

        elif mode == "head":
            num_neg = self.num_neg_h
            pos_entities_cpu = h_cpu
            neg_samples = torch.zeros((batch_size, num_neg), dtype=torch.long)

            for i in range(batch_size):
                g = torch.Generator()
                g.manual_seed(self.seed + idx_cpu[i])
                r_i, t_i = r_cpu[i].item(), t_cpu[i].item()

                generated_negs = []
                oversample_factor = 2
                while len(generated_negs) < num_neg:
                    candidates = torch.randint(0, self.num_entities, (num_neg * oversample_factor,), generator=g, dtype=torch.long)
                    for neg in candidates.tolist():
                        if (neg, r_i, t_i) not in self.train_triples_set:
                            generated_negs.append(neg)
                        if len(generated_negs) >= num_neg:
                            break
                neg_samples[i, :] = torch.tensor(generated_negs[:num_neg], dtype=torch.long)

            pos_entities_tensor = pos_entities_cpu.view(-1, 1)
            return torch.cat([pos_entities_tensor, neg_samples], dim=1)

        else:
            raise ValueError(f"Unknown mode: {mode}")



def get_dataloader(triples_arrays, batch_size=128, seed_order=42, shuffle=True, return_indices=False):
    """
    Creates a DataLoader for training/validation/test triples from numpy arrays.
    """
    dataset = KGETriples(triples_arrays, return_indices=return_indices)
    
 
    # Créer le générateur pour la reproductibilité
    generator = torch.Generator()
    generator.manual_seed(seed_order)
    
    # Créer le DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0
    )
    return loader, generator
