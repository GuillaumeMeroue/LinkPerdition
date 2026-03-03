import torch
import numpy as np
import random

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compare_model_params(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if not torch.equal(p1, p2):
            return False
    return True

def compare_negative_samples(neg1, neg2):
    return set(neg1) == set(neg2)

def compare_triple_order(order1, order2):
    return order1 == order2

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(path, model_class, num_entities, num_relations, embedding_dim):
    model = model_class(num_entities, num_relations, embedding_dim)
    model.load_state_dict(torch.load(path))
    return model

def get_init_function(init_type):
    if init_type == 'xavier_normal':
        return torch.nn.init.xavier_normal_
    elif init_type == 'xavier_uniform':
        return torch.nn.init.xavier_uniform_
    elif init_type == 'kaiming_normal':
        return torch.nn.init.kaiming_normal_
    elif init_type == 'kaiming_uniform':
        return torch.nn.init.kaiming_uniform_
    elif init_type == 'normal':
        return torch.nn.init.normal_
    elif init_type == 'uniform':
        return torch.nn.init.uniform_
    else:
        raise ValueError(f"Unknown init type: {init_type}")
    # return getattr(torch.nn.init, init_type)

def check_gpu_availability():
    if not torch.cuda.is_available():
        import warnings
        warnings.warn("CUDA is not available. Falling back to CPU.")
        return False
    return True