import os
import torch
import numpy as np
from kge.train import init_training, get_loss_fn, one_epoch
from kge.eval import evaluate
from kge.data import load_triples, load_entity_relation_dicts, convert_to_id_arrays
from training_utils import init_model

def load_train_val_data(data_dir):
    """Charge les données d'entraînement et de validation."""
    train_path = os.path.join(data_dir, "train.txt")
    valid_path = os.path.join(data_dir, "valid.txt")
    train_triples = load_triples(train_path)
    valid_triples = load_triples(valid_path)
    entity2id, relation2id = load_entity_relation_dicts(data_dir)
    
    h_train, r_train, t_train = convert_to_id_arrays(train_triples, entity2id, relation2id)
    h_valid, r_valid, t_valid = convert_to_id_arrays(valid_triples, entity2id, relation2id)
    
    return (h_train, r_train, t_train), (h_valid, r_valid, t_valid), entity2id, relation2id

def train_and_eval_model(args, data_dir="data/nations"):
    """
    Entraîne et évalue un modèle avec les paramètres donnés.
    
    Args:
        args: Namespace contenant les paramètres du modèle et de l'entraînement
        data_dir: Chemin vers le dossier contenant les données
        
    Returns:
        tuple: (train_losses, val_losses, val_mrrs)
    """
    # Chargement des données
    (h_train, r_train, t_train), (h_valid, r_valid, t_valid), entity2id, relation2id = load_train_val_data(data_dir)

    # Initialisation du modèle et des loaders
    model = init_model(args, entity2id, relation2id, args.init_function, args.seed_forward, args.seed_init)
    model, optimizer, train_loader, valid_loader, neg_sampler, valid_neg_sampler, train_generator, valid_generator = init_training(
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
    
    # Initialiser la fonction de perte avec la marge si nécessaire
    loss_fn = get_loss_fn(args.loss_type, margin=getattr(args, 'ce', None))
    device = torch.device("cuda" if args.use_gpu else "cpu")
    model.to(device)

    # Initialisation des listes de métriques
    train_losses = []
    val_losses = []
    val_mrrs = []

    # Création des triplets pour l'évaluation
    all_h = np.concatenate([h_train, h_valid], axis=0)
    all_r = np.concatenate([r_train, r_valid], axis=0)
    all_t = np.concatenate([t_train, t_valid], axis=0)
    all_triples = (all_h, all_r, all_t)

    all_neg_triples_list = []
    batch_orders_list = []
    
    # Initialisation de model_state avec l'état initial du modèle
    model_state = model.state_dict()
    
    # Boucle d'entraînement
    for epoch in range(1, args.max_epochs + 1):
        # Phase d'entraînement
        model_state, batch_orders, all_neg_triples, loss = one_epoch(
            model, optimizer, train_loader, neg_sampler, 
            epoch, loss_fn, loss_type=args.loss_type, is_train=True, debug=True
        )
        train_losses.append(loss)
        all_neg_triples_list.append(all_neg_triples)
        batch_orders_list.append(batch_orders)

        # Phase de validation
        _, _, _, eval_loss = one_epoch(
            model, optimizer, valid_loader, valid_neg_sampler, 
            epoch, loss_fn, loss_type=args.loss_type, is_train=False
        )
        val_losses.append(eval_loss)

        # Évaluation des métriques
        metrics = evaluate(
            model,
            (h_valid, r_valid, t_valid),
            (h_train, r_train, t_train),
            hits_k=[1, 3, 10],
            use_inverse=args.use_inverse
        )
        print(f"Epoch {epoch}: {metrics}")
        val_mrrs.append(metrics['pessimistic']['MRR'])

    return train_losses, val_losses, val_mrrs, model_state, batch_orders_list, all_neg_triples_list
