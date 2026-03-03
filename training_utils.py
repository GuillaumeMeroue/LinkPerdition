import os
import torch
import json
from kge.data import load_entity_relation_dicts, load_torch_geometric_data
from kge.models import TransE, DistMult, ComplEx, RotatE, ConvE, RGCN, Transformer
from kge.eval import evaluate
from kge.train import train_kge
import numpy as np

def dump_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def val_evaluator(model, triples, all_triples, entity2id, relation2id, use_inverse=False):
    metrics = evaluate(model, triples, all_triples, use_inverse=use_inverse)
    print("Val Metrics (filtered):", metrics["pessimistic"])
    return metrics


def init_model(args, entity2id, relation2id, init_function, seed_forward, seed_init):
    print(args)
    if args.model == 'TransE':
        model = TransE(
            num_entities=len(entity2id),
            num_relations=len(relation2id),
            embedding_dim=args.embedding_dim,
            seed_init=seed_init,
            seed_forward=seed_forward,
            use_inverse=args.use_inverse,
            transE_norm=args.transE_norm,
            dropout_entity=args.dropout_entity,
            dropout_relation=args.dropout_relation,
            init_function=init_function,
        )
    elif args.model == 'DistMult':
        model = DistMult(
            num_entities=len(entity2id),
            num_relations=len(relation2id),
            embedding_dim=args.embedding_dim,
            seed_init=seed_init,
            seed_forward=seed_forward,
            use_inverse=args.use_inverse,
            init_function=init_function,
            dropout_entity=args.dropout_entity,
            dropout_relation=args.dropout_relation,
        )
    elif args.model == 'ComplEx':
        model = ComplEx(
            num_entities=len(entity2id),
            num_relations=len(relation2id),
            embedding_dim=args.embedding_dim,
            seed_init=seed_init,
            seed_forward=seed_forward,
            use_inverse=args.use_inverse,
            init_function=init_function,
            dropout_entity=args.dropout_entity,
            dropout_relation=args.dropout_relation,
        )
    elif args.model == 'RotatE':
        # Get optional RotatE-specific parameters
        rotate_norm = getattr(args, 'rotate_norm', 1)
        normalize_phases = getattr(args, 'normalize_phases', True)
        model = RotatE(
            num_entities=len(entity2id),
            num_relations=len(relation2id),
            embedding_dim=args.embedding_dim,
            seed_init=seed_init,
            seed_forward=seed_forward,
            use_inverse=args.use_inverse,
            init_function=init_function,
            dropout_entity=args.dropout_entity,
            dropout_relation=args.dropout_relation,
            rotate_norm=rotate_norm,
            normalize_phases=normalize_phases,
        )
    elif args.model == 'ConvE':
        model = ConvE(
            num_entities=len(entity2id),
            num_relations=len(relation2id),
            filter_size=3,
            padding=1,  # shit, it's 0 in the original paper
            stride=1,
            feature_map_drop=args.dropout_feature_map,
            hidden_drop=args.dropout_projection,
            embedding_dim=args.embedding_dim,
            embedding_shape1=args.convE_embedding_shape1,
            seed_init=seed_init,
            seed_forward=seed_forward,
            use_inverse=args.use_inverse,
            dropout_entity=args.dropout_entity,
            dropout_relation=args.dropout_relation,
            init_function=init_function,
        )
    elif args.model == 'Fixed':
        from kge.models import FixedModel
        model = FixedModel(
            num_entities=len(entity2id),
            num_relations=len(relation2id),
            embedding_dim=args.embedding_dim,
            seed_forward=seed_forward,
            seed_init=seed_init,
            use_inverse=args.use_inverse,
            dropout_entity=args.dropout_entity,
            dropout_relation=args.dropout_relation,
            init_function=init_function,
        )
    elif args.model == 'RGCN':
        edge_index, edge_type = load_torch_geometric_data(args.data_dir, entity2id, relation2id)
        # Get use_batched_encoding from args if available, otherwise default to True
        use_batched_encoding = getattr(args, 'use_batched_encoding_RGCN', True)
        use_batched_encoding = False
        print("use_batched_encoding: ", use_batched_encoding)
        encoder_batch_size = getattr(args, 'encoder_batch_size_RGCN', 1024)
        model = RGCN(
            num_entities=len(entity2id),
            num_relations=len(relation2id),
            embedding_dim=args.embedding_dim,
            edge_index=edge_index,
            edge_type=edge_type,
            seed_forward=seed_forward,
            seed_init=seed_init,
            use_inverse=args.use_inverse,
            dropout_entity=args.dropout_entity,
            dropout_relation=args.dropout_relation,
            init_function=init_function,
            use_batched_encoding=use_batched_encoding,
            encoder_batch_size=encoder_batch_size
        )
    elif args.model == 'Transformer':
        model = Transformer(
            num_entities=len(entity2id),
            num_relations=len(relation2id),
            embedding_dim=args.embedding_dim,
            seed_forward=seed_forward,
            seed_init=seed_init,
            use_inverse=args.use_inverse,
            dropout_entity=args.dropout_entity,
            dropout_relation=args.dropout_relation,
            init_function=init_function
        )
    else:
        raise ValueError(f"Unknown model type: {args.model}")
    return model


def train_and_eval(
    model,
    train_triples,
    valid_triples,
    test_triples,
    entity2id,
    relation2id,
    run_dir,
    data_dir,
    seed_neg,
    seed_order,
    seed_forward,
    seed_init,
    batch_size,
    lr,
    max_epochs,
    eval_every,
    early_stop_metric,
    early_stop_patience,
    early_stop_delta,
    embedding_dim,
    model_name,
    num_neg_h,
    num_neg_t,
    loss_type,
    margin,
    reg_type,
    reg_entity_weight,
    reg_relation_weight,
    use_inverse,
    log_to_wandb,
    device,
    wandb_run=None,
    config=None,
    max_hours=float('inf'),
    resume_checkpoint=False
):
    
    # Charger les dictionnaires d'entités et de relations
    entity2id, relation2id = load_entity_relation_dicts(data_dir)
    
    # Convertir les triplets en tenseurs d'IDs
    # train_triples = convert_to_id_arrays(train_triples, entity2id, relation2id)
    # valid_triples = convert_to_id_arrays(valid_triples, entity2id, relation2id) if valid_triples is not None else None
    # test_triples = convert_to_id_arrays(test_triples, entity2id, relation2id) if test_triples is not None else None
    
    # Concaténer tous les triplets pour les statistiques
    all_h = np.concatenate([train_triples[0], valid_triples[0], test_triples[0]], axis=0)
    all_r = np.concatenate([train_triples[1], valid_triples[1], test_triples[1]], axis=0)
    all_t = np.concatenate([train_triples[2], valid_triples[2], test_triples[2]], axis=0)  
    all_triples = (all_h, all_r, all_t)

    print("Training model started on device: ", device)

    model, history = train_kge(
        model=model,
        train_triples=train_triples,
        valid_triples=valid_triples,
        entity2id=entity2id,
        relation2id=relation2id,
        seed_neg=seed_neg,
        seed_order=seed_order,
        seed_forward=seed_forward,
        seed_init=seed_init,
        batch_size=batch_size,
        lr=lr,
        max_epochs=max_epochs,
        eval_every=eval_every,
        early_stop_metric=early_stop_metric,
        early_stop_patience=early_stop_patience,
        early_stop_delta=early_stop_delta,
        num_neg_h=num_neg_h,
        num_neg_t=num_neg_t,
        evaluator=lambda m, v, e, r: val_evaluator(m, v, all_triples, e, r, use_inverse=use_inverse),
        loss_type=loss_type,
        margin=margin,
        reg_type=reg_type,
        reg_entity_weight=reg_entity_weight,
        reg_relation_weight=reg_relation_weight,
        log_to_wandb=log_to_wandb,
        wandb_run=wandb_run,
        max_hours=max_hours,
        resume_checkpoint=resume_checkpoint,
        checkpoint_dir=run_dir
    )
    metrics, preds = evaluate(model, test_triples, all_triples, use_inverse=use_inverse, return_preds=True)
    print("Test Metrics (filtered, pessimistic):", metrics["pessimistic"])
    print("Test Metrics (filtered, realistic):", metrics["realistic"])

    os.makedirs(run_dir, exist_ok=True)
    save_model(model, run_dir, metrics, history, config=config, preds=preds)
    if wandb_run:
        wandb_run.log({"test_metrics": metrics})
    return metrics


def save_model(model, run_dir, metrics, history, config, preds=None):
    print("Saving model to", run_dir)
    os.makedirs(run_dir, exist_ok=True)
    
    # Save model weights
    torch.save(model.state_dict(), os.path.join(run_dir, "model.pth"))
    
    # Save config
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f)
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f)
    
    # Save training history
    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump(history, f)
    
    if preds:
        with open(os.path.join(run_dir, "preds.json"), "w") as f:
            json.dump(preds, f, indent=2)
