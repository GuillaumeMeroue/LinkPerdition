import wandb
import torch
from collections import defaultdict
import json
from kge.data import load_triples, load_entity_relation_dicts, convert_to_id_arrays
from kge.train import train_kge
from kge.eval import evaluate
import torch
from training_utils import init_model
import os
from training_utils import val_evaluator
import numpy as np
from kge.utils import check_gpu_availability

SOURCE_PREFIX = "133_10"
# SOURCE_PREFIX = "27_10"
# SOURCE_PREFIX = "30_10"
MINI_GRID = True

def get_test_mrr(run):
    try:
        history = list(run.scan_history(keys=["test_metrics.realistic.MRR"]))
        return history[-1].get("test_metrics.realistic.MRR")
    except Exception as e:
        print(f"Erreur lors de la récupération du MRR max pour le run {run.id}: {e}")
        return 0.0


def get_diverses_configs_from_latest_sweep(project):
    api = wandb.Api()
    runs = api.runs(f"{project}")
    print("project:", project)
    print("runs:", runs)

    # Group runs by sweep_id
    sweeps_dict = defaultdict(list)
    for run in runs:
        if run.sweep is not None:
            sweeps_dict[run.sweep.id].append(run)

    if not sweeps_dict:
        raise ValueError("Aucun sweep trouvé dans ce projet.")

    # Find the latest sweep
    sweep_id_to_latest_time = {
        sweep_id: max(run.created_at for run in sweep_runs)
        for sweep_id, sweep_runs in sweeps_dict.items()
    }
    latest_sweep_id = max(sweep_id_to_latest_time.items(), key=lambda x: x[1])[0]

    # Get runs from the latest sweep
    sweep_runs = sweeps_dict[latest_sweep_id]

    # Extract MRR and configs, filter out runs with max MRR < 5%
    results = []
    for run in sweep_runs:
        try:
            mrr = get_test_mrr(run)
            if mrr >= 0.05:  # Only keep runs with MRR >= 5%
                results.append((mrr, run.config))
        except KeyError:
            continue

    if not results:
        raise ValueError("Aucune exécution avec MRR >= 5% trouvée.")

    results.sort(key=lambda x: x[0], reverse=True)

    best = results[0][1]
    worst = results[-1][1]
    median = results[len(results) // 2][1]

    return best, median, worst

def init_sweep(args):
    if MINI_GRID:
        sweep_config = {
            'method': 'grid',
            'parameters': {
                'embedding_dim': {'values': [128, 256, 512]},
                'lr': {'values': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]},
                # 'embedding_dim': {'values': [128, 256]},
                # 'lr': {'values': [1e-5, 1e-3, 1e-1]},


                'dropout_entity': {'value': 0.2},
                'dropout_relation': {'value': 0.2},
                'batch_size': {'value': 256},
                'model': {'value': args.model},
                'data_dir': {'value': args.data_dir},
                'seed_neg': {'value': 42},
                'seed_order': {'value': 42},
                'seed_init': {'value': 42},
                'seed_forward': {'value': 42},
                'loss_type': {'value': 'ce'},



                'use_gpu': {'value': True},
                'GPU_reproductibility': {'value': True}
            }
        }
    
    else:
        sweep_config = {
            'method': 'random',
            'parameters': {
            'embedding_dim': {'values': [128, 256, 512]},
            'lr': {'distribution': 'log_uniform_values', 'min': 1e-5, 'max': 1},
            # 'entity_dropout': {'min': 0.0, 'max': 0.5},
            # 'relation_dropout': {'min': 0.0, 'max': 0.5},
            'dropout_entity': {'min': 0.0, 'max': 0.5},
            'dropout_relation': {'min': 0.0, 'max': 0.5},
            # 'loss_type': {'values': ['bce','ce']},
            'model': {'value': args.model},
            'data_dir': {'value': args.data_dir},
            'seed_neg': {'value': 42},
            'seed_order': {'value': 42},
            'seed_init': {'value': 42},
            'seed_forward': {'value': 42},
            'batch_size': {'value': 256},
            'loss_type': {'value': 'ce'},



            'use_gpu': {'value': True},
            'GPU_reproductibility': {'value': True}
            }
        }

    #WARNING : ça a été lancé avec 10 num_neg pour kinship et nations et 500 pour les autres

    if args.data_dir in ["kinship", "nations"]:
        if MINI_GRID:
            sweep_config['parameters']['num_neg'] = {'value': 10}
        else:
            sweep_config['parameters']['num_neg'] = {'distribution': 'q_log_uniform_values', 'q': 1, 'min': 1, 'max': 10}
    else:
        if MINI_GRID:
            sweep_config['parameters']['num_neg'] = {'value': 500}
        else:
            sweep_config['parameters']['num_neg'] = {'distribution': 'q_log_uniform_values', 'q': 1, 'min': 1, 'max': 1000}

    if args.model == "TransE":
        sweep_config['parameters']['transE_norm'] = {'value': 2} 
        if MINI_GRID:
            sweep_config['parameters']['use_inverse'] = {'value': True}
        else:
            sweep_config['parameters']['use_inverse'] = {'values': [True, False]}


    if args.model == "DistMult":
        if MINI_GRID:
            sweep_config['parameters']['use_inverse'] = {'value': True}
        else:
            sweep_config['parameters']['use_inverse'] = {'values': [True, False]}

    if args.model == "ComplEx":
        if MINI_GRID:
            sweep_config['parameters']['use_inverse'] = {'value': True}
        else:
            sweep_config['parameters']['use_inverse'] = {'values': [True, False]}

    if args.model == "RotatE":
        if MINI_GRID:
            sweep_config['parameters']['use_inverse'] = {'value': True}
            sweep_config['parameters']['rotate_norm'] = {'value': 1}
            sweep_config['parameters']['normalize_phases'] = {'value': True}
        else:
            sweep_config['parameters']['use_inverse'] = {'values': [True, False]}
            sweep_config['parameters']['rotate_norm'] = {'values': [1, 2]}
            sweep_config['parameters']['normalize_phases'] = {'values': [True, False]}

    if args.model == "ConvE":
        sweep_config['parameters']['use_inverse'] = {'value': True}
        # sweep_config['parameters']['feature_map_drop'] = {'min': 0.0, 'max': 0.5} command used for the run #todo delete
        # sweep_config['parameters']['hidden_drop'] = {'min': 0.0, 'max': 0.5}
        if MINI_GRID:
            sweep_config['parameters']['dropout_projection'] = {'value': 0.3} # Default in LibKGE (and original paper)
            sweep_config['parameters']['dropout_feature_map'] = {'value': 0.2} 
        else:
            sweep_config['parameters']['dropout_projection'] = {'min': 0.0, 'max': 0.5}
            sweep_config['parameters']['dropout_feature_map'] = {'min': 0.0, 'max': 0.5}

    if args.model == "Transformer":
        sweep_config['parameters']['use_inverse'] = {'value': True}
    
    if args.model == "RGCN":
        if MINI_GRID:
            sweep_config['parameters']['use_inverse'] = {'value': True}
            # sweep_config['parameters']['batch_size'] = {'value': 4096}
            # sweep_config['parameters']['use_batched_encoding_RGCN'] = {'value': True}
            
        else:
            sweep_config['parameters']['use_inverse'] = {'values': [True, False]}

    # Add args not in sweep_config
    for element in args.__dict__.items():
        if element[0] not in sweep_config['parameters'] and element[0] != "max_hours":
            sweep_config['parameters'][element[0]] = {'value': element[1]}
    project = f'{SOURCE_PREFIX}_{args.data_dir.split("/")[-1]}_{args.model}'
    sweep_id = wandb.sweep(sweep_config, project=project)
    print(f"Sweep initialized. Sweep ID: {sweep_id}")
    return sweep_id

def run_sweep(sweep_id, args):
    def sweep_train():
        # Initialiser wandb pour ce run
        wandb.init()
        # Charger les hyperparamètres sweep
        config = wandb.config
        # Charger les données et dictionnaires
        data_dir = config.data_dir
        
        device = 'cpu'
        if config.use_gpu and check_gpu_availability():
            device = 'cuda'

        print("Device:", device)
        
        data_dir = config.data_dir
        entity2id, relation2id = load_entity_relation_dicts(data_dir)

        # Load triples as strings first
        train_triples_str = load_triples(os.path.join(data_dir, "train.txt"))
        valid_triples_str = load_triples(os.path.join(data_dir, "valid.txt"))
        test_triples_str = load_triples(os.path.join(data_dir, "test.txt"))

        # Convert to ID arrays (NumPy)
        train_triples = convert_to_id_arrays(train_triples_str, entity2id, relation2id, use_inverse=config.use_inverse, split="train")
        valid_triples = convert_to_id_arrays(valid_triples_str, entity2id, relation2id, use_inverse=config.use_inverse, split="valid")
        test_triples = convert_to_id_arrays(test_triples_str, entity2id, relation2id, use_inverse=config.use_inverse, split="test")

        all_h = np.concatenate([train_triples[0], valid_triples[0], test_triples[0]], axis=0)
        all_r = np.concatenate([train_triples[1], valid_triples[1], test_triples[1]], axis=0)
        all_t = np.concatenate([train_triples[2], valid_triples[2], test_triples[2]], axis=0)  
        all_triples = (all_h, all_r, all_t)

        if config.use_inverse:
            # Add inverse relations
            num_rel = len(relation2id)
            for rel, idx in list(relation2id.items()):
                inv_rel = rel + '_inv'
                relation2id[inv_rel] = idx + num_rel
        
        # Créer le modèle
        model = init_model(config, entity2id, relation2id, config.init_function, config.seed_forward, config.seed_init).to(device)

        if hasattr(config, 'num_neg'):
            num_neg_h = config.num_neg
            num_neg_t = config.num_neg
        else:
            num_neg_h = config.num_neg_h
            num_neg_t = config.num_neg_t

        
        # Appel cohérent avec train_kge
        run_dir = f"runs/{data_dir.split('/')[-1]}_{config.model}_{config.embedding_dim}_{config.lr}_{config.seed_init}_{config.seed_neg}_{config.seed_order}"
        run_name = f"{config.embedding_dim}_{config.lr}_{config.seed_init}_{config.seed_neg}_{config.seed_order}"
        wandb.run.name = run_name
        
        # Check if a checkpoint exists for this run and enable resume if so
        checkpoint_path = os.path.join(run_dir, 'checkpoint.pth')
        resume_from_checkpoint = args.resume_checkpoint
        
        if resume_from_checkpoint:
            print(f"Found existing checkpoint for sweep run: {checkpoint_path}")
            print("Resuming training from checkpoint...")
        else:
            print(f"No checkpoint found for sweep run, starting from scratch...")
        
        model, history = train_kge(
            model,
            train_triples,
            valid_triples,
            entity2id,
            relation2id,
            seed_neg=config.seed_neg,
            seed_order=config.seed_order,
            seed_init=config.seed_init,
            seed_forward=config.seed_forward,
            batch_size=config.batch_size,
            lr=config.lr,
            max_epochs=config.max_epochs,
            eval_every=config.eval_every,
            early_stop_metric=config.early_stop_metric,
            early_stop_patience=config.early_stop_patience,
            early_stop_delta=config.early_stop_delta,
            num_neg_h=num_neg_h,
            num_neg_t=num_neg_t,
            evaluator=lambda m, v, e, r: val_evaluator(m, v, all_triples, e, r, use_inverse=config.use_inverse),
            loss_type=config.loss_type,
            margin=config.margin,
            reg_type=config.reg_type,
            reg_entity_weight=config.reg_entity_weight,
            reg_relation_weight=config.reg_relation_weight,
            log_to_wandb=True,
            wandb_run=wandb.run,
            resume_checkpoint=resume_from_checkpoint,
            checkpoint_dir=run_dir
        )
        # Évaluation finale
        metrics = evaluate(model, test_triples, all_triples, use_inverse=config.use_inverse)
        wandb.log({"test_metrics": metrics})
        print("Test Metrics (filtered, pessimistic):", metrics["pessimistic"])
        print("Test Metrics (filtered, realistic):", metrics["realistic"])
        # Sauvegarde des résultats

        os.makedirs(run_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(run_dir, "model.pth"))
        with open(os.path.join(run_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f)
        with open(os.path.join(run_dir, "history.json"), "w") as f:
            json.dump(history, f)
    project = f'{SOURCE_PREFIX}_{args.data_dir.split("/")[-1]}_{args.model}'
    wandb.agent(sweep_id, project=project, function=sweep_train, count=1)