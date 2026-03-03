from ast import arg, parse
import os

from torch.nn.modules.loss import F
from kge.data import load_triples, load_entity_relation_dicts, convert_to_id_arrays
from kge.models import TransE, DistMult, ComplEx, RotatE, ConvE
from kge.train import train_kge
from kge.utils import load_model, check_gpu_availability
import torch
import argparse
import wandb
from stability import stability_experiment
from training_utils import init_model, train_and_eval
from sweep_utils import SOURCE_PREFIX




def parse_args():
    parser = argparse.ArgumentParser(description='Train a knowledge graph embedding model with full seed control and batch sweep modes.')

    # Model
    parser.add_argument('--model', type=str, default='RotatE', choices=['TransE', 'DistMult', 'ComplEx', 'RotatE', 'ConvE', 'RGCN', 'Transformer'])
    parser.add_argument('--data_dir', type=str, default='data/kinship', help='Path to the dataset directory')

    # Hyperparameters
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--embedding_dim', type=int, default=128, help='Embedding size (must be a perfect square for ConvE)')
    parser.add_argument('--use_inverse', action=argparse.BooleanOptionalAction, default=True, help='Use inverse relations (required for ConvE)')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--init_function', type=str, default='xavier_normal', choices=['xavier_normal', 'xavier_uniform', 'kaiming_normal', 'kaiming_uniform', 'normal', 'uniform'], help='Max epochs for the training')
    parser.add_argument('--num_neg_h', type=int, default=10, help='Number of head negatives')
    parser.add_argument('--num_neg_t', type=int, default=10, help='Number of tail negatives')

    parser.add_argument('--transE_norm', type=int, default=2, choices=[1,2], help='L_p norm for TransE (1 or 2)')
    parser.add_argument('--rotate_norm', type=int, default=1, choices=[1,2], help='L_p norm for RotatE (1 or 2)') # todo : merge with TransE norm
    parser.add_argument('--normalize_phases', action=argparse.BooleanOptionalAction, default=True, help='Normalize relation phases in RotatE to [-pi, pi]')
    parser.add_argument('--convE_embedding_shape1', type=int, default=16, help='The first dimension of the reshaped 2D embedding. The second dimension is infered')
    parser.add_argument('--use_batched_encoding_RGCN', action=argparse.BooleanOptionalAction, default=True, help='Use batched encoding for RGCN (default: True)')
    parser.add_argument('--encoder_batch_size_RGCN', type=int, default=256, help='Batch size for encoding entities in RGCN eval mode (default: 1024)')

    parser.add_argument('--loss_type', type=str, default='ce', choices=['bce', 'margin', 'ce'], help='Loss function: Binary Cross Entropy, Margin Ranking, or Cross Entropy')
    parser.add_argument('--margin', type=float, default=1.0, help='Margin for margin ranking loss')

    # Regularization
    parser.add_argument('--reg_type', type=str, default='none', choices=['none', 'l1', 'l2', 'l3'], help='Regularization type (L1, L2, L3, None)')
    parser.add_argument('--reg_entity_weight', type=float, default=0.0001, help='Entity embedding regularization weight')
    parser.add_argument('--reg_relation_weight', type=float, default=0.0001, help='Relation embedding regularization weight')
    parser.add_argument('--dropout_entity', type=float, default=0.2, help='Dropout on entity embeddings')
    parser.add_argument('--dropout_relation', type=float, default=0.2, help='Dropout on relation embeddings')
    parser.add_argument('--dropout_feature_map', type=float, default=0.2, help='Feature map dropout (ConvE only)')
    parser.add_argument('--dropout_projection', type=float, default=0.3, help='Projection dropout (ConvE only)')

    # Seeds
    parser.add_argument('--seed_init', type=int, default=42)
    parser.add_argument('--seed_neg', type=int, default=42)
    parser.add_argument('--seed_order', type=int, default=42)
    parser.add_argument('--seed_forward', type=int, default=42)    
    parser.add_argument('--seed_varying', type=str, default=None, choices=['seed_init', 'seed_neg', 'seed_order', 'seed_forward', 'all', "hardware", None])

    # Training
    parser.add_argument('--max_epochs', type=int, default=500)
    parser.add_argument('--eval_every', type=int, default=5)
    parser.add_argument('--early_stop_metric', type=str, default='MRR')
    parser.add_argument('--early_stop_patience', type=int, default=50)
    parser.add_argument('--early_stop_delta', type=float, default=1e-3)
    parser.add_argument('--max_hours', type=float, default=float('inf'), help='Max hours for the training')

    parser.add_argument('--use_gpu', action='store_true', help='Use GPU for training if available')
    parser.add_argument('--GPU_reproductibility', action='store_true', help='If true, then the training will be submitted as besteffort OAR jobs')
    parser.add_argument('--resume_checkpoint', action='store_true', help='Resume training from the last checkpoint')
    parser.add_argument('--oar', action='store_true', help='If true, then the training will be submitted as OAR jobs')
    parser.add_argument('--oar_besteffort', action='store_true', help='If true, then the training will be submitted as besteffort OAR jobs')
    parser.add_argument('--run_dir', type=str, default=None, help='Run directory')

    # Wandb
    parser.add_argument('--init_sweep', action='store_true', help='Initialise and print sweep id for wandb')
    parser.add_argument('--sweep_id', type=str, default=None)
    parser.add_argument('--log_to_wandb', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--wandb_project', type=str, default=None, help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='Wandb run name')


    # Functionnalities
    parser.add_argument('--stability_training', action='store_true', help='Start training the model with stability experiment')
    parser.add_argument('--visualise', action='store_true', help='Générer les visualisations de stabilité.')
    parser.add_argument('--stability_measures', action='store_true', help='If true, then the training will be submitted as besteffort OAR jobs')
    parser.add_argument('--GPU_stability_experiment', action='store_true', help='If true, then the training will be submitted as besteffort OAR jobs')
    parser.add_argument('--GPU_stability_measures', action='store_true', help='If true, then the training will be submitted as besteffort OAR jobs')
    parser.add_argument('--ablation_norm', action='store_true', help='Run embedding norm ablation study')
    parser.add_argument('--ablation_voting', action='store_true', help='Run voting/aggregation ablation study')
    parser.add_argument('--model_families', type=str, nargs='+', default=None, choices=['TransE', 'DistMult', 'ComplEx', 'RotatE', 'ConvE', 'RGCN', 'Transformer'], help='Model families to analyze in ablation study')
    
    # Meta-model arguments
    parser.add_argument('--use_meta_model', action='store_true', help='Use meta-model for aggregating multiple trained models')
    parser.add_argument('--use_intermodel', action='store_true', help='Allow meta-model to use different model types (TransE, ConvE, etc.)')
    # parser.add_argument('--model_types', type=str, nargs='+', default=['ConvE'], choices=['TransE', 'DistMult', 'ComplEx', 'RotatE', 'ConvE', 'RGCN', 'Transformer'], help='List of model types to use for intermodel (e.g., TransE DistMult ConvE). If not specified with use_intermodel, all types will be used.')
    parser.add_argument('--model_types', type=str, nargs='+', default=['ConvE'], help='List of model types to use for intermodel (e.g., TransE DistMult ConvE). If not specified with use_intermodel, all types will be used.')
    parser.add_argument('--model_agreg', type=str, default='softmax_mean', choices=['majority', 'mean', 'weighted_mean', 'softmax_mean', 'geometric_mean', 'harmonic_mean', 'winner_take_all', 'all'], help='Aggregation method for meta-model')
    parser.add_argument('--score_norm', type=str, default='minmax', choices=['zscore', 'minmax', 'none', 'all'], help='Score normalization method')
    parser.add_argument('--competency_measure', type=str, default='MRR', choices=['MRR', 'normalised_MR', 'inverse_MR', 'Hits@1', 'Hits@10', 'median_rank', 'all'], help='Competency measure for model weighting')
    # parser.add_argument('--similarity_methods', type=str, nargs='+', default=['neighbor'], choices=['neighbor', 'embedding', 'jaccard', 'types'], help='Methods for finding similar triples')
    parser.add_argument('--similarity_methods', type=str, nargs='+', default=['neighbor'], help='Methods for finding similar triples')
    parser.add_argument('--keep_top_k_triples', type=int, default=10, help='Number of top similar triples to keep for competancy evaluation')
    parser.add_argument('--relation_weight_multiplier', type=float, default=10.0, help='Multiplier for same-relation triples in similarity scoring')
    parser.add_argument('--normalize_before_filtering', action='store_true', help='If set, normalize scores before filtering (includes train/val entities in normalization). Default: normalize after filtering.')
    parser.add_argument('--similar_triples_source', type=str, default='val', choices=['train', 'val'], help='Source for similar triples search: train or val set (default: val)')
    parser.add_argument('--stability_base_dir', type=str, default='24_09/stability', help='Base directory for stability runs')
    parser.add_argument('--model_pattern', type=str, default='best_all*', help='Pattern to filter model directories')
    parser.add_argument('--meta_model_grid', action='store_true', help='Perform exhaustive grid search over all meta-model configurations')
    parser.add_argument('--multi_grid', action='store_true', help='Launch multiple grid search jobs via OAR for all model/similarity combinations')
    
    return parser.parse_args()


def main(args):
    # Vérifier et configurer le GPU si demandé
    device = 'cpu'    
    if args.use_gpu:
        if check_gpu_availability():
            device = 'cuda'
            print("GPU available, using CUDA")

        if args.GPU_reproductibility:
            print("GPU reproductibility enabled")
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


    data_dir = args.data_dir
    entity2id, relation2id = load_entity_relation_dicts(data_dir)

    # Load triples as strings first
    train_triples_str = load_triples(os.path.join(data_dir, "train.txt"))
    valid_triples_str = load_triples(os.path.join(data_dir, "valid.txt"))
    test_triples_str = load_triples(os.path.join(data_dir, "test.txt"))

    # Convert to ID arrays (NumPy)
    train_triples = convert_to_id_arrays(train_triples_str, entity2id, relation2id, use_inverse=args.use_inverse, split="train")
    valid_triples = convert_to_id_arrays(valid_triples_str, entity2id, relation2id, use_inverse=args.use_inverse, split="valid")
    test_triples = convert_to_id_arrays(test_triples_str, entity2id, relation2id, use_inverse=args.use_inverse, split="test")
    if args.use_inverse:
        # Add inverse relations
        num_rel = len(relation2id)
        for rel, idx in list(relation2id.items()):
            inv_rel = rel + '_inv'
            relation2id[inv_rel] = idx + num_rel

    model = init_model(args, entity2id, relation2id, args.init_function, args.seed_forward, args.seed_init)
    model = model.to(device)  # Déplacer le modèle sur le bon appareil


    if args.run_dir is None:
        run_dir = f"runs_{SOURCE_PREFIX}/{data_dir.split('/')[-1]}_{args.model}_{args.embedding_dim}_{args.seed_init}_{args.seed_neg}_{args.seed_order}_{args.max_epochs}"
    else:
        run_dir = args.run_dir  
    if args.log_to_wandb:
        try:
            wandb.init(
                project=args.wandb_project if args.wandb_project else f"{SOURCE_PREFIX}_{args.model}_{data_dir.split('/')[-1]}",
            name=args.wandb_run_name,
                config=vars(args)
            )
        except Exception as e:
            print(f"[WARN] wandb.init failed ({e}). Proceeding without wandb logging.")
            args.log_to_wandb = False
    train_and_eval(
        model=model,
        train_triples=train_triples,
        valid_triples=valid_triples,
        test_triples=test_triples,
        entity2id=entity2id,
        relation2id=relation2id,
        run_dir=run_dir,
        data_dir=args.data_dir,
        seed_neg=args.seed_neg,
        seed_order=args.seed_order,
        seed_init=args.seed_init,
        seed_forward=args.seed_forward,
        batch_size=args.batch_size,
        lr=args.lr,
        max_epochs=args.max_epochs,
        eval_every=args.eval_every,
        early_stop_metric=args.early_stop_metric,
        early_stop_patience=args.early_stop_patience,
        early_stop_delta=args.early_stop_delta,
        embedding_dim=args.embedding_dim,
        model_name=args.model,
        num_neg_h=args.num_neg_h,
        num_neg_t=args.num_neg_t,
        loss_type=args.loss_type,
        margin=args.margin,
        reg_type=args.reg_type,
        reg_entity_weight=args.reg_entity_weight,
        reg_relation_weight=args.reg_relation_weight,
        use_inverse=args.use_inverse,
        log_to_wandb=args.log_to_wandb,
        device=device,
        wandb_run=wandb.run if args.log_to_wandb else None,
        config=args.__dict__,
        max_hours=args.max_hours,
        resume_checkpoint=args.resume_checkpoint
    )
    print("GG well done")
    


if __name__ == "__main__":
    args = parse_args()
    print(args)

    if args.use_meta_model:
        from meta_model_main import run_meta_model
        print("Launching meta-model evaluation...")
        run_meta_model(args)
    elif args.ablation_norm:
        from stability_measures.ablation_norm import run_ablation_norm
        print("Launching embedding norm ablation study...")
        run_ablation_norm(args)
    elif args.ablation_voting:
        from stability_measures.ablation_voting import run_ablation_voting
        print("Launching voting/aggregation ablation study...")
        run_ablation_voting(args)
    elif args.visualise:
        from stability_measures import stability_visualisation
        print("Lunch stability visualisation...")
        stability_visualisation(args)
    elif args.stability_measures or args.GPU_stability_measures:
        from stability_measures import stability_all_measures
        if args.GPU_stability_measures:
            stability_type = "GPU"
        else:
            stability_type = "stability"
        print(f"Lunch stability measures {stability_type}...")
        stability_all_measures(args)
    elif args.stability_training or args.GPU_stability_experiment:
        from stability import stability_experiment
        stability_type = "GPU" if args.GPU_stability_experiment else "stability"
        print(f"Lunch stability training {stability_type}...")
        stability_experiment(args)
    elif args.init_sweep:
        from sweep_utils import init_sweep
        init_sweep(args)
    elif args.sweep_id is not None:
        from sweep_utils import run_sweep
        run_sweep(args.sweep_id, args)
    elif args.stability_measures:
        from stability_measures import stability_all_measures
        stability_all_measures(args)
    elif args.multi_grid:
        print("Launching multi-grid experiments via OAR...")
        print("This will submit multiple grid search jobs for all model/similarity combinations.")
        print("Please use the multi_grid_launcher.py script directly for more control.")
        print("\nUsage: python3 multi_grid_launcher.py [OPTIONS]")
        print("  --dry_run              : Print commands without executing")
        print("  --besteffort           : Submit as besteffort jobs")
        print("  --individual_only      : Only individual model jobs")
        print("  --intermodel_only      : Only intermodel jobs")
        print("  --models MODEL [MODEL ...] : Specific models to run")
        print("  --similarity_methods METHOD [METHOD ...] : Specific similarity methods")
        print("\nOr use the wrapper: ./multi_grid_lunch.sh [OPTIONS]")
        import sys
        sys.exit(0)
    else:
        print("Lunch main training...")
        main(args)