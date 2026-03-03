from pickletools import read_bytes1
import wandb
import os
from kge.data import load_triples, load_entity_relation_dicts, convert_to_id_arrays
from kge.train import train_kge
import torch
from training_utils import init_model, train_and_eval
from sweep_utils import SOURCE_PREFIX
from argparse import Namespace
import subprocess
import time
import copy
import json

SEEDS_VALUES = [42,698,887,283,358]
SEED_TYPES = ['seed_init', 'seed_neg', 'seed_order', 'seed_forward', 'all']
QUALITIES = ['best', 'median', 'worst']
# QUALITIES = ['best']
SOURCE_PREFIX = "30_10"
PROJECT_PREFIX = "30_10"

def stability_experiment(args):
    from sweep_utils import get_diverses_configs_from_latest_sweep

    dataset_key = args.data_dir.split('/')[-1]
    project_name = f"{SOURCE_PREFIX}_{dataset_key}_{args.model}"

    print("open the project", project_name)

    # Try to load configs from the pre-saved 'all_config' file first
    # Added at the last moment for reviewer, it may be buggy
    loaded_from_file = False
    # if os.path.exists('all_config'):
    #     try:
    #         with open('all_config', 'r') as f:
    #             all_configs = json.load(f)
    #         key = f"{dataset_key}_{args.model}"
    #         if key in all_configs:
    #             cfgs = all_configs[key]
    #             best_cfg = cfgs.get('best')
    #             median_cfg = cfgs.get('median')
    #             worst_cfg = cfgs.get('worst')
    #             if best_cfg and median_cfg and worst_cfg:
    #                 print(f"[INFO] Loaded configs for {key} from 'all_config'. Skipping wandb retrieval.")
    #                 loaded_from_file = True
    #     except Exception as e:
    #         print(f"[WARN] Failed to load 'all_config': {e}. Falling back to wandb retrieval.")

    if not loaded_from_file:
        best_cfg, median_cfg, worst_cfg = get_diverses_configs_from_latest_sweep(project=project_name)
    best_cfg['oar'] = args.oar
    median_cfg['oar'] = args.oar
    worst_cfg['oar'] = args.oar
    best_cfg['oar_besteffort'] = args.oar_besteffort
    median_cfg['oar_besteffort'] = args.oar_besteffort
    worst_cfg['oar_besteffort'] = args.oar_besteffort
    best_cfg['use_gpu'] = args.use_gpu
    median_cfg['use_gpu'] = args.use_gpu
    worst_cfg['use_gpu'] = args.use_gpu
    best_cfg['GPU_reproductibility'] = args.GPU_reproductibility
    median_cfg['GPU_reproductibility'] = args.GPU_reproductibility
    worst_cfg['GPU_reproductibility'] = args.GPU_reproductibility
    best_cfg = Namespace(**best_cfg)
    median_cfg = Namespace(**median_cfg)
    worst_cfg = Namespace(**worst_cfg)
    print("Best config:", best_cfg, "Median config:", median_cfg, "Worst config:", worst_cfg)
    if args.stability_training: 
        stability_training(best_cfg, "best", do_not_relunch=True, GPU_stability_experiment=False)
        stability_training(median_cfg, "median", do_not_relunch=True, GPU_stability_experiment=False) # only for all_seeds
        stability_training(worst_cfg, "worst", do_not_relunch=True, GPU_stability_experiment=False) # only for all_seeds
    elif args.GPU_stability_experiment:
        stability_training(best_cfg, "best", do_not_relunch=False, GPU_stability_experiment=True)
    else:
        raise ValueError("Unknown experiment type")
        

def stability_training(args, name, do_not_relunch=True, GPU_stability_experiment=False):
    if GPU_stability_experiment:
        global SEEDS_VALUES, QUALITIES
        SEEDS_VALUES = SEEDS_VALUES[:2]
        QUALITIES = ["best"]
    print("Stability training for", name, "with args:", args)
    data_dir = args.data_dir
    if GPU_stability_experiment:
        wandb_project = f"Stability_GPU_{PROJECT_PREFIX}_{data_dir.split('/')[-1]}_{args.model}"
    else:
        wandb_project = f"Stability_{PROJECT_PREFIX}_{data_dir.split('/')[-1]}_{args.model}"
    procs = []

    # Si relunch, on récupère la liste des runs existantes dans wandb
    run_status_dict = dict()
    if do_not_relunch:
        try:
            api = wandb.Api()
            runs = api.runs(wandb_project)

            # Trier tous les runs par date de création (du plus ancien au plus récent)
            sorted_runs = sorted(runs, key=lambda r: r.created_at)
            print("Sorted runs:", sorted_runs)

            # Pour chaque run, on garde le dernier state pour un nom donné
            run_status_dict = dict()
            for run in sorted_runs:
                run_status_dict[run.name] = run.state  # écrase les anciens

        except Exception as e:
            print(f"[WARN] Impossible de récupérer les runs wandb: {e}")
            run_status_dict = dict()


    def run_training_instance(seed_type, seed_order, seed_neg, seed_init, seed_forward, max_epochs_value, run_name_prefix, GPU_reproductibility):
        if seed_type in ["seed_init", "all", "hardware"]:
            seed_value = seed_init
        elif seed_type == "seed_neg":
            seed_value = seed_neg
        elif seed_type == "seed_forward":
            seed_value = seed_forward
        else:
            seed_value = seed_order

        run_name = f"{run_name_prefix}_{seed_type}_{seed_value}_{max_epochs_value}_reprod_{GPU_reproductibility}"
        
        # Déterminer si on doit utiliser resume_checkpoint
        use_resume_checkpoint = False
        if do_not_relunch and run_name in run_status_dict:
            if run_status_dict[run_name] in ["finished", "running"]:
                print(f"[INFO] Run {run_name} déjà existante avec état {run_status_dict[run_name]}, on skip.")
                return
            elif run_status_dict[run_name] in ["failed", "crashed"]:
                print(f"[INFO] Run {run_name} existe mais a échoué ({run_status_dict[run_name]}), on reprend avec checkpoint.")
                use_resume_checkpoint = True
            else:
                print(f"[INFO] Run {run_name} existe avec état {run_status_dict[run_name]}, on reprend avec checkpoint.")
                use_resume_checkpoint = True
        else:
            print(f"[INFO] Run {run_name} n'existe pas, on lance sans checkpoint.")

        data_dir = args.data_dir
        if not args.oar:
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
            print("Relation2id:", relation2id)

            if args.GPU_reproductibility:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                torch.use_deterministic_algorithms(True)
                os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            else:
                torch.backends.cudnn.deterministic = False
                torch.backends.cudnn.benchmark = True
                torch.use_deterministic_algorithms(False)

            device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
            model = init_model(args, entity2id, relation2id, args.init_function, seed_forward, seed_init).to(device)

        procs = []
        if GPU_stability_experiment:
            run_dir = f"{PROJECT_PREFIX}/GPU_XP/{data_dir.split('/')[-1]}_{args.model}_{run_name}"
        else:
            run_dir = f"{PROJECT_PREFIX}/stability/{data_dir.split('/')[-1]}_{args.model}_{run_name}"
        args_copy = copy.copy(args)
        args_copy.max_epochs = max_epochs_value
        args_copy.seed_forward = seed_forward
        args_copy.seed_init = seed_init
        args_copy.seed_order = seed_order
        args_copy.seed_neg = seed_neg
        args_copy.seed_varying = seed_type
        args_copy.GPU_reproductibility = GPU_reproductibility

        if hasattr(args, 'num_neg'):
            num_neg_h   = args.num_neg
            num_neg_t   = args.num_neg
        else:
            num_neg_h = args.num_neg_h
            num_neg_t = args.num_neg_t
        if args.oar:
            pythoncommand = ""
            pythoncommand += f"python3 main.py --wandb_project {wandb_project} --wandb_run_name {run_name} --run_dir {run_dir} \
                    --seed_order {seed_order} --seed_neg {seed_neg} --seed_init {seed_init} --seed_forward {seed_forward} --seed_varying {seed_type}"

            for key, value in args.__dict__.items():
                # entity_dropout and relation_dropout are just here because I have used this order in wandb, but not in argparse :(, à enlever
                # init_sweep because it's in the sweep config I guess
                # stability_training : To avoid infinite loop
                # seed_order, seed_neg, seed_init, seed_varying : Because I pass them as argument
                # num_neg_h, num_neg_t : Because I pass them as argument and num_neg because it exist on sweep config but not in main args
                # hidden_drop, feature_map_drop : I used this name in wandb :(
                # seed_range : Because I don't use it anymore
                if key in ["stability_training", "init_sweep",\
                    "run_dir", "wandb_project", "wandb_run_name",\
                    "seed_order", "seed_neg", "seed_init", "seed_forward",\
                    "seed_varying", "num_neg_h", "num_neg_t", "num_neg", "max_epochs", "GPU_reproductibility", "seed_range"]:
                    continue
                if value is None:
                    continue
                if isinstance(value, bool):
                    if value:
                        pythoncommand += f" --{key}"
                else:
                    pythoncommand += f" --{key} {value}"

            pythoncommand += f" --num_neg_h {num_neg_h} --num_neg_t {num_neg_t}"
            pythoncommand += f" --max_epochs {max_epochs_value}"

            if GPU_reproductibility:
                pythoncommand += " --GPU_reproductibility"
            
            if use_resume_checkpoint:
                pythoncommand += " --resume_checkpoint"
            
            if data_dir.split('/')[-1] in ["kinship", "nations"]:
                walltime="05:00:00"
                cores=1
            else:
                walltime="48:00:00"
                cores=2

            if max_epochs_value == 0:
                walltime="00:30:00"
                cores=1

            if args.use_gpu:
                oar_ressources = f"gpu=1,walltime={walltime}"
                pythoncommand += " --use_gpu"
            else:
                oar_ressources = f"core={cores},walltime={walltime}"

            command = [
                "oarsub", "-l", oar_ressources,
                "-n", f"stability_{args.model}_{data_dir.split('/')[-1]}_{run_name}"]
            
            if run_name_prefix.endswith("_hardware_1"):
                command.append("-p")
                command.append("gpu_model='Tesla T4'")
            elif run_name_prefix.endswith("_hardware_2"):
                command.append("-p")
                command.append("gpu_model='A40'")
            elif run_name_prefix.endswith("_hardware_3"):
                command.append("-p")
                command.append("gpu_model='A100-SXM4-40GB'")
            elif run_name_prefix.endswith("_hardware_4"):
                command.append("-p")
                command.append("gpu_model='GeForce GTX 1080 Ti'")
            
            else :
                if args.model in ["ConvE", "DistMult", "TransE", "RGCN", "Transformer", "ComplEx", "RotatE"]:
                    command.append("-p")
                    command.append("gpu_model='GeForce RTX 2080 Ti'")
                    # command.append("gpu_model='Quadro RTX 8000'")
            
            if args.oar_besteffort:
                command.append("-t")
                command.append("besteffort")

            command.append(pythoncommand)
            print("Command:", command)
            proc = subprocess.Popen(command)
            procs.append(proc)
        else:
            # Try to init wandb, but bypass logging if it fails
            log_to_wandb = True
            wandb_run = None
            try:
                wandb.init(project=wandb_project, name=run_name, config=args_copy.__dict__)
                wandb_run = wandb.run
            except Exception as e:
                print(f"[WARN] wandb.init failed ({e}). Proceeding without wandb logging.")
                log_to_wandb = False

            train_and_eval(
                model,
                train_triples,
                valid_triples,
                test_triples,
                entity2id,
                relation2id,
                run_dir,
                args.data_dir,
                seed_neg=seed_neg,
                seed_order=seed_order,
                seed_init=seed_init,
                seed_forward=seed_forward,
                batch_size=args.batch_size,
                lr=args.lr,
                max_epochs=max_epochs_value,
                eval_every=args.eval_every,
                early_stop_metric=args.early_stop_metric,
                early_stop_patience=args.early_stop_patience,
                early_stop_delta=args.early_stop_delta,
                embedding_dim=args.embedding_dim,
                model_name=args.model,
                num_neg_h=num_neg_h,
                num_neg_t=num_neg_t,
                loss_type=args.loss_type,
                margin=args.margin,
                reg_type=args.reg_type,
                reg_entity_weight=args.reg_entity_weight,
                reg_relation_weight=args.reg_relation_weight,
                use_inverse=args.use_inverse,
                device=device,
                log_to_wandb=log_to_wandb,
                wandb_run=wandb_run,
                config=args_copy.__dict__,
                resume_checkpoint=use_resume_checkpoint
            )
            if log_to_wandb:
                try:
                    wandb.finish()
                except Exception as e:
                    print(f"[WARN] wandb.finish failed: {e}")

    # Standard runs
    for seed_type in SEED_TYPES:
        for seed_value in SEEDS_VALUES:
            if name == "best" or seed_type == "all" or (seed_type == "seed_init" and seed_value == SEEDS_VALUES[0]):
                print("Training", seed_type, seed_value)
                # Don't train 4 times 42_42_42
                if seed_value == SEEDS_VALUES[0] and seed_type != SEED_TYPES[0]: 
                    continue
                seed_order = seed_value if seed_type in ['seed_order', 'all'] else SEEDS_VALUES[0]
                seed_neg = seed_value if seed_type in ['seed_neg', 'all'] else SEEDS_VALUES[0]
                seed_init = seed_value if seed_type in ['seed_init', 'all'] else SEEDS_VALUES[0]
                seed_forward = seed_value if seed_type in ['seed_forward', 'all'] else SEEDS_VALUES[0]
                run_training_instance(seed_type, seed_order, seed_neg, seed_init, seed_forward, args.max_epochs, name, True)

    # 5 runs with only seed_init varying and max_epochs=0
    if name == "best":
        for seed_value in SEEDS_VALUES:
            run_training_instance('seed_init', SEEDS_VALUES[0], SEEDS_VALUES[0], seed_value, SEEDS_VALUES[0], 0, name, True)

        if GPU_stability_experiment:
            for i in range(1, 2):
                run_training_instance('all', SEEDS_VALUES[0], SEEDS_VALUES[0], SEEDS_VALUES[0], SEEDS_VALUES[0], 500, f'{name}_control_{i}', True)

            for i in range(0, 2):
                run_training_instance('all', SEEDS_VALUES[0], SEEDS_VALUES[0], SEEDS_VALUES[0], SEEDS_VALUES[0], 500, f'{name}_control_{i}_reprod', False)
        else:
            # 4 runs control avec la même seed (all_42) et même config, mais runs_dir suffixé _1 à _4
            # for i in range(1, 5):
            #     run_training_instance('all', SEEDS_VALUES[0], SEEDS_VALUES[0], SEEDS_VALUES[0], SEEDS_VALUES[0], 500, f'{name}_control_{i}', True)
            # pass
            for i in range(1, 5):
                run_training_instance('hardware', SEEDS_VALUES[0], SEEDS_VALUES[0], SEEDS_VALUES[0], SEEDS_VALUES[0], 500, f'{name}_hardware_{i}', True)
            # # 5 runs control avec la même seed (all_42) et même config, mais sans GPU reproductibilité
            # for i in range(0, 5):
            #     run_training_instance('all', SEEDS_VALUES[0], SEEDS_VALUES[0], SEEDS_VALUES[0], SEEDS_VALUES[0], 500, f'{name}_control_{i}_reprod', False)
        
