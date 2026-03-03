# stability_measures.py
"""
Main orchestrator:
- Finds run directories
- Calls stability_space and stability_prediction modules
- Saves metrics in the stability_measures directory

Enhancements:
- Also captures and saves each run's configuration from `{run_dir}/config.json`.
- Adds structured logging of infos and warnings during metric computations to a log file.
"""
import os
import json
from typing import Dict, List
import os
import json
import time
import logging
from wandb.env import MODE
from .stability_space_GPU import load_embedding_from_pth, compute_all_neighbor_space_metrics
from .stability_prediction import (
    compute_ambiguity, compute_discrepancy, compute_rank_stats,
    compute_all_prediction_metrics, compute_mrr_stats_from_runs
)
from .stability_utils import get_preds_list
from stability import QUALITIES, SEED_TYPES, SEEDS_VALUES, PROJECT_PREFIX
from kge.models import MODELS


# PROJECT_PREFIX = "24_09"
PROJECT_PREFIX = "30_10"
PROJECT_PREFIX = "25_11"

K_VALUES = [1, 5, 10]
K = K_VALUES[2]

import resource

def print_ram_usage(context=""):
    usage_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"[RAM] {context} : {usage_mb:.2f} MB used")


def stability_all_measures(args):
    """
    Main function to compute and save stability metrics.
    For each (model, quality, seed_type), computes similarity between runs with different seeds.
    """
    data_dir = args.data_dir
    data_name = data_dir.split('/')[-1]

    if args.stability_measures:
        if os.path.exists(f"{PROJECT_PREFIX}/stability"):
            experiment_dir = f"{PROJECT_PREFIX}/stability"
        else:
            experiment_dir = f"stability"  # for retrocompatibility, todo to remove

    elif args.GPU_stability_measures:
        experiment_dir = f"{PROJECT_PREFIX}/GPU_XP"
        global SEEDS_VALUES, QUALITIES
        SEEDS_VALUES = SEEDS_VALUES[:2]
        QUALITIES = QUALITIES[:1]
    else:
        raise ValueError("Unknown experiment type")

    # Prepare output directory and logging early
    if args.GPU_stability_measures:
        output_dir = f"{PROJECT_PREFIX}/stability_measures_results_gpu"
    else:
        output_dir = f"{PROJECT_PREFIX}/stability_measures_results"
    os.makedirs(output_dir, exist_ok=True)
    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # Configure logger (one file per data/model run)
    logger = logging.getLogger("stability_measures")
    logger.setLevel(logging.INFO)
    # Avoid adding multiple handlers if function is called multiple times
    if not logger.handlers:
        log_path = os.path.join(
            logs_dir, f"{data_name}_{args.model}_{int(time.time())}.log"
        )
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    # 1. Collect data in a structured way
    all_runs_data = {}
    MODELS = [args.model]
    for model in MODELS:
        all_runs_data[model] = {}
        for quality in QUALITIES:
            all_runs_data[model][quality] = {}
            for seed_type in SEED_TYPES:
                all_runs_data[model][quality][seed_type] = {}
                for epoch in [500,0]:
                    if epoch == 0 and seed_type != "seed_init":
                        continue
                    runs = []
                    for seed_value in SEEDS_VALUES:
                        if seed_value == SEEDS_VALUES[0] and seed_type != SEED_TYPES[0]: 
                            run_dir = f"{experiment_dir}/{data_name}_{model}_{quality}_{SEED_TYPES[0]}_{seed_value}_{epoch}_reprod_True"
                        else:
                            run_dir = f"{experiment_dir}/{data_name}_{model}_{quality}_{seed_type}_{seed_value}_{epoch}_reprod_True"
                        if not os.path.exists(run_dir):
                            msg = f"{run_dir} n'existe pas, skip."
                            print(f"[WARN] {msg}")
                            logger.warning(msg)
                            continue
                        print(run_dir)
                        run_data = {"run_dir": run_dir}
                        model_pth = os.path.join(run_dir, "model.pth")
                        preds_json = os.path.join(run_dir, "preds.json")
                        config_json = os.path.join(run_dir, "config.json")

                        if os.path.exists(model_pth):
                            run_data["model_pth"] = model_pth
                        else:
                            msg = f"{model_pth} manquant."
                            print(f"[WARN] {msg}")
                            logger.warning(msg)
                        if os.path.exists(preds_json):
                            with open(preds_json, "r") as f:
                                run_data["preds"] = json.load(f)
                        else:
                            msg = f"{preds_json} manquant."
                            print(f"[WARN] {msg}")
                            logger.warning(msg)
                        if os.path.exists(config_json):
                            try:
                                with open(config_json, "r") as f:
                                    run_data["config"] = json.load(f)
                            except Exception as e:
                                msg = f"Impossible de charger {config_json}: {e}"
                                print(f"[WARN] {msg}")
                                logger.warning(msg)
                        else:
                            msg = f"{config_json} manquant."
                            print(f"[WARN] {msg}")
                            logger.warning(msg)
                        runs.append(run_data)
                    all_runs_data[model][quality][seed_type][epoch] = runs

            # --- Factorized: collect special groups of runs ---
            def collect_runs(run_dirs):
                runs = []
                for run_dir in run_dirs:
                    print(run_dir)
                    if not os.path.exists(run_dir):
                        msg = f"{run_dir} n'existe pas, skip."
                        print(f"[WARN] {msg}")
                        logger.warning(msg)
                        continue
                    run_data = {"run_dir": run_dir}
                    model_pth = os.path.join(run_dir, "model.pth")
                    preds_json = os.path.join(run_dir, "preds.json")
                    config_json = os.path.join(run_dir, "config.json")
                    if os.path.exists(model_pth):
                        run_data["model_pth"] = model_pth
                    else:
                        msg = f"{model_pth} manquant."
                        print(f"[WARN] {msg}")
                        logger.warning(msg)
                    if os.path.exists(preds_json):
                        with open(preds_json, "r") as f:
                            run_data["preds"] = json.load(f)
                    else:
                        msg = f"{preds_json} manquant."
                        print(f"[WARN] {msg}")
                        logger.warning(msg)
                    if os.path.exists(config_json):
                        try:
                            with open(config_json, "r") as f:
                                run_data["config"] = json.load(f)
                        except Exception as e:
                            msg = f"Impossible de charger {config_json}: {e}"
                            print(f"[WARN] {msg}")
                            logger.warning(msg)
                    else:
                        msg = f"{config_json} manquant."
                        print(f"[WARN] {msg}")
                        logger.warning(msg)
                    runs.append(run_data)
                return runs

            # Raw
            raw_run_dirs = [f"{experiment_dir}/{data_name}_{model}_{quality}_seed_init_{seed_value}_0_reprod_True" for seed_value in SEEDS_VALUES]
            all_runs_data[model][quality]["Raw"] = collect_runs(raw_run_dirs)

            if args.GPU_stability_measures:
                nb = 2
            else:
                nb = 5

            # Control_Reproc
            control_reproc_dirs = [f"{experiment_dir}/{data_name}_{model}_{quality}_control_{i}_all_42_500_reprod_True" for i in range(1, nb)]
            control_reproc_dirs.append(f"{experiment_dir}/{data_name}_{model}_{quality}_{SEED_TYPES[0]}_{SEEDS_VALUES[0]}_500_reprod_True")
            all_runs_data[model][quality]["Control_Reproc"] = collect_runs(control_reproc_dirs)

            # Hardware
            # hardware_dirs = [f"{experiment_dir}/{data_name}_{model}_{quality}_hardware_{i}_500_reprod_True" for i in range(0, nb)]
            hardware_dirs = [f"{experiment_dir}/{data_name}_{model}_{quality}_hardware_{i}_hardware_42_500_reprod_True" for i in range(1, nb)] # For retrocompatibilty : I missname the training
            hardware_dirs.append(f"{experiment_dir}/{data_name}_{model}_{quality}_{SEED_TYPES[0]}_{SEEDS_VALUES[0]}_500_reprod_True")
            all_runs_data[model][quality]["Hardware"] = collect_runs(hardware_dirs)

    # 2. Calculer les métriques
    all_metrics = {}
    
    def process_runs(runs, epoch=None, model=None, quality=None, seed_type=None):

        print("---------")
        # How to print only the name of the run_dir?
        for run in runs:
            print(run["run_dir"])
        print("---------")
        if not runs:
            return {}
            
        # --- Embedding-based metrics ---
        embeddings = []
        valid_runs_for_preds = []
        
        for run in runs:
            if "model_pth" in run:
                try:
                    emb = load_embedding_from_pth(run["model_pth"])
                    embeddings.append(emb)
                except Exception as e:
                    msg = f"Impossible de charger l'embedding de {run['model_pth']}: {e}"
                    print(f"[ERR] {msg}")
                    logger.error(msg)
            if "preds" in run:
                valid_runs_for_preds.append(run)
            else:
                msg = f"{run['run_dir']} n'a pas de preds.json."
                print(f"[WARN] {msg}")
                logger.warning(msg)
        print_ram_usage("After collecting embeddings")

        space_metrics = {}
        emb_list = embeddings
        if len(emb_list) > 1:
            print("Calcul des métriques d'espace...")
            logger.info("Calcul des métriques d'espace...")
            space_metrics = compute_all_neighbor_space_metrics(
                emb_list,
                k_values=K_VALUES
            )
            space_metrics["n_runs"] = len(emb_list)
            print("space_metrics:", space_metrics)
            logger.info(f"space_metrics: {space_metrics}")
            # --- Additional metrics from GitHub comparison.py (prefixed with 'cos_') ---
            # try:
            #     cos_metrics = compute_cos_metrics_via_github(emb_list, k_values=K_VALUES)
            #     space_metrics.update(cos_metrics)
            #     print("cos_github_metrics:", cos_metrics)
            #     logger.info(f"cos_github_metrics: {cos_metrics}")
            # except Exception as e:
            #     print(f"[WARN] Error computing cos_github_metrics: {e}")
            #     logger.warning(f"Error computing cos_github_metrics: {e}")
        print_ram_usage("After computing embedding metrics")

        # --- Prediction-based metrics ---
        prediction_metrics = {}
        preds_runs = [run["preds"] for run in valid_runs_for_preds if "preds" in run]
        print_ram_usage("After collecting predictions")
        # Compute metrics for all K values
        print("Calcul des métriques des premières predictions metrics...")
        logger.info("Calcul des métriques des premières predictions metrics...")
        ambiguity_metrics = compute_ambiguity(valid_runs_for_preds, K_VALUES)
        discrepancy_metrics = compute_discrepancy(valid_runs_for_preds, K_VALUES)
        rank_stats = compute_rank_stats(valid_runs_for_preds)
        print("rank_stats:", rank_stats)
        print("ambiguity_metrics:", ambiguity_metrics)
        print("discrepancy_metrics:", discrepancy_metrics)
        logger.info(f"rank_stats: {rank_stats}")
        logger.info(f"ambiguity_metrics: {ambiguity_metrics}")
        logger.info(f"discrepancy_metrics: {discrepancy_metrics}")
        print_ram_usage("After computing basic prediction metrics")
        # Get prediction scores using stability_utils
        try:
            # Add data_dir to each run if not present
            for run in valid_runs_for_preds:
                if 'data_dir' not in run:
                    run['data_dir'] = data_dir
            
            # Get the minimum k_value across all runs to ensure compatibility
            max_k_needed = max(K_VALUES)
            
            # Check if we need to recompute with larger k
            need_recompute = False
            if valid_runs_for_preds and 'preds' in valid_runs_for_preds[0]:
                current_k = valid_runs_for_preds[0]['preds'].get('k_value', 0)
                if current_k < max_k_needed:
                    msg = f"Current k_value ({current_k}) is smaller than required ({max_k_needed}). Recomputing predictions..."
                    print(f"[INFO] {msg}")
                    logger.info(msg)
                    need_recompute = True
            
            # Get predictions (recompute if necessary)
            if need_recompute:
                preds_list = get_preds_list(valid_runs_for_preds, max_k_needed)
            else:
                preds_list = [run['preds'] for run in valid_runs_for_preds if 'preds' in run]
            
            print_ram_usage("After loading predictions")
            
            # Compute prediction-based metrics
            pred_metrics = {}
            if len(preds_list) > 1:
                pred_metrics = compute_all_prediction_metrics(
                    preds_list, 
                    k_values=K_VALUES,
                    output_dir=output_dir,
                    model=model,
                    dataset=data_name,
                    quality=quality,
                    seed_type=seed_type,
                    epoch=epoch
                )
                print("pred_metrics:", pred_metrics)
                logger.info(f"pred_metrics: {pred_metrics}")
            
            # Combine all metrics
            prediction_metrics = {
                "n_runs": len(preds_runs),
                **ambiguity_metrics,
                **discrepancy_metrics,
                **rank_stats,
                **pred_metrics,
                **ambiguity_metrics,  # Add all ambiguity@K metrics
                **discrepancy_metrics  # Add all discrepancy@K metrics
            }
            
        except Exception as e:
            msg = f"Error computing prediction metrics: {str(e)}"
            print(f"[WARNING] {msg}")
            logger.warning(msg)
            prediction_metrics = {
                "n_runs": len(preds_runs),
                **rank_stats,
                "error": str(e)
            }
            
        # --- MRR metrics from metrics.json ---
        mrr_metrics = compute_mrr_stats_from_runs(runs)

        # --- Configs: attach configs of the runs used ---
        configs = {}
        for run in runs:
            try:
                run_name = os.path.basename(run.get("run_dir", "")) or run.get("run_dir", "")
                if "config" in run:
                    configs[run_name] = run["config"]
                else:
                    # try to load as fallback
                    cfg_path = os.path.join(run.get("run_dir", ""), "config.json")
                    if cfg_path and os.path.exists(cfg_path):
                        with open(cfg_path, "r") as f:
                            configs[run_name] = json.load(f)
                    else:
                        # mark missing config
                        configs[run_name] = {"_error": "config.json missing"}
                        logger.warning(f"Config missing for run {run_name}")
            except Exception as e:
                logger.warning(f"Error attaching config for run {run.get('run_dir','')}: {e}")

        return {
            "space_metrics": space_metrics,
            "prediction_metrics": prediction_metrics,
            "mrr_metrics": mrr_metrics,
            "configs": configs,
        }

    # Main loop
    for model, quality_data in all_runs_data.items():
        all_metrics[model] = {}
        for quality, seed_type_data in quality_data.items():
            all_metrics[model][quality] = {}
            for seed_type, epochs_data in seed_type_data.items():
                all_metrics[model][quality][seed_type] = {}
                
                if isinstance(epochs_data, dict):  # Data with epochs
                    for epoch, runs in epochs_data.items():
                        if not runs:
                            continue
                        print("Processing runs for", model, quality, seed_type, epoch)
                        logger.info(f"Processing runs for {model} {quality} {seed_type} {epoch}")
                        metrics = process_runs(runs, epoch, model, quality, seed_type)
                        if metrics:
                            all_metrics[model][quality][seed_type][epoch] = metrics
                else:  # Special data (Raw, Control_Reproc, etc.)
                    print("Processing runs for", model, quality, seed_type)
                    logger.info(f"Processing runs for {model} {quality} {seed_type}")
                    metrics = process_runs(epochs_data, None, model, quality, seed_type)
                    if metrics:
                        all_metrics[model][quality][seed_type] = metrics
                
                # 3. Save results
                if args.GPU_stability_measures:
                    output_dir = f"{PROJECT_PREFIX}/stability_measures_results_gpu"
                else:
                    output_dir = f"{PROJECT_PREFIX}/stability_measures_results"
                os.makedirs(output_dir, exist_ok=True)
                
                for model, model_metrics in all_metrics.items():
                    model_metrics["timestamp"] = time.time()
                    out_path = os.path.join(output_dir, f"{data_name}_{model}_stability.json")
                    with open(out_path, "w") as f:
                        json.dump(model_metrics, f, indent=2)
                
                msg = f"Stability metrics computed and saved in {output_dir}/"
                print(f"[INFO] {msg}")
                logger.info(msg)
