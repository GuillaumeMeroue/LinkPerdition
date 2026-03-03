import json
import os
from typing import Dict, Any, List, Optional, Tuple

# Constants per user spec
DATASETS = ["WN18RR", "kinship", "nations", "codex-s", "FB15k-237"]
MODELS = ["ConvE", "TransE", "DistMult", "RGCN", "Transformer", "ComplEx", "RotatE"]
QUALITIES = ["best", "median", "worst"]
# In JSON, Control_Reproc and Hardware appear as seed_type names as well
SEED_TYPES = ["seed_init", "seed_forward", "seed_neg", "seed_order", "all", "Control_Reproc", "Hardware"]
RAW_ALIAS = "Raw"  # maps to seed_init @ run_id 0

# Metrics of interest - ALL available metrics for loading
SPACE_METRICS = [
    "space_overlap@10_mean", 
    "space_jaccard_l2@10_mean", 
    "space_jaccard_cosine@10_mean",
    "rbo@10_mean"
]
PREDICTION_METRICS = [
    "pred_overlap@1", 
    "pred_overlap@10", 
    "pred_rbo@10",
    "ambiguity@10",
    "super_pred_jaccard@10_mean",
    "super_pred_overlap@10_mean"
]
# MRR metrics live in separate block
MRR_MEAN_KEY = "mrr_mean"
MRR_STD_KEY = "mrr_std"

# Metrics for specific research questions
RQ1_METRICS = [
    "space_overlap@10_mean",
    "space_jaccard_l2@10_mean",
    "rbo@10_mean",
    "pred_rbo@10",
    "ambiguity@10",
    "pred_overlap@1",
    "pred_overlap@10",
]

RQ2_VARIANT_METRICS = [
    "pred_overlap@10",
    "space_overlap@10_mean",
]

RQ3_VARIANT_METRICS = [
    "pred_overlap@10",
    "space_overlap@10_mean",
    "super_pred_jaccard@10_mean",
]

OTHER_METRICS = [
    "ambiguity@10",
    "space_jaccard_l2@10_mean",
    "space_jaccard_cosine@10_mean",
    "pred_overlap@10",
    "pred_rbo@10",
]

RQ3_ALL_METRICS = [
    "pred_overlap@10",  # Consistancy
    "super_pred_overlap@10_mean",  # Homogeneity
    "pred_overlap@10",  # Pred-Overlap (duplicate but needed)
    "space_overlap@10_mean",  # Space-Overlap
    "mrr_mean",
    "hit@1_mean",
    "hit@10_mean",
]

# Helper: resolve metric std counterpart key
STD_SUFFIXES = {
    "space_overlap@10_mean": "space_overlap@10_std",
    "space_jaccard_l2@10_mean": "space_jaccard_l2@10_std",
    "space_jaccard_cosine@10_mean": "space_jaccard_cosine@10_std",
    "rbo@10_mean": "rbo@10_std",
    "pred_overlap@1": "pred_overlap@1_std",
    "pred_overlap@10": "pred_overlap@10_std",
    "pred_rbo@10": "pred_rbo@10_std",
    "ambiguity@10": None,  # No std for ambiguity
    "super_pred_jaccard@10_mean": "super_pred_jaccard@10_std",
    "super_pred_overlap@10_mean": "super_pred_overlap@10_std",
}


class MetricsError(Exception):
    pass


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _extract_seed_block(quality_block: Dict[str, Any], seed: str, run_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve metrics block for (seed, run_id). For special seeds Control_Reproc/Hardware,
    data may be stored directly under the seed without a run_id level.
    """
    seed_block = quality_block.get(seed)
    if not isinstance(seed_block, dict):
        return None
    # Special-case: some seeds (Control_Reproc, Hardware) may not use run_id nesting
    direct_has_metrics = any(k in seed_block for k in ("space_metrics", "prediction_metrics", "mrr_metrics"))
    if direct_has_metrics and seed in ("Control_Reproc", "Hardware"):
        return seed_block
    # Otherwise expect nested by run_id (e.g., '500' or '0')
    run_block = seed_block.get(run_id)
    if not isinstance(run_block, dict):
        return None
    return run_block


def _get_metric_with_std(container: Dict[str, Any], mean_key: str) -> Tuple[Optional[float], Optional[float]]:
    mean = container.get(mean_key)
    std_key = STD_SUFFIXES.get(mean_key)
    std = container.get(std_key) if std_key else None
    return mean, std


def _validate_n_runs(run_block: Dict[str, Any], dataset: str, model: str, quality: str, seed_label: str) -> List[str]:
    """Return list of error strings if invalid n_runs."""
    errs: List[str] = []
    sm = run_block.get("space_metrics", {})
    pm = run_block.get("prediction_metrics", {})
    # Only validate if keys present
    sm_n = sm.get("n_runs")
    pm_n = pm.get("n_runs")
    for split, n in [("space_metrics", sm_n), ("prediction_metrics", pm_n)]:
        if n is None:
            errs.append(f"[{dataset}][{model}][{quality}][{seed_label}] missing n_runs in {split}")
        elif n != 5:
            errs.append(f"[{dataset}][{model}][{quality}][{seed_label}] invalid n_runs={n} in {split} (expected 5)")
    return errs


def load_metrics_for_pair(dataset: str, model: str, base_dir: str = "25_11_new/stability_measures_results") -> Tuple[Dict[str, Dict[str, Dict[str, Dict[str, Optional[float]]]]], List[str]]:
    """
    Load metrics for a (dataset, model) pair from 07_09/{dataset}_{model}_stability.json.

    Returns
    -------
    data: dict structured as data[quality][seed_label][metric] = {"mean": float|None, "std": float|None}
    errors: list of strings describing missing metrics or invalid n_runs
    """
    errors: List[str] = []
    result: Dict[str, Dict[str, Dict[str, Dict[str, Optional[float]]]]] = {}

    path = os.path.join(base_dir, f"{dataset}_{model}_stability.json")
    if not os.path.exists(path):
        raise MetricsError(f"Missing file: {path}")

    obj = _read_json(path)

    for quality in QUALITIES:
        qblock = obj.get(quality)
        if not isinstance(qblock, dict):
            errors.append(f"[{dataset}][{model}] missing quality '{quality}' block")
            continue

        result.setdefault(quality, {})

        # Determine which seed labels to load for this quality
        if quality == "best":
            seeds_to_load = SEED_TYPES.copy()
            include_raw = True
        else:
            # For median and worst, only seed_type = all (user spec)
            seeds_to_load = ["all"]
            include_raw = False
        # Always consider that Control_Reproc and Hardware live as seed_types@500

        # Load each requested seed@500
        for seed in seeds_to_load:
            run_block = _extract_seed_block(qblock, seed, "500")
            if run_block is None:
                if seed in ("Control_Reproc", "Hardware"):
                    errors.append(f"[{dataset}][{model}][{quality}] missing seed_type '{seed}' block")
                else:
                    errors.append(f"[{dataset}][{model}][{quality}] missing seed_type '{seed}' run_id 500")
                continue

            seed_label = seed  # display label
            result[quality].setdefault(seed_label, {})

            # Validate n_runs (skip for Control_Reproc/Hardware which may not have n_runs semantics)
            if seed not in ("Control_Reproc", "Hardware"):
                errors.extend(_validate_n_runs(run_block, dataset, model, quality, seed_label))

            # Extract space metrics
            sm = run_block.get("space_metrics", {})
            if not isinstance(sm, dict):
                errors.append(f"[{dataset}][{model}][{quality}][{seed_label}] missing space_metrics")
            else:
                for m in SPACE_METRICS:
                    mean, std = _get_metric_with_std(sm, m)
                    if mean is None:
                        errors.append(f"[{dataset}][{model}][{quality}][{seed_label}] missing metric '{m}' in space_metrics")
                    result[quality][seed_label][m] = {"mean": mean, "std": std}

            # Extract prediction metrics
            pm = run_block.get("prediction_metrics", {})
            if not isinstance(pm, dict):
                errors.append(f"[{dataset}][{model}][{quality}][{seed_label}] missing prediction_metrics")
            else:
                for m in PREDICTION_METRICS:
                    mean, std = _get_metric_with_std(pm, m)
                    if mean is None:
                        errors.append(f"[{dataset}][{model}][{quality}][{seed_label}] missing metric '{m}' in prediction_metrics")
                    result[quality][seed_label][m] = {"mean": mean, "std": std}

            # MRR metrics
            mrr = run_block.get("mrr_metrics", {})
            if not isinstance(mrr, dict):
                errors.append(f"[{dataset}][{model}][{quality}][{seed_label}] missing mrr_metrics")
            else:
                result[quality][seed_label][MRR_MEAN_KEY] = {"mean": mrr.get(MRR_MEAN_KEY), "std": None}
                result[quality][seed_label][MRR_STD_KEY] = {"mean": mrr.get(MRR_STD_KEY), "std": None}
                result[quality][seed_label]["hit@1_mean"] = {"mean": mrr.get("hit@1_mean"), "std": mrr.get("hit@1_std")}
                result[quality][seed_label]["hit@10_mean"] = {"mean": mrr.get("hit@10_mean"), "std": mrr.get("hit@10_std")}

        # Raw is seed_init @ run_id 0 for 'best' only (as per spec)
        if include_raw:
            run_block = _extract_seed_block(qblock, "seed_init", "0")
            if run_block is None:
                errors.append(f"[{dataset}][{model}][{quality}] missing Raw (seed_init run_id 0)")
            else:
                seed_label = RAW_ALIAS
                result[quality].setdefault(seed_label, {})
                errors.extend(_validate_n_runs(run_block, dataset, model, quality, seed_label))

                sm = run_block.get("space_metrics", {})
                if not isinstance(sm, dict):
                    errors.append(f"[{dataset}][{model}][{quality}][{seed_label}] missing space_metrics")
                else:
                    for m in SPACE_METRICS:
                        mean, std = _get_metric_with_std(sm, m)
                        if mean is None:
                            errors.append(f"[{dataset}][{model}][{quality}][{seed_label}] missing metric '{m}' in space_metrics")
                        result[quality][seed_label][m] = {"mean": mean, "std": std}

                pm = run_block.get("prediction_metrics", {})
                if not isinstance(pm, dict):
                    errors.append(f"[{dataset}][{model}][{quality}][{seed_label}] missing prediction_metrics")
                else:
                    for m in PREDICTION_METRICS:
                        mean, std = _get_metric_with_std(pm, m)
                        if mean is None:
                            errors.append(f"[{dataset}][{model}][{quality}][{seed_label}] missing metric '{m}' in prediction_metrics")
                        result[quality][seed_label][m] = {"mean": mean, "std": std}

                mrr = run_block.get("mrr_metrics", {})
                if not isinstance(mrr, dict):
                    errors.append(f"[{dataset}][{model}][{quality}][{seed_label}] missing mrr_metrics")
                else:
                    result[quality][seed_label][MRR_MEAN_KEY] = {"mean": mrr.get(MRR_MEAN_KEY), "std": None}
                    result[quality][seed_label][MRR_STD_KEY] = {"mean": mrr.get(MRR_STD_KEY), "std": None}
                    result[quality][seed_label]["hit@1_mean"] = {"mean": mrr.get("hit@1_mean"), "std": mrr.get("hit@1_std")}
                    result[quality][seed_label]["hit@10_mean"] = {"mean": mrr.get("hit@10_mean"), "std": mrr.get("hit@10_std")}

    return result, errors


def load_all(base_dir: str = "Results/Stability_results") -> Dict[Tuple[str, str], Dict[str, Dict[str, Dict[str, Dict[str, Optional[float]]]]]]:
    """Load metrics for all (dataset, model) pairs that have a json file present."""
    data: Dict[Tuple[str, str], Dict[str, Dict[str, Dict[str, Dict[str, Optional[float]]]]]] = {}
    for ds in DATASETS:
        for model in MODELS:
            path = os.path.join(base_dir, f"{ds}_{model}_stability.json")
            print(path)
            if not os.path.exists(path):
                continue
            pair_data, errs = load_metrics_for_pair(ds, model, base_dir=base_dir)
            for e in errs:
                print(f"ERROR: {e}")
            data[(ds, model)] = pair_data
    return data


def ordered_seed_labels_for_quality(quality: str) -> List[str]:
    if quality == "best":
        return [
            "seed_init",
            "seed_forward",
            "seed_neg",
            "seed_order",
            "all",
            RAW_ALIAS,
            "Control_Reproc",
            "Hardware",
        ]
    else:
        return ["all"]


def all_metric_rows() -> List[str]:
    """Return the list of metric names to display as rows in tables/plots."""
    return SPACE_METRICS + PREDICTION_METRICS + [MRR_MEAN_KEY, MRR_STD_KEY]


if __name__ == "__main__":
    # Simple sanity run
    all_data = load_all()
    print(f"Loaded pairs: {len(all_data)}")
    # Emphasize most important metric presence
    missing = []
    for (ds, mdl), d in all_data.items():
        for q, qd in d.items():
            for s, sd in qd.items():
                pj = sd.get("pred_jaccard@1", {}).get("mean")
                if pj is None:
                    missing.append((ds, mdl, q, s))
    if missing:
        print("Missing pred_jaccard@1 in:")
        for item in missing:
            print(" - ", item)
    else:
        print("All pred_jaccard@1 present.")
