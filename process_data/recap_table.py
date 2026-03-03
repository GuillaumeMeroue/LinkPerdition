import os
from typing import Dict, Tuple, List, Optional

from .load_metrics import (
    load_all,
    ordered_seed_labels_for_quality,
)

RECAP_METRICS = [
    ("mrr", "mrr_mean", "mrr_std"),  # (label, mean_key, std_key)
    ("jaccard@1", "jaccard@1_mean", "jaccard@1_std"),
    ("cka", "cka_mean", "cka_std"),
    ("pred_jaccard@1", "pred_jaccard@1", "pred_jaccard@1_std"),
    ("discrepancy", "discrepancy@1", None),
]

MODELS = ["ConvE", "TransE", "DistMult", "RGCN", "Transformer"]
DATASETS = ["WN18RR", "kinship", "nations", "codex-s"]


def fmt_cell(mean: Optional[float], std: Optional[float]) -> str:
    try:
        m = None if mean is None else float(mean)
    except (TypeError, ValueError):
        m = None
    try:
        s = None if std is None else float(std)
    except (TypeError, ValueError):
        s = None
    if m is None:
        return "--"
    if s is None:
        return f"${m:.3f}$"
    return f"${m:.3f} \\pm {s:.3f}$"


def esc(s: str) -> str:
    return s.replace("_", "\\_")


def generate_recap_table(output_path: str = "process_data/tables/recap.tex", quality: str = "best", landscape: bool = True) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    all_data = load_all()

    seeds = ordered_seed_labels_for_quality(quality)

    # Build LaTeX header
    header = []
    header.append("% Auto-generated recap table\n")
    if landscape:
        header.append("\\begin{sidewaystable*}[t]\n")
    else:
        header.append("\\begin{table*}[t]\n")
    header.append("\\centering\n")
    header.append("\\small\n")
    header.append("\\setlength{\\tabcolsep}{3pt}% tighter columns\n")

    # Columns: 1 (metric) + for each model: len(seeds) subcolumns
    n_metric_cols = 1
    n_sub = len(seeds)
    total_cols = n_metric_cols + len(MODELS) * n_sub

    # Column alignment: first left, others centered
    col_spec = "l" + ("c" * (total_cols - 1))
    header.append(f"\\resizebox{{\\textwidth}}{{!}}{{%\n")
    header.append(f"\\begin{{tabular}}{{{col_spec}}}\n")
    header.append("\\toprule\n")

    # First header row: empty for metric, then models spanning subcols
    first_row = ["metric"]
    for model in MODELS:
        first_row.append(f"\\multicolumn{{{n_sub}}}{{c}}{{{esc(model)}}}")
    header.append(" \\ & ".join(first_row) + " \\\\ \n")

    # Second header row: per-model seed names
    second_row = [" "]
    for _ in MODELS:
        second_row.extend([esc(s) for s in seeds])
    header.append(" \\ & ".join(second_row) + " \\\\ \n")
    header.append("\\midrule\n")

    lines: List[str] = []

    for dataset in DATASETS:
        # Dataset separator row
        lines.append(f"\\multicolumn{{{total_cols}}}{{l}}{{\\textbf{{{esc(dataset)}}}}} \\\\ \n")
        # For each metric row
        for label, mean_key, std_key in RECAP_METRICS:
            row_cells = [esc(label)]
            for model in MODELS:
                pair = all_data.get((dataset, model))
                # default missing
                if not pair:
                    row_cells.extend(["--"] * n_sub)
                    continue
                qblock = pair.get(quality, {})
                for seed in seeds:
                    md = qblock.get(seed, {})
                    if label == "mrr":
                        mean = md.get("mrr_mean", {}).get("mean")
                        std = md.get("mrr_std", {}).get("mean")
                    else:
                        mean = md.get(mean_key, {}).get("mean")
                        std = md.get(std_key, {}).get("std") if std_key else None
                        # For space/pred metrics in our structure, std is stored in the metric's std entry's "std" via load_metrics
                        # But load_metrics stored std alongside mean in same dict entries
                        if std is None and std_key:
                            # Try reading std from the same key's dict
                            std = md.get(mean_key, {}).get("std")
                    row_cells.append(fmt_cell(mean, std))
            lines.append(" \\ & ".join(row_cells) + " \\\\ \n")
        lines.append("\\midrule\n")

    footer = []
    footer.append("\\bottomrule\n")
    footer.append("\\end{tabular}\n")
    footer.append("}% resizebox\n")
    footer.append("\\caption{Récapitulatif des métriques: mrr, jaccard@1, cka, pred\\_jaccard@1, discrepancy. Qualité=best; sous-colonnes par seed.}\n")
    footer.append("\\label{tab:recap}\n")
    if landscape:
        footer.append("\\end{sidewaystable*}\n")
    else:
        footer.append("\\end{table*}\n")

    with open(output_path, "w") as f:
        f.writelines(header + lines + footer)
    print(f"Saved recap table: {output_path}")


if __name__ == "__main__":
    generate_recap_table()
