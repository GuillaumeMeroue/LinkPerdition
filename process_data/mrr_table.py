import os
from typing import List

from .load_metrics import load_all

MODELS = ["ConvE", "TransE", "DistMult", "RGCN", "Transformer"]
DATASETS = ["WN18RR", "kinship", "nations", "codex-s"]


def esc(s: str) -> str:
    return s.replace("_", "\\_")


def fmt_cell(mean, std) -> str:
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


essentials = {
    "all": "all",
}


def generate_mrr_table(output_path: str = "process_data/tables/mrr_all_seed.tex") -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    all_data = load_all()

    header: List[str] = []
    header.append("% Auto-generated MRR-only table (seed=all, quality=best)\n")
    header.append("\\begin{table*}[t]\n")
    header.append("\\centering\n")
    header.append("\\small\n")
    header.append("\\setlength{\\tabcolsep}{3pt}% tighter columns\n")

    # Columns: 1 (dataset) + one per model (quality fixed to best)
    n_metric_cols = 1
    n_models = len(MODELS)
    total_cols = n_metric_cols + n_models

    col_spec = "l" + ("c" * (total_cols - 1))
    header.append(f"\\resizebox{{\\textwidth}}{{!}}{{%\n")
    header.append(f"\\begin{{tabular}}{{{col_spec}}}\n")
    header.append("\\toprule\n")

    # Header row: dataset + model names
    first = ["dataset"] + [esc(m) for m in MODELS]
    header.append(" \\ & ".join(first) + " \\\\ \n")
    header.append("\\midrule\n")

    lines: List[str] = []
    for dataset in DATASETS:
        row = [esc(dataset)]
        for model in MODELS:
            pair = all_data.get((dataset, model), {})
            qd = pair.get("best", {})
            md = qd.get("all", {})
            mean = md.get("mrr_mean", {}).get("mean")
            std = md.get("mrr_std", {}).get("mean")
            row.append(fmt_cell(mean, std))
        lines.append(" \\ & ".join(row) + " \\\\ \n")

    footer: List[str] = []
    footer.append("\\bottomrule\n")
    footer.append("\\end{tabular}\n")
    footer.append("}% resizebox\n")
    footer.append("\\caption{MRR (seed=all, qualité=best) pour chaque dataset et modèle.}\n")
    footer.append("\\label{tab:mrr_all}\n")
    footer.append("\\end{table*}\n")

    with open(output_path, "w") as f:
        f.writelines(header + lines + footer)
    print(f"Saved MRR table: {output_path}")


if __name__ == "__main__":
    generate_mrr_table()
