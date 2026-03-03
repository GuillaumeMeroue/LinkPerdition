from typing import Dict, Tuple, List, Optional
import os
import pandas as pd

from .load_metrics import (
    load_all,
    ordered_seed_labels_for_quality,
    all_metric_rows,
    QUALITIES,
    RAW_ALIAS,
)


def fmt_cell(mean: Optional[float], std: Optional[float]) -> str:
    # Guard against None, NaN, and non-numeric values
    def _num(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    m = _num(mean)
    s = _num(std)
    if m is None:
        return "--"
    if s is None:
        return f"${m:.3f}$"
    return f"${m:.3f} \\pm {s:.3f}$"


def build_table_for_pair(pair_data: Dict[str, Dict[str, Dict[str, Dict[str, Optional[float]]]]]) -> pd.DataFrame:
    metrics = all_metric_rows()
    # Columns are MultiIndex: (quality, seed_label)
    cols: List[Tuple[str, str]] = []
    data_cells: Dict[str, List[str]] = {m: [] for m in metrics}

    def esc(s: str) -> str:
        return s.replace("_", "\\_")

    for q in QUALITIES:
        seeds = ordered_seed_labels_for_quality(q)
        for s in seeds:
            cols.append((q, esc(s)))
            for m in metrics:
                entry = pair_data.get(q, {}).get(s, {}).get(m, {"mean": None, "std": None})
                data_cells[m].append(fmt_cell(entry.get("mean"), entry.get("std")))

    columns = pd.MultiIndex.from_tuples(cols, names=["quality", "seed"]) 
    df = pd.DataFrame({m: data_cells[m] for m in metrics})
    df.index = columns
    # We want metrics as rows, so transpose
    df = df.T
    # Escape metric names for LaTeX
    df.index = [m.replace("_", "\\_") for m in df.index]
    df.index.name = "metric"
    return df


def make_tables(output_dir: str = "process_data/tables") -> None:
    os.makedirs(output_dir, exist_ok=True)
    all_data = load_all()
    for (dataset, model), pair_data in all_data.items():
        df = build_table_for_pair(pair_data)
        fname = f"{dataset}_{model}.tex"
        fpath = os.path.join(output_dir, fname)
        with open(fpath, "w") as f:
            f.write(df.to_latex(escape=False, multirow=True, multicolumn=True))
        print(f"Saved table: {fpath}")


if __name__ == "__main__":
    make_tables()
