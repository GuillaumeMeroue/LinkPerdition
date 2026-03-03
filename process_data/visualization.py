import os
from typing import Dict, Tuple, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap
import pandas as pd
import numpy as np

from .load_metrics import (
    load_all,
    ordered_seed_labels_for_quality,
    all_metric_rows,
    QUALITIES,
    RQ1_METRICS,
    RQ2_VARIANT_METRICS,
    RQ3_VARIANT_METRICS,
    OTHER_METRICS,
    RQ3_ALL_METRICS,
)

# Custom palette as requested: [284B63, D86E6E, 0077B6, F4978E, 90E0EF, FBC4AB]
PALETTE = ["#284B63", "#D86E6E", "#0077B6", "#F4978E", "#90E0EF", "#FBC4AB"]
# PALETTE = sns.color_palette("tab10")
# PALETTE = sns.color_palette("Set2")

MODEL_ORDER = ["TransE", "ConvE", "DistMult", "Transformer", "RGCN", "ComplEx", "RotatE"]
# DATASET_ORDER = ["WN18RR", "FB15k-237", "codex-s", "kinship", "nations"]

DATASET_ORDER = ["WN18RR", "FB15k-237", "codex-s"]
HEATMAP_MODEL_ORDER = ["TransE", "ConvE", "DistMult", "Transformer", "RGCN", "ComplEx", "RotatE"]

# Toggles for including special seeds in best quality visualizations
SHOW_RAW = False
SHOW_CONTROL_REPROC = False
SHOW_HARDWARE = True

def _best_seed_order_filtered():
    order = ordered_seed_labels_for_quality("best")
    keep = []
    for s in order:
        if s == "Raw" and not SHOW_RAW:
            continue
        if s == "Control_Reproc" and not SHOW_CONTROL_REPROC:
            continue
        if s == "Hardware" and not SHOW_HARDWARE:
            continue
        keep.append(s)
    return keep


def _collect_flat_records(all_data) -> List[Dict[str, object]]:
    rows = []
    for (dataset, model), pair_data in all_data.items():
        for q in QUALITIES:
            for seed in ordered_seed_labels_for_quality(q):
                metrics = pair_data.get(q, {}).get(seed, {})
                for m_name, v in metrics.items():
                    rows.append({
                        "dataset": dataset,
                        "model": model,
                        "quality": q,
                        "seed": seed,
                        "metric": m_name,
                        "mean": v.get("mean"),
                        "std": v.get("std"),
                    })
    return rows


# RQ1: For each metric a groupbar.
# - Group by model
# - One bar per dataset

def make_rq1(output_dir: str = "process_data/RQ1"):
    os.makedirs(output_dir, exist_ok=True)
    all_data = load_all()
    rows = _collect_flat_records(all_data)
    df = pd.DataFrame(rows)

    # We'll do RQ1 with quality=best and seed=all (most representative), unless seed not present then skip
    filt = (df["quality"] == "best") & (df["seed"] == "all")
    dfb = df[filt]

    metrics = RQ1_METRICS
    palette_rq1 = [PALETTE[-1], PALETTE[0], PALETTE[1]]

    for metric in metrics:
        dmf = dfb[dfb["metric"] == metric].copy()
        # Mark RGCN + FB15k-237 for N/A label (keep in dataframe)
        rgcn_fb_mask = (dmf["model"] == "RGCN") & (dmf["dataset"] == "FB15k-237")
        # Coerce mean to numeric
        dmf["mean"] = pd.to_numeric(dmf["mean"], errors="coerce")
        # Set RGCN + FB15k-237 to 0 for plotting (will add N/A text)
        dmf.loc[rgcn_fb_mask, "mean"] = 0.0
        dmf.loc[rgcn_fb_mask, "std"] = 0.0
        # Drop other NaNs
        dmf = dmf.dropna(subset=["mean"]) 
        if dmf.empty:
            continue
        plt.figure(figsize=(10, 4))
        # Enforce category orders for reproducible layout/colors
        dmf["model"] = pd.Categorical(dmf["model"], categories=MODEL_ORDER, ordered=True)
        dmf["dataset"] = pd.Categorical(dmf["dataset"], categories=DATASET_ORDER, ordered=True)
        dmf.sort_values(["model", "dataset"], inplace=True)
        ax = sns.barplot(data=dmf, x="model", y="mean", hue="dataset", palette=palette_rq1, errorbar=None)
        # Add error bars if std available
        dmf["std"] = pd.to_numeric(dmf["std"], errors="coerce")
        # Build stds in plotting order (model major, dataset minor)
        x_models = [c for c in MODEL_ORDER if c in dmf["model"].cat.categories]
        h_datasets = [c for c in DATASET_ORDER if c in dmf["dataset"].cat.categories]
        std_list = []
        height_list = []
        for m in x_models:
            subm = dmf[dmf["model"] == m]
            for d in h_datasets:
                row = subm[subm["dataset"] == d]
                if not row.empty:
                    std_list.append(row.iloc[0]["std"])
                    height_list.append(row.iloc[0]["mean"])
        # Iterate bars and attach errorbars
        for patch, yerr in zip(ax.patches, std_list):
            if pd.notna(yerr) and yerr > 0:
                x = patch.get_x() + patch.get_width() / 2
                y = patch.get_height()
                ax.errorbar(x, y, yerr=yerr, fmt='none', ecolor='black', capsize=3, lw=1)
        # Force scale for ambiguity/discrepancy and all jaccard/rbo/overlap metrics
        if ("ambiguity" in metric) or ("discrepancy" in metric) or ("jaccard" in metric) or ("rbo" in metric) or ("overlap" in metric):
            ax.set_ylim(0.0, 1.0)
        # No titles/legends and no axis labels
        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel("")
        # Print color mapping (dataset -> color)
        hue_levels = [d for d in DATASET_ORDER if d in dmf["dataset"].cat.categories]
        mapping = {lvl: PALETTE[i] for i, lvl in enumerate(hue_levels)}
        print(f"[RQ1][{metric}] color map (dataset): {mapping}")
        # Remove any residual legend and enlarge model ticks
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
        # Larger tick labels for models on x-axis
        ax.tick_params(axis='x', labelsize=14)
        # Ensure spines visible and add tiny axis margins to avoid clipping
        for s in ax.spines.values():
            s.set_visible(True)
        ax.margins(x=0.02, y=0.02)

       # Add N/A text for RGCN + FB15k-237
        if "RGCN" in x_models and "FB15k-237" in h_datasets:
            rgcn_idx = x_models.index("RGCN")
            # FB15k-237 est le dataset du milieu (index 1 sur 3), donc sa position
            # correspond exactement à l'index entier du modèle sur l'axe X.
            x_pos = rgcn_idx
            y_pos = ax.get_ylim()[1] * 0.02
            ax.text(x_pos, y_pos, "N/A", ha='center', va='bottom', fontsize=10)
        # Save without visible whitespace but avoid clipping the right edge
        fname = os.path.join(output_dir, f"rq1_{metric.replace('@','at').replace('/','_')}.png")
        plt.savefig(fname, bbox_inches='tight', pad_inches=0.03)
        plt.close()
        print(f"Saved {fname}")
        # Alternate plot with 1 - value for ambiguity/discrepancy
        if ("ambiguity" in metric) or ("discrepancy" in metric):
            dmf_alt = dmf.copy()
            dmf_alt["mean"] = 1.0 - dmf_alt["mean"]
            plt.figure(figsize=(10, 4))
            ax = sns.barplot(data=dmf_alt, x="model", y="mean", hue="dataset", palette=palette_rq1, errorbar=None)
            # reuse same std_list order since std is invariant to 1 - x
            for patch, yerr in zip(ax.patches, std_list):
                if pd.notna(yerr) and yerr > 0:
                    x = patch.get_x() + patch.get_width() / 2
                    y = patch.get_height()
                    ax.errorbar(x, y, yerr=yerr, fmt='none', ecolor='black', capsize=3, lw=1)

            ax.set_ylim(0.0, 1.0)
            # Match styling: no title/legend/labels
            ax.set_title("")
            ax.set_xlabel("")
            ax.set_ylabel("")
            # Print color mapping
            hue_levels = [d for d in DATASET_ORDER if d in dmf_alt["dataset"].cat.categories]
            mapping = {lvl: PALETTE[i] for i, lvl in enumerate(hue_levels)}
            print(f"[RQ1][alt(1-{metric})] color map (dataset): {mapping}")
            # Larger ticks and avoid clipping
            ax.tick_params(axis='x', labelsize=14)
            for s in ax.spines.values():
                s.set_visible(True)
            ax.margins(x=0.02, y=0.02)
# Add N/A text for RGCN + FB15k-237
            if "RGCN" in x_models and "FB15k-237" in h_datasets:
                    rgcn_idx = x_models.index("RGCN")
                    # FB15k-237 est le dataset du milieu (index 1 sur 3), donc sa position
                    # correspond exactement à l'index entier du modèle sur l'axe X.
                    x_pos = rgcn_idx
                    y_pos = ax.get_ylim()[1] * 0.02
                    ax.text(x_pos, y_pos, "N/A", ha='center', va='bottom', fontsize=10)
                    # Remove any residual legend and enlarge model ticks
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()
            fname = os.path.join(output_dir, f"rq1_alt_{metric.replace('@','at').replace('/','_')}.png")
            plt.savefig(fname, bbox_inches='tight', pad_inches=0.03)
            plt.close()
            print(f"Saved {fname}")


def make_rq2_variant(output_dir: str = "process_data/RQ2_variant"):
    """
    Variant of RQ2 where colors map to models and grouping is by seed type across x=model.
    One subplot per dataset in a (2,2) grid.
    """
    os.makedirs(output_dir, exist_ok=True)
    all_data = load_all()
    rows = _collect_flat_records(all_data)
    df = pd.DataFrame(rows)

    metrics = RQ2_VARIANT_METRICS
    # Filter to only FB15k-237, WN18RR, codex-s (exclude kinship and nations)
    datasets = DATASET_ORDER     

    def add_errorbars(ax, x_levels, hue_levels, df, x_col, hue_col, y_col="mean", err_col="std"):
        expected = []
        for xv in x_levels:
            for hv in hue_levels:
                expected.append((xv, hv))
        yerrs = []
        for xv, hv in expected:
            row = df[(df[x_col] == xv) & (df[hue_col] == hv)]
            if not row.empty:
                yerrs.append(row.iloc[0][err_col])
            else:
                yerrs.append(np.nan)
        for patch, yerr in zip(ax.patches, yerrs):
            if pd.notna(yerr) and yerr > 0:
                x = patch.get_x() + patch.get_width() / 2
                y = patch.get_height()
                ax.errorbar(x, y, yerr=yerr, fmt='none', ecolor='black', capsize=3, lw=1)

    for metric in metrics:
        for ds in datasets:
            plt.figure(figsize=(10, 5))
            sub = df[(df["dataset"] == ds) & (df["metric"] == metric) & (df["quality"] == "best")].copy()
            # Mark RGCN + FB15k-237 for N/A label
            rgcn_fb_mask = False
            if ds == "FB15k-237":
                rgcn_fb_mask = sub["model"] == "RGCN"
                sub.loc[rgcn_fb_mask, "mean"] = 0.0
                sub.loc[rgcn_fb_mask, "std"] = 0.0
            seed_order = _best_seed_order_filtered()
            sub["seed"] = pd.Categorical(sub["seed"], categories=seed_order, ordered=True)
            sub["model"] = pd.Categorical(sub["model"], categories=MODEL_ORDER, ordered=True)
            sub.sort_values(["model", "seed"], inplace=True)
            sub["mean"] = pd.to_numeric(sub["mean"], errors="coerce")
            sub["std"] = pd.to_numeric(sub["std"], errors="coerce")
            sub = sub.dropna(subset=["mean"]) 
            ax = sns.barplot(data=sub, x="model", y="mean", hue="seed", palette=PALETTE, errorbar=None)
            add_errorbars(ax, [m for m in MODEL_ORDER if m in sub["model"].cat.categories], list(sub["seed"].cat.categories), sub, "model", "seed")
            if ("ambiguity" in metric) or ("discrepancy" in metric) or ("jaccard" in metric) or ("rbo" in metric) or ("overlap" in metric):
                ax.set_ylim(0.0, 1.0)
            # No titles/legends and no axis labels
            ax.set_title("")
            ax.set_xlabel("")
            ax.set_ylabel("")
            plt.xticks(rotation=0)
            # Print color mapping (seed -> color)
            hue_levels = list(sub["seed"].cat.categories)
            mapping = {lvl: PALETTE[i] for i, lvl in enumerate(hue_levels)}
            print(f"[RQ2_variant][{metric}][{ds}] color map (seed): {mapping}")
            # Larger tick labels for models on x-axis
            ax.tick_params(axis='x', labelsize=14)
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()
            for s in ax.spines.values():
                s.set_visible(True)
            ax.margins(x=0.02, y=0.02)
            # Add N/A text for RGCN + FB15k-237
            if ds == "FB15k-237":
                x_models = [m for m in MODEL_ORDER if m in sub["model"].cat.categories]
                rgcn_idx = x_models.index("RGCN") if "RGCN" in x_models else -1
                if rgcn_idx >= 0:
                    # The exact center of a categorical group on the x-axis is its integer index
                    x_pos = rgcn_idx
                    y_pos = ax.get_ylim()[1] * 0.05
                    ax.text(x_pos, y_pos, "N/A", ha='center', va='bottom', fontsize=12)
            fname = os.path.join(output_dir, f"rq2_variant_{metric.replace('@','at').replace('/','_')}_{ds}.png")
            plt.savefig(fname, bbox_inches='tight', pad_inches=0.03)
            plt.close()
            print(f"Saved {fname}")
            # Alt plot 1 - value
            if ("ambiguity" in metric) or ("discrepancy" in metric):
                sub_alt = sub.copy()
                sub_alt["mean"] = 1.0 - sub_alt["mean"]
                plt.figure(figsize=(10, 5))
                ax = sns.barplot(data=sub_alt, x="model", y="mean", hue="seed", palette=PALETTE, errorbar=None)
                add_errorbars(ax, [m for m in MODEL_ORDER if m in sub_alt["model"].cat.categories], list(sub_alt["seed"].cat.categories), sub_alt, "model", "seed")
                ax.set_ylim(0.0, 1.0)
                ax.set_title("")
                ax.set_xlabel("")
                ax.set_ylabel("")
                plt.xticks(rotation=0)
                hue_levels = list(sub_alt["seed"].cat.categories)
                mapping = {lvl: PALETTE[i] for i, lvl in enumerate(hue_levels)}
                print(f"[RQ2_variant][alt(1-{metric})][{ds}] color map (seed): {mapping}")
                leg = ax.get_legend()
                if leg is not None:
                    leg.remove()
                for s in ax.spines.values():
                    s.set_visible(True)
                ax.margins(x=0.02, y=0.02)
                # Add N/A text for RGCN + FB15k-237 in alt plot
                if ds == "FB15k-237":
                    x_models = [m for m in MODEL_ORDER if m in sub_alt["model"].cat.categories]
                    rgcn_idx = x_models.index("RGCN") if "RGCN" in x_models else -1
                    if rgcn_idx >= 0:
                        seed_cats = list(sub_alt["seed"].cat.categories)
                        num_seeds = len(seed_cats)
                        bar_start_idx = rgcn_idx * num_seeds
                        if bar_start_idx < len(ax.patches):
                            first_patch = ax.patches[bar_start_idx]
                            last_patch = ax.patches[min(bar_start_idx + num_seeds - 1, len(ax.patches) - 1)]
                            x_pos = (first_patch.get_x() + last_patch.get_x() + last_patch.get_width()) / 2
                            y_pos = ax.get_ylim()[1] * 0.05
                            ax.text(x_pos, y_pos, "N/A", ha='center', va='bottom', fontsize=12)
                fname = os.path.join(output_dir, f"rq2_variant_alt_{metric.replace('@','at').replace('/','_')}_{ds}.png")
                plt.savefig(fname, bbox_inches='tight', pad_inches=0.03)
                plt.close()
                print(f"Saved {fname}")



def make_rq3_variant(output_dir: str = "process_data/RQ3_variant"):
    os.makedirs(output_dir, exist_ok=True)
    all_data = load_all()
    rows = _collect_flat_records(all_data)
    df = pd.DataFrame(rows)

    metrics = RQ3_VARIANT_METRICS
    # Select seed=all and only best/median/worst and filter datasets to exclude kinship and nations
    dfa = df[(df["seed"] == "all") & (df["quality"].isin(["best", "median", "worst"]))].copy()
    dfa = dfa[dfa["dataset"].isin(["FB15k-237", "WN18RR", "codex-s"])]

    for metric in metrics:
        sub = dfa[dfa["metric"] == metric].copy()
        if sub.empty:
            continue
        sub["pair"] = sub["model"] + " | " + sub["dataset"]
        sub["mean"] = pd.to_numeric(sub["mean"], errors="coerce")
        pivot = sub.pivot_table(index="pair", columns="quality", values="mean")
        # Skip if pivot is empty or all NaN
        if pivot.empty or pivot.isna().all().all():
            print(f"Skipping {metric} - no valid data")
            continue
        def pair_key(p):
            model = p.split(" | ")[0]
            return HEATMAP_MODEL_ORDER.index(model) if model in HEATMAP_MODEL_ORDER else len(HEATMAP_MODEL_ORDER)
        pivot = pivot.reindex(sorted(pivot.index, key=pair_key))
        pivot = pivot.reindex(columns=["best", "median", "worst"])
        # Fixed height for all heatmaps, adjust width to maintain aspect ratio
        num_rows = len(pivot)
        fig_height = 8.0  # Fixed height for all plots
        fig_width = 3.5  # Narrower width to fit 3 side by side
        plt.figure(figsize=(fig_width, fig_height))
        
        # Determine if we should hide colorbar and/or y-labels
        hide_cbar = metric == "pred_overlap@10"
        hide_cbar_and_ylabel = "space_overlap@10_mean" in metric
        hide_ylabel_only = "super_pred_jaccard@10_mean" in metric
        
        if "jaccard" in metric or "overlap" in metric:
            # Anchored continuous colormap: black at 0, red at 0.5, yellow at 0.85, green at 1.0
            positions = [0.0, 0.5, 0.85, 1.0]
            colors = ["black", "red", "yellow", "green"]
            cmap = LinearSegmentedColormap.from_list("jaccard_fixed", list(zip(positions, colors)))
            # Hide colorbar for all except the first plot to maintain uniform width
            show_cbar = not (hide_cbar or hide_cbar_and_ylabel or hide_ylabel_only)
            sns.heatmap(pivot, annot=True, fmt=".3f", cmap=cmap, vmin=0.0, vmax=1.0, 
                       cbar=show_cbar,
                       annot_kws={"size": 13}, cbar_kws={"shrink": 0.8})
        elif "rbo" in metric:
            sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn", vmin=0.0, vmax=1.0, cbar=False,
                       annot_kws={"size": 13})
        else:
            sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn",
                       annot_kws={"size": 13}, cbar_kws={"shrink": 0.8})
        plt.title("")
        plt.xlabel("", fontsize=13)
        
        # Hide y-labels for specific metrics
        if hide_cbar_and_ylabel or hide_ylabel_only:
            plt.ylabel("")
            plt.yticks([])
        else:
            plt.ylabel("", fontsize=13)
            plt.yticks(fontsize=12)
        plt.xticks(fontsize=13)
            
        fname = os.path.join(output_dir, f"rq3_variant_{metric.replace('@','at').replace('/','_')}.png")
        plt.savefig(fname, bbox_inches='tight', pad_inches=0.01)
        plt.close()
        print(f"Saved {fname}")


def make_other_metrics(output_dir: str = "process_data/other_metrics"):
    os.makedirs(output_dir, exist_ok=True)
    all_data = load_all()
    rows = _collect_flat_records(all_data)
    df = pd.DataFrame(rows)

    filt = (df["quality"] == "best") & (df["seed"] == "all")
    dfb = df[filt]

    metrics = OTHER_METRICS
    for metric in metrics:
        dmf = dfb[dfb["metric"] == metric].copy()
        # Set RGCN + FB15k-237 to NaN
        mask = (dmf["model"] == "RGCN") & (dmf["dataset"] == "FB15k-237")
        dmf.loc[mask, "mean"] = np.nan
        dmf.loc[mask, "std"] = np.nan
        
        dmf["mean"] = pd.to_numeric(dmf["mean"], errors="coerce")
        dmf = dmf.dropna(subset=["mean"]) 
        if dmf.empty:
            continue
        plt.figure(figsize=(10, 4))
        dmf["model"] = pd.Categorical(dmf["model"], categories=MODEL_ORDER, ordered=True)
        dmf["dataset"] = pd.Categorical(dmf["dataset"], categories=DATASET_ORDER, ordered=True)
        dmf.sort_values(["model", "dataset"], inplace=True)
        
        # For ambiguity, plot 1 - value
        if "ambiguity" in metric:
            dmf["mean"] = 1.0 - dmf["mean"]
            
        ax = sns.barplot(data=dmf, x="model", y="mean", hue="dataset", palette=PALETTE, errorbar=None)
        dmf["std"] = pd.to_numeric(dmf["std"], errors="coerce")
        x_models = [c for c in MODEL_ORDER if c in dmf["model"].cat.categories]
        h_datasets = [c for c in DATASET_ORDER if c in dmf["dataset"].cat.categories]
        std_list = []
        for m in x_models:
            subm = dmf[dmf["model"] == m]
            for d in h_datasets:
                row = subm[subm["dataset"] == d]
                if not row.empty:
                    std_list.append(row.iloc[0]["std"])
        for patch, yerr in zip(ax.patches, std_list):
            if pd.notna(yerr) and yerr > 0:
                x = patch.get_x() + patch.get_width() / 2
                y = patch.get_height()
                ax.errorbar(x, y, yerr=yerr, fmt='none', ecolor='black', capsize=3, lw=1)
        if ("jaccard" in metric) or ("rbo" in metric) or ("overlap" in metric) or ("ambiguity" in metric):
            ax.set_ylim(0.0, 1.0)
        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel("")
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
        ax.tick_params(axis='x', labelsize=14)
        for s in ax.spines.values():
            s.set_visible(True)
        ax.margins(x=0.02, y=0.02)
        fname = os.path.join(output_dir, f"rq1_{metric.replace('@','at').replace('/','_')}.png")
        if "ambiguity" in metric:
            fname = os.path.join(output_dir, f"rq1_alt_{metric.replace('@','at').replace('/','_')}.png")
        plt.savefig(fname, bbox_inches='tight', pad_inches=0.03)
        plt.close()
        print(f"Saved {fname}")


def make_rq3_all(output_dir: str = "process_data/rq3_all"):
    os.makedirs(output_dir, exist_ok=True)
    all_data = load_all()
    rows = _collect_flat_records(all_data)
    df = pd.DataFrame(rows)

    dfa = df[(df["seed"] == "all") & (df["quality"].isin(["best", "median", "worst"]))].copy()
    dfa = dfa[dfa["dataset"].isin(["FB15k-237", "WN18RR", "codex-s"])]

    # Mapping for output filenames
    metric_names = {
        "pred_overlap@10": "Consistancy_at10_mean",
        "super_pred_overlap@10_mean": "Homogeneity_at10_mean",
        "space_overlap@10_mean": "space_overlapat10_mean",
        "mrr_mean": "MRR",
        "hit@1_mean": "Hit@1",
        "hit@10_mean": "Hit@10",
    }
    
    metrics_to_plot = [
        "pred_overlap@10",
        "super_pred_overlap@10_mean",
        "pred_overlap@10",  # duplicate for pred_overlapat10
        "space_overlap@10_mean",
        "mrr_mean",
        "hit@1_mean",
        "hit@10_mean",
    ]
    
    for i, metric in enumerate(metrics_to_plot):
        sub = dfa[dfa["metric"] == metric].copy()
        if sub.empty:
            continue
        sub["pair"] = sub["model"] + " | " + sub["dataset"]
        sub["mean"] = pd.to_numeric(sub["mean"], errors="coerce")
        pivot = sub.pivot_table(index="pair", columns="quality", values="mean")
        if pivot.empty or pivot.isna().all().all():
            continue
        def pair_key(p):
            model = p.split(" | ")[0]
            return HEATMAP_MODEL_ORDER.index(model) if model in HEATMAP_MODEL_ORDER else len(HEATMAP_MODEL_ORDER)
        pivot = pivot.reindex(sorted(pivot.index, key=pair_key))
        pivot = pivot.reindex(columns=["best", "median", "worst"])
        
        # Larger figure with bigger fonts
        plt.figure(figsize=(6, max(4, 0.5 * len(pivot))))
        
        if "jaccard" in metric or "overlap" in metric or "Homogeneity" in metric_names.get(metric, "") or "Consistancy" in metric_names.get(metric, ""):
            positions = [0.0, 0.5, 0.85, 1.0]
            colors = ["black", "red", "yellow", "green"]
            cmap = LinearSegmentedColormap.from_list("jaccard_fixed", list(zip(positions, colors)))
            sns.heatmap(pivot, annot=True, fmt=".3f", cmap=cmap, vmin=0.0, vmax=1.0, 
                       annot_kws={"size": 12}, cbar_kws={"shrink": 0.8})
        elif "rbo" in metric:
            sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn", vmin=0.0, vmax=1.0, 
                       annot_kws={"size": 12}, cbar_kws={"shrink": 0.8})
        else:
            sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn", 
                       annot_kws={"size": 12}, cbar_kws={"shrink": 0.8})
        
        plt.title("")
        plt.ylabel("", fontsize=12)
        plt.xlabel("", fontsize=12)
        plt.yticks(fontsize=11)
        plt.xticks(fontsize=12)
        
        # Use metric_names mapping for filename
        if i == 2:  # Third occurrence of pred_overlap@10
            fname_metric = "pred_overlapat10"
        else:
            fname_metric = metric_names.get(metric, metric.replace("@", "at").replace("/", "_"))
        fname = os.path.join(output_dir, f"rq3_variant_{fname_metric}.png")
        plt.savefig(fname, bbox_inches='tight', pad_inches=0.01)
        plt.close()
        print(f"Saved {fname}")


def make_rq3_variant_with_colorbar(output_dir: str = "process_data/RQ3_variant"):
    """Version alternative avec colorbar pour super_pred_jaccard"""
    os.makedirs(output_dir, exist_ok=True)
    all_data = load_all()
    rows = _collect_flat_records(all_data)
    df = pd.DataFrame(rows)

    # Select seed=all and only best/median/worst and filter datasets to exclude kinship and nations
    dfa = df[(df["seed"] == "all") & (df["quality"].isin(["best", "median", "worst"]))].copy()
    dfa = dfa[dfa["dataset"].isin(["FB15k-237", "WN18RR", "codex-s"])]

    metrics = RQ3_VARIANT_METRICS
    for metric in metrics:
        # Only generate for super_pred_jaccard with colorbar
        if "super_pred_jaccard" not in metric:
            continue
            
        sub = dfa[dfa["metric"] == metric].copy()
        if sub.empty:
            continue
        sub["pair"] = sub["model"] + " | " + sub["dataset"]
        sub["mean"] = pd.to_numeric(sub["mean"], errors="coerce")
        pivot = sub.pivot_table(index="pair", columns="quality", values="mean")
        if pivot.empty or pivot.isna().all().all():
            print(f"Skipping {metric} - no valid data")
            continue
        def pair_key(p):
            model = p.split(" | ")[0]
            return HEATMAP_MODEL_ORDER.index(model) if model in HEATMAP_MODEL_ORDER else len(HEATMAP_MODEL_ORDER)
        pivot = pivot.reindex(sorted(pivot.index, key=pair_key))
        pivot = pivot.reindex(columns=["best", "median", "worst"])
        
        # Fixed height for all heatmaps, adjust width to maintain aspect ratio
        num_rows = len(pivot)
        fig_height = 8.0
        fig_width = 4.2  # Slightly wider to accommodate colorbar
        plt.figure(figsize=(fig_width, fig_height))
        
        # Anchored continuous colormap
        positions = [0.0, 0.5, 0.85, 1.0]
        colors = ["black", "red", "yellow", "green"]
        cmap = LinearSegmentedColormap.from_list("jaccard_fixed", list(zip(positions, colors)))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap=cmap, vmin=0.0, vmax=1.0, 
                   cbar=True,  # Show colorbar
                   annot_kws={"size": 13}, cbar_kws={"shrink": 0.8})
        
        plt.title("")
        plt.xlabel("", fontsize=13)
        plt.ylabel("", fontsize=13)
        plt.yticks(fontsize=12)
        plt.xticks(fontsize=13)
            
        fname = os.path.join(output_dir, f"rq3_variant_{metric.replace('@','at').replace('/','_')}_with_colorbar.png")
        plt.savefig(fname, bbox_inches='tight', pad_inches=0.01)
        plt.close()
        print(f"Saved {fname}")


def generate_colorbar_only(output_dir: str = "process_data/RQ3_variant"):
    """Génère uniquement le colorbar comme image séparée"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a figure with just the colorbar
    fig = plt.figure(figsize=(1.5, 8.0))
    ax = fig.add_axes([0.05, 0.05, 0.25, 0.9])
    
    # Create the same colormap as used in heatmaps
    positions = [0.0, 0.5, 0.85, 1.0]
    colors = ["black", "red", "yellow", "green"]
    cmap = LinearSegmentedColormap.from_list("jaccard_fixed", list(zip(positions, colors)))
    
    # Create a colorbar
    norm = plt.Normalize(vmin=0.0, vmax=1.0)
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), 
                     cax=ax, orientation='vertical')
    cb.ax.tick_params(labelsize=13)
    
    fname = os.path.join(output_dir, "colorbar_only.png")
    plt.savefig(fname, bbox_inches='tight', pad_inches=0.01)
    plt.close()
    print(f"Saved {fname}")


if __name__ == "__main__":
    make_rq1()
    make_rq2_variant()
    make_rq3_variant()
    make_other_metrics()
    make_rq3_all()
    make_rq3_variant_with_colorbar()
    generate_colorbar_only()