"""
ablation_voting_visualization.py

Visualization module for ablation voting results.
Reads JSON files and generates plots.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Optional


# Custom palette: [284B63, D86E6E, 0077B6, F4978E, 90E0EF, FBC4AB]
COLORS_METHODS = {
    'borda': '#284B63',     # dark blue
    'norm': '#D86E6E',      # coral red
    'control': '#FBC4AB'    # light peach
}

COLORS_COMPARATIVE = {
    'random_control': '#FBC4AB',  # light peach
    'borda': '#284B63',           # dark blue
    'norm': '#D86E6E'             # coral red
}

# Model order (same as visualization.py)
MODEL_ORDER = ["TransE", "ConvE", "DistMult", "Transformer", "RGCN", "ComplEx", "RotatE"]

K_VALUES = [1, 5, 10]


def load_ablation_json(json_path: str) -> Dict:
    """Load ablation voting JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def load_all500_data(dataset: str, model: str, stability_dir: str = None, use_ablation_file: bool = False, ablation_result: Dict = None) -> Optional[Dict]:
    """
    Load all_500 data from the corresponding stability.json file OR from ablation voting result.
    
    Args:
        dataset: Dataset name (e.g., 'FB15k-237', 'WN18RR', 'codex-s')
        model: Model name (e.g., 'ConvE', 'DistMult', etc.)
        stability_dir: Directory containing stability JSON files (default: auto-detect)
        use_ablation_file: If True, use data from ablation_result instead of stability.json
        ablation_result: Dictionary containing ablation voting results (required if use_ablation_file=True)
    
    Returns:
        Dictionary with the same structure as random_control, or None if file not found
    """
    
    # If using ablation file data
    if use_ablation_file:
        if ablation_result is None:
            print(f"[WARNING] use_ablation_file=True but no ablation_result provided")
            return None
        
        # Try all_500_control first, then init_control as fallback
        for control_key in ['all_500_control', 'init_control']:
            if control_key in ablation_result and ablation_result[control_key] is not None:
                control_data = ablation_result[control_key]
                print(f"[INFO] Using {control_key} data from ablation voting JSON")
                return control_data
        
        print(f"[WARNING] No all_500_control or init_control found in ablation result")
        return None
    if stability_dir is None:
        # Auto-detect: look in ../stability_measures_results/ relative to the ablation voting file
        # or in the common pattern: XX_YY/stability_measures_results/
        possible_dirs = [
            f"01_12/stability_measures_results",
            f"25_11/stability_measures_results",
            f"30_10/stability_measures_results",
            f"24_09/stability_measures_results",
            f"07_09"
        ]
        
        stability_path = None
        for dir_path in possible_dirs:
            candidate = os.path.join(dir_path, f"{dataset}_{model}_stability.json")
            if os.path.exists(candidate):
                stability_path = candidate
                break
        
        if stability_path is None:
            print(f"[WARNING] Could not find stability file for {dataset}_{model}")
            return None
    else:
        stability_path = os.path.join(stability_dir, f"{dataset}_{model}_stability.json")
        if not os.path.exists(stability_path):
            print(f"[WARNING] Stability file not found: {stability_path}")
            return None
    
    print(f"[INFO] Loading all_500 data from {stability_path}")
    
    try:
        with open(stability_path, 'r') as f:
            stability_data = json.load(f)
        
        # Extract data from ["best"]["all"]["500"]["prediction_metrics"]
        all500_metrics = stability_data["best"]["all"]["500"]["prediction_metrics"]
        
        # Format to match random_control structure
        formatted_data = {
            "mean_of_rank_stds": all500_metrics.get("mean_of_rank_stds", 0),
            "std_of_rank_stds": all500_metrics.get("std_of_rank_stds", 0),
            "ambiguity@1": all500_metrics.get("ambiguity@1", 0),
            "ambiguity@5": all500_metrics.get("ambiguity@5", 0),
            "ambiguity@10": all500_metrics.get("ambiguity@10", 0),
            "discrepancy@1": all500_metrics.get("discrepancy@1", 0),
            "discrepancy@5": all500_metrics.get("discrepancy@5", 0),
            "discrepancy@10": all500_metrics.get("discrepancy@10", 0),
            "pred_jaccard@1": all500_metrics.get("pred_jaccard@1", 0),
            "pred_jaccard@1_std": all500_metrics.get("pred_jaccard@1_std", 0),
            "pred_overlap@1": all500_metrics.get("pred_overlap@1", 0),
            "pred_overlap@1_std": all500_metrics.get("pred_overlap@1_std", 0),
            "pred_rbo@1": all500_metrics.get("pred_rbo@1", 0),
            "pred_rbo@1_std": all500_metrics.get("pred_rbo@1_std", 0),
            "pred_jaccard@10": all500_metrics.get("pred_jaccard@10", 0),
            "pred_jaccard@10_std": all500_metrics.get("pred_jaccard@10_std", 0),
            "pred_overlap@10": all500_metrics.get("pred_overlap@10", 0),
            "pred_overlap@10_std": all500_metrics.get("pred_overlap@10_std", 0),
            "pred_rbo@10": all500_metrics.get("pred_rbo@10", 0),
            "pred_rbo@10_std": all500_metrics.get("pred_rbo@10_std", 0),
            "pred_kl@1": all500_metrics.get("pred_kl@1", 0),
            "pred_kl@1_std": all500_metrics.get("pred_kl@1_std", 0),
            "super_pred_jaccard@1_mean": all500_metrics.get("super_pred_jaccard@1_mean", 0),
            "super_pred_jaccard@1_std": all500_metrics.get("super_pred_jaccard@1_std", 0),
            "super_pred_overlap@1_mean": all500_metrics.get("super_pred_overlap@1_mean", 0),
            "super_pred_overlap@1_std": all500_metrics.get("super_pred_overlap@1_std", 0),
        }
        
        # Add per-K data if available (for comparative distributions)
        if "super_pred_jaccards_per_k" in all500_metrics:
            formatted_data["super_jaccards_per_k"] = all500_metrics["super_pred_jaccards_per_k"]
        if "super_pred_overlaps_per_k" in all500_metrics:
            formatted_data["super_overlaps_per_k"] = all500_metrics["super_pred_overlaps_per_k"]
        
        # Add histograms if available
        if "homogeneity@1_histogram" in all500_metrics:
            formatted_data["homogeneity@1_histogram"] = all500_metrics["homogeneity@1_histogram"]
        if "consistency@1_histogram" in all500_metrics:
            formatted_data["consistency@1_histogram"] = all500_metrics["consistency@1_histogram"]
        
        return formatted_data
        
    except Exception as e:
        print(f"[ERROR] Failed to load all_500 data from {stability_path}: {e}")
        return None


def create_comparative_distribution_plots(all_results: Dict, output_dir: str, use_all500: bool = True, use_ablation_file: bool = False):
    """
    Create comparative distribution plots showing all voting methods together.
    One plot for jaccard, one for overlap, with subplots for each K value.
    Each subplot shows overlapping histograms for all voting methods.
    
    Args:
        all_results: Dictionary mapping model names to their results
        output_dir: Output directory for plots
        use_all500: If True, use all_500 data from stability.json instead of random_control
        use_ablation_file: If True, use control data from ablation JSON instead of stability.json
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if all_results is already the inner dict (single model file)
    # or if it's the outer dict (multiple models)
    if 'aggregation_results' in all_results:
        # Single model file - wrap it with the model name
        model_name = all_results.get('model', 'unknown')
        model_data = {model_name: all_results}
    else:
        # Multiple models - extract valid model data
        model_data = {}
        for model_name, result in all_results.items():
            if isinstance(result, dict) and 'aggregation_results' in result:
                model_data[model_name] = result
    
    for model_name, result in model_data.items():
        
        data_name = result.get('dataset', 'unknown')
        
        # Collect super_pred data for all methods (including random_control or all_500)
        methods_jaccard_data = {}  # method -> {k -> [values]}
        methods_overlap_data = {}  # method -> {k -> [values]}
        
        # Add control data (either all_500, init_control, or random_control)
        control_data = None
        if use_all500:
            control_data = load_all500_data(data_name, model_name, use_ablation_file=use_ablation_file, ablation_result=result)
        
        if control_data is None:
            # Fallback to random_control
            control_data = result.get('random_control', {})
        
        if 'super_jaccards_per_k' in control_data:
            methods_jaccard_data['random_control'] = control_data['super_jaccards_per_k']
        if 'super_overlaps_per_k' in control_data:
            methods_overlap_data['random_control'] = control_data['super_overlaps_per_k']
        
        # Add aggregation methods
        for method_name, agg_result in result['aggregation_results'].items():
            if 'super_jaccards_per_k' in agg_result:
                methods_jaccard_data[method_name] = agg_result['super_jaccards_per_k']
            if 'super_overlaps_per_k' in agg_result:
                methods_overlap_data[method_name] = agg_result['super_overlaps_per_k']
        
        if not methods_jaccard_data:
            print(f"[INFO] No per-query data found for {model_name}, skipping comparative plots")
            continue
        
        # Create comparative Jaccard plot
        fig, axes = plt.subplots(1, len(K_VALUES), figsize=(5 * len(K_VALUES), 4))
        if len(K_VALUES) == 1:
            axes = [axes]
        
        for idx, k in enumerate(K_VALUES):
            ax = axes[idx]
            k_str = str(k)  # Convert to string for JSON key lookup
            
            for method_name in ['random_control', 'borda', 'norm']:
                if method_name in methods_jaccard_data and k_str in methods_jaccard_data[method_name]:
                    values = methods_jaccard_data[method_name][k_str]
                    label = 'Control' if method_name == 'random_control' else method_name.capitalize()
                    ax.hist(values, bins=30, alpha=0.5, color=COLORS_COMPARATIVE[method_name], 
                           label=label, edgecolor='black', linewidth=0.5)
            
            ax.set_xlabel(f'K={k}', fontsize=12)
            if idx == 0:
                ax.set_ylabel('Frequency', fontsize=12)
            else:
                ax.set_yticklabels([])
            
            if idx == len(K_VALUES) - 1:
                ax.legend(fontsize=18, loc='upper left')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        jaccard_comp_path = os.path.join(output_dir, f"{model_name}_{data_name}_comparative_super_pred_jaccard_distribution.png")
        plt.savefig(jaccard_comp_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Saved comparative Jaccard plot: {jaccard_comp_path}")
        
        # Create comparative Overlap plot
        fig, axes = plt.subplots(1, len(K_VALUES), figsize=(5 * len(K_VALUES), 4))
        if len(K_VALUES) == 1:
            axes = [axes]
        
        for idx, k in enumerate(K_VALUES):
            ax = axes[idx]
            k_str = str(k)  # Convert to string for JSON key lookup
            
            for method_name in ['random_control', 'borda', 'norm']:
                if method_name in methods_overlap_data and k_str in methods_overlap_data[method_name]:
                    values = methods_overlap_data[method_name][k_str]
                    label = 'Control' if method_name == 'random_control' else method_name.capitalize()
                    ax.hist(values, bins=30, alpha=0.5, color=COLORS_COMPARATIVE[method_name], 
                           label=label, edgecolor='black', linewidth=0.5)
            
            ax.set_xlabel(f'K={k}', fontsize=12)
            if idx == 0:
                ax.set_ylabel('Frequency', fontsize=12)
            else:
                ax.set_yticklabels([])
            
            if idx == len(K_VALUES) - 1:
                ax.legend(fontsize=14, loc='upper left')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        overlap_comp_path = os.path.join(output_dir, f"{model_name}_{data_name}_comparative_super_pred_overlap_distribution.png")
        plt.savefig(overlap_comp_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Saved comparative Overlap plot: {overlap_comp_path}")


def create_grouped_by_dataset_plots(all_results_list: List[Dict], output_dir: str, use_all500: bool = True, use_ablation_file: bool = False):
    """
    Create plots grouped by dataset, with each model having its own grouped bars.
    One plot per metric per dataset.
    
    Args:
        all_results_list: List of result dictionaries from all JSON files
        output_dir: Output directory for plots
        use_all500: If True, use all_500 data from stability.json instead of random_control
        use_ablation_file: If True, use control data from ablation JSON instead of stability.json
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Organize data by dataset
    data_by_dataset = {}  # dataset -> [(model, result_dict), ...]
    
    for result_dict in all_results_list:
        # Handle both formats
        if 'aggregation_results' in result_dict:
            # Single model format
            dataset = result_dict.get('dataset', 'unknown')
            model = result_dict.get('model', 'unknown')
            if dataset not in data_by_dataset:
                data_by_dataset[dataset] = []
            data_by_dataset[dataset].append((model, result_dict))
    
    if not data_by_dataset:
        print("[INFO] No data found for grouped by dataset plots")
        return
    
    # Metrics to plot
    metrics_to_plot = [
        'pred_jaccard@1', 'pred_jaccard@10', 'pred_rbo@10',
        'pred_overlap@1', 'pred_overlap@10'
    ]
    
    # Aggregation methods
    aggregation_methods = ['control', 'borda', 'norm']
    
    # Create plots for each dataset
    for dataset, models_data in data_by_dataset.items():
        # Sort models according to MODEL_ORDER
        def model_sort_key(item):
            model_name = item[0]
            return MODEL_ORDER.index(model_name)
        
        
        models_data = sorted(models_data, key=model_sort_key)
        model_names = [m[0] for m in models_data]
        
        for metric in metrics_to_plot:
            fig, ax = plt.subplots(figsize=(12, 5))
            
            # Prepare data
            data_rows = []
            for model_name, result in models_data:
                # Add control metrics (either all_500, init_control, or random_control)
                control_data = None
                if use_all500:
                    control_data = load_all500_data(dataset, model_name, use_ablation_file=use_ablation_file, ablation_result=result)
                
                if control_data is None:
                    # Fallback to random_control
                    control_data = result.get('random_control', {})
                
                if metric in control_data:
                    std_key = f"{metric}_std"
                    std = control_data.get(std_key, 0)
                    data_rows.append({
                        'model': model_name,
                        'method': 'control',
                        'value': control_data[metric],
                        'std': std
                    })
                
                # Add aggregation methods
                if 'aggregation_results' in result:
                    for method in ['borda', 'norm']:
                        if method in result['aggregation_results']:
                            agg_result = result['aggregation_results'][method]
                            if metric in agg_result:
                                std_key = f"{metric}_std"
                                std = agg_result.get(std_key, 0)
                                data_rows.append({
                                    'model': model_name,
                                    'method': method,
                                    'value': agg_result[metric],
                                    'std': std
                                })
            
            if not data_rows:
                plt.close()
                continue
            
            df = pd.DataFrame(data_rows)
            
            # Create grouped bar plot
            x_pos = np.arange(len(model_names))
            width = 0.25  # Width of each bar
            
            for i, method in enumerate(aggregation_methods):
                df_method = df[df['method'] == method]
                if df_method.empty:
                    continue
                
                values = []
                stds = []
                for model in model_names:
                    row = df_method[df_method['model'] == model]
                    if not row.empty:
                        values.append(row.iloc[0]['value'])
                        stds.append(row.iloc[0]['std'])
                    else:
                        values.append(0)
                        stds.append(0)
                
                offset = (i - 1) * width
                ax.bar(x_pos + offset, values, width, 
                       label=method.capitalize(),
                       color=COLORS_METHODS[method], 
                       yerr=stds, capsize=3, ecolor='black')
            
            # Customize plot - keep only model names on x-axis
            ax.set_xlabel('')  # Remove x-axis label
            ax.set_ylabel('')  # Remove y-axis label
            ax.set_title('')   # Remove title
            ax.set_xticks(x_pos)
            ax.set_xticklabels(model_names, fontsize=18)
            
            # Remove legend
            if ax.get_legend():
                ax.get_legend().remove()
            
            ax.grid(True, alpha=0.3, axis='y')
            
            # Set y-axis limits for metrics in [0, 1]
            if 'jaccard' in metric or 'rbo' in metric or 'overlap' in metric:
                ax.set_ylim(0.0, 1.0)
            
            plt.tight_layout()
            
            # Save plot
            metric_name = metric.replace('@', 'at').replace('/', '_')
            output_path = os.path.join(output_dir, f"{dataset}_{metric_name}_grouped.png")
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"[INFO] Saved grouped plot: {output_path}")


def create_voting_visualizations(all_results: Dict, output_dir: str, use_all500: bool = True, use_ablation_file: bool = False):
    """
    Create visualizations for ablation voting results for a single model.
    
    Args:
        all_results: Result dictionary (assumes single model format)
        output_dir: Output directory for plots
        use_all500: If True, use all_500 data from stability.json instead of random_control
        use_ablation_file: If True, use control data from ablation JSON instead of stability.json
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Ensure we have the right format
    if 'aggregation_results' not in all_results:
        print("[INFO] No aggregation_results found, skipping voting visualizations")
        return
    
    model_name = all_results.get('model', 'unknown')
    
    # Metrics to plot
    metrics_to_plot = [
        'pred_jaccard@1', 'pred_jaccard@10', 'pred_rbo@10', 
        'pred_overlap@1', 'pred_overlap@10'
    ]
    
    # Aggregation methods
    aggregation_methods = ['control', 'borda', 'norm']
    
    for metric in metrics_to_plot:
        fig, ax = plt.subplots(figsize=(8, 4))
        
        # Collect data for this metric
        values = []
        stds = []
        labels = []
        colors = []
        
        # Add control (either all_500, init_control, or random_control)
        control_data = None
        if use_all500:
            dataset = all_results.get('dataset', 'unknown')
            model_name = all_results.get('model', 'unknown')
            control_data = load_all500_data(dataset, model_name, use_ablation_file=use_ablation_file, ablation_result=all_results)
        
        if control_data is None:
            # Fallback to random_control
            control_data = all_results.get('random_control', {})
        
        if metric in control_data:
            values.append(control_data[metric])
            stds.append(control_data.get(f"{metric}_std", 0))
            labels.append('Control')
            colors.append(COLORS_METHODS['control'])
        
        # Add borda and norm
        for method in ['borda', 'norm']:
            if method in all_results['aggregation_results']:
                agg_result = all_results['aggregation_results'][method]
                if metric in agg_result:
                    values.append(agg_result[metric])
                    stds.append(agg_result.get(f"{metric}_std", 0))
                    labels.append(method.capitalize())
                    colors.append(COLORS_METHODS[method])
        
        if not values:
            print(f"[INFO] No data for metric {metric}, skipping...")
            plt.close()
            continue
        
        # Create bar plot
        x_pos = np.arange(len(values))
        ax.bar(x_pos, values, yerr=stds, capsize=5, color=colors, 
               edgecolor='black', linewidth=1.2)
        
        # Customize plot
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_ylabel(metric.replace('@', ' @ ').replace('_', ' ').title(), fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Set y-axis limits for metrics in [0, 1]
        if 'jaccard' in metric or 'rbo' in metric or 'overlap' in metric:
            ax.set_ylim(0.0, 1.0)
        
        plt.tight_layout()
        
        # Save plot
        metric_name = metric.replace('@', 'at').replace('/', '_')
        output_path = os.path.join(output_dir, f"ablation_voting_{metric_name}.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[INFO] Saved {output_path}")


def generate_all_visualizations(json_path: str, output_dir: str = None, use_all500: bool = True, use_ablation_file: bool = False):
    """
    Generate all visualizations from an ablation voting JSON file.
    
    Args:
        json_path: Path to ablation voting JSON file
        output_dir: Output directory for plots (defaults to ablation_voting/)
        use_all500: If True, use all_500 data from stability.json instead of random_control (default: True)
        use_ablation_file: If True, use control data from ablation JSON instead of stability.json (default: False)
    """
    if output_dir is None:
        output_dir = 'ablation_voting'
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load JSON
    print(f"[INFO] Loading {json_path}")
    all_results = load_ablation_json(json_path)
    
    # Generate comparative distribution plots
    print(f"\n[INFO] Generating comparative distribution plots...")
    create_comparative_distribution_plots(all_results, output_dir, use_all500=use_all500, use_ablation_file=use_ablation_file)
    
    # Generate voting visualizations
    print(f"\n[INFO] Generating voting visualizations...")
    create_voting_visualizations(all_results, output_dir, use_all500=use_all500, use_ablation_file=use_ablation_file)
    
    print(f"\n[INFO] All visualizations saved to {output_dir}")


def process_directory(results_dir: str, output_dir: str = None, use_all500: bool = True, use_ablation_file: bool = False):
    """
    Process all ablation voting JSON files in a directory.
    
    Args:
        results_dir: Directory containing ablation voting JSON files
        output_dir: Output directory for plots (defaults to ablation_voting/)
        use_all500: If True, use all_500 data from stability.json instead of random_control (default: True)
        use_ablation_file: If True, use control data from ablation JSON instead of stability.json (default: False)
    """
    if output_dir is None:
        output_dir = 'ablation_voting'
    
    # Find all ablation voting JSON files
    json_files = [f for f in os.listdir(results_dir) if f.endswith('_ablation_voting.json')]
    
    print(f"[INFO] Found {len(json_files)} ablation voting JSON files in {results_dir}")
    
    # Collect all results for grouped plots
    all_results_list = []
    
    for json_file in json_files:
        json_path = os.path.join(results_dir, json_file)
        
        # Load and store for grouped plots
        result_data = load_ablation_json(json_path)
        all_results_list.append(result_data)
        
        # Create subdirectory for this file
        base_name = json_file.replace('_ablation_voting.json', '')
        file_output_dir = os.path.join(output_dir, base_name)
        
        print(f"\n[INFO] Processing {json_file}")
        generate_all_visualizations(json_path, file_output_dir, use_all500=use_all500, use_ablation_file=use_ablation_file)
    
    # Generate grouped by dataset plots
    print(f"\n[INFO] Generating grouped by dataset plots...")
    create_grouped_by_dataset_plots(all_results_list, output_dir, use_all500=use_all500, use_ablation_file=use_ablation_file)


if __name__ == '__main__':
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate visualizations for ablation voting results')
    parser.add_argument('--input_path', type=str, default='01_12/ablation_voting_results/',
                        help='Path to JSON file or directory containing ablation voting results')
    parser.add_argument('--output_dir', type=str, default='process_data/ablation_voting/',
                        help='Output directory for plots')
    parser.add_argument('--use_all500', action='store_true', default=True,
                        help='Use all_500 data from stability.json instead of random_control (default: True)')
    parser.add_argument('--use_random_control', dest='use_all500', action='store_false',
                        help='Use random_control data from ablation voting JSON instead of all_500')
    parser.add_argument('--use_ablation_file', action='store_true', default=False,
                        help='Use control data (all_500_control/init_control) from ablation JSON instead of stability.json (default: False)')
    
    args = parser.parse_args()
    
    input_path = args.input_path
    output_dir = args.output_dir
    use_all500 = args.use_all500
    use_ablation_file = args.use_ablation_file
    
    if use_ablation_file:
        print(f"[INFO] Using control data (all_500_control/init_control) from ablation voting JSON")
    else:
        print(f"[INFO] Using {'all_500 data from stability.json' if use_all500 else 'random_control data from ablation voting JSON'}")
    
    if os.path.isdir(input_path):
        # Process directory
        process_directory(input_path, output_dir, use_all500=use_all500, use_ablation_file=use_ablation_file)
    elif os.path.isfile(input_path):
        # Process single file
        generate_all_visualizations(input_path, output_dir, use_all500=use_all500, use_ablation_file=use_ablation_file)
    else:
        print(f"[ERROR] Path not found: {input_path}")
        sys.exit(1)
