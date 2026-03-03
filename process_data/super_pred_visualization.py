"""
Module to regenerate super_pred plots from JSON stability results.

This module reads JSON files containing histogram data for super_pred_jaccard and super_pred_overlap
metrics and regenerates the distribution plots with custom styling.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple


# Custom palette: [284B63, D86E6E, 0077B6, F4978E, 90E0EF, FBC4AB]
COLOR_MAIN = '#284B63'  # dark blue
COLOR_MEAN = '#D86E6E'  # coral red
COLOR_STD = '#F4978E'   # light coral


def load_json_file(json_path: str) -> Dict:
    """Load JSON file and return the data. Returns None if file is corrupted."""
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"[ERROR] Corrupted JSON file: {json_path}")
        print(f"        {e}")
        return None


def extract_histogram_data(histogram_dict: Dict) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Extract histogram data from JSON dictionary.
    
    Args:
        histogram_dict: Dictionary containing 'bins', 'frequencies', 'mean', 'std'
    
    Returns:
        Tuple of (bins, frequencies, mean, std)
    """
    bins = np.array(histogram_dict['bins'])
    frequencies = np.array(histogram_dict['frequencies'])
    mean = histogram_dict['mean']
    std = histogram_dict['std']
    return bins, frequencies, mean, std


def reconstruct_values_from_histogram(bins: np.ndarray, frequencies: np.ndarray) -> np.ndarray:
    """
    Reconstruct approximate individual values from histogram bins and frequencies.
    
    Args:
        bins: Array of bin edges
        frequencies: Array of frequencies for each bin
    
    Returns:
        Array of reconstructed values
    """
    values = []
    for i in range(len(frequencies)):
        if frequencies[i] > 0:
            # Use bin center as representative value
            if i < len(bins) - 1:
                bin_center = (bins[i] + bins[i+1]) / 2
            else:
                bin_center = bins[i]
            values.extend([bin_center] * int(frequencies[i]))
    return np.array(values)


def plot_super_pred_from_histogram(bins: np.ndarray, frequencies: np.ndarray, 
                                   mean: float, std: float, 
                                   metric_name: str, k: int, output_path: str,
                                   minimal: bool = False, n_queries: int = None):
    """
    Plot super_pred distribution from histogram data.
    
    Args:
        bins: Histogram bin edges
        frequencies: Histogram frequencies
        mean: Mean value
        std: Standard deviation
        metric_name: Name of metric (e.g., 'jaccard' or 'overlap')
        k: K value (e.g., 10)
        output_path: Path to save the plot
        minimal: If True, create minimal plot without legend/title
        n_queries: Number of queries (for normalization)
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Plot histogram using bar chart
    bin_centers = []
    for i in range(len(bins) - 1):
        bin_centers.append((bins[i] + bins[i+1]) / 2)
    
    if len(bin_centers) < len(frequencies):
        # Last bin
        bin_centers.append(bins[-1])
    
    bin_width = bins[1] - bins[0] if len(bins) > 1 else 0.02
    
    # Normalize frequencies if minimal and n_queries is provided
    if minimal and n_queries is not None and n_queries > 0:
        normalized_frequencies = frequencies / n_queries
        ax.bar(bin_centers, normalized_frequencies, width=bin_width * 0.9, 
               alpha=0.7, color=COLOR_MAIN, edgecolor='black')
    else:
        ax.bar(bin_centers, frequencies, width=bin_width * 0.9, 
               alpha=0.7, color=COLOR_MAIN, edgecolor='black')
    
    # Add mean and std lines (more visible for minimal)
    if minimal:
        ax.axvline(mean, color=COLOR_MEAN, linestyle='--', linewidth=3, alpha=0.9)
        ax.axvline(mean - std, color=COLOR_STD, linestyle=':', linewidth=2.5, alpha=0.9)
        ax.axvline(mean + std, color=COLOR_STD, linestyle=':', linewidth=2.5, alpha=0.9)
    else:
        ax.axvline(mean, color=COLOR_MEAN, linestyle='--', linewidth=2)
        ax.axvline(mean - std, color=COLOR_STD, linestyle=':', linewidth=1.5)
        ax.axvline(mean + std, color=COLOR_STD, linestyle=':', linewidth=1.5)
    
    if minimal:
        # Minimal styling: no xlabel, no ylabel, only numerical values
        ax.text(0.98, 0.98, f'Mean: {mean:.3f}\nStd: {std:.3f}', 
                transform=ax.transAxes, fontsize=12, verticalalignment='top', 
                horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        # Remove axis labels completely
        ax.set_xlabel('')
        ax.set_ylabel('')
    else:
        # Standard styling with legend
        ax.text(0.98, 0.98, f'Mean: {mean:.3f}\nStd: {std:.3f}', 
                transform=ax.transAxes, fontsize=12, verticalalignment='top', 
                horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.set_xlabel(f'{metric_name.capitalize()}@{k}', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'Super-Pred {metric_name.capitalize()}@{k} Distribution', fontsize=14)
    
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Plot saved to {output_path}")


def regenerate_super_pred_plots_from_json(json_path: str, output_dir: str = None, 
                                          quality: str = 'best', seed_type: str = 'all',
                                          k_values: List[int] = [1, 10],
                                          minimal: bool = False):
    """
    Regenerate super_pred plots from a JSON stability file.
    
    Args:
        json_path: Path to JSON file
        output_dir: Output directory for plots (defaults to same dir as JSON)
        quality: Quality level to extract ('best', 'median', 'worst')
        seed_type: Seed type to extract ('all', 'seed_init', etc.)
        k_values: List of K values to plot
        minimal: If True, create minimal plots without legend/title
    """
    # Load JSON
    data = load_json_file(json_path)
    
    # Skip if file is corrupted
    if data is None:
        return
    
    # Set output directory
    if output_dir is None:
        output_dir = os.path.dirname(json_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract base filename
    base_name = os.path.splitext(os.path.basename(json_path))[0]
    
    # Navigate to the correct section
    if quality not in data:
        print(f"[WARNING] Quality '{quality}' not found in JSON")
        return
    
    quality_data = data[quality]
    
    if seed_type not in quality_data:
        print(f"[WARNING] Seed type '{seed_type}' not found in quality '{quality}'")
        return
    
    seed_data = quality_data[seed_type]
    
    # Handle different JSON structures (with or without run_id)
    if '500' in seed_data:
        # Nested structure with run_id
        metrics_data = seed_data['500'].get('prediction_metrics', {})
    else:
        # Direct structure
        metrics_data = seed_data.get('prediction_metrics', {})
    
    if not metrics_data:
        print(f"[WARNING] No prediction_metrics found")
        return
    
    # Process each K value
    for k in k_values:
        # Jaccard
        jaccard_key = f'homogeneity@{k}_histogram'
        if jaccard_key in metrics_data:
            histogram = metrics_data[jaccard_key]
            bins, frequencies, mean, std = extract_histogram_data(histogram)
            n_queries = histogram.get('n_queries', None)
            
            suffix = '_minimal' if minimal else ''
            output_path = os.path.join(output_dir, 
                                      f'{base_name}_super_pred_jaccard_at{k}{suffix}.png')
            plot_super_pred_from_histogram(bins, frequencies, mean, std, 
                                          'jaccard', k, output_path, minimal, n_queries)
        
        # Overlap
        overlap_key = f'consistency@{k}_histogram'
        if overlap_key in metrics_data:
            histogram = metrics_data[overlap_key]
            bins, frequencies, mean, std = extract_histogram_data(histogram)
            n_queries = histogram.get('n_queries', None)
            
            suffix = '_minimal' if minimal else ''
            output_path = os.path.join(output_dir, 
                                      f'{base_name}_super_pred_overlap_at{k}{suffix}.png')
            plot_super_pred_from_histogram(bins, frequencies, mean, std, 
                                          'overlap', k, output_path, minimal, n_queries)


def regenerate_all_super_pred_plots(results_dir: str, output_dir: str = None,
                                    quality: str = 'best', seed_type: str = 'all',
                                    k_values: List[int] = [10],
                                    minimal: bool = True):
    """
    Regenerate super_pred plots for all JSON files in a directory.
    
    Args:
        results_dir: Directory containing JSON stability files
        output_dir: Output directory for plots (defaults to super_metrics/)
        quality: Quality level to extract
        seed_type: Seed type to extract
        k_values: List of K values to plot
        minimal: If True, create minimal plots
    """
    if output_dir is None:
        output_dir = 'super_metrics'
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all JSON files
    json_files = [f for f in os.listdir(results_dir) if f.endswith('_stability.json')]
    
    print(f"[INFO] Found {len(json_files)} JSON files in {results_dir}")
    
    for json_file in json_files:
        json_path = os.path.join(results_dir, json_file)
        print(f"\n[INFO] Processing {json_file}")
        regenerate_super_pred_plots_from_json(json_path, output_dir, quality, seed_type, k_values, minimal)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python super_pred_visualization.py <json_path_or_directory> [output_dir] [--minimal]")
        print("\nExample:")
        print("  python super_pred_visualization.py 25_11/stability_measures_results/")
        print("  python super_pred_visualization.py FB15k-237_RotatE_stability.json super_metrics/ --minimal")
        print("\nDefault output directory: super_metrics/")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else None
    minimal = '--minimal' in sys.argv
    
    if os.path.isdir(input_path):
        # Process directory
        regenerate_all_super_pred_plots(input_path, output_dir, minimal=minimal)
    elif os.path.isfile(input_path):
        # Process single file
        regenerate_super_pred_plots_from_json(input_path, output_dir, minimal=minimal)
    else:
        print(f"[ERROR] Path not found: {input_path}")
        sys.exit(1)
