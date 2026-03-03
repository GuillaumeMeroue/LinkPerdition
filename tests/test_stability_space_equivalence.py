"""
Test equivalence of different implementations of space metrics
"""
import numpy as np
import torch
import pytest
from stability_measures.stability_space import compute_all_neighbor_space_metrics as original_metrics
from stability_measures.stability_space_GPU import compute_all_neighbor_space_metrics as gpu_metrics

# Test constants
N_SAMPLES = 1000
H_DIM = 32
N_MODELS = 5
K_VALUES = [1, 5, 10]

@pytest.fixture(scope="module")
def test_embeddings():
    """Generate test embeddings."""
    np.random.seed(42)
    torch.manual_seed(42)
    # return [np.random.randn(N_SAMPLES, H_DIM).astype(np.float32) 
    #         for _ in range(N_MODELS)]
    x = np.random.randn(N_SAMPLES, H_DIM).astype(np.float32)
    return [x + 0.1*np.random.randn(N_SAMPLES, H_DIM).astype(np.float32) for _ in range(N_MODELS)]


def test_original_vs_optimized_consistency(test_embeddings):
    """Test that original and optimized CPU implementations of space metrics are consistent."""
    # Original implementation
    original_results = original_metrics(test_embeddings, K_VALUES)
    
        # GPU implementation (may include extra metrics)
    gpu_results = gpu_metrics(test_embeddings, K_VALUES, batch_size=128)
    
    # Compare results for each metric present in CPU implementation
    for key in original_results:

        assert key in gpu_results, f"Metric {key} missing in GPU results"
        orig_val = original_results[key]
        gpu_val = gpu_results[key]
        print(f"{key}: CPU={orig_val:.6f}, GPU={gpu_val:.6f}")
        # Allow small numerical differences due to different computation paths
        assert np.isclose(orig_val, gpu_val, rtol=1e-3, atol=1e-3), \
            f"Significant difference for {key}: CPU={orig_val} vs GPU={gpu_val}"
