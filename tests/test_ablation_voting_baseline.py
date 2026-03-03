"""
test_ablation_voting_baseline.py

Tests to verify that refactored aggregation functions in stability_measures/ablation_voting.py
produce the same results as the original baseline code.

This ensures backward compatibility and correctness of the refactored implementation.
"""

import os
import sys
import pytest
import torch
import pickle
import tempfile
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stability_measures.ablation_voting import (
    majority_aggregation,
    borda_aggregation,
    average_aggregation,
    norm_aggregation
)


# ============================================================================
# BASELINE IMPLEMENTATIONS (Original Code) + small change for using pickle
# ============================================================================

batch_size = 200  # Global variable used in baseline code


def majority_aggregation_baseline(score_paths):
    """Original majority aggregation implementation."""
    with open(score_paths[0], 'rb') as f:
        scores = pickle.load(f)
    total_samples = scores.shape[0]
    agg_scores = []
    for i in tqdm(range(0, total_samples, batch_size), desc="Baseline Majority"):
        score_tensor = []
        for score_path in score_paths:
            with open(score_path, 'rb') as f:
                scores = pickle.load(f)
            score_tensor.append(scores[i:i+batch_size, :])
        # aggregate scores
        score_tensor = torch.stack(score_tensor, dim=2).cuda() if torch.cuda.is_available() else torch.stack(score_tensor, dim=2)
        # average aggregation here
        _, max_indices = torch.max(score_tensor, dim=1, keepdim=True)
        trunk_scores = torch.zeros_like(score_tensor)
        trunk_scores.scatter_(dim=1, index=max_indices, value=1)
        trunk_scores = torch.mean(trunk_scores, dim=2)
        agg_scores.append(trunk_scores.cpu())
    return torch.cat(agg_scores)


def borda_aggregation_baseline(score_paths):
    """Original borda aggregation implementation."""
    def borda_scores(scores):
        """
        input: scores of multiple checkpoints (#batch*#entity*#rep)
        output: aggregated scores with voting scores
        """
        _, num_entities, _ = scores.shape
        sorted_indices = torch.argsort(scores, dim=1)
        range_tensor = torch.arange(1, num_entities + 1, device=scores.device, dtype=scores.dtype).view(1, -1, 1)
        borda_scores = torch.zeros_like(scores)
        borda_scores.scatter_(1, sorted_indices, range_tensor.expand_as(sorted_indices))
        sum_borda_scores = torch.sum(borda_scores, dim=2)
        return sum_borda_scores

    with open(score_paths[0], 'rb') as f:
        scores = pickle.load(f)
    total_samples = scores.shape[0]

    agg_scores = []
    for i in tqdm(range(0, total_samples, batch_size), desc="Baseline Borda"):
        score_tensor = []
        for score_path in score_paths:
            with open(score_path, 'rb') as f:
                scores = pickle.load(f)
            score_tensor.append(scores[i:i+batch_size, :])
        # aggregate scores
        score_tensor = torch.stack(score_tensor, dim=2).cuda() if torch.cuda.is_available() else torch.stack(score_tensor, dim=2)
        trunk_scores = borda_scores(score_tensor)
        agg_scores.append(trunk_scores.cpu())
    return torch.cat(agg_scores)


def average_aggregation_baseline(score_paths):
    """
    Original average aggregation implementation.
    Note: The original code had `*random.randint(0, 1000)` which is non-deterministic.
    For testing purposes, we remove this randomness.
    """
    with open(score_paths[0], 'rb') as f:
        scores = pickle.load(f)
    total_samples = scores.shape[0]

    agg_scores = []
    for i in tqdm(range(0, total_samples, batch_size), desc="Baseline Average"):
        score_tensor = []
        for score_path in score_paths:
            with open(score_path, 'rb') as f:
                scores = pickle.load(f)
            # NOTE: Original had *random.randint(0, 1000) - removed for deterministic testing
            score_tensor.append(scores[i:i+batch_size, :])
        # aggregate scores
        score_tensor = torch.stack(score_tensor, dim=2).cuda() if torch.cuda.is_available() else torch.stack(score_tensor, dim=2)
        # average aggregation here
        trunk_scores = torch.mean(score_tensor, dim=2)
        agg_scores.append(trunk_scores.cpu())
    return torch.cat(agg_scores)


def norm_aggregation_baseline(score_paths):
    """Original normalized aggregation implementation."""
    with open(score_paths[0], 'rb') as f:
        scores = pickle.load(f)
    total_samples = scores.shape[0]

    agg_scores = []
    for i in tqdm(range(0, total_samples, batch_size), desc="Baseline Norm"):
        score_tensor = []
        for score_path in score_paths:
            with open(score_path, 'rb') as f:
                scores = pickle.load(f)
            min_score = torch.min(scores)
            max_score = torch.max(scores)
            norm_scores = (scores[i:i+batch_size, :] - min_score)/(max_score-min_score)
            score_tensor.append(norm_scores)
        # aggregate scores
        score_tensor = torch.stack(score_tensor, dim=2).cuda() if torch.cuda.is_available() else torch.stack(score_tensor, dim=2)
        # average aggregation here
        trunk_scores = torch.mean(score_tensor, dim=2)
        agg_scores.append(trunk_scores.cpu())
    return torch.cat(agg_scores)


# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture
def synthetic_scores():
    """
    Create synthetic score tensors for testing.
    Returns paths to temporary pickle files.
    """
    num_samples = 500  # Small enough for fast testing
    num_entities = 100
    num_models = 5
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    score_paths = []
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    for i in range(num_models):
        # Generate random scores with different distributions per model
        scores = torch.randn(num_samples, num_entities) * (i + 1) + i * 10
        
        # Save to temporary file
        path = os.path.join(temp_dir, f'scores_model_{i}.pkl')
        with open(path, 'wb') as f:
            pickle.dump(scores, f)
        score_paths.append(path)
    
    yield score_paths
    
    # Cleanup
    for path in score_paths:
        if os.path.exists(path):
            os.remove(path)
    os.rmdir(temp_dir)


@pytest.fixture
def edge_case_scores():
    """
    Create edge case score tensors (ties, zeros, negatives).
    """
    num_samples = 100
    num_entities = 50
    num_models = 3
    
    temp_dir = tempfile.mkdtemp()
    score_paths = []
    
    torch.manual_seed(123)
    
    for i in range(num_models):
        # Create scores with edge cases
        scores = torch.zeros(num_samples, num_entities)
        
        # Add some patterns
        scores[:, 0] = 100.0  # Always highest
        scores[:, 1] = -100.0  # Always lowest
        scores[:, 2:10] = torch.randn(num_samples, 8)  # Random middle
        scores[:, 10:20] = 0.0  # Ties (all zeros)
        scores[:, 20:] = torch.randn(num_samples, 30) * 0.01  # Very small values
        
        path = os.path.join(temp_dir, f'edge_scores_model_{i}.pkl')
        with open(path, 'wb') as f:
            pickle.dump(scores, f)
        score_paths.append(path)
    
    yield score_paths
    
    # Cleanup
    for path in score_paths:
        if os.path.exists(path):
            os.remove(path)
    os.rmdir(temp_dir)


# ============================================================================
# COMPARISON TESTS
# ============================================================================

def compare_tensors(tensor1, tensor2, method_name, rtol=1e-5, atol=1e-6):
    """
    Compare two tensors and provide detailed error information.
    
    Args:
        tensor1: First tensor (baseline)
        tensor2: Second tensor (refactored)
        method_name: Name of the method being tested
        rtol: Relative tolerance
        atol: Absolute tolerance
    
    Returns:
        bool: True if tensors are close enough
    """
    
    # Check shapes
    assert tensor1.shape == tensor2.shape, f"{method_name}: Shape mismatch {tensor1.shape} vs {tensor2.shape}"
    

    is_close = torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol)
    
    if not is_close:
        diff = torch.abs(tensor1 - tensor2)
        max_diff = torch.max(diff).item()
        mean_diff = torch.mean(diff).item()
        
        print(f"\n{method_name} - Difference statistics:")
        print(f"  Max difference: {max_diff:.10f}")
        print(f"  Mean difference: {mean_diff:.10f}")
        print(f"  Relative tolerance: {rtol}")
        print(f"  Absolute tolerance: {atol}")
        
        # Show some examples of differences
        large_diffs = diff > atol
        if large_diffs.any():
            num_large = large_diffs.sum().item()
            print(f"  Number of large differences: {num_large} / {tensor1.numel()}")
            
            # Show first few large differences
            indices = torch.nonzero(large_diffs, as_tuple=False)[:5]
            for idx in indices:
                i, j = idx[0].item(), idx[1].item()
                print(f"    [{i}, {j}]: baseline={tensor1[i,j]:.6f}, refactored={tensor2[i,j]:.6f}, diff={diff[i,j]:.6f}")
    
    return is_close


@pytest.mark.parametrize("batch_size_override", [50, 100, 200])
def test_majority_aggregation(synthetic_scores, batch_size_override):
    """Test majority aggregation against baseline."""
    global batch_size
    old_batch_size = batch_size
    batch_size = batch_size_override
    
    try:
        print(f"\n=== Testing Majority Aggregation (batch_size={batch_size_override}) ===")
        
        # Run baseline
        baseline_result = majority_aggregation_baseline(synthetic_scores)
        
        # Run refactored
        refactored_result = majority_aggregation(synthetic_scores, batch_size=batch_size_override)
        
        assert compare_tensors(
            baseline_result, 
            refactored_result, 
            "Majority Aggregation",
            rtol=1e-4,  # More lenient for majority voting
            atol=1e-5
        ), "Majority aggregation results differ from baseline"
        
        print("✓ Majority aggregation matches baseline")
        
    finally:
        batch_size = old_batch_size


@pytest.mark.parametrize("batch_size_override", [50, 100, 200])
def test_borda_aggregation(synthetic_scores, batch_size_override):
    """Test borda aggregation against baseline."""
    global batch_size
    old_batch_size = batch_size
    batch_size = batch_size_override
    
    try:
        print(f"\n=== Testing Borda Aggregation (batch_size={batch_size_override}) ===")
        
        # Run baseline
        baseline_result = borda_aggregation_baseline(synthetic_scores)
        
        # Run refactored
        refactored_result = borda_aggregation(synthetic_scores, batch_size=batch_size_override)
        
        assert compare_tensors(
            baseline_result, 
            refactored_result, 
            "Borda Aggregation",
            rtol=1e-5,
            atol=1e-6
        ), "Borda aggregation results differ from baseline"
        
        print("✓ Borda aggregation matches baseline")
        
    finally:
        batch_size = old_batch_size


@pytest.mark.parametrize("batch_size_override", [50, 100, 200])
def test_average_aggregation(synthetic_scores, batch_size_override):
    """Test average aggregation against baseline."""
    global batch_size
    old_batch_size = batch_size
    batch_size = batch_size_override
    
    try:
        print(f"\n=== Testing Average Aggregation (batch_size={batch_size_override}) ===")
        
        # Run baseline
        baseline_result = average_aggregation_baseline(synthetic_scores)
        
        # Run refactored
        refactored_result = average_aggregation(synthetic_scores, batch_size=batch_size_override)
        
        assert compare_tensors(
            baseline_result, 
            refactored_result, 
            "Average Aggregation",
            rtol=1e-5,
            atol=1e-6
        ), "Average aggregation results differ from baseline"
        
        print("✓ Average aggregation matches baseline")
        
    finally:
        batch_size = old_batch_size


@pytest.mark.parametrize("batch_size_override", [50, 100, 200])
def test_norm_aggregation(synthetic_scores, batch_size_override):
    """Test normalized aggregation against baseline."""
    global batch_size
    old_batch_size = batch_size
    batch_size = batch_size_override
    
    try:
        print(f"\n=== Testing Norm Aggregation (batch_size={batch_size_override}) ===")
        
        # Run baseline
        baseline_result = norm_aggregation_baseline(synthetic_scores)
        
        # Run refactored
        refactored_result = norm_aggregation(synthetic_scores, batch_size=batch_size_override)
        
        assert compare_tensors(
            baseline_result, 
            refactored_result, 
            "Norm Aggregation",
            rtol=1e-5,
            atol=1e-6
        ), "Norm aggregation results differ from baseline"
        
        print("✓ Norm aggregation matches baseline")
        
    finally:
        batch_size = old_batch_size


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

def test_majority_edge_cases(edge_case_scores):
    """Test majority aggregation with edge cases (ties, zeros, negatives)."""
    print("\n=== Testing Majority Aggregation Edge Cases ===")
    
    baseline_result = majority_aggregation_baseline(edge_case_scores)
    refactored_result = majority_aggregation(edge_case_scores, batch_size=50)
    
    assert compare_tensors(
        baseline_result, 
        refactored_result, 
        "Majority Edge Cases",
        rtol=1e-4,
        atol=1e-5
    ), "Majority aggregation edge cases differ from baseline"
    
    print("✓ Majority aggregation handles edge cases correctly")


def test_borda_edge_cases(edge_case_scores):
    """Test borda aggregation with edge cases."""
    print("\n=== Testing Borda Aggregation Edge Cases ===")
    
    baseline_result = borda_aggregation_baseline(edge_case_scores)
    refactored_result = borda_aggregation(edge_case_scores, batch_size=50)
    
    assert compare_tensors(
        baseline_result, 
        refactored_result, 
        "Borda Edge Cases",
        rtol=1e-5,
        atol=1e-6
    ), "Borda aggregation edge cases differ from baseline"
    
    print("✓ Borda aggregation handles edge cases correctly")


def test_average_edge_cases(edge_case_scores):
    """Test average aggregation with edge cases."""
    print("\n=== Testing Average Aggregation Edge Cases ===")
    
    baseline_result = average_aggregation_baseline(edge_case_scores)
    refactored_result = average_aggregation(edge_case_scores, batch_size=50)
    
    assert compare_tensors(
        baseline_result, 
        refactored_result, 
        "Average Edge Cases",
        rtol=1e-5,
        atol=1e-6
    ), "Average aggregation edge cases differ from baseline"
    
    print("✓ Average aggregation handles edge cases correctly")


def test_norm_edge_cases(edge_case_scores):
    """Test normalized aggregation with edge cases."""
    print("\n=== Testing Norm Aggregation Edge Cases ===")
    
    baseline_result = norm_aggregation_baseline(edge_case_scores)
    refactored_result = norm_aggregation(edge_case_scores, batch_size=50)
    
    assert compare_tensors(
        baseline_result, 
        refactored_result, 
        "Norm Edge Cases",
        rtol=1e-5,
        atol=1e-6
    ), "Norm aggregation edge cases differ from baseline"
    
    print("✓ Norm aggregation handles edge cases correctly")


# ============================================================================
# PROPERTY TESTS
# ============================================================================

def test_majority_properties(synthetic_scores):
    """Test mathematical properties of majority aggregation."""
    print("\n=== Testing Majority Aggregation Properties ===")
    
    result = majority_aggregation(synthetic_scores, batch_size=100)
    
    # Property 1: All values should be in [0, 1] (since we average votes)
    min_val = result.min().item()
    max_val = result.max().item()
    assert min_val >= -1e-5 and max_val <= 1.0 + 1e-5, \
        f"Majority aggregation should produce values in [0, 1], got range [{min_val}, {max_val}]"
    
    # Property 2: For each query, sum of votes should be close to 1.0
    # (after averaging across models)
    row_sums = result.sum(dim=1)
    mean_sum = row_sums.mean().item()
    assert 0.9 < mean_sum < 1.1, \
        f"Mean row sum should be close to 1.0, got {mean_sum}"
    
    print("✓ Majority aggregation satisfies expected properties")


def test_borda_properties(synthetic_scores):
    """Test mathematical properties of borda aggregation."""
    print("\n=== Testing Borda Aggregation Properties ===")
    
    result = borda_aggregation(synthetic_scores, batch_size=100)
    
    # Property 1: All values should be positive (Borda counts are positive)
    assert torch.all(result >= 0.0), "Borda aggregation should produce non-negative values"
    
    # Property 2: For each query, the sum should be consistent
    # (sum of ranks 1 to N across all models)
    num_entities = result.shape[1]
    num_models = len(synthetic_scores)
    expected_sum = num_models * (num_entities * (num_entities + 1)) / 2
    
    row_sums = result.sum(dim=1)
    mean_sum = row_sums.mean().item()
    
    assert abs(mean_sum - expected_sum) / expected_sum < 0.01, \
        f"Mean Borda sum should be close to {expected_sum}, got {mean_sum}"
    
    print("✓ Borda aggregation satisfies expected properties")


def test_average_properties(synthetic_scores):
    """Test mathematical properties of average aggregation."""
    print("\n=== Testing Average Aggregation Properties ===")
    
    result = average_aggregation(synthetic_scores, batch_size=100)
    
    # Property: Result should be the mean of input scores
    # Load first score to check
    with open(synthetic_scores[0], 'rb') as f:
        first_scores = pickle.load(f)
    
    # Average should preserve approximate range
    all_scores = []
    for path in synthetic_scores:
        with open(path, 'rb') as f:
            all_scores.append(pickle.load(f))
    
    stacked = torch.stack(all_scores, dim=2)
    expected_mean = stacked.mean(dim=2)
    
    assert torch.allclose(result, expected_mean, rtol=1e-4, atol=1e-5), \
        "Average aggregation should produce mean of inputs"
    
    print("✓ Average aggregation satisfies expected properties")


def test_norm_properties(synthetic_scores):
    """Test mathematical properties of normalized aggregation."""
    print("\n=== Testing Norm Aggregation Properties ===")
    
    result = norm_aggregation(synthetic_scores, batch_size=100)
    
    # Property: After normalization, values should be in reasonable range
    # Each model's scores are normalized to [0, 1], then averaged
    # So result should be in [0, 1]
    assert torch.all(result >= -1e-5) and torch.all(result <= 1.0 + 1e-5), \
        "Norm aggregation should produce values in [0, 1]"
    
    print("✓ Norm aggregation satisfies expected properties")


# ============================================================================
# INTEGRATION TEST
# ============================================================================

def test_all_methods_consistency(synthetic_scores):
    """
    Test that all methods produce consistent outputs.
    This is a sanity check to ensure all methods run without errors.
    """
    print("\n=== Testing All Methods Consistency ===")
    
    methods = {
        'majority': majority_aggregation,
        'borda': borda_aggregation,
        'average': average_aggregation,
        'norm': norm_aggregation
    }
    
    results = {}
    for name, method in methods.items():
        print(f"  Running {name}...")
        result = method(synthetic_scores, batch_size=100)
        results[name] = result
        
        # Basic sanity checks
        assert result.shape[0] > 0, f"{name}: Empty result"
        assert not torch.isnan(result).any(), f"{name}: Contains NaN"
        assert not torch.isinf(result).any(), f"{name}: Contains Inf"
    
    print("✓ All methods produce valid outputs")
    
    # Check that different methods produce different results
    # (they should, since they use different aggregation strategies)
    for name1 in methods:
        for name2 in methods:
            if name1 < name2:  # Compare each pair once
                are_different = not torch.allclose(results[name1], results[name2], rtol=0.1)
                print(f"  {name1} vs {name2}: {'Different' if are_different else 'Similar'}")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
