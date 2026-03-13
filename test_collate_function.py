"""
Unit test for the homogeneous batch collate function.

Tests the collate function in isolation without requiring full datasets.
"""

import torch
from src.eegpp3.utils.collate import HomogeneousBatchCollate


def create_mock_batch(batch_size=4, w_out=5, num_classes=7):
    """Create a mock batch as if from EEGDataset.__getitem__."""
    batch = []
    for _ in range(batch_size):
        # Each sample from dataset: (seqs, lbs, lbs_binary)
        seqs = torch.randn(3, 5120)  # [channels=3, samples=5120] at 256Hz
        lbs = torch.randn(w_out, num_classes)
        lbs_binary = torch.randn(w_out, 2)
        batch.append((seqs, lbs, lbs_binary))
    return batch


def test_collate_no_degradation():
    """Test collate function with degradation disabled."""
    print("=" * 80)
    print("Test 1: Collate WITHOUT degradation")
    print("=" * 80)

    collate_fn = HomogeneousBatchCollate(enable_degradation=False, degradation_prob=0.0)

    # Create mock batch
    batch = create_mock_batch(batch_size=4)

    # Collate
    seqs, lbs, lbs_binary = collate_fn(batch)

    print(f"\nInput: 4 samples of shape [3, 5120]")
    print(f"Output shape: {seqs.shape}")
    print(f"Expected: torch.Size([4, 3, 5120]) at 256Hz")

    assert seqs.shape == torch.Size([4, 3, 5120]), f"Wrong shape: {seqs.shape}"
    assert lbs.shape == torch.Size([4, 5, 7]), f"Wrong lbs shape: {lbs.shape}"
    assert lbs_binary.shape == torch.Size([4, 5, 2]), f"Wrong lbs_binary shape: {lbs_binary.shape}"

    print("✅ PASSED: All samples at 256Hz (5120 samples)")
    print()


def test_collate_with_degradation():
    """Test collate function with degradation enabled."""
    print("=" * 80)
    print("Test 2: Collate WITH degradation (probabilistic)")
    print("=" * 80)

    collate_fn = HomogeneousBatchCollate(enable_degradation=True, degradation_prob=0.5)

    hz_256_count = 0
    hz_128_count = 0
    num_trials = 100

    print(f"\nRunning {num_trials} trials with 50% degradation probability...")

    for i in range(num_trials):
        batch = create_mock_batch(batch_size=4)
        seqs, lbs, lbs_binary = collate_fn(batch)

        # Check shape consistency within batch
        assert seqs.dim() == 3, "Should be 3D tensor"
        assert seqs.shape[0] == 4, "Batch size should be 4"
        assert seqs.shape[1] == 3, "Should have 3 channels"

        # Track Hz distribution
        if seqs.shape[2] == 5120:
            hz_256_count += 1
        elif seqs.shape[2] == 2560:
            hz_128_count += 1
        else:
            raise ValueError(f"Unexpected shape: {seqs.shape}")

    ratio_256 = hz_256_count / num_trials * 100
    ratio_128 = hz_128_count / num_trials * 100

    print(f"\nResults:")
    print(f"  256Hz batches: {hz_256_count}/{num_trials} ({ratio_256:.1f}%)")
    print(f"  128Hz batches: {hz_128_count}/{num_trials} ({ratio_128:.1f}%)")
    print(f"  Expected: ~50% each")

    # Allow some variance (30-70% range is reasonable for random 50% probability)
    assert 30 <= ratio_256 <= 70, f"256Hz ratio {ratio_256}% outside expected range"
    assert 30 <= ratio_128 <= 70, f"128Hz ratio {ratio_128}% outside expected range"

    print("✅ PASSED: Degradation working with expected distribution")
    print()


def test_shape_consistency():
    """Test that all samples in a batch have consistent shapes."""
    print("=" * 80)
    print("Test 3: Shape consistency (no collation errors)")
    print("=" * 80)

    collate_fn = HomogeneousBatchCollate(enable_degradation=True, degradation_prob=0.5)

    print("\nTesting 50 batches for shape consistency...")

    for trial in range(50):
        batch = create_mock_batch(batch_size=8)

        # This would raise RuntimeError if shapes don't match (like the original error)
        try:
            seqs, lbs, lbs_binary = collate_fn(batch)
        except RuntimeError as e:
            print(f"\n❌ FAILED at trial {trial}: {e}")
            raise

    print("✅ PASSED: No shape mismatch errors in 50 trials")
    print()


def test_degradation_preserves_channels():
    """Test that degradation only affects time dimension, not channels."""
    print("=" * 80)
    print("Test 4: Degradation preserves channel dimension")
    print("=" * 80)

    collate_fn = HomogeneousBatchCollate(enable_degradation=True, degradation_prob=1.0)

    batch = create_mock_batch(batch_size=4)
    seqs, lbs, lbs_binary = collate_fn(batch)

    print(f"\nInput: [batch=4, channels=3, samples=5120]")
    print(f"Output: {seqs.shape}")

    assert seqs.shape[0] == 4, "Batch dimension changed"
    assert seqs.shape[1] == 3, "Channel dimension changed"
    assert seqs.shape[2] == 2560, "Should be degraded to 2560 samples at 128Hz"

    print("✅ PASSED: Channels preserved, only time dimension halved")
    print()


if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("HOMOGENEOUS BATCH COLLATION - UNIT TESTS")
    print("=" * 80)
    print()

    test_collate_no_degradation()
    test_collate_with_degradation()
    test_shape_consistency()
    test_degradation_preserves_channels()

    print("=" * 80)
    print("✅ ALL TESTS PASSED!")
    print("=" * 80)
    print("\nSummary:")
    print("  - Collate function works correctly")
    print("  - Degradation applied at batch level (all samples in batch same Hz)")
    print("  - No shape mismatch errors")
    print("  - Ready to use in training!")
    print()
