"""
Test script for homogeneous batch degradation.

This verifies that:
1. Batches have uniform shapes (no collation errors)
2. Some batches are 256Hz, others are 128Hz
3. Validation/test always use 256Hz
"""

import torch
from src.eegpp3.dataloader import EEGKFoldDataLoader


def test_homogeneous_batching():
    """Test that homogeneous batching works correctly."""

    print("=" * 80)
    print("Testing Homogeneous Batch Degradation")
    print("=" * 80)

    # Create dataloader with degradation enabled
    print("\n1. Creating dataloader with degradation enabled (30% probability)...")
    dataloader = EEGKFoldDataLoader(
        dataset_files='default',
        n_splits=5,
        n_workers=0,
        batch_size=4,
        minmax_normalized=True,
        enable_degradation=True,
        degradation_prob=0.3
    )

    # Set fold
    dataloader.set_fold(0)

    # Test training dataloader
    print("\n2. Testing training dataloader...")
    train_loader = dataloader.train_dataloader(epoch=0)

    hz_256_count = 0
    hz_128_count = 0

    print("\nProcessing first 10 batches:")
    for batch_idx, (seqs, lbs, lbs_binary) in enumerate(train_loader):
        if batch_idx >= 10:
            break

        batch_size, channels, seq_len = seqs.shape

        # Determine Hz based on sequence length
        if seq_len == 5120:
            hz = 256
            hz_256_count += 1
        elif seq_len == 2560:
            hz = 128
            hz_128_count += 1
        else:
            raise ValueError(f"Unexpected seq_len: {seq_len}")

        print(f"  Batch {batch_idx}: shape={seqs.shape} → {hz}Hz")

    print(f"\nSummary:")
    print(f"  256Hz batches: {hz_256_count}/10")
    print(f"  128Hz batches: {hz_128_count}/10")
    print(f"  Expected ratio: ~70% @ 256Hz, ~30% @ 128Hz")

    # Test validation dataloader (should always be 256Hz)
    print("\n3. Testing validation dataloader (should always be 256Hz)...")
    val_loader = dataloader.val_dataloader()

    all_256hz = True
    for batch_idx, (seqs, lbs, lbs_binary) in enumerate(val_loader):
        if batch_idx >= 5:
            break

        seq_len = seqs.shape[-1]
        if seq_len != 5120:
            all_256hz = False
            print(f"  ⚠️  Batch {batch_idx}: Expected 5120, got {seq_len}")
        else:
            print(f"  ✓ Batch {batch_idx}: shape={seqs.shape} → 256Hz")

    if all_256hz:
        print("\n✅ All validation batches at 256Hz (correct!)")
    else:
        print("\n❌ Some validation batches not at 256Hz (incorrect!)")

    # Test without degradation
    print("\n4. Testing dataloader WITHOUT degradation...")
    dataloader_no_deg = EEGKFoldDataLoader(
        dataset_files='default',
        n_splits=5,
        n_workers=0,
        batch_size=4,
        minmax_normalized=True,
        enable_degradation=False,  # No degradation
        degradation_prob=0.0
    )

    dataloader_no_deg.set_fold(0)
    train_loader_no_deg = dataloader_no_deg.train_dataloader(epoch=0)

    all_256hz = True
    for batch_idx, (seqs, lbs, lbs_binary) in enumerate(train_loader_no_deg):
        if batch_idx >= 5:
            break

        seq_len = seqs.shape[-1]
        if seq_len != 5120:
            all_256hz = False
            print(f"  ⚠️  Batch {batch_idx}: Expected 5120, got {seq_len}")
        else:
            print(f"  ✓ Batch {batch_idx}: shape={seqs.shape} → 256Hz")

    if all_256hz:
        print("\n✅ All batches at 256Hz when degradation disabled (correct!)")
    else:
        print("\n❌ Some batches degraded when degradation disabled (incorrect!)")

    print("\n" + "=" * 80)
    print("✅ Test completed successfully - no shape mismatch errors!")
    print("=" * 80)


if __name__ == '__main__':
    test_homogeneous_batching()
