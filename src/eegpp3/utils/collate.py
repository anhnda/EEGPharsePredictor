"""
Custom collate functions for EEG data loading.

Implements homogeneous batch degradation for mixed Hz training.
"""

import torch
from ..utils.augmentation import SamplingRateDegradation


class HomogeneousBatchCollate:
    """
    Collate function that applies degradation at batch level.

    This ensures all samples in a batch have the same shape:
    - Some batches: all samples at 256Hz → [B, 3, 5120]
    - Other batches: all samples at 128Hz → [B, 3, 2560]

    Benefits:
    - No shape mismatch in torch.stack()
    - Simpler than padding or adaptive models
    - Still trains on both Hz for robustness
    """

    def __init__(self, enable_degradation=False, degradation_prob=0.3):
        """
        Args:
            enable_degradation: Whether to enable 256Hz→128Hz degradation
            degradation_prob: Probability of degrading entire batch (0.3 = 30%)
        """
        self.enable_degradation = enable_degradation
        self.degradation_prob = degradation_prob

        if self.enable_degradation:
            self.degrader = SamplingRateDegradation(orig_freq=256, new_freq=128)
        else:
            self.degrader = None

    def __call__(self, batch):
        """
        Collate a batch with optional batch-level degradation.

        Args:
            batch: List of tuples (seqs, lbs, lbs_binary) from dataset
                   Each seqs is [3, 5120] at 256Hz

        Returns:
            seqs: Batched tensor [B, 3, 5120] or [B, 3, 2560]
            lbs: Batched labels [B, W_OUT, num_classes]
            lbs_binary: Batched binary labels [B, W_OUT, 2]
        """
        # Unpack batch
        seqs_list = [item[0] for item in batch]
        lbs_list = [item[1] for item in batch]
        lbs_binary_list = [item[2] for item in batch]

        # Stack into batches (all same shape from dataset)
        seqs = torch.stack(seqs_list, dim=0)  # [B, 3, 5120]
        lbs = torch.stack(lbs_list, dim=0)
        lbs_binary = torch.stack(lbs_binary_list, dim=0)

        # Apply batch-level degradation
        if self.degrader is not None and torch.rand(1).item() < self.degradation_prob:
            # Degrade entire batch: [B, 3, 5120] → [B, 3, 2560]
            seqs = self.degrader(seqs)

        return seqs, lbs, lbs_binary


def create_collate_fn(enable_degradation=False, degradation_prob=0.3):
    """
    Factory function to create collate_fn for DataLoader.

    Usage:
        collate_fn = create_collate_fn(enable_degradation=True, degradation_prob=0.3)
        dataloader = DataLoader(..., collate_fn=collate_fn)
    """
    return HomogeneousBatchCollate(
        enable_degradation=enable_degradation,
        degradation_prob=degradation_prob
    )
