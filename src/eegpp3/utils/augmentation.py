"""
Data augmentation utilities for EEG signals.

Includes degradation from 256Hz to 128Hz for training robustness.
"""

import torch
import torchaudio.transforms as T


class SamplingRateDegradation:
    """
    Degrades 256Hz EEG signals to 128Hz by downsampling.

    This simulates lower sampling rate data during training, making the model
    robust to both 256Hz and 128Hz inputs at inference time.

    Strategy:
    1. Resample from 256Hz to 128Hz (includes built-in anti-aliasing low-pass filter)
    2. The resampling naturally low-pass filters at Nyquist/2 (64Hz for 128Hz target)

    Usage during training:
        - Apply to 30% of training samples randomly
        - Keep 70% at original 256Hz
    """

    def __init__(self, orig_freq=256, new_freq=128):
        """
        Args:
            orig_freq: Original sampling frequency (default: 256Hz)
            new_freq: Target sampling frequency (default: 128Hz)
        """
        self.orig_freq = orig_freq
        self.new_freq = new_freq

        # torchaudio.Resample includes anti-aliasing by default
        # This will low-pass filter at new_freq/2 = 64Hz before downsampling
        self.resampler = T.Resample(
            orig_freq=orig_freq,
            new_freq=new_freq,
            lowpass_filter_width=64,  # High-quality anti-aliasing filter
            rolloff=0.99,
            resampling_method='sinc_interp_kaiser',
            beta=14.769656459379492  # Kaiser window parameter
        )

    def __call__(self, signal):
        """
        Degrade signal from 256Hz to 128Hz.

        Args:
            signal: Input tensor of shape [..., seq_len]
                   For 4s @ 256Hz: seq_len = 1024
                   For 20s @ 256Hz: seq_len = 5120 (5 chunks concatenated)

        Returns:
            Downsampled tensor of shape [..., seq_len//2]
                   For 4s: 1024 -> 512 samples
                   For 20s: 5120 -> 2560 samples
        """
        # Move to CPU if needed (torchaudio.Resample works best on CPU)
        device = signal.device
        signal_cpu = signal.cpu() if device.type != 'cpu' else signal

        # Apply resampling (includes anti-aliasing)
        degraded = self.resampler(signal_cpu)

        # Move back to original device
        if device.type != 'cpu':
            degraded = degraded.to(device)

        return degraded


def apply_random_degradation(signal, degradation_prob=0.3, degrader=None):
    """
    Randomly apply sampling rate degradation to a batch of signals.

    Args:
        signal: Input tensor [batch, channels, seq_len] or [channels, seq_len]
        degradation_prob: Probability of degrading each sample (default: 0.3 = 30%)
        degrader: SamplingRateDegradation instance (creates one if None)

    Returns:
        Degraded signal with same batch/channel structure but potentially shorter seq_len
        Boolean mask indicating which samples were degraded
    """
    if degrader is None:
        degrader = SamplingRateDegradation()

    # Handle both batched [B, C, T] and single [C, T] inputs
    is_batched = signal.dim() == 3
    if not is_batched:
        signal = signal.unsqueeze(0)

    batch_size = signal.size(0)
    channels = signal.size(1)

    # Random degradation mask
    degrade_mask = torch.rand(batch_size) < degradation_prob

    # If no degradation needed, return original
    if not degrade_mask.any():
        result = signal if is_batched else signal.squeeze(0)
        return result, degrade_mask

    # Apply degradation to selected samples
    degraded_signals = []
    for i in range(batch_size):
        if degrade_mask[i]:
            # Degrade all channels of this sample
            degraded_sample = degrader(signal[i])  # [channels, seq_len//2]
            degraded_signals.append(degraded_sample)
        else:
            degraded_signals.append(signal[i])

    # Stack back
    result = torch.stack(degraded_signals, dim=0)

    if not is_batched:
        result = result.squeeze(0)

    return result, degrade_mask


# Convenience function for training loops
def create_degrader(orig_freq=256, new_freq=128):
    """Create a degradation transform for training."""
    return SamplingRateDegradation(orig_freq=orig_freq, new_freq=new_freq)
