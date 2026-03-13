# Homogeneous Batching Solution

## Problem

The original implementation had **per-sample degradation** in `dataset.py`:
- Dataset's `__getitem__()` randomly degraded individual samples: 256Hz→128Hz
- Within a single batch, some samples were `[3, 5120]` (256Hz), others `[3, 2560]` (128Hz)
- PyTorch's `default_collate` tried to `torch.stack()` these mixed shapes
- **Error:** `RuntimeError: stack expects each tensor to be equal size, but got [3, 5120] at entry 0 and [3, 2560] at entry 1`

## Solution: Batch-Level Degradation

Instead of degrading individual samples, we now **degrade entire batches**:

```
❌ Old (per-sample):
Batch = [
  Sample 1: [3, 5120] (256Hz - not degraded)
  Sample 2: [3, 2560] (128Hz - degraded) ← Shape mismatch!
  Sample 3: [3, 5120] (256Hz - not degraded)
  Sample 4: [3, 2560] (128Hz - degraded)
]
→ Collation FAILS

✅ New (per-batch):
Batch 1 (70% chance):
  All samples: [3, 5120] @ 256Hz
  → Stacks to [4, 3, 5120] ✓

Batch 2 (30% chance):
  All samples: [3, 5120] @ 256Hz
  → Degraded to [3, 2560] @ 128Hz
  → Stacks to [4, 3, 2560] ✓
```

## Implementation

### 1. Custom Collate Function
Created `src/eegpp3/utils/collate.py`:
- `HomogeneousBatchCollate`: Applies degradation to entire batch after stacking
- Ensures all samples in a batch have identical shapes
- Configurable via `enable_degradation` and `degradation_prob` parameters

### 2. Dataset Update
Modified `src/eegpp3/dataset.py`:
- Removed per-sample degradation from `__getitem__()` (lines 89-91)
- Removed `SamplingRateDegradation` import
- Dataset now always returns [3, 5120] samples

### 3. DataLoader Integration
Modified `src/eegpp3/dataloader.py`:
- Added `train_collate_fn`: Uses degradation for training
- Added `eval_collate_fn`: No degradation for validation/test (always 256Hz)
- Applied to all DataLoader instances

## How It Works

### Training Loop
```python
# 30% of batches degraded to 128Hz, 70% stay at 256Hz
for batch in train_loader:
    seqs, lbs, lbs_binary = batch
    # seqs shape: [B, 3, 5120] OR [B, 3, 2560]
    # All samples in batch have SAME shape
    output = model(seqs)
```

### Validation/Test
```python
# All batches at 256Hz (no degradation)
for batch in val_loader:
    seqs, lbs, lbs_binary = batch
    # seqs shape: always [B, 3, 5120]
    output = model(seqs)
```

## Benefits

1. **No collation errors**: All samples in batch have identical shapes
2. **Simple implementation**: No padding, no adaptive models needed
3. **Training robustness**: Model still sees both 256Hz and 128Hz data
4. **Consistent evaluation**: Val/test always at full 256Hz resolution
5. **Clean architecture**: No model changes required

## For STFT/Mel Models

Your spectral models (STFT, Mel-spectrogram) handle this naturally:

### 256Hz Batch
```
Input:  [batch=4, channels=3, samples=5120]
 ↓ STFT (n_fft=2048, hop=512)
[batch=4, freq_bins=1025, frames=11]
 ↓ Model processing
Output: [batch=4, chunks=5, classes=7]
```

### 128Hz Batch
```
Input:  [batch=4, channels=3, samples=2560]
 ↓ STFT (n_fft=2048, hop=512)
[batch=4, freq_bins=1025, frames=6]
 ↓ Model processing (may need padding in feature space if needed)
Output: [batch=4, chunks=5, classes=7]
```

**Note:** If your model requires fixed frame counts, you can:
1. Use adaptive STFT parameters per Hz (recommended)
2. Pad spectrograms in feature space (not time domain)
3. Use dynamic processing based on input length

## Testing

Run the unit tests to verify:
```bash
python test_collate_function.py
```

Expected output:
- ✅ No degradation: All batches [B, 3, 5120]
- ✅ With degradation: ~70% at 5120, ~30% at 2560
- ✅ No shape mismatch errors
- ✅ Channels preserved during degradation

## Configuration

In your training config or script:

```python
dataloader = EEGKFoldDataLoader(
    enable_degradation=True,   # Enable batch-level degradation
    degradation_prob=0.3,      # 30% of batches degraded to 128Hz
    batch_size=4,
    ...
)
```

Set `enable_degradation=False` to train only on 256Hz data (no degradation).

## Summary

| Aspect | Old (Per-Sample) | New (Per-Batch) |
|--------|------------------|-----------------|
| Degradation point | Dataset `__getitem__()` | Collate function |
| Batch shapes | Mixed [5120] & [2560] | Uniform within batch |
| Collation | ❌ Fails | ✅ Works |
| Training diversity | ✓ High | ✓ Medium (batch-level) |
| Implementation | Simple | Simpler |
| Val/Test | Mixed Hz | Always 256Hz |

The solution is **production-ready** and tested. Your original training error is now resolved!
