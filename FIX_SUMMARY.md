# Fix Summary: Mel Spectrogram Frequency Alignment Issue

## Problem

After implementing 128Hz degradation, the model's f1x score dropped dramatically from **70-80% to 44%** and got stuck without improvement.

## Root Cause

The mel filterbanks for 256Hz and 128Hz were creating **incompatible frequency representations**:

### Before Fix:
- **256Hz mel spectrogram**: 64 bins covering **0-128Hz**
- **128Hz mel spectrogram**: 64 bins covering **0-64Hz** only

This meant the **same mel bin index represented different frequency ranges**:
- Bin #30 in 256Hz data ≈ 60-80Hz frequency range
- Bin #30 in 128Hz data ≈ 30-40Hz frequency range

The model was completely confused because it was trying to learn from incompatible feature representations!

## Solution Implemented

Applied **Option 1: Zero-padding for high-frequency bins**

### After Fix:
- **256Hz mel spectrogram**: 64 bins covering **0-128Hz**
  - Bins 0-31: 0-64Hz (real data)
  - Bins 32-63: 64-128Hz (real data)

- **128Hz mel spectrogram**: 64 bins covering **0-128Hz**
  - Bins 0-31: 0-64Hz (real data, 32 bins)
  - Bins 32-63: 64-128Hz (zero-padded, 32 bins)

### Key Changes in `src/eegpp3/models/melstftcnn1dnc.py`:

1. **128Hz mel filterbank uses half the bins (32) for 0-64Hz**:
   ```python
   self.n_mels_128 = n_mels // 2  # 32 bins for 0-64Hz
   self.mel_scale_128 = torchaudio.transforms.MelScale(
       n_mels=self.n_mels_128,
       sample_rate=128,
       f_min=0,
       f_max=64,  # Nyquist limit for 128Hz
       ...
   )
   ```

2. **Zero-padding applied to high-frequency bins (64-128Hz)**:
   ```python
   # For 128Hz: compute mel on 0-64Hz (32 bins), then pad
   mel_spec_low = self.mel_scale_128(magnitude)  # [batch, 32, frames]

   # Pad with zeros for missing 64-128Hz range
   high_freq_padding = torch.zeros(batch, 32, frames, ...)
   mel_spec = torch.cat([mel_spec_low, high_freq_padding], dim=1)  # [batch, 64, frames]
   ```

## Verification

### Test Results:
- ✓ Both sampling rates produce **aligned mel spectrograms** [64, num_frames]
- ✓ Peak bins for low-frequency signals (< 64Hz) are **identical** for both rates
- ✓ High-frequency bins (32-63) for 128Hz are properly **zero-padded** (log(1e-9) ≈ -20.72)
- ✓ Model can now learn from **consistent frequency representations**

### Example with 20Hz test signal:
- 256Hz peak bin: **10**
- 128Hz peak bin: **10**
- **Perfectly aligned!**

## Expected Outcome

The model should now:
1. **Learn meaningful features** from both 256Hz and 128Hz data
2. **Converge properly** during training
3. **Achieve f1x scores in the 70-80% range** again
4. Be **robust to both sampling rates** at inference time

## Next Steps

1. **Delete old checkpoints** to start fresh training:
   ```bash
   rm out/checkpoints/melstftcnn1dnc_*.pkl
   ```

2. **Restart training** with the fixed model:
   ```bash
   python main.py --mode train --model_type melstftcnn1dnc --batch_size 10 --n_epochs 50
   ```

3. **Monitor the f1x score** - it should now improve steadily instead of being stuck at 44%

## Files Modified

1. `src/eegpp3/models/melstftcnn1dnc.py`:
   - Updated `MelSTFTEmbedding.__init__()` to use 32 bins for 128Hz
   - Updated `MelSTFTEmbedding.forward()` to zero-pad high-frequency bins
   - Updated docstring to explain the alignment strategy

## Test Files Created

1. `test_melstft_dimensions.py` - Validates shape compatibility
2. `test_mel_frequency_alignment.py` - Verifies frequency alignment correctness
