# Implementation Summary: Dual Sampling Rate MelSTFT Model

## ✅ Verified Implementation

The new `melstftcnn1dnc` model correctly implements dual sampling rate support while maintaining the **5-chunk (20s) concatenation strategy**.

### Key Confirmation

**Dataset Strategy:**
```
5 chunks of 4s each = 20s total
├── Chunk at idx-2 (4s)
├── Chunk at idx-1 (4s)
├── Chunk at idx   (4s) ← Main segment for evaluation (POS_IDX=2)
├── Chunk at idx+1 (4s)
└── Chunk at idx+2 (4s)

Concatenated into single tensor: [3 channels, 5120 samples] for 256Hz
                                 [3 channels, 2560 samples] for 128Hz
```

**Model Output:**
- Predictions for **all 5 chunks**: `[batch, 5, num_classes]`
- Training/validation uses **middle chunk** (POS_IDX=2) only
- Temporal context from full 20s window informs all predictions

## Test Results

All dimension tests passed successfully:

### Test 1: 256Hz Processing
```
Input:  [batch=2, channels=3, samples=5120]
         ↓ (per channel)
        [batch=2, samples=5120]
         ↓ (STFT: n_fft=2048, hop=512)
        [batch=2, freq_bins=1025, frames=11]
         ↓ (Mel filterbank: 64 bins)
        [batch=2, n_mels=64, frames=11]
         ↓ (CNN + Classifier)
Output: [batch=2, chunks=5, classes=7] ✓
```

### Test 2: 128Hz Processing
```
Input:  [batch=2, channels=3, samples=2560]
         ↓ (per channel)
        [batch=2, samples=2560]
         ↓ (STFT: n_fft=1024, hop=256)
        [batch=2, freq_bins=513, frames=6]
         ↓ (Mel filterbank: 64 bins)
        [batch=2, n_mels=64, frames=6]
         ↓ (Pad to 11 frames)
        [batch=2, n_mels=64, frames=11]
         ↓ (CNN + Classifier)
Output: [batch=2, chunks=5, classes=7] ✓
```

### Critical Fixes Applied

1. **Sampling Rate Detection Fix**
   - **Before:** Threshold at 900 samples (incorrect for 5-chunk concatenation)
   - **After:** Threshold at 3840 samples (midpoint between 2560 and 5120)
   - **Result:** Correctly detects 128Hz for 2560-sample inputs ✓

2. **Frame Count Normalization**
   - **Problem:** 128Hz produces 6 frames, 256Hz produces 11 frames
   - **Issue:** CNN pooling layers caused dimension collapse for 6 frames
   - **Solution:** Pad 128Hz mel spectrograms to 11 frames (replicate last frame)
   - **Result:** Both sampling rates work through full CNN pipeline ✓

## Data Flow Verification

### Training
```python
# In dataset.py __getitem__():
1. Load 5 chunks: [idx-2, idx-1, idx, idx+1, idx+2]
   Each chunk: [3, 1024] for 256Hz

2. Concatenate along time: torch.concat(seqs, dim=-1)
   Result: [3, 5120]

3. Apply random degradation (30% probability):
   degrader(seqs)  # [3, 5120] → [3, 2560]

4. Return: ([3, 5120 or 2560], labels_5chunks, labels_binary_5chunks)
```

### Model Forward Pass
```python
# In melstftcnn1dnc.py forward():
1. Input: [batch, 3, seq_len]

2. For each channel i:
   - Extract: xi = x[:, i, :]  # [batch, seq_len]
   - Detect sampling rate from seq_len
   - Apply adaptive STFT (n_fft=2048 or 1024)
   - Apply mel filterbank (64 bins)
   - Pad to 11 frames if needed
   - Process through CNN → [batch, 512]

3. Concatenate all channels: [batch, 1536]

4. Classifier: [batch, 1536] → [batch, 5*7] → [batch, 5, 7]

5. Return: (predictions_5chunks, binary_5chunks)
```

### Validation/Inference
```python
# In trainer.py validation loop:
pred, pred_binary = model(x)  # [batch, 5, 7], [batch, 5, 2]

# Use only main segment (POS_IDX=2):
test_pred.append(pred[:, params.POS_IDX, :-1])  # [batch, 6]
test_pred_binary.append(pred_binary[:, params.POS_IDX, :])  # [batch, 2]
```

## Temporal Context Preservation

✅ **Confirmed:** The model processes the entire 20s window as a continuous signal:

1. **Dataset concatenation:** Joins 5 chunks seamlessly along time dimension
2. **STFT processing:** Computes frequency content over entire 20s window
3. **CNN receptive field:** Can attend to patterns across chunk boundaries
4. **Classifier output:** Generates predictions for all 5 time points
5. **Evaluation focus:** Uses middle chunk (POS_IDX=2) to avoid edge effects

This design allows the model to leverage temporal context from neighboring chunks when predicting the main segment's sleep stage.

## Configuration Parameters

All correctly configured in `params.py`:
- `W_OUT = 5` ✓ (5 chunks)
- `MAX_SEQ_SIZE = 1024` ✓ (samples per chunk at 256Hz)
- `POS_IDX = 2` ✓ (middle chunk)
- `NUM_CLASSES = 7` ✓ (6 sleep stages + 1 other)

## Files Modified/Created

### Created
- `src/eegpp3/models/melstftcnn1dnc.py` - New model with dual sampling rate support
- `src/eegpp3/configs/melstftcnn1dnc_config.yml` - Model configuration
- `src/eegpp3/utils/augmentation.py` - Training degradation transforms
- `MELSTFT_USAGE.md` - Comprehensive usage documentation
- `test_melstft_dimensions.py` - Dimension verification tests

### Modified
- `src/eegpp3/dataset.py` - Added degradation support
- `src/eegpp3/dataloader.py` - Pass degradation parameters
- `src/eegpp3/trainer.py` - Auto-enable degradation for melstft
- `src/eegpp3/utils/model_utils.py` - Register new model

## Next Steps

1. **Train the model:**
   ```bash
   python main.py --mode train --model_type melstftcnn1dnc --n_epochs 100
   ```

2. **Verify training metrics:**
   - Check that degradation is being applied (monitor input shapes in logs)
   - Ensure validation metrics are reasonable

3. **Test inference:**
   - Test with 256Hz data (should work out-of-box)
   - For 128Hz data: comment out upsampling in `data_utils.py:174-177`

4. **Compare performance:**
   - Baseline: `stftcnn1dnc` (256Hz only)
   - New: `melstftcnn1dnc` (256Hz + 128Hz)

## Potential Improvements

Future enhancements to consider:
1. **Adaptive pooling:** Replace fixed pooling with adaptive pooling for more flexibility
2. **Variable degradation rate:** Randomize degradation between 20-50% per epoch
3. **Frequency masking:** Add SpecAugment-style augmentation
4. **Multi-scale processing:** Process different time scales simultaneously

---

**Status:** ✅ Implementation verified and ready for training
**Date:** 2026-03-12
