# Quick Start: MelSTFT Dual Sampling Rate Model

## ✅ Verification Complete

The implementation **correctly maintains the 5-chunk (20s) concatenation strategy** while adding dual sampling rate support.

### Confirmed Behavior

**5-Chunk Concatenation (20s window):**
```
Dataset loads: [idx-2, idx-1, idx, idx+1, idx+2]
              └─────────── 5 chunks × 4s = 20s ───────────┘
Concatenates: [3 channels, 5120 samples] (256Hz)
           or [3 channels, 2560 samples] (128Hz after degradation)
```

**Model Output:**
- Predictions for all 5 chunks: `[batch, 5, num_classes]`
- Evaluation uses middle chunk: `POS_IDX = 2`
- Full 20s context informs predictions ✓

## Usage

### 1. Train Model

```bash
python main.py \
  --mode train \
  --model_type melstftcnn1dnc \
  --n_epochs 100 \
  --batch_size 8
```

**What happens during training:**
- ✓ Automatically applies 30% degradation (256Hz → 128Hz)
- ✓ Processes 5 concatenated chunks (20s) per sample
- ✓ Model learns to handle both sampling rates
- ✓ Validates on middle chunk (POS_IDX=2) of each 20s window

### 2. Inference (256Hz Data)

```bash
python main.py \
  --mode infer \
  --model_type melstftcnn1dnc \
  --yaml_config_path config.yml
```

**Model automatically:**
- ✓ Detects 5120 samples → 256Hz
- ✓ Uses n_fft=2048, hop=512
- ✓ Processes full 20s windows
- ✓ Outputs predictions for all 5 chunks

### 3. Inference (128Hz Data)

**Prepare data without upsampling:**

Edit `src/eegpp3/utils/data_utils.py` lines 174-177:
```python
# Comment out these lines:
# if len(tmp_eeg) == params.MAX_SEQ_SIZE // 2:
#     tmp_eeg = resample_poly(tmp_eeg, 2, 1)
#     tmp_emg = resample_poly(tmp_emg, 2, 1)
#     tmp_mot = resample_poly(tmp_mot, 2, 1)
```

Then run inference:
```bash
python main.py \
  --mode infer \
  --model_type melstftcnn1dnc \
  --yaml_config_path config.yml
```

**Model automatically:**
- ✓ Detects 2560 samples → 128Hz
- ✓ Uses n_fft=1024, hop=256
- ✓ Processes full 20s windows
- ✓ Pads to consistent dimensions
- ✓ Outputs predictions for all 5 chunks

## Architecture Summary

```
Input: [batch, 3 channels, 5120 or 2560 samples]
  ↓
Per-channel processing:
  ├─ Auto-detect sampling rate (by sequence length)
  ├─ Adaptive STFT (2048/1024 for n_fft)
  ├─ Mel filterbank (64 bins, 0-128 Hz)
  ├─ Log compression
  └─ Pad to 11 frames (if needed)
  ↓
CNN Feature Extraction:
  ├─ Conv1D + MaxPool
  └─ Conv1D + MaxPool
  ↓
Flatten + FC → 512-dim embeddings per channel
  ↓
Concatenate: 3 channels × 512 = 1536 dims
  ↓
Classifier → [batch, 5 chunks, 7 classes]
  ↓
Output: Predictions for all 5 chunks (20s window)
        Evaluation uses middle chunk (POS_IDX=2)
```

## Key Features

✅ **5-chunk (20s) temporal context** - Preserved from original design
✅ **Dual sampling rate** - Single model handles 128Hz and 256Hz
✅ **Mel-scale preprocessing** - Better frequency representation
✅ **Training augmentation** - 30% degradation for robustness
✅ **Auto-detection** - No manual configuration needed
✅ **Dimension-safe** - Padding prevents CNN collapse

## Test Verification

Run the test suite to verify:
```bash
python test_melstft_dimensions.py
```

Expected output:
```
✓ Dataset correctly concatenates 5 chunks of 1024 samples (20s total)
✓ 256Hz: [3, 5120] → Mel [3, 64, 11 frames]
✓ 128Hz: [3, 2560] → Mel [3, 64, 11 frames] (padded)
✓ Model outputs [5, NUM_CLASSES] predictions for all chunks
✓ Evaluation uses POS_IDX=2 (middle segment)
✓ Temporal context from 20s window is preserved
```

## Files & Documentation

- **`IMPLEMENTATION_SUMMARY.md`** - Technical details and verification results
- **`MELSTFT_USAGE.md`** - Comprehensive usage guide
- **`test_melstft_dimensions.py`** - Test suite for dimensions
- **Model:** `src/eegpp3/models/melstftcnn1dnc.py`
- **Config:** `src/eegpp3/configs/melstftcnn1dnc_config.yml`

## Troubleshooting

**Issue:** Model receives wrong dimensions
**Solution:** Verify W_OUT=5 in `params.py` and check data loading

**Issue:** Poor 128Hz performance
**Solution:** Increase degradation_prob in `trainer.py` (line 73)

**Issue:** CUDA OOM
**Solution:** Reduce batch_size or n_mels in config

---

**Ready to train!** The implementation is verified and ready for use.
