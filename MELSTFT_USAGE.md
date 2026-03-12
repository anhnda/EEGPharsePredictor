# MelSTFT Model with Dual Sampling Rate Support

This document explains how to use the new `melstftcnn1dnc` model that supports both 128Hz and 256Hz EEG sampling rates.

## Overview

The `MelSTFTCNN1DnCModel` extends the original STFT-based model with:
1. **Mel-scale filterbank** for better frequency representation
2. **Adaptive STFT parameters** for 128Hz and 256Hz inputs
3. **Training augmentation** that randomly degrades 30% of 256Hz signals to 128Hz
4. **Auto-detection** of sampling rate during inference

## Architecture Pipeline

```
Signal (128Hz or 256Hz)
    ↓
STFT (adaptive parameters)
    ↓
Mel filterbank (64 bins, 0-128 Hz)
    ↓
Log compression
    ↓
CNN feature extraction
    ↓
Sleep stage classification
```

### STFT Parameters

**For 256Hz signals** (1024 samples per 4s chunk):
- `n_fft = 2048`
- `win_length = 2048`
- `hop_length = 512`
- Captures frequencies: 0-128 Hz

**For 128Hz signals** (512 samples per 4s chunk):
- `n_fft = 1024`
- `win_length = 1024`
- `hop_length = 256`
- Captures frequencies: 0-64 Hz (naturally limited by Nyquist)

### Mel Filterbank

- **n_mels**: 64 mel frequency bins
- **fmin**: 0 Hz
- **fmax**: 128 Hz (for 256Hz) or 64 Hz (for 128Hz)
- **Log compression**: Applied for dynamic range compression

## Training

### Basic Training Command

```bash
python main.py --mode train --model_type melstftcnn1dnc --n_epochs 100 --batch_size 8
```

### Training Strategy

The model automatically applies **random degradation augmentation** during training:
- **70%** of batches use original 256Hz data
- **30%** of batches are degraded to 128Hz (downsampled with anti-aliasing)

This mixed training makes the model robust to both sampling rates.

### How Degradation Works

1. Original signal: `[batch, 3, 5120]` (256Hz, 5 chunks × 1024 samples)
2. Apply `torchaudio.transforms.Resample(256 → 128)` with anti-aliasing
3. Degraded signal: `[batch, 3, 2560]` (128Hz, 5 chunks × 512 samples)
4. Model processes with adaptive STFT parameters

### Training Configuration

The degradation is automatically enabled for `melstftcnn1dnc` model in `trainer.py`:
```python
# Enable degradation for melstft models
enable_degradation = 'melstft' in self.model_type.lower()
degradation_prob = 0.3  # 30% degradation rate
```

To modify degradation rate, edit `src/eegpp3/trainer.py` line ~73.

## Inference

### Basic Inference Command

```bash
python main.py --mode infer --model_type melstftcnn1dnc --yaml_config_path config.yml
```

### Auto-Detection of Sampling Rate

The model **automatically detects** sampling rate from input sequence length:

| Input Length (4s chunk) | Detected Rate | STFT Config |
|-------------------------|---------------|-------------|
| ~1024 samples           | 256Hz         | n_fft=2048  |
| ~512 samples            | 128Hz         | n_fft=1024  |

**Example:**
- 256Hz input: `[batch, 3, 5120]` → Uses 256Hz STFT params
- 128Hz input: `[batch, 3, 2560]` → Uses 128Hz STFT params

### Detection Logic

Located in `src/eegpp3/models/melstftcnn1dnc.py`:
```python
def _detect_sampling_rate(self, seq_length):
    """
    Detect sampling rate from sequence length.

    For W_OUT=5 (5 chunks of 4s each = 20s total):
    - 256Hz: 5 × 1024 = 5120 samples
    - 128Hz: 5 × 512 = 2560 samples

    Threshold at midpoint: (5120 + 2560) / 2 = 3840
    """
    if seq_length >= 3840:  # Closer to 5120 = 256Hz
        return 256
    else:  # Closer to 2560 = 128Hz
        return 128
```

### Frame Count Normalization

To prevent dimension collapse in CNN pooling layers, the mel spectrograms from both sampling rates are padded to a consistent frame count (11 frames):

- **256Hz**: Naturally produces ~11 frames → No padding needed
- **128Hz**: Produces ~6 frames → Padded to 11 frames by replicating last frame

This ensures the CNN architecture works correctly for both sampling rates.

### Preparing 128Hz Data for Inference

**Current behavior:** The data loading pipeline automatically upsamples 512-sample chunks to 1024 samples (see `data_utils.py` lines 174-177).

**For true 128Hz inference:**
1. Comment out the upsampling code in `src/eegpp3/utils/data_utils.py`:
   ```python
   # Lines 174-177 - Comment these out for native 128Hz support
   # if len(tmp_eeg) == params.MAX_SEQ_SIZE // 2:
   #     tmp_eeg = resample_poly(tmp_eeg, 2, 1)
   #     tmp_emg = resample_poly(tmp_emg, 2, 1)
   #     tmp_mot = resample_poly(tmp_mot, 2, 1)
   ```

2. Update `params.py` if needed to accept variable sequence lengths
3. Re-prepare your data dumps with native 128Hz chunks

Alternatively, prepare 128Hz data files with 512 samples per 4-second chunk, and the model will detect and process them correctly.

## Model Files

### Created Files

1. **Model**: `src/eegpp3/models/melstftcnn1dnc.py`
   - `MelSTFTEmbedding`: Adaptive mel-STFT preprocessing
   - `MelSTFTCNN1DnCModel`: Complete model architecture

2. **Config**: `src/eegpp3/configs/melstftcnn1dnc_config.yml`
   - STFT parameters for both sampling rates
   - Mel filterbank configuration
   - CNN layer specifications

3. **Augmentation**: `src/eegpp3/utils/augmentation.py`
   - `SamplingRateDegradation`: 256Hz→128Hz degradation transform
   - `apply_random_degradation`: Helper for training loops

### Modified Files

1. **Dataset**: `src/eegpp3/dataset.py`
   - Added `enable_degradation` parameter
   - Applies random degradation during training

2. **DataLoader**: `src/eegpp3/dataloader.py`
   - Passes degradation parameters to dataset

3. **Trainer**: `src/eegpp3/trainer.py`
   - Auto-enables degradation for melstft models

4. **Model Factory**: `src/eegpp3/utils/model_utils.py`
   - Registered `melstftcnn1dnc` model
   - Added 'mel' to Fourier Transform check

## Example Workflow

### 1. Train Model on 256Hz Data

```bash
python main.py \
  --mode train \
  --model_type melstftcnn1dnc \
  --n_epochs 100 \
  --batch_size 8 \
  --n_splits 5
```

During training:
- 70% batches: Original 256Hz (5120 samples)
- 30% batches: Degraded 128Hz (2560 samples)
- Model learns to handle both sampling rates

### 2. Inference on 256Hz Data

```bash
python main.py \
  --mode infer \
  --model_type melstftcnn1dnc \
  --yaml_config_path configs/inference_256hz.yml
```

Model detects 1024 samples/chunk → Uses 256Hz STFT parameters

### 3. Inference on 128Hz Data

Prepare 128Hz data (512 samples per 4s chunk), then:

```bash
python main.py \
  --mode infer \
  --model_type melstftcnn1dnc \
  --yaml_config_path configs/inference_128hz.yml
```

Model detects 512 samples/chunk → Uses 128Hz STFT parameters

## Technical Details

### Frequency Range Handling

**256Hz signals:**
- Nyquist frequency: 128 Hz
- Mel filterbank: 0-128 Hz across 64 bins
- Full frequency content preserved

**128Hz signals:**
- Nyquist frequency: 64 Hz
- Mel filterbank: 0-64 Hz across 64 bins
- Upper frequency range (64-128 Hz) naturally limited
- Model trained to recognize this pattern as "128Hz signal"

### Why This Works

1. **Shared mel representation**: Both sampling rates produce 64 mel bins
2. **Training on mixed data**: Model learns that limited bandwidth = 128Hz signal
3. **Adaptive STFT**: Correct time-frequency resolution for each rate
4. **Log compression**: Normalizes dynamic range across different inputs

### Performance Expectations

- **256Hz data**: Full model performance (trained on 70% real + 30% synthetic)
- **128Hz data**: Slightly reduced performance due to limited frequency content
- **Mixed deployment**: Single model works for both rates without retraining

## Troubleshooting

### Issue: Model receives wrong sequence length

**Symptom:** Input is [batch, 3, 5120] but expected [batch, 3, 2560]

**Solution:** Check your data preparation. For 128Hz:
- Ensure 512 samples per 4s chunk
- Disable automatic upsampling in `data_utils.py`

### Issue: Poor performance on 128Hz data

**Symptom:** Model works well on 256Hz but poorly on 128Hz

**Solution:**
- Increase degradation probability during training (e.g., 0.5 instead of 0.3)
- Train longer to expose model to more 128Hz examples
- Check that degradation is actually being applied (add logging)

### Issue: CUDA out of memory

**Symptom:** OOM error during training

**Solution:**
- Reduce batch size (try 4 instead of 8)
- Use smaller mel bins (try 48 instead of 64)
- Reduce number of workers

## Configuration Reference

### Mel Filterbank Parameters

Edit `src/eegpp3/configs/melstftcnn1dnc_config.yml`:

```yaml
n_mels: 64        # Number of mel bins (try 48, 64, 80, 128)
fmin: 0           # Minimum frequency in Hz
fmax: 128         # Maximum frequency in Hz (for 256Hz data)
```

### Degradation Parameters

Edit `src/eegpp3/trainer.py` around line 73:

```python
degradation_prob = 0.3  # Change to 0.5 for 50% degradation rate
```

Or pass to dataset directly:
```python
dataset = EEGDataset(
    dump_path,
    enable_degradation=True,
    degradation_prob=0.5  # 50% degradation
)
```

## Comparison: STFT vs MelSTFT

| Feature | STFTCNN1DnC | MelSTFTCNN1DnC |
|---------|-------------|-----------------|
| Frequency representation | Linear (1025 bins) | Mel-scale (64 bins) |
| Input channels to CNN | 1025 | 64 |
| Sampling rate support | 256Hz only | 128Hz + 256Hz |
| Training augmentation | None | 30% degradation |
| Parameter count | Higher | Lower (fewer input channels) |
| Frequency resolution | Uniform | Perceptually-motivated |

## Future Improvements

Possible enhancements:
1. **Variable degradation rate**: Randomly vary degradation between 20-40%
2. **Frequency masking**: Add mel frequency masking augmentation
3. **Time masking**: Add time masking augmentation
4. **Multi-resolution**: Use multiple mel scales simultaneously
5. **Learned filterbank**: Replace fixed mel filterbank with learned filters

## References

- **Mel-scale**: HTK mel scale implementation (better for EEG than slaney)
- **STFT**: Hamming window for reduced spectral leakage
- **Resampling**: Kaiser window with β=14.769 for high-quality anti-aliasing
- **Log compression**: log(x + ε) with ε=1e-9 for numerical stability

## Support

For issues or questions:
1. Check this documentation
2. Review `src/eegpp3/models/melstftcnn1dnc.py` for implementation details
3. Inspect `src/eegpp3/utils/augmentation.py` for degradation logic
4. File an issue with your training/inference logs
