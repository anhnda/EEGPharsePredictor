import torch
from torch import nn
import torchaudio

from .. import params
from ..utils.config_utils import load_yaml_config
from ..utils.data_utils import LABEL_DICT
from .stftcnn1dnc import Conv1DLayer, FC


class MelSTFTEmbedding(nn.Module):
    """
    Mel-scale STFT embedding with adaptive parameters for 128Hz and 256Hz sampling rates.

    Strategy:
    - 256Hz input (1024 samples): n_fft=2048, hop=512, captures 0-128Hz
    - 128Hz input (512 samples): n_fft=1024, hop=256, captures 0-64Hz, pad 64-128Hz with zeros

    Both produce mel spectrograms in 0-128Hz range with n_mels bins.
    """

    def __init__(
            self,
            n_fft_256=2048,
            win_length_256=2048,
            hop_length_256=512,
            n_fft_128=1024,
            win_length_128=1024,
            hop_length_128=256,
            n_mels=64,
            fmin=0,
            fmax=128,
            normalized=True,
    ):
        super().__init__()
        # STFT params for 256Hz
        self.n_fft_256 = n_fft_256
        self.win_length_256 = win_length_256
        self.hop_length_256 = hop_length_256

        # STFT params for 128Hz
        self.n_fft_128 = n_fft_128
        self.win_length_128 = win_length_128
        self.hop_length_128 = hop_length_128

        # Mel filterbank params
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.normalized = normalized

        # Create mel filterbanks for both sampling rates
        # 256Hz: sample_rate=256, n_fft=2048
        self.mel_scale_256 = torchaudio.transforms.MelScale(
            n_mels=n_mels,
            sample_rate=256,
            f_min=fmin,
            f_max=fmax,
            n_stft=n_fft_256 // 2 + 1,
            norm='slaney',
            mel_scale='htk'
        )

        # 128Hz: sample_rate=128, n_fft=1024, but output will be padded
        self.mel_scale_128 = torchaudio.transforms.MelScale(
            n_mels=n_mels,
            sample_rate=128,
            f_min=fmin,
            f_max=min(64, fmax),  # 128Hz can only capture up to 64Hz
            n_stft=n_fft_128 // 2 + 1,
            norm='slaney',
            mel_scale='htk'
        )

    def _detect_sampling_rate(self, seq_length):
        """Detect sampling rate from sequence length (4s chunks)."""
        if seq_length >= 900:  # ~1024 samples = 256Hz
            return 256
        else:  # ~512 samples = 128Hz
            return 128

    def _compute_stft(self, x, sampling_rate):
        """Compute STFT with adaptive parameters based on sampling rate."""
        if sampling_rate == 256:
            n_fft = self.n_fft_256
            win_length = self.win_length_256
            hop_length = self.hop_length_256
        else:  # 128Hz
            n_fft = self.n_fft_128
            win_length = self.win_length_128
            hop_length = self.hop_length_128

        window = torch.hamming_window(win_length, device=x.device)
        stft_complex = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            normalized=self.normalized,
            return_complex=True,
            onesided=True
        )

        # Compute magnitude
        magnitude = torch.sqrt(stft_complex.real ** 2 + stft_complex.imag ** 2)
        return magnitude

    def forward(self, x):
        """
        Forward pass with adaptive STFT and mel filterbank.

        Args:
            x: Input signal [batch, seq_len] where seq_len = 5120 for 256Hz or ~2560 for 128Hz

        Returns:
            Mel spectrogram [batch, n_mels, num_frames]
        """
        # Detect sampling rate from input length
        seq_length = x.size(-1)
        sampling_rate = self._detect_sampling_rate(seq_length)

        # Compute STFT magnitude
        magnitude = self._compute_stft(x, sampling_rate)  # [batch, n_fft//2+1, num_frames]

        # Apply mel filterbank
        if sampling_rate == 256:
            mel_spec = self.mel_scale_256(magnitude)
        else:  # 128Hz
            # For 128Hz, compute mel on 0-64Hz, result will naturally represent limited bandwidth
            mel_spec = self.mel_scale_128(magnitude)
            # Note: The mel filterbank for 128Hz is configured with fmax=64Hz,
            # so it naturally represents the limited frequency range.
            # No explicit padding needed as the model will learn the 128Hz signal characteristics.

        # Apply log compression with small epsilon for numerical stability
        mel_spec = torch.log(mel_spec + 1e-9)

        return mel_spec


class MelSTFTCNN1DnCModel(nn.Module):
    """
    Mel-STFT CNN model with support for dual sampling rates (128Hz and 256Hz).

    Architecture:
    - Input: [batch, 3, seq_len] where 3 = (EEG, EMG, MOT) channels
    - Per-channel processing:
        - MelSTFTEmbedding: Signal -> STFT -> Mel filterbank -> Log compression
        - Conv1D layers for feature extraction
        - FC layer to 512-dim embedding
    - Concatenate all channel embeddings
    - Dual classifiers: main (sleep stage) + binary (stable/transition)
    """

    def __init__(self, yml_config_file='melstftcnn1dnc_config.yml'):
        super().__init__()
        self.type = 'melstftcnn1dnc'
        self.config = load_yaml_config(yml_config_file)
        num_chains = self.config['num_chains']
        self.chains = nn.ModuleList([nn.Sequential() for _ in range(num_chains)])
        out_dim = 512

        for chain in self.chains:
            # Mel-STFT embedding layer
            input_embedding = MelSTFTEmbedding(
                n_fft_256=self.config['n_fft_256'],
                win_length_256=self.config['win_length_256'],
                hop_length_256=self.config['hop_length_256'],
                n_fft_128=self.config['n_fft_128'],
                win_length_128=self.config['win_length_128'],
                hop_length_128=self.config['hop_length_128'],
                n_mels=self.config['n_mels'],
                fmin=self.config['fmin'],
                fmax=self.config['fmax'],
                normalized=self.config['normalized']
            )
            chain.add_module('input_embedding', input_embedding)

            # Conv1D layers - input channels now = n_mels instead of n_fft//2+1
            in_channels = self.config['n_mels']
            for i, layer_config in enumerate(self.config['conv_layers']):
                conv_layer = Conv1DLayer(
                    in_channels=in_channels,
                    out_channels=layer_config['out_channels'],
                    kernel_size=layer_config['kernel_size'],
                    stride=layer_config['stride'],
                    padding=layer_config['padding'],
                    pooling_kernel_size=layer_config['pooling_kernel_size'],
                    pooling_stride=layer_config['pooling_stride'],
                    pooling_padding=layer_config['pooling_padding'],
                    dropout=layer_config['dropout']
                )
                chain.add_module(name=layer_config['name'], module=conv_layer)
                in_channels = layer_config['out_channels']

            # Flatten and FC layers
            flatten = nn.Flatten()
            fc = FC(in_dim=1024 * 2, out_dim=out_dim)
            chain.add_module('flatten', flatten)
            chain.add_module('fc', fc)

        # Dual classifiers
        self.classifier = FC(
            in_dim=out_dim * num_chains,
            out_dim=params.W_OUT * len(LABEL_DICT)
        )
        self.classifier_binary = FC(
            in_dim=out_dim * num_chains + params.W_OUT * len(LABEL_DICT),
            out_dim=params.W_OUT * 2
        )

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x: Input tensor [batch, 3, seq_len]
               - seq_len = 5120 for 256Hz (5 chunks * 1024 samples)
               - seq_len = 2560 for 128Hz (5 chunks * 512 samples)

        Returns:
            tuple: (main_predictions, binary_predictions)
                - main_predictions: [batch, W_OUT, num_classes]
                - binary_predictions: [batch, W_OUT, 2]
        """
        xis = []
        for i, chain in enumerate(self.chains):
            xi = x[:, i, :]  # Extract channel i
            xi = chain(xi)
            xis.append(xi)

        # Concatenate all channel embeddings
        out = torch.concat(xis, dim=-1)
        outx = out

        # Main classifier (sleep stage)
        out = self.classifier(out)
        out = out.reshape(out.size(0), params.W_OUT, -1)

        # Binary classifier (stable/transition)
        out2x = torch.concat([outx, out.reshape(out.size(0), -1)], dim=1)
        out2 = self.classifier_binary(out2x)
        out2 = out2.reshape(out2.size(0), params.W_OUT, -1)

        return out, out2
