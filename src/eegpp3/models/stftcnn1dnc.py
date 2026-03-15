import torch
from torch import nn

from .. import params
from ..utils.config_utils import load_yaml_config
from ..utils.data_utils import LABEL_DICT


class MNAPooling1D(nn.Module):
    def __init__(self, kernel_size=2, stride=2, padding=0):
        super().__init__()
        self.max_pooling = nn.MaxPool1d(kernel_size=kernel_size, stride=stride, padding=padding)
        self.average_pooling = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        mx = self.max_pooling(x)
        avg = self.average_pooling(x)
        return torch.concat([mx, avg], dim=1)


class BiMaxPooling1D(nn.Module):
    def __init__(self, kernel_size=2, stride=2, padding=0):
        super().__init__()
        self.max_pooling = nn.MaxPool1d(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        mx1 = self.max_pooling(x)
        mx2 = self.max_pooling(-x)
        return torch.concat([mx1, mx2], dim=1)


class STFTEmbedding(nn.Module):
    """
    Adaptive STFT embedding for 128Hz and 256Hz sampling rates.

    Strategy:
    - 256Hz input (5120 samples): n_fft=2048, hop=512, produces 1025 frequency bins
    - 128Hz input (2560 samples): n_fft=1024, hop=256, produces 513 frequency bins + zero-padding to 1025

    Both produce aligned STFT spectrograms [n_fft//2+1=1025, num_frames] for consistent CNN processing.
    """

    def __init__(
            self,
            n_fft_256=2048,
            win_length_256=2048,
            hop_length_256=512,
            n_fft_128=1024,
            win_length_128=1024,
            hop_length_128=256,
            return_complex=True,
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

        self.return_complex = return_complex
        self.normalized = normalized

        # Target frequency bins (from 256Hz STFT)
        self.target_freq_bins = n_fft_256 // 2 + 1  # 1025 bins

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

    def forward(self, x):
        """
        Forward pass with adaptive STFT.

        Args:
            x: Input signal [batch, seq_len] where seq_len = 5120 for 256Hz or 2560 for 128Hz

        Returns:
            STFT magnitude [batch, target_freq_bins, num_frames] - padded to consistent dimensions
        """
        # Detect sampling rate from input length
        seq_length = x.size(-1)
        sampling_rate = self._detect_sampling_rate(seq_length)

        # Select STFT parameters based on sampling rate
        if sampling_rate == 256:
            n_fft = self.n_fft_256
            win_length = self.win_length_256
            hop_length = self.hop_length_256
        else:  # 128Hz
            n_fft = self.n_fft_128
            win_length = self.win_length_128
            hop_length = self.hop_length_128

        # Compute STFT
        window = torch.hamming_window(win_length, device=x.device)
        stft_complex = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            normalized=self.normalized,
            return_complex=self.return_complex,
            onesided=True
        )

        # Compute magnitude
        magnitude = torch.sqrt(stft_complex.real ** 2 + stft_complex.imag ** 2)
        # magnitude shape: [batch, n_fft//2+1, num_frames]

        # Pad frequency dimension if needed (for 128Hz data)
        if sampling_rate == 128:
            # 128Hz produces 513 bins, need to pad to 1025 bins
            current_freq_bins = magnitude.size(1)
            if current_freq_bins < self.target_freq_bins:
                batch_size = magnitude.size(0)
                num_frames = magnitude.size(2)
                padding_bins = self.target_freq_bins - current_freq_bins

                # Pad with zeros for high-frequency bins (128Hz can't capture >64Hz)
                freq_padding = torch.zeros(
                    batch_size, padding_bins, num_frames,
                    device=magnitude.device, dtype=magnitude.dtype
                )
                magnitude = torch.cat([magnitude, freq_padding], dim=1)

        return magnitude


class Conv1DLayer(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            pooling_kernel_size,
            pooling_stride,
            pooling_padding=0,
            dropout=0.1
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pooling_kernel_size, stride=pooling_stride, padding=pooling_padding)
        )

    def forward(self, x):
        return self.conv(x)


class FC(nn.Module):
    def __init__(
            self,
            in_dim: int = 1024,
            dim_feedforward: int = 2048,
            out_dim: int = 7,
            dropout: float = 0.1,
    ):
        super().__init__()
        self.ff1 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, dim_feedforward),
            nn.ReLU(),
        )
        self.ff2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, out_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.ff1(x)
        x = self.ff2(x)
        return x


class STFTCNN1DnCModel(nn.Module):
    """
    STFT CNN model with support for dual sampling rates (128Hz and 256Hz).

    Architecture:
    - Input: [batch, 3, seq_len] where 3 = (EEG, EMG, MOT) channels
    - Per-channel processing:
        - STFTEmbedding: Signal -> STFT magnitude (adaptive for 128Hz/256Hz)
        - Conv1D layers for feature extraction
        - FC layer to 512-dim embedding
    - Concatenate all channel embeddings
    - Dual classifiers: main (sleep stage) + binary (stable/transition)
    """

    def __init__(self, yml_config_file='stftcnn1dnc_config.yml'):
        super().__init__()
        self.type = 'stftcnn1dnc'
        self.config = load_yaml_config(yml_config_file)
        num_chains = self.config['num_chains']
        self.chains = nn.ModuleList([nn.Sequential() for _ in range(num_chains)])
        out_dim = 512

        for chain in self.chains:
            # Adaptive STFT embedding layer
            input_embedding = STFTEmbedding(
                n_fft_256=self.config.get('n_fft_256', self.config.get('n_fft', 2048)),
                win_length_256=self.config.get('win_length_256', self.config.get('win_length', 2048)),
                hop_length_256=self.config.get('hop_length_256', self.config.get('hop_length', 512)),
                n_fft_128=self.config.get('n_fft_128', 1024),
                win_length_128=self.config.get('win_length_128', 1024),
                hop_length_128=self.config.get('hop_length_128', 256),
                return_complex=True,
                normalized=self.config['normalized']
            )
            chain.add_module('input_embedding', input_embedding)

            # Conv1D layers - input channels from target frequency bins (256Hz STFT)
            in_channels = self.config.get('n_fft_256', self.config.get('n_fft', 2048)) // 2 + 1
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
        self.classifier = FC(in_dim=out_dim * num_chains, out_dim=params.W_OUT * len(LABEL_DICT))
        self.classifier_binary = FC(in_dim=out_dim * num_chains + params.W_OUT * len(LABEL_DICT), out_dim=params.W_OUT * 2)

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

    @staticmethod
    def chain_forward(chain, x):
        return chain(x)
