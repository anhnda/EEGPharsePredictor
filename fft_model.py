import torch
from torch import nn
from torch.nn import MaxPool1d
import params
class FFTModel(torch.nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.name = "FFT"
        self.dropout1 = torch.nn.Dropout(0.1)
        self.ff1 = torch.nn.Linear(params.MAX_SEQ_SIZE, params.MAX_SEQ_SIZE)
        self.ff2 = torch.nn.Linear(params.MAX_SEQ_SIZE, n_class)
        self.act = torch.nn.ReLU()

    def forward(self, x):
        x = self.dropout1(x)
        x = torch.fft.fft(x)
        x = torch.real(x)
        x = self.act(self.ff1(x))
        x = self.ff2(x)
        x = torch.squeeze(x, 1)

        return x
