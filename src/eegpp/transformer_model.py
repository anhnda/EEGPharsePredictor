from torch import nn
import torch
import math

from .import params


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        # print("X Shape", x.shape)
        pex = self.pe[:x.size(0)]
        # print("PEX: ", pex.shape)
        x = x + pex
        return self.dropout(x)


class EGGPhrasePredictor(nn.Module):
    def __init__(self, n_class, dmodel=params.D_MODEL, nhead=8):
        super().__init__()
        self.dmodel = dmodel
        self.type = "Transformer"
        self.nhead = nhead
        self.pe = PositionalEncoding(dmodel)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dmodel, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.ff1 = torch.nn.Linear(dmodel, dmodel * 2)
        self.ff2 = torch.nn.Linear(dmodel * 2, n_class)
        self.act = torch.nn.ReLU()
        # self.sm = torch.nn.Softmax(dim=-1)

    def forward2(self, x):
        x = self.pe(x)
        # print(x.dtype, x.shape)
        x = self.transformer_encoder(x)
        x = x[0, :, :]
        # print("FW: ", x.shape)

        pd = self.sm(self.ff2(self.act(self.ff1(x))))
        # print("PD: ", pd.shape)
        return pd

    def forward(self, x):
        x = self.pe(x)
        # print(x.dtype, x.shape)
        x = self.transformer_encoder(x)
        x = x[0, :, :]
        # print("FW: ", x.shape)

        pd = self.ff2(self.act(self.ff1(x)))
        # pd = self.sm(pd)
        # print("PD: ", pd.shape)
        return pd
