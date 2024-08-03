import torch
from torch import nn
torch.manual_seed(1)
DSIZE = 16
encoder_layer = nn.TransformerEncoderLayer(d_model=DSIZE, nhead=8)
transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
src = torch.ones(2, 4, DSIZE)
print(src.dtype, src.shape)
transformer_encoder.eval()
src_mask = nn.Transformer.generate_square_subsequent_mask(len(src))

out = transformer_encoder(src[0,:,:])
print(out.shape, out)
# src = torch.rand(2, 32)
print("Next")
