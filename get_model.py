import params
from dataset import EGGDataset
from transformer_model import EGGPhrasePredictor
from cnn_model import CNNModel
# from cnn_model_2d import CNNModel2
from cnn_model_3c import CNNModel3C
from cnn_model_3c_3out import CNNModel3C3Out
from fft_model import FFTModel
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from sklearn.metrics import roc_auc_score, average_precision_score
import params
import torch
from tqdm import tqdm
from dev import get_device
import numpy as np
import joblib

device = get_device(params.DEVICE)
model_type = params.MODE_TYPE
SIDE_FLAG = params.SIDE_FLAG

# model_type = "Transformer"
TILE_SEQ = False
if model_type == "Transformer":
    TILE_SEQ = True




def get_model(n_class, out_collapsed=True):
    if model_type == "CNN":
        model = CNNModel(n_class=n_class, flag=SIDE_FLAG).to(device)
    elif model_type == "CNN3C":
        if params.OUT_3C:
            model = CNNModel3C3Out(n_class=n_class, flag=SIDE_FLAG, out_collapsed=out_collapsed).to(device)
        else:
            model = CNNModel3C(n_class=n_class, flag=SIDE_FLAG).to(device)
    elif model_type == "FFT":
        model = FFTModel(n_class=n_class).to(device)
    else:
        model = EGGPhrasePredictor(n_class=n_class, dmodel=params.D_MODEL).to(device)
    return model
