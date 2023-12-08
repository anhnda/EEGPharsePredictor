import shap

import params
from dataset import EGGDataset
from transformer_model import EGGPhrasePredictor
from cnn_model import CNNModel
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

device = get_device("cpu")
model_type = "CNN"
TWO_SIDE_WINDOWS = True
# model_type = "Transformer"
if model_type == "Transformer":
    tile_seq = True
else:
    tile_seq = False


def get_model(n_class):
    if model_type == "CNN":
        model = CNNModel(n_class=n_class, ).to(device)
    elif model_type == "FFT":
        model = FFTModel(n_class=n_class).to(device)
    else:
        model = EGGPhrasePredictor(n_class=n_class, dmodel=params.D_MODEL).to(device)
    return model


def train():
    dataset = EGGDataset(tile_seq=tile_seq, two_side=TWO_SIDE_WINDOWS)
    n_class = dataset.get_num_class()
    model = get_model(n_class)
    generator1 = torch.Generator().manual_seed(params.RD_SEED)
    train_dt, test_dt = random_split(dataset, [0.8, 0.2], generator=generator1)
    train_dataloader = DataLoader(train_dt, batch_size=params.BATCH_SIZE, num_workers=1, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dt, batch_size=1, shuffle=False)
    sm = torch.nn.Softmax(dim=-1)
    xs = []
    lbs = []
    for _, data in enumerate(test_dataloader):
        x, lb = data

        if model.type == "Transformer":
            x = x.transpose(1, 0)
        else:
            x = torch.unsqueeze(x, 1)
        x = x.float().to(device)
        # print("X in", x.shape)
        prediction = model(x)
        shap.Explainer