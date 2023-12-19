import shap

import params
from dataset import EGGDataset
from transformer_model import EGGPhrasePredictor
from cnn_model import CNNModel
from fft_model import FFTModel
from cnn_model_2d import CNNModel2

from torch.utils.data import DataLoader
from torch.utils.data import random_split
from sklearn.metrics import roc_auc_score, average_precision_score
import params
import torch
from tqdm import tqdm
from dev import get_device
import numpy as np
import joblib
from shap import DeepExplainer

device = get_device("cpu")
model_type = "CNN"  # "FFT" # "Transformer"
TWO_SIDE_WINDOWS = False
params.THREE_SIGNAL_TYPES = False
# model_type = "Transformer"
if model_type == "Transformer":
    tile_seq = True
else:
    tile_seq = False


def get_model(n_class):
    if model_type == "CNN":
        model = CNNModel(n_class=n_class, ).to(device)
    elif model_type == "CNN2":
        model = CNNModel2(n_class=n_class, ).to(device)
    elif model_type == "FFT":
        model = FFTModel(n_class=n_class).to(device)
    else:
        model = EGGPhrasePredictor(n_class=n_class, dmodel=params.D_MODEL).to(device)
    return model



def load_model(model, path):
    model.load_state_dict(torch.load(path))


def xshap():
    dataset = EGGDataset(tile_seq=tile_seq, two_side=TWO_SIDE_WINDOWS)
    n_class = dataset.get_num_class()
    model = get_model(n_class)
    load_model(model, "out/model.pkl")
    generator1 = torch.Generator().manual_seed(params.RD_SEED)
    train_dt, test_dt = random_split(dataset, [0.8, 0.2], generator=generator1)
    train_dataloader = DataLoader(train_dt, batch_size=params.BATCH_SIZE * 2, num_workers=1, shuffle=True,
                                  drop_last=True)
    samples, lb = next(iter(train_dataloader))
    if model.type == "Transformer":
        samples = samples.transpose(1, 0)
    else:
        samples = torch.unsqueeze(samples, 1)
    samples = samples.float().to(device)
    explainer = DeepExplainer(model, samples)
    #
    test_dataloader = DataLoader(test_dt, batch_size=1, shuffle=False)
    sm = torch.nn.Softmax(dim=-1)
    xs = []
    lbs = []
    shap_values = []
    ic = 0
    for _, data in tqdm(enumerate(test_dataloader)):
        ic += 1
        if ic == 40:
            break
        x, lb = data

        if model.type == "Transformer":
            x = x.transpose(1, 0)
        else:
            x = torch.unsqueeze(x, 1)
        x = x.float().to(device)
        # print("X in", x.shape)
        shap_v = explainer.shap_values(x, check_additivity=False)
        xs.append(x)
        lbs.append(lb)
        shap_values.append(shap_v)
        # print(prediction.shape, prediction)
        # exit(-1)
    xs = torch.concat(xs, dim=0).detach().cpu().numpy()
    lbs = torch.concat(lbs, dim=0).detach().cpu().numpy()
    joblib.dump([xs, lbs, shap_values,dataset.idx_2lb], "out/xmodel.pkl")


if __name__ == "__main__":
    xshap()
