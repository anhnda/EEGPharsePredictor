import params
from dataset import EGGDataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from sklearn.metrics import roc_auc_score, average_precision_score
import params
import torch
from tqdm import tqdm
import numpy as np
import joblib
from params import get_dump_filename

params.DEVICE = "mps"

from get_model import get_model, TILE_SEQ, SIDE_FLAG, device
import shap
from shap import DeepExplainer, GradientExplainer
from train import get_model_dirname
import params
from optparse import OptionParser
import sys
from train import parse_x


def load_model(model, path):
    model.load_state_dict(torch.load(path))


def xshap(model_id=1, test_id=1):
    generator1 = torch.Generator().manual_seed(params.RD_SEED)
    torch.random.manual_seed(params.RD_SEED)

    MODEL_ID = model_id
    TEST_ID = test_id
    params.DID = TEST_ID
    print("Data info: DID: ", params.DID, get_dump_filename())
    print(model_id, test_id)
    dataset = EGGDataset(tile_seq=TILE_SEQ, dump_path=get_dump_filename(), side_flag=SIDE_FLAG)
    n_class = dataset.get_num_class()
    params.DID = MODEL_ID
    model = get_model(n_class, out_collapsed=not params.OUT_3C).to(device)
    model_path = "%s/model_%s.pkl" % (get_model_dirname(), MODEL_ID)
    print("Model path: ", model_path)
    load_model(model, model_path)

    train_dt, test_dt = random_split(dataset, [0.8, 0.2], generator=generator1)
    train_dataloader = DataLoader(train_dt, batch_size=params.BATCH_SIZE * 2, num_workers=0, shuffle=True,
                                  drop_last=True)
    test_dataloader = DataLoader(test_dt, batch_size=1, num_workers=0, shuffle=True)

    samples, lb, _, _,_ = next(iter(train_dataloader))
    # print(samples)
    if model.type == "Transformer":
        samples = samples.transpose(1, 0)
    else:
        samples = torch.unsqueeze(samples, 1)

    samples = samples.float().to(device)
    # print("\nSample: ", samples.shape, samples[0,:,2,-100:])
    # print(params.OFF_MOT)
    # exit(-1)
    explainer = GradientExplainer(model, samples)
    #
    sm = torch.nn.Softmax(dim=-1)
    xs = []
    lbs = []
    lbws = []
    epoches = []
    shap_values = []
    ic = 0
    MX = 202  # len(test_dataloader)
    preds = []
    for _, data in tqdm(enumerate(test_dataloader)):
        ic += 1
        if ic == MX - 1:
            break
        x, lb, lbw, _, epoch_ids = data

        if model.type == "Transformer":
            x = x.transpose(1, 0)
        else:
            x = torch.unsqueeze(x, 1)
        x = x.float().to(device)
        # print("X in", x.shape)
        pred = model(x)
        # print("Pred: ", pred.shape)
        if params.OUT_3C:
            pred = pred.reshape(pred.shape[0], -1, 3)[:, :, 1]
        pred = sm(pred).detach().cpu()
        preds.append(pred)
        # print("\nX", x.shape, x[:, :, 2, -100:])
        shap_v = explainer.shap_values(x)
        # print("\nShap V: ", shap_v[0][:,:,2,-100:])
        xs.append(x)
        lbs.append(lb)
        lbws.append(lbw)
        epoches.append(epoch_ids)
        # print("Shap value shape: ", shap_v[0].shape)
        shap_values.append(shap_v)
        # print(prediction.shape, prediction)
        # exit(-1)
        if ic % 100 == 0 and ic > 0:
            xss = torch.concat(xs, dim=0).detach().cpu().numpy()
            lbss = torch.concat(lbs, dim=0).detach().cpu().numpy()
            lbwss = torch.concat(lbws, dim=0).detach().cpu().numpy()
            epochess = torch.concat(epoches).detach().cpu().numpy()
            predss = torch.concat(preds, dim=0).numpy()
            joblib.dump([xss, lbss, lbwss, shap_values, dataset.idx_2lb, epochess, predss],
                        "%s/xmodel_%s_%s.pkl" % (get_model_dirname(), MODEL_ID, TEST_ID))

    xs = torch.concat(xs, dim=0).detach().cpu().numpy()
    lbs = torch.concat(lbs, dim=0).detach().cpu().numpy()
    lbws = torch.concat(lbws, dim=0).detach().cpu().numpy()
    epochess = torch.concat(epoches).detach().cpu().numpy()
    predss = torch.concat(preds, dim=0).numpy()

    joblib.dump([xs, lbs, lbws, shap_values, dataset.idx_2lb, epochess, predss],
                "%s/xmodel_%s_%s.pkl" % (get_model_dirname(), MODEL_ID, TEST_ID))


if __name__ == "__main__":
    parse_x()
    print("ARG: ", params.TRAIN_ID, params.TEST_ID)
    xshap(params.TRAIN_ID, params.TEST_ID)
