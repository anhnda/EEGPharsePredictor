import math

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
from get_model import get_model, TILE_SEQ, SIDE_FLAG, device


def train():
    dataset = EGGDataset(tile_seq=TILE_SEQ, side_flag=SIDE_FLAG)
    n_class = dataset.get_num_class()
    model = get_model(n_class)
    generator1 = torch.Generator().manual_seed(params.RD_SEED)
    train_dt, test_dt = random_split(dataset, [0.8, 0.2], generator=generator1)
    train_dataloader = DataLoader(train_dt, batch_size=params.BATCH_SIZE, num_workers=1, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dt, batch_size=params.BATCH_SIZE, shuffle=False)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    min_test_loss = 1e6
    min_id = -1
    sm = torch.nn.Softmax(dim=-1)
    is_first_test = True
    for epoch_id in range(params.N_EPOCH):
        model.train()
        for it, data in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            x, lb, _ = data
            # print(x.shape, lb)
            # exit(-1)
            if model.type == "Transformer":
                x = x.transpose(1, 0)
            else:
                x = torch.unsqueeze(x, 1)
            x = x.float().to(device)
            # print("X in", x.shape)
            prediction = model(x)
            # print(lb.dtype, lb.shape, prediction.dtype, prediction.shape)
            loss = loss_function(prediction, lb.to(device))
            loss.backward()
            # if it % 10 == 0:
            #     print(it, loss)
            optimizer.step()
        true_test = []
        predicted_test = []
        xs = []
        lbs = []
        lbws = []
        print("Train last loss: ", loss)
        model.eval()
        for _, data in enumerate(test_dataloader):
            x, lb, lws = data
            if is_first_test:
                xs.append(x)
                lbs.append(lb)
                lbws.append(lws)
            if model.type == "Transformer":
                x = x.transpose(1, 0)
            else:
                x = torch.unsqueeze(x, 1)
            x = x.float().to(device)
            # print("X in", x.shape)
            prediction = model(x)
            true_test.append(lb)
            # print("P S: ", prediction.shape)
            predicted_test.append(prediction)

        true_test = torch.concat(true_test, dim=0).detach().cpu()[:, :-1]
        predicted_test = torch.concat(predicted_test, dim=0).detach().cpu()[:, :-1]
        auc, aupr = roc_auc_score(true_test, predicted_test), average_precision_score(true_test, predicted_test)
        f1x = 2* auc * aupr / (auc + aupr + 1e-10)
        test_loss = loss_function(predicted_test, true_test)
        print(torch.sum(true_test, dim=0))
        print(sm(predicted_test[:2, :]), true_test[:2, :])
        ss = sm(predicted_test)
        test_loss2 = test_loss
        if params.CRITERIA == "F1X":
            test_loss2 = - f1x
        if min_test_loss > test_loss2:
            min_test_loss = test_loss2
            min_id = epoch_id
            np.savetxt("out/predicted.txt", ss, fmt="%.4f")
            torch.save(model.state_dict(), "out/model.pkl")

            print("Find new Best: ", test_loss, math.fabs(test_loss2), math.fabs(min_test_loss))
            if is_first_test:
                xs = torch.concat(xs, dim=0).detach().cpu().numpy()
                lbs = torch.concat(lbs, dim=0).detach().cpu().numpy()
                lbws = torch.concat(lbws, dim=0).detach().cpu().numpy()
                np.savetxt("out/true.txt", true_test, fmt="%d")
                joblib.dump([xs, lbs, lbws, dataset.idx_2lb], "out/test_data.pkl")
                is_first_test = False
        print("Error Test: ", params.CRITERIA, test_loss, math.fabs(test_loss2), math.fabs(min_test_loss), epoch_id, min_id, auc, aupr)


if __name__ == "__main__":
    train()
