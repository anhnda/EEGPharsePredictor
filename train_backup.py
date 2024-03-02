import math
from optparse import OptionParser

import params
import utils
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

CLASS_WEIGHT = None  # torch.tensor([2, 1, 0.1,  2, 2, 0.5, 0]).float().to(device)
CLASS_WEIGHT2 = None  # torch.tensor([2, 1, 0.1,  2, 2, 0.5]).float()

loss_function = torch.nn.CrossEntropyLoss(weight=CLASS_WEIGHT)


def get_model_dirname():
    return "out/M_EGG{}_EMG{}_MOT{}_RD{}".format(not params.OFF_EGG, not params.OFF_EMG, not params.OFF_MOT,
                                                 params.RD_SEED)


def parse_x():
    parser = OptionParser()
    parser.add_option("-a", "--regg", dest="regg", action="store_true", help="advanced model")
    parser.add_option("-b", "--remg", dest="remg", action="store_true", help="advanced model")
    parser.add_option("-c", "--rmot", dest="rmot", action="store_true", help="advanced model")

    parser.add_option("-m", "--train", dest="train", type='int', default=1, help="validation fold id")
    parser.add_option("-e", "--test", dest="test", type='int', default=1, help="validation fold id")

    (cmd_options, args) = parser.parse_args()
    if cmd_options.regg:
        params.OFF_EGG = True
    if cmd_options.remg:
        params.OFF_EMG = True
    if cmd_options.rmot:
        params.OFF_MOT = True
    print(cmd_options, params.OFF_EGG, params.OFF_EMG, params.OFF_MOT, params.RD_SEED)
    params.TRAIN_ID = cmd_options.train
    params.TEST_ID = cmd_options.test


def get_loss_c3(c3_predictions, lws, device, w3=None):
    if w3 is None:
        w3 = [0.85, 1, 0.85]
    out = c3_predictions.reshape(c3_predictions.shape[0], 3)
    target = torch.zeros(out.shape).to(device)
    # batch_size, dim_size, n_channels = out.shape
    loss = 0
    for i in range(3):
        lbs = lws[:, i]
        tps = [(j, lbs[j], i) for j in range(len(lbs))]
        ids = tuple(np.transpose(tps))
        target[ids] = 1
        lossi = loss_function(out[:, :, i], target[:, :, i], ignore_index=-1)
        loss += w3[i] * lossi
    return loss


def train():
    torch.manual_seed(params.RD_SEED)
    generator1 = torch.Generator().manual_seed(params.RD_SEED)

    model_dir = get_model_dirname()
    utils.ensureDir(model_dir)

    from my_logger.logger2 import MyLogger
    logger = MyLogger(model_dir + "/log.txt")

    dataset = EGGDataset(tile_seq=TILE_SEQ, side_flag=SIDE_FLAG)
    n_class = dataset.get_num_class()
    model = get_model(n_class)
    train_dt, test_dt = random_split(dataset, [0.8, 0.2], generator=generator1)
    train_dataloader = DataLoader(train_dt, batch_size=params.BATCH_SIZE, num_workers=0, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dt, batch_size=params.BATCH_SIZE, num_workers=0, shuffle=False)
    loss_function2 = torch.nn.CrossEntropyLoss(weight=CLASS_WEIGHT2)

    optimizer = torch.optim.Adam(model.parameters())
    min_test_loss = 1e6
    min_id = -1
    sm = torch.nn.Softmax(dim=-1)
    is_first_test = True
    all_res = []
    for epoch_id in range(params.N_EPOCH):
        model.train()
        for it, data in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            x, lb, _, _ = data
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
        itest = 0
        for _, data in tqdm(enumerate(test_dataloader)):
            x, lb, lws, _ = data
            itest += 1
            # print("\r%s", itest, end="")

            if is_first_test and itest <= 2001:
                # xs.append(x)
                # lbs.append(lb)
                # lbws.append(lws)
                pass
            if model.type == "Transformer":
                x = x.transpose(1, 0)
            else:
                x = torch.unsqueeze(x, 1)
            x = x.float().to(device)
            # print("X in", x.shape)
            prediction = model(x)
            # print(lb.shape, prediction.shape)
            true_test.append(lb.detach().cpu())
            # print("P S: ", prediction.shape)
            predicted_test.append(prediction.detach().cpu())
        # exit(-1)
        true_test = torch.concat(true_test, dim=0).detach().cpu()[:, :-1]
        predicted_test = torch.concat(predicted_test, dim=0).detach().cpu()[:, :-1]
        auc, aupr = roc_auc_score(true_test, predicted_test), average_precision_score(true_test, predicted_test)
        f1x = 2 * auc * aupr / (auc + aupr + 1e-10)
        test_loss = loss_function2(predicted_test, true_test)
        logger.infoAll(torch.sum(true_test, dim=0))
        logger.infoAll((sm(predicted_test[:2, :]), true_test[:2, :]))
        ss = sm(predicted_test)
        test_loss2 = test_loss

        if params.CRITERIA == "F1X":
            test_loss2 = - f1x
        if min_test_loss > test_loss2:
            min_test_loss = test_loss2
            min_id = epoch_id
            np.savetxt("%s/predicted_%s.txt" % (get_model_dirname(), params.DID), ss, fmt="%.4f")
            torch.save(model.state_dict(), "%s/model_%s.pkl" % (get_model_dirname(), params.DID))

            logger.infoAll(("Find new Best: ", test_loss, math.fabs(test_loss2), math.fabs(min_test_loss)))
            if is_first_test:
                # xs = torch.concat(xs, dim=0).detach().cpu().numpy()
                # lbs = torch.concat(lbs, dim=0).detach().cpu().numpy()
                # lbws = torch.concat(lbws, dim=0).detach().cpu().numpy()
                np.savetxt("%s/true.txt" % (get_model_dirname()), true_test, fmt="%d")
                joblib.dump([xs, lbs, lbws, dataset.idx_2lb], "%s/test_data.pkl" % get_model_dirname())
                is_first_test = False
        logger.infoAll(("Error Test: ", params.CRITERIA, test_loss, math.fabs(test_loss2), math.fabs(min_test_loss)
                        , epoch_id, min_id, auc, aupr))
        logger.infoAll(("Best lost: ", math.fabs(min_test_loss), min_id))
        all_res.append([test_loss, math.fabs(test_loss2), math.fabs(min_test_loss)
                           , epoch_id, min_id, auc, aupr])
    logger.infoAll(("Best values: ", all_res[min_id]))


if __name__ == "__main__":
    parse_x()
    train()
