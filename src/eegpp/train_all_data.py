import math
import os.path
from optparse import OptionParser

from . import utils
from .dataset import EGGDataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from sklearn.metrics import roc_auc_score, average_precision_score
from . import params
import torch
from tqdm import tqdm
import numpy as np
import joblib
from .get_model import get_model, TILE_SEQ, SIDE_FLAG, device

CLASS_WEIGHT = None  # torch.tensor([2, 1, 0.1,  2, 2, 0.5, 0]).float().to(device)
CLASS_WEIGHT2 = None  # torch.tensor([2, 1, 0.1,  2, 2, 0.5]).float()
BINARY_WEIGHT = torch.tensor([0.1, 1]).float().to(device)

loss_function = torch.nn.CrossEntropyLoss(weight=CLASS_WEIGHT)
loss_functionx = torch.nn.CrossEntropyLoss(weight=CLASS_WEIGHT, ignore_index=-1)
loss_binary_function = torch.nn.CrossEntropyLoss(weight=BINARY_WEIGHT)
# def init_rd_seed():
torch.manual_seed(params.RD_SEED)
generator1 = torch.Generator().manual_seed(params.RD_SEED)
N_TRAIN_FILE = 3


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

    import yaml
    config = yaml.safe_load(open(params.DATA_CONFIG_PATH))
    params.DUMP_FILE_PATTERN = config["datasets"]["dump_file_pattern"]
    global N_TRAIN_FILE
    N_TRAIN_FILE = len(config["datasets"]["dump_files"])
    print("N_TRAIN_FILE", N_TRAIN_FILE)


def get_loss_c3(out, lws_array, device, w3=None):
    if w3 is None:
        if params.WINDOW_SIZE == 3:
            w3 = [0.85, 1, 0.85]
        elif params.WINDOW_SIZE == 5:
            w3 = [0.4, 0.85, 1, 0.85, 0.4]
    target = lws_array.to(device)
    loss = 0
    for i in range(params.WINDOW_SIZE):
        lossi = loss_functionx(out[:, :, i], target[:, :, i])
        loss += w3[i] * lossi
    return loss


def get_loss_c3_binary(out, lws_array, device, w3=None):
    if w3 is None:
        if params.WINDOW_SIZE == 3:
            w3 = [0.85, 1, 0.85]
        elif params.WINDOW_SIZE == 5:
            w3 = [0.4, 0.85, 1, 0.85, 0.4]
    target = lws_array.to(device)
    loss = 0
    for i in range(params.WINDOW_SIZE):
        lossi = loss_binary_function(out[:, :, i], target[:, :, i])
        loss += w3[i] * lossi
    return loss


def get_train_test_all():
    datasets = []
    if not os.path.exists(params.DUMP_FILE_PATTERN % 1):
        print("Regenerating pkl files...")
        from .data_loader import force_load_all_with_labels
        force_load_all_with_labels()
    for i in range(1, N_TRAIN_FILE + 1):
        dump_path = params.DUMP_FILE_PATTERN % i
        print("Loading ", dump_path)
        dataset = EGGDataset(dump_path=dump_path, tile_seq=TILE_SEQ, side_flag=SIDE_FLAG)
        datasets.append(dataset)
    train_dts, test_dts = [], []
    for i in range(N_TRAIN_FILE):
        train_dt, test_dt = random_split(datasets[i], [0.8, 0.2], generator=generator1)
        train_dts.append(train_dt)
        test_dts.append(test_dt)

    train_dt = torch.utils.data.ConcatDataset(train_dts)
    test_dt = torch.utils.data.ConcatDataset(test_dts)
    return train_dt, test_dt, dataset.idx_2lb


def train():
    model_dir = get_model_dirname()
    utils.ensureDir(model_dir)

    from .my_logger.logger2 import MyLogger
    logger = MyLogger(model_dir + "/log.txt")

    n_class = params.NUM_CLASSES
    model = get_model(n_class)
    train_dt, test_dt, idx_2lb = get_train_test_all()

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
            x, lb, _, lbws_array, _ = data
            # print(lb.shape, lbws_array.shape)
            bcls = torch.zeros((lbws_array.shape[0], 2, lbws_array.shape[-1]))
            for ii, li in enumerate(lbws_array):
                for i in range(lbws_array.shape[-1]):
                    v = li[:, i]

                    if v[0] + v[2] + v[4] >= 1:
                        bcls[ii, 0, i] = 1
                    else:
                        bcls[ii, 1, i] = 0
            # print(x.shape, lb)
            # exit(-1)
            bcls = bcls.to(device)

            if model.type == "Transformer":
                x = x.transpose(1, 0)
            else:
                x = torch.unsqueeze(x, 1)
            x = x.float().to(device)
            prediction, prediction2 = model(x)

            # print("X in", x.shape)
            # print("Predicted", prediction2.shape, prediction2[:2])
            # print(bcls.shape, torch.sum(bcls), bcls[:2])
            # print("TB: ", torch.sum(bcls))

            loss = get_loss_c3(prediction, lbws_array, device)
            loss += params.W_STAR * get_loss_c3_binary(prediction2, bcls, device)


            # loss = get_loss_c3_binary(prediction2, bcls, device)

            loss.backward()
            # if it % 10 == 0:
            #     print(it, loss)
            optimizer.step()

        true_test = []
        predicted_test = []
        predicted_binary = []
        true_binary = []
        xs = []
        lbs = []
        lbws = []
        print("Train last loss: ", loss)
        model.eval()
        itest = 0
        with torch.no_grad():
            for cc, data in tqdm(enumerate(test_dataloader)):
                x, lb, _, lbws_array, _ = data

                x = torch.unsqueeze(x, 1)
                x = x.float().to(device)
                prediction, prediction2 = model(x)

                bcls = []
                for li in lb:
                    if li[0] + li[2] + li[4] >= 1:
                        bcls.append([1, 0])
                    else:
                        bcls.append([0, 1])
                bcls = torch.tensor(bcls)
                # prediction, prediction2 = model(x)
                # print("Test: x", x.shape)
                # print(prediction2.shape, prediction2[:2])
                # print(bcls.shape, torch.sum(bcls), bcls[:2])
                # print(lb.shape, prediction.shape)
                true_test.append(lb.detach().cpu())
                true_binary.append(bcls.detach().cpu())
                # print("P S: ", prediction.shape)

                predicted_test.append(prediction.detach().cpu())
                predicted_binary.append(prediction2.detach().cpu())

        # exit(-1)

        true_test = torch.concat(true_test, dim=0).detach().cpu()[:, :-1]
        true_binary = torch.concat(true_binary, dim=0).detach().cpu()
        if params.OUT_3C:
            predicted_test = torch.concat(predicted_test, dim=0).detach().cpu()[:, :-1, params.POS_ID]
            predicted_binary = torch.concat(predicted_binary, dim=0).detach().cpu()[:,:, params.POS_ID]

        else:
            raise 'Only OUT_3C is supported'

        predicted_test = sm(predicted_test)
        predicted_binary = sm(predicted_binary)

        print("BINARY: ", true_binary[:2], predicted_binary[:2])


        auc, aupr = roc_auc_score(true_test, predicted_test), average_precision_score(true_test, predicted_test)
        auc2, aupr2 = roc_auc_score(true_binary, predicted_binary), average_precision_score(true_binary,
                                                                                            predicted_binary)

        f1x = 2 * auc * aupr / (auc + aupr + 1e-10)
        f2x = 2 * auc2 * aupr2 / (auc2 + aupr2 + 1e-10)

        test_loss = loss_function2(predicted_test, true_test)
        logger.infoAll(torch.sum(true_test, dim=0))
        logger.infoAll((predicted_test[:2, :], true_test[:2, :]))
        # ss = sm(predicted_test)
        test_loss2 = test_loss

        if params.CRITERIA == "F1X":
            test_loss2 = - f1x
        elif params.CRITERIA == "F2X":
            test_loss2 = - f2x

        if min_test_loss > test_loss2:
            min_test_loss = test_loss2
            min_id = epoch_id
            np.savetxt("%s/predicted_%s.txt" % (get_model_dirname(), params.DID), predicted_test, fmt="%.12f")
            np.savetxt("%s/predicted_binary_%s.txt" % (get_model_dirname(), params.DID), predicted_binary, fmt="%.12f")

            torch.save(model.state_dict(), "%s/model_%s.pkl" % (get_model_dirname(), params.DID))

            logger.infoAll(("Find new Best: ", test_loss, math.fabs(test_loss2), math.fabs(min_test_loss)))
            if is_first_test:
                # xs = torch.concat(xs, dim=0).detach().cpu().numpy()
                # lbs = torch.concat(lbs, dim=0).detach().cpu().numpy()
                # lbws = torch.concat(lbws, dim=0).detach().cpu().numpy()
                np.savetxt("%s/true.txt" % (get_model_dirname()), true_test, fmt="%d")
                joblib.dump([xs, lbs, lbws, idx_2lb], "%s/test_data.pkl" % get_model_dirname())
                is_first_test = False
        logger.infoAll(("Error Test: ", params.CRITERIA, test_loss, math.fabs(test_loss2), math.fabs(min_test_loss)
                        , epoch_id, min_id, auc, aupr, auc2, aupr2))
        logger.infoAll(("Best lost: ", math.fabs(min_test_loss), min_id))
        all_res.append([test_loss, math.fabs(test_loss2), math.fabs(min_test_loss)
                           , epoch_id, min_id, auc, aupr])
    logger.infoAll(("Best values: ", all_res[min_id]))


def run_training_all():
    parse_x()
    train()


if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    run_training_all()
